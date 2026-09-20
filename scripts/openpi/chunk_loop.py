#!/usr/bin/env python3
"""When to ask for the next action chunk, and which action to send this tick.

The loop half of real-time chunking, and the only half that is about THIS RIG rather
than about the model:

    openpi.models.rtc         the algorithm -- prefix weights, guided velocity
    openpi.policies.rtc       RealTimeChunker: which previous actions to hand it
    THIS MODULE               when to ask, and what d actually is

Shared by the two loops that drive this checkpoint, because `d` has to mean the same
thing in both and because a second copy of this arithmetic would drift from the first:

    webui/openpi_worker.py    its own thread, its own 30 Hz pacing, send_action()
    evals/rtc.py              an Inspect Robots Controller, called once per step

Pure index arithmetic: no jax, no lerobot, no eval framework, no threads, no clock and
no hardware. That is what makes it testable without a GPU or an arm
(scripts/openpi/chunk_loop_test.py) -- which matters more here than usual, because the
one number this module exists to get right cannot be checked after the fact from a log.

THE THREE NUMBERS

    prefix_start  index, in the chunk now executing, of the action at the tick the
                  observation was captured. Index 0 of the new chunk lines up with it.
    d             how many actions the arm will have executed between that capture and
                  the moment the new chunk takes over.
    s             where prefix influence has decayed to zero: the whole overlap with
                  the chunk being retired. Chosen by openpi.policies.rtc, not here.

WHY d IS EXACT HERE. A loop that swaps chunks ON ARRIVAL cannot know d when it submits:
it has to predict its own latency, typically from a rolling maximum of measured delays,
and a prediction is all it can ever be. This loop retires a chunk at a FIXED index
instead. `end` is set when the window opens and never moves
while an inference is in flight, so `end - idx` is exactly how many actions will be
sent before the swap. If the inference overruns, the loop stalls AT `end` and still
sends no more than that, so a slow inference costs motion rather than correctness. If
`end` moved after a submit -- retuning the window live, say -- d would quietly become a
lie and RTC would optimise for a seam that never happens. Hence set_horizon() refuses
while an inference is in flight, which is the one rule in this file with teeth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# What plan() is telling the caller to do this tick.
#   "infer"    no chunk to execute (or nothing in flight at the boundary): produce one
#              NOW, blocking, then adopt() it.
#   "collect"  the window is used up and an inference is in flight: take it, blocking
#              if it has not landed, then adopt() it.
#   "execute"  keep playing the current chunk; submit `request` first if it is set.
Action = Literal["infer", "collect", "execute"]


@dataclass(frozen=True)
class Request:
    """One inference to start, and the two promises its sampler is given."""

    prefix_start: int
    inference_delay: int


@dataclass(frozen=True)
class Plan:
    """What to do this tick. `request` is set only when an inference should start."""

    action: Action
    request: Request | None = None


class ChunkSchedule:
    """Cursor over the chunk being executed, and the replanning boundary.

    One instance per episode-ish: reset() puts it back to "no chunk". The caller owns
    the chunk itself, the threading and the clock; this owns only indices, so the two
    loops that use it can look nothing alike and still agree on d.

    Args:
        horizon: actions to execute per chunk before replanning (`--actions`, 15 here).
        overlap: predict the next chunk while the current one still plays. False is the
            synchronous loop -- the arm holds still through every inference.
        rtc: promise a real d. False promises 0, which is honest for a synchronous
            loop (nothing is executed while it thinks) and a deliberate lie for an
            overlapped one, so an unguided async loop should simply not be given a
            chunker to guide with.
    """

    def __init__(self, horizon: int, *, overlap: bool = True, rtc: bool = True) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon
        self.overlap = overlap
        self.rtc = rtc
        self.reset()

    def reset(self) -> None:
        """Forget the current chunk. The next plan() asks for a blocking inference."""
        self._len = 0
        self.idx = 0
        self.start = 0
        self.end = 0

    @property
    def has_chunk(self) -> bool:
        """True once a chunk has been adopted and not reset."""
        return self._len > 0

    @property
    def at_boundary(self) -> bool:
        """True when the window is used up, i.e. this tick swaps chunks."""
        return self._len > 0 and self.idx >= self.end

    def set_horizon(self, horizon: int, *, in_flight: bool) -> bool:
        """Retune the execution window; returns whether it took.

        REFUSED while an inference is in flight, because that inference was told how
        many actions would be executed before it lands, and that count is `end - idx`
        measured against the window as it was. Widening the window afterwards would
        execute more actions than promised and RTC would pin the seam to the wrong
        place; narrowing it would swap early onto a chunk sampled for a later seam.
        The web UI's live `actions <n>` command is exactly this case.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if in_flight:
            return False
        self.horizon = horizon
        if self._len:
            self.end = min(self.start + horizon, self._len)
        return True

    def plan(self, *, in_flight: bool) -> Plan:
        """Decide this tick, given whether an inference is already in flight."""
        if self._len == 0:
            # First chunk of the episode. Nothing has been executed, so there is no
            # prefix to honour and no delay to promise; it has to be waited for.
            return Plan("infer", Request(prefix_start=0, inference_delay=0))
        if self.idx >= self.end:
            if in_flight:
                return Plan("collect")
            # The synchronous path: the arm holds still while we think, so nothing is
            # executed in the meantime and d is genuinely 0. prefix_start is the index
            # the retiring chunk reached, which still gives the sampler a soft seam to
            # start from -- the previous plan's tail, not a frozen prefix.
            return Plan("infer", Request(prefix_start=self.idx, inference_delay=0))
        request = None
        # `idx >= 1` and not `idx > start`: after a swap the cursor opens at d >= 1, so
        # this fires on the window's FIRST tick and the inference gets the whole window
        # as cover. Only the episode's first chunk starts at 0, and there a prefix
        # would pin the sampler to actions the arm has not begun executing.
        if self.overlap and not in_flight and self.idx >= 1:
            request = Request(prefix_start=self.idx, inference_delay=self.promise())
        return Plan("execute", request)

    def promise(self) -> int:
        """d for an inference submitted right now: exact, not estimated.

        `end` is fixed for the life of the window (see set_horizon), so every action
        from `idx` to `end - 1` will be sent before the swap and none after. Clamped to
        the chunk's last index, because a d past the end would pin the whole chunk and
        freeze the policy on a stale plan.
        """
        if not (self.rtc and self.overlap):
            return 0
        return max(min(self.end - self.idx, self._len - 1), 0)

    def adopt(self, chunk_len: int, *, delay: int) -> None:
        """Start executing a chunk that was sampled for `delay`.

        The cursor opens at `delay`, not 0: those actions of this chunk are already
        behind us, which is precisely what the sampler was told to reproduce. Passing
        the delay back rather than remembering it keeps a request the caller chose not
        to send from silently offsetting the next chunk.
        """
        if chunk_len < 1:
            raise ValueError(f"a chunk needs at least one action, got {chunk_len}")
        self._len = chunk_len
        # A chunk shorter than the promise clamps to its last index. That is not
        # hypothetical: evals/deadline.py ends a trial with a ONE-action hold-still
        # chunk, and indexing it at d would raise instead of stopping the arm.
        self.idx = self.start = min(max(delay, 0), chunk_len - 1)
        self.end = min(self.start + self.horizon, chunk_len)

    def take(self) -> int:
        """Index of the action to send this tick, advancing the cursor."""
        if self._len == 0:
            raise RuntimeError("take() with no chunk adopted")
        if self.idx >= self._len:
            raise RuntimeError(
                f"take() past the end of the chunk ({self.idx} >= {self._len}); "
                "the boundary plan must be handled first"
            )
        i = self.idx
        self.idx += 1
        return i
