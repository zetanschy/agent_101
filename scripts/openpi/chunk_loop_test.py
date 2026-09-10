#!/usr/bin/env python3
"""Tests for chunk_loop.ChunkSchedule -- the arithmetic no log can check afterwards.

    python3 scripts/openpi/chunk_loop_test.py      # no deps, no GPU, no arm
    pytest scripts/openpi/chunk_loop_test.py

The point of these is `d`. Everything else about real-time chunking is visible in a
recording: a jump at the seam, a stalled arm, an absmax past 1. Whether the sampler was
told the RIGHT number of committed actions is not -- a wrong d produces smooth,
plausible motion that is pinned to the wrong instant, and the only way to know is to
count the actions the loop actually sends. So the simulation below counts them.

The fake loop mirrors evals/rtc.py's structure (plan, then submit, then send in the
same tick). webui/openpi_worker.py skips the send on a swap tick instead, which changes
which tick an action lands on but not how many are sent per window -- the property
under test.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_spec = importlib.util.spec_from_file_location(
    "chunk_loop", pathlib.Path(__file__).resolve().parent / "chunk_loop.py"
)
chunk_loop = importlib.util.module_from_spec(_spec)
# REGISTERED BEFORE EXECUTION. @dataclass resolves its own module out of sys.modules to
# check for KW_ONLY, so a by-path import that skips this step dies with a confusing
# "NoneType has no attribute __dict__" on the first dataclass. Both real callers load
# this module the same way and need the same line.
sys.modules[_spec.name] = chunk_loop
_spec.loader.exec_module(chunk_loop)

ChunkSchedule = chunk_loop.ChunkSchedule
HORIZON = 15
CHUNK = 50


def simulate(ticks, *, horizon=HORIZON, chunk_len=CHUNK, latency=3, overlap=True, rtc=True):
    """Run the schedule for `ticks`, with an inference that takes `latency` ticks.

    Returns one record per tick: what was submitted, what was collected, what was sent.
    `latency` is in ticks so a slow inference (latency > horizon) is expressible, which
    is the case that stalls the loop.
    """
    sched = ChunkSchedule(horizon, overlap=overlap, rtc=rtc)
    log = []
    pending = None  # (ready_at_tick, Request)
    for t in range(ticks):
        entry = {"t": t, "submitted": None, "collected": None, "sent": None, "stalled": False}
        plan = sched.plan(in_flight=pending is not None)
        if plan.action == "infer":
            # Blocking: the caller waits, so the tick "contains" the whole inference.
            sched.adopt(chunk_len, delay=plan.request.inference_delay)
            entry["submitted"] = plan.request
            entry["collected"] = plan.request
        elif plan.action == "collect":
            ready_at, request = pending
            entry["stalled"] = ready_at > t
            pending = None
            sched.adopt(chunk_len, delay=request.inference_delay)
            entry["collected"] = request
        elif plan.request is not None:
            pending = (t + latency, plan.request)
            entry["submitted"] = plan.request
        entry["sent"] = sched.take()
        log.append(entry)
    return log


def test_first_window_is_blocking_and_unguided():
    log = simulate(3)
    assert log[0]["collected"].prefix_start == 0
    assert log[0]["collected"].inference_delay == 0, "the first chunk has no prefix"
    assert log[0]["sent"] == 0
    # Nothing is submitted at idx 0: a prefix there would pin the sampler to actions
    # the arm has not started executing.
    assert log[0]["submitted"].inference_delay == 0
    assert log[1]["submitted"] == chunk_loop.Request(prefix_start=1, inference_delay=14)


def test_steady_state_window_is_the_horizon():
    log = simulate(200)
    swaps = [e["t"] for e in log if e["collected"] is not None]
    # First swap after the blocking chunk plays its 15 actions, then every 15 ticks.
    assert swaps[:4] == [0, 15, 30, 45], swaps[:4]
    gaps = {b - a for a, b in zip(swaps[1:], swaps[2:])}
    assert gaps == {HORIZON}, gaps


def test_the_promise_is_exactly_what_gets_sent():
    """d == the number of old-chunk actions sent from the observation to the swap.

    This is the whole file. If this passes, RTC is pinning the seam to the instant the
    seam actually happens.
    """
    log = simulate(200)
    for entry in log:
        request = entry["submitted"]
        if request is None or entry["collected"] is not None:
            continue
        swap = next(
            (e["t"] for e in log if e["t"] > entry["t"] and e["collected"] is not None), None
        )
        if swap is None:
            continue  # the simulation ended before this one landed
        sent = [e["sent"] for e in log if entry["t"] <= e["t"] < swap]
        assert len(sent) == request.inference_delay, (
            f"submitted at t={entry['t']} promising d={request.inference_delay}, "
            f"but {len(sent)} actions were sent before the swap at t={swap}"
        )
        # ... and they are exactly the tail of the old window, starting where the
        # sampler was told index 0 of the new chunk lines up.
        assert sent == list(range(request.prefix_start, request.prefix_start + len(sent)))


def test_the_new_chunk_takes_over_at_index_d():
    """The first action executed off a new chunk is the one just past its pinned prefix."""
    log = simulate(200)
    for entry in log[1:]:
        if entry["collected"] is not None:
            assert entry["sent"] == entry["collected"].inference_delay


def test_a_late_inference_stalls_but_keeps_d_honest():
    """A slow inference costs motion, not correctness.

    latency 40 ticks against a 15-tick window: every swap waits. The promise still
    holds because the window is retired at a fixed index -- the loop cannot run past
    `end` while it waits.
    """
    log = simulate(200, latency=40)
    stalled = [e for e in log if e["stalled"]]
    assert stalled, "a 40-tick inference in a 15-tick window must stall"
    for entry in log:
        request = entry["submitted"]
        if request is None or entry["collected"] is not None:
            continue
        swap = next(
            (e["t"] for e in log if e["t"] > entry["t"] and e["collected"] is not None), None
        )
        if swap is None:
            continue
        sent = [e["sent"] for e in log if entry["t"] <= e["t"] < swap]
        assert len(sent) == request.inference_delay


def test_synchronous_loop_promises_zero_and_replays_from_zero():
    log = simulate(60, overlap=False)
    for entry in log:
        if entry["collected"] is not None:
            assert entry["collected"].inference_delay == 0
            assert entry["sent"] == 0, "with no overlap the new chunk starts at its head"
    # The retiring index is still handed over, so the sampler gets a soft seam.
    swaps = [e["collected"].prefix_start for e in log if e["collected"] is not None]
    assert swaps[:3] == [0, HORIZON, HORIZON]


def test_rtc_off_promises_nothing_even_when_overlapped():
    log = simulate(60, rtc=False)
    assert all(
        e["submitted"].inference_delay == 0 for e in log if e["submitted"] is not None
    )
    # ... and then the chunk is replayed from its head, cold. That jump at the seam is
    # what --mode async is, and why it is not the default.
    assert all(e["sent"] == 0 for e in log if e["collected"] is not None)


def test_a_one_action_chunk_stops_the_arm_instead_of_raising():
    """evals/deadline.py ends a trial with a single hold-still action."""
    sched = ChunkSchedule(HORIZON)
    sched.adopt(CHUNK, delay=0)
    for _ in range(5):
        sched.take()
    sched.adopt(1, delay=14)  # d from the window it replaced, chunk far shorter
    assert sched.take() == 0


def test_retuning_the_window_is_refused_mid_flight():
    sched = ChunkSchedule(HORIZON)
    sched.adopt(CHUNK, delay=0)
    sched.take()
    promised = sched.promise()
    assert sched.set_horizon(30, in_flight=True) is False
    assert sched.promise() == promised, "d must not move under an in-flight inference"
    assert sched.set_horizon(30, in_flight=False) is True
    assert sched.end == 30


def test_a_chunk_shorter_than_the_horizon_clips_the_window():
    sched = ChunkSchedule(HORIZON)
    sched.adopt(8, delay=0)
    assert sched.end == 8
    assert sched.promise() == 7, "d is clamped to the last index, never past the chunk"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
