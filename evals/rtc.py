"""Overlapped chunk execution for the eval loop, as an Inspect Robots Controller.

    ./robot eval-openpi --mode rtc        # this file
    ./robot eval-openpi --mode async      # this file, without the guidance
    ./robot eval-openpi --mode sync       # NOT this file: DefaultController + act()

WHY A CONTROLLER AND NOT A POLICY. Overlapping inference with execution needs an
observation from the MIDDLE of a window, and a policy is only handed one when it is
asked for a chunk. The Controller seam is called once per control step with that
step's observation and decides which action to send, which is exactly the shape of the
problem -- so the framework already has the hook, and evals/openpi_policy.py stays a
model wrapper rather than growing a control loop.

WHAT LIVES WHERE, because this is the third file in a three-file mechanism:

    scripts/openpi/chunk_loop.py   the index arithmetic and d, shared with the web UI
    evals/openpi_policy.py         the model, the chunker, the in-flight inference
    THIS FILE                      the framework seam, and the stall accounting

The arithmetic is deliberately not here. webui/openpi_worker.py drives the same
ChunkSchedule with its own thread and its own 30 Hz pacing, and the whole point of RTC
is that `d` means the same thing in both loops -- so it is defined once, tested once
without a GPU (scripts/openpi/chunk_loop_test.py), and imported by both.

A NOTE ON WHAT IS MEASURED HERE. A late chunk is the one failure this layer can see
and neither the model nor the schedule can: the schedule keeps d honest by stalling at
the retirement index, and the stall shows up as the arm holding still for a moment.
That is a real cost -- it drops the effective control rate below the 30 Hz the
checkpoint was tuned at -- so every blocked collect is timed and handed to the policy,
which owns the per-trial log entry.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from inspect_robots.types import Action, ActionChunk, Observation

# The key DefaultController writes its (latency, chunk_len) pairs under, and the one
# the rollout diffs after every next_action() to emit an inference event and flush the
# policy's transcript delta. Imported rather than copied: a controller that invented
# its own key would run correctly and log nothing, and this repo has already shipped
# one empty report. If a future Inspect Robots renames it, this fails at import.
from inspect_robots.controller import _INFER_KEY

ROOT = pathlib.Path(__file__).resolve().parent.parent
_MODULE_NAME = "agent101_chunk_loop"

# A collect that blocked for less than this is scheduling noise, not a late chunk.
LATE_S = 0.001


def chunk_loop():
  """scripts/openpi/chunk_loop.py, imported by path and cached.

  Same reason evals/openpi_policy.py loads evaluate.py this way: scripts/ is not a
  package, and putting scripts/openpi/ on sys.path would shadow the installed openpi.
  Registering the module in sys.modules is NOT optional -- @dataclass resolves its
  own module out of there, and a module that skipped this step dies on its first
  dataclass with an unrecognisable AttributeError.
  """
  cached = sys.modules.get(_MODULE_NAME)
  if cached is not None:
    return cached
  path = ROOT / "scripts" / "openpi" / "chunk_loop.py"
  spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"could not load {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[_MODULE_NAME] = module
  spec.loader.exec_module(module)
  return module


@dataclass
class _Trial:
  """One trial's loop state. Lives in the rollout's per-trial `store`."""

  schedule: Any
  actions: list[Action] = field(default_factory=list)
  request: Any = None

  def adopt(self, chunk: ActionChunk, delay: int, store: dict[str, Any]) -> None:
    """Take a chunk into execution and record the inference that produced it."""
    self.actions = list(chunk.actions)
    self.schedule.adopt(len(self.actions), delay=delay)
    # How many of this chunk's actions will actually be executed -- the same thing
    # DefaultController records, so a report cannot overstate a 50-action chunk.
    executed = self.schedule.end - self.schedule.start
    store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, executed))


class AsyncChunkController:
  """Play the current chunk while the next one is being predicted.

  Args:
    replan_interval: actions executed per chunk before replanning (15 here).
    rtc: ask the policy to honour a prefix. Requires a policy that says it can;
      False is the cold-splice `--mode async`, which exists to show what the
      guidance is worth rather than to be run.
  """

  def __init__(self, replan_interval: int, *, rtc: bool = True) -> None:
    self._replan = replan_interval
    self._rtc = rtc
    self._key = "_openpi_chunk_loop"

  def _require(self, policy: Any) -> None:
    """Refuse a policy that cannot do this, rather than degrading quietly.

    Both halves are checked because they fail differently: a policy without
    submit()/collect() cannot overlap at all, and one without a chunker would run
    the whole session looking fine while every seam was spliced cold.
    """
    if not getattr(policy, "supports_async", False):
      raise RuntimeError(
        f"{type(policy).__name__} does not declare supports_async: overlapped "
        "execution needs a policy that can start an inference and pick it up later. "
        "Run with --mode sync, which uses the framework's own controller."
      )
    for name in ("submit", "collect"):
      if not callable(getattr(policy, name, None)):
        raise RuntimeError(
          f"{type(policy).__name__} declares supports_async but has no {name}()"
        )
    if self._rtc and not getattr(policy, "supports_prefix", False):
      raise RuntimeError(
        f"{type(policy).__name__} cannot honour an action prefix, so --mode "
        "rtc would splice every seam cold while claiming to be guided. Build "
        "the policy with rtc=True, or run --mode async and say so."
      )

  def next_action(
    self, policy: Any, observation: Observation, t: int, store: dict[str, Any]
  ) -> Action:
    """Choose step `t`'s action, submitting and collecting inferences as due."""
    trial: _Trial | None = store.get(self._key)
    if trial is None:
      self._require(policy)
      schedule = chunk_loop().ChunkSchedule(
        self._replan, overlap=True, rtc=self._rtc
      )
      trial = store[self._key] = _Trial(schedule)

    plan = trial.schedule.plan(in_flight=bool(getattr(policy, "busy", False)))
    if plan.action == "infer":
      # The trial's first chunk: nothing is executing, so this one is waited for.
      # Not counted as a stall -- there is no motion to interrupt yet.
      self._submit(policy, observation, t, plan.request)
      chunk = policy.collect(block=True)
      trial.adopt(chunk, plan.request.inference_delay, store)
    elif plan.action == "collect":
      started = time.perf_counter()
      chunk = policy.collect(block=True)
      stalled = time.perf_counter() - started
      note = getattr(policy, "note_replan", None)
      if callable(note):
        note(late=stalled > LATE_S, stall_s=stalled)
      # The delay the SAMPLER was given, not one recomputed now: recomputing it
      # after the fact is how a loop ends up executing a chunk from an index it
      # was never sampled for.
      trial.adopt(chunk, trial.request.inference_delay, store)
    elif plan.request is not None:
      self._submit(policy, observation, t, plan.request)
      trial.request = plan.request

    return trial.actions[trial.schedule.take()]

  @staticmethod
  def _submit(policy: Any, observation: Observation, t: int, request: Any) -> None:
    """Hand one inference to the policy, naming the step the observation is from.

    `at_step` is what lets the report resolve this inference to the frame the model
    actually saw: an overlapped inference is submitted mid-window, so its step is
    not a multiple of the replan interval and the policy cannot infer it.
    """
    policy.submit(
      observation,
      at_step=t,
      prefix_start=request.prefix_start,
      inference_delay=request.inference_delay,
    )
