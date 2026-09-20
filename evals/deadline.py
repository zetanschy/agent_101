"""A wall-clock guard around any policy.

Nothing in Inspect Robots bounds a trial in TIME. `max_steps` and `max_seconds` are
both step counts (the latter converted at the embodiment's declared control_hz), and
`max_llm_calls` bounds decisions -- all counts, none of them a clock. That is usually
fine, because a decision takes about 8 seconds. It stops being fine when the provider
is degraded but not erroring: the agent's HTTP client waits 120 s per request and
retries three times, so a single decision can burn six minutes without failing, and a
40-decision trial can in principle run for hours with nobody watching.

This wrapper ends a trial the way the framework already lets a policy end one: an
action carrying `request_stop` in its meta, which the rollout records as a truncation
with the given reason. Three properties matter:

  * THE STOP ACTION IS THE CURRENT POSE. Commanding where the arm already is means the
    guard cannot itself become a motion, which a zero vector or a home pose would.
  * IT TRUNCATES, IT DOES NOT ERROR. The log gets status success with
    termination_reason "wall_clock_timeout", the same shape as running out of steps --
    so a slow trial is distinguishable from a broken one, and from an operator's
    Ctrl-C, at read time.
  * THE DEADLINE IS PER TRIAL, reset in reset(), because that is the unit an operator
    supervises.

Everything else is forwarded to the wrapped policy, including the optional hooks the
framework probes with hasattr -- bind(), transcript(), on_trial_start/end -- because a
wrapper that hid transcript() would empty the report's camera player, which is a
failure this repo has already had once.

THE GUARD SITS ON THREE METHODS, not one. In --mode async/rtc the controller
(evals/rtc.py) never calls act(): it calls submit() and collect() instead. A deadline
that only guarded act() would therefore never fire on the openpi path -- the path whose
trials are 3600 steps long and whose inference runs on another thread.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

STOP_REASON = "wall_clock_timeout"


class Deadline:
  """Wrap a policy so each trial gets at most `minutes` of wall clock."""

  def __init__(self, inner: Any, minutes: float) -> None:
    if minutes <= 0:
      raise ValueError(f"minutes must be positive, got {minutes}")
    self._inner = inner
    self._budget_s = minutes * 60.0
    self._started = time.monotonic()
    self._fired = False
    self._stop_chunk: ActionChunk | None = None

  def __getattr__(self, name: str) -> Any:
    """Forward everything this class does not define to the wrapped policy.

    That includes `info` and `config`, and they must NOT be copied in __init__.
    An embodiment-adaptive policy rebuilds its spaces in bind(), which eval() calls
    AFTER construction and BEFORE the compatibility check: the LLM agent declares a
    1-D placeholder until bind() gives it the embodiment's 6-D joint space. A snapshot
    taken here therefore fails the compat check with "policy emits 1-D actions but
    embodiment expects 6-D" -- which is exactly what an earlier version of this file
    did. Forwarding keeps them live.

    __getattr__ fires only for attributes this class does not define, so it cannot
    shadow reset or act, and hasattr() probes see the inner policy's surface, which is
    how the framework detects the optional hooks (bind, transcript, on_trial_*).
    """
    return getattr(self._inner, name)

  def reset(self, scene: Scene) -> None:
    """Start the inner policy's trial, and this trial's clock."""
    self._started = time.monotonic()
    self._fired = False
    self._stop_chunk = None
    self._inner.reset(scene)

  def elapsed(self) -> float:
    """Seconds since this trial's reset."""
    return time.monotonic() - self._started

  def expired(self) -> bool:
    """Whether this trial's budget is spent."""
    return self.elapsed() >= self._budget_s

  def _stop(self, observation: Observation) -> ActionChunk:
    """A one-action chunk holding the current pose, flagged to end the trial."""
    if not self._fired:
      print(
        f"[deadline] {self.elapsed() / 60:.1f} min elapsed, budget "
        f"{self._budget_s / 60:.1f} min: stopping the trial (arm holds position)",
        flush=True,
      )
      self._fired = True
    hold = np.asarray(observation.state["joint_pos"], dtype=np.float64)
    return ActionChunk(
      actions=[
        Action(data=hold, meta={"request_stop": True, "stop_reason": STOP_REASON})
      ]
    )

  def act(self, observation: Observation) -> ActionChunk:
    """The inner policy's chunk, or a hold-still stop once the budget is spent.

    Checked BEFORE delegating: past the deadline the point is not to pay for one more
    inference, and with the agent that call is both the slow part and the expensive
    part.
    """
    if self.expired():
      return self._stop(observation)
    return self._inner.act(observation)

  # --- the overlapped path (evals/rtc.py) --------------------------------------

  @property
  def busy(self) -> bool:
    """True while an inference is in flight, or a stop is waiting to be collected."""
    return self._stop_chunk is not None or bool(self._inner.busy)

  def submit(self, observation: Observation, **kwargs: Any) -> None:
    """Start the inner policy's inference, or arm the stop in its place.

    The stop is built from THIS observation, so the hold-still pose is where the arm
    actually is at the moment the guard fires rather than wherever it was when the
    controller last asked.

    A deadline that expires while an inference is already in flight cannot preempt it:
    the controller submits only when nothing is pending, so the stop arms one window
    later (half a second at 15 actions and 30 Hz). The alternative -- discarding a
    chunk that is already paid for -- would leave the arm mid-motion for longer.
    """
    if self.expired():
      self._stop_chunk = self._stop(observation)
      return
    self._inner.submit(observation, **kwargs)

  def collect(self, *, block: bool = True) -> ActionChunk | None:
    """The armed stop if there is one, else the inner policy's chunk."""
    if self._stop_chunk is not None:
      chunk, self._stop_chunk = self._stop_chunk, None
      return chunk
    return self._inner.collect(block=block)
