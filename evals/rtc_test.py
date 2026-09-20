"""Tests for evals/rtc.py -- the framework seam, against a fake policy.

    python -m evals.rtc_test            # needs inspect-robots; no openpi, no arm
    pytest evals/rtc_test.py

scripts/openpi/chunk_loop_test.py already proves the arithmetic. What is left to check
is the wiring around it, and every one of these has a specific way of going wrong that
a hardware session would NOT make obvious:

  * the chunk is executed from the index it was sampled for (a wrong offset looks like
    a slightly clumsy policy);
  * `at_step` names the step the observation came from, or the report's camera player
    shows the operator a frame the model never saw;
  * the inference is recorded under the framework's own key, or the log has scores and
    no inference events -- this repo has shipped an empty report before;
  * a policy that cannot honour a prefix is refused rather than silently splicing
    every seam cold while the banner says "rtc".

The fake policy answers instantly, so these say nothing about latency. Lateness is
covered where it belongs: in the schedule's own tests, where a 40-tick inference in a
15-tick window still has to keep d exact.
"""

from __future__ import annotations

import time

import numpy as np
from inspect_robots.controller import _INFER_KEY
from inspect_robots.types import Action, ActionChunk, Observation

from evals.rtc import AsyncChunkController

HORIZON = 15
CHUNK = 50


class FakePolicy:
  """Records what it was asked for; answers with an identifiable chunk.

  Each action's data is (chunk_id, index), so a rollout log says exactly which action
  of which chunk reached the arm -- the thing an offset bug gets wrong.
  """

  supports_async = True

  def __init__(self, *, chunk_len: int = CHUNK, prefix: bool = True, collect_s: float = 0.0):
    self.supports_prefix = prefix
    self._chunk_len = chunk_len
    self._collect_s = collect_s
    self._pending: dict | None = None
    self._chunks = 0
    self.submits: list[dict] = []
    self.replans: list[dict] = []

  @property
  def busy(self) -> bool:
    return self._pending is not None

  def submit(self, observation, *, at_step, prefix_start=0, inference_delay=0) -> None:
    assert self._pending is None, "the controller submitted twice without collecting"
    request = {
      "at_step": at_step,
      "prefix_start": prefix_start,
      "inference_delay": inference_delay,
    }
    self._pending = request
    self.submits.append(request)

  def collect(self, *, block: bool = True) -> ActionChunk:
    assert self._pending is not None, "collect with nothing in flight"
    self._pending = None
    if self._collect_s:
      time.sleep(self._collect_s)
    chunk_id = self._chunks
    self._chunks += 1
    return ActionChunk(
      actions=[
        Action(data=np.array([chunk_id, i], dtype=np.float64))
        for i in range(self._chunk_len)
      ],
      inference_latency_s=0.05,
    )

  def note_replan(self, *, late: bool, stall_s: float = 0.0) -> None:
    self.replans.append({"late": late, "stall_s": stall_s})


def observation() -> Observation:
  return Observation(images={}, state={"joint_pos": np.zeros(6)})


def drive(ticks: int, *, policy: FakePolicy | None = None, rtc: bool = True):
  """Run the controller the way the rollout does: once per step, one action out."""
  policy = policy or FakePolicy(prefix=rtc)
  controller = AsyncChunkController(HORIZON, rtc=rtc)
  store: dict = {}
  sent = []
  for t in range(ticks):
    action = controller.next_action(policy, observation(), t, store)
    chunk_id, index = (int(v) for v in np.asarray(action.data))
    sent.append((t, chunk_id, index))
  return policy, store, sent


def test_the_first_window_plays_the_first_chunk_from_its_head():
  _, _, sent = drive(HORIZON)
  assert sent == [(t, 0, t) for t in range(HORIZON)]


def test_a_new_chunk_is_executed_from_the_index_it_was_sampled_for():
  policy, _, sent = drive(60)
  swaps = [(t, c, i) for t, c, i in sent if i == 0 or (t and c != sent[t - 1][1])]
  # Chunk 1 takes over at index 14: d actions of it are already behind us.
  first_of_chunk_1 = next((t, c, i) for t, c, i in sent if c == 1)
  assert first_of_chunk_1 == (15, 1, 14), first_of_chunk_1
  # ... and it plays 15 actions from there before chunk 2 takes over.
  first_of_chunk_2 = next((t, c, i) for t, c, i in sent if c == 2)
  assert first_of_chunk_2 == (30, 2, 14), first_of_chunk_2
  assert swaps  # the listing above is only meaningful while chunks do change


def test_the_promise_matches_what_the_arm_was_sent():
  """d, end to end through the controller: promised at submit, counted at the seam."""
  policy, _, sent = drive(200)
  swap_ticks = [t for t, c, i in sent if i == 14 and c > 0]
  for request, swap in zip(policy.submits[1:], swap_ticks):
    executed = [(t, c, i) for t, c, i in sent if request["at_step"] <= t < swap]
    assert len(executed) == request["inference_delay"], (request, swap, executed)


def test_at_step_names_the_step_the_observation_came_from():
  policy, _, _ = drive(60)
  # The first is the trial's blocking chunk at step 0; then one tick into each window.
  assert [r["at_step"] for r in policy.submits[:4]] == [0, 1, 16, 31]


def test_every_chunk_is_recorded_under_the_frameworks_key():
  policy, store, sent = drive(60)
  recorded = store[_INFER_KEY]
  chunks = len({c for _, c, _ in sent})
  assert len(recorded) == chunks
  # (latency, actions actually executed) -- never the 50 that were predicted.
  assert all(entry == (0.05, HORIZON) for entry in recorded), recorded


def test_async_mode_splices_from_the_head_and_asks_for_no_prefix():
  policy, _, sent = drive(60, policy=FakePolicy(prefix=False), rtc=False)
  assert all(r["inference_delay"] == 0 for r in policy.submits)
  assert next((t, c, i) for t, c, i in sent if c == 1) == (15, 1, 0)


def test_a_policy_that_cannot_prefix_is_refused_in_rtc_mode():
  controller = AsyncChunkController(HORIZON, rtc=True)
  try:
    controller.next_action(FakePolicy(prefix=False), observation(), 0, {})
  except RuntimeError as exc:
    assert "prefix" in str(exc)
  else:
    raise AssertionError("rtc mode accepted a policy that cannot honour a prefix")


def test_a_policy_that_cannot_overlap_is_refused():
  class Blocking:
    supports_prefix = True  # claims the model half, has none of the loop half

  controller = AsyncChunkController(HORIZON, rtc=True)
  try:
    controller.next_action(Blocking(), observation(), 0, {})
  except RuntimeError as exc:
    assert "supports_async" in str(exc)
  else:
    raise AssertionError("overlapped execution accepted a policy with no submit()")


def test_a_one_action_stop_chunk_is_executed_rather_than_indexed_past_the_end():
  """What evals/deadline.py hands back when a trial runs out of wall clock."""
  policy, _, sent = drive(20, policy=FakePolicy(chunk_len=1))
  assert sent[0] == (0, 0, 0)
  # Every later chunk is one action long too, so each swap sends its only action.
  assert all(i == 0 for _, _, i in sent)


def test_a_seam_that_had_to_wait_is_reported_to_the_policy():
  policy = FakePolicy(collect_s=0.01)
  drive(40, policy=policy)
  # The trial's first chunk blocks by construction and is not a late seam; the swaps
  # after it are, because this fake makes every collect wait.
  assert policy.replans, "no seam was reported"
  assert all(entry["late"] for entry in policy.replans)
  assert all(entry["stall_s"] >= 0.01 for entry in policy.replans)


class AsyncScripted:
  """The mock oracle, wrapped in the submit/collect seam this controller needs.

  Deliberately NOT a mock of OpenPiPolicy: the point is to run the controller through
  the real eval() -- rollout, per-trial store, approver, log -- and see that a trial
  starts clean. A leftover chunk or a leaked store would make the second trial execute
  from index d with no prefix behind it, which on hardware is a lurch at step 0 and in
  a log is nothing at all.
  """

  supports_async = True
  supports_prefix = True

  def __init__(self) -> None:
    from inspect_robots.mock.policies import ScriptedPolicy

    self._inner = ScriptedPolicy(chunk_size=8)
    self.info = self._inner.info
    self.config = self._inner.config
    self._pending: tuple[dict, ActionChunk] | None = None
    self.requests: list[dict] = []
    self.trial = -1

  @property
  def busy(self) -> bool:
    return self._pending is not None

  def reset(self, scene) -> None:
    self._pending = None  # the drain evals/openpi_policy.py does for real
    self.trial += 1
    self._inner.reset(scene)

  def submit(self, observation, *, at_step, prefix_start=0, inference_delay=0) -> None:
    request = {
      "trial": self.trial,
      "at_step": at_step,
      "prefix_start": prefix_start,
      "inference_delay": inference_delay,
    }
    self.requests.append(request)
    self._pending = (request, self._inner.act(observation))

  def collect(self, *, block: bool = True) -> ActionChunk:
    assert self._pending is not None
    _, chunk = self._pending
    self._pending = None
    return chunk


def test_a_real_eval_run_keeps_each_trial_independent():
  """Through the actual eval(): two scenes, two epochs, the CubePick mock world."""
  import tempfile

  from inspect_robots import Scene, Task, eval as robot_eval
  from inspect_robots.mock import CubePickEmbodiment

  policy = AsyncScripted()
  task = Task(
    name="rtc-controller-smoke",
    scenes=[Scene(id="a", instruction="pick the cube"),
            Scene(id="b", instruction="pick the cube")],
    scorer="success_at_end",
    max_steps=24,
    epochs=2,
  )
  with tempfile.TemporaryDirectory() as log_dir:
    (log,) = robot_eval(
      task,
      policy,
      CubePickEmbodiment(),
      controller=AsyncChunkController(HORIZON, rtc=True),
      log_dir=log_dir,
      grader=None,
    )
  assert log.status == "success", log.status
  assert log.results.total_trials == 4, log.results.total_trials
  # Every trial opens with a blocking, unguided inference on its own step 0. If the
  # controller's state outlived a trial, the second one would open mid-window.
  for trial in range(4):
    first = next(r for r in policy.requests if r["trial"] == trial)
    assert (first["at_step"], first["prefix_start"], first["inference_delay"]) == (0, 0, 0), first
  # And the run really did overlap: more than one inference per trial, all of them
  # after the first guided, which is what distinguishes this from --mode sync.
  per_trial = [len([r for r in policy.requests if r["trial"] == t]) for t in range(4)]
  assert all(n > 1 for n in per_trial), per_trial
  assert any(r["inference_delay"] > 0 for r in policy.requests)


def main() -> int:
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
