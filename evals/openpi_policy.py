"""The openpi (JAX) pi0.5 checkpoint as an Inspect Robots `Policy`.

This is the model that actually works on this bench -- the orbax checkpoint the web UI
loads as `openpi:/checkpoints/openpi_pi05_lora_cap_to_cup_*` -- and it is NOT the
lerobot checkpoint inspect-robots-so101's own `LeRobotPolicy` can load. openpi is a
different stack: JAX, a `params/` + `assets/` directory rather than a safetensors
repo, its own slash-separated observation keys, and its own container.

WHAT THIS DOES NOT DO IS REDEFINE AN OBSERVATION. scripts/openpi/evaluate.py already
owns that: which camera fills which openpi slot, how state is packed, and how a
checkpoint's TrainConfig is discovered from its assets/ directory. webui/openpi_worker.py
imports it by path for exactly that reason and this does the same, so there is one
definition to be wrong in rather than three.

THREE SETTINGS MUST MATCH webui/openpi_worker.py, which is the configuration known to
drive this checkpoint well on this bench in both its modes. They were all wrong in the
first version of this file, and the arm moved strangely for all three reasons at once.

  UNITS: DEGREES. scripts/openpi/evaluate.py defaults `--units normalized`, with a
  comment about datasets recorded before lerobot 0.6.1 made degrees the default -- but
  webui/openpi_worker.py defaults `--units degrees`, and that is the path that works
  here. The webui default wins because it is the one with evidence behind it. Nothing
  downstream would catch this: the embodiment commands policy output verbatim after
  clamping, and Inspect Robots compares state KEYS rather than units, so the wrong
  choice is a silent per-joint rescaling (a factor of half_span/100 -- 1.69 on
  wrist_roll), not an exception.

  REPLAN INTERVAL: 15 of the 50 predicted actions. openpi_worker and evaluate.py both
  default `--actions 15`, so the last 35 actions of every chunk are normally discarded
  and re-planned from a fresh observation. Inspect Robots' DefaultController plays a
  WHOLE chunk when replan_interval is None, so leaving it unset executes 35 stale
  actions per chunk open-loop. `PolicyConfig.replan_interval` is what eval() reads to
  build that controller.

  SETTLING: OFF. The embodiment can wait for the arm to arrive before observing, which
  is right for an LLM agent taking one deliberate action at a time and wrong here: it
  changes the cadence of chunk replay, which is timing this policy was tuned against.
  inspect-robots-so101 leaves it off by default for exactly this reason. The runner
  passes settle_tolerance=None for this policy.

The first two live here, the third in evals/run.py; all three are keyed off the policy
so they cannot drift apart.

THE FOURTH SETTING IS THE EXECUTION MODE, and it was wrong here for longer than the
other three. openpi_worker.py defaults to `--mode async` and offers `--mode rtc`; this
file only ever did the equivalent of `--mode sync`, because Inspect Robots'
DefaultController calls act(), waits for it, and only then plays the chunk. On a
self-paced embodiment that is a real stall at every replan boundary: SOArmEmbodiment
sleeps to control_hz measured from the END of the previous step, so a 200 ms inference
is 200 ms of the arm holding still, once every 15 actions.

  async   the next chunk is predicted while the current one is still executing, so the
          arm never pauses -- but the new chunk is spliced in cold and the arm jumps at
          the seam, because nothing told the sampler which actions were committed while
          it was thinking.
  rtc     async, plus real-time chunking (arXiv:2506.07339): the chunk is sampled under
          a soft constraint pinning its first `d` actions to the ones the arm will have
          executed by the time it lands, so the seam is continuous by construction.

RTC IS THREE PARTS, and only one of them belongs in this file:

  openpi.models.rtc          the algorithm -- prefix weights, guidance weight, and the
                             guided velocity inside the flow sampler.
  openpi.policies.rtc        RealTimeChunker: which previous actions to hand it, and
                             the clipping of d and s.
  THIS FILE + evals/rtc.py   the control loop: what d actually is on this rig.

The loop part is the one that cannot be borrowed. `inference_delay` is a promise about
the future -- how many actions the arm will have executed before the new chunk takes
over -- and a loop that swaps chunks ON ARRIVAL can only estimate it from measured
latency, typically a rolling maximum of recent delays. This loop retires each chunk at a FIXED
index, so d is exact and known at submit time, and RTC optimises for the seam that
actually happens. A wrong d is not an error; it is a quiet mis-optimisation.

So this file owns the model, the chunker, the in-flight request and the per-episode
reset, while evals/rtc.py owns timing and bookkeeping. That split is not tidiness: the
inference has to be drained BEFORE the chunker is reset, or a trial's last inference
lands after the reset and pins the next trial to a chunk nobody is executing -- and
only the object holding both can order those two.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import pathlib
import statistics
import time
from typing import Any

import numpy as np

from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation
from inspect_robots_so101.config import action_box, observation_space

ROOT = pathlib.Path(__file__).resolve().parent.parent

# See the module docstring. Properties of the CHECKPOINT and of the loop it was tuned
# in, not preferences -- both taken from webui/openpi_worker.py's working defaults.
USE_DEGREES = True
REPLAN_INTERVAL = 15

# RTC tuning, also from webui/openpi_worker.py's rtc mode. "exp" releases the prefix
# constraint later and more gently than a linear ramp. "identity" is the cheap Jacobian
# the LeRobot reference uses; "full" is the true VJP at roughly double the inference
# time, which on this rig would eat the very window RTC exists to hide.
RTC_SCHEDULE = "exp"
RTC_MAX_GUIDANCE = 5.0
RTC_JACOBIAN = "identity"

# The checkpoint the web UI's dropdown shows as the working one. /checkpoints is the
# read-only mount of ~/Downloads/hf_models (docker-compose.yml).
DEFAULT_CHECKPOINT = "/checkpoints/openpi_pi05_lora_cap_to_cup_200"


def _evaluate_module():
  """scripts/openpi/evaluate.py, imported by path.

  Not importable as a package (scripts/ has no __init__ and openpi/ shadows the
  installed openpi), so it is loaded the way webui/openpi_worker.py loads it.
  """
  path = ROOT / "scripts" / "openpi" / "evaluate.py"
  spec = importlib.util.spec_from_file_location("agent101_openpi_evaluate", path)
  if spec is None or spec.loader is None:
    raise ImportError(f"could not load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _refuse_delta_actions(train_config: Any) -> None:
  """RTC needs a prefix that means the same thing in the new chunk's frame.

  openpi's `extra_delta_transform` encodes actions as deltas from the FIRST STATE of
  each chunk (openpi/training/config.py). A prefix carried over from a chunk anchored
  at an older state is then not comparable to the one being sampled, and guiding on it
  pins the arm to a trajectory offset by the difference between the two anchors.
  Supporting it needs a re-anchoring pass -- subtract the old anchor, add the new --
  and refusing is the honest alternative to guessing.

  This bench's checkpoint sets it False (pi05_soarm101_lora_cap_to_cup), so the prefix
  is directly comparable across frames and nothing here has to be re-anchored.
  """
  data = getattr(train_config, "data", None)
  if getattr(data, "extra_delta_transform", False):
    raise ValueError(
      f"{getattr(train_config, 'name', '?')} trains on DELTA actions "
      "(extra_delta_transform=True), so a prefix from an earlier chunk is anchored at "
      "a different state and cannot guide this one without re-anchoring. Run this "
      "checkpoint with --mode async or --mode sync."
    )


class OpenPiPolicy:
  """An openpi pi0.5 checkpoint, driving the 6-D SO-ARM joint contract."""

  def __init__(
    self,
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    config_name: str | None = None,
    cameras: tuple[str, ...] = ("front", "grip"),
    cam_height: int = 480,
    cam_width: int = 640,
    action_horizon: int = 50,
    replan_interval: int = REPLAN_INTERVAL,
    rtc: bool = False,
    rtc_schedule: str = RTC_SCHEDULE,
    rtc_max_guidance: float = RTC_MAX_GUIDANCE,
    rtc_jacobian: str = RTC_JACOBIAN,
    rtc_horizon: int | None = None,
    warmup: bool = True,
  ) -> None:
    """Load the checkpoint. Slow: this builds a JAX policy and JIT-compiles it.

    `rtc` decides whether this policy can honour an action prefix at all. It is not
    the same choice as whether inference overlaps execution -- that one belongs to the
    controller (evals/rtc.py) -- but RTC is meaningless without the overlap, so
    evals/run.py sets the two together off `--mode`.
    """
    self._eval = _evaluate_module()
    self._checkpoint = checkpoint
    self._prompt: str | None = None
    self._cameras = cameras

    from openpi.policies import policy_config
    from openpi.training import config as pi0_config

    # The checkpoint knows which dataset it trained on; asking it beats a flag that
    # can disagree with the assets/ directory (evaluate.infer_config has the long note).
    name = config_name or self._eval.infer_config(checkpoint)
    self._config_name = name
    train_config = pi0_config.get_config(name)
    self._policy = policy_config.create_trained_policy(train_config, checkpoint)

    # RealTimeChunker wraps the policy rather than replacing it: an unguided infer()
    # still has to go THROUGH it, because that is what keeps `_prev` -- the chunk the
    # next prefix is taken from -- in step with what the arm is executing.
    self._chunker = None
    self.rtc_settings: dict[str, Any] = {}
    if rtc:
      _refuse_delta_actions(train_config)
      from openpi.policies.rtc import RealTimeChunker

      # No prefix_attention_horizon: the default is the whole overlap with the
      # chunk being retired, which is PI's own value (action_horizon minus the
      # replan interval) and needs no telling what the replan interval is.
      self._chunker = RealTimeChunker(
        self._policy,
        prefix_attention_horizon=rtc_horizon,
        prefix_attention_schedule=rtc_schedule,
        max_guidance_weight=rtc_max_guidance,
        jacobian=rtc_jacobian,
      )
      self.rtc_settings = {
        "schedule": rtc_schedule,
        "max_guidance_weight": rtc_max_guidance,
        "jacobian": rtc_jacobian,
        "prefix_attention_horizon": rtc_horizon or "leftover",
      }

    self.info = PolicyInfo(
      name="openpi",
      # No bounds: the embodiment owns the safety limits, as LeRobotPolicy also does.
      action_space=action_box(),
      observation_space=observation_space(
        cam_height, cam_width, cameras, use_degrees=USE_DEGREES
      ),
    )
    # replan_interval is the load-bearing one: eval() builds
    # DefaultController(policy.config.replan_interval), and None there means "play the
    # whole 50-action chunk open-loop".
    self.config = PolicyConfig(
      action_horizon=action_horizon, replan_interval=replan_interval
    )
    self._replan = replan_interval
    self._step = 0
    self._messages: list[dict[str, Any]] = []

    # ONE worker thread, and never two chunks in flight: the JAX policy is not
    # re-entrant and the loop has no use for a second. The pool belongs to the policy
    # rather than to the controller because it has to outlive a trial -- see reset().
    self._pool = concurrent.futures.ThreadPoolExecutor(
      max_workers=1, thread_name_prefix="openpi-infer"
    )
    self._pending: concurrent.futures.Future[ActionChunk] | None = None
    self._reset_trial_stats()
    if warmup:
      self._warmup(cam_height, cam_width)

  def _reset_trial_stats(self) -> None:
    """Per-trial inference counters, summarised into the log by on_trial_end()."""
    self._latencies: list[float] = []
    self._guided = 0
    self._absmax = 0.0
    self._late = 0
    self._stall_s = 0.0

  def _warmup(self, cam_height: int, cam_width: int) -> None:
    """Pay the JIT compile at load, on synthetic pixels, with nothing commanded.

    JAX traces and compiles on the first infer() -- tens of seconds -- and a compile
    inside the control loop is a stalled arm. webui/openpi_worker.py warms up on a real
    observation before it will accept a `run`; here the policy is built before the
    embodiment exists, so the shapes come from the declared spaces instead. Only shapes
    and dtypes reach the tracer, and no action leaves this method.

    THE GUIDED SAMPLER IS A SECOND TRACE. A prefix changes the argument tree, so it
    compiles separately, and without this the first guided call would be the first
    replan of the first trial -- mid-motion, arm holding its last command, for as long
    as the compile takes. Hence two chunker calls: the first has no prefix to honour
    and exists only to give the chunker one.
    """
    dim = self.info.action_space.dim
    obs = {
      "prompt": "warmup",
      "observation/state": np.zeros(dim, dtype=np.float32),
      "observation/images/front": np.zeros((cam_height, cam_width, 3), dtype=np.uint8),
      "observation/images/side": np.zeros((cam_height, cam_width, 3), dtype=np.uint8),
    }
    # THE WARMUP MUST NOT BE OBSERVABLE. openpi's Policy.infer splits self._rng on
    # every call, so throwaway inferences would shift the sampling noise of every
    # real one -- and a run's first trial, which starts from a fresh key 0, would
    # stop reproducing the logs recorded before this method existed. The key is put
    # back afterwards, so the only thing warmup leaves behind is the compile. (A
    # torch-backed policy has no _rng; then there is nothing to restore.)
    rng = getattr(self._policy, "_rng", None)
    started = time.perf_counter()
    try:
      if self._chunker is None:
        self._policy.infer(obs)
      else:
        self._chunker.infer(obs, prefix_start=0, inference_delay=0)
        self._chunker.infer(obs, prefix_start=1, inference_delay=1)
        # That second call left a prefix from synthetic pixels behind.
        self._chunker.reset()
    finally:
      if rng is not None:
        self._policy._rng = rng
    guided = ", guided sampler included" if self._chunker is not None else ""
    print(
      f"[openpi] warmup {time.perf_counter() - started:.0f} s (jit compile{guided})",
      flush=True,
    )

  # --- the async seam the controller drives -------------------------------------

  @property
  def supports_prefix(self) -> bool:
    """True when this policy can be ASKED to honour a prefix, i.e. built with rtc.

    A distinct name from "can this run asynchronously", which every policy here can:
    overlapping inference with execution is the controller's business, honouring the
    prefix is the model's. Conflating the two would let an unguided policy claim a
    guided seam -- the same split any serving stack has to draw between "will accept a
    prefix" and "drives its own realtime loop".
    """
    return self._chunker is not None

  @property
  def supports_async(self) -> bool:
    """True when submit()/collect() can be used instead of act().

    Always, for this policy: the inference thread exists whether or not the loop
    chooses to overlap. Declared rather than assumed so a controller can refuse an
    LLM-agent policy loudly instead of calling a submit() that is not there.
    """
    return True

  @property
  def busy(self) -> bool:
    """True while an inference is in flight."""
    return self._pending is not None

  def submit(
    self,
    observation: Observation,
    *,
    at_step: int | None,
    prefix_start: int = 0,
    inference_delay: int = 0,
  ) -> None:
    """Start one inference on the worker thread. Returns immediately.

    `prefix_start` is the index, in the chunk now executing, of the action that lines
    up with index 0 of the chunk being sampled -- i.e. the action at the tick this
    observation was captured. `inference_delay` is how many actions the arm will have
    executed by the time this chunk takes over. Both are ignored without a chunker.

    The OBSERVATION IS PACKED HERE, on the calling thread, deliberately: it is the
    caller's snapshot and packing it later would race the next step's read.
    """
    if self._pending is not None:
      raise RuntimeError("an inference is already in flight; collect() it first")
    packed = self._pack(observation)
    self._pending = self._pool.submit(
      self._infer,
      packed,
      at_step=at_step,
      prefix_start=prefix_start,
      inference_delay=inference_delay,
    )

  def collect(self, *, block: bool = True) -> ActionChunk | None:
    """The submitted chunk, or None when it is not ready and `block` is False."""
    pending = self._pending
    if pending is None:
      raise RuntimeError("collect() with no inference in flight")
    if not block and not pending.done():
      return None
    try:
      return pending.result()
    finally:
      self._pending = None

  def note_replan(self, *, late: bool, stall_s: float = 0.0) -> None:
    """The controller reporting a seam it had to wait for; see evals/rtc.py.

    The loop is the only thing that knows a chunk arrived late, and this policy is the
    only thing that gets a per-trial log hook, so the number is handed over rather
    than counted in two places.
    """
    if late:
      self._late += 1
    self._stall_s += stall_s

  def _drain(self) -> None:
    """Discard any in-flight inference, waiting out one that is already running."""
    pending, self._pending = self._pending, None
    if pending is not None and not pending.cancel():
      try:
        pending.result()
      except Exception:  # noqa: BLE001 - a discarded chunk's failure is nobody's trial
        pass

  def close(self) -> None:
    """Release the inference thread. Idempotent; safe to call without a run."""
    self._drain()
    self._pool.shutdown(wait=False)

  def reset(self, scene: Scene) -> None:
    """Take the scene's instruction as openpi's prompt, and start a fresh trial.

    DRAINS, THEN RESETS, in that order. A trial's last inference is usually still in
    flight when the horizon or the operator ends it; if it lands after the chunker is
    reset, RealTimeChunker keeps ITS chunk as the prefix and the next trial's first
    guided sample is pinned to actions from the previous trial -- a stale plan nobody
    is executing, on a table that has been rearranged in between.
    """
    self._drain()
    if self._chunker is not None:
      self._chunker.reset()
    self._prompt = scene.instruction
    self._step = 0
    self._messages = []
    self._reset_trial_stats()

  def _pack(self, observation: Observation) -> tuple[dict[str, Any], np.ndarray]:
    """The observation in openpi's layout, plus the state the chunk is checked against."""
    state = np.asarray(observation.state["joint_pos"], dtype=np.float32)
    images = observation.images
    missing = [c for c in self._cameras if c not in images]
    if missing:
      raise KeyError(
        f"observation missing camera(s) {missing}; the embodiment was built with "
        f"{sorted(images)} -- check CAM_*_INDEX in .env"
      )
    obs: dict[str, Any] = {
      "prompt": self._prompt or (observation.instruction or ""),
      "observation/state": state,
      "observation/images/front": images["front"],
      # grip fills openpi's "side" slot, as in scripts/openpi/evaluate.py.
      "observation/images/side": images["grip"],
    }
    return obs, state

  def act(self, observation: Observation) -> ActionChunk:
    """One blocking inference on the calling thread: the synchronous path.

    This is what DefaultController calls, so `--mode sync` is this policy plus that
    controller -- the path every eval log recorded before RTC existed, left alone on
    purpose so those numbers stay comparable. (One thing did change for it: the JIT
    compile is now paid at load rather than inside the first trial. That moves wall
    clock around; it does not change an action.)

    It REFUSES for an RTC policy instead of quietly dropping the guidance. act() is
    handed an observation and nothing else: it does not know which actions have been
    executed, so the only prefix it could offer is a wrong one, and a silently
    unguided seam is the failure this file exists to remove.
    """
    if self._chunker is not None:
      raise RuntimeError(
        "this policy was built with rtc=True, so its chunks must be requested through "
        "submit()/collect() by evals.rtc.AsyncChunkController -- the only caller that "
        "knows which actions the arm has committed. act() cannot supply a prefix."
      )
    return self._infer(
      self._pack(observation), at_step=None, prefix_start=0, inference_delay=0
    )

  def _infer(
    self,
    packed: tuple[dict[str, Any], np.ndarray],
    *,
    at_step: int | None,
    prefix_start: int,
    inference_delay: int,
  ) -> ActionChunk:
    """One inference. Runs on the worker thread under submit(); touches no hardware.

    Everything worth knowing about the call is recorded HERE rather than by the
    caller, because the caller is a controller that never sees `outputs` -- and
    because the chunker's own clipping of d and s is only readable right after it.
    """
    obs, state = packed
    # Read BEFORE the call: infer() sets the prefix for the next one.
    guided = self._chunker is not None and self._chunker.has_prefix
    started = time.perf_counter()
    if self._chunker is None:
      outputs = self._policy.infer(obs)
    else:
      outputs = self._chunker.infer(
        obs, prefix_start=prefix_start, inference_delay=inference_delay
      )
    latency = time.perf_counter() - started
    actions = np.asarray(outputs["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < state.shape[0]:
      raise ValueError(
        f"expected a (horizon, >={state.shape[0]}) action chunk, got {actions.shape}"
      )
    meta: dict[str, Any] = {
      "checkpoint": self._checkpoint,
      "config": self._config_name,
      "guided": guided,
    }
    if guided:
      assert self._chunker is not None
      # What the chunker ACTUALLY used, after clipping d and s to the overlap it had.
      meta |= {
        "prefix_start": prefix_start,
        "inference_delay": self._chunker.last_delay,
        "prefix_horizon": self._chunker.last_horizon,
      }
      self._guided += 1
    raw = outputs.get("raw_actions")
    if raw is not None:
      # Guidance is ADDITIVE and can push the normalized chunk past the +/-1 the model
      # was trained in -- measurably so, on other rigs. Nothing downstream would show
      # it here: the arm is commanded in degrees and clamped twice, so
      # an overshoot reads as a joint sitting on its limit. Reported per inference and
      # summarised per trial instead of being assumed away.
      absmax = float(np.abs(np.asarray(raw)).max())
      meta["raw_absmax"] = round(absmax, 4)
      self._absmax = max(self._absmax, absmax)
    self._latencies.append(latency)
    self._record(state, actions, latency, at_step=at_step, meta=meta)
    return ActionChunk(
      actions=[Action(data=a[: state.shape[0]]) for a in actions],
      inference_latency_s=latency,
      meta=meta,
    )

  # The HTML report's camera player is built from the POLICY TRANSCRIPT, not from the
  # frames directory: _frame_references() walks the transcript looking for a text part
  # matching "camera '<name>' (step N):" immediately followed by the placeholder below,
  # and resolves each to <frames_dir>/<trial_prefix>_<name>_<step:06d>.npy. A policy
  # that reports nothing therefore renders an empty flipbook however many frames the
  # embodiment stored -- which is exactly what this policy did until now.
  _PLACEHOLDER = "[image omitted: streamed camera frame]"

  def _record(
    self,
    state: np.ndarray,
    actions: np.ndarray,
    latency: float,
    *,
    at_step: int | None = None,
    meta: dict[str, Any] | None = None,
  ) -> None:
    """Append one inference to the transcript, in the shape the report parses.

    `at_step` is the trial step this observation was captured at, and the async path
    passes it because it knows: an overlapped inference is submitted mid-window, so
    the step it saw is NOT a multiple of the replan interval, and naming the wrong one
    would show the operator a frame the model never saw.

    Without it (the synchronous path) the counter advances by the REPLAN INTERVAL
    rather than by the chunk length, because that is how many of the predicted actions
    the controller executes before calling act() again. Frames are stored every step,
    so every referenced step exists; referencing chunk-length steps instead would
    point past the trial's end.

    Images are named, never embedded -- the frame sidecars already hold the pixels, and
    the transcript hook is explicitly required not to duplicate them.
    """
    step = self._step if at_step is None else at_step
    parts: list[dict[str, Any]] = []
    for camera in self._cameras:
      parts.append({"type": "text", "text": f"camera '{camera}' (step {step}):"})
      parts.append({"type": "text", "text": self._PLACEHOLDER})
    parts.append(
      {"type": "text", "text": "joint_pos: " + np.array2string(state, precision=2)}
    )
    detail = ""
    if meta and meta.get("guided"):
      detail += (
        f"; guided from index {meta['prefix_start']} of the previous chunk, "
        f"d={meta['inference_delay']} s={meta['prefix_horizon']}"
      )
    if meta and "raw_absmax" in meta:
      detail += f"; raw absmax {meta['raw_absmax']:.2f}"
    self._messages.append({"role": "user", "content": parts})
    self._messages.append(
      {
        "role": "assistant",
        "content": (
          f"predicted {len(actions)} actions, executing {self._replan}; "
          f"first target {np.array2string(actions[0][: state.shape[0]], precision=2)}; "
          f"{latency * 1000:.0f} ms{detail}"
        ),
      }
    )
    if at_step is None:
      self._step += self._replan

  def transcript(self) -> list[dict[str, Any]]:
    """This trial's inferences, for the log and the report's camera player.

    Called once per trial at trial end. Must be idempotent, must not mutate policy
    state, and must not alias it -- hence the copy.
    """
    return [dict(message) for message in self._messages]

  def on_trial_end(self, record: Any, log_dir: str, run_stamp: str) -> None:
    """Summarise this trial's inference behaviour into the log, and onto the console.

    `record.metadata` is the framework's per-policy trial slot and lands in the log as
    `trial_metadata`, which is what lets a reader tell an RTC trial from a synchronous
    one WITHOUT trusting the flags someone remembers typing. The console line is for
    the operator standing at the arm: a run whose seams are late is a run whose
    control rate has quietly dropped below the 30 Hz the checkpoint was tuned at.
    """
    if not self._latencies:
      return
    mean_ms = statistics.mean(self._latencies) * 1000
    max_ms = max(self._latencies) * 1000
    summary: dict[str, Any] = {
      "mode": "rtc" if self._chunker is not None else "plain",
      "inferences": len(self._latencies),
      "guided": self._guided,
      "mean_latency_ms": round(mean_ms, 1),
      "max_latency_ms": round(max_ms, 1),
      "late_seams": self._late,
      "stall_s": round(self._stall_s, 2),
      "raw_absmax": round(self._absmax, 4),
    }
    if self._chunker is not None:
      summary["rtc"] = dict(self.rtc_settings)
    metadata = getattr(record, "metadata", None)
    if isinstance(metadata, dict):
      metadata["openpi"] = summary
    print(
      f"[openpi] {len(self._latencies)} inferences, {self._guided} guided, "
      f"{mean_ms:.0f} ms mean / {max_ms:.0f} ms max, {self._late} late seam(s) "
      f"costing {self._stall_s:.2f} s, raw absmax {self._absmax:.2f}",
      flush=True,
    )
