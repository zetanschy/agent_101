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
"""

from __future__ import annotations

import importlib.util
import pathlib
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
  ) -> None:
    """Load the checkpoint. Slow: this builds a JAX policy and JIT-compiles it."""
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
    self._policy = policy_config.create_trained_policy(
      pi0_config.get_config(name), checkpoint
    )

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

  def reset(self, scene: Scene) -> None:
    """Take the scene's instruction as openpi's prompt, and start a fresh transcript."""
    self._prompt = scene.instruction
    self._step = 0
    self._messages = []

  def act(self, observation: Observation) -> ActionChunk:
    """One inference: pack the observation openpi's way, return the whole chunk."""
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
    started = time.perf_counter()
    actions = np.asarray(self._policy.infer(obs)["actions"], dtype=np.float64)
    latency = time.perf_counter() - started
    if actions.ndim != 2 or actions.shape[1] < state.shape[0]:
      raise ValueError(
        f"expected a (horizon, >={state.shape[0]}) action chunk, got {actions.shape}"
      )
    self._record(state, actions, latency)
    return ActionChunk(
      actions=[Action(data=a[: state.shape[0]]) for a in actions],
      inference_latency_s=latency,
      meta={"checkpoint": self._checkpoint, "config": self._config_name},
    )

  # The HTML report's camera player is built from the POLICY TRANSCRIPT, not from the
  # frames directory: _frame_references() walks the transcript looking for a text part
  # matching "camera '<name>' (step N):" immediately followed by the placeholder below,
  # and resolves each to <frames_dir>/<trial_prefix>_<name>_<step:06d>.npy. A policy
  # that reports nothing therefore renders an empty flipbook however many frames the
  # embodiment stored -- which is exactly what this policy did until now.
  _PLACEHOLDER = "[image omitted: streamed camera frame]"

  def _record(self, state: np.ndarray, actions: np.ndarray, latency: float) -> None:
    """Append one inference to the transcript, in the shape the report parses.

    The step counter advances by the REPLAN INTERVAL rather than by the chunk length,
    because that is how many of the predicted actions the controller actually executes
    before calling act() again. Frames are stored every step, so every referenced step
    exists; referencing chunk-length steps instead would point past the trial's end.

    Images are named, never embedded -- the frame sidecars already hold the pixels, and
    the transcript hook is explicitly required not to duplicate them.
    """
    parts: list[dict[str, Any]] = []
    for camera in self._cameras:
      parts.append({"type": "text", "text": f"camera '{camera}' (step {self._step}):"})
      parts.append({"type": "text", "text": self._PLACEHOLDER})
    parts.append(
      {"type": "text", "text": "joint_pos: " + np.array2string(state, precision=2)}
    )
    self._messages.append({"role": "user", "content": parts})
    self._messages.append(
      {
        "role": "assistant",
        "content": (
          f"predicted {len(actions)} actions, executing {self._replan}; "
          f"first target {np.array2string(actions[0][: state.shape[0]], precision=2)}; "
          f"{latency * 1000:.0f} ms"
        ),
      }
    )
    self._step += self._replan

  def transcript(self) -> list[dict[str, Any]]:
    """This trial's inferences, for the log and the report's camera player.

    Called once per trial at trial end. Must be idempotent, must not mutate policy
    state, and must not alias it -- hence the copy.
    """
    return [dict(message) for message in self._messages]
