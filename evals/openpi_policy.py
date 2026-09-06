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

THE UNITS ARE NORMALIZED, NOT DEGREES, and this is the single most dangerous thing in
this file. These checkpoints were trained on datasets recorded before lerobot 0.6.1
made degrees the default, so their joint values are lerobot's normalized +/-100.
evaluate.py defaults `--units normalized` for the same reason. Nothing downstream will
catch a mistake here: the embodiment commands policy output verbatim after clamping,
and Inspect Robots' compatibility check compares state KEYS, not units -- so degrees
fed to a normalized-trained policy is a silent 4x error in joint space, not an
exception. `USE_DEGREES = False` here and `rig.so_arm_config(use_degrees=False)` in the
runner are one decision recorded twice; changing either alone is the bug.

A chunk is returned whole. openpi predicts an action horizon per inference and the
framework logs it; how much of it gets executed before re-planning is the rollout's
business, which is the same division of labour evaluate.py's `--actions` implements.
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

# See the module docstring. This is a property of the CHECKPOINTS, not a preference.
USE_DEGREES = False

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
    action_horizon: int = 15,
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
    self.config = PolicyConfig(action_horizon=action_horizon)

  def reset(self, scene: Scene) -> None:
    """Take the scene's instruction as openpi's prompt for this trial."""
    self._prompt = scene.instruction

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
    return ActionChunk(
      actions=[Action(data=a[: state.shape[0]]) for a in actions],
      inference_latency_s=latency,
      meta={"checkpoint": self._checkpoint, "config": self._config_name},
    )
