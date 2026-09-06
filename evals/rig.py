"""This bench, as Inspect Robots needs to see it.

One place that turns agent_101's existing truth -- .env, the lerobot calibration file,
config/home_pose.json -- into an `SOArmConfig`. Nothing here is a new setting to keep
in step with the rest of the repo; it is a translation.

THE JOINT LIMITS ARE A SAFETY CLAMP, and inspect-robots-so101 is explicit that its
defaults are placeholders (+/-180 degrees) rather than limits: every command is clipped
to them inside `step()`, beneath any approver, so a placeholder is the difference
between a backstop and a formality. They are computed here from the arm's own
calibration rather than typed in.

lerobot's degree mode is, in `MotorsBus._normalize`:

    degrees = (raw - mid) * 360 / max_res,  mid = (range_min + range_max) / 2

so a joint's travel is symmetric about zero and its half-span is the limit. `max_res`
is the model resolution minus one, 4095 for the STS3215. Applied to
zetans_follower.json that gives, in degrees:

    shoulder_pan  +/-104.13    wrist_flex  +/-104.09
    shoulder_lift +/-104.31    wrist_roll  +/-168.57
    elbow_flex     +/-97.89    gripper      0..100 (RANGE_0_100, never degrees)

Two independent checks that the formula is the right one, rather than merely a formula:
the URDF's own limits are +/-110, +/-100, -100..90, +/-95, +/-160, which bracket these
within a couple of degrees; and config/home_pose.json's recorded shoulder_lift is
-104.31, i.e. the arm was homed sitting exactly on this computed limit.

That last coincidence is also a trap. The recorded home is -104.31 and the computed
limit is -104.3077, so the home pose is 0.0023 degrees OUTSIDE the clamp and
SOArmConfig's constructor rejects it outright. HOME_INSET pulls the home pose back
inside instead of widening the clamp to admit it -- a safety bound must not be relaxed
to fit a rounded number, and 0.0023 degrees is far below anything the servo resolves
(one STS3215 count is 0.088 degrees).
"""

from __future__ import annotations

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The 6-D contract inspect-robots-so101 declares, in its order. Do not reorder: the
# embodiment packs actions and state positionally against this.
JOINTS = (
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
)
GRIPPER = "gripper"

# STS3215: 4096 counts of resolution, and lerobot divides by resolution - 1.
_MAX_RES = 4095.0

# Keep the home pose this far inside each limit. Chosen to be smaller than anything
# the hardware can resolve (one count is 0.088 degrees) and larger than the rounding
# in home_pose.json, so it fixes the constructor rejection without moving the arm.
HOME_INSET = 0.01


def _env(key: str, default: str) -> str:
  return os.environ.get(key, default)


def calibration_path() -> pathlib.Path:
  """The calibration lerobot itself would load for this robot_type/robot_id."""
  robot_type = _env("ROBOT_TYPE", "so101_follower")
  robot_id = _env("ROBOT_ID", "zetans_follower")
  return ROOT / "calibration" / "robots" / robot_type / f"{robot_id}.json"


def half_spans() -> dict[str, float]:
  """Each arm joint's calibrated half-travel, in degrees. The limit in degree mode."""
  cal = json.loads(calibration_path().read_text())
  return {
    j: (cal[j]["range_max"] - cal[j]["range_min"]) / 2.0 * 360.0 / _MAX_RES
    for j in JOINTS
    if j != GRIPPER
  }


def joint_limits(use_degrees: bool = True) -> tuple[tuple[float, ...], tuple[float, ...]]:
  """(low, high) in the units the embodiment commands.

  Degree mode is computed from the calibration file, so recalibrating the arm moves
  the safety clamp with it instead of silently invalidating it. NORMALIZED mode needs
  no calibration at all: lerobot maps the calibrated range onto exactly +/-100 by
  construction, so +/-100 IS the calibrated limit there and computing it would only
  add a way to get it wrong.
  """
  if not use_degrees:
    low = tuple(-100.0 if j != GRIPPER else 0.0 for j in JOINTS)
    high = tuple(100.0 for _ in JOINTS)
    return low, high
  cal = json.loads(calibration_path().read_text())
  low: list[float] = []
  high: list[float] = []
  for joint in JOINTS:
    if joint == GRIPPER:
      # lerobot hardcodes the gripper to MotorNormMode.RANGE_0_100 regardless of
      # use_degrees; kinematics.py has the long version of this note.
      low.append(0.0)
      high.append(100.0)
      continue
    entry = cal[joint]
    half_span = (entry["range_max"] - entry["range_min"]) / 2.0
    limit = half_span * 360.0 / _MAX_RES
    low.append(-limit)
    high.append(limit)
  return tuple(low), tuple(high)


def home_pose(use_degrees: bool = True) -> tuple[float, ...]:
  """config/home_pose.json in JOINTS order and the requested units, inside the clamp.

  Returns the same pose `./robot home` and webui/home.py drive to, so an eval trial
  starts where every other tool in this repo starts.

  THE FILE IS IN DEGREES -- webui/home.py records it from a robot built with
  use_degrees=True, and its shoulder_lift of -104.31 could not be anything else,
  since normalized units stop at 100. So normalized mode converts rather than
  reading it straight: degrees and normalized are both linear in the raw count and
  both centred on the calibrated midpoint, so the conversion is one ratio,
  normalized = degrees / half_span * 100. The gripper is RANGE_0_100 in BOTH modes
  and is passed through untouched.
  """
  saved = json.loads((ROOT / "config" / "home_pose.json").read_text())
  low, high = joint_limits(use_degrees)
  spans = half_spans()
  pose: list[float] = []
  for joint, lo, hi in zip(JOINTS, low, high):
    value = float(saved[joint])
    if not use_degrees and joint != GRIPPER:
      value = value / spans[joint] * 100.0
    pose.append(min(max(value, lo + HOME_INSET), hi - HOME_INSET))
  return tuple(pose)


def camera_configs(names: tuple[str, ...]):
  """lerobot camera configs for the requested names, from .env's indices.

  Mirrors webui/app.py's cameras_json so an eval sees the cameras the rest of the
  repo does. Imported lazily because evals/tasks.py must stay importable without
  lerobot for the mock-world smoke test.
  """
  from lerobot.cameras.opencv import OpenCVCameraConfig

  index_env = {
    "front": "CAM_FRONT_INDEX",
    "grip": "CAM_GRIP_INDEX",
    "side": "CAM_SIDE_INDEX",
  }
  defaults = {"front": "4", "grip": "6", "side": "1"}
  width = int(_env("CAM_WIDTH", "640"))
  height = int(_env("CAM_HEIGHT", "480"))
  fps = int(_env("CAM_FPS", "30"))
  fourcc = _env("CAM_FOURCC", "MJPG")
  out = {}
  for name in names:
    if name not in index_env:
      raise ValueError(f"unknown camera {name!r}; known: {sorted(index_env)}")
    out[name] = OpenCVCameraConfig(
      index_or_path=int(_env(index_env[name], defaults[name])),
      width=width,
      height=height,
      fps=fps,
      fourcc=fourcc,
    )
  return out


def so_arm_config(
  cameras: tuple[str, ...] = ("front", "grip"),
  *,
  with_cameras: bool = True,
  settle_tolerance: float | None = 2.0,
  use_degrees: bool = True,
):
  """The `SOArmConfig` for this bench.

  `max_relative_target` is lerobot's per-step slew limit in native motor units and is
  REQUIRED once home_pose is set, because homing sends one absolute command: without
  it the arm slams to the home pose at full speed from wherever it stopped. 10 counts
  per step is 0.88 degrees, matching the leash scripts/robot/policy_bridge.py puts on
  a sim-trained policy for the same reason.

  `settle_tolerance` makes step() wait for the arm to arrive before observing. It is
  off by default upstream to preserve closed-loop VLA cadence, and on by default here
  because the first user of this rig is an LLM agent taking one deliberate action at a
  time -- planning the next move from a pose the arm has not reached is a worse failure
  than a slow step. Pass None to restore upstream cadence for a chunked VLA.

  `use_degrees` MUST match what the policy was trained in, and nothing will tell you
  if it does not: the embodiment commands policy output verbatim after the clamp, and
  Inspect Robots compares state KEYS rather than units, so a mismatch drives the arm
  with numbers that mean something else. The agent policy reads the declared bounds
  and adapts, so degrees is the friendlier choice there. The openpi pi0.5 checkpoints
  in this repo are NORMALIZED -- scripts/openpi/evaluate.py defaults --units to it
  for them, because they were trained on datasets recorded before lerobot 0.6.1 made
  degrees the default. evals/openpi_policy.py carries that default with it.
  """
  from inspect_robots_so101 import SOArmConfig

  low, high = joint_limits(use_degrees)
  return SOArmConfig(
    port=_env("ROBOT_PORT", "/dev/ttyACM1"),
    robot_type=_env("ROBOT_TYPE", "so101_follower"),
    robot_id=_env("ROBOT_ID", "zetans_follower"),
    # The repo's version-controlled calibration, which docker-compose mounts over
    # lerobot's default cache location; passed explicitly so a host run finds it too.
    calibration_dir=str(ROOT / "calibration"),
    cameras=cameras,
    camera_configs=camera_configs(cameras) if with_cameras else None,
    cam_width=int(_env("CAM_WIDTH", "640")),
    cam_height=int(_env("CAM_HEIGHT", "480")),
    control_hz=float(_env("CAM_FPS", "30")),
    joint_low=low,
    joint_high=high,
    home_pose=home_pose(use_degrees),
    max_relative_target=10.0,
    use_degrees=use_degrees,
    disable_torque_on_disconnect=True,
    settle_tolerance=settle_tolerance,
  )
