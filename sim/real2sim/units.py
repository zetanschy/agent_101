"""LeRobot joint values <-> URDF joint radians, and the real joint limits.

The dataset was recorded by lerobot 0.6.1's SOFollower with use_degrees=True (the
default; joint_units report, sections 1-2). Its bus normalises each motor as:

    arm joints, DEGREES:   v = (P - mid) * 360 / 4095,   mid = (range_min + range_max) / 2
                           goal P = int(v * 4095 / 360 + mid)        (not clamped)
    gripper, RANGE_0_100:  pct = (clip(P, min, max) - min) / (max - min) * 100
                           goal P = int(pct / 100 * (max - min) + min)

with P the Present_Position tick (0..4095) and the ranges from the follower
calibration file (paths.FOLLOWER_CALIBRATION). MEASURED: every recorded state value
decodes to an integer tick through the follower calibration to 1e-4 tick, so these
formulas are exact for this dataset (tests/test_units.py re-checks it).

ARM -> URDF. q_rad = radians(v + offset_deg). The five offsets default to zero: the
repo convention is identity degrees (sim/sim_agent101/kinematics.py:80-91), and the
URDF zero is the range midpoint, not the homing pose (those differ by up to 7.7 deg
on the elbow). They are a scene parameter because the data suggest they are not
zero: a front-camera silhouette fit puts the elbow at about -2.8 deg, a grasp-height
fit at -2.1 deg. CALIB fits them. (360/4095 vs the physical 360/4096 per tick is a
0.024 % scale difference, 0.04 deg at 165 deg, and is ignored as lerobot ignores it.)

GRIPPER -> URDF Jaw. jaw_deg = a + b * pct, with two candidates (joint_units §4):

    tick_physical   a = -13.258, b = 1.33594   the horn drives the jaw 1:1, so 1 % is
                    15.2 ticks = 1.336 deg; anchored where the jaws shut on nothing
                    (present 2049 ticks, all four episodes) = URDF Jaw -11.5 deg, where
                    the finger meshes touch (0.10 mm gap at -11.5, 1.94 mm at -10).
                    DEFAULT (provisional).
    kinematics_py   a = -10.0,   b = 1.1       sim/sim_agent101/kinematics.py:40-55:
                    0-100 % spread over the URDF's -10..100 deg limits. Leaves the
                    sim's shut jaws 4 mm apart.

They cross at 13.4 %, near the ~15 % hold, so only the far ends distinguish them;
the moving jaw is visible in the wrist camera and CALIB fits a and b.

LIMITS. `limits_rad` returns the REAL joint range: the calibration's
Min/Max_Position_Limit, which the servo firmware enforces (a goal past them is
clamped), pushed through the same maps. Four of those exceed the URDF limits the
MJCF carries (Pitch +-104.3 vs +-100, Elbow +97.9 vs +90, Wrist_Pitch +-104.1 vs
+-95, Wrist_Roll +-168.6 vs +-160 deg) and the dataset really goes there (rest pose
Elbow 95-96 deg), so engines must widen their models to these. Shoulder_pan's real
range (+-104.1) is narrower than the URDF's +-110. Jaw: pct 0..100 maps to about
-13.3..+120 deg under tick_physical, but the jaws physically touch at -11.5 deg
(JAW_TOUCH_DEG) so pct 0 is never reached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths

LEROBOT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
URDF_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
N_ARM = 5
GRIPPER = 5  # index of the gripper / Jaw in every 6-vector

LEROBOT_MAX_RES = 4095  # lerobot divides by model resolution - 1 (motors_bus.py:876-877)
TICKS_PER_REV = 4096  # STS3215 physical resolution

# measured: finger meshes of moving_jaw_so101_v1 and wrist_roll_follower_so101_v1 touch
JAW_TOUCH_DEG = -11.5
# measured: present tick when the jaws shut on nothing, all four episodes (2049-2051)
JAW_SHUT_TICK = 2049

GRIPPER_CANDIDATES = {
    "tick_physical": {"a_deg": -13.258, "b_deg_per_pct": 1.33594,
                      "source": "estimated: horn 1:1, 15.2 ticks/% x 360/4096, anchored at "
                                "2049 ticks = mesh touch -11.5 deg (joint_units report §4)"},
    "kinematics_py": {"a_deg": -10.0, "b_deg_per_pct": 1.1,
                      "source": "READ sim/sim_agent101/kinematics.py:40-55 (URDF -10..100 deg over 0..100 %)"},
}


def tick_physical_map(calib: dict | None = None) -> tuple[float, float]:
    """(a_deg, b_deg_per_pct) of the tick_physical candidate, from the calibration."""
    g = (calib or calibration())["gripper"]
    ticks_per_pct = (g["range_max"] - g["range_min"]) / 100.0
    b = ticks_per_pct * 360.0 / TICKS_PER_REV
    a = JAW_TOUCH_DEG + (g["range_min"] - JAW_SHUT_TICK) * 360.0 / TICKS_PER_REV
    return a, b


@dataclass(frozen=True)
class Units:
    """The lerobot <-> URDF map of one scene. Arrays are (..., 6) in joint order."""

    offsets_deg: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
    grip_a_deg: float = GRIPPER_CANDIDATES["tick_physical"]["a_deg"]
    grip_b_deg_per_pct: float = GRIPPER_CANDIDATES["tick_physical"]["b_deg_per_pct"]

    @classmethod
    def from_scene(cls, scene) -> "Units":
        r = scene["robot"]
        return cls(tuple(float(x) for x in r["joint_offsets"]["deg"]),
                   float(r["gripper_map"]["a_deg"]), float(r["gripper_map"]["b_deg_per_pct"]))

    def to_urdf(self, v) -> np.ndarray:
        """LeRobot values (deg x5, gripper %) -> URDF radians."""
        v = np.asarray(v, dtype=float)
        q = np.empty_like(v)
        q[..., :N_ARM] = np.radians(v[..., :N_ARM] + np.asarray(self.offsets_deg))
        q[..., GRIPPER] = self.jaw_rad(v[..., GRIPPER])
        return q

    def to_lerobot(self, q) -> np.ndarray:
        """URDF radians -> LeRobot values (deg x5, gripper %)."""
        q = np.asarray(q, dtype=float)
        v = np.empty_like(q)
        v[..., :N_ARM] = np.degrees(q[..., :N_ARM]) - np.asarray(self.offsets_deg)
        v[..., GRIPPER] = self.jaw_pct(q[..., GRIPPER])
        return v

    def jaw_rad(self, pct):
        return np.radians(self.grip_a_deg + self.grip_b_deg_per_pct * np.asarray(pct, dtype=float))

    def jaw_pct(self, jaw_rad):
        return (np.degrees(np.asarray(jaw_rad, dtype=float)) - self.grip_a_deg) / self.grip_b_deg_per_pct


def calibration(path: str | Path | None = None) -> dict:
    """The follower calibration JSON: {motor: {id, drive_mode, homing_offset, range_min, range_max}}."""
    return json.loads(Path(path or paths.FOLLOWER_CALIBRATION).read_text())


def _cal(calib: dict):
    lo = np.array([calib[j]["range_min"] for j in LEROBOT_JOINTS], dtype=float)
    hi = np.array([calib[j]["range_max"] for j in LEROBOT_JOINTS], dtype=float)
    if any(calib[j].get("drive_mode", 0) for j in LEROBOT_JOINTS):
        raise NotImplementedError("drive_mode != 0 inverts a joint; not modelled (all 0 here)")
    return lo, hi, (lo + hi) / 2.0


def ticks_to_lerobot(ticks, calib: dict | None = None) -> np.ndarray:
    """Present ticks (..., 6) -> lerobot values, exactly as the bus normalises them."""
    lo, hi, mid = _cal(calib or calibration())
    P = np.asarray(ticks, dtype=float)
    v = np.empty_like(P)
    v[..., :N_ARM] = (P[..., :N_ARM] - mid[:N_ARM]) * 360.0 / LEROBOT_MAX_RES
    v[..., GRIPPER] = (np.clip(P[..., GRIPPER], lo[GRIPPER], hi[GRIPPER]) - lo[GRIPPER]) / (hi[GRIPPER] - lo[GRIPPER]) * 100.0
    return v


def lerobot_to_ticks(v, calib: dict | None = None) -> np.ndarray:
    """lerobot values -> ticks as floats (no rounding: exact data land on integers)."""
    lo, hi, mid = _cal(calib or calibration())
    v = np.asarray(v, dtype=float)
    P = np.empty_like(v)
    P[..., :N_ARM] = v[..., :N_ARM] * LEROBOT_MAX_RES / 360.0 + mid[:N_ARM]
    P[..., GRIPPER] = v[..., GRIPPER] / 100.0 * (hi[GRIPPER] - lo[GRIPPER]) + lo[GRIPPER]
    return P


def limits_lerobot(calib: dict | None = None) -> np.ndarray:
    """(6, 2) real range in lerobot units: the firmware position limits, normalised."""
    lo, hi, _ = _cal(calib or calibration())
    return np.stack([ticks_to_lerobot(lo, calib), ticks_to_lerobot(hi, calib)], axis=1)


def limits_rad(units: Units | None = None, calib: dict | None = None) -> np.ndarray:
    """(6, 2) real range in URDF radians [lower, upper], for widening engine models."""
    lim = (units or Units()).to_urdf(limits_lerobot(calib).T).T
    return np.sort(lim, axis=1)  # a negative gripper slope would swap the ends


def urdf_limits_rad(mjcf: str | Path | None = None) -> np.ndarray:
    """(6, 2) the joint ranges the MJCF itself carries (the URDF's), read from the XML."""
    import xml.etree.ElementTree as ET

    rng = {j.get("name"): j.get("range") for j in ET.parse(mjcf or paths.MJCF).getroot().iter("joint")}
    return np.array([[float(x) for x in rng[n].split()] for n in URDF_JOINTS])


def describe(units: Units | None = None, calib: dict | None = None) -> str:
    """One line per joint: real range vs the URDF's, for `./robot real2sim core info`."""
    real, urdf = np.degrees(limits_rad(units, calib)), np.degrees(urdf_limits_rad())
    return "\n".join(f"  {n:12s} real {r[0]:+8.2f} .. {r[1]:+8.2f} deg   URDF {u[0]:+8.2f} .. {u[1]:+8.2f}"
                     for n, r, u in zip(URDF_JOINTS, real, urdf))

