"""Forward kinematics for the SO-ARM101, from its URDF, with no solver dependency.

lerobot's RobotKinematics wants `placo`, which is not installed in any environment
here and pulls in a whole IK stack we do not need. This is a serial 6-DoF chain
where every joint turns about its own local Z, so FK is a product of fixed origin
transforms and Z rotations -- about forty lines, and it can be checked against the
simulator, which `sim_calibrate_extrinsics.py` does.

The URDF is `so101_new_calib`, the same model name as the root prim of the workshop's
USD, so the sim and this agree by construction rather than by luck.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

URDF = pathlib.Path(__file__).parent / "urdf" / "so101_new_calib.urdf"

# Base to gripper. `Jaw` drives the finger and is not part of this chain.
CHAIN = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _load(urdf: pathlib.Path = URDF) -> dict:
    root = ET.parse(urdf).getroot()
    out = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0")).split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0")).split()]
        T = np.eye(4)
        T[:3, :3] = _rpy(*rpy)
        T[:3, 3] = xyz
        axis = [float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "0 0 1").split()]
        out[j.get("name")] = {"T": T, "axis": np.array(axis)}
    return out


_JOINTS = _load()


def _rot_about(axis: np.ndarray, q: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def fk_gripper(joint_rad: dict[str, float] | np.ndarray) -> np.ndarray:
    """4x4 transform from the robot base to the `gripper` link.

    Accepts a dict keyed by joint name, or an array ordered as CHAIN. Angles in
    RADIANS -- lerobot reports degrees, so convert before calling.
    """
    if not isinstance(joint_rad, dict):
        joint_rad = dict(zip(CHAIN, np.asarray(joint_rad, dtype=float), strict=True))
    T = np.eye(4)
    for name in CHAIN:
        j = _JOINTS[name]
        T = T @ j["T"] @ _rot_about(j["axis"], float(joint_rad[name]))
    return T


def fk_chain(joint_rad) -> dict[str, np.ndarray]:
    """Every link frame along the way, for debugging a mismatch."""
    if not isinstance(joint_rad, dict):
        joint_rad = dict(zip(CHAIN, np.asarray(joint_rad, dtype=float), strict=True))
    T, out = np.eye(4), {}
    for name in CHAIN:
        j = _JOINTS[name]
        T = T @ j["T"] @ _rot_about(j["axis"], float(joint_rad[name]))
        out[name] = T.copy()
    return out


if __name__ == "__main__":
    for q in (np.zeros(5), np.array([0.3, -0.6, -0.1, 1.5, -1.6])):
        T = fk_gripper(q)
        print(f"q={np.round(q,3)}  gripper xyz(mm) {np.round(T[:3,3]*1000,2)}")
