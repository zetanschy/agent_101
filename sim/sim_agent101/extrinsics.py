"""Measured camera poses, converted into what the Isaac scene wants.

Reads config/extrinsics.json, written by ./robot calib-solve. Free of isaaclab so
the calibration tools can share it.

Two conversions matter and both are easy to get silently wrong:

FRAME. calib-solve reports poses in the ROBOT BASE frame. The overhead camera is a
prim in the env, not under the robot, so it needs env->base composed in front. The
wrist camera IS a child of Robot/gripper, so its pose is used as-is.

CONVENTION. OpenCV cameras look along +Z with +Y down. USD/Isaac "opengl" cameras
look along -Z with +Y up. Skip the 180-degree flip about X and the camera points
backwards -- which renders a plausible-looking image of the wrong thing.
"""

from __future__ import annotations

import json
import math
import pathlib

CONFIG = pathlib.Path(__file__).parent / "config" / "extrinsics.json"

# Where the robot sits in the env, from assets/so101.py: pos (-0.05, 0, 0), yaw 90.
ROBOT_POS = (-0.05, 0.0, 0.0)
ROBOT_YAW_DEG = 90.0


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _rotz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _mat_to_quat(m):
    """(w, x, y, z) from the rotation part of a 4x4."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(m[i][i] - m[j][j] - m[k][k] + 1.0) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k][j] - m[j][k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (m[j][i] + m[i][j]) / s
    q[k + 1] = (m[k][i] + m[i][k]) / s
    return tuple(q)


def _cv_to_gl(m):
    """OpenCV camera axes -> USD/opengl camera axes: 180 degrees about X."""
    flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    return _matmul(m, flip)


def available() -> bool:
    return CONFIG.exists()


def load() -> dict:
    if not CONFIG.exists():
        raise FileNotFoundError(f"{CONFIG} -- run ./robot calib-solve")
    return json.loads(CONFIG.read_text())


def _pose(m):
    return tuple(m[i][3] for i in range(3)), _mat_to_quat(m)


def overhead_in_env():
    """(pos, quat) for the overhead camera prim, in the env frame."""
    T_base_cam = _cv_to_gl(load()["top_cam_in_base"])
    T_env_base = _rotz(ROBOT_YAW_DEG)
    for i in range(3):
        T_env_base[i][3] = ROBOT_POS[i]
    return _pose(_matmul(T_env_base, T_base_cam))


def wrist_in_gripper():
    """(pos, quat) for the wrist camera prim, relative to the gripper body."""
    return _pose(_cv_to_gl(load()["wrist_cam_in_gripper"]))


if __name__ == "__main__":
    d = load()
    print(f"source: {d.get('method')}  poses {d.get('poses_used')}  "
          f"reproj {d.get('reproj_rms_px', float('nan')):.2f} px")
    p, q = overhead_in_env()
    print(f"overhead in env:     pos {tuple(round(v,5) for v in p)}  quat {tuple(round(v,5) for v in q)}")
    p, q = wrist_in_gripper()
    print(f"wrist in gripper:    pos {tuple(round(v,5) for v in p)}  quat {tuple(round(v,5) for v in q)}")
