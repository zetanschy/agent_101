"""Rigid transforms in numpy: 4x4 matrices and (w, x, y, z) quaternions.

The one convention every real2sim module and every engine adapter shares. MuJoCo,
Isaac Lab and USD all store quaternions scalar-first; scipy and trimesh do not
(scipy is x, y, z, w), so nothing here goes through scipy.
"""

from __future__ import annotations

import numpy as np


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv_T(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def apply_T(T, P) -> np.ndarray:
    """Transform points (..., 3) by a 4x4."""
    T = np.asarray(T, dtype=float)
    return np.asarray(P, dtype=float) @ T[:3, :3].T + T[:3, 3]


def quat_to_mat(q) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3). Normalises first."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def mat_to_quat(R) -> np.ndarray:
    """(3, 3) rotation -> wxyz with w >= 0 (Shepperd's method, stable near 180 deg)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a * b (apply b, then a), wxyz."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=float), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def axis_angle_quat(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * a])


def pose_to_T(pos, quat) -> np.ndarray:
    return make_T(quat_to_mat(quat), pos)


def T_to_pose(T) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), mat_to_quat(T[:3, :3])


def is_rotation(R, tol: float = 1e-6) -> bool:
    R = np.asarray(R, dtype=float)
    return R.shape == (3, 3) and np.allclose(R.T @ R, np.eye(3), atol=tol) and np.linalg.det(R) > 0
