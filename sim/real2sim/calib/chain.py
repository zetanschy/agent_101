"""Vectorised forward kinematics of the SO-ARM101 with joint zero offsets, in numpy.

The calibration evaluates the arm at thousands of (frame, parameter) combinations
per Jacobian, and the joint zero offsets are among the parameters, so FK must be
cheap and batched. MuJoCo's mj_kinematics is exact but one pose per Python call
(~20 us each); this is the same chain written out once from the compiled MJCF:

    T_world_link(q) = prod_i  T(body_pos_i, body_quat_i) @ Rz(q_i)

MEASURED on the MJCF (paths.MJCF, mujoco 3.6): every joint of the chain sits at its
body's origin (jnt_pos = 0) and turns about the body's local +z (jnt_axis = 0 0 1),
so that product is the whole model. tests: agrees with kinematics.Kinematics.link_T
to 1e-12 m on random poses (calib/tests/test_chain.py).

q here is URDF radians AFTER the lerobot -> URDF map (units.Units.to_urdf): the
calibration applies its own offsets and gripper map first (params.Params.q_urdf).
"""

from __future__ import annotations

import functools

import numpy as np

from .. import paths
from ..kinematics import LINKS

CHAIN = LINKS[1:]  # bodies with a joint, base -> jaw order; `base` is the world identity


@functools.lru_cache(maxsize=None)
def _chain(mjcf: str):
    import mujoco

    m = mujoco.MjModel.from_xml_path(mjcf)
    out = []
    for name in CHAIN:
        b = m.body(name).id
        j = m.body_jntadr[b]
        if not (np.allclose(m.jnt_pos[j], 0) and np.allclose(m.jnt_axis[j], [0, 0, 1])):
            raise ValueError(f"{name}: joint not at the body origin about +z; chain.py assumes it")
        from ..transforms import quat_to_mat

        out.append((name, m.body(m.body_parentid[b]).name, np.array(m.body_pos[b]), quat_to_mat(m.body_quat[b])))
    return tuple(out)


def _rz(q):
    c, s = np.cos(q), np.sin(q)
    R = np.zeros(q.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


def link_T(Q, mjcf=None) -> dict:
    """{link: (T, 4, 4)} world poses of every link for joint vectors Q (T, 6), URDF rad."""
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    n = len(Q)
    eye = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    out = {"base": eye}
    for i, (name, parent, pos, R0) in enumerate(_chain(str(mjcf or paths.MJCF))):
        Tp = out[parent]
        R = R0 @ _rz(Q[:, i])  # (n, 3, 3)
        T = np.empty((n, 4, 4))
        T[:, :3, :3] = Tp[:, :3, :3] @ R
        T[:, :3, 3] = Tp[:, :3, :3] @ pos + Tp[:, :3, 3]
        T[:, 3] = (0, 0, 0, 1)
        out[name] = T
    return out


def transform(T, P) -> np.ndarray:
    """Points P (..., N, 3) in a link frame -> world, for per-frame poses T (..., 4, 4)."""
    return P @ np.swapaxes(T[..., :3, :3], -1, -2) + T[..., None, :3, 3]
