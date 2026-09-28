"""Track-mode goals: the servo command that makes the arm follow the recorded state.

The MuJoCo track's definition (real2sim.mujoco.replay.track_goals), computed here on
the ISAAC model's inertia so the feed-forward matches the arm it drives:

    q_ref(t)      a natural cubic spline through observation.state (URDF rad, 5 arm joints)
    tau_ff        inverse dynamics of the arm along q_ref (MuJoCo mj_inverse on
                  paths.MJCF, which carries the same link inertias as the workshop USD --
                  checked link by link in README -- with the gripper link's inertial
                  replaced by the Isaac gripper + mount + webcam of scene.post_spawn, the
                  servo's armature and its outside-the-clamp damping on the joints),
                  plus the frictionloss as a smoothed Coulomb term fl tanh(qd / 0.02)
    u             q_ref + (tau_ff + kd qd_ref) / kp, one per physics step

The servos keep their torque clamp and their compliance, so a contact the recording
knew nothing about still pushes the arm back. The Jaw is not tracked: it stays on the
recorded action through the servo and its torque limit (the jaw state is blocked by
the cap and carries no squeeze).
"""

from __future__ import annotations

import numpy as np

from .. import paths
from ..units import URDF_JOINTS


def track_goals(model, state_rad: np.ndarray, fps: float, substeps: int, gripper_inertial: dict | None) -> np.ndarray:
    """(T * substeps, 5) arm goals; gripper_inertial = {mass, com, diag, quat} in the
    gripper link frame (scene.post_spawn's payload), or None for the bare URDF link."""
    import mujoco
    from scipy.interpolate import CubicSpline

    m = mujoco.MjModel.from_xml_path(str(paths.MJCF))
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in URDF_JOINTS]
    dadr = [m.jnt_dofadr[m.joint(n).id] for n in URDF_JOINTS]
    arm = URDF_JOINTS[:5]
    for n, a in zip(arm, dadr[:5]):
        j = model.joints[n]
        m.dof_armature[a] = j.armature
        m.dof_damping[a] = j.damping
        m.dof_frictionloss[a] = 0.0
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    if gripper_inertial:
        g = m.body("gripper").id
        m.body_mass[g] = gripper_inertial["mass"]
        m.body_ipos[g] = gripper_inertial["com"]
        m.body_inertia[g] = gripper_inertial["diag"]
        m.body_iquat[g] = gripper_inertial["quat"]
    kp = np.array([model.joints[n].kp for n in arm])
    kd = np.array([model.joints[n].kd for n in arm])
    fl = np.array([model.joints[n].frictionloss for n in arm])
    T = len(state_rad)
    t = np.arange(T) / fps
    sp = CubicSpline(t, state_rad[:, :5], bc_type="natural")
    ts = np.minimum(np.arange(T * substeps) / (fps * substeps), t[-1])
    q, qd, qdd = sp(ts), sp(ts, 1), sp(ts, 2)
    u = np.empty((len(ts), 5))
    for i in range(len(ts)):
        d.qpos[:] = 0.0
        d.qpos[qadr[:5]] = q[i]
        d.qpos[qadr[5]] = state_rad[0, 5]
        d.qvel[:] = 0.0
        d.qvel[dadr[:5]] = qd[i]
        d.qacc[:] = 0.0
        d.qacc[dadr[:5]] = qdd[i]
        mujoco.mj_inverse(m, d)
        tau = d.qfrc_inverse[dadr[:5]] + fl * np.tanh(qd[i] / 0.02)
        u[i] = q[i] + (tau + kd * qd[i]) / kp
    return u
