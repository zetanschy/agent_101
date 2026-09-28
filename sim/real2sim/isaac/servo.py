"""The STS3215 servo on a PhysX articulation: the MuJoCo track's model, re-expressed.

ONE MODEL, TWO ENGINES. The parameters and their meaning are the MuJoCo track's
(real2sim.mujoco.servo.ServoModel, read from the scene's `actuator` group, i.e.
MUJOCO's servo.json once it exists, else the provisional flat keys):

    tau = clip(kp (u - q) - kd qd, +-effort) - damping qd - frictionloss sign(qd)
    u   = the recorded goal, clamped to the firmware limits, zero-order held per frame,
          delayed by dead_time (s), slewed at <= max_velocity (the firmware target)
    Jaw = the HORN (the encoder, what observation.state records) driving the finger
          through a series spring flex_stiffness / flex_damping (grip_hold.py: the
          encoder angle falls linearly with the squeeze on every real hold)

PhysX cannot express this directly: its drive has one stiffness and one damping and
clips the WHOLE drive force at maxForce, it has no dry friction, and it cannot stack a
second hinge in the jaw link. So, per physics step:

  arm joints   the PhysX implicit drive carries stiffness kp and damping kd + damping,
               and its TARGET is set to q + (clip(kp (u - q) - kd qd) + kd qd) / kp,
               which makes the drive torque exactly clip(kp (u - q) - kd qd) - damping qd
               (the clip inside, the motor damping outside, as in MuJoCo), integrated
               implicitly. PhysX maxForce is only a backstop (effort + (kd + damping)
               NO_LOAD_SPEED) that never binds.
  dry friction an elasto-plastic (bristle) friction as a feed-forward effort:
               tau_f = clip(k_s z - c_s qd, +-frictionloss), z = q_anchor - q, and the
               anchor slips with the joint whenever |k_s z| would exceed frictionloss.
               Stuck, it holds a static load up to frictionloss, as MuJoCo's friction
               constraint does (the fitted Pitch/Elbow frictionloss 0.28-0.29 N.m is a
               1.8 deg deadband at kp 9 -- a velocity-only friction would let the arm
               creep through it); sliding, it is Coulomb friction. The bristle
               deflection is at most frictionloss / k_s (STICTION_K below: <= 0.08 deg),
               and k_s, c_s are explicit-stable at 480 Hz for any joint inertia >= its
               armature (k_s dt^2 / J_a = 0.06, c_s dt / J_a = 0.49 at J_a 0.0145).
  Jaw (flex)   the horn is a state integrated HERE (semi-implicit Euler, inertia =
               armature, exact Coulomb friction with stiction), and the PhysX Jaw drive
               IS the series spring: stiffness flex_stiffness, damping flex_damping,
               target = horn angle, target velocity = horn velocity; PhysX Jaw armature 0
               (the rotor inertia belongs to the horn). The finger is the PhysX Jaw
               joint; the horn angle is what the encoder reads.
  goal stream  GoalStream's semantics (index = floor((t - dead_time) fps), slew
               clip(goal, target -+ vmax dt)), vectorised over envs in torch. MEASURED
               equal to real2sim.mujoco.servo.GoalStream (isaac/tests/test_servo.py).

MEASURED (probe1, ep0 arm-only action replay at 480 Hz, provisional kp 8 / damping 0.8
/ armature 0.01 / frictionloss 0.052 / 1 frame, no slew): this emulation tracks
MuJoCo's own replay of the same model to 0.06/0.23/0.40/0.05/0.04 deg RMSE per arm
joint (the elbow/pitch residual is the real firmware limit, which Isaac enforces and
the MuJoCo reference ignored); a literal port (PhysX drive with maxForce = effort,
no friction) was 0.31/0.30/0.46/0.30/0.30. Units are SI throughout: a degree-based
stiffness (57x) could not land within 0.1 deg.

Pure torch + numpy: no Isaac import.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..units import GRIPPER, URDF_JOINTS

NO_LOAD_SPEED = 5.3  # rad/s, spec: STS3215 no-load (BAM m1 5.07-5.3); sizes the PhysX backstop only
STICTION_K = 200.0  # N.m/rad, assumed bristle stiffness (module doc: <= 0.08 deg deflection)
STICTION_ZETA = 1.0  # critically damped bristle on the armature


def load_model(scene):
    """The scene's servo model through the MuJoCo track's parser (one reading of the
    config for both engines)."""
    from ..mujoco.servo import ServoModel

    return ServoModel.from_scene(scene)


class GoalStreamT:
    """GoalStream (real2sim.mujoco.servo) for N envs at once: goals (T, 6) URDF rad,
    the same for every env; at(t) returns (N, 6)."""

    def __init__(self, goals, fps: float, dead_time: float, max_velocity, limits, q0, device):
        g = np.clip(np.asarray(goals, dtype=float), limits[:, 0], limits[:, 1])
        self.goals = torch.tensor(g, dtype=torch.float64, device=device)
        self.fps, self.dead_time, self.vmax = float(fps), float(dead_time), max_velocity
        self.target = torch.as_tensor(q0, dtype=torch.float64, device=device).clone()
        self._t = None

    def index(self, t: float) -> int:
        return int(min(max(math.floor((t - self.dead_time) * self.fps + 1e-9), 0), len(self.goals) - 1))

    def at(self, t: float) -> torch.Tensor:
        goal = self.goals[self.index(t)].expand_as(self.target)
        if self.vmax is None:
            return goal.clone()
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        step = self.vmax * dt
        self.target = torch.minimum(torch.maximum(goal, self.target - step), self.target + step)
        return self.target.clone()


def physx_gains(model, flex: bool = True) -> dict:
    """{stiffness, damping, armature, max_force: {URDF joint: value}}: what PhysX's drive
    must carry for this model (module doc). With the flex on, the Jaw drive IS the
    series spring and carries no rotor inertia."""
    out = {k: {} for k in ("stiffness", "damping", "armature", "max_force")}
    for i, n in enumerate(URDF_JOINTS):
        j = model.joints[n]
        d = float(j.kd or 0.0) + float(j.damping)
        if i == GRIPPER and flex and j.flex_stiffness:
            out["stiffness"][n], out["damping"][n], out["armature"][n] = float(j.flex_stiffness), float(j.flex_damping or 0.0), 0.0
            # the spring carries whatever the horn pushes; 10x the clamp is a backstop only
            out["max_force"][n] = 10.0 * float(j.effort) + float(j.flex_damping or 0.0) * NO_LOAD_SPEED
        else:
            out["stiffness"][n], out["damping"][n], out["armature"][n] = float(j.kp), d, float(j.armature)
            out["max_force"][n] = float(j.effort) + d * NO_LOAD_SPEED
    return out


class PhysxServo:
    """Per-physics-step PhysX drive targets and feed-forward efforts, N envs x 6 joints.

    `perm[i]` is the articulation index of URDF joint i. Tensors passed to and returned
    by step() are in ARTICULATION order (what robot.set_joint_* expects)."""

    def __init__(self, model, perm, dt: float, num_envs: int, device, flex: bool = True):
        self.model, self.dt, self.N, self.dev = model, float(dt), num_envs, device
        self.perm = list(perm)
        n = len(perm)

        def vec(attr):
            v = torch.zeros(n, device=device, dtype=torch.float32)
            for i, name in enumerate(URDF_JOINTS):
                v[perm[i]] = float(getattr(model.joints[name], attr) or 0.0)
            return v

        self.kp, self.kd, self.damp = vec("kp"), vec("kd"), vec("damping")
        self.arm, self.fl, self.eff = vec("armature"), vec("frictionloss"), vec("effort")
        jaw = model.joints[URDF_JOINTS[GRIPPER]]
        self.j = perm[GRIPPER]
        self.flex = bool(flex and jaw.flex_stiffness)
        if self.flex and not jaw.armature:
            raise ValueError("the flexed Jaw needs a horn inertia: armature must be > 0")
        self.kf = float(jaw.flex_stiffness or 0.0)
        self.cf = float(jaw.flex_damping or 0.0)
        self.horn = torch.zeros(num_envs, device=device)
        self.horn_vel = torch.zeros(num_envs, device=device)
        self.horn_tau = torch.zeros(num_envs, device=device)
        self.horn_act = torch.zeros(num_envs, device=device)
        self.anchor = torch.zeros(num_envs, n, device=device)
        self.c_s = 2.0 * STICTION_ZETA * torch.sqrt(STICTION_K * self.arm.clamp(min=1e-6))

    def physx_tensors(self) -> dict:
        """physx_gains(...) as (6,) tensors in articulation order."""
        g = physx_gains(self.model, self.flex)
        out = {}
        for k, d in g.items():
            v = torch.zeros(len(self.perm), device=self.dev)
            for i, n in enumerate(URDF_JOINTS):
                v[self.perm[i]] = d[n]
            out[k] = v
        return out

    def reset(self, q) -> None:
        """Start at rest at joint angles q (N, n) articulation order: the horn at the
        jaw angle (the flex relaxed), the friction anchors unloaded."""
        q = torch.as_tensor(q, dtype=torch.float32, device=self.dev)
        self.anchor[:] = q
        self.horn[:] = q[:, self.j]
        self.horn_vel[:] = 0.0
        self.horn_tau[:] = 0.0
        self.horn_act[:] = 0.0

    def _friction(self, q, qd):
        z = self.anchor - q
        slip = STICTION_K * z.abs() > self.fl
        z = torch.where(slip, torch.sign(z) * self.fl / STICTION_K, z)
        self.anchor = q + z
        return torch.clamp(STICTION_K * z - self.c_s * qd, -self.fl, self.fl)

    def step(self, q: torch.Tensor, qd: torch.Tensor, u: torch.Tensor):
        """(position target, velocity target, feed-forward effort) for this physics step."""
        kp, kd = self.kp, self.kd
        tau_c = torch.clamp(kp * (u - q) - kd * qd, -self.eff, self.eff)
        tgt = q + (tau_c + kd * qd) / kp
        vel = torch.zeros_like(q)
        ff = self._friction(q, qd)
        if self.flex:
            j = self.j
            h, w = self.horn, self.horn_vel
            act = torch.clamp(kp[j] * (u[:, j] - h) - kd[j] * w, -self.eff[j], self.eff[j])
            tau_s = act - self.damp[j] * w
            spring = self.kf * (h - q[:, j]) + self.cf * (w - qd[:, j])
            w_new = w + self.dt * (tau_s - spring) / self.arm[j]
            cap = self.dt * self.fl[j] / self.arm[j]  # Coulomb friction impulse, with stiction
            w_new = torch.sign(w_new) * torch.clamp(w_new.abs() - cap, min=0.0)
            self.horn = h + self.dt * w_new
            self.horn_vel = w_new
            self.horn_tau = tau_s
            self.horn_act = act
            tgt[:, j] = self.horn
            vel[:, j] = self.horn_vel
            ff[:, j] = 0.0
        return tgt, vel, ff

    def torque(self, q, qd, tgt, vel, ff) -> torch.Tensor:
        """Joint torques after the step (articulation order); the Jaw's is the servo
        torque at the horn when the flex is on (what the torque limit applies to)."""
        tau = self.kp * (tgt - q) - (self.kd + self.damp) * qd + ff
        if self.flex:
            tau[:, self.j] = self.horn_tau
        return tau

    def actuator_force(self, q, qd, u) -> torch.Tensor:
        """The clamped controller output clip(kp (u - q) - kd qd, +-effort) per joint
        (articulation order): MuJoCo's actuator_force (the motor damping and the friction
        act outside it). The Jaw's is the horn's when the flex is on."""
        a = torch.clamp(self.kp * (u - q) - self.kd * qd, -self.eff, self.eff)
        if self.flex:
            a[:, self.j] = self.horn_act
        return a

    def encoder(self, q) -> torch.Tensor:
        """What the real encoders would read: the joint angles, with the horn for the Jaw."""
        e = q.clone()
        if self.flex:
            e[:, self.j] = self.horn
        return e


def firmware_limits(units) -> np.ndarray:
    """(6, 2) URDF rad: the goal clamp, the calibration's Min/Max_Position_Limit (as
    real2sim.mujoco.servo.firmware_limits_rad)."""
    from ..units import limits_rad

    return limits_rad(units)
