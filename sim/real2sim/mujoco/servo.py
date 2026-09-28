"""The STS3215 servo model, engine-agnostic, and the goal stream the firmware acts on.

WHAT THE REAL SERVO IS (joint_units report sections 2 and 5, READ + MEASURED). A
Feetech STS3215 (1:345) in position mode with P_Coefficient=16, I=0, D=32 (lerobot
0.6.1 so_follower.py:159-168). Its firmware turns the position error into a PWM duty
cycle, clamps the duty to Max_Torque_Limit (100 % on the arm, 50 % on the gripper),
and a DC motor turns the duty into torque. So, at the output shaft:

    tau = clip(kp (u - q) - kd qdot, -effort, +effort)     the controller, clamped
          - damping qdot - frictionloss sign(qdot)          the motor, NOT clamped

The clamp is on the controller's output (the duty), so the motor's back-EMF and
viscous friction act OUTSIDE it: at full duty the torque still falls with speed.
That split is the answer to the understand phase's "kv discrepancy":

    mjlab's constants (kp 17.8, kv 2.0, armature 0.1, 1.5 N.m) were measured as
    7-14 deg RMSE with an 8-frame lag when kv was JOINT damping, and 1-3 deg when
    kv was the ACTUATOR's velocity bias. MEASURED here (servo_fit.kv_crosscheck, ep0,
    no dead time): the integrator is not the cause (Euler and implicitfast agree to
    0.01 deg at dt 1/480 and 1/900 s); the clamp is. Passive damping b beside a force
    clamp F caps the joint speed at F / b = 1.5 / 2.0 = 0.75 rad/s: the sim peaks at
    55 deg/s against the real 146, lags 9.4 frames (real 4.7) and misses by 7.93 deg
    RMSE; with a 3 N.m clamp, 97 deg/s and 1.63 deg. As an actuator bias the same kv
    is inside the clamp and caps nothing: 156 deg/s, 1.16 deg, either clamp. The
    physical servo has both kinds, which is why the fit identifies kd (inside) and
    damping (outside) separately.

    u           the goal: the recorded ACTION (what lerobot sent), clamped to the
                firmware position limits, held for one frame, and delayed by
                dead_time (bus + firmware latency; fitted). The firmware then
                slews its internal target toward the goal at most at max_velocity
                (Rhoban BAM's STS3215 m1 model: 5.29 rad/s), which rounds off the
                30 Hz staircase.
    armature    reflected rotor inertia (kg m^2), added to the joint.
    effort      the clamp: stall torque x Max_Torque_Limit.

THE GRIPPER adds a series compliance. The encoder reads the output shaft, but the
cap is held by a PLA finger that bends, a cap that squashes, a horn that gives. On
the real holds the encoder angle falls LINEARLY with the commanded squeeze
(grip_hold.py, 6 plateaus: slope -0.45 % per % of error, residual +-0.3 %), which a
rigid jaw cannot do and a spring in series with the P servo does exactly:
theta_enc = theta_contact - (kp / k_flex) (theta_enc - u). So the Jaw joint here is
the HORN (the encoder, the actuator's joint), and a coaxial `Jaw_flex` hinge of
stiffness k_flex carries the finger. Where MuJoCo stacks both hinges in the jaw body
the finger turns by Jaw + Jaw_flex, the rotor inertia (armature) sits on the horn
side and the finger's own inertia on the other: a series-elastic actuator with no
extra body. PhysX cannot stack coaxial joints in one link: ISAAC needs a horn link.

UNITS. SI: N.m/rad, N.m.s/rad, kg m^2, N.m, s, rad/s. Joint names are
units.URDF_JOINTS. The parameters live in the scene's `actuator` group, written by
servo_fit to config/<ds>.servo.json; every joint and the flexure carry a `source`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..units import GRIPPER, URDF_JOINTS

MODEL_ID = "sts3215/1"
FLEX_JOINT = "Jaw_flex"

# Rhoban BAM, feetech_sts3215_7_4V/m1.json (READ, joint_units report bam/ clone):
# the firmware slews its internal target at most at this speed (3400 ticks/s nominal,
# fitted 5.288 rad/s). The recorded goals move at up to 3.8 rad/s, so it binds only
# on the 30 Hz steps of the staircase.
BAM_MAX_VELOCITY = 5.2883
# The arm's torque clamp: full-duty stall of the same BAM model, max_pwm 0.97 x 7.4 V x
# kt 1.1776 N.m/A / R 2.4787 ohm = 3.41 N.m (READ, m1.json). MEASURED consistency: the
# recorded ep2 motion needs up to 3.16 N.m on Pitch and 2.81 on Elbow (inverse dynamics
# of the identified model along observation.state), above both vendor ratings (1.91 N.m
# at 7.4 V, 2.94 at 12 V) and below this; Robonine measured ~3.43 N.m at 12 V.
BAM_STALL = 0.97 * 7.4 * 1.1776 / 2.4787


@dataclass(frozen=True)
class JointServo:
    """One joint's servo parameters (SI). kd is inside the effort clamp, damping outside."""

    kp: float
    kd: float
    damping: float
    armature: float
    frictionloss: float
    effort: float
    flex_stiffness: float | None = None  # Jaw only: series finger compliance, N.m/rad
    flex_damping: float | None = None  # N.m.s/rad
    source: str = "provisional"

    def replace(self, **kw) -> "JointServo":
        return replace(self, **kw)


@dataclass(frozen=True)
class ServoModel:
    joints: dict  # {URDF joint name: JointServo}
    dead_time: float  # s
    max_velocity: float | None = BAM_MAX_VELOCITY  # rad/s firmware target slew; None = off
    source: str = "provisional"
    extra: dict = field(default_factory=dict)  # stall torque, metrics: round-tripped, not used

    def __post_init__(self):
        missing = [j for j in URDF_JOINTS if j not in self.joints]
        if missing:
            raise ValueError(f"servo model lacks joints {missing}")

    def joint(self, name: str) -> JointServo:
        return self.joints[name]

    def replace(self, **kw) -> "ServoModel":
        return replace(self, **kw)

    def with_joint(self, name: str, **kw) -> "ServoModel":
        js = dict(self.joints)
        js[name] = js[name].replace(**kw)
        return self.replace(joints=js)

    # --- config round trip ---------------------------------------------------------
    @classmethod
    def from_scene(cls, scene) -> "ServoModel":
        return cls.from_config(scene.actuator())

    @classmethod
    def from_config(cls, a: dict) -> "ServoModel":
        """From the scene's `actuator` group: per-joint `joints` if present (servo.json),
        else the provisional flat keys (core config: one kp/damping/... for all)."""
        if "joints" in a:
            js = {}
            for n in URDF_JOINTS:
                g = a["joints"][n]
                flex = g.get("flex", {})
                js[n] = JointServo(float(g["kp"]), float(g.get("kd", 0.0)), float(g["damping"]), float(g["armature"]),
                                   float(g["frictionloss"]), float(g["effort_limit"]),
                                   float(flex["stiffness"]) if flex else None,
                                   float(flex["damping"]) if flex else None, g.get("source", ""))
            mv = a.get("max_velocity")
            return cls(js, float(a["dead_time_s"]), None if mv is None else float(mv), a.get("source", ""),
                       {k: a[k] for k in ("stall_torque", "metrics", "fit") if k in a})
        # provisional core config: flat numbers, frames of dead time
        eff = a.get("effort_limit", {})
        js = {n: JointServo(float(a.get("kp", 8.0)), 0.0, float(a.get("damping", 0.8)), float(a.get("armature", 0.01)),
                            float(a.get("frictionloss", 0.052)),
                            float(eff.get("gripper" if i == GRIPPER else "arm", 1.47 if i == GRIPPER else 2.94)),
                            source=a.get("source", "provisional"))
              for i, n in enumerate(URDF_JOINTS)}
        return cls(js, float(a.get("dead_time_frames", 1)) / 30.0, BAM_MAX_VELOCITY, a.get("source", "provisional"))

    def to_config(self, sources: dict | None = None, **top) -> dict:
        """The `actuator` group for config/<ds>.servo.json (see module doc for units).
        Also writes the provisional flat keys (kp, damping, ...) from the Rotation joint
        and the effort limits, so a reader of the old layout gets consistent numbers."""
        s = sources or {}
        joints = {}
        for n in URDF_JOINTS:
            j = self.joints[n]
            g = {"source": s.get(n, j.source), "kp": _r(j.kp), "kd": _r(j.kd), "damping": _r(j.damping),
                 "armature": _r(j.armature), "frictionloss": _r(j.frictionloss), "effort_limit": _r(j.effort)}
            if j.flex_stiffness is not None:
                g["flex"] = {"source": s.get(FLEX_JOINT, "fitted: grip_hold.py"), "stiffness": _r(j.flex_stiffness),
                             "damping": _r(j.flex_damping or 0.0)}
            joints[n] = g
        r = self.joints["Rotation"]
        out = {
            "source": top.pop("source", self.source),
            "model": f"{MODEL_ID}: tau = clip(kp (u - q) - kd qdot, +-effort_limit) - damping qdot - frictionloss "
                     "sign(qdot); u = recorded action clamped to the firmware limits, delayed by dead_time_s, "
                     "slewed at <= max_velocity; Jaw = horn (encoder) + coaxial series flex hinge to the finger",
            "dead_time_s": _r(self.dead_time), "max_velocity": None if self.max_velocity is None else _r(self.max_velocity),
            "joints": joints,
            # the core config's flat layout, kept consistent (Rotation's values; see joints for all)
            "kp": _r(r.kp), "damping": _r(r.damping), "armature": _r(r.armature), "frictionloss": _r(r.frictionloss),
            "dead_time_frames": _r(self.dead_time * 30.0),
            "effort_limit": {"source": s.get("effort_limit", "fitted: see joints.*.effort_limit"),
                             "arm": _r(r.effort), "gripper": _r(self.joints["Jaw"].effort)},
        }
        out.update(self.extra)
        out.update(top)
        return out


def _r(x: float, n: int = 6) -> float:
    return float(f"{float(x):.{n}g}")


# --- the goal stream ----------------------------------------------------------------

class GoalStream:
    """u(t) for the servos: recorded goals (T, 6) in URDF rad at `fps`, clamped to the
    firmware limits, zero-order held, delayed by dead_time, and (optionally) slewed at
    max_velocity by a stateful firmware target -- call `at` with increasing t only.

    Before the first goal takes effect (t < dead_time) the servo holds goals[0]: the
    real follower was already tracking a goal like it when recording started."""

    def __init__(self, goals, fps: float, dead_time: float, max_velocity: float | None = None,
                 limits=None, q0=None):
        g = np.asarray(goals, dtype=float)
        if limits is not None:
            lim = np.asarray(limits, dtype=float)
            g = np.clip(g, lim[:, 0], lim[:, 1])
        self.goals, self.fps, self.dead_time, self.vmax = g, float(fps), float(dead_time), max_velocity
        self._target = np.array(q0 if q0 is not None else g[0], dtype=float)
        self._t = None

    def index(self, t: float) -> int:
        return int(min(max(math.floor((t - self.dead_time) * self.fps + 1e-9), 0), len(self.goals) - 1))

    def raw(self, t: float) -> np.ndarray:
        return self.goals[self.index(t)]

    def at(self, t: float) -> np.ndarray:
        goal = self.raw(t)
        if self.vmax is None:
            return goal
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        step = self.vmax * dt
        self._target = np.clip(goal, self._target - step, self._target + step)
        return self._target.copy()


def firmware_limits_rad(units) -> np.ndarray:
    """(6, 2) goal clamp: the calibration's Min/Max_Position_Limit through `units`."""
    from ..units import limits_rad

    return limits_rad(units)


def describe(servo: ServoModel) -> str:
    rows = [f"servo {MODEL_ID}: dead time {servo.dead_time * 1000:.1f} ms, max_velocity "
            f"{servo.max_velocity if servo.max_velocity is None else round(servo.max_velocity, 3)} rad/s"]
    for n in URDF_JOINTS:
        j = servo.joints[n]
        d = asdict(j)
        d.pop("source")
        rows.append(f"  {n:12s} " + " ".join(f"{k} {v:.4g}" for k, v in d.items() if v is not None))
    return "\n".join(rows)
