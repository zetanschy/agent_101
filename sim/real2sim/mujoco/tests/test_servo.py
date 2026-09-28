"""Servo model: config round trip, the goal stream, and the kv-clamp finding."""
import math

import numpy as np
import pytest

from real2sim.mujoco.servo import BAM_MAX_VELOCITY, GoalStream, JointServo, ServoModel
from real2sim.units import URDF_JOINTS


def test_config_round_trip(servo):
    back = ServoModel.from_config(servo.to_config())
    assert math.isclose(back.dead_time, servo.dead_time, rel_tol=1e-5)
    for n in URDF_JOINTS:
        a, b = servo.joint(n), back.joint(n)
        for k in ("kp", "kd", "damping", "armature", "frictionloss", "effort"):
            assert math.isclose(getattr(a, k), getattr(b, k), rel_tol=1e-5), (n, k)
    assert (back.joint("Jaw").flex_stiffness is None) == (servo.joint("Jaw").flex_stiffness is None)


def test_goal_stream_delay_hold_clamp_slew():
    g = np.zeros((4, 6))
    g[1:] = 1.0  # a unit step at frame 1
    lim = np.array([[-0.5, 0.5]] * 6)
    s = GoalStream(g, 30.0, 0.010, None, lim)
    assert s.raw(1 / 30 + 0.009)[0] == 0.0  # dead time not elapsed
    assert s.raw(1 / 30 + 0.011)[0] == 0.5  # step, clamped to the firmware limit
    s = GoalStream(g, 30.0, 0.0, BAM_MAX_VELOCITY)
    t = np.arange(0, 12 / 30, 1 / 900)
    u = np.array([s.at(x)[0] for x in t])
    assert np.all(np.diff(u) <= BAM_MAX_VELOCITY / 900 + 1e-12)  # slewed, never faster
    k = int(np.argmax(t >= 1 / 30 + 0.05))
    assert u[k] == pytest.approx(BAM_MAX_VELOCITY * (t[k] - 1 / 30), abs=2 * BAM_MAX_VELOCITY / 900)  # ramping
    assert u[-1] == pytest.approx(1.0)  # arrived (the step is 1 rad, the ramp 0.19 s)


def test_kv_outside_the_clamp_caps_the_speed(scene):
    """The understand phase's kv discrepancy: passive damping b beside a force clamp F
    caps the speed at ~F/b (plus gravity); as an actuator bias it does not."""
    from real2sim.mujoco import servo_fit as F

    data = F.episode_data(scene, 0)
    data = {k: (v[:260] if hasattr(v, "__len__") and len(v) > 10 else v) for k, v in data.items()}
    sim = F.ArmSim(scene)
    peaks = {}
    for where in ("joint", "actuator"):
        js = {n: JointServo(17.8, 2.0 if where == "actuator" else 0.0, 2.0 if where == "joint" else 0.0, 0.1, 0.052, 1.5)
              for n in URDF_JOINTS}
        q = sim.run(ServoModel(js, 0.0, None), data)
        peaks[where] = np.abs(np.gradient(q[:, :5], axis=0) * 30).max()
    assert peaks["joint"] < 1.0 < peaks["actuator"]  # rad/s: 0.75 cap (+ gravity) vs uncapped


def test_identified_model_is_physical(servo):
    """Every identified number inside its physical bracket (servo_fit.BOUNDS, STS3215 spec)."""
    for n in URDF_JOINTS:
        j = servo.joint(n)
        assert 2 < j.kp < 40 and j.damping > 0 and 1e-4 <= j.armature <= 0.2 and j.frictionloss >= 0
    stall = servo.joint("Rotation").effort
    assert 1.8 <= stall <= 3.6  # N.m: STS3215 1.91 (7.4 V spec) .. 3.52 (BAM full duty)
    assert servo.joint("Jaw").effort <= 0.5 * 3.6
