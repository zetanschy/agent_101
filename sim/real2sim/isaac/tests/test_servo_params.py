"""The PhysX servo emulation's arithmetic, the goal stream vs MUJOCO's, and param overrides."""
import numpy as np
import pytest

from real2sim.isaac import params as P


def test_goal_stream_matches_mujoco(scene):
    torch = pytest.importorskip("torch")
    from real2sim.mujoco.servo import GoalStream
    from real2sim.isaac.servo import GoalStreamT, firmware_limits

    rng = np.random.default_rng(1)
    goals = np.cumsum(rng.normal(0, 0.05, (60, 6)), 0)
    lim = firmware_limits(scene.units())
    q0 = goals[0] + 0.3
    ref = GoalStream(goals, 30, 0.0231, 5.2883, lim, q0)
    mine = GoalStreamT(goals, 30, 0.0231, 5.2883, lim, torch.tensor(np.tile(q0, (3, 1))), "cpu")
    for k in range(59 * 16):
        t = k / 480.0
        a, b = ref.at(t), mine.at(t).numpy()
        assert np.allclose(b, a[None], atol=1e-9), (k, a, b[0])


def test_servo_step_saturates_and_frictions():
    torch = pytest.importorskip("torch")
    from real2sim.isaac.servo import PhysxServo
    from real2sim.mujoco.servo import JointServo, ServoModel
    from real2sim.units import URDF_JOINTS

    js = {n: JointServo(9.0, 0.03, 0.85, 0.0145, 0.1, 2.2) for n in URDF_JOINTS}
    s = PhysxServo(ServoModel(js, 0.023), list(range(6)), 1 / 480, 1, "cpu", flex=False)
    q = torch.zeros(1, 6)
    s.reset(q)
    u = torch.full((1, 6), 1.0)  # 1 rad error: kp err = 9 N.m >> 2.2 N.m
    tgt, vel, ff = s.step(q, torch.zeros(1, 6), u)
    # the drive's spring part equals the clamp exactly
    assert torch.allclose(9.0 * (tgt - q), torch.full((1, 6), 2.2), atol=1e-5)
    # stuck: a small push is held by the bristle up to frictionloss, never beyond
    s.reset(q)
    for _ in range(50):
        f = s._friction(q + 1e-4, torch.zeros(1, 6))
    assert torch.all(f.abs() <= 0.1 + 1e-6)
    f = s._friction(q + 0.5, torch.zeros(1, 6))  # a big slip: Coulomb
    assert torch.allclose(f.abs(), torch.full((1, 6), 0.1))


def test_param_overrides_and_scene_overrides(scene):
    p = P.override(P.defaults(), ["hz=960", "materials.cap.static_friction=0.6"])
    assert p["physics"]["hz"] == 960 and p["materials"]["cap"]["static_friction"] == 0.6
    with pytest.raises(KeyError):
        P.override(P.defaults(), ["no_such_key=1"])
    sc2, done = P.scene_with_overrides(scene, ["table.z=0.011"])
    assert sc2.table_z() == 0.011 and sc2.hash != scene.hash and done["table.z"]["was"] == scene.table_z()
