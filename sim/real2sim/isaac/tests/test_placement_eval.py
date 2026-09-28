"""Antipodal grasp placement on a recorded grasp; offline penetration and eval on a real log."""
import numpy as np

from real2sim import episodes, kinematics, poselog
from real2sim.isaac import penetration as PEN
from real2sim.isaac.placement import grasp_centre, pad_gap


def test_grasp_centre_is_antipodal(scene, extracted):
    kin = kinematics.default()
    q = scene.units().to_urdf(episodes.load()[0].state[231])
    zmin = min(v[:, 2].min() for g, v in kin.world_vertices(q, visual=False) if g["link"] in ("gripper", "jaw"))
    r = grasp_centre(kin, q, 0.029, zmin - 0.001, zmin + 0.011)
    assert r["residual_m"] < 1e-4
    qq = q.copy()
    qq[5] = r["jaw_rad"]
    gap, a, b = pad_gap(kin, qq, zmin - 0.001, zmin + 0.011)
    assert abs(gap - 0.029) < 1e-4 and np.allclose(r["xy"], (a + b) / 2)
    # a smaller cap closes the jaw further
    assert grasp_centre(kin, q, 0.024, zmin - 0.001, zmin + 0.011)["jaw_rad"] < r["jaw_rad"]


def test_penetration_zero_far_from_the_real_arm(scene, extracted):
    log = poselog.from_dataset(episode=0)
    T = len(log["q"])
    log["objects"] = np.array(["cap_A"])
    log["obj_pos"] = np.tile([0.3, 0.3, scene.table_z()], (T, 1, 1)).astype(float)  # away from the arm
    log["obj_quat"] = np.tile([1.0, 0, 0, 0], (T, 1, 1))
    out = PEN.run(log, scene, kinematics.default(), stride=25)
    assert out["finger_cap"].max() == 0.0 and out["cap_table"].max() < 1e-6
