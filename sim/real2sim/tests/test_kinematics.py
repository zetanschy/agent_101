"""FK against the repo's URDF FK, the grasp site, and the finger-gap geometry."""
import importlib.util

import numpy as np
import pytest

from real2sim import kinematics, paths


@pytest.fixture(scope="module")
def kin():
    return kinematics.default()


def _sim_agent_kinematics():
    spec = importlib.util.spec_from_file_location("sim_agent101_kinematics", paths.SIM_AGENT_KINEMATICS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fk_matches_sim_agent101(kin):
    ref = _sim_agent_kinematics()
    rng = np.random.default_rng(0)
    worst_p = worst_r = 0.0
    for q in rng.uniform(-2.5, 2.5, (300, 6)):
        T_ref, T = ref.fk_gripper(q[:5]), kin.link_T(q, "gripper")
        worst_p = max(worst_p, np.abs(T_ref[:3, 3] - T[:3, 3]).max())
        worst_r = max(worst_r, np.abs(T_ref[:3, :3] - T[:3, :3]).max())
    assert worst_p < 1e-5 and worst_r < 1e-5, (worst_p, worst_r)  # 0.01 mm


def test_link_poses_batch_and_names(kin):
    Q = np.random.default_rng(1).uniform(-1, 1, (5, 6))
    pos, quat = kin.link_poses_batch(Q)
    single = kin.link_poses(Q[3])
    assert list(single) == list(kinematics.LINKS)
    for i, n in enumerate(kinematics.LINKS):
        assert np.allclose(pos[3, i], single[n][0]) and np.allclose(quat[3, i], single[n][1])
    assert np.allclose(single["base"][0], 0) and np.allclose(single["base"][1], [1, 0, 0, 0])


def test_fingertips_near_grasp_site(kin):
    q = np.array(kinematics.HOME)
    T = np.linalg.inv(kin.link_T(q, "gripper"))
    mid = T[:3, :3] @ kin.fingertips(q)["mid"] + T[:3, 3]
    assert np.linalg.norm(mid - np.array(kinematics.GRASP_SITE_POS)) < 0.003


def test_jaw_gap_matches_joint_units_report(kin):
    for deg, mm in ((5.5, 22.8), (6.6, 24.2), (7.9, 25.8)):
        assert abs(kin.jaw_gap(np.radians(deg)) * 1000 - mm) < 0.3
    assert abs(np.degrees(kin.jaw_touch_rad()) - (-11.5)) < 0.3
    assert abs(np.degrees(kin.jaw_rad_for_gap(0.0242)) - 6.6) < 0.3


def test_world_vertices(kin):
    q = np.zeros(6)
    vis = kin.world_vertices(q)
    assert len(vis) >= 13 and all(v.shape[1] == 3 for _, v in vis)
