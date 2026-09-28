"""Ensemble draws, the latency-corrected pixel comparison, the cap collision shapes, and the
guard that keeps what-if runs off the nominal logs."""
import argparse

import numpy as np
import pytest


def test_ensemble_draws_the_fitted_cap_uncertainty(scene):
    """CALIB's sigma [D, h] wins over the provisional 23-32 mm diameter_range."""
    from real2sim.mujoco.evaluate import draw_perturbation

    cap = scene["objects"]["cap"]
    if not isinstance(cap.get("sigma"), list):
        pytest.skip("the scene has no fitted cap sigma")
    rng = np.random.default_rng(0)
    P = [draw_perturbation(scene, 0, rng) for _ in range(400)]
    D = np.array([p.cap_diameter for p in P])
    H = np.array([p.cap_height for p in P])
    assert abs(D.std() / cap["sigma"][0] - 1) < 0.15 and abs(D.mean() - cap["diameter"]) < 2e-4
    assert abs(H.std() / cap["sigma"][1] - 1) < 0.15 and abs(H.mean() - cap["height"]) < 5e-4
    assert D.min() >= cap["diameter_range"][0] and D.max() <= cap["diameter_range"][1]


def test_fractional_frame_interpolation():
    from real2sim.mujoco.evaluate import _at, _quat_at

    a = np.arange(10.0)[:, None] * np.array([1.0, 2.0, 3.0])
    assert np.allclose(_at(a, 2.27), [2.27, 4.54, 6.81]) and np.allclose(_at(a, -3), a[0]) and np.allclose(_at(a, 99), a[-1])
    q = np.array([[1.0, 0, 0, 0], [-np.cos(0.1), 0, 0, -np.sin(0.1)]])  # the same rotation family, opposite sign
    m = _quat_at(q, 0.5)
    assert np.isclose(np.linalg.norm(m), 1) and np.isclose(abs(m[3]), np.sin(0.05), atol=1e-4)


def test_taper_collision_is_calibs_frustum(scene, servo):
    import mujoco

    from real2sim.mujoco import model as mdl

    cap = scene.cap_dims()
    if "diameter_open_end" not in cap:
        pytest.skip("no fitted taper")
    b = mdl.build(scene, 0, servo, mdl.Physics(cap_shape="taper"), cameras=())
    m, d = b.model, mujoco.MjData(b.model)
    mujoco.mj_forward(m, d)
    ob = b.obj("cap_A")
    g = ob["geoms"][0]
    assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    mid = m.geom_dataid[g]
    V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    W = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]  # world
    L = (W - d.xpos[ob["body"]]) @ d.xmat[ob["body"]].reshape(3, 3)  # cap frame: rim plane z = 0
    r = np.hypot(L[:, 0], L[:, 1])
    lo, hi = L[:, 2] < cap["height"] / 2, L[:, 2] > cap["height"] / 2
    assert np.allclose(L[lo, 2], 0, atol=1e-6) and np.allclose(L[hi, 2], cap["height"], atol=1e-6)
    assert np.isclose(r[lo].max(), cap["diameter_open_end"] / 2, atol=1e-6)
    assert np.isclose(r[hi].max(), cap["diameter_closed_end"] / 2, atol=1e-6)


def test_what_if_runs_need_a_tag():
    from real2sim.mujoco import cli

    base = dict(tag="", set=[], sigma=None, placement="config", no_flex=False)
    cli._need_tag(argparse.Namespace(**base))  # the nominal run
    for extra in ({"set": ["table.z=0.02"]}, {"placement": "grasp"}, {"cap_shape": "taper"}, {"no_flex": True}):
        with pytest.raises(SystemExit):
            cli._need_tag(argparse.Namespace(**{**base, **extra}))
        cli._need_tag(argparse.Namespace(**{**base, **extra, "tag": "t"}))
