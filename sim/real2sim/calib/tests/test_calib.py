"""CALIB building blocks: FK chain, SDFs, polygons, camera latency, parameter vectors,
the silhouette lift, and the saved observations."""
import numpy as np
import pytest

from real2sim import kinematics
from real2sim.calib import chain, masks, objfit, params, timing


def test_chain_matches_mujoco_fk():
    kin = kinematics.default()
    rng = np.random.default_rng(0)
    Q = rng.uniform(-2.0, 2.0, (50, 6))
    L = chain.link_T(Q)
    err = max(np.abs(kin.link_T(q, n) - L[n][i]).max() for i, q in enumerate(Q) for n in kinematics.LINKS)
    assert err < 1e-9


def test_sdf_zero_level_is_the_edge():
    yy, xx = np.mgrid[:101, :101]
    disk = (xx - 50.0) ** 2 + (yy - 50.0) ** 2 <= 30.0 ** 2
    s = masks.sdf(disk)
    assert s[50, 50] < -29 and s[0, 0] > 30
    # along a row the zero crossing sits between the last inside and first outside pixel
    row = s[50]
    k = int(np.argmax(row[50:] > 0)) + 50
    assert row[k - 1] < 0 < row[k] and abs(row[k] + row[k - 1]) < 1.0
    assert abs(masks.bilinear(s, np.array([[50.0, 50.0]]))[0] - s[50, 50]) < 1e-6


def test_polygon_sdf_square():
    sq = objfit.convex_hull(np.array([[0, 0], [10, 0], [10, 10], [0, 10.0]]))
    d = objfit.polygon_sdf(np.array([[5, 5], [12, 5], [5, -3], [10, 5.0]]), sq)
    assert np.allclose(d, [-5, 2, 3, 0], atol=1e-9)


def test_timing_interpolates_within_the_episode():
    st = np.arange(20, dtype=float)[:, None] * np.ones((1, 6))
    ck = timing.Clock(st, [0, 10], [10, 20])
    assert np.allclose(ck.at([5], 0.0), 5) and np.allclose(ck.at([5], 1.5), 3.5)
    assert np.allclose(ck.at([11], 3.0), 10)  # clamped at its episode's first row, not episode 0
    assert np.allclose(ck.at([19], -2.0), 19)


def test_layout_round_trip_and_rebase():
    from real2sim.scene import load

    L, meta = params.from_scene(load())
    L.only_free(["front.rot", "front.t", "robot.off"])
    full = L.full0().copy()
    sl = L.slices()
    full[sl["front.rot"]] = [0.01, -0.02, 0.005]
    full[sl["robot.off"]] = [1, 2, 3, 4, 5]
    assert np.allclose(L.from_solver(L.to_solver(full)), full)
    T = params.Model(L, full, meta).T_parent_cam("front")
    L2 = L.rebase(full)
    assert np.allclose(params.Model(L2, L2.full0(), meta).T_parent_cam("front"), T, atol=1e-12)
    assert np.allclose(L2.block("front.rot").x0, 0)


def test_silhouette_points_lie_on_the_arm():
    pytest.importorskip("mujoco")
    from scipy.spatial import cKDTree

    from real2sim import camera
    from real2sim.calib import render

    q = np.array(kinematics.HOME)
    T_wc = camera.look_at([0.25, -0.35, 0.45], chain.link_T(q[None])["wrist"][0][:3, 3])
    spec = camera.RenderSpec(640, 480, (600.0, 600.0, 319.5, 239.5))
    try:
        ar = render.ArmRenderer({"c": ("world", spec)})
    except Exception as e:  # no EGL on this machine
        pytest.skip(f"no offscreen GL: {e}")
    r = ar.render("c", q, T_wc)
    ar.close()
    link, pc, _ = render.outline_points(r)
    assert len(link) > 200
    pw = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
    rng = np.random.default_rng(0)
    samp = []
    for g, v in kinematics.default().world_vertices(q):
        tri = v[g["faces"]]
        a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        n = int(a.sum() / 1e-7) + 100
        i = rng.choice(len(tri), n, p=a / a.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        samp.append((1 - s)[:, None] * tri[i, 0] + (s * (1 - r2))[:, None] * tri[i, 1] + (s * r2)[:, None] * tri[i, 2])
    d, _ = cKDTree(np.concatenate(samp)).query(pw)
    assert np.percentile(d, 90) < 0.0015  # MEASURED 0.35 mm at 1 m, sample spacing ~0.3 mm


def test_observations_if_extracted():
    from real2sim.calib import observations

    if not (observations.out_dir() / "detections.npz").exists():
        pytest.skip("run ./robot real2sim calib observe first")
    obs = observations.load()
    n = len(__import__("real2sim.episodes", fromlist=["x"]).npz()["index"])
    assert len(obs.rim()) == n and len(obs.summary()) == n
    t = obs.teal("front", rows=range(0, 5))
    assert len(t) >= 5 and {"u", "v", "r_eq", "area"} <= set(t.dtype.names)
    assert obs.mask("grip", 0).shape == (480, 640)
