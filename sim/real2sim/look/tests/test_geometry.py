"""Plates median, band refinement, table grid and sampling, the shadow model."""
import numpy as np
import pytest

from real2sim.look import LABELS, lighting, plates, segment, table


def test_masked_median_matches_nanmedian():
    rng = np.random.default_rng(0)
    st = rng.integers(0, 256, (9, 20, 30, 3)).astype(np.uint8)
    ok = rng.random((9, 20, 30)) > 0.4
    ok[:, 0, 0] = False  # never seen
    med, cnt = plates.masked_median_u8(st, ok, chunk_rows=7)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the all-masked pixel
        ref = np.nanmedian(np.where(ok[..., None], st.astype(float), np.nan), axis=0)
    both = np.isfinite(ref)
    assert np.allclose(med[both], ref[both]) and np.isnan(med[0, 0]).all() and cnt[0, 0] == 0


def test_refine_keeps_core_and_connected_evidence():
    model = np.zeros((60, 60), bool)
    model[20:40, 20:40] = True
    ev = np.zeros_like(model)
    ev[20:40, 36:44] = True  # evidence from inside the core to 4 px past the model edge
    ev[5:8, 5:8] = True  # far away: outside the band
    out = segment.refine(model, ev, 2, 6)
    assert out[22:38, 22:38].all() and out[25, 42] and not out[6, 6]


def test_assign_caps_rule():
    lab = np.zeros((40, 40), np.uint8)
    mug = np.zeros((40, 40), bool)
    mug[0:20, 0:20] = True
    teal = np.zeros_like(mug)
    teal[5:9, 5:9] = True  # inside the mug, away from the robot -> mug
    teal[30:34, 30:34] = True  # outside the mug -> cap
    out = segment.assign_caps(lab, teal, mug)
    assert (out[5:9, 5:9] == LABELS["mug"]).all() and (out[30:34, 30:34] == LABELS["cap"]).all()


def test_grid_round_trip_and_layout():
    g = table.Grid(*table.EXTENT, 0.002)
    X, Y = g.xy()
    r, c = g.rc(X, Y)
    R, C = g.shape
    assert np.allclose(r, np.arange(R)[:, None]) and np.allclose(c, np.arange(C)[None, :])
    assert X[0, 0] < X[0, -1] and Y[0, 0] > Y[-1, 0]  # right = +x, up = +y


def test_sample_matches_remap_beyond_shrt_max():
    import cv2

    rng = np.random.default_rng(3)
    img = rng.random((50, 70, 3)).astype(np.float32)
    uv = rng.uniform(0, 68, (40000, 2)).astype(np.float32)
    got = table.sample(img, uv)
    ref = cv2.remap(img, uv[:30000, 0].reshape(1, -1), uv[:30000, 1].reshape(1, -1), cv2.INTER_LINEAR)[0]
    assert got.shape == (40000, 3) and np.allclose(got[:30000], ref, atol=1e-6)


def test_rectify_recovers_a_table_pattern(scene):
    """Paint a pattern on the table through the front camera; rectify it back."""
    cam, z = scene.camera("front"), scene.table_z()
    g = table.Grid(-0.2, 0.2, -0.4, -0.1, 0.004)
    v, u = np.mgrid[0:cam.height, 0:cam.width]
    P = cam.unproject_to_plane(np.stack([u.ravel(), v.ravel()], 1).astype(float), z=z).reshape(cam.height, cam.width, 3)
    img = (0.5 + 0.4 * np.sin(P[..., 0] * 60) * np.cos(P[..., 1] * 45))[..., None].repeat(3, -1).astype(np.float32)
    tex, ok = table.rectify(img, np.ones(img.shape[:2], bool), cam, g, z)
    X, Y = g.xy()
    truth = 0.5 + 0.4 * np.sin(X * 60) * np.cos(Y * 45)
    assert ok.mean() > 0.99 and np.nanmax(np.abs(tex[..., 0] - truth)) < 0.01


def test_shadow_model_umbra_penumbra_lit(scene):
    d = lighting.direction(np.radians(-122.0), np.radians(44.0))
    occ = lighting.occluders(scene, 0)
    c = np.asarray(scene.episode(0)["mug"]["xy"])
    tz = scene.table_z()
    away = -d[:2] / np.linalg.norm(d[:2])  # the shadow falls away from the light
    pts = np.array([[*(c + away * 0.06), tz], [*(c - away * 0.20), tz], [*(c + away * 2.0), tz]])
    vis = lighting.visibility(pts, d, np.radians(1.0), occ)
    assert vis[0] == pytest.approx(0.0) and vis[1] == pytest.approx(1.0) and vis[2] == pytest.approx(1.0)
