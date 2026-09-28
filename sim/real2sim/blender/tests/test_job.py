"""The render job (host side, no Blender): camera time, dedupe, reflections, materials, camera model."""
import numpy as np
import pytest

from real2sim.blender import camera_model as CM
from real2sim.blender import job as J
from real2sim.blender import materials as MAT


def _log(T=5):
    fr = np.arange(T)
    pos = np.zeros((T, 1, 3))
    pos[:, 0, 0] = fr * 0.01  # 1 cm per row along x
    ang = np.radians(10.0) * fr  # 10 deg per row about z
    quat = np.stack([np.cos(ang / 2), 0 * ang, 0 * ang, np.sin(ang / 2)], 1)[:, None]
    return {"frame": fr, "links_pos": pos, "links_quat": quat, "obj_pos": np.zeros((T, 0, 3)),
            "obj_quat": np.zeros((T, 0, 4))}


def test_camera_time_interpolates_between_rows_and_clamps():
    lp, lq, _, _ = J.poses_at(_log(), [3, 0], lag=1.5)
    assert lp[0, 0, 0] == pytest.approx(0.015)  # row 3 of the image = state at t 1.5
    assert np.degrees(2 * np.arctan2(lq[0, 0, 3], lq[0, 0, 0])) == pytest.approx(15.0, abs=0.01)
    assert lp[1, 0, 0] == pytest.approx(0.0)  # before the log: clamped to its first row


def test_dedupe_collapses_identical_poses_in_first_seen_order():
    a = np.array([[0.0], [0.0], [1.0], [0.0], [1.0]])
    uniq, inv = J.dedupe(a)
    assert uniq.tolist() == [0, 2] and inv.tolist() == [0, 0, 1, 0, 1]


def test_untracked_objects_are_parked_out_of_view():
    M = J.to_matrices(np.full((1, 3), np.nan), np.full((1, 4), np.nan))
    assert M[0, 2, 3] == -100.0 and np.allclose(M[0, :3, :3], np.eye(3))


@pytest.fixture(scope="module")
def look():
    try:
        return J.load_look()
    except FileNotFoundError:
        pytest.skip("look assets not built")


def test_reflections_see_only_the_observed_room(look, tmp_path):
    tex = J.textures(look[0], look[1], tmp_path)
    env, glossy, unseen = (np.load(tmp_path / tex[k]) for k in ("env", "env_glossy", "env_unseen"))
    assert np.allclose(glossy[..., :3] + unseen[..., :3], env[..., :3])  # a partition of the environment
    assert ((glossy[..., :3] == 0) | (unseen[..., :3] == 0)).all()
    cyl = np.load(tmp_path / tex["backdrop"])
    assert set(np.unique(cyl[..., 3])) <= {0.0, 1.0}  # alpha = the wall's seen mask
    assert cyl[..., 3].mean() == pytest.approx(tex["backdrop_seen_fraction"], abs=1e-3)


def test_material_table(look):
    mats = MAT.materials(look[0])
    assert all(m["ior"] == pytest.approx(1.5) for m in mats.values() if m["metallic"] < 0.5)  # F0 0.04
    assert [n for n, m in mats.items() if m.get("room_reflection")] == ["mug_steel_interior"]
    grip = mats["cap_teal"]["per_camera"]["grip"]["base_color"]
    assert grip == pytest.approx(look[0]["materials"]["cap_teal"]["wrist_camera_albedo"])
    assert mats["mat_black_glossy"]["texture"] == "table_albedo"


def test_front_camera_model_maps_lit_white_pla_to_the_white_reference(look):
    """A horizontal white PLA face under the total irradiance 1 has radiance rho / pi;
    the front camera must return LOOK's L_PLA for it (reference white balance)."""
    L = np.asarray(look[0]["cameras"]["front"]["white_reference"]["L_PLA_lin"])
    rho = look[0]["cameras"]["front"]["white_reference"]["rho_PLA"]
    r = CM.Response.__new__(CM.Response)
    r.cam, r.rho, r.L, r.wb_mode, r.state = "front", rho, L.astype(np.float32), "reference", None
    out = r.colour(np.full((1, 1, 3), rho / np.pi, np.float32), 0, 0, 0)
    assert np.allclose(out[0, 0], L, rtol=1e-5)
