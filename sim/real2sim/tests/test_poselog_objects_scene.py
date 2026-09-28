"""Pose-log round trip, object meshes and poses, scene layering and validation."""
import copy

import numpy as np
import pytest

from real2sim import objects, poselog, scene as S
from real2sim.metrics import cap_in_mug


def test_poselog_round_trip(tmp_path, scene):
    T = 12
    q = np.random.default_rng(0).uniform(-1, 1, (T, 6))
    log = poselog.make(30, np.arange(T), q, q + 0.01, objects=["cap_A", "mug"],
                       obj_pos=np.zeros((T, 2, 3)), obj_quat=np.tile([1.0, 0, 0, 0], (T, 2, 1)),
                       meta=poselog.meta("test", "unit", 0, scene.hash, dt=1 / 600, substeps=20, seed=1),
                       x_pen_max=np.zeros(T))
    poselog.fill_links_from_q(log)
    path = tmp_path / "log.npz"
    poselog.write(path, log)
    back = poselog.read(path)
    for k in ("q", "ctrl", "links_pos", "links_quat", "obj_pos", "obj_quat", "x_pen_max"):
        assert np.array_equal(back[k], log[k]), k
    assert back["meta"]["scene_hash"] == scene.hash and back["links"].tolist()[5] == "gripper"
    bad = copy.deepcopy(log)
    bad["q"] = bad["q"][:-1]
    with pytest.raises(ValueError):
        poselog.validate(bad)


def test_real_poselog(extracted):
    log = poselog.from_dataset(episode=0)
    poselog.validate(log)
    assert log["q"].shape == (488, 6) and log["links_pos"].shape == (488, 7, 3)


def test_meshes_watertight_with_dims(scene):
    cd, md = scene.cap_dims(), scene.mug_dims()
    cap = objects.cap_mesh(cd)
    assert cap.is_watertight and cap.is_volume
    assert np.allclose(cap.bounds, [[-cd["diameter"] / 2] * 2 + [0], [cd["diameter"] / 2] * 2 + [cd["height"]]], atol=1e-6)
    parts = objects.mug_parts(md)
    for p in parts.values():
        assert p.is_watertight and p.is_volume
    Ro = md["outer_diameter"] / 2
    assert np.allclose(parts["body"].bounds, [[-Ro, -Ro, 0], [Ro, Ro, md["height"]]], atol=1e-6)
    hb = parts["handle"].bounds
    assert abs(hb[1, 0] - (Ro + md["handle"]["protrusion"])) < 2e-4  # reaches the protrusion
    assert hb[0, 0] > Ro - md["wall"]  # never pokes into the cavity
    inner = np.pi * (Ro - md["wall"]) ** 2 * (md["height"] - md["floor"])
    assert abs(parts["body"].volume - (np.pi * Ro ** 2 * md["height"] - inner)) / parts["body"].volume < 0.01


def test_object_poses_and_in_mug(scene):
    objs = {o["name"]: o for o in objects.episode_objects(scene, 1)}
    h = scene.cap_dims()["height"]
    assert np.isclose(objs["cap_A"]["pos"][2], scene.table_z())  # closed-up rests on its rim
    assert np.isclose(objs["cap_B"]["pos"][2], scene.table_z() + h)  # open-up is flipped
    c = objects.cap_centre(objs["cap_B"]["pos"], objs["cap_B"]["quat"], scene.cap_dims())
    assert np.isclose(c[2], scene.table_z() + h / 2)
    mug = objs["mug"]
    inside = mug["pos"] + [0.0, 0.0, 0.02]
    args = (mug["pos"], mug["quat"], scene.cap_dims(), scene.mug_dims())
    assert cap_in_mug(inside, [1.0, 0, 0, 0], *args)
    assert not cap_in_mug(objs["cap_A"]["pos"], objs["cap_A"]["quat"], *args)


def test_scene_merge_hash_and_validation(scene):
    assert not S.validate(dict(scene)) and scene.used_episodes() == [0, 1, 2]
    over = {"table": {"source": "fitted: test", "z": 0.025}, "robot": {"joint_offsets": {"deg": [0, 0, -2.8, 0, 0]}}}
    merged = S.deep_merge(dict(scene), over)
    assert merged["table"]["z"] == 0.025 and merged["table"]["range"] == scene["table"]["range"]
    assert merged["robot"]["joint_offsets"]["source"] == scene["robot"]["joint_offsets"]["source"]
    assert S.content_hash(merged) != scene.hash and S.content_hash(dict(scene)) == scene.hash
    broken = copy.deepcopy(dict(scene))
    del broken["cameras"]["front"]["pose"]["source"]
    broken["episodes"]["0"]["picks"][0]["close"] = [300, 200]
    errs = S.validate(broken)
    assert any("cameras.front.pose" in e for e in errs) and any("close" in e for e in errs)
    cam = scene.camera("front")
    again = S.Scene(S.deep_merge(dict(scene), {"cameras": {"front": cam.to_config({"camera": "spec: x", "intrinsics": "fitted: x",
                                                                                   "distortion": "fitted: x", "pose": "fitted: x"})}}),
                    scene.ds, scene.layers, {})
    assert not S.validate(dict(again)) and np.allclose(again.camera("front").T_parent_cam, cam.T_parent_cam, atol=1e-8)
