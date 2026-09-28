"""The ensemble draws are the MuJoCo track's, seed by seed; the ensemble block has its keys;
LOOK's textures convert and carry the scene units look.json states."""
import json

import numpy as np
import pytest

from real2sim import paths


def test_mujoco_draws_are_the_mujoco_tracks(scene):
    from real2sim.isaac.replay import mujoco_draws
    from real2sim.mujoco.evaluate import draw_perturbation

    pert = mujoco_draws(scene, 1, 4, 1000)
    assert pert[0] == {"env": 0, "nominal": True}
    d = scene.cap_dims()
    for e in (1, 2, 3):
        ref = draw_perturbation(scene, 1, np.random.default_rng(1000 + e - 1)).as_dict()
        p = pert[e]
        assert p["mujoco_seed"] == 1000 + e - 1 and p["mujoco"] == ref
        assert p["table_dz"] == ref["table_dz"] and p["friction_scales"] == ref["friction_scale"]
        assert p["cap_dxy"] == {c: list(v) for c, v in ref["cap_dxy"].items()}
        assert abs(p["cap_scale"] * d["diameter"] - ref["cap_diameter"]) < 1e-12
        if ref.get("cap_height") is not None:
            assert abs(p["cap_zscale"] * d["height"] - ref["cap_height"]) < 1e-12
    # a --sigma string reaches the MuJoCo function as its dict
    s = mujoco_draws(scene, 0, 2, 7, "cap_xy=0.003,cap_d=0.029:0.030")[1]
    assert s["sigma"] == {"cap_xy": 0.003, "cap_d": (0.029, 0.03)} and 0.029 <= s["mujoco"]["cap_diameter"] <= 0.030


def test_ensemble_block_has_the_mujoco_keys():
    from real2sim.isaac.evaluate import ensemble_summary

    pick = {"cap": "A", "grasped": True, "lifted": True, "carried": True, "in_mug": True, "drop_err_frames": 1,
            "jaw_hold_err_deg": 0.2, "slip_mm": 0.01, "pen_fc_max_mm": 0.1}
    row = {"env": "env_001", "picks": [pick], "success": True, "finite": True, "outcome": {"agree": True},
           "pen_fc": {"max_mm": 0.1, "p99_mm": 0.05}, "mug_moved_mm": 0.0, "perturbation": {"mujoco_seed": 1000},
           "isaac": {"penetration_mm": {"offline_finger_cap": {"max_mm": 0.2}}}}
    out = ensemble_summary([row, {**row, "env": "env_002", "success": False,
                                  "picks": [{**pick, "in_mug": False, "drop_err_frames": None}]}],
                           {"mode": "action", "args": {"perturb": "mujoco", "seed": 1000}})
    # the keys mujoco.evaluate.ensemble writes, same statistics
    for k in ("n", "success_rate", "outcome_agreement", "per_pick", "pen_fc_max_mm", "pen_fc_p99_mm_median",
              "mug_moved_mm_max", "errors", "seeds"):
        assert k in out
    assert out["n"] == 2 and out["success_rate"] == 0.5 and out["seed0"] == 1000
    assert out["per_pick"]["A"]["in_mug"] == 0.5 and out["per_pick"]["A"]["drop_err_frames_median"] == 1.0


def test_look_textures_and_units(scene):
    lj = paths.out_dir(scene.ds, "look") / "look.json"
    if not lj.exists():
        pytest.skip("run ./robot real2sim look build first")
    pytest.importorskip("cv2")
    from real2sim.isaac.look import env_horizontal_irradiance, prepare_textures

    look = json.loads(lj.read_text())
    tex = prepare_textures(look, lj.parent, paths.out_dir(scene.ds, "isaac", "look_tex", create=True))
    # the environment's fill is the ambient share (look README: total horizontal irradiance 1)
    e = env_horizontal_irradiance(tex["env"])
    assert abs(e - look["lights"]["ambient"]["horizontal_irradiance_scene_units"]) < 0.02
    k = look["lights"]["key"]
    assert abs(e + k["normal_irradiance_scene_units"] * k["direction_to_light"][2] - 1.0) < 0.02
    # reflections see only the observed room: next to nothing overhead
    assert env_horizontal_irradiance(tex["env_glossy"]) < 0.01
