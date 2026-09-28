"""The replay driver: modes, pose logs, metrics plumbing, headless viewer behaviour."""
import numpy as np
import pytest

from real2sim import poselog
from real2sim.mujoco import replay


@pytest.fixture(scope="module")
def feasible(scene, built):
    """A table 1 mm below the lowest recorded arm point of episode 0: the tests check the
    driver, not the (provisional) table height."""
    from real2sim.mujoco import evaluate, model

    dz = (evaluate.recorded_clearance_mm(scene, built, 0) - 1.0) / 1000
    return model.Perturbation(table_dz=float(dz))


@pytest.mark.parametrize("mode", ["action", "track", "kinematic"])
def test_short_window_logs_validly(scene, servo, feasible, mode):
    res = replay.run(scene, 0, replay.Config(mode=mode, frames=(0, 45), perturb=feasible), servo)
    log = res.log
    poselog.validate(log)
    assert res.finite and len(log["q"]) == 45
    assert log["meta"]["scene_hash"] == scene.hash
    assert log["meta"]["physical"] == (mode != "kinematic")
    real = scene.units().to_urdf(np.asarray(__import__("real2sim.episodes", fromlist=["x"]).load()[0].state[:45]))
    err = np.degrees(np.abs(log["q"][:, :5] - real[:, :5])).max()
    assert err < {"kinematic": 1e-6, "track": 0.5, "action": 6.0}[mode]


def test_episode_metrics_keys(scene, servo, feasible):
    from real2sim.mujoco import evaluate

    res = replay.run(scene, 0, replay.Config(mode="track", perturb=feasible), servo)
    m = evaluate.episode_metrics(scene, res.log, res.built, pixels=False)
    assert {"picks", "joint_rmse_deg", "fingertip_err_mm", "pen_fc", "mug_moved_mm", "outcome"} <= set(m)
    assert [p["cap"] for p in m["picks"]] == [pk["cap"] for pk in scene.episode(0)["picks"]]


def test_viewer_refuses_without_display(monkeypatch, built):
    from real2sim.mujoco import viewer

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="no display"):
        viewer.play_log(built, {"q": np.zeros((1, 6)), "fps": 30, "objects": []})
