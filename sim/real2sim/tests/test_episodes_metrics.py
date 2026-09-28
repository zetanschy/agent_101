"""Episode slicing on the extracted data, and the metrics on real signals."""
import numpy as np

from real2sim import episodes, metrics


def test_episode_slicing(extracted):
    eps = episodes.load()
    assert sorted(eps) == [0, 1, 2] and 3 in episodes.load(include_excluded=True)
    assert [len(eps[i]) for i in (0, 1, 2)] == [488, 898, 782]
    z = episodes.npz()
    for e in eps.values():
        assert np.array_equal(e.state, z["state"][e.start:e.stop])
        f = e.frames("front")
        assert f.shape == (len(e), 480, 640, 3) and np.array_equal(f[0], episodes.frames(cam="front")[e.start])
        assert e.frames("grip")[-1].mean() > 1


def test_real_follower_lag(extracted, scene):
    """The follower trails the leader by 4.5-4.9 frames (joint_units report)."""
    u = scene.units()
    for e in episodes.load().values():
        r = metrics.joint_rmse_deg(u.to_urdf(e.action), e.state, u, max_lag=10)
        assert -6 <= r["lag"] <= -4  # the ACTION leads the state, so as "sim" it is early
        assert r["mean_arm_at_lag"] < r["mean_arm"]


def test_gripper_events_episode0(extracted):
    ev = metrics.gripper_events(episodes.load()[0].state[:, 5])
    closes = [e for e in ev if e["kind"] == "close"]
    assert any(205 <= e["start"] <= 235 for e in closes), ev  # the close at 212-231


def test_fingertip_error_zero_on_itself(extracted, scene):
    e = episodes.load()[0]
    u = scene.units()
    r = metrics.fingertip_error_mm(u.to_urdf(e.state[:50]), e.state[:50], u)
    assert r["max"] < 1e-6


def test_image_metrics():
    rng = np.random.default_rng(0)
    a = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-10, 11, a.shape), 0, 255).astype(np.uint8)
    assert metrics.masked_psnr(a, a) == float("inf") and 25 < metrics.masked_psnr(a, b) < 40
    assert abs(metrics.masked_ssim(a, a) - 1) < 1e-9 and metrics.masked_ssim(a, b) < 1
    m = np.zeros((60, 80), bool)
    m[10:30, 10:30] = True
    assert metrics.silhouette_iou(m, m) == 1.0 and metrics.silhouette_iou(m, ~m) == 0.0
    assert metrics.penetration_summary([0.0002, 0.0011])["max_mm"] == 1.1
