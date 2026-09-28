"""Units: lerobot <-> ticks <-> URDF radians, against the follower calibration."""
import numpy as np

from real2sim import units


def test_ticks_round_trip_integers():
    cal = units.calibration()
    lo, hi = units._cal(cal)[:2]
    rng = np.random.default_rng(0)
    P = rng.integers(lo, hi + 1, size=(1000, 6)).astype(float)
    back = units.lerobot_to_ticks(units.ticks_to_lerobot(P, cal), cal)
    assert np.abs(back - P).max() < 1e-9


def test_dataset_values_are_integer_ticks(extracted):
    from real2sim import episodes

    st = episodes.npz()["state"].astype(np.float64)
    P = units.lerobot_to_ticks(st, units.calibration())
    # float32 storage: 1e-3 tick is ~1e-4 deg, far below one tick (0.088 deg)
    assert np.abs(P - np.rint(P)).max() < 2e-3


def test_urdf_round_trip_and_offsets():
    u = units.Units(offsets_deg=(1.0, -2.0, 3.0, 0.5, -0.5))
    v = np.array([[10.0, -98.0, 95.9, 75.5, -165.0, 14.8], [0, 0, 0, 0, 0, 0]])
    q = u.to_urdf(v)
    assert np.allclose(q[0, :5], np.radians(v[0, :5] + np.array(u.offsets_deg)))
    assert np.allclose(u.to_lerobot(q), v)


def test_gripper_maps():
    a, b = units.tick_physical_map()
    c = units.GRIPPER_CANDIDATES["tick_physical"]
    assert abs(a - c["a_deg"]) < 1e-3 and abs(b - c["b_deg_per_pct"]) < 1e-5
    u = units.Units()
    # the shut stop (2049 ticks = 1.316 %) is the finger-mesh touch angle
    pct_shut = (units.JAW_SHUT_TICK - 2029) / 15.2
    assert abs(np.degrees(u.jaw_rad(pct_shut)) - units.JAW_TOUCH_DEG) < 0.01
    k = units.Units(grip_a_deg=-10.0, grip_b_deg_per_pct=1.1)
    assert abs(np.degrees(k.jaw_rad(13.4)) - np.degrees(u.jaw_rad(13.4))) < 0.1  # the maps cross at 13.4 %


def test_limits_widened_and_hold_the_data(extracted):
    from real2sim import episodes

    lim = np.degrees(units.limits_rad())
    urdf = np.degrees(units.urdf_limits_rad())
    assert lim[1, 0] < urdf[1, 0] and lim[2, 1] > urdf[2, 1] and lim[3, 1] > urdf[3, 1] and lim[4, 0] < urdf[4, 0]
    st = units.Units().to_urdf(episodes.npz()["state"])
    L = units.limits_rad()
    assert np.all(st >= L[:, 0] - 1e-6) and np.all(st <= L[:, 1] + 1e-6)
