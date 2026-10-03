"""Domain randomization (live/dr.py): reproducible draws, applied from the nominal model so
episodes never compound, and a level leaves what it does not randomize alone."""

import numpy as np
import pytest

from real2sim import scene as scene_mod
from real2sim.live import dr, layouts


@pytest.fixture(scope="module")
def eng():
    from real2sim.live.engine_mujoco import MujocoEngine

    sc = scene_mod.load()
    e = MujocoEngine(sc, cameras=("front",))
    e.rest = sc.units().to_urdf(layouts.rest_state(sc))
    e.reset(layouts.random(sc, np.random.default_rng(0), 1), e.rest)
    yield e
    e.close()


def test_draws_are_seeded_and_in_range():
    a, b = dr.draw("all", np.random.default_rng(5)), dr.draw("all", np.random.default_rng(5))
    assert a == b
    assert dr.draw("off", np.random.default_rng(5)) == {"level": "off"}
    assert set(dr.draw("visual", np.random.default_rng(1))) == {"level", "visual"}
    f = a["physics"]
    lo, hi = dr.RANGES["kp_gain"]
    assert all(lo <= k <= hi for k in f["kp_gain"]) and len(f["kp_gain"]) == 6
    with pytest.raises(ValueError):
        dr.draw("heavy", np.random.default_rng(0))


def test_applied_from_nominal_never_compounding(eng):
    m, act = eng.b.model, eng.b.act
    kp0, fl0, cam0 = m.actuator_gainprm[act, 0].copy(), m.dof_frictionloss.copy(), m.cam_pos.copy()
    for s in range(3):
        p = dr.draw("all", np.random.default_rng(s))
        eng.apply_dr(p, np.random.default_rng(s))
        assert np.allclose(m.actuator_gainprm[act, 0], kp0 * p["physics"]["kp_gain"])
        assert np.allclose(-m.actuator_biasprm[act, 1], m.actuator_gainprm[act, 0])
    eng.apply_dr(dr.draw("visual", np.random.default_rng(9)), np.random.default_rng(0))
    assert np.allclose(m.actuator_gainprm[act, 0], kp0) and np.allclose(m.dof_frictionloss, fl0)
    assert eng._bias is None and eng._db is None and eng.goals.dead_time == eng.servo.dead_time
    assert not np.allclose(m.cam_pos, cam0)  # visual moved the camera mounts


def test_sensor_bias_is_in_the_encoder(eng):
    p = dr.draw("physics", np.random.default_rng(4))
    p["physics"]["sensor_noise_deg"] = 0.0
    eng.reset(layouts.random(eng.scene, np.random.default_rng(1), 1), eng.rest)
    eng.apply_dr(p, np.random.default_rng(0))
    bias = np.radians(p["physics"]["sensor_bias_deg"])
    q_true = eng.state() - bias
    for _ in range(30):  # holding the reported pose keeps the joints where they are
        eng.command(q_true + bias)
        eng.step()
    assert np.allclose(eng.state() - bias, q_true, atol=np.radians(1.0))
