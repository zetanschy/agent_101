"""The live sim's pieces, and the claim that matters: fed the recorded actions in
lockstep, the live MuJoCo follower IS the evaluated replay (bitwise)."""

import json
import socket

import numpy as np
import pytest

from real2sim import paths, poselog, scene as scene_mod, episodes
from real2sim.live import goals as G, layouts, protocol


@pytest.fixture(scope="module")
def sc():
    return scene_mod.load()


def test_protocol_round_trip():
    a, b = socket.socketpair()
    msg = {"op": "x", "img": np.arange(24, dtype=np.uint8).reshape(2, 4, 3), "q": np.linspace(0, 1, 6),
           "nested": [{"k": np.array([1, 2], np.int64)}, 3.5, "s"], "none": None}
    protocol.send(a, msg)
    got = protocol.recv(b)
    assert got["op"] == "x" and got["nested"][1:] == [3.5, "s"] and got["none"] is None
    for k in ("img", "q"):
        assert got[k].dtype == msg[k].dtype and np.array_equal(got[k], msg[k])
    assert got["img"].flags.writeable
    protocol.send(b, {"error": "boom"})  # queued as the reply to the next call
    with pytest.raises(RuntimeError, match="boom"):
        protocol.call(a, "noop")
    a.close(), b.close()


def test_online_goals_equal_goalstream():
    from real2sim.mujoco.servo import GoalStream

    rng = np.random.default_rng(0)
    goals = rng.normal(0, 0.5, (40, 6))
    lim = np.tile([[-1.0, 1.0]], (6, 1))
    fps, dead, vmax = 30.0, 1 / 30, 5.29
    ref = GoalStream(goals, fps, dead, vmax, lim, q0=np.zeros(6))
    on = G.OnlineGoals(dead, lim, q0=np.clip(goals[0], -1, 1), max_velocity=vmax, target0=np.zeros(6))
    n = 8
    for k in range(len(goals) - 1):
        on.push(k / fps, goals[k])
        for s in range(n):
            t = (k + s / n) / fps
            assert np.allclose(on.at(t), ref.at(t), atol=1e-12), (k, s)


def test_random_layouts_keep_their_clearances(sc):
    rng = np.random.default_rng(1)
    for _ in range(50):
        L = layouts.random(sc, rng)
        mug = np.array(L["mug"]["xy"])
        caps = [np.array(c["xy"]) for c in L["caps"]]
        assert 1 <= len(caps) <= 3
        for i, c in enumerate(caps):
            assert np.linalg.norm(c - mug) >= layouts.CAP_MUG_CLEARANCE - 1e-9
            assert c[0] >= layouts.X_MIN and c[1] < 0
            assert all(np.linalg.norm(c - o) >= layouts.CAP_CAP_CLEARANCE - 1e-9 for o in caps[:i])
    s2, ep = layouts.scene_with_layout(sc, L)
    assert s2.episode(ep)["caps"] == L["caps"] and s2.hash != sc.hash
    assert layouts.real(sc, 0)["caps"][0]["xy"] == sc.episode(0)["caps"][0]["xy"]


def test_bus_units_round_trip(sc):
    from real2sim import units as U
    from real2sim.live.server import Server

    class Args:
        layout, clock, fps, autoreset = "real:0", "lockstep", 30.0, 0.0

    srv = Server(engine=None, scene=sc, a=Args(), rng=np.random.default_rng(0))
    v = episodes.load()[1].state[::50]  # real readings, degrees + percent
    for x in v:
        n = srv._to_bus(x, "normalized")
        assert np.all(np.abs(n[:5]) <= 100) and n[5] == x[5]
        assert np.allclose(srv._from_bus(n, "normalized"), x, atol=1e-9)
    # the bus's own definition: the calibrated range_min reads -100
    lo, hi, _ = U._cal(U.calibration())
    at_min = U.ticks_to_lerobot(np.r_[lo[:5], lo[5]])
    assert np.allclose(srv._to_bus(at_min, "normalized")[:5], -100.0)


def test_live_mujoco_is_the_replay(sc):
    log = paths.out_dir(sc.ds) / "mujoco" / "logs" / "ep0_action_seed0.npz"
    if not log.exists():
        pytest.skip("no nominal MuJoCo replay log (./robot real2sim mujoco eval)")
    ref = poselog.read(str(log))
    meta = ref["meta"] if isinstance(ref["meta"], dict) else json.loads(str(ref["meta"]))
    if meta.get("scene_hash") != sc.hash:
        pytest.skip("the replay log is from another scene")
    from real2sim.live.engine_mujoco import MujocoEngine

    ep, u = episodes.load()[0], sc.units()
    eng = MujocoEngine(sc, look=False)
    eng.reset(layouts.real(sc, 0), u.to_urdf(ep.state[0]))
    eng.goals.reset(u.to_urdf(ep.action[0]), target0=u.to_urdf(ep.state[0]))  # GoalStream's start
    q = [eng.state()]
    for k in range(len(ep) - 1):
        eng.command(u.to_urdf(ep.action[k]))
        eng.step()
        q.append(eng.state())
    assert np.array_equal(np.array(q), np.asarray(ref["q"]))
    assert all(eng.status()["caps_in_mug"])
    eng.close()
