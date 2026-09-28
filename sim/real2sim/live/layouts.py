"""Where the caps and the mug start, and where the arm rests.

    real:N       episode N's calibrated layout (config/<ds>.calib.json): the same caps,
                 orientations and mug pose the real episode started from
    random[:K]   K caps (default 1-3) and the mug drawn in the reachable, camera-visible
                 part of the mat -- what data collection varies between episodes

A layout is {"caps": [{id, xy, yaw, up}], "mug": {xy, handle_yaw_deg}} in the URDF base
frame, the same shape as an episode entry in the scene, so the engines place it with
the code that places the replayed episodes (scene_with_layout).

WHERE RANDOM LAYOUTS MAY GO (base frame, metres; base -y is forward). The six real caps
sit 0.13-0.40 m from the base, at x -0.20..+0.23; the mugs at 0.19-0.32 m. The overhead
camera sees x -0.31..+0.41 and y +0.04..-0.50 on the table. Caps are drawn 0.14-0.34 m
out and mugs 0.16-0.30 m, within 70 deg of straight ahead, clipped to x >= -0.27 so both
stay in the overhead view. A cap keeps at least evals.tasks' CUP_CLEARANCE_CM (10 cm)
from the mug axis, as the real trials do, so the two are never one blob in the front
camera. It also keeps 50 mm from the other caps and 70 mm from the resting fingertips.
"""

from __future__ import annotations

import copy

import numpy as np

LIVE_EPISODE = 900  # the scene key a live layout is written under (never a real episode index)
CAP_R, MUG_R = (0.14, 0.34), (0.16, 0.30)
AZIMUTH_DEG = 70.0
X_MIN = -0.27
CAP_MUG_CLEARANCE = 0.10
CAP_CAP_CLEARANCE = 0.05
REST_CLEARANCE = 0.07


def rest_state(scene) -> np.ndarray:
    """The rest pose in lerobot units: the median first frame of the used episodes (the
    arm starts every real episode folded at rest; they agree to a few degrees)."""
    from .. import episodes

    eps = episodes.load(scene.ds)
    return np.median(np.stack([eps[i].state[0] for i in scene.used_episodes()]), axis=0)


def real(scene, episode: int) -> dict:
    e = scene.episode(episode)
    caps = [{"id": c["id"], "xy": list(c["xy"]), "yaw": float(c.get("yaw", 0.0)), "up": c.get("up", "closed")}
            for c in e["caps"]]
    mug = {"xy": list(e["mug"]["xy"]), "handle_yaw_deg": float(e["mug"].get("handle_yaw_deg", 0.0))}
    return {"kind": f"real:{episode}", "caps": caps, "mug": mug}


def _polar(rng, r_range) -> np.ndarray:
    for _ in range(1000):
        r = rng.uniform(*r_range)
        a = np.deg2rad(rng.uniform(-AZIMUTH_DEG, AZIMUTH_DEG))
        xy = np.array([r * np.sin(a), -r * np.cos(a)])
        if xy[0] >= X_MIN:
            return xy
    raise RuntimeError("no admissible point")


def random(scene, rng: np.random.Generator, n_caps: int | None = None) -> dict:
    from .. import kinematics

    rest_q = scene.units().to_urdf(rest_state(scene))
    tips = kinematics.fingertips(rest_q)
    rest_xy = np.asarray(tips["mid"][:2], dtype=float)
    n = int(n_caps) if n_caps else int(rng.integers(1, 4))
    for _ in range(2000):
        mug = _polar(rng, MUG_R)
        if np.linalg.norm(mug - rest_xy) < REST_CLEARANCE + 0.045:
            continue
        caps = []
        for _ in range(200):
            c = _polar(rng, CAP_R)
            if (np.linalg.norm(c - mug) >= CAP_MUG_CLEARANCE and np.linalg.norm(c - rest_xy) >= REST_CLEARANCE
                    and all(np.linalg.norm(c - np.asarray(o["xy"])) >= CAP_CAP_CLEARANCE for o in caps)):
                caps.append({"id": "ABC"[len(caps)], "xy": [round(float(c[0]), 4), round(float(c[1]), 4)],
                             "yaw": 0.0, "up": "open" if rng.random() < 0.5 else "closed"})
                if len(caps) == n:
                    return {"kind": f"random:{n}", "caps": caps,
                            "mug": {"xy": [round(float(mug[0]), 4), round(float(mug[1]), 4)],
                                    "handle_yaw_deg": round(float(rng.uniform(-180.0, 180.0)), 1)}}
    raise RuntimeError(f"could not place {n} caps and the mug")


def parse(spec: str, scene, rng: np.random.Generator) -> dict:
    """'real:N' | 'random' | 'random:K'."""
    kind, _, arg = spec.partition(":")
    if kind == "real":
        return real(scene, int(arg or scene.used_episodes()[0]))
    if kind == "random":
        return random(scene, rng, int(arg) if arg else None)
    raise ValueError(f"layout {spec!r}: expected real:N or random[:K]")


def scene_with_layout(scene, layout: dict):
    """(scene copy, episode index) with the layout as a used episode, for the engines'
    episode-based builders. The copy has its own hash; the layout is recorded under it."""
    from ..scene import Scene

    data = copy.deepcopy(dict(scene))
    data["episodes"][str(LIVE_EPISODE)] = {
        "source": f"live layout {layout.get('kind', '?')}", "use": True, "length": 0, "picks": [],
        "caps": copy.deepcopy(layout["caps"]), "mug": copy.deepcopy(layout["mug"])}
    out = Scene(data, scene.ds, scene.layers, scene.origin)
    out.overrides = getattr(scene, "overrides", {})
    return out, LIVE_EPISODE
