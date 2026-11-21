"""Where the caps and the mug start, and where the arm rests.

    real:N       episode N's calibrated layout (config/<ds>.calib.json): the same caps,
                 orientations and mug pose the real episode started from
    random[:K]   K caps (default 1-3) and the mug drawn in the reachable, camera-visible
                 part of the mat -- what data collection varies between episodes
    organizer    the clear compartment organizer near where it sits on the bench, and one
                 cap beside it, always closed side up (cavity down); no mug (ORG_* below)

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
# 'organizer': the clear compartment organizer (mujoco/model.py ORGANIZER) and one cap, no
# mug. Its pose is drawn around where it sat in the 2026-11-20 front frame (centre x, y in
# m, yaw in deg; yaw 0 = its length along base x, latches toward the robot), kept with its
# rim in the overhead view, every compartment within ORG_REACH of the base and ORG_REST_
# CLEARANCE from the resting fingertips. The cap keeps ORG_CAP_CLEARANCE from it and its lid,
# and lies closed side up only, to keep the task about the prompt, not the grasp.
ORG_POSE, ORG_DX, ORG_DY, ORG_DYAW = (-0.029, -0.311, 2.0), 0.06, 0.04, 20.0  # pose: its rim at 5 cm
ORG_REACH, ORG_REST_CLEARANCE, ORG_CAP_CLEARANCE = 0.40, 0.015, 0.03


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


def organizer_local(xy, org: dict) -> np.ndarray:
    """Base-frame table points (..., 2) in the organizer's frame (x along it, +y its front)."""
    a = np.radians(org["yaw_deg"])
    d = np.asarray(xy, dtype=float) - np.asarray(org["xy"], dtype=float)
    return np.stack([d[..., 0] * np.cos(a) + d[..., 1] * np.sin(a), -d[..., 0] * np.sin(a) + d[..., 1] * np.cos(a)], -1)


def organizer_gap(xy, org: dict, lid: bool = True) -> np.ndarray:
    """Distance on the table from points (..., 2) to the organizer's footprint (0 inside),
    with or without the open lid behind it."""
    from ..mujoco.model import ORGANIZER as O

    loc = organizer_local(xy, org)
    back = O["depth"] / 2 + (np.sqrt(max(O["lid_depth"] ** 2 - O["height"] ** 2, 0.0)) if lid else 0.0)
    dx = np.maximum(np.abs(loc[..., 0]) - O["length"] / 2, 0.0)
    dy = np.maximum(np.maximum(-back - loc[..., 1], loc[..., 1] - O["depth"] / 2), 0.0)
    return np.hypot(dx, dy)


def organizer(scene, rng: np.random.Generator) -> dict:
    """The organizer near ORG_POSE and one cap beside it (see ORG_* above)."""
    from .. import kinematics
    from ..mujoco.model import ORGANIZER as O

    rest_q = scene.units().to_urdf(rest_state(scene))
    rest_xy = np.asarray(kinematics.fingertips(rest_q)["mid"][:2], dtype=float)
    L, D = O["length"], O["depth"]
    li, di = L - 2 * O["wall"], D - 2 * O["wall"]
    cells = np.array([[li / 2 - (c + 0.5) * li / O["cols"], di / 2 - (r + 0.5) * di / O["rows"]]
                      for r in range(O["rows"]) for c in range(O["cols"])])
    rim = np.array([[sx * L / 2, sy * D / 2] for sx in (-1, 1) for sy in (-1, 1)])
    cap_r = scene.cap_dims()["diameter"] / 2
    for _ in range(2000):
        org = {"xy": [round(float(ORG_POSE[0] + rng.uniform(-ORG_DX, ORG_DX)), 4),
                      round(float(ORG_POSE[1] + rng.uniform(-ORG_DY, ORG_DY)), 4)],
               "yaw_deg": round(float(ORG_POSE[2] + rng.uniform(-ORG_DYAW, ORG_DYAW)), 1)}
        a = np.radians(org["yaw_deg"])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        corners, centres = rim @ R.T + org["xy"], cells @ R.T + org["xy"]
        if (corners[:, 0].min() < X_MIN or corners[:, 1].min() < -0.50
                or np.linalg.norm(centres, axis=1).max() > ORG_REACH
                or organizer_gap(rest_xy, org, lid=False) < ORG_REST_CLEARANCE):
            continue
        for _ in range(200):
            c = _polar(rng, CAP_R)
            if organizer_gap(c, org) >= cap_r + ORG_CAP_CLEARANCE and np.linalg.norm(c - rest_xy) >= REST_CLEARANCE:
                cap = {"id": "A", "xy": [round(float(c[0]), 4), round(float(c[1]), 4)], "yaw": 0.0, "up": "closed"}
                return {"kind": "organizer", "caps": [cap], "organizer": org}
    raise RuntimeError("could not place the organizer and a cap")


def parse(spec: str, scene, rng: np.random.Generator) -> dict:
    """'real:N' | 'random' | 'random:K' | 'organizer'."""
    kind, _, arg = spec.partition(":")
    if kind == "real":
        return real(scene, int(arg or scene.used_episodes()[0]))
    if kind == "random":
        return random(scene, rng, int(arg) if arg else None)
    if kind == "organizer":
        return organizer(scene, rng)
    raise ValueError(f"layout {spec!r}: expected real:N, random[:K] or organizer")


def scene_with_layout(scene, layout: dict):
    """(scene copy, episode index) with the layout as a used episode, for the engines'
    episode-based builders. The copy has its own hash; the layout is recorded under it."""
    from ..scene import Scene

    data = copy.deepcopy(dict(scene))
    data["episodes"][str(LIVE_EPISODE)] = {
        "source": f"live layout {layout.get('kind', '?')}", "use": True, "length": 0, "picks": [],
        **{k: copy.deepcopy(layout[k]) for k in ("caps", "mug", "organizer") if k in layout}}
    out = Scene(data, scene.ds, scene.layers, scene.origin)
    out.overrides = getattr(scene, "overrides", {})
    return out, LIVE_EPISODE
