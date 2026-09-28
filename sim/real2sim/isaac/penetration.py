"""Offline overlap of fingers, caps, table and mug, from a pose log and the meshes.

PhysX reports its own separations (contact.py), but those are the solver's view of
its own shapes (the SDF of the fingers, the convex pieces of the cap). This is the
independent check RULE 2 asks for: the LOGGED poses, the real meshes, exact geometry.

    fingers -> cap    surface samples of the two finger meshes (the same STL the USD's
                      SDF was cooked from; kinematics.meshes), placed by the logged
                      gripper / jaw link poses, and their depth inside each cap's
                      solid (geometry.revolved_depth: exact, seam-free)
    cap -> table      depth of the cap's surface below the table top
    cap -> mug        depth of the cap's surface inside the mug body's solid
    fingers -> table  depth of the finger samples below the table top

Accuracy: the finger samples are ~0.3 mm apart on the tips (40k per finger), and the
cap's collision pieces differ from its revolved solid by their chord sagitta
(< 0.26 mm, geometry.py), so a reported depth is good to about 0.3 mm.
"""

from __future__ import annotations

import functools

import numpy as np

from ..kinematics import FIXED_FINGER_MESH, MOVING_JAW_MESH
from ..transforms import quat_to_mat
from . import geometry as G

N_FINGER = 40000


@functools.lru_cache(maxsize=2)
def finger_samples(kin) -> dict:
    """{'gripper': (N,3) fixed-finger samples in the gripper frame, 'jaw': moving jaw in the jaw frame}."""
    out = {}
    for link, name in (("gripper", FIXED_FINGER_MESH), ("jaw", MOVING_JAW_MESH)):
        g = next(m for m in kin.meshes(False) if m["mesh"] == name)
        out[link] = G.sample_surface(g["verts"], g["faces"], N_FINGER, seed=2)
    return out


@functools.lru_cache(maxsize=8)
def _cap_surface(d: float, h: float, wall: float, top: float, n: int = 6000) -> np.ndarray:
    from ..objects import cap_mesh

    m = cap_mesh({"diameter": d, "height": h, "wall": wall, "top": top})
    return G.sample_surface(np.asarray(m.vertices), np.asarray(m.faces), n, seed=3)


def _world(points, pos, quat):
    return points @ quat_to_mat(quat).T + pos


def run(log: dict, scene, kin, cap_scale: float = 1.0, table_dz: float = 0.0, stride: int = 1,
        cap_zscale: float = 1.0) -> dict:
    """Per-frame overlap depths (m) for one pose log. Returns {name: (T,) or (T, ncaps)}.
    cap_scale / cap_zscale / table_dz: an ensemble env's perturbed cap and table."""
    links = list(log["links"])
    gi, ji = links.index("gripper"), links.index("jaw")
    objs = list(log["objects"])
    caps = [i for i, n in enumerate(objs) if n.startswith("cap_")]
    mug = objs.index("mug") if "mug" in objs else None
    cd, md = dict(scene.cap_dims()), scene.mug_dims()
    cd["diameter"] *= cap_scale
    cd["height"] *= cap_zscale
    cd["top"] *= cap_zscale
    prof_cap, prof_mug = G.cap_profile(cd), G.mug_body_profile(md)
    capsurf = _cap_surface(cd["diameter"], cd["height"], cd["wall"], cd["top"])
    fs = finger_samples(kin)
    sx, sy = G.table_tilt(scene)
    tz0 = scene.table_z() + table_dz

    def below_table(P):  # depth of points below the (possibly tilted) table top
        return float(np.max(tz0 + sx * P[:, 0] + sy * P[:, 1] - P[:, 2], initial=0.0))

    T = len(log["frame"])
    r_reach = cd["diameter"] / 2 + 0.003  # finger samples farther than this from the cap axis cannot touch it
    out = {"finger_cap": np.zeros((T, len(caps))), "cap_table": np.zeros((T, len(caps))),
           "cap_mug": np.zeros((T, len(caps))), "finger_table": np.zeros(T), "finger_mug": np.zeros(T)}
    for t in range(0, T, stride):
        P = np.concatenate([_world(fs["gripper"], log["links_pos"][t, gi], log["links_quat"][t, gi]),
                            _world(fs["jaw"], log["links_pos"][t, ji], log["links_quat"][t, ji])])
        out["finger_table"][t] = below_table(P)
        if mug is not None:
            loc = G.to_local(P, log["obj_pos"][t, mug], log["obj_quat"][t, mug])
            near = np.hypot(loc[:, 0], loc[:, 1]) < md["outer_diameter"] / 2 + 1e-3
            if near.any():
                out["finger_mug"][t] = G.revolved_depth(loc[near], prof_mug).max(initial=0.0)
        for k, j in enumerate(caps):
            cp, cq = log["obj_pos"][t, j], log["obj_quat"][t, j]
            if not np.all(np.isfinite(cp)):
                continue
            loc = G.to_local(P, cp, cq)
            near = (np.hypot(loc[:, 0], loc[:, 1]) < r_reach) & (loc[:, 2] > -0.003) & (loc[:, 2] < cd["height"] + 0.003)
            if near.any():
                out["finger_cap"][t, k] = G.revolved_depth(loc[near], prof_cap).max(initial=0.0)
            S = _world(capsurf, cp, cq)
            out["cap_table"][t, k] = below_table(S)
            if mug is not None:
                sl = G.to_local(S, log["obj_pos"][t, mug], log["obj_quat"][t, mug])
                out["cap_mug"][t, k] = G.revolved_depth(sl, prof_mug).max(initial=0.0)
    return out
