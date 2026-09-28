"""The pose log: one .npz per simulated (or real) episode, the same for every engine.

    log = poselog.make(fps=30, frame=..., q=..., ctrl=..., objects=["cap_A", "mug"],
                       obj_pos=..., obj_quat=..., meta=poselog.meta(engine="mujoco", ...))
    poselog.fill_links_from_q(log)   # an engine may log q + objects only
    poselog.write(path, log);  log = poselog.read(path)

FIELDS (T frames, L links, O objects; all poses in the URDF BASE frame, quats wxyz):
    fps        ()        int, frames per second of `frame` (the dataset's 30)
    frame      (T,)      int, episode-local frame index each row corresponds to
    q          (T, 6)    float, URDF joint radians (units.URDF_JOINTS order) ACTUALLY reached
    ctrl       (T, 6)    float, the joint targets commanded at that frame (URDF radians)
    links      (L,)      str, link names (kinematics.LINKS by default)
    links_pos  (T, L, 3), links_quat (T, L, 4)
    objects    (O,)      str, 'cap_<id>' / 'mug' (objects.episode_objects names)
    obj_pos    (T, O, 3), obj_quat (T, O, 4): object ORIGIN poses (objects.py conventions);
                         NaN rows where an object is not tracked
    meta       ()        str, JSON: engine, mode, episode, seed, scene_hash, dt, substeps,
                         perturbation, git (+ anything else the engine wants to record)
Extra arrays are allowed if their name starts with "x_" (e.g. x_pen_max (T,) for the
finger-cap penetration each frame); validate() checks only that their first dim is T.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np

from . import paths

REQUIRED = ("fps", "frame", "q", "ctrl", "links", "links_pos", "links_quat", "objects", "obj_pos", "obj_quat", "meta")
META_KEYS = ("engine", "mode", "episode", "seed", "scene_hash", "dt", "substeps", "perturbation", "git")


def git_describe() -> str:
    try:
        return subprocess.run(["git", "-C", str(paths.REPO), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def meta(engine: str, mode: str, episode: int, scene_hash: str, dt: float | None = None, substeps: int | None = None,
         seed: int | None = None, perturbation: dict | None = None, **extra) -> dict:
    """The meta dict with every META_KEYS entry filled (git describe looked up here)."""
    return {"engine": engine, "mode": mode, "episode": int(episode), "seed": seed, "scene_hash": scene_hash,
            "dt": dt, "substeps": substeps, "perturbation": perturbation or {}, "git": git_describe(), **extra}


def make(fps, frame, q, ctrl=None, links=(), links_pos=None, links_quat=None, objects=(), obj_pos=None,
         obj_quat=None, meta=None, **extra) -> dict:
    """Assemble a log dict with the right dtypes; missing link/object arrays become empty."""
    q = np.asarray(q, dtype=float)
    T = len(q)
    L, O = len(links), len(objects)
    log = {
        "fps": np.int64(fps), "frame": np.asarray(frame, dtype=np.int64), "q": q,
        "ctrl": np.asarray(ctrl, dtype=float) if ctrl is not None else np.full_like(q, np.nan),
        "links": np.array(list(links), dtype=str), "objects": np.array(list(objects), dtype=str),
        "links_pos": np.asarray(links_pos, dtype=float) if links_pos is not None else np.zeros((T, L, 3)),
        "links_quat": np.asarray(links_quat, dtype=float) if links_quat is not None else np.zeros((T, L, 4)),
        "obj_pos": np.asarray(obj_pos, dtype=float) if obj_pos is not None else np.zeros((T, O, 3)),
        "obj_quat": np.asarray(obj_quat, dtype=float) if obj_quat is not None else np.zeros((T, O, 4)),
        "meta": dict(meta or {}),
    }
    for k, v in extra.items():
        if not k.startswith("x_"):
            raise ValueError(f"extra array {k!r} must be named x_...")
        log[k] = np.asarray(v)
    return log


def validate(log: dict) -> None:
    """Raise ValueError listing every problem; return None when the log is well-formed."""
    err = [f"missing '{k}'" for k in REQUIRED if k not in log]
    if err:
        raise ValueError("; ".join(err))
    T = len(log["q"])
    L, O = len(log["links"]), len(log["objects"])
    shapes = {"frame": (T,), "q": (T, 6), "ctrl": (T, 6), "links_pos": (T, L, 3), "links_quat": (T, L, 4),
              "obj_pos": (T, O, 3), "obj_quat": (T, O, 4)}
    for k, s in shapes.items():
        if np.shape(log[k]) != s:
            err.append(f"{k} has shape {np.shape(log[k])}, expected {s}")
    if int(log["fps"]) <= 0:
        err.append("fps must be > 0")
    if T and np.any(np.diff(np.asarray(log["frame"])) <= 0):
        err.append("frame must be strictly increasing")
    for names in ("links", "objects"):
        if len(set(np.asarray(log[names]).tolist())) != len(log[names]):
            err.append(f"{names} has duplicate names")
    for k in ("links_quat", "obj_quat"):
        qq = np.asarray(log[k], dtype=float)
        if qq.size:
            n = np.linalg.norm(qq, axis=-1)
            ok = np.isnan(n) | (np.abs(n - 1) < 1e-4)
            if not ok.all():
                err.append(f"{k}: {int((~ok).sum())} quaternions are not unit (NaN is allowed for untracked)")
    if not np.all(np.isfinite(np.asarray(log["q"], dtype=float))):
        err.append("q must be finite")
    m = log["meta"] if isinstance(log["meta"], dict) else json.loads(str(log["meta"]))
    missing = [k for k in META_KEYS if k not in m]
    if missing:
        err.append(f"meta lacks {missing} (use poselog.meta(...))")
    for k, v in log.items():
        if k.startswith("x_") and np.shape(v)[:1] != (T,):
            err.append(f"{k}: first dimension must be T={T}")
    if err:
        raise ValueError("pose log invalid: " + "; ".join(err))


def write(path, log: dict | None = None, **arrays) -> None:
    """write(path, log) or write(path, **fields). Validates first."""
    log = dict(log or {}, **arrays)
    validate(log)
    out = {k: (np.array(json.dumps(v, default=float)) if k == "meta" else np.asarray(v)) for k, v in log.items()}
    np.savez_compressed(path, **out)


def read(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        log = {k: z[k] for k in z.files}
    log["meta"] = json.loads(str(log["meta"]))
    log["fps"] = int(log["fps"])
    return log


def fill_links_from_q(log: dict, kin=None, links=None) -> dict:
    """Fill links/links_pos/links_quat by forward kinematics of log['q'] (in place)."""
    from .kinematics import LINKS, default

    kin = kin or default()
    links = tuple(links or LINKS)
    log["links_pos"], log["links_quat"] = kin.link_poses_batch(log["q"], links)
    log["links"] = np.array(links, dtype=str)
    return log


def from_dataset(ds: str | None = None, episode: int = 0, units=None) -> dict:
    """The REAL episode as a pose log (engine 'real'): q from observation.state, ctrl
    from action, links by FK. No objects (they are tracked by CALIB, not recorded)."""
    from . import episodes, scene

    sc = scene.load(ds)
    u = units or sc.units()
    ep = episodes.load(ds, include_excluded=True)[episode]
    log = make(ep.fps, np.arange(len(ep)), u.to_urdf(ep.state), u.to_urdf(ep.action),
               meta=meta("real", "recorded", episode, sc.hash, dt=1.0 / ep.fps, substeps=1, seed=None))
    return fill_links_from_q(log)
