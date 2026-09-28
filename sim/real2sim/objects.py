"""The cap and the mug: procedural meshes from the scene's dimensions, and their poses.

No CAD or scan of either object exists (tooling report: nothing on disk is a teal cap
or a red enamel mug), so both are built from the few dimensions the data give, in
config objects.cap / objects.mug. Rebuilding is deterministic, and the file names
carry a hash of the dimensions, so a changed dimension can never be confused with a
stale mesh.

CONVENTIONS (one each, used by every engine):
    cap   origin at the centre of the RIM plane (the open end), +z towards the closed
          top. A skirt tube (outer radius d/2, wall `wall`) under a top disk (`top`
          thick): hollow underneath. Closed-up vs open-up is a POSE, not a mesh:
          closed-up = identity, resting on its rim at z = table;
          open-up   = 180 deg about x, resting on its top, origin at z = table + height.
    mug   origin at the centre of the bottom face, +z up, handle along +x. A revolved
          body (outer wall, rim, inner wall, floor) plus a handle: a round tube swept
          along a half-ellipse from z_bottom to z_top, reaching `protrusion` beyond
          the outer wall, its ends buried 1 mm into the wall (never into the cavity).

WATERTIGHTNESS. The cap and the mug body are single closed 2-manifolds. The mug is
body + handle, two closed parts that overlap inside the wall: this box has no
boolean backend (no manifold3d), and renderers and convex decomposition do not need
the union. `mug_parts()` returns them separately.

Meshes need trimesh (host python3, 45pysaac, mjlab env; imported lazily). Poses are
pure numpy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import paths
from .transforms import axis_angle_quat, quat_mul, quat_to_mat

MESH_VERSION = 1  # bump when the generators change shape, so the hash changes too
_FLIP_X = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x, wxyz


CAP_KEYS = ("diameter", "height", "wall", "top")
MUG_KEYS = ("outer_diameter", "height", "wall", "floor")
HANDLE_KEYS = ("protrusion", "z_bottom", "z_top", "thickness")


def dims_hash(dims: dict) -> str:
    """10 hex of the GEOMETRY the generators use (mass or colour changes keep the name)."""
    geo = {k: dims[k] for k in CAP_KEYS + MUG_KEYS if k in dims}
    if "handle" in dims:
        geo["handle"] = {k: dims["handle"][k] for k in HANDLE_KEYS}
    blob = json.dumps({"v": MESH_VERSION, **geo}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _revolve(profile, sections: int):
    import trimesh

    m = trimesh.creation.revolve(np.asarray(profile, dtype=float), sections=sections)
    if m.volume < 0:
        m.invert()
    return m


def cap_mesh(dims: dict, sections: int = 64):
    """Hollow cap, origin at the rim-plane centre, +z to the closed top."""
    R, h, w, t = dims["diameter"] / 2, dims["height"], dims["wall"], dims["top"]
    Ri = R - w
    return _revolve([(Ri, 0), (R, 0), (R, h), (0, h), (0, h - t), (Ri, h - t), (Ri, 0)], sections)


def _tube(path, radius: float, n: int = 20):
    """Closed tube of circular section along a polyline (parallel-transport frames)."""
    import trimesh

    P = np.asarray(path, dtype=float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.cross(T[0], [0.0, 1.0, 0.0]) if abs(T[0][1]) < 0.9 else np.cross(T[0], [1.0, 0.0, 0.0])
    N = [ref / np.linalg.norm(ref)]
    for i in range(1, len(P)):
        v = N[-1] - (N[-1] @ T[i]) * T[i]
        N.append(v / np.linalg.norm(v))
    N = np.array(N)
    B = np.cross(T, N)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = P[:, None, :] + radius * (np.cos(ang)[None, :, None] * N[:, None, :] + np.sin(ang)[None, :, None] * B[:, None, :])
    V = np.concatenate([rings.reshape(-1, 3), P[[0, -1]]])
    F = []
    M = len(P)
    for i in range(M - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = a + n, b + n
            F += [[a, b, d], [a, d, c]]
    c0, c1 = M * n, M * n + 1
    for j in range(n):
        F.append([c0, (j + 1) % n, j])
        F.append([c1, (M - 1) * n + j, (M - 1) * n + (j + 1) % n])
    m = trimesh.Trimesh(V, np.array(F), process=True)
    if m.volume < 0:
        m.invert()
    return m


def mug_parts(dims: dict, sections: int = 96) -> dict:
    """{'body': revolved cup, 'handle': swept tube}, each watertight, mug frame."""
    Ro, H, w, F = dims["outer_diameter"] / 2, dims["height"], dims["wall"], dims["floor"]
    Ri = Ro - w
    body = _revolve([(0, 0), (Ro, 0), (Ro, H), (Ri, H), (Ri, F), (0, F), (0, 0)], sections)
    h = dims["handle"]
    rt = h["thickness"] / 2
    x_in = Ro - 0.001  # ends buried 1 mm in the wall; with rt < wall + Ri-clearance never in the cavity
    a = Ro + h["protrusion"] - rt - x_in
    b = (h["z_top"] - h["z_bottom"]) / 2
    zc = (h["z_top"] + h["z_bottom"]) / 2
    phi = np.linspace(-np.pi / 2, np.pi / 2, 41)
    path = np.stack([x_in + a * np.cos(phi), np.zeros_like(phi), zc + b * np.sin(phi)], 1)
    return {"body": body, "handle": _tube(path, rt)}


def mug_mesh(dims: dict, sections: int = 96):
    """Body + handle concatenated into one mesh (two closed overlapping parts)."""
    import trimesh

    p = mug_parts(dims, sections)
    return trimesh.util.concatenate([p["body"], p["handle"]])


def export_meshes(scene, out: Path | None = None, force: bool = False) -> dict:
    """Write cap/mug OBJ + STL (and mug_body / mug_handle) to out_dir(ds)/meshes/.
    Returns {name: {"obj": path, "stl": path, "hash": ..., "watertight": bool}}."""
    out = Path(out or paths.out_dir(scene.ds, "meshes", create=True))
    out.mkdir(parents=True, exist_ok=True)
    cap_d, mug_d = scene.cap_dims(), scene.mug_dims()
    hc, hm = dims_hash(cap_d), dims_hash(mug_d)
    parts = mug_parts(mug_d)
    todo = {f"cap_{hc}": lambda: cap_mesh(cap_d), f"mug_{hm}": lambda: mug_mesh(mug_d),
            f"mug_body_{hm}": lambda: parts["body"], f"mug_handle_{hm}": lambda: parts["handle"]}
    res = {}
    for name, make in todo.items():
        obj, stl = out / f"{name}.obj", out / f"{name}.stl"
        mesh = make()
        if force or not (obj.exists() and stl.exists()):
            mesh.export(obj)
            mesh.export(stl)
        key = name.rsplit("_", 1)[0]
        res[key] = {"obj": obj, "stl": stl, "hash": name.rsplit("_", 1)[1],
                    "watertight": bool(mesh.is_watertight), "volume_m3": float(mesh.volume)}
    return res


# --- poses ---------------------------------------------------------------------------

def cap_pose(scene, episode: int, cap_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(pos, quat wxyz) of a cap's ORIGIN at reset, resting on the table."""
    ep = scene.episode(episode)
    c = next(c for c in ep["caps"] if c["id"] == cap_id)
    yaw = axis_angle_quat([0, 0, 1], float(c.get("yaw", 0.0)))
    if c["up"] == "closed":
        return np.array([*c["xy"], scene.table_z()]), yaw
    return np.array([*c["xy"], scene.table_z() + scene.cap_dims()["height"]]), quat_mul(yaw, _FLIP_X)


def mug_pose(scene, episode: int) -> tuple[np.ndarray, np.ndarray]:
    ep = scene.episode(episode)
    m = ep["mug"]
    return np.array([*m["xy"], scene.table_z()]), axis_angle_quat([0, 0, 1], np.radians(m["handle_yaw_deg"]))


def episode_objects(scene, episode: int) -> list[dict]:
    """Every object to place at reset: [{name, kind, pos, quat}], names 'cap_<id>' and
    'mug' -- the object names every pose log uses."""
    ep = scene.episode(episode)
    out = [{"name": f"cap_{c['id']}", "kind": "cap", "up": c["up"], **dict(zip(("pos", "quat"), cap_pose(scene, episode, c["id"])))}
           for c in ep["caps"]]
    if "mug" in ep:
        out.append({"name": "mug", "kind": "mug", **dict(zip(("pos", "quat"), mug_pose(scene, episode)))})
    return out


def cap_centre(pos, quat, dims: dict) -> np.ndarray:
    """Centre of the cap's bounding cylinder from its origin pose(s): origin + R (0, 0, h/2)."""
    R = quat_to_mat(quat)
    return np.asarray(pos, dtype=float) + R[..., :, 2] * dims["height"] / 2


if __name__ == "__main__":
    import sys

    from .scene import load

    for k, v in export_meshes(load(), force="--force" in sys.argv).items():
        print(f"{k:12s} {v['obj']}  watertight={v['watertight']}  {v['volume_m3'] * 1e6:.2f} cm3")
