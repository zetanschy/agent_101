"""Finger collision geometry that follows the real inner faces: CoACD of the finger parts.

WHY. MuJoCo collides a mesh geom as its convex hull, and mjlab's SO-101 gives each
finger part one mesh geom. The fixed finger lives in `wrist_roll_follower_so101_v1`,
a part that also spans the whole wrist (x -35..+30 mm, z -104..+1 mm in the gripper
frame), so its hull fills the grasp opening from above: the understand phase measured
hull contact normals with n_z = +0.31..+0.36, the hull riding up over the cap and the
jaw closing past it, 0/45 caps in the mug. The real inner faces are STEPPED planes
(MEASURED, gripper frame, fingers at Jaw = 0):

    fixed finger   x = -7.9 mm for z -104..-94, -9.9 for -94..-78, -11.6 for -78..-50
    moving jaw     x = +7.9 mm for z -105..-95, +9.9 for -95..-85, +11.9 for -85..-45

CoACD (threshold 0.05, seed 0) cuts both parts exactly at those steps (48 convex
pieces for the fixed-finger part, 22 for the jaw; piece volumes sum to 1.09x and 1.08x
the part volumes, against 2.58x and 2.33x for the single hulls), so a cap between the
fingers meets the true flat faces at the true gap. model.py replaces the two hull
geoms by these pieces; every other link keeps its hull.

CoACD is only installed in the astribot_simu conda env ($R2S_COACD_PY), so `build`
runs there once and caches OBJ pieces + a manifest (source STL sha256, parameters);
`load` (any env) reads the cache and refuses a stale one.

    ./robot real2sim mujoco fingers          (build; ~60 s)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .. import paths

FINGER_MESHES = ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")  # on `gripper`, on `jaw`
COACD_PARAMS = {"threshold": 0.05, "max_convex_hull": -1, "preprocess_mode": "auto", "resolution": 2000,
                "mcts_nodes": 20, "mcts_iterations": 150, "mcts_max_depth": 3, "seed": 0}
VERSION = 1


def cache_dir(ds: str | None = None) -> Path:
    return paths.out_dir(ds, "mujoco", "coacd")


def _stl(name: str) -> Path:
    return paths.MJCF.parent / "assets" / f"{name}.stl"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _key(name: str) -> dict:
    return {"version": VERSION, "stl_sha256": _sha(_stl(name)), "params": COACD_PARAMS}


def build(ds: str | None = None, force: bool = False) -> dict:
    """Decompose FINGER_MESHES with CoACD into cache_dir(ds). Needs `coacd` + trimesh."""
    import coacd
    import trimesh

    out = cache_dir(ds)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in FINGER_MESHES:
        mf = out / f"{name}.json"
        key = _key(name)
        if not force and mf.exists() and json.loads(mf.read_text()).get("key") == key:
            manifest[name] = json.loads(mf.read_text())
            continue
        m = trimesh.load(_stl(name), force="mesh")
        parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), **COACD_PARAMS)
        files, vols = [], []
        for i, (v, f) in enumerate(parts):
            p = out / f"{name}_{i:02d}.obj"
            t = trimesh.Trimesh(v, f)
            t.export(p)
            files.append(p.name)
            vols.append(float(t.volume))
        info = {"key": key, "parts": files, "part_volumes_m3": vols, "part_volume_sum_m3": float(sum(vols)),
                "mesh_volume_m3": float(m.volume), "hull_volume_m3": float(m.convex_hull.volume)}
        mf.write_text(json.dumps(info, indent=1))
        manifest[name] = info
    return manifest


def load(ds: str | None = None) -> dict:
    """{mesh name: [(V, 3) float vertices of each convex piece, in the STL's own frame]}."""
    out = cache_dir(ds)
    res = {}
    for name in FINGER_MESHES:
        mf = out / f"{name}.json"
        if not mf.exists():
            raise FileNotFoundError(f"{mf} missing: run ./robot real2sim mujoco fingers (CoACD, ~60 s)")
        info = json.loads(mf.read_text())
        if info.get("key") != _key(name):
            raise RuntimeError(f"{mf} is stale (the STL or the CoACD parameters changed): "
                               "rerun ./robot real2sim mujoco fingers --force")
        res[name] = [_read_obj_vertices(out / f) for f in info["parts"]]
    return res


def _read_obj_vertices(p: Path) -> np.ndarray:
    """Vertex positions of an OBJ (the pieces are convex: faces are not needed, MuJoCo
    rebuilds the hull). Plain parser, so any interpreter can read the cache."""
    v = [list(map(float, ln.split()[1:4])) for ln in p.read_text().splitlines() if ln.startswith("v ")]
    return np.asarray(v, dtype=float)


if __name__ == "__main__":
    import sys

    for n, i in build(force="--force" in sys.argv).items():
        print(f"{n:32s} {len(i['parts']):3d} pieces  volume {i['part_volume_sum_m3'] * 1e6:6.2f} cm3 "
              f"(part {i['mesh_volume_m3'] * 1e6:.2f}, hull {i['hull_volume_m3'] * 1e6:.2f})")
