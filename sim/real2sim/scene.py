"""The scene description of one dataset: three JSON layers merged, validated, hashed.

    scene = real2sim.scene.load()            # or load("<dataset name>")
    scene.camera("front")  -> camera.Camera
    scene.table_z(), scene.episode(1), scene.cap_dims(), scene.mug_dims(),
    scene.actuator(), scene.units(), scene.used_episodes(), scene.hash

LAYERS, merged in order (a later file overrides a key of an earlier one; dicts merge
recursively, anything else -- numbers, strings, LISTS -- is replaced whole):

    config/<ds>.json         provisional values from the understand phase   CORE
    config/<ds>.calib.json   dataset-fitted cameras, table, offsets, poses   CALIB
    config/<ds>.servo.json   identified actuator model                       MUJOCO

Only the first must exist. Keys starting with "_" are comments and are ignored.
`scene.hash` is a sha256 over the canonical merged JSON (comments excluded): write it
into every output (poselog meta) so a result can be traced to the exact scene.

PROVENANCE. A GROUP is any dict holding at least one non-dict value, and every group
must carry "source": a string saying where its numbers come from, starting with one
of the labels measured / fitted / estimated / assumed / spec / read / provisional /
conflict (e.g. "fitted: silhouette IoU 0.676 held-out (cameras report §2)"). A group
may carry "sigma" (1-sigma uncertainty, in the unit of its main value) or "range".
A layer overriding a group should replace its "source" too; `info` shows which layer
each group came from.

SCHEMA (real2sim.scene/1). Lengths in m, angles in rad unless the key says _deg.
Frame: the URDF base frame (see real2sim/__init__.py).

    schema       "real2sim.scene/1"
    dataset      "<ds>"
    robot
      joint_offsets   {source, deg: [5]}     q = radians(v + offset), arm joints
      gripper_map     {source, a_deg, b_deg_per_pct, alternatives: {...}}  jaw = a + b pct
      limits          {source, calibration: repo-relative path}  (values: units.limits_rad)
    table        {source, z, range: [lo, hi]}   top surface height in the base frame
    cameras
      <name>     {source, width, height,
                  intrinsics {source, K: [fx, fy, cx, cy]},
                  distortion {source, dist: [k1, k2, p1, p2, k3]},
                  pose {source, parent: "world" | link, T_parent_cam: 4x4 OpenCV axes}}
    objects
      cap        {source, diameter, height, wall, top, mass}
                 origin: centre of the RIM plane, +z towards the closed top
      mug        {source, outer_diameter, height, wall, floor, mass,
                  handle: {source, protrusion, z_bottom, z_top, thickness}}
                 origin: centre of the bottom face, +z up, handle along +x
    actuator     {source, ...}  servo model; provisional here, MUJOCO's servo.json owns it
    episodes
      "<i>"      {source, use: bool, reason, length, outcome: {source, in_mug: [ids]},
                  caps: [{source, id, xy: [x, y], up: "closed" | "open", px, r_px}],
                  mug:  {source, xy: [x, y], handle_yaw_deg, rim_px},
                  picks: [{source, cap, close, lift, release, drop, (first_close, reopen)}]}

FRAME RANGES in picks are INCLUSIVE [first, last] EPISODE-LOCAL frame indices, as in
the episodes report tables (unlike episodes.npz's half-open global [from, to)).
Cap pose: up="closed" rests on its rim, origin z = table; up="open" is flipped
(180 deg about x) and rests on its top, origin z = table + height (objects.cap_pose).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import SCHEMA, paths

LABELS = ("measured", "fitted", "estimated", "assumed", "spec", "read", "provisional", "conflict", "computed")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _strip_comments(x):
    if isinstance(x, dict):
        return {k: _strip_comments(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [_strip_comments(v) for v in x]
    return x


def content_hash(data: dict) -> str:
    blob = json.dumps(_strip_comments(data), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _groups(x, path=""):
    """Yield (path, dict) for every provenance group (dict with a non-dict value)."""
    if isinstance(x, dict):
        vals = {k: v for k, v in x.items() if not k.startswith("_")}
        if any(not isinstance(v, (dict, list)) or (isinstance(v, list) and not all(isinstance(e, dict) for e in v))
               for v in vals.values()):
            yield path, x
        for k, v in vals.items():
            yield from _groups(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            if isinstance(v, dict):
                yield from _groups(v, f"{path}[{i}]")


def _num_list(x, n=None) -> bool:
    return isinstance(x, list) and (n is None or len(x) == n) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in x)


def validate(data: dict) -> list[str]:
    """Hand-written schema check. Returns a list of problems (empty = valid)."""
    err = []
    need = lambda cond, msg: None if cond else err.append(msg)  # noqa: E731
    need(data.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}")
    for key in ("dataset", "robot", "table", "cameras", "objects", "episodes"):
        need(key in data, f"missing top-level '{key}'")
    for path, g in _groups(data):
        if path in ("",):  # the top level holds schema/dataset strings, not parameters
            continue
        src = g.get("source")
        need(isinstance(src, str) and src.split(":")[0].strip().split()[0].lower() in LABELS if src else False,
             f"{path}: 'source' must be a string starting with one of {LABELS}")
    r = data.get("robot", {})
    need(_num_list(r.get("joint_offsets", {}).get("deg"), 5), "robot.joint_offsets.deg must be 5 numbers")
    gm = r.get("gripper_map", {})
    need(all(isinstance(gm.get(k), (int, float)) for k in ("a_deg", "b_deg_per_pct")),
         "robot.gripper_map needs a_deg, b_deg_per_pct")
    need(isinstance(data.get("table", {}).get("z"), (int, float)), "table.z must be a number")
    for name, c in data.get("cameras", {}).items():
        p = f"cameras.{name}"
        need(isinstance(c.get("width"), int) and isinstance(c.get("height"), int), f"{p}: width/height ints")
        need(_num_list(c.get("intrinsics", {}).get("K"), 4), f"{p}.intrinsics.K must be 4 numbers")
        need(_num_list(c.get("distortion", {}).get("dist"), 5), f"{p}.distortion.dist must be 5 numbers")
        pose = c.get("pose", {})
        need(isinstance(pose.get("parent"), str), f"{p}.pose.parent must be a string")
        T = pose.get("T_parent_cam")
        if isinstance(T, list) and len(T) == 4 and all(_num_list(row, 4) for row in T):
            from .transforms import is_rotation
            need(is_rotation([row[:3] for row in T[:3]], 1e-6) and T[3] == [0, 0, 0, 1],
                 f"{p}.pose.T_parent_cam is not a rigid transform")
        else:
            err.append(f"{p}.pose.T_parent_cam must be 4x4")
    o = data.get("objects", {})
    for k in ("diameter", "height", "wall", "top", "mass"):
        need(isinstance(o.get("cap", {}).get(k), (int, float)) and o["cap"][k] > 0, f"objects.cap.{k} must be > 0")
    for k in ("outer_diameter", "height", "wall", "floor", "mass"):
        need(isinstance(o.get("mug", {}).get(k), (int, float)) and o["mug"][k] > 0, f"objects.mug.{k} must be > 0")
    for key, e in data.get("episodes", {}).items():
        p = f"episodes.{key}"
        need(key.isdigit(), f"{p}: episode keys are decimal strings")
        need(isinstance(e.get("use"), bool), f"{p}.use must be true/false")
        ids = [c.get("id") for c in e.get("caps", [])]
        need(len(ids) == len(set(ids)), f"{p}: duplicate cap ids")
        for c in e.get("caps", []):
            need(_num_list(c.get("xy"), 2), f"{p} cap {c.get('id')}: xy must be 2 numbers")
            need(c.get("up") in ("closed", "open"), f"{p} cap {c.get('id')}: up must be 'closed' or 'open'")
        if e.get("caps"):
            need(_num_list(e.get("mug", {}).get("xy"), 2), f"{p}.mug.xy must be 2 numbers")
        n = e.get("length")
        for pk in e.get("picks", []):
            need(pk.get("cap") in ids, f"{p}: pick of unknown cap {pk.get('cap')}")
            for k in ("close", "lift", "release", "drop", "first_close", "reopen"):
                if k in pk:
                    rng = pk[k]
                    ok = _num_list(rng, 2) and rng[0] <= rng[1] and (n is None or 0 <= rng[0] and rng[1] < n)
                    need(ok, f"{p} pick {pk.get('cap')}.{k}: inclusive [first, last] within the episode")
    return err


class Scene(dict):
    """The merged scene (a dict) plus accessors. Build with scene.load()."""

    def __init__(self, data: dict, ds: str, layers: list[Path], origin: dict):
        super().__init__(data)
        self.ds, self.layers, self.origin = ds, layers, origin
        self.hash = content_hash(data)

    def camera(self, name: str):
        from .camera import Camera

        if name not in self["cameras"]:
            raise KeyError(f"no camera {name!r}; have {sorted(self['cameras'])}")
        return Camera.from_config(name, self["cameras"][name])

    def table_z(self) -> float:
        return float(self["table"]["z"])

    def episode(self, i) -> dict:
        return self["episodes"][str(int(i))]

    def used_episodes(self) -> list[int]:
        return sorted(int(k) for k, e in self["episodes"].items() if e["use"])

    def cap_dims(self) -> dict:
        return dict(self["objects"]["cap"])

    def mug_dims(self) -> dict:
        return copy.deepcopy(self["objects"]["mug"])

    def actuator(self) -> dict:
        return copy.deepcopy(self.get("actuator", {}))

    def units(self):
        from .units import Units

        return Units.from_scene(self)

    def provenance(self) -> list[tuple[str, str, str, str]]:
        """[(group path, source, sigma/range, layer file)] for every group."""
        rows = []
        for path, g in _groups(dict(self)):
            if not path:
                continue
            unc = g.get("sigma", g.get("range", ""))
            rows.append((path, g.get("source", ""), json.dumps(unc) if unc != "" else "",
                         self.origin.get(path, self.layers[0].name)))
        return rows


def _origins(layer: dict, name: str, out: dict) -> None:
    for path, _ in _groups(layer):
        if path:
            out[path] = name


def load(ds: str | None = None, strict: bool = True) -> Scene:
    """Merge config/<ds>.json <- .calib.json <- .servo.json and validate."""
    ds = paths.dataset(ds)
    data, layers, origin = {}, [], {}
    for layer in paths.CONFIG_LAYERS:
        p = paths.config_path(ds, layer)
        if not p.exists():
            if not layer:
                raise FileNotFoundError(f"no scene config {p}")
            continue
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: {e}") from None
        _origins(d, p.name, origin)
        data = deep_merge(data, d)
        layers.append(p)
    problems = validate(data)
    if problems and strict:
        raise ValueError(f"scene {ds} is invalid:\n  " + "\n  ".join(problems))
    return Scene(data, ds, layers, origin)


def info(ds: str | None = None) -> str:
    """Human-readable dump for `./robot real2sim core info`."""
    from .units import describe

    s = load(ds, strict=False)
    lines = [f"scene {s.ds}  hash {s.hash}", "layers: " + ", ".join(str(p) for p in s.layers)]
    problems = validate(dict(s))
    lines += [f"INVALID: {p}" for p in problems] or ["schema: valid"]
    lines.append(f"table z = {s.table_z():+.4f} m;  used episodes {s.used_episodes()}")
    for name in s["cameras"]:
        c = s.camera(name)
        lines.append(f"camera {name}: {c.width}x{c.height} K {tuple(round(k, 2) for k in c.K)} "
                     f"dist {tuple(round(k, 4) for k in c.dist)} parent {c.parent} "
                     f"t {tuple(round(float(v), 4) for v in c.T_parent_cam[:3, 3])}")
    lines.append("joint ranges (real = calibration limits through the scene's units):")
    lines.append(describe(s.units()))
    lines.append("provenance (group | source | sigma/range | layer):")
    for path, src, unc, layer in s.provenance():
        lines.append(f"  {path:34s} | {src[:110]:110s} | {unc:14s} | {layer}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    print(info(sys.argv[1] if len(sys.argv) > 1 else None))
