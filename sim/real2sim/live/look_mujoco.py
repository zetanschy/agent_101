"""LOOK's dataset-derived appearance, as far as MuJoCo's rasterizer can carry it.

The live MuJoCo images are rasterized: no reflections and no soft light. So they carry
the parts of look/ that a rasterizer can:
- THE MAT AS THE OVERHEAD CAMERA SAW IT: look/table/radiance, the arm-free plates
  rectified onto the table at 1 mm, fluorescent marks and glare included. It lies on the
  table plane, so the wrist camera sees it with the right parallax instead of a flat colour.
- THE KEY LIGHT fitted to the mug shadows (look.json lights.key: direction and ambient
  share), as a directional light with shadows. There is no headlight, whose brightness
  would change with the viewing camera.

Lighting is scaled so a horizontal surface receives 1.0 in total (ambient +
diffuse x cos(elevation)). The radiance texture then shows at exactly its recorded value
in the overhead view, and the arm and objects get the measured split between key and ambient.

Visual only (group VISUAL); the table the physics uses is untouched. What stays a
rasterizer's: the polished mug interior (no reflections of the caps), the room behind the
arm, and the cameras' own response. Blender or Isaac RTX render those.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _srgb8(lin: np.ndarray) -> np.ndarray:
    x = np.clip(lin, 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.round(s * 255.0).astype(np.uint8)


def table_png(ds: str) -> tuple[Path, dict]:
    """The radiance texture as an 8-bit sRGB PNG MuJoCo can load, cached by look build."""
    from .. import paths

    look_dir = paths.out_dir(ds, "look")
    look = json.loads((look_dir / "look.json").read_text())
    t = look["textures"]["table"]
    src = look_dir / "table" / t["files"]["radiance_lin"]
    out = paths.out_dir(ds, "live", create=True) / f"table_radiance_{look.get('inputs_hash', 'x')}.png"
    if not out.exists():
        import imageio.v3 as iio

        iio.imwrite(out, _srgb8(np.load(src)))
    return out, {"t": t, "light": look["lights"]["key"]}


def hook(ds: str):
    """spec_hook for mujoco.model.build: the mat texture and the fitted key light."""
    png, info = table_png(ds)

    def apply(spec, scene) -> None:
        import mujoco

        from ..mujoco import model as mdl

        t, key = info["t"], info["light"]
        spec.add_texture(name="look_table", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
        mat = spec.add_material(name="look_table")
        mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "look_table"
        mat.texuniform, mat.texrepeat = False, [1.0, 1.0]
        mat.specular, mat.shininess, mat.reflectance = 0.0, 0.0, 0.0
        # the texture's metric extent, a hair above the collision table (visual only)
        cx, cy = (t["x0"] + t["x1"]) / 2, (t["y0"] + t["y1"]) / 2
        hx, hy = (t["x1"] - t["x0"]) / 2, (t["y1"] - t["y0"]) / 2
        z = scene.table_z() + 0.0003
        spec.worldbody.add_geom(name="look_table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.0002],
                                pos=[cx, cy, z - 0.0002], material="look_table", rgba=[1, 1, 1, 1],
                                group=mdl.VISUAL_GROUP, contype=0, conaffinity=0)
        for g in spec.geoms:  # the flat-colour mat and desk the texture replaces
            if g.name in ("mat_visual", "desk_visual"):
                g.rgba = [0, 0, 0, 0]
        d_to_light = np.asarray(key["direction_to_light"], dtype=float)
        cos_el = float(d_to_light[2] / np.linalg.norm(d_to_light))
        ratio = float(key.get("ambient_over_key_horizontal", 1.0))  # ambient / (key x cos)
        kd = 1.0 / (cos_el * (1.0 + ratio))  # so ambient + kd cos = 1 on a horizontal face
        ka = ratio * kd * cos_el
        for li in list(spec.lights):
            if li.name == "key":
                li.dir = (-d_to_light).tolist()
                li.pos = (d_to_light * 1.5).tolist()
                li.diffuse, li.ambient, li.specular = [kd] * 3, [0.0] * 3, [0.15] * 3
                li.castshadow = True
                # the fitted source is a 12 deg disk: a bulb this size at 1.5 m softens the edges
                li.bulbradius = float(1.5 * np.tan(np.deg2rad(key.get("angular_radius_deg", 12.0))))
                if hasattr(mujoco, "mjtLightType"):
                    li.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        # the ambient share goes on the headlight's AMBIENT term, which does not depend on
        # the viewing camera (MEASURED: a per-light ambient left shadowed faces black here);
        # its directional part stays off
        spec.visual.headlight.ambient = [ka] * 3
        spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
        spec.visual.headlight.specular = [0.0, 0.0, 0.0]
        spec.visual.quality.shadowsize = 8192

    return apply
