"""LOOK's assets on the Isaac scene (ISAAC part 2): the real frames' materials, lights and
backdrop, rendered by RTX as LINEAR SCENE RADIANCE.

    look = LookAssets(scene, units=calibration(ds)["rt"])   # reads look.json, textures once
    sim = S.make_sim(opts, look); cfg = S.scene_cfg(sc, opts, gains, look); ...
    S.post_spawn(stage, iscene, sc, opts, look)   # materials, table, backdrop, lights
    look.set_mode("rt" | "pt")           # carb render settings
    look.bind_table("albedo" | "grey" | "emit")   # the table quad's material (checks)

Everything the CAMERA does (lens distortion, exposure, white balance, sRGB transfer) is
real2sim.blender.camera_model, applied on the host to the linear pinholes rendered here,
identically for Blender and Isaac: the two renderers are scored through one camera.

SCENE UNITS (look README): the total horizontal irradiance at the table is 1, so a
horizontal white Lambertian face has radiance 1/pi. RTX's light units are its own, so
they are MEASURED (lookrender.py --calibrate, cached per look build in
calibration.json): the table quad made a grey Lambertian (albedo 0.5, no specular lobe),
everything else hidden, lit by the key alone (one Kit process) and by the diffuse dome
alone (another), and then made a unit emitter. The render intensities are 1 / (render
units per scene unit), so the render IS scene radiance; each render re-checks it on the
grey quad after its images (lookrender.py).

ONE LIGHT SETUP PER PROCESS. The lights are authored once, at spawn, at the calibrated
intensities of the process's render mode (the dome's units differ 2.9x between RTX
real-time and path tracing, MEASURED), and only materials change afterwards (the grey
check quad, the calibration emitter). Light edits at runtime were never needed, so they
are not relied on: an early probe that seemed to show RTX ignoring them was dominated by
the renderer's own ambient light (/rtx/sceneDb/ambientLightIntensity 1 in Isaac Lab's
'quality' preset), which set_mode turns off.

DOME ORIENTATION (probe_dome in lookrender.py, MEASURED): RTX puts LOOK's +x at -x and its
+y at +y, i.e. its latlong azimuth runs the other way round (as Blender's does, blender/
job.py). The dome therefore uses the column-flipped map rotated 180 deg about z, which
brings every LOOK azimuth back onto itself. The diffuse dome's inputs:specular 0 is
honoured: a mirror facing its bright wedge rendered 0.0.

WHAT IS IN IT
  key light   a DistantLight towards lights.key.direction_to_light, angle = 2 x the fitted
              angular radius (a large soft source), white (the cameras' balance is theirs)
  ambient     one DomeLight on LOOK's env.hdr, which carries the ambient share as its fill
              (the key is not in it), diffuse only (inputs:specular 0). RTX real-time lights
              with ONE dome: a second, specular-only dome halved the grey check quad (0.0798
              against 0.159, MEASURED). Specular off is Blender's convention (blender/job.py
              textures): LOOK's dielectric albedos are EFFECTIVE ones that already hold the
              ambient's reflection, so mirroring the uniform fill as well counts it twice
              (+0.016 camera-linear on the mat, Blender's first fitted-scene render). What
              reflections do see is the observed room: the backdrop's seen part (below).
              The steel is the one true specular: its floor and rim emit the room they
              mirror (_mug), as Blender's steel does.
  table       a quad over LOOK's texture extent at the table height with its de-lit albedo
              (linear EXR) and the glare fit's GGX roughness, F0 0.04; the physics box and
              the ground plane stay, invisible
  backdrop    LOOK's proxy cylinder (R 0.8 m, 0.6 m tall) as an unlit emitter of its scene-
              radiance texture, twice: the wall where a wrist frame saw it (cutout
              elsewhere) for every ray, and 1.6 mm behind it the whole wall with the fill
              where unseen, for the cameras only. Neither casts a shadow
  materials   look.json's: white PLA printed parts, black servos, black mount and webcam,
              teal cap, red enamel mug outside, stainless inside and on the rim (face
              subsets of the mug's visual mesh). UsdPreviewSurface, and the robot's own
              OmniPBR materials edited in place (_robot); perceptual roughness (GGX alpha =
              r^2, as look.json's alpha_ggx / roughness pairs are)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .. import paths

try:  # Kit: the scene hooks. Host python has no isaaclab, but the texture helpers still work.
    from . import scene as S

    _LookBase = S.Look
except ImportError:
    S, _LookBase = None, object

TEX_VERSION = 1  # bump when prepare_textures changes what it writes
TEST_ALBEDO = 0.5  # calibration plane

# Robot materials by the name of the material the workshop USD binds (MEASURED on the
# first look render: run 'look_probe' in isaac/README). Anything else keeps its own.
ROBOT_MATERIALS = {"material_a_3d_printed": "white_pla"}
DARK_ROBOT_MATERIAL = "servo_black"  # a robot material whose own colour is dark (< 0.2)


def _look_dir(ds) -> Path:
    return paths.out_dir(ds, "look")


def _write_float_image(path: Path, rgb: np.ndarray) -> None:
    """RGB(A) float image -> .exr (half) or .hdr (RGBE, RGB only), by extension, via cv2."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    a = np.asarray(rgb, np.float32)
    bgr = np.ascontiguousarray(np.concatenate([a[..., 2::-1], a[..., 3:]], -1))  # RGB(A) -> BGR(A)
    if path.suffix == ".exr":
        ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    else:
        ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"could not write {path}")


def prepare_textures(look: dict, look_dir: Path, out: Path, force: bool = False) -> dict:
    """LOOK's linear arrays as files RTX reads (host numpy + cv2), once per look build:
    table albedo (EXR), backdrop cylinder radiance with the fill where unseen (EXR, the
    blender job's rule), and env.hdr split into the full map and its observed part."""
    import cv2

    key = f"v{TEX_VERSION}_{look['inputs_hash']}_{look['built'].replace(':', '')}"
    out = Path(out) / key
    files = {"table_albedo": out / "table_albedo.exr", "cylinder": out / "cylinder.exr",
             "cylinder_seen": out / "cylinder_seen.exr",
             "env": out / "env.hdr", "env_glossy": out / "env_glossy.hdr",
             "env_mirrored": out / "env_mirrored.hdr", "env_glossy_mirrored": out / "env_glossy_mirrored.hdr"}
    if force or not all(p.exists() for p in files.values()):
        out.mkdir(parents=True, exist_ok=True)
        tb = look["textures"]["table"]
        _write_float_image(files["table_albedo"], np.load(look_dir / "table" / tb["files"]["albedo_lin"]))
        cyl = look["textures"]["backdrop_cylinder"]
        ctex = np.load(look_dir / "backdrop" / cyl["files"]["cylinder_lin"]).astype(np.float32)
        seen = cv2.imread(str(look_dir / "backdrop" / cyl["files"]["cylinder_seen"]), cv2.IMREAD_GRAYSCALE) > 0
        fill = float(look["textures"]["environment"]["fill_radiance"])
        _write_float_image(files["cylinder"], np.where(seen[..., None], ctex, fill))
        _write_float_image(files["cylinder_seen"], np.concatenate([ctex * seen[..., None], seen[..., None]], 2))
        env = cv2.cvtColor(cv2.imread(str(look_dir / "backdrop" / "env.hdr"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        env_seen = cv2.imread(str(look_dir / "backdrop" / cyl["files"]["env_seen"]), cv2.IMREAD_GRAYSCALE) > 0
        _write_float_image(files["env"], env)
        _write_float_image(files["env_glossy"], env * env_seen[..., None])
        # RTX's latlong runs azimuth the other way round (probe_dome, MEASURED): columns flipped
        _write_float_image(files["env_mirrored"], env[:, ::-1])
        _write_float_image(files["env_glossy_mirrored"], (env * env_seen[..., None])[:, ::-1])
    return {k: str(v) for k, v in files.items()}


def env_horizontal_irradiance(env_path: str) -> float:
    """E on an up-facing plane from an equirect map, row 0 = +z (LOOK's layout): the
    cosine-weighted integral over the upper hemisphere, luminance (Rec. 709)."""
    import cv2

    env = cv2.cvtColor(cv2.imread(env_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float64)
    H, W = env.shape[:2]
    theta = (np.arange(H) + 0.5) / H * np.pi  # polar angle from +z
    w = np.cos(theta) * np.sin(theta) * (np.pi / H) * (2 * np.pi / W)
    up = theta < np.pi / 2
    lum = env @ np.array([0.2126, 0.7152, 0.0722])
    return float((lum[up] * w[up, None]).sum())


class LookAssets(_LookBase):
    """scene.Look with LOOK's assets (module doc).

    units: {'key', 'dome', 'emit'} RTX render units per scene unit (calibration()); None =
    unit intensities (a calibration setup). setup: what is lit and shown at spawn:
      'render'  the scene as LOOK measured it
      'key'     calibration: the key alone on the grey table quad, nothing else shown
      'dome'    calibration: the diffuse dome alone on the grey table quad
      'probe'   the dome orientation probe: a synthetic env (lookrender.probe_env) alone"""

    SETUPS = ("render", "key", "dome", "probe")

    def __init__(self, sc, units: dict | None = None, setup: str = "render", probe_env: str | None = None,
                 dome_rotation_deg: float | None = None, steel_fill: bool = True):
        if setup not in self.SETUPS:
            raise ValueError(f"setup {setup!r} not in {self.SETUPS}")
        self.sc, self.setup, self.probe_env, self.steel_fill = sc, setup, probe_env, steel_fill
        self.dir = _look_dir(sc.ds)
        self.look = json.loads((self.dir / "look.json").read_text())
        if self.look.get("scene_hash") != sc.hash:
            print(f"WARNING: look.json was built on scene {self.look.get('scene_hash')}, rendering {sc.hash}", flush=True)
        self.tex = prepare_textures(self.look, self.dir, paths.out_dir(sc.ds, "isaac", "look_tex", create=True))
        self.mat = self.look["materials"]
        self.units = dict(units) if units else {"key": 1.0, "dome": 1.0, "emit": 1.0}
        cal = calibration(sc.ds, self.tex)
        rot = cal.get("dome_rotation_deg", DOME_ROTATION_DEG)
        self.dome_rotation_deg = float(rot["value"] if dome_rotation_deg is None else dome_rotation_deg)
        # the probe renders LOOK's own layout; everything else follows its measured convention
        self.dome_mirrored = bool(cal.get("dome_probe", {}).get("mirrored", False)) and setup != "probe"
        self.info = {"setup": setup, "units": self.units, "dome_rotation_deg": {**rot, "value": self.dome_rotation_deg},
                     "dome_mirrored": self.dome_mirrored}
        self.paths = {}
        self.stage = None
        self.mode = None

    # --- the scene.Look hooks ----------------------------------------------------------
    def object_looks(self) -> dict:
        m = self.mat
        return {"cap": {"color": tuple(m["cap_teal"]["base_color_linear"]), "roughness": m["cap_teal"]["roughness"],
                        "metallic": 0.0},
                "mug": {"color": tuple(m["mug_enamel_red"]["base_color_linear"]),
                        "roughness": m["mug_enamel_red"]["roughness"], "metallic": 0.0}}

    def lights(self) -> dict:
        return {}  # post_spawn authors them (textures, per-light diffuse / specular)

    def render_cfg(self):
        import isaaclab.sim as sim_utils

        return sim_utils.RenderCfg(rendering_mode="quality", enable_translucency=False)

    def post_spawn(self, stage, env_paths: list) -> None:
        from pxr import UsdGeom

        self.stage = stage
        env = env_paths[0]
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/ground")).MakeInvisible()
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{env}/Table")).MakeInvisible()
        self.info["robot_materials"] = self._robot(stage, f"{env}/Robot")
        self.info["mug_subsets"] = self._mug(stage, f"{env}/mug")
        self._table(stage, env)
        self._backdrop(stage, env)
        self._lights(stage)
        self.paths["grey"] = self._preview(stage, "/World/Looks/calib_grey", (TEST_ALBEDO,) * 3, 1.0, 0.0,
                                           no_specular=True)
        self.paths["unit_emitter"] = self._preview(stage, "/World/Looks/calib_emit", (0.0, 0.0, 0.0), 1.0, 0.0,
                                                   emissive=(1.0, 1.0, 1.0), no_specular=True)
        if self.setup != "render":  # calibration / probe: the grey table quad alone
            hidden = [f"{env}/Robot", f"{env}/mug", self.paths["backdrop"], self.paths["backdrop_seen"]]
            hidden += [str(p.GetPath()) for p in stage.GetPrimAtPath(env).GetChildren() if p.GetName().startswith("cap_")]
            for h in hidden:
                UsdGeom.Imageable(stage.GetPrimAtPath(h)).MakeInvisible()
            self.bind_table("grey")

    def post_reset(self, sim) -> None:
        pass

    def bind_table(self, kind: str) -> None:
        """'albedo' (LOOK's texture), 'grey' (the calibration Lambertian) or 'emit' (unit emitter)."""
        from pxr import UsdShade

        prim = self.stage.GetPrimAtPath(self.paths["table"])
        UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
        self._bind(prim, self.paths[{"albedo": "table_material", "grey": "grey", "emit": "unit_emitter"}[kind]])

    # --- materials ---------------------------------------------------------------------
    def _preview(self, stage, path: str, base, rough: float, metallic: float = 0.0, emissive=None,
                 no_specular: bool = False, texture: str | None = None, emissive_texture: str | None = None,
                 cutout: bool = False):
        """A UsdPreviewSurface material; textures are linear ('raw') and read through st.
        cutout: opacity = the emissive texture's alpha, threshold 0.5."""
        from pxr import Sdf, UsdShade

        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        if no_specular:  # a pure diffuse / pure emitter: specular workflow with a black specular colour
            sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        tex_in = {"diffuseColor": texture, "emissiveColor": emissive_texture}
        vals = {"diffuseColor": base, "emissiveColor": emissive}
        reader = None
        for name in ("diffuseColor", "emissiveColor"):
            inp = sh.CreateInput(name, Sdf.ValueTypeNames.Color3f)
            if tex_in[name]:
                if reader is None:
                    reader = UsdShade.Shader.Define(stage, f"{path}/st")
                    reader.CreateIdAttr("UsdPrimvarReader_float2")
                    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                tx = UsdShade.Shader.Define(stage, f"{path}/tex_{name}")
                tx.CreateIdAttr("UsdUVTexture")
                tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_in[name])
                tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
                tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
                tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                inp.ConnectToSource(tx.ConnectableAPI(), "rgb")
                if cutout and name == "emissiveColor":
                    tx.CreateOutput("a", Sdf.ValueTypeNames.Float)
                    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tx.ConnectableAPI(), "a")
                    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
            else:
                inp.Set(tuple(float(c) for c in (vals[name] if vals[name] is not None else (0.0, 0.0, 0.0))))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _bind(self, prim, mat) -> None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

    def _named(self, stage, key: str, emissive=None):
        m = self.mat[key]
        return self._preview(stage, f"/World/Looks/{key}", m["base_color_linear"], m["roughness"],
                             m.get("metallic", 0.0), emissive=emissive)

    def _robot(self, stage, robot: str) -> dict:
        """look.json's materials on the robot, by the material the workshop USD binds
        (ROBOT_MATERIALS; dark unnamed ones are servo black), and mount black on the mount
        and the webcam box. The arm's meshes are INSTANCE PROXIES (all 17 visual meshes,
        MEASURED), which take no binding, so those materials are edited in place, as
        sim_agent101's set_robot_color does. Returns what was done, for the manifest."""
        from pxr import Usd, UsdGeom, UsdShade

        mount = self._named(stage, "mount_black")
        found = {}
        for pr in Usd.PrimRange(stage.GetPrimAtPath(robot), Usd.TraverseInstanceProxies()):
            path = str(pr.GetPath())
            if (not pr.IsA(UsdGeom.Gprim) or "/collisions/" in path
                    or UsdGeom.Imageable(pr).ComputePurpose() == UsdGeom.Tokens.guide):
                continue
            if "/klip_support" in path or "/kwc500_head" in path:
                rec = found.setdefault("payload", {"target": "mount_black", "meshes": 0})
                rec["meshes"] += 1
                self._bind(pr, mount)
                continue
            bound = UsdShade.MaterialBindingAPI(pr).ComputeBoundMaterial()[0]
            src = str(bound.GetPath()) if bound else "none"
            target = ROBOT_MATERIALS.get(Path(src).name) or (DARK_ROBOT_MATERIAL if _is_dark(bound) else None)
            rec = found.setdefault(src, {"target": target, "meshes": 0})
            rec["meshes"] += 1
            if target and "edited" not in rec:
                rec["edited"] = _edit_material(bound, self.mat[target])
        return found

    def _mug(self, stage, mug: str) -> dict:
        """Stainless inside and on the rim, enamel elsewhere: GeomSubsets of the visual mesh
        by face (the body's faces come first, objects.mug_mesh), classified in the mug frame:
        inner wall = normal towards the axis; floor top and rim = normal up above the base.

        THE ROOM IN THE STEEL. The steel's colour is a spec F0 with nothing of the room in it,
        and the room is not in the scene (the dome is diffuse only, see the module doc), so
        the surfaces that mirror the room above, the floor and the rim, emit what they would
        reflect: ROOM_SCALE x F0 x the fill (the unobserved room is the uniform fill). The
        walls mirror the mug itself and are traced. This is the Blender renderer's rule
        (bl_render._add_room_reflection: emission where the mirror ray leaves upwards,
        ROOM_SCALE imported from blender/materials.py), by face instead of by ray. MEASURED
        here: a constant emission on every steel face made the interior 3-4x too bright (the
        cavity reflects it into itself) and lit the mat around the mug (wrist background
        19.3 dB against 22.3 dB without it); no emission at all rendered the interior black
        (front mug dE 31)."""
        from pxr import UsdGeom, UsdShade, Vt

        from .. import objects
        from ..blender.materials import ROOM_SCALE

        vis = UsdGeom.Mesh(stage.GetPrimAtPath(f"{mug}/visual"))
        P = np.array(vis.GetPointsAttr().Get(), dtype=float)
        F = np.array(vis.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3)
        n_body = len(objects.mug_parts(self.sc.mug_dims())["body"].faces)
        c = P[F].mean(1)
        n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        r_hat = c[:, :2] / (np.linalg.norm(c[:, :2], axis=1, keepdims=True) + 1e-12)
        radial = (n[:, :2] * r_hat).sum(1)
        body = np.arange(len(F)) < n_body
        up = body & (n[:, 2] > 0.9) & (c[:, 2] > 0.5 * self.sc.mug_dims()["floor"])  # floor top and rim
        wall = body & (radial < -0.5) & ~up
        m = self.mat["mug_steel_interior"]
        fill = float(self.look["textures"]["environment"]["fill_radiance"]) if self.steel_fill else 0.0
        room = tuple(ROOM_SCALE * fill * np.asarray(m["base_color_linear"]) / self.units["emit"])
        mats = {"steel_up": self._preview(stage, "/World/Looks/mug_steel_up", m["base_color_linear"], m["roughness"], 1.0,
                                          emissive=room),
                "steel_wall": self._preview(stage, "/World/Looks/mug_steel_wall", m["base_color_linear"], m["roughness"],
                                            1.0),
                "enamel": self._named(stage, "mug_enamel_red")}
        for name, sel in (("steel_up", up), ("steel_wall", wall), ("enamel", ~(up | wall))):
            sub = UsdGeom.Subset.CreateGeomSubset(vis, name, UsdGeom.Tokens.face, Vt.IntArray(np.nonzero(sel)[0].tolist()),
                                                  UsdShade.Tokens.materialBind)
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mats[name])
        return {"faces": int(len(F)), "steel_up": int(up.sum()), "steel_wall": int(wall.sum()),
                "enamel": int((~(up | wall)).sum()), "room_emission_scene_units": [round(v * self.units["emit"], 5) for v in room],
                "room_scale": ROOM_SCALE}

    def _table(self, stage, env: str) -> None:
        """LOOK's texture on a quad at the table height (no tilt in this scene; the blender
        job's nadir scaling is 1 when the table is at LOOK's z, as here)."""
        from pxr import Gf, Sdf, UsdGeom, Vt

        tb = self.look["textures"]["table"]
        z = self.sc.table_z()
        if abs(z - float(tb["z"])) > 1e-6 or self.sc["table"].get("tilt"):
            raise NotImplementedError(f"table at {z} (tilt {self.sc['table'].get('tilt')}) vs LOOK's texture at {tb['z']}")
        x0, x1, y0, y1 = (float(tb[k]) for k in ("x0", "x1", "y0", "y1"))
        mesh = UsdGeom.Mesh.Define(stage, f"{env}/LookTable")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z), Gf.Vec3f(x1, y1, z),
                                             Gf.Vec3f(x0, y1, z)]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y1, z)]))
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
        m = self.mat["mat_black_glossy"]
        self.paths["table"] = f"{env}/LookTable"
        self.paths["table_material"] = self._preview(stage, "/World/Looks/table", (1.0, 1.0, 1.0), m["roughness"], 0.0,
                                                     texture=self.tex["table_albedo"])
        self._bind(mesh.GetPrim(), self.paths["table_material"])

    def _backdrop(self, stage, env: str, segments: int = 256) -> None:
        """LOOK's proxy cylinder, twice: the wall where a wrist frame saw it (cutout
        elsewhere) for every ray, and just behind it the whole wall with the fill where
        unseen for the cameras only. Reflections thus see the observed room and nothing
        where nobody looked, as in Blender (job.textures), while a camera sees the fill."""
        cyl = self.look["textures"]["backdrop_cylinder"]
        for name, tex, radius, primary_only, cutout in (("LookBackdrop", "cylinder", 1.002, True, False),
                                                         ("LookBackdropSeen", "cylinder_seen", 1.0, False, True)):
            self._cylinder(stage, f"{env}/{name}", cyl, float(cyl["radius"]) * radius, segments, tex, primary_only,
                           cutout)
        self.paths["backdrop"] = f"{env}/LookBackdrop"
        self.paths["backdrop_seen"] = f"{env}/LookBackdropSeen"

    def _cylinder(self, stage, path: str, cyl: dict, R: float, segments: int, tex: str, primary_only: bool,
                  cutout: bool) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        cx, cy = map(float, cyl["centre_xy"])
        z0, h = float(cyl["z0"]), float(cyl["height"])
        phi = np.linspace(-np.pi, np.pi, segments + 1)  # texture column 0 is phi = -pi
        pts, uv = [], []
        for zz, v in ((z0, 0.0), (z0 + h, 1.0)):  # row 0 is the top: v = 1 there
            for k, p in enumerate(phi):
                pts.append(Gf.Vec3f(cx + R * math.cos(p), cy + R * math.sin(p), zz))
                uv.append(Gf.Vec2f(k / segments, v))
        n = segments + 1
        idx = []
        for k in range(segments):  # wound to face the axis: the cameras are inside
            idx += [k, n + k, n + k + 1, k + 1]
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * segments))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        pv = UsdGeom.PrimvarsAPI(mesh)
        pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray(uv))
        pv.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        if primary_only:
            pv.CreatePrimvar("invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
        mpath = f"/World/Looks/{Path(path).name}"
        mat = self._preview(stage, mpath, (0.0, 0.0, 0.0), 1.0, 0.0, no_specular=True, emissive_texture=self.tex[tex],
                            cutout=cutout)
        tx = UsdShade.Shader(stage.GetPrimAtPath(f"{mpath}/tex_emissiveColor"))
        tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((1.0 / self.units["emit"],) * 3 + (1.0,))  # radiance
        self._bind(mesh.GetPrim(), mat)

    # --- lights --------------------------------------------------------------------------
    def _lights(self, stage) -> None:
        """The key and the two domes at their calibrated intensities; a calibration setup
        authors only the light it measures, at unit intensity (module doc)."""
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        from ..transforms import axis_angle_quat

        want = {"render": ("key", "dome_diffuse"), "key": ("key",), "dome": ("dome_diffuse",),
                "probe": ("dome_diffuse",)}[self.setup]
        self.paths["lights"] = list(want)
        if "key" in want:
            k = self.look["lights"]["key"]
            d = np.asarray(k["direction_to_light"], float)
            d /= np.linalg.norm(d)
            key = UsdLux.DistantLight.Define(stage, "/World/LookKey")
            key.CreateAngleAttr(float(2 * k["angular_radius_deg"]))
            key.CreateIntensityAttr(float(1.0 / self.units["key"]))
            key.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            ax = np.cross([0.0, 0.0, 1.0], d)  # the light shines along its -z: +z goes to the light
            q = axis_angle_quat(ax / np.linalg.norm(ax), math.acos(float(np.clip(d[2], -1, 1))))
            xf = UsdGeom.Xformable(key.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, q)))
        sfx = "_mirrored" if self.dome_mirrored else ""
        for name, tex, diffuse, specular, primary in (("dome_diffuse", self.tex["env" + sfx], 1.0, 0.0, True),
                                                      ("dome_glossy", self.tex["env_glossy" + sfx], 0.0, 1.0, False)):
            if name not in want:
                continue
            dl = UsdLux.DomeLight.Define(stage, f"/World/Look_{name}")
            dl.CreateTextureFileAttr(self.probe_env if self.setup == "probe" else tex)
            dl.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
            dl.CreateIntensityAttr(float(1.0 / self.units["dome"]))
            dl.CreateDiffuseAttr(diffuse)
            dl.CreateSpecularAttr(specular)
            dl.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(primary)
            dxf = UsdGeom.Xformable(dl.GetPrim())
            dxf.ClearXformOpOrder()
            dxf.AddRotateZOp().Set(float(self.dome_rotation_deg if self.setup != "probe" else 0.0))

    # --- calibration ------------------------------------------------------------------------
    def expected(self) -> dict:
        """Scene radiance of the grey table quad under each light alone (look.json units)."""
        k = self.look["lights"]["key"]
        e_key = float(k["normal_irradiance_scene_units"]) * float(np.asarray(k["direction_to_light"])[2])
        e_dome = env_horizontal_irradiance(self.tex["env"])
        return {"key": TEST_ALBEDO * e_key / math.pi, "dome": TEST_ALBEDO * e_dome / math.pi, "emit": 1.0,
                "all": TEST_ALBEDO * (e_key + e_dome) / math.pi, "E_key_horizontal": e_key, "E_dome_horizontal": e_dome}

    # --- render settings ----------------------------------------------------------------------
    def set_mode(self, mode: str, pt_spp: int = 32, pt_total_spp: int = 256, denoise: bool = True) -> dict:
        """'rt' RTX real-time or 'pt' path tracing, with the settings that keep the output
        LINEAR and PHYSICAL: no auto exposure, no fake ambient light, no ambient occlusion."""
        import carb

        s = carb.settings.get_settings()
        common = {"/rtx/post/histogram/enabled": False, "/rtx/sceneDb/ambientLightIntensity": 0.0,
                  "/rtx/ambientOcclusion/enabled": False, "/rtx/indirectDiffuse/enabled": True,
                  "/rtx/reflections/enabled": True, "/rtx/translucency/enabled": False,
                  "/rtx/directLighting/sampledLighting/enabled": True, "/rtx/shadows/enabled": True}
        if mode == "rt":
            extra = {"/rtx/rendermode": "RayTracedLighting", "/rtx/post/aa/op": 4}  # DLAA: full-resolution AA
        elif mode == "pt":
            extra = {"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": int(pt_spp),
                     "/rtx/pathtracing/totalSpp": int(pt_total_spp), "/rtx/pathtracing/maxBounces": 6,
                     "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 6,
                     "/rtx/pathtracing/optixDenoiser/enabled": bool(denoise), "/rtx/pathtracing/clampSpp": 0}
        else:
            raise ValueError(mode)
        for k, v in {**common, **extra}.items():
            s.set(k, v)
        self.mode = mode
        return {**common, **extra}


def _edit_material(material, m: dict) -> dict:
    """Set a bound material's base colour, roughness and metallic to look.json's entry `m`,
    whatever its shader: OmniPBR MDL inputs or UsdPreviewSurface inputs. A colour texture
    would override the constant, so it is cleared. Returns {input: value} as set."""
    from pxr import Sdf, UsdShade

    names = {"diffuse_color_constant": tuple(m["base_color_linear"]), "diffuseColor": tuple(m["base_color_linear"]),
             "reflection_roughness_constant": m["roughness"], "roughness": m["roughness"],
             "metallic_constant": m.get("metallic", 0.0), "metallic": m.get("metallic", 0.0)}
    done = {}
    for c in material.GetPrim().GetChildren():
        sh = UsdShade.Shader(c)
        if not sh:
            continue
        for inp in sh.GetInputs():
            n = inp.GetBaseName()
            if n in names:
                v = names[n]
                inp.Set(tuple(float(x) for x in v) if isinstance(v, tuple) else float(v))
                done[n] = v
            elif n == "diffuse_texture" and inp.Get():
                inp.Set(Sdf.AssetPath(""))
                done[n] = "cleared"
        # MDL shaders author only the inputs that differ from their defaults (the workshop's
        # carry diffuse_color_constant alone, MEASURED): create the rest
        if sh.GetImplementationSource() == "sourceAsset":
            for n, typ in (("diffuse_color_constant", Sdf.ValueTypeNames.Color3f),
                           ("reflection_roughness_constant", Sdf.ValueTypeNames.Float),
                           ("metallic_constant", Sdf.ValueTypeNames.Float)):
                if n not in done:
                    v = names[n]
                    sh.CreateInput(n, typ).Set(tuple(float(x) for x in v) if isinstance(v, tuple) else float(v))
                    done[n] = v
    return done


def _is_dark(material) -> bool:
    """A material whose own base colour is dark (the servo plastic), whatever its shader."""
    if not material:
        return False
    from pxr import UsdShade

    for c in material.GetPrim().GetChildren():
        sh = UsdShade.Shader(c)
        if not sh:
            continue
        for name in ("diffuse_color_constant", "diffuseColor", "base_color"):
            inp = sh.GetInput(name)
            if inp and inp.Get() is not None:
                return float(np.mean(inp.Get())) < 0.2
    return False


# the rotation about +z that puts LOOK's equirect azimuth (column = atan2(y - cy, x - cx)
# from -pi) onto RTX's latlong mapping; lookrender --probe-dome measures it into
# calibration.json, which overrides this default
DOME_ROTATION_DEG = {"value": 0.0, "source": "provisional: not measured"}


def calibration_path(ds, tex: dict | None = None) -> Path:
    """calibration.json beside the look build's textures (a new look build re-measures)."""
    if tex is None:
        d = _look_dir(ds)
        look = json.loads((d / "look.json").read_text())
        tex = prepare_textures(look, d, paths.out_dir(ds, "isaac", "look_tex", create=True))
    return Path(tex["env"]).parent / "calibration.json"


def calibration(ds, tex: dict | None = None) -> dict:
    p = calibration_path(ds, tex)
    return json.loads(p.read_text()) if p.exists() else {}


def update_calibration(ds, patch: dict, tex: dict | None = None) -> dict:
    """Deep-merge `patch` into calibration.json (one Kit process measures one light)."""
    p = calibration_path(ds, tex)
    cur = calibration(ds, tex)

    def merge(a, b):
        for k, v in b.items():
            a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return a

    out = merge(cur, patch)
    p.write_text(json.dumps(out, indent=1))
    return out


def units_for(ds, mode: str, tex: dict | None = None) -> dict:
    """{'key', 'dome', 'emit'} for a render mode, or KeyError naming what to run."""
    cal = calibration(ds, tex).get("modes", {}).get(mode, {})
    missing = [k for k in ("key", "dome", "emit") if k not in cal]
    if missing:
        raise KeyError(f"no {missing} calibration for mode {mode!r}: run ./robot real2sim isaac calibrate")
    return {k: float(cal[k]["render_units_per_scene_unit"]) for k in ("key", "dome", "emit")}
