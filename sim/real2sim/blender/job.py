"""The render job: everything Blender needs, resolved on the host into one directory.

    job = build_job(ds, episode=0, source=Source.kinematic(), engine="eevee", frames=None, out=dir)

Blender's Python has numpy but not mujoco, cv2, trimesh or scipy (and it ignores
PYTHONPATH), so every decision is taken here, in host python3, and written down:

    job.json   cameras (to_blender of each oversized pinhole, latency), lights, world,
               table, backdrop, materials, engine settings, the output frames and,
               per camera, which unique pose each output frame is rendered from
    job.npz    mesh vertices / faces / per-face material slots, and the 4x4 world
               matrices of every link and object for every unique render pose
    tex_*.npy  float32 textures already in Blender's pixel order (bottom row first)

bl_render.py (run inside Blender) only reads these. It never looks at the scene
config, the look assets or a pose log itself, so what was rendered is exactly what
the job says, and the job can be inspected, diffed and re-rendered.

SOURCES. A pose log (any engine; real2sim.poselog) supplies the arm links and the
objects frame by frame. `--kinematic` builds one from the recording: the arm by FK of
observation.state (poselog.from_dataset), and the caps and the mug STATIC at their
reset poses (objects.episode_objects). That is not a physical replay -- after a grasp
the real cap leaves the table and the kinematic one does not -- so its mode is
recorded as "kinematic: arm = FK(observation.state), objects static at reset" in the
job, the manifest and the side-by-side caption.

SCENE OF A LOG. A physics run may have changed the scene: `meta.scene_overrides`
({path: {was, now}}, both engines) and MuJoCo ensemble perturbations
(`meta.perturbation.cap_diameter`, `meta.physics.table_z`). They are applied here, so
the rendered cap is the simulated cap and the rendered table is where the simulated
objects rest. `scene_for_log` returns that scene; its hash goes into the job.

CAMERA TIME. CALIB fits a latency per camera (cameras.<cam>.latency.frames: "compare a
sim frame of state time t with the real image of row t + frames"). The image of row k
therefore shows the world at t = k - latency, and each camera is rendered there: link
and object poses are interpolated between log rows (positions linearly, rotations by
normalised quaternion lerp; a frame at 150 deg/s moves 5 deg, where nlerp is exact to
< 0.01 deg). Without a CALIB latency the provisional value is 0 and the job says so.

DEDUPE. Identical render poses (to 1e-7 m / 1e-7 in quaternion components) are
rendered once per camera: the arm rests for 30-90 frames at the start and end of
every episode (episodes report), and a static wrist camera sees the same picture.
The front camera needs every link and object; the wrist camera the same set (it
sees the arm and the scene too), so both use one key.

TABLE. The table texture (look/table/albedo_lin.npy) was rectified through the front
camera onto z_look (look.json textures.table.z). A physics log may rest its objects at
another table height. The plane goes to the render's z, scaled about the front camera's
nadir by (C_z - z_render) / (C_z - z_look): for a horizontal plane that central
projection is exactly a uniform scaling, so the front camera sees the texture exactly
where LOOK measured it (a 9 mm height change would otherwise move texels by 3-4 mm
at the frame edge; cameras report 0.32-0.62 mm per mm).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import objects as OBJ
from .. import paths, poselog
from ..camera import render_spec, to_blender
from ..scene import Scene, content_hash
from ..transforms import make_T, quat_to_mat
from . import JOB_SCHEMA, materials as MAT

LINKS_NEEDED = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")


# --- where the poses come from --------------------------------------------------------

@dataclass(frozen=True)
class Source:
    """kind 'kinematic' (FK of the recorded state, static objects) or 'poselog' (a file)."""

    kind: str
    path: str | None = None

    @classmethod
    def kinematic(cls) -> "Source":
        return cls("kinematic")

    @classmethod
    def log(cls, path) -> "Source":
        return cls("poselog", str(Path(path).resolve()))

    @property
    def label(self) -> str:
        if self.kind == "kinematic":
            return "kinematic: arm = FK(observation.state), objects static at reset (not physical)"
        return f"poselog: {self.path}"

    @property
    def tag(self) -> str:
        """A short name for output directories: 'kinematic' or '<engine>-<log stem>'."""
        if self.kind == "kinematic":
            return "kinematic"
        p = Path(self.path)
        stem = p.stem if p.stem != "poselog" else p.parent.name  # isaac/<run>/poselog.npz
        return f"{_engine_of(p)}-{stem}"


def _engine_of(p: Path) -> str:
    try:
        return str(poselog.read(p)["meta"].get("engine", "log"))
    except Exception:
        return "log"


def set_path(d: dict, path: str, value) -> None:
    """d['a']['b'][...] = value for 'a.b...' (the meta.scene_overrides key format)."""
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def scene_for_log(scene: Scene, meta: dict) -> tuple[Scene, dict]:
    """The scene a physics run actually simulated: meta.scene_overrides (both engines)
    plus MuJoCo's ensemble perturbation of the cap diameter and the table height.
    Returns (scene, {path: value applied})."""
    import copy

    data = copy.deepcopy(dict(scene))
    applied = {}
    for path, v in (meta.get("scene_overrides") or {}).items():
        now = v["now"] if isinstance(v, dict) and "now" in v else v
        set_path(data, path, now)
        applied[path] = now
    pert = meta.get("perturbation") or {}
    if pert.get("cap_diameter"):
        data["objects"]["cap"]["diameter"] = float(pert["cap_diameter"])
        applied["objects.cap.diameter"] = float(pert["cap_diameter"])
    tz = (meta.get("physics") or {}).get("table_z")
    if tz is not None:
        data["table"]["z"] = float(tz)
        applied["table.z"] = float(tz)
    elif pert.get("table_dz"):
        data["table"]["z"] = float(data["table"]["z"]) + float(pert["table_dz"])
        applied["table.z"] = data["table"]["z"]
    return Scene(data, scene.ds, scene.layers, scene.origin), applied


def load_source(scene: Scene, episode: int, source: Source) -> tuple[dict, Scene, dict]:
    """(pose log with links and objects filled, the scene it lives in, overrides applied)."""
    if source.kind == "kinematic":
        log = poselog.from_dataset(scene.ds, episode, units=scene.units())
        objs = OBJ.episode_objects(scene, episode)
        T = len(log["q"])
        log["objects"] = np.array([o["name"] for o in objs], dtype=str)
        log["obj_pos"] = np.repeat(np.array([o["pos"] for o in objs], float)[None], T, 0)
        log["obj_quat"] = np.repeat(np.array([o["quat"] for o in objs], float)[None], T, 0)
        log["meta"]["mode"] = source.label
        return log, scene, {}
    log = poselog.read(source.path)
    if int(log["meta"].get("episode", episode)) != int(episode):
        raise ValueError(f"{source.path} is episode {log['meta'].get('episode')}, not {episode}")
    if not len(log["links"]) or not np.any(log["links_pos"]):
        poselog.fill_links_from_q(log)
    sc, applied = scene_for_log(scene, log["meta"])
    return log, sc, applied


# --- poses at camera time ---------------------------------------------------------------

def _nlerp(q0, q1, w):
    q1 = np.where((np.sum(q0 * q1, -1, keepdims=True) < 0), -q1, q1)
    q = (1 - w) * q0 + w * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def poses_at(log: dict, frames, lag: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Link and object poses at episode time t = frame - lag, interpolated between
    the log rows that bracket it (clamped to the log). Returns
    (links_pos (N, L, 3), links_quat (N, L, 4), obj_pos (N, O, 3), obj_quat (N, O, 4)).
    An object row that is NaN (untracked) stays NaN."""
    fr = np.asarray(log["frame"], float)
    t = np.clip(np.asarray(frames, float) - float(lag), fr[0], fr[-1])
    j = np.clip(np.searchsorted(fr, t, side="right") - 1, 0, len(fr) - 1)
    j1 = np.minimum(j + 1, len(fr) - 1)
    span = np.where(j1 > j, fr[j1] - fr[j], 1.0)
    w = np.clip((t - fr[j]) / span, 0.0, 1.0)
    out = []
    for pk, qk in (("links_pos", "links_quat"), ("obj_pos", "obj_quat")):
        P, Q = np.asarray(log[pk], float), np.asarray(log[qk], float)
        if P.shape[1] == 0:
            out += [P[j], Q[j]]
            continue
        wp = w[:, None, None]
        out.append((1 - wp) * P[j] + wp * P[j1])
        out.append(_nlerp(Q[j], Q[j1], wp))
    return tuple(out)


def to_matrices(pos, quat) -> np.ndarray:
    """(..., 3), (..., 4) -> (..., 4, 4); NaN poses become a far-away parking matrix
    (an untracked object must not appear in the picture)."""
    pos, quat = np.asarray(pos, float), np.asarray(quat, float)
    bad = ~np.isfinite(pos).all(-1) | ~np.isfinite(quat).all(-1)
    pos = np.where(bad[..., None], [0.0, 0.0, -100.0], pos)
    quat = np.where(bad[..., None], [1.0, 0.0, 0.0, 0.0], quat)
    M = np.zeros(pos.shape[:-1] + (4, 4))
    M[..., :3, :3] = quat_to_mat(quat)
    M[..., :3, 3] = pos
    M[..., 3, 3] = 1.0
    return M


def dedupe(*arrays, decimals: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Rows identical across all arrays (after rounding) collapse to one.
    Returns (unique row indices in first-seen order, map row -> position in it)."""
    n = len(arrays[0])
    keys = {}
    first, inv = [], np.empty(n, int)
    for i in range(n):
        h = hashlib.sha1(b"".join(np.round(np.nan_to_num(a[i], nan=9e9), decimals).tobytes() for a in arrays)).digest()
        if h not in keys:
            keys[h] = len(first)
            first.append(i)
        inv[i] = keys[h]
    return np.array(first, int), inv


# --- geometry ------------------------------------------------------------------------------

def _rpy_extrinsic_mat(rpy_deg) -> np.ndarray:
    r, p, y = np.radians(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _box(half) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = half
    V = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def arm_meshes() -> list[dict]:
    """The FK arm as rendered: every visual geom of paths.MJCF in its body frame, plus
    the printed klip_support bracket and the KWC-500 body on the gripper. The bracket
    pose and the body box are the ones LOOK's label renderer uses (look/segment.py,
    READ from mjlab klip_camera.py and sim_agent101 objects.py), so the renderer and
    the label maps agree on what the wrist carries."""
    import trimesh

    from ..kinematics import FIXED_FINGER_MESH, MOVING_JAW_MESH, default
    from ..look import segment as SEG

    out = []
    for i, g in enumerate(default().meshes(visual=True)):
        mesh = g["mesh"]
        part = "fingers" if mesh in (FIXED_FINGER_MESH, MOVING_JAW_MESH) else \
            "servo" if mesh.startswith("sts3215") else "printed"
        out.append({"name": f"arm{i:02d}_{mesh}", "link": g["link"], "part": part,
                    "verts": np.asarray(g["verts"], np.float32), "faces": np.asarray(g["faces"], np.int32)})
    T_mount = make_T(_rpy_extrinsic_mat(SEG.KLIP_MOUNT_RPY_DEG), SEG.KLIP_MOUNT_POS)
    klip = trimesh.load(str(SEG.KLIP_STL), force="mesh", process=True)
    V = np.asarray(klip.vertices) @ T_mount[:3, :3].T + T_mount[:3, 3]
    out.append({"name": "mount_klip_support", "link": "gripper", "part": "mount",
                "verts": V.astype(np.float32), "faces": np.asarray(klip.faces, np.int32)})
    Vb, Fb = _box(SEG.KWC500_BODY_HALF)
    Vb = (Vb + SEG.KWC500_BODY_POS) @ T_mount[:3, :3].T + T_mount[:3, 3]
    out.append({"name": "mount_kwc500_body", "link": "gripper", "part": "mount",
                "verts": Vb.astype(np.float32), "faces": Fb.astype(np.int32)})
    return out


def mug_face_slots(verts, faces, dims) -> np.ndarray:
    """Per-face material slot of the mug body: 1 steel (inner wall, floor, rim top), 0
    red enamel (outer wall, bottom). The real mug shows a polished rim ring and interior
    from above and red enamel outside (front and wrist frames, episodes report)."""
    c = np.asarray(verts)[np.asarray(faces)].mean(1)
    r = np.hypot(c[:, 0], c[:, 1])
    Ro, H, w, F = dims["outer_diameter"] / 2, dims["height"], dims["wall"], dims["floor"]
    inner = (r < Ro - w / 2) & (c[:, 2] > F / 2)
    rim = c[:, 2] > H - 1e-5
    return (inner | rim).astype(np.int32)


def object_meshes(scene: Scene, names) -> list[dict]:
    """Meshes for the log's objects: 'cap_*' -> the core cap (scene dims), 'mug' -> body
    (two slots) + handle. Vertices in the OBJECT frame (objects.py conventions)."""
    cap_d, mug_d = scene.cap_dims(), scene.mug_dims()
    out = []
    cap = OBJ.cap_mesh(cap_d)
    parts = OBJ.mug_parts(mug_d) if "mug" in names else {}
    for o, name in enumerate(names):
        if name.startswith("cap"):
            out.append({"name": name, "object": o, "slots": ["cap_teal"], "verts": np.asarray(cap.vertices, np.float32),
                        "faces": np.asarray(cap.faces, np.int32)})
        elif name == "mug":
            b, h = parts["body"], parts["handle"]
            out.append({"name": "mug_body", "object": o, "slots": ["mug_enamel_red", "mug_steel_interior"],
                        "verts": np.asarray(b.vertices, np.float32), "faces": np.asarray(b.faces, np.int32),
                        "face_slot": mug_face_slots(b.vertices, b.faces, mug_d)})
            out.append({"name": "mug_handle", "object": o, "slots": ["mug_enamel_red"],
                        "verts": np.asarray(h.vertices, np.float32), "faces": np.asarray(h.faces, np.int32)})
        else:
            raise ValueError(f"unknown object {name!r} in the pose log")
    return out


# --- look assets ---------------------------------------------------------------------------

def load_look(ds=None) -> tuple[dict, Path]:
    d = paths.out_dir(ds, "look")
    p = d / "look.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run ./robot real2sim look build")
    return json.loads(p.read_text()), d


def look_staleness(look: dict, scene: Scene) -> str | None:
    """None when look.json was built from this scene's look inputs, else a warning."""
    from ..look.build import inputs_hash

    h = inputs_hash(scene)
    if look.get("inputs_hash") == h:
        return None
    return (f"look assets are STALE (built from inputs {look.get('inputs_hash')}, scene now {h}): "
            "re-run ./robot real2sim look build")


def _blender_rows(img) -> np.ndarray:
    """(H, W, C) image, row 0 = top -> Blender pixel order (row 0 = bottom), RGBA float32."""
    a = np.asarray(img, np.float32)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], 2)
    return np.ascontiguousarray(a[::-1])


def textures(look: dict, look_dir: Path, job_dir: Path) -> dict:
    """Table albedo, backdrop cylinder and environment as float32 npy in Blender order.

    The cylinder is stored with 0 where no wrist frame saw it (look/backdrop.py); those
    texels get the environment's fill radiance, the same uniform 'rest of the room' the
    environment map carries in every direction nobody saw. The environment is mirrored
    left-right: LOOK's column runs with azimuth atan2(y, x) (0 = +x, increasing to +y),
    Blender's equirect lookup u = 0.5 - atan2(y, x) / 2 pi runs the other way.

    REFLECTIONS SEE ONLY THE OBSERVED ROOM. LOOK's dielectric albedos (table texture,
    cap, PLA, enamel, servo, mount) are effective albedos: rho_PLA L / L_PLA with only the
    KEY's specular lobe removed (look/table.py ALBEDO, look/materials.py), so whatever
    the ambient's reflection adds on the real surface is already inside them. Rendered
    with a glossy BSDF that also mirrors the uniform fill, the ambient is counted twice:
    for the mat that is F0 0.04 x fill 0.16 = 0.0064 radiance, +0.015 in front
    camera-linear against a plain-mat total of 0.006-0.008 (measured: +0.016 background
    bias, both cameras, first fitted-scene render). So the fill lights diffusely only:
    glossy rays see env_glossy (the environment where a wrist frame observed it, 0
    elsewhere; tex_env_glossy.npy) and the cylinder's alpha is its seen mask, which
    bl_render.py makes transparent to glossy rays where no frame saw the wall. The one
    material that is not an effective albedo, the steel, gets the unobserved rest
    (tex_env_unseen.npy) explicitly (materials.py ROOM REFLECTION)."""
    import cv2

    tb = look["textures"]["table"]
    alb = np.load(look_dir / "table" / tb["files"]["albedo_lin"]).astype(np.float32)
    np.save(job_dir / "tex_table_albedo.npy", _blender_rows(alb))
    cyl = look["textures"]["backdrop_cylinder"]
    ctex = np.load(look_dir / "backdrop" / cyl["files"]["cylinder_lin"]).astype(np.float32)
    seen = cv2.imread(str(look_dir / "backdrop" / cyl["files"]["cylinder_seen"]), cv2.IMREAD_GRAYSCALE) > 0
    fill = float(look["textures"]["environment"]["fill_radiance"])
    ctex = np.where(seen[..., None], ctex, fill)
    np.save(job_dir / "tex_backdrop.npy", _blender_rows(np.concatenate([ctex, seen[..., None]], 2)))
    env = cv2.cvtColor(cv2.imread(str(look_dir / "backdrop" / "env.hdr"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    env_seen = cv2.imread(str(look_dir / "backdrop" / cyl["files"]["env_seen"]), cv2.IMREAD_GRAYSCALE) > 0
    np.save(job_dir / "tex_env.npy", _blender_rows(env[:, ::-1]))
    np.save(job_dir / "tex_env_glossy.npy", _blender_rows((env * env_seen[..., None])[:, ::-1]))
    np.save(job_dir / "tex_env_unseen.npy", _blender_rows((env * ~env_seen[..., None])[:, ::-1]))
    return {"table_albedo": "tex_table_albedo.npy", "backdrop": "tex_backdrop.npy", "env": "tex_env.npy",
            "env_glossy": "tex_env_glossy.npy", "env_unseen": "tex_env_unseen.npy",
            "backdrop_seen_fraction": round(float(seen.mean()), 4),
            "env_seen_fraction": round(float(env_seen.mean()), 4), "backdrop_fill_radiance": fill,
            "reflections": "observed room only: the fill lights diffusely (job.textures)"}


def table_placement(look: dict, scene: Scene) -> dict:
    """The table quad at the render's z, with LOOK's texture extent scaled about the front
    camera's nadir so that the front camera sees every texel where LOOK measured it."""
    tb = look["textures"]["table"]
    z_look, z = float(tb["z"]), scene.table_z()
    C = scene.camera("front").T_world_cam()[:3, 3]
    s = (C[2] - z) / (C[2] - z_look)
    x0, x1 = C[0] + s * (tb["x0"] - C[0]), C[0] + s * (tb["x1"] - C[0])
    y0, y1 = C[1] + s * (tb["y0"] - C[1]), C[1] + s * (tb["y1"] - C[1])
    return {"z": z, "z_look": z_look, "scale_about_front_nadir": round(float(s), 6),
            "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "note": "UV (0,0) at (x0, y0), (1,1) at (x1, y1); texture rows flipped to Blender order"}


def lights(look: dict) -> dict:
    """The key as a Blender sun and the ambient as the world. MEASURED on this Blender
    (build-phase units check, both engines): a sun of strength S on a white Lambertian
    face renders S cos(theta) / pi, and a uniform world of radiance L renders L, i.e.
    Blender's sun strength IS normal irradiance and its world colour IS radiance. So
    look.json's scene units go in unchanged: strength = normal_irradiance_scene_units,
    angle = 2 x angular radius, world = env.hdr at strength 1."""
    k = look["lights"]["key"]
    return {"sun": {"direction_to_light": k["direction_to_light"], "strength": k["normal_irradiance_scene_units"],
                    "angle_deg": 2 * k["angular_radius_deg"],
                    "source": "fitted: LOOK lights.key (mug shadows); Blender units measured"},
            "world": {"strength": 1.0, "source": "fitted: LOOK backdrop/env.hdr, fill = ambient share"},
            "ambient_fraction": look["lights"]["ambient"]["horizontal_irradiance_scene_units"]}


# --- the job -------------------------------------------------------------------------------

@dataclass
class JobSpec:
    ds: str | None
    episode: int
    source: Source
    engine: str = "eevee"
    frames: list | None = None  # episode-local output frames; None = every frame of the log
    cams: tuple = ("front", "grip")
    quality: dict = field(default_factory=dict)


def camera_latency(scene: Scene, cam: str) -> tuple[float, str]:
    lat = scene["cameras"][cam].get("latency")
    if lat and "frames" in lat:
        return float(lat["frames"]), lat.get("source", "calib")
    return 0.0, "provisional: no cameras.<cam>.latency in the scene (CALIB has not published); 0 frames"


def build_job(spec: JobSpec, out: Path, log_fn=print) -> dict:
    """Resolve everything and write job.json + job.npz + textures into `out`."""
    from ..scene import load as load_scene

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_scene(spec.ds)
    log, scene, applied = load_source(base, spec.episode, spec.source)
    look, look_dir = load_look(scene.ds)
    warnings = [w for w in [look_staleness(look, base)] if w]
    if abs(scene.table_z() - float(look["textures"]["table"]["z"])) > 1e-4:
        warnings.append(f"table z {scene.table_z():.4f} differs from the look texture's {look['textures']['table']['z']}: "
                        "the texture is rescaled about the front nadir (job.py TABLE)")
    frames = np.asarray(spec.frames if spec.frames is not None else log["frame"], int)
    links = [str(n) for n in log["links"]]
    missing = [n for n in LINKS_NEEDED if n not in links]
    if missing:
        raise ValueError(f"pose log lacks links {missing}")
    li = [links.index(n) for n in LINKS_NEEDED]
    objects = [str(n) for n in log["objects"]]

    cams, npz = {}, {}
    for cam in spec.cams:
        c = scene.camera(cam)
        rs = render_spec(c)
        lag, lag_src = camera_latency(scene, cam)
        lp, lq, op, oq = poses_at(log, frames, lag)
        lp, lq = lp[:, li], lq[:, li]
        uniq, inv = dedupe(lp, lq, op, oq)
        npz[f"{cam}_links"] = to_matrices(lp[uniq], lq[uniq]).astype(np.float64)
        npz[f"{cam}_objects"] = to_matrices(op[uniq], oq[uniq]).astype(np.float64) if objects else np.zeros((len(uniq), 0, 4, 4))
        b = to_blender(rs.camera(c))
        cams[cam] = {"blender": b, "clip_start": 0.004 if c.parent != "world" else 0.05, "clip_end": 10.0,
                     "pinhole": {"width": rs.width, "height": rs.height, "K": list(rs.K)},
                     "latency_frames": lag, "latency_source": lag_src,
                     "renders": int(len(uniq)), "frame_to_render": inv.tolist()}
        log_fn(f"  {cam}: pinhole {rs.width}x{rs.height}, latency {lag:g} frames, {len(frames)} frames -> "
               f"{len(uniq)} unique renders")

    meshes = arm_meshes()
    for i, m in enumerate(meshes):
        npz[f"arm{i}_v"], npz[f"arm{i}_f"] = m["verts"], m["faces"]
    omeshes = object_meshes(scene, objects)
    for i, m in enumerate(omeshes):
        npz[f"obj{i}_v"], npz[f"obj{i}_f"] = m["verts"], m["faces"]
        if "face_slot" in m:
            npz[f"obj{i}_slot"] = m["face_slot"]
    np.savez(out / "job.npz", **npz)
    tex = textures(look, look_dir, out)
    mats = MAT.materials(look)
    job = {
        "schema": JOB_SCHEMA, "dataset": scene.ds, "episode": int(spec.episode),
        "source": {"kind": spec.source.kind, "path": spec.source.path, "label": spec.source.label,
                   "log_meta": {k: log["meta"].get(k) for k in ("engine", "mode", "seed", "scene_hash", "git")}},
        "scene_hash": scene.hash, "scene_hash_base": base.hash, "scene_overrides_applied": applied,
        "look": {"dir": str(look_dir), "inputs_hash": look.get("inputs_hash"), "built": look.get("built")},
        "engine": spec.engine, "quality": spec.quality, "frames": frames.tolist(),
        "links": list(LINKS_NEEDED), "objects": objects, "cameras": cams,
        "arm": [{"name": m["name"], "link": m["link"], "material": MAT.PART_MATERIAL[m["part"]], "part": m["part"],
                 "key": f"arm{i}"} for i, m in enumerate(meshes)],
        "object_meshes": [{"name": m["name"], "object": m["object"], "slots": m["slots"], "key": f"obj{i}",
                           "has_slots": "face_slot" in m} for i, m in enumerate(omeshes)],
        "materials": mats, "lights": lights(look), "textures": tex,
        "table": table_placement(look, scene),
        "backdrop": {k: look["textures"]["backdrop_cylinder"][k] for k in ("centre_xy", "radius", "height")}
        | {"z0": scene.table_z()},
        "warnings": warnings,
    }
    job["job_hash"] = content_hash({k: v for k, v in job.items() if k not in ("warnings",)})
    (out / "job.json").write_text(json.dumps(job, indent=1, default=_json_default))
    for w in warnings:
        log_fn(f"  WARNING: {w}")
    return job


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def sample_frames(ds, episode: int) -> list[dict]:
    """LOOK's sample set rows for one episode: [{id, frame, phase, ...}] (samples/index.json)."""
    from ..look.imagemetrics import load_index

    index, _ = load_index(ds)
    return [s for s in index["samples"] if int(s["episode"]) == int(episode)]
