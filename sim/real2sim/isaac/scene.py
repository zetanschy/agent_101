"""The Isaac Lab scene of one real episode, in the URDF base frame, built in ONE place.

    look = Look()                                      # ISAAC part 2 subclasses this
    opts = Options(episode=0, num_envs=1, render=True)
    sim = make_sim(opts, look)                         # SimulationContext (PhysX + RTX settings)
    cfg = scene_cfg(scene, opts, gains, look)          # InteractiveSceneCfg (gains: servo.py)
    iscene = InteractiveScene(cfg)
    post_spawn(sim.stage, iscene, scene, opts, look)   # USD edits before PhysX parses
    sim.reset()
    after_reset(iscene, opts)                          # read-backs, runtime materials

WHAT IS IN IT (per env; world = env origin = URDF base frame):
    Robot     the workshop SO-101 USD (SDF finger collisions) at identity. MEASURED
              (probe2): at identity the PhysX link frames of all 7 bodies equal
              real2sim.kinematics' URDF frames (quaternions equal up to sign); the
              understand-phase "USD = URDF rotated +90 deg" was the push-T placement.
              Servo: servo.py (implicit PhysX drive + saturation + friction feed-
              forward, the MuJoCo model). Joint limits widened IN THE USD to the real
              firmware limits (units.limits_rad), Jaw lower limit = the finger-touch
              angle -11.5 deg (the real shut stop, units.JAW_TOUCH_DEG): authored before
              PhysX parses, so no reset can restore the URDF's, and read back after.
              Self-collision stays off (PhysX filters parent-child links anyway, so the
              fingers could not collide with each other; the touch limit is the stop).
    Payload   the printed klip_support and the KWC-500 body on the gripper link:
              colliders (mount: PhysX convexDecomposition of the repo's baked
              klip_support.usd; body: a box) and their mass folded into the gripper
              link's MassAPI (params.PAYLOAD).
    Table     a static box whose top is scene.table_z(), physics material "mat", and
              a ground plane 0.75 m below it so nothing falls forever. Contacts
              between the table and the fixed robot base are filtered out.
    Objects   caps and the mug: usd_assets (visual mesh + convex pieces + material),
              dynamic, placed once at reset (placement.py), moved only by contact.
    Lights    Look.lights(); Cameras (render=True): TiledCameras from the scene
              config, rendered as the oversized square-pixel pinhole
              camera.render_spec(cam, square=True), later warped into the real
              distorted image (render.py). The principal point goes through
              sim_agent101's camera_cfg aperture offsets (imported, not copied).

LOOK HOOKS (the rendering track's extension points; physics never lives there):
    Look.object_looks()      colours/roughness baked into the cap/mug USD
    Look.table_visual()      the table's visual material
    Look.lights()            {name: AssetBaseCfg}
    Look.render_cfg()        sim_utils.RenderCfg
    Look.post_spawn(stage, env_paths)   bind MDLs, add an HDRI dome, textures ...
    Look.post_reset(sim)     runtime carb settings (e.g. path tracing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg

from .. import camera as cam_mod
from .. import objects, paths, units as U
from ..transforms import mat_to_quat, quat_to_mat
from . import params as P
from . import usd_assets
from .geometry import table_tilt, table_top_at

TABLE_SIZE = (1.4, 1.0, 0.03)  # m; x, y, thickness. Covers every cap/mug xy with margin
TABLE_CENTRE_XY = (0.0, -0.30)  # base -y is forward; the base itself stands on it too
FLOOR_BELOW = 0.75  # m below the table top: the room floor, only to catch drops
GRIP_NEAR = 0.02  # m, near clip past the modelled webcam housing (sim_agent101 camera_cfg note)


@dataclass
class Options:
    episode: int
    num_envs: int = 1
    env_spacing: float = 2.0
    render: bool = False
    cameras: tuple = ("front", "grip")
    placement: str = "config"  # "config" | "grasp" (placement.py)
    kinematic_objects: bool = False  # mode kinematic: objects stay where they are placed
    params: dict = field(default_factory=P.defaults)
    # per-env perturbations (env 0 is always nominal); see replay.draw_perturbations
    perturb: list = field(default_factory=list)

    @property
    def replicate_physics(self) -> bool:
        """Per-env GEOMETRY (table height, cap size) or a per-env TABLE material needs each
        env parsed on its own: the table is a static collider, which the tensor API cannot
        re-material at runtime, so its friction is authored in each env's USD."""
        return not any(p.get("table_dz") or p.get("cap_scale", 1.0) != 1.0 or p.get("cap_zscale", 1.0) != 1.0
                       or p.get("friction_scales", {}).get("mat", 1.0) != 1.0 for p in self.perturb)


class Look:
    """Rendering hooks with plain defaults (see module doc). Part 2 overrides these."""

    def object_looks(self) -> dict:
        return {k: {kk: vv for kk, vv in v.items() if kk != "source"} for k, v in P.LOOK_DEFAULTS.items()}

    def table_visual(self):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.015, 0.015, 0.017), roughness=0.2, metallic=0.0)

    def lights(self) -> dict:
        # key light from the image's top-right (base -x, -y), where the real glare is
        return {
            "dome_light": AssetBaseCfg(prim_path="/World/DomeLight",
                                       spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.85, 0.87, 0.92))),
            "key_light": AssetBaseCfg(prim_path="/World/KeyLight",
                                      spawn=sim_utils.DiskLightCfg(intensity=5000.0, radius=0.15, color=(1.0, 0.96, 0.9)),
                                      init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, -0.45, 0.8))),
        }

    def render_cfg(self):
        return sim_utils.RenderCfg(rendering_mode="quality", enable_translucency=False)

    def post_spawn(self, stage, env_paths: list) -> None:
        """Paint the printed parts white: the workshop USD ships them yellow, the real
        arm is white PLA. The same shader sim_agent101's set_robot_color edits (all printed
        geometry binds Looks/material_a_3d_printed)."""
        for p in env_paths:
            sh = stage.GetPrimAtPath(f"{p}/Robot/Looks/material_a_3d_printed/Shader")
            if sh and sh.GetAttribute("inputs:diffuse_color_constant"):
                sh.GetAttribute("inputs:diffuse_color_constant").Set((0.93, 0.93, 0.95))

    def post_reset(self, sim) -> None:
        pass


# --- the simulation context ---------------------------------------------------------

def _material(m: dict):
    return sim_utils.RigidBodyMaterialCfg(static_friction=m["static_friction"], dynamic_friction=m["dynamic_friction"],
                                          restitution=m["restitution"],
                                          friction_combine_mode=m.get("friction_combine", "average"),
                                          restitution_combine_mode=m.get("restitution_combine", "average"))


def make_sim(opts: Options, look: Look):
    ph = opts.params["physics"]
    m = opts.params["materials"]["robot"]
    cfg = sim_utils.SimulationCfg(
        dt=1.0 / ph["hz"], render_interval=int(ph["hz"] // 30), device="cuda:0",
        physx=sim_utils.PhysxCfg(solver_type=1 if ph["solver"] == "TGS" else 0, enable_ccd=bool(ph["ccd"]),
                                 bounce_threshold_velocity=float(ph["bounce_threshold"]),
                                 enable_enhanced_determinism=bool(ph["enhanced_determinism"])),
        # the default material is what every shape without its own gets: the robot's
        # links (Isaac Lab 2.1 cannot bind one through UsdFileCfg).
        physics_material=_material(m),
        render=look.render_cfg(),
    )
    return sim_utils.SimulationContext(cfg)


# --- entity configs ------------------------------------------------------------------

def joint_limits_rad(scene, margin_deg: float = P.PHYSICS["limit_margin_deg"]["value"]) -> np.ndarray:
    """(6, 2) the joint range authored into the USD: the firmware range +- margin (the
    firmware clamps the GOAL, servo.firmware_limits; the joint itself has no stop there,
    the MuJoCo track's LIMIT_MARGIN_DEG), and the Jaw's lower stop at the finger-touch
    angle, a real mechanical stop (the fingers meet)."""
    lim = U.limits_rad(scene.units()).copy()
    m = np.radians(margin_deg)
    lim[:, 0] -= m
    lim[:, 1] += m
    lim[U.GRIPPER, 0] = np.radians(U.JAW_TOUCH_DEG)
    return lim


def _q0(scene, episode: int, margin_deg: float) -> dict:
    """The recorded first frame, nudged 1e-4 rad inside the limits: the shut jaw reads
    exactly the touch angle, and Isaac Lab rejects a default ON a limit in float32."""
    from .. import episodes

    ep = episodes.load(scene.ds, include_excluded=True)[episode]
    lim = joint_limits_rad(scene, margin_deg)
    q = np.clip(scene.units().to_urdf(ep.state[0]), lim[:, 0] + 1e-4, lim[:, 1] - 1e-4)
    return {n: float(v) for n, v in zip(U.URDF_JOINTS, q)}


def robot_cfg(scene, opts: Options, gains: dict) -> ArticulationCfg:
    """gains: {stiffness, damping, armature, max_force: {URDF joint: value}} from
    servo.PhysxServo.physx_gains (written again at runtime; these only seed PhysX)."""
    from sim_agent101.assets.so101 import SO101_CFG

    ph = opts.params["physics"]
    spawn = SO101_CFG.spawn.replace(
        activate_contact_sensors=True,
        rigid_props=SO101_CFG.spawn.rigid_props.replace(max_depenetration_velocity=ph["max_depenetration_velocity"]),
        articulation_props=SO101_CFG.spawn.articulation_props.replace(
            solver_position_iteration_count=ph["robot_position_iterations"],
            solver_velocity_iteration_count=ph["velocity_iterations"]),
    )
    return SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                                                   joint_pos=_q0(scene, opts.episode, ph["limit_margin_deg"])),
        actuators={"sts3215": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=gains["stiffness"], damping=gains["damping"], armature=gains["armature"],
            friction=0.0, effort_limit_sim=gains["max_force"],
            velocity_limit_sim=10.0)},  # 10 rad/s: the USD's; never binds (real peak 3.0 rad/s)
    )


def table_cfg(scene, opts: Options, look: Look) -> AssetBaseCfg:
    m = opts.params["materials"]["mat"]
    sx, sy = table_tilt(scene)
    # the box's top face passes through (centre xy, table_top_at(centre)) with the slope
    n = np.array([-sx, -sy, 1.0]) / math.sqrt(1 + sx * sx + sy * sy)
    R = np.eye(3)
    if sx or sy:
        ax = np.cross([0.0, 0.0, 1.0], n)
        ang = math.asin(min(1.0, np.linalg.norm(ax)))
        from ..transforms import axis_angle_quat

        R = quat_to_mat(axis_angle_quat(ax / np.linalg.norm(ax), ang))
    top = np.array([*TABLE_CENTRE_XY, table_top_at(scene, *TABLE_CENTRE_XY)])
    centre = top - n * TABLE_SIZE[2] / 2
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE, collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_material(m),
            visual_material=look.table_visual()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in centre),
                                                 rot=tuple(float(v) for v in mat_to_quat(R))),
    )


def object_placements(scene, opts: Options) -> list[dict]:
    """objects.episode_objects(...) with the cap xy replaced per opts.placement, and z
    lifted onto a tilted table (table_tilt) where the config has one."""
    objs = objects.episode_objects(scene, opts.episode)
    for o in objs:
        o["pos"] = np.array(o["pos"], dtype=float)
        o["pos"][2] += table_top_at(scene, o["pos"][0], o["pos"][1]) - scene.table_z()
    if opts.placement == "config":
        return objs
    if opts.placement != "grasp":
        raise ValueError(f"placement must be config|grasp, got {opts.placement!r}")
    from .. import episodes, kinematics
    from .placement import grasp_centre

    kin = kinematics.default()
    ep = episodes.load(scene.ds, include_excluded=True)[opts.episode]
    u, cd = scene.units(), scene.cap_dims()
    picks = {p["cap"]: p for p in scene.episode(opts.episode)["picks"]}
    for o in objs:
        if o["kind"] != "cap":
            continue
        pk = picks[o["name"][4:]]
        f = int(pk["close"][1])  # last frame of the close: the jaw is blocked on the cap
        tz = table_top_at(scene, o["pos"][0], o["pos"][1])
        z_rel = o["pos"][2] - tz  # the cap origin's height above the table (0 closed-up, h open-up)
        gc = grasp_centre(kin, u.to_urdf(ep.state[f]), cd["diameter"], tz, tz + cd["height"])
        o["pos"] = np.array([gc["xy"][0], gc["xy"][1], z_rel + table_top_at(scene, *gc["xy"])])
        o["placement"] = {"frame": f, **gc}
    return objs


def rigid_cfg(name: str, usd: Path, pos, quat, opts: Options) -> RigidObjectCfg:
    ph = opts.params["physics"]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=ph["object_position_iterations"],
                solver_velocity_iteration_count=ph["velocity_iterations"],
                max_depenetration_velocity=ph["max_depenetration_velocity"],
                kinematic_enabled=opts.kinematic_objects),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=ph["contact_offset"],
                                                             rest_offset=ph["rest_offset"]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in pos), rot=tuple(float(v) for v in quat)),
    )


def camera_cfgs(scene, opts: Options) -> dict:
    """{name: TiledCameraCfg} for the scene's cameras, as square-pixel pinholes.

    The principal point goes through sim_agent101's camera_cfg, which centres on W/2
    (pixel EDGES); real2sim pixels centre on (W-1)/2, so cx, cy are handed over +0.5
    (render check: README 'cameras')."""
    from sim_agent101.assets.objects import camera_cfg

    out = {}
    for name in opts.cameras:
        cam = scene.camera(name)
        spec = cam_mod.render_spec(cam, square=True)
        pin = spec.camera(cam)
        T_gl = cam_mod.opencv_to_opengl(pin.T_parent_cam)
        intr = {name: {"width": pin.width, "height": pin.height, "fx": pin.fx, "fy": pin.fy,
                       "cx": pin.cx + 0.5, "cy": pin.cy + 0.5}}
        parent = "{ENV_REGEX_NS}" if cam.parent == "world" else "{ENV_REGEX_NS}/Robot/" + cam.parent
        out[f"cam_{name}"] = camera_cfg(name, f"{parent}/cam_{name}", pos=tuple(T_gl[:3, 3]),
                                        rot_quat=tuple(mat_to_quat(T_gl[:3, :3])), width=pin.width,
                                        height=pin.height, intrinsics=intr,
                                        near_m=GRIP_NEAR if cam.parent != "world" else 0.05)
    return out


def scene_cfg(scene, opts: Options, gains: dict, look: Look | None = None) -> InteractiveSceneCfg:
    """The InteractiveSceneCfg of one episode. Entity order matters (parents first)."""
    look = look or Look()
    cfg = InteractiveSceneCfg(num_envs=opts.num_envs, env_spacing=opts.env_spacing,
                              replicate_physics=opts.replicate_physics)
    cfg.robot = robot_cfg(scene, opts, gains)
    cfg.table = table_cfg(scene, opts, look)
    cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.2, 0.2, 0.2)),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, scene.table_z() - FLOOR_BELOW)))
    mats = opts.params["materials"]
    usd = usd_assets.build(scene, paths.out_dir(scene.ds, "isaac", "usd", create=True),
                           {"cap": mats["cap"], "mug": mats["mug"]}, look.object_looks())
    for o in object_placements(scene, opts):
        setattr(cfg, o["name"], rigid_cfg(o["name"], usd[o["kind"]], o["pos"], o["quat"], opts))
    for k, v in look.lights().items():
        setattr(cfg, k, v)
    if opts.render:
        for k, v in camera_cfgs(scene, opts).items():
            setattr(cfg, k, v)
    return cfg


# --- USD edits between spawning and PhysX parsing -----------------------------------------

def _set_limits_in_usd(stage, robot_path: str, scene, margin_deg: float) -> dict:
    """Author joint_limits_rad on the joint prims (degrees, as USD wants them)."""
    from pxr import UsdPhysics

    lim = np.degrees(joint_limits_rad(scene, margin_deg))
    out = {}
    for i, n in enumerate(U.URDF_JOINTS):
        j = UsdPhysics.RevoluteJoint.Get(stage, f"{robot_path}/joints/{n}")
        if not j:
            raise RuntimeError(f"no joint prim {robot_path}/joints/{n}")
        j.GetLowerLimitAttr().Set(float(lim[i, 0]))
        j.GetUpperLimitAttr().Set(float(lim[i, 1]))
        out[n] = [float(lim[i, 0]), float(lim[i, 1])]
    return out


def _payload(stage, robot_path: str, opts: Options) -> dict:
    """Mount + webcam colliders on the gripper link, their mass in its MassAPI."""
    import trimesh
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    from sim_agent101.assets.objects import KLIP_SUPPORT_CFG, KWC500_BODY_CFG

    from . import geometry as G

    pl = opts.params["payload"]
    g = stage.GetPrimAtPath(f"{robot_path}/gripper")
    mass_api = UsdPhysics.MassAPI(g)
    m0 = float(mass_api.GetMassAttr().Get())
    c0 = np.array(mass_api.GetCenterOfMassAttr().Get(), dtype=float)
    d0 = np.array(mass_api.GetDiagonalInertiaAttr().Get(), dtype=float)
    pa = mass_api.GetPrincipalAxesAttr().Get()  # Gf.Quatf (real, imaginary)
    R0 = quat_to_mat([pa.GetReal(), *pa.GetImaginary()])
    parts = [(m0, c0, R0 @ np.diag(d0) @ R0.T)]

    # the printed mount, from the repo's own converted USD (pose baked into the points)
    klip_usd = Path(KLIP_SUPPORT_CFG.spawn.usd_path)
    if not klip_usd.exists():
        raise FileNotFoundError(f"{klip_usd} missing: run ./robot sim-assets (it bakes the mount pose)")
    mp = stage.DefinePrim(f"{robot_path}/gripper/klip_support", "Xform")
    mp.GetReferences().AddReference(str(klip_usd))
    mesh_prim = None
    for pr in Usd.PrimRange(mp):
        if pr.IsA(UsdGeom.Mesh):
            mesh_prim = pr
            break
    if mesh_prim is None:
        raise RuntimeError(f"no mesh in {klip_usd}")
    UsdPhysics.CollisionAPI.Apply(mesh_prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr("convexDecomposition")
    PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
    pts = np.array(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=float)
    Tm = np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    Tg = np.array(UsdGeom.Xformable(g).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    pts_g = (np.linalg.inv(Tg) @ Tm @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    faces = np.array(UsdGeom.Mesh(mesh_prim).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    tm = trimesh.Trimesh(pts_g, faces, process=True)
    mm = float(pl["mount_mass"])
    if tm.is_volume:  # uniform density: trimesh's inertia is about the centre of mass, density 1
        parts.append((mm, tm.center_mass, tm.moment_inertia * (mm / tm.mass)))
    else:  # not closed: its bounding box
        parts.append((mm, tm.bounds.mean(0), G.box_inertia(mm, tm.extents)))

    # the KWC-500 head: a box where sim_agent101 models it (bore-derived), colliding
    size = tuple(KWC500_BODY_CFG.spawn.size)
    pos = np.array(KWC500_BODY_CFG.init_state.pos, dtype=float)
    rot = np.array(KWC500_BODY_CFG.init_state.rot, dtype=float)
    cube = UsdGeom.Cube.Define(stage, f"{robot_path}/gripper/kwc500_head")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddOrientOp().Set(Gf.Quatf(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([(0.02, 0.02, 0.02)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mw = float(pl["webcam_mass"])
    Rw = quat_to_mat(rot)
    parts.append((mw, pos, Rw @ G.box_inertia(mw, size) @ Rw.T))

    m, c, I = G.combine_mass(parts)
    diag, R = G.principal(I)
    q = mat_to_quat(R)
    mass_api.GetMassAttr().Set(m)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*map(float, c)))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, diag)))
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    return {"gripper_mass_urdf": m0, "gripper_mass_with_payload": m, "com_shift_mm": float(np.linalg.norm(c - c0) * 1e3),
            "mount_watertight": bool(tm.is_volume),
            # the gripper link's inertial as authored (link frame): track.py builds its
            # inverse dynamics on exactly this
            "inertial": {"mass": m, "com": c.tolist(), "diag": diag.tolist(), "quat": q.tolist()}}


def _filter_base_table(stage, robot_path: str, table_path: str) -> None:
    """The fixed base and the static table: their contact carries no information (the
    real base is clamped) and would only cost solver work if the table top were above
    the base's 30.1 mm mesh bottom."""
    from pxr import Sdf, UsdPhysics

    base = stage.GetPrimAtPath(f"{robot_path}/base")
    api = UsdPhysics.FilteredPairsAPI.Apply(base)
    api.CreateFilteredPairsRel().AddTarget(Sdf.Path(f"{table_path}/geometry/mesh"))  # the collider prim


def post_spawn(stage, iscene, scene, opts: Options, look: Look | None = None) -> dict:
    """Edits every env's USD before sim.reset() parses physics. Returns what it did."""
    from pxr import Gf, UsdGeom

    from pxr import PhysxSchema

    look = look or Look()
    env_paths = [f"/World/envs/env_{i}" for i in range(opts.num_envs)]
    names = [o["name"] for o in objects.episode_objects(scene, opts.episode)]
    info = {}
    # Replicated envs INHERIT env_0 (Isaac Lab clone_environments(copy_from_source=False))
    # and PhysX parses env_0 once for all of them: edit env_0 only. Editing the clones as
    # well added the payload twice (the first 8-env run died on the webcam box's
    # existing xformOp, and the gripper MassAPI would have summed the payload again).
    edit = env_paths[:1] if opts.replicate_physics and opts.num_envs > 1 else env_paths
    for i, ep_path in enumerate(edit):
        rp = f"{ep_path}/Robot"
        info["limits_deg"] = _set_limits_in_usd(stage, rp, scene, opts.params["physics"]["limit_margin_deg"])
        info["payload"] = _payload(stage, rp, opts)
        _filter_base_table(stage, rp, f"{ep_path}/Table")
        if opts.params["physics"]["ccd"]:  # sweep CCD on the objects (Isaac Lab 2.1 has no cfg field)
            for n in names:
                PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(f"{ep_path}/{n}")).CreateEnableCCDAttr(True)
        pert = opts.perturb[i] if i < len(opts.perturb) else {}
        if pert.get("table_dz"):
            tp = UsdGeom.Xformable(stage.GetPrimAtPath(f"{ep_path}/Table"))
            op = next(o for o in tp.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            v = op.Get()
            op.Set(Gf.Vec3d(v[0], v[1], v[2] + float(pert["table_dz"])))
        if pert.get("cap_scale", 1.0) != 1.0 or pert.get("cap_zscale", 1.0) != 1.0:
            s, sz = float(pert.get("cap_scale", 1.0)), float(pert.get("cap_zscale", 1.0))
            for o in objects.episode_objects(scene, opts.episode):
                if o["kind"] != "cap":
                    continue
                xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"{ep_path}/{o['name']}"))
                ops = [op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                (ops[0] if ops else xf.AddScaleOp()).Set(Gf.Vec3d(s, s, sz))  # diameter in x-y, height in z
        mat_s = pert.get("friction_scales", {}).get("mat", 1.0)
        if mat_s != 1.0:
            info.setdefault("table_friction", {})[ep_path] = _scale_friction(stage, f"{ep_path}/Table", mat_s)
    look.post_spawn(stage, env_paths)
    return info


def _scale_friction(stage, root: str, s: float) -> list:
    """Scale the static and dynamic friction of every physics material under `root`
    (the table's: {root}/geometry/material, per env when envs are not replicated)."""
    from pxr import Usd, UsdPhysics

    out = []
    for pr in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if pr.HasAPI(UsdPhysics.MaterialAPI):
            m = UsdPhysics.MaterialAPI(pr)
            for a in (m.GetStaticFrictionAttr(), m.GetDynamicFrictionAttr()):
                a.Set(float(a.Get()) * s)
            out.append([str(pr.GetPath()), float(m.GetStaticFrictionAttr().Get()), float(m.GetDynamicFrictionAttr().Get())])
    if not out:
        raise RuntimeError(f"no physics material under {root}")
    return out


def after_reset(iscene, scene, opts: Options) -> dict:
    """Read back what PhysX actually parsed (limits, gains, masses) for the log."""
    robot = iscene["robot"]
    v = robot.root_physx_view
    names = list(robot.joint_names)
    perm = [names.index(n) for n in U.URDF_JOINTS]
    lim = np.degrees(v.get_dof_limits()[0].cpu().numpy())[perm]
    masses = v.get_masses()[0].cpu().numpy()
    bodies = list(robot.body_names)
    return {"limits_deg_physx": lim.round(3).tolist(),
            "stiffness": v.get_dof_stiffnesses()[0].cpu().numpy()[perm].tolist(),
            "damping": v.get_dof_dampings()[0].cpu().numpy()[perm].tolist(),
            "armature": v.get_dof_armatures()[0].cpu().numpy()[perm].tolist(),
            "max_force": v.get_dof_max_forces()[0].cpu().numpy()[perm].tolist(),
            "masses": {b: float(m) for b, m in zip(bodies, masses)}}
