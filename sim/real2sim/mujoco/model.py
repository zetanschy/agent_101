"""The MuJoCo scene of one episode, built with MjSpec in the URDF base frame.

    b = build(scene, episode=0, servo=ServoModel.from_scene(scene))
    b.model, b.qadr, b.act, b.objects, b.finger_geoms, b.cameras ...

CONTENTS (every number below carries its provenance; the scene supplies the rest):

    arm      paths.MJCF, base at identity. Visual copies -> group 2, collision copies ->
             group 3 with contype 1 / conaffinity 0 (no self-collision: adjacent hulls
             overlap by construction, mjlab so101_constants.py). The two finger hulls
             are replaced by their CoACD pieces (fingers.py). Joint ranges widened to
             the real firmware range + 5 deg (units.limits_rad; the data exceed the
             URDF on 4 joints); the Jaw's lower limit is JAW_TOUCH_DEG, where the
             finger meshes touch (the real shut stop, 2049 ticks in all 4 episodes).
    servo    servo.ServoModel: joint damping / armature / frictionloss, one position
             actuator per joint (kp, kd inside the effort clamp), and the Jaw's
             series flex hinge `Jaw_flex` coaxial with Jaw in the jaw body.
    mount    klip_support (18 g, mjlab klip_camera.py _MOUNT_MASS) and the KWC-500
             body (a 30 x 26 x 16 mm block, sim_agent101 objects.py BODY_W/D/L) as
             welded child bodies of `gripper`, visual only, mass only.
    cameras  `front` in the world and `grip` on `gripper`, each an oversized pinhole
             (camera.render_spec) that render.py warps through the real distortion.
    table    a box whose top is at table z; mat friction.
    objects  objects.episode_objects: caps (primitive cylinder collision, hollow-cap
             visual mesh, inertia of the hollow cap) and the mug (floor cylinder + a
             ring of N wall boxes + handle capsules; mesh visuals; inertia of the
             mesh). Free bodies, placed once, moved only by contact.

CONTACT (Physics). A 2.5 g cap between PLA fingers is the hard case for MuJoCo's soft
contacts: their stiffness is an ACCELERATION per metre of penetration, so the force
per metre scales with the effective mass, and the cap's is tiny. Equilibrium
penetration is r = (1 - d)/d * F A / k, with A ~ 1/m_cap = 400 /kg, k =
d/(d_width^2 tc^2): MuJoCo's defaults (tc 0.02, d 0.9..0.95) sink a cap 7.6 mm per
newton, which is the 5-6.6 mm at 0.5-0.9 N the understand phase measured on its
finger pads. Every contact of the objects and the fingers therefore uses
solref (obj_timeconst, 1) with obj_timeconst = 2 dt (the stability floor) and solimp
(0.99, 0.999, 0.5 mm). MEASURED at dt 1/1800 s: a 9.6 N squeeze sinks the cap 0.05 mm,
a cap dropped into the mug from 15 cm above the rim penetrates <= 0.26 mm on impact.
Critically damped (dampratio 1): contacts dissipate, they cannot return more energy
than an impact brings. Friction coefficients combine by MuJoCo's element-wise max, so
each MATERIAL carries its lowest pairing. MEASURED alternatives (ep0 track replay,
cap 27.1 mm): mjlab's single finger hulls lift the cap 1.2 mm and lose it (jaw closes
past it, -5.5 deg vs real); a rigid jaw (no flex) holds 1.4 deg too open; condim 3 / 4 / 6
give the same grasp; impratio 10 lets the held cap slip 0.8 mm vs 0.09 mm at 100.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .. import objects as objlib
from .. import paths
from ..camera import render_spec, to_mujoco
from ..kinematics import GRASP_SITE_POS
from ..transforms import mat_to_quat, quat_to_mat
from ..units import JAW_TOUCH_DEG, URDF_JOINTS, limits_rad, urdf_limits_rad
from . import fingers as fingerlib
from .servo import FLEX_JOINT, ServoModel

VISUAL_GROUP, COLLISION_GROUP, MARKER_GROUP = 2, 3, 5
LIMIT_MARGIN_DEG = 5.0  # beyond the firmware range: the real joints have no hard stop inside it

# mjlab klip_camera.py:37-45 (READ): gripper -> klip_support, mount EXTRINSIC xyz euler.
MOUNT_POS = (-0.01455, 0.08710, -0.01039)
MOUNT_RPY_DEG = (-179.755, -34.920, -90.061)
MOUNT_MASS = 0.018  # printed PLA bracket, mjlab _MOUNT_MASS (READ)
# sim_agent101 objects.py:58-85 (READ): bore centre in the support frame; KWC-500 body block
BORE_CENTRE = (0.017, -0.0175, 0.0)
WEBCAM_BLOCK = (0.030, 0.026, 0.016)
# ESTIMATED: the 30 x 26 x 16 mm plastic + PCB block at ~1.0 g/cm^3 is 12.5 g, the lens
# barrel ~2 g, plus the part of the USB cable the wrist carries ~5 g. Range 10-40 g.
WEBCAM_MASS = 0.020

# Materials: friction coefficient (slide, torsion m, roll m). Pair friction is the
# element-wise MAX of the two geoms, so cap = the lowest pairing it has (steel).
# ESTIMATED from handbook dry-friction ranges; ensembles perturb them (evaluate.py).
MATERIALS = {
    "cap": (0.30, 0.001, 0.0),  # PP/HDPE cap; also cap-on-steel and cap-on-cap
    "finger": (0.40, 0.001, 0.0),  # PLA on PP/HDPE 0.3-0.5
    "mat": (0.60, 0.001, 0.0),  # black glossy mat (vinyl/silicone) 0.4-0.9
    "mug": (0.30, 0.001, 0.0),  # enamel outside, polished stainless inside 0.2-0.4
}
COLORS = {
    "printed": (0.93, 0.93, 0.95, 1.0), "servo": (0.05, 0.05, 0.05, 1.0), "mount": (0.45, 0.45, 0.47, 1.0),
    "webcam": (0.02, 0.02, 0.02, 1.0), "cap": (105 / 255, 160 / 255, 159 / 255, 1.0),  # episodes report cap top RGB
    "mug": (0.55, 0.04, 0.03, 1.0), "steel": (0.49, 0.49, 0.43, 1.0),  # episodes report: steel RGB (126,126,110)
    "mat": (0.063, 0.082, 0.078, 1.0), "desk": (0.80, 0.80, 0.78, 1.0),  # mat RGB (16, 21, 20)
}


@dataclass(frozen=True)
class Physics:
    """Solver and contact settings. dt must divide the 30 fps frame (substeps)."""

    # dt = 1/1800 s. MEASURED: at 30 substeps (1/900 s) a cap dropped into the mug from
    # 5-15 cm above the rim penetrated 0.9-1.4 mm on impact and the held cap 0.16 mm; at 60,
    # 0.02-0.26 mm and 0.05 mm, for the same outcomes, at 3.7-5x real time.
    substeps: int = 60
    fps: float = 30.0
    integrator: str = "implicitfast"
    cone: str = "elliptic"  # pyramidal + impratio 1 never lifted the cap (understand phase)
    # MEASURED (ep0 track, cap held 4 s): impratio 10 let a grasp near the fingertips creep
    # 2 mm and pop out before the release; 30 and 100 held it to the release. Prior-art slip
    # test: 0.18 mm (10) vs 0.02 mm (100) under a 7.4 m/s^2 shake.
    impratio: float = 100.0
    iterations: int = 50
    ls_iterations: int = 50
    tolerance: float = 1e-10
    obj_timeconst: float | None = None  # None = 2 dt
    obj_dampratio: float = 1.0  # critically damped: no bounce energy
    obj_solimp: tuple = (0.99, 0.999, 0.0005, 0.5, 2.0)
    condim: int = 4  # torsional friction on object contacts
    # The servos' dry friction (frictionloss) as a HARD constraint. MEASURED: with MuJoCo's
    # default soft friction (solref 0.02, solimp 0.9/0.95) the ep0 Pitch creeps +0.74 deg over
    # frames 150-195 where the real encoder does not move a tick (stiction); at (2 dt, 1) and
    # (0.99, 0.999) it creeps 0.30 deg, and held-out ep2 Elbow RMSE falls 1.17 -> 0.82 deg.
    joint_friction_solimp: tuple = (0.99, 0.999, 0.001, 0.5, 2.0)
    margin: float = 0.0
    fingers: str = "coacd"  # "coacd" (fingers.py) or "hull" (mjlab's single hulls, for comparison)
    flex: bool = True  # the Jaw's series compliance (servo.py)
    contacts: bool = True
    gravity: bool = True
    materials: dict = field(default_factory=lambda: dict(MATERIALS))
    webcam_mass: float = WEBCAM_MASS
    mug_walls: int = 32
    # Cap collision: "cylinder" = a primitive of the scene's mid-height `diameter`; "taper" =
    # the convex frustum CALIB fitted (diameter_closed_end -> diameter_open_end), for
    # comparison. MEASURED (fitted scene 6f282e81, action mode): the fingers hold the four
    # open-up caps on their rims, the frustum's 32.3 mm end, so their jaw holds go from
    # +0.52..+1.17 to +1.33..+1.79 deg too open, and ep0 is still not released.
    cap_shape: str = "cylinder"

    @property
    def dt(self) -> float:
        return 1.0 / (self.fps * self.substeps)

    @property
    def timeconst(self) -> float:
        return self.obj_timeconst if self.obj_timeconst is not None else 2.0 * self.dt

    def replace(self, **kw) -> "Physics":
        return replace(self, **kw)


@dataclass(frozen=True)
class Perturbation:
    """Per-run deviations from the scene (ensembles). Units SI; empty = the scene as is."""

    cap_dxy: dict = field(default_factory=dict)  # {cap id: (dx, dy)}
    mug_dxy: tuple = (0.0, 0.0)
    table_dz: float = 0.0
    cap_diameter: float | None = None  # absolute override
    cap_height: float | None = None  # absolute override
    cap_mass: float | None = None
    friction_scale: dict = field(default_factory=dict)  # {material: factor on the slide coefficient}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Perturbation":
        """Inverse of as_dict (a pose log's meta.perturbation)."""
        d = d or {}
        return cls(cap_dxy={k: tuple(v) for k, v in d.get("cap_dxy", {}).items()}, mug_dxy=tuple(d.get("mug_dxy", (0.0, 0.0))),
                   table_dz=float(d.get("table_dz", 0.0)), cap_diameter=d.get("cap_diameter"),
                   cap_height=d.get("cap_height"), cap_mass=d.get("cap_mass"),
                   friction_scale=dict(d.get("friction_scale", {})))

    def as_dict(self) -> dict:
        return {"cap_dxy": {k: list(map(float, v)) for k, v in self.cap_dxy.items()}, "mug_dxy": list(self.mug_dxy),
                "table_dz": self.table_dz, "cap_diameter": self.cap_diameter, "cap_height": self.cap_height,
                "cap_mass": self.cap_mass, "friction_scale": dict(self.friction_scale)}


@dataclass
class Built:
    model: object
    physics: Physics
    qadr: np.ndarray  # qpos address of the 6 URDF joints (Jaw = the horn / encoder)
    dadr: np.ndarray
    act: np.ndarray  # actuator ids, URDF order
    flex_qadr: int | None
    flex_dadr: int | None
    objects: list  # [{name, kind, body, qadr, dadr, geoms, pos0, quat0}]
    finger_geoms: dict  # {"fixed": [...], "moving": [...]}
    table_geom: int | None
    cameras: dict  # {name: {"id", "spec", "cam"}} (pinhole render cameras)
    site_grasp: int
    info: dict
    arm_geoms: np.ndarray = None  # collision geoms of the moving links (fingers included)

    def obj(self, name: str) -> dict:
        return next(o for o in self.objects if o["name"] == name)


# --- helpers ----------------------------------------------------------------------

def _euler_xyz_extrinsic(rpy_deg) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll): scipy's extrinsic 'xyz' (mjlab _quat_extrinsic)."""
    r, p, y = (math.radians(a) for a in rpy_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _add_mesh(spec, name: str, verts, faces=None):
    m = spec.add_mesh(name=name)
    m.uservert = np.asarray(verts, dtype=np.float64).ravel().tolist()
    if faces is not None:
        m.userface = np.asarray(faces, dtype=np.int32).ravel().tolist()
    return m


def _mass_props(mesh, mass: float):
    """(com (3,), inertia (3, 3)) of a watertight trimesh scaled to `mass`."""
    rho = mass / mesh.volume
    return np.array(mesh.center_mass), np.array(mesh.moment_inertia) * rho


def _set_inertial(body, mass: float, com, I) -> None:
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    body.explicitinertial = True
    body.mass = float(mass)
    body.ipos = [float(x) for x in com]
    body.inertia = [float(x) for x in np.maximum(w, 1e-12)]
    body.iquat = mat_to_quat(V).tolist()


def _contact(g, phys: Physics, material: str, perturb: Perturbation) -> None:
    f = phys.materials[material]
    s = perturb.friction_scale.get(material, 1.0)
    g.friction = [f[0] * s, f[1], f[2]]
    g.condim = phys.condim
    g.solref = [phys.timeconst, phys.obj_dampratio]
    g.solimp = list(phys.obj_solimp)
    g.margin = phys.margin


def joint_ranges(units) -> np.ndarray:
    """(6, 2) MuJoCo joint ranges: URDF range united with the real firmware range +
    LIMIT_MARGIN_DEG; Jaw from JAW_TOUCH_DEG (a real, physical stop)."""
    real, urdf = limits_rad(units), urdf_limits_rad()
    m = math.radians(LIMIT_MARGIN_DEG)
    r = np.stack([np.minimum(urdf[:, 0], real[:, 0] - m), np.maximum(urdf[:, 1], real[:, 1] + m)], 1)
    r[5, 0] = math.radians(JAW_TOUCH_DEG)
    return r


# --- the builder ------------------------------------------------------------------

def build(scene, episode: int | None = 0, servo: ServoModel | None = None, physics: Physics | None = None,
          perturb: Perturbation | None = None, cameras=("front", "grip"), objects: bool = True,
          markers=(), placement: str = "config") -> Built:
    """The MjModel of `episode` (None or objects=False: arm, table and cameras only).
    placement: 'config' (the scene's cap xy) or 'grasp' (grasp_placements: a diagnostic).
    markers: [(body name or 'world', (x, y, z), radius)] visual spheres named marker<i>,
    for projection tests."""
    import mujoco

    phys = physics or Physics()
    perturb = perturb or Perturbation()
    servo = servo or ServoModel.from_scene(scene)
    units = scene.units()
    spec = mujoco.MjSpec.from_file(str(paths.MJCF))
    spec.modelname = f"real2sim_{scene.ds}_ep{episode}"
    o = spec.option
    o.timestep = phys.dt
    o.integrator = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "euler": mujoco.mjtIntegrator.mjINT_EULER,
                    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT, "rk4": mujoco.mjtIntegrator.mjINT_RK4}[phys.integrator]
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if phys.cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.impratio = phys.impratio
    o.iterations, o.ls_iterations, o.tolerance = phys.iterations, phys.ls_iterations, phys.tolerance
    if not phys.contacts:
        o.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    if not phys.gravity:
        o.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY

    # arm geoms: groups, colours, collision filter; the finger hulls out (replaced below)
    finger_hulls = {}
    for g in list(spec.geoms):
        visual = g.contype == 0 and g.conaffinity == 0
        shade = tuple(round(float(v), 2) for v in g.rgba[:3])
        g.rgba = COLORS["printed"] if shade == (1.0, 0.82, 0.12) else COLORS["servo"]
        g.group = VISUAL_GROUP if visual else COLLISION_GROUP
        if not visual:
            g.contype, g.conaffinity = 1, 0
            if g.meshname in fingerlib.FINGER_MESHES and phys.fingers == "coacd":
                finger_hulls[g.meshname] = (g.parent, list(g.pos), list(g.quat))
                spec.delete(g)
    finger_geom_names = {"fixed": [], "moving": []}
    if phys.fingers == "coacd":
        pieces = fingerlib.load(scene.ds)
        for mesh, (body, pos, quat) in finger_hulls.items():
            side = "fixed" if body.name == "gripper" else "moving"
            for i, v in enumerate(pieces[mesh]):
                nm = f"{side}_finger_{i:02d}"
                _add_mesh(spec, nm, v)
                g = body.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=nm, pos=pos, quat=quat,
                                  group=COLLISION_GROUP, contype=1, conaffinity=0, rgba=(0.2, 0.9, 0.3, 1))
                _contact(g, phys, "finger", perturb)
                finger_geom_names[side].append(nm)
    else:
        for g in spec.geoms:
            if g.contype == 1 and g.meshname in fingerlib.FINGER_MESHES:
                nm = f"{'fixed' if g.parent.name == 'gripper' else 'moving'}_finger_hull"
                g.name = nm
                _contact(g, phys, "finger", perturb)
                finger_geom_names["fixed" if g.parent.name == "gripper" else "moving"].append(nm)

    # joints: ranges and servo model
    ranges = joint_ranges(units)
    for i, n in enumerate(URDF_JOINTS):
        j, s = spec.joint(n), servo.joint(n)
        j.range = [float(ranges[i, 0]), float(ranges[i, 1])]
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        # mujoco 3.11: damping/stiffness are polynomials [linear, quadratic, cubic]
        j.damping, j.armature, j.frictionloss = [s.damping, 0.0, 0.0], s.armature, s.frictionloss
        j.solref_friction = [phys.timeconst, 1.0]
        j.solimp_friction = list(phys.joint_friction_solimp)
        a = spec.add_actuator(name=n, target=n)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = s.kp
        a.biasprm[0], a.biasprm[1], a.biasprm[2] = 0.0, -s.kp, -s.kd
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-s.effort, s.effort]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    jaw = servo.joint("Jaw")
    if phys.flex and jaw.flex_stiffness:
        jb = spec.body("jaw")
        fj = jb.add_joint(name=FLEX_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], pos=[0, 0, 0])
        fj.stiffness, fj.damping, fj.springref = [jaw.flex_stiffness, 0.0, 0.0], [jaw.flex_damping or 0.0, 0.0, 0.0], 0.0
        fj.limited = mujoco.mjtLimited.mjLIMITED_FALSE

    # mount + webcam: welded children of `gripper` (mass only, visual only)
    gripper = spec.body("gripper")
    R_m = _euler_xyz_extrinsic(MOUNT_RPY_DEG)
    ms = spec.add_mesh(name="klip_support", file=str(paths.MJCF.parent / "assets" / "klip_support.stl"))
    del ms
    mount = gripper.add_body(name="klip_support", pos=list(MOUNT_POS), quat=mat_to_quat(R_m).tolist())
    mount.add_geom(name="klip_support_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="klip_support",
                   mass=MOUNT_MASS, group=VISUAL_GROUP, contype=0, conaffinity=0, rgba=COLORS["mount"])
    # KWC-500 block glued to the hidden face of the bore plate (objects.py CAMERA_SIDE = -1)
    cam_body = mount.add_body(name="kwc500", pos=[BORE_CENTRE[0], BORE_CENTRE[1], -WEBCAM_BLOCK[2] / 2])
    cam_body.add_geom(name="kwc500_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[WEBCAM_BLOCK[0] / 2, WEBCAM_BLOCK[1] / 2, WEBCAM_BLOCK[2] / 2], mass=phys.webcam_mass,
                      group=VISUAL_GROUP, contype=0, conaffinity=0, rgba=COLORS["webcam"])
    gripper.add_site(name="grasp_site", pos=list(GRASP_SITE_POS), group=VISUAL_GROUP, size=[0.002, 0, 0])

    # table: top at table_top(x, y) (a plane: z + tilt, when CALIB fits a tilt)
    tz = scene.table_z() + perturb.table_dz
    top = lambda x, y: table_top(scene, x, y) + perturb.table_dz  # noqa: E731
    tq = _table_quat(scene)
    n = quat_to_mat(tq)[:, 2]
    for name, size, xy, off, grp, col in (
            ("table", (0.75, 0.6, 0.02), (0.0, -0.25), 0.0, COLLISION_GROUP, "mat"),
            # visuals: the black mat the overhead sees edge to edge, the white desk beyond it (episodes report)
            ("mat_visual", (0.40, 0.34, 0.0005), (0.04, -0.26), 0.0001, VISUAL_GROUP, "mat"),
            ("desk_visual", (1.0, 1.0, 0.001), (0.0, -0.3), -0.0002, VISUAL_GROUP, "desk")):
        c = np.array([xy[0], xy[1], top(*xy)]) - n * (size[2] - off)
        g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=c.tolist(),
                                    quat=tq.tolist(), group=grp, contype=int(grp == COLLISION_GROUP),
                                    conaffinity=int(grp == COLLISION_GROUP), rgba=COLORS[col])
        if name == "table":
            _contact(g, phys, "mat", perturb)
    spec.worldbody.add_light(name="key", pos=[-0.5, -0.65, 1.3], dir=[0.45, 0.35, -1.0], diffuse=[0.9, 0.9, 0.9],
                             castshadow=True)
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.headlight.ambient = [0.15, 0.15, 0.15]

    for i, (body, pos, r) in enumerate(markers):
        parent = spec.worldbody if body == "world" else spec.body(body)
        parent.add_geom(name=f"marker{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], pos=list(pos),
                        group=MARKER_GROUP, contype=0, conaffinity=0, mass=0, rgba=(1, 0, 1, 1))

    # objects
    objs = []
    if objects and episode is not None:
        objs = _add_objects(spec, scene, episode, phys, perturb, top, placement)

    # cameras (oversized pinholes; render.py remaps them through the real distortion)
    cams, offw, offh = {}, 640, 480
    for name in cameras:
        cam = scene.camera(name)
        rs = render_spec(cam, square=False)
        mj = to_mujoco(rs.camera(cam))
        parent = spec.worldbody if cam.parent == "world" else spec.body(cam.parent)
        c = parent.add_camera(name=name, pos=mj["pos"], quat=mj["quat"])
        c.resolution = mj["resolution"]
        c.sensor_size = mj["sensorsize"]
        c.focal_pixel = mj["focalpixel"]
        c.principal_pixel = mj["principalpixel"]
        cams[name] = {"spec": rs, "cam": cam}
        offw, offh = max(offw, rs.width), max(offh, rs.height)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = offw, offh

    m = spec.compile()
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in URDF_JOINTS])
    dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in URDF_JOINTS])
    act = np.array([m.actuator(n).id for n in URDF_JOINTS])
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, FLEX_JOINT)
    for name in cams:
        cams[name]["id"] = m.camera(name).id
    for ob in objs:
        b = m.body(ob["body_name"])
        ob["body"] = b.id
        j = m.jnt_qposadr[m.body_jntadr[b.id]]
        ob["qadr"], ob["dadr"] = int(j), int(m.jnt_dofadr[m.body_jntadr[b.id]])
        ob["geoms"] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b.id and m.geom_contype[g]]
    fg = {k: [m.geom(n).id for n in v] for k, v in finger_geom_names.items()}
    arm_b = {m.body(n).id for n in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")}
    arm_geoms = np.array([g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] in arm_b])
    info = {"dt": phys.dt, "substeps": phys.substeps, "timeconst": phys.timeconst, "fingers": phys.fingers,
            "placement": placement, "table_tilt": list(scene["table"].get("tilt") or (0.0, 0.0)),
            "flex": bool(phys.flex and jaw.flex_stiffness), "table_z": tz, "ngeom": int(m.ngeom),
            "n_finger_pieces": {k: len(v) for k, v in fg.items()}, "joint_ranges_deg": np.degrees(ranges).round(2).tolist(),
            "webcam_mass": phys.webcam_mass, "cap_shape": phys.cap_shape,
            "materials": {k: list(v) for k, v in phys.materials.items()}}
    return Built(m, phys, qadr, dadr, act, int(m.jnt_qposadr[fid]) if fid >= 0 else None,
                 int(m.jnt_dofadr[fid]) if fid >= 0 else None, objs, fg, m.geom("table").id, cams,
                 m.site("grasp_site").id, info, arm_geoms)


def table_top(scene, x, y):
    """Table top height (m, base frame) at x, y: table.z plus table.tilt (dz/dx, dz/dy)
    when the scene carries one (CALIB's params fit it; the Isaac track reads the same key)."""
    t = scene["table"].get("tilt") or (0.0, 0.0)
    return scene.table_z() + float(t[0]) * np.asarray(x) + float(t[1]) * np.asarray(y)


def _table_quat(scene) -> np.ndarray:
    t = scene["table"].get("tilt") or (0.0, 0.0)
    n = np.array([-float(t[0]), -float(t[1]), 1.0])
    n /= np.linalg.norm(n)
    ax = np.cross([0.0, 0.0, 1.0], n)
    if np.linalg.norm(ax) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    from ..transforms import axis_angle_quat

    return axis_angle_quat(ax / np.linalg.norm(ax), math.asin(min(1.0, np.linalg.norm(ax))))


def grasp_placements(scene, episode: int, cap_d: dict, top) -> dict:
    """{cap id: (x, y)}: where a rigid cap of this diameter must have stood for BOTH real
    fingers to hold it antipodally at the last frame of its real close (the jaw blocked
    on it), from the recorded joints. The SAME estimate the Isaac track offers
    (real2sim.isaac.placement.grasp_centre, pure numpy; imported so both engines place
    identically). A DIAGNOSTIC that separates placement error from contact physics:
    the default placement is the scene's (camera) estimate. A cap whose height band no
    finger reaches at that frame (the table perturbed low) keeps its config xy."""
    from .. import episodes as eplib
    from .. import kinematics
    from ..isaac.placement import grasp_centre

    kin = kinematics.default()
    ep = eplib.load(scene.ds, include_excluded=True)[episode]
    u = scene.units()
    out = {}
    for pk in scene.episode(episode)["picks"]:
        c = next(c for c in scene.episode(episode)["caps"] if c["id"] == pk["cap"])
        z0 = float(top(*c["xy"]))
        try:
            gc = grasp_centre(kin, u.to_urdf(ep.state[int(pk["close"][1])]), cap_d["diameter"], z0,
                              z0 + cap_d["height"])
        except ValueError:  # no finger reaches down into the cap's height band: keep the config xy
            continue
        out.setdefault(pk["cap"], tuple(gc["xy"]))  # a regrasped cap: its first pick
    return out


def _add_objects(spec, scene, episode: int, phys: Physics, perturb: Perturbation, top, placement: str) -> list:
    import mujoco

    cap_d = scene.cap_dims()
    if perturb.cap_diameter is not None:
        cap_d["diameter"] = float(perturb.cap_diameter)
    if perturb.cap_height is not None:
        cap_d["height"] = float(perturb.cap_height)
    if perturb.cap_mass is not None:
        cap_d["mass"] = float(perturb.cap_mass)
    mug_d = scene.mug_dims()
    cap_vis = objlib.cap_mesh(cap_d)
    com_c, I_c = _mass_props(cap_vis, cap_d["mass"])
    _add_mesh(spec, "cap_visual", cap_vis.vertices, cap_vis.faces)
    taper = phys.cap_shape == "taper"
    if taper:
        if "diameter_open_end" not in cap_d or "diameter_closed_end" not in cap_d:
            raise ValueError("cap_shape 'taper' needs objects.cap.diameter_open_end / diameter_closed_end (CALIB)")
        # the frustum CALIB fitted, rim plane (open end) at z = 0; a perturbed diameter shifts both ends
        dd = cap_d["diameter"] - scene.cap_dims()["diameter"]
        ang = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
        ring = lambda d, z: np.stack([d / 2 * np.cos(ang), d / 2 * np.sin(ang), np.full_like(ang, z)], 1)  # noqa: E731
        _add_mesh(spec, "cap_collision_taper", np.concatenate([ring(cap_d["diameter_open_end"] + dd, 0.0),
                                                               ring(cap_d["diameter_closed_end"] + dd, cap_d["height"])]))
    parts = objlib.mug_parts(mug_d)
    for k in ("body", "handle"):
        _add_mesh(spec, f"mug_{k}_visual", parts[k].vertices, parts[k].faces)
    liner = _mug_liner(mug_d)
    _add_mesh(spec, "mug_liner_visual", liner.vertices, liner.faces)
    com_m, I_m = _mass_props(objlib.mug_mesh(mug_d), mug_d["mass"])

    if placement not in ("config", "grasp"):
        raise ValueError(f"placement must be config|grasp, got {placement!r}")
    where = grasp_placements(scene, episode, cap_d, top) if placement == "grasp" else {}
    out = []
    for ob in objlib.episode_objects(scene, episode):
        pos, quat = np.array(ob["pos"], dtype=float), np.array(ob["quat"], dtype=float)
        if ob["kind"] == "cap":
            cid = ob["name"].split("_", 1)[1]
            if cid in where:
                pos[:2] = where[cid]
            pos[:2] += np.asarray(perturb.cap_dxy.get(cid, (0.0, 0.0)), dtype=float)
            # resting on the (tilted, perturbed) table: closed-up on its rim, open-up on its top
            pos[2] = float(top(*pos[:2])) + (cap_d["height"] if ob["up"] == "open" else 0.0)
            b = spec.worldbody.add_body(name=ob["name"], pos=pos.tolist(), quat=quat.tolist())
            b.add_freejoint(name=f"{ob['name']}_free")
            _set_inertial(b, cap_d["mass"], com_c, I_c)
            if taper:
                g = b.add_geom(name=f"{ob['name']}_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
                               meshname="cap_collision_taper", group=COLLISION_GROUP, contype=1, conaffinity=1,
                               mass=0, rgba=COLORS["cap"])
            else:
                g = b.add_geom(name=f"{ob['name']}_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                               size=[cap_d["diameter"] / 2, cap_d["height"] / 2, 0], pos=[0, 0, cap_d["height"] / 2],
                               group=COLLISION_GROUP, contype=1, conaffinity=1, mass=0, rgba=COLORS["cap"])
            _contact(g, phys, "cap", perturb)
            b.add_geom(name=f"{ob['name']}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="cap_visual",
                       group=VISUAL_GROUP, contype=0, conaffinity=0, mass=0, rgba=COLORS["cap"])
        else:
            pos[:2] += np.asarray(perturb.mug_dxy, dtype=float)
            pos[2] = float(top(*pos[:2]))
            b = spec.worldbody.add_body(name="mug", pos=pos.tolist(), quat=quat.tolist())
            b.add_freejoint(name="mug_free")
            _set_inertial(b, mug_d["mass"], com_m, I_m)
            for g in _mug_collision(b, mug_d, phys.mug_walls):
                _contact(g, phys, "mug", perturb)
            for k, col in (("body", "mug"), ("handle", "mug"), ("liner", "steel")):
                b.add_geom(name=f"mug_{k}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"mug_{k}_visual",
                           group=VISUAL_GROUP, contype=0, conaffinity=0, mass=0, rgba=COLORS[col])
        out.append({"name": ob["name"], "kind": ob["kind"], "up": ob.get("up"), "body_name": ob["name"],
                    "pos0": pos, "quat0": quat})
    return out


def _mug_collision(body, d: dict, n: int) -> list:
    """Floor cylinder + n wall boxes whose INNER faces are tangent to the inner radius
    (so the cavity is exactly Ri at every face centre) + handle capsules."""
    import mujoco

    Ro, H, w, F = d["outer_diameter"] / 2, d["height"], d["wall"], d["floor"]
    Ri = Ro - w
    geoms = [body.add_geom(name="mug_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[Ro, F / 2, 0],
                           pos=[0, 0, F / 2], group=COLLISION_GROUP, contype=1, conaffinity=1, mass=0,
                           rgba=COLORS["mug"])]
    half_w = (Ri + w) * math.tan(math.pi / n)  # outer corners of neighbours meet: no gap
    for i in range(n):
        th = 2 * math.pi * (i + 0.5) / n
        r = Ri + w / 2
        geoms.append(body.add_geom(name=f"mug_wall{i:02d}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                   size=[w / 2, half_w, (H - F) / 2], pos=[r * math.cos(th), r * math.sin(th), (H + F) / 2],
                                   quat=[math.cos(th / 2), 0, 0, math.sin(th / 2)], group=COLLISION_GROUP,
                                   contype=1, conaffinity=1, mass=0, rgba=COLORS["mug"]))
    h = d["handle"]
    rt = h["thickness"] / 2
    x_in = Ro - 0.001
    a = Ro + h["protrusion"] - rt - x_in
    b = (h["z_top"] - h["z_bottom"]) / 2
    zc = (h["z_top"] + h["z_bottom"]) / 2
    phi = np.linspace(-np.pi / 2, np.pi / 2, 7)
    P = np.stack([x_in + a * np.cos(phi), np.zeros_like(phi), zc + b * np.sin(phi)], 1)
    for i in range(len(P) - 1):
        geoms.append(body.add_geom(name=f"mug_handle{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[rt, 0, 0],
                                   fromto=[*P[i], *P[i + 1]], group=COLLISION_GROUP, contype=1, conaffinity=1,
                                   mass=0, rgba=COLORS["mug"]))
    return geoms


def _mug_liner(d: dict):
    """Visual only: the polished steel inside (inner wall + floor top), 0.3 mm proud of the enamel."""
    Ro, H, w, F = d["outer_diameter"] / 2, d["height"], d["wall"], d["floor"]
    Ri = Ro - w - 0.0003
    return objlib._revolve([(0, F + 0.0003), (Ri, F + 0.0003), (Ri, H - 0.0005), (Ri - 0.0004, H - 0.0005),
                            (Ri - 0.0004, F + 0.0007), (0, F + 0.0007)], 96)


def set_arm_state(b: Built, data, q, qdot=None) -> None:
    """Joint positions (URDF rad, 6) and velocities onto the arm; the flex hinge relaxed."""
    data.qpos[b.qadr] = q
    data.qvel[b.dadr] = 0.0 if qdot is None else qdot
    if b.flex_qadr is not None:
        data.qpos[b.flex_qadr] = 0.0
        data.qvel[b.flex_dadr] = 0.0


def arm_q(b: Built, data) -> np.ndarray:
    return data.qpos[b.qadr].copy()


def finger_angle(b: Built, data) -> float:
    """The moving finger's angle: horn (Jaw) + flex."""
    return float(data.qpos[b.qadr[5]] + (data.qpos[b.flex_qadr] if b.flex_qadr is not None else 0.0))


