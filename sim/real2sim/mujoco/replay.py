"""Replay one real episode in MuJoCo: the driver, its three modes, and the pose log.

    res = run(scene, episode, Config(mode="action"))
    res.log (real2sim.poselog dict), res.wall_s, res.built

MODES
    action     THE PHYSICALLY HONEST DEFAULT. All six servos get the recorded ACTION
               (what lerobot sent the follower), clamped to the firmware limits, held
               per frame and delayed by the identified dead time (servo.GoalStream),
               through the identified servo model. Nothing else steers the arm: its
               open-loop error against the real state is the servo model's.
    track      HYBRID. The five arm servos get a synthetic goal that makes the servo
               model reproduce the recorded STATE: inverse dynamics of the arm along
               the recorded trajectory (cubic spline through observation.state) gives
               the torque the servo must produce, and u = q_ref + (tau + kd qdot_ref)/kp
               is the goal that produces it, so the arm is where the cameras saw it.
               The servos keep their torque clamp and their compliance: a contact
               the recording knew nothing about still pushes the arm back. The Jaw
               stays on the recorded action (the state is blocked by the cap and
               carries no squeeze), through the servo model and its torque limit.
    kinematic  NOT PHYSICAL, for camera and render checks only: the joints are set to
               the recorded state every frame, objects stay where they were placed,
               no dynamics. Logged with meta physical=false.

TIMING. Frame k is t_k = k / fps. The logged state of frame k is the simulation at t_k,
sampled BEFORE the substeps to t_(k+1) (the recorder reads state[k] and then sends
action[k]). ctrl is the goal in force at t_k.

WHAT IS LOGGED per frame (x_* extras of the pose log, T = frames):
    x_pen_fc      (T,)     max finger-cap penetration over the frame's substeps (m)
    x_pen_obj     (T,)     max penetration of any contact involving an object (m)
    x_touch       (T, C, 2) cap c touched by the fixed / moving finger during the frame
    x_fc_force    (T, C, 2) squeeze: normal force of the fixed / moving finger on cap c, summed
                           over its contact points, averaged over the frame's substeps (N)
    x_arm_obj     (T,)     max normal force of any non-finger arm geom on an object (N)
    x_flex        (T,)     Jaw_flex deflection (rad); x_finger = Jaw + flex
    x_ncon        (T,)     contacts at t_k
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import episodes as eplib
from .. import poselog
from ..kinematics import LINKS
from ..units import URDF_JOINTS
from . import model as mdl
from .servo import GoalStream, ServoModel, firmware_limits_rad


@dataclass
class Config:
    mode: str = "action"  # action | track | kinematic
    physics: mdl.Physics = field(default_factory=mdl.Physics)
    perturb: mdl.Perturbation = field(default_factory=mdl.Perturbation)
    seed: int | None = None
    frames: tuple | None = None  # (first, stop) episode-local window, for debugging
    settle_s: float = 0.3  # objects settle under gravity before t_0, arm held at state[0]
    placement: str = "config"  # "config" (the scene's cap xy) | "grasp" (model.grasp_placements)


@dataclass
class Result:
    log: dict
    built: object
    wall_s: float
    sim_s: float
    finite: bool


# --- geom categories for the vectorised contact scan ------------------------------------

OTHER, FIXED, MOVING, ARM, TABLE, MUG = 0, 1, 2, 3, 4, 5
CAP0 = 10  # cap c -> CAP0 + c


def geom_categories(b: mdl.Built) -> np.ndarray:
    m = b.model
    cat = np.full(m.ngeom, OTHER, dtype=np.int32)
    arm_bodies = {m.body(n).id for n in LINKS}
    for g in range(m.ngeom):
        if m.geom_contype[g] and m.geom_bodyid[g] in arm_bodies:
            cat[g] = ARM
    cat[b.finger_geoms["fixed"]] = FIXED
    cat[b.finger_geoms["moving"]] = MOVING
    cat[b.table_geom] = TABLE
    ci = 0
    for o in b.objects:
        if o["kind"] == "cap":
            cat[o["geoms"]] = CAP0 + ci
            ci += 1
        else:
            cat[o["geoms"]] = MUG
    return cat


class ContactScan:
    """Accumulates the x_* contact metrics over the substeps of one frame (numpy only)."""

    def __init__(self, b: mdl.Built):
        self.cat = geom_categories(b)
        self.ncap = sum(o["kind"] == "cap" for o in b.objects)
        self.reset()

    def reset(self):
        self.pen_fc = 0.0
        self.pen_obj = 0.0
        self.touch = np.zeros((self.ncap, 2), bool)
        self.force = np.zeros((self.ncap, 2))  # summed normal force per finger, accumulated
        self.n = 0
        self.arm_obj = 0.0

    def mean_force(self) -> np.ndarray:
        return self.force / max(self.n, 1)

    def scan(self, d) -> None:
        self.n += 1
        n = d.ncon
        if n == 0:
            return
        c = d.contact
        g1, g2 = c.geom1[:n], c.geom2[:n]
        dist = c.dist[:n]
        c1, c2 = self.cat[g1], self.cat[g2]
        obj = (c1 >= MUG) | (c2 >= MUG)
        if obj.any():
            self.pen_obj = max(self.pen_obj, float(-dist[obj].min()))
        if not self.ncap:
            return
        ea = c.efc_address[:n]
        # normal force = the contact's first constraint row (elliptic cones); -1 = not in the solver
        fn = np.where(ea >= 0, d.efc_force[np.maximum(ea, 0)], 0.0) if d.nefc else np.zeros(n)
        cap = np.where(c1 >= CAP0, c1 - CAP0, np.where(c2 >= CAP0, c2 - CAP0, -1))
        other = np.where(c1 >= CAP0, c2, c1)
        for side, code in ((0, FIXED), (1, MOVING)):
            k = (cap >= 0) & (other == code)
            if k.any():
                self.pen_fc = max(self.pen_fc, float(-dist[k].min()))
                np.add.at(self.force[:, side], cap[k], fn[k])
                self.touch[cap[k], side] = True
        a = (((c1 == ARM) & (c2 >= MUG)) | ((c2 == ARM) & (c1 >= MUG)))
        if a.any():
            self.arm_obj = max(self.arm_obj, float(fn[a].max()))


# --- the track-mode goal --------------------------------------------------------------

def track_goals(scene, servo: ServoModel, state_rad: np.ndarray, fps: float, substeps: int) -> np.ndarray:
    """(T * substeps, 5) arm goals that make the servo model follow the recorded state:
    u = q_ref + (tau_ff + kd qdot_ref) / kp, tau_ff from mj_inverse on the arm-only
    model (damping included, frictionloss added as a smoothed Coulomb term)."""
    import mujoco
    from scipy.interpolate import CubicSpline

    T = len(state_rad)
    t = np.arange(T) / fps
    sp = CubicSpline(t, state_rad[:, :5], bc_type="natural")
    ts = np.arange(T * substeps) / (fps * substeps)
    ts = np.minimum(ts, t[-1])
    q, qd, qdd = sp(ts), sp(ts, 1), sp(ts, 2)
    phys = mdl.Physics(substeps=substeps, contacts=False, fingers="hull", flex=False)
    b = mdl.build(scene, None, servo, phys, cameras=(), objects=False)
    m = b.model
    m.dof_frictionloss[:] = 0.0
    d = mujoco.MjData(m)
    kp = np.array([servo.joint(n).kp for n in URDF_JOINTS[:5]])
    kd = np.array([servo.joint(n).kd for n in URDF_JOINTS[:5]])
    fl = np.array([servo.joint(n).frictionloss for n in URDF_JOINTS[:5]])
    jaw_q = state_rad[0, 5]
    u = np.empty((len(ts), 5))
    for i in range(len(ts)):
        d.qpos[b.qadr[:5]] = q[i]
        d.qpos[b.qadr[5]] = jaw_q
        d.qvel[b.dadr[:5]] = qd[i]
        d.qacc[:] = 0.0
        d.qacc[b.dadr[:5]] = qdd[i]
        mujoco.mj_inverse(m, d)
        tau = d.qfrc_inverse[b.dadr[:5]] + fl * np.tanh(qd[i] / 0.02)
        u[i] = q[i] + (tau + kd * qd[i]) / kp
    return u


# --- the run -------------------------------------------------------------------------

def run(scene, episode: int, cfg: Config | None = None, servo: ServoModel | None = None, built=None,
        progress=None, warn=print) -> Result:
    import mujoco

    cfg = cfg or Config()
    servo = servo or ServoModel.from_scene(scene)
    b = built or mdl.build(scene, episode, servo, cfg.physics, cfg.perturb, placement=cfg.placement)
    m = b.model
    d = mujoco.MjData(m)
    u = scene.units()
    ep = eplib.load(scene.ds, include_excluded=True)[episode]
    lo, hi = cfg.frames or (0, len(ep))
    state, action = u.to_urdf(ep.state[lo:hi]), u.to_urdf(ep.action[lo:hi])
    T, fps, n_sub = len(state), float(ep.fps), b.physics.substeps
    dt = m.opt.timestep
    limits = firmware_limits_rad(u)

    # reset: arm at the recorded state, objects where the scene placed them, settle
    mdl.set_arm_state(b, d, state[0])
    d.ctrl[b.act] = state[0]
    mujoco.mj_forward(m, d)
    if cfg.mode != "kinematic" and cfg.settle_s > 0:
        arm_q0 = d.qpos[b.qadr].copy()
        for _ in range(int(round(cfg.settle_s / dt))):
            mujoco.mj_step(m, d)
            d.qpos[b.qadr] = arm_q0  # the arm is held exactly (it is at rest at t_0 in the data)
            d.qvel[b.dadr] = 0.0
        d.time = 0.0
    mdl.set_arm_state(b, d, state[0])
    mujoco.mj_forward(m, d)
    start_pen = _arm_table_penetration(b, d)
    if start_pen > 5e-4 and warn:
        warn(f"episode {episode}: the recorded start pose puts the arm {start_pen * 1000:.1f} mm INSIDE the table "
             f"(table z {b.info['table_z'] * 1000:.1f} mm): the scene's table height / joint offsets are not "
             "consistent with the data, and the start is not physically feasible")

    stream = GoalStream(action, fps, servo.dead_time, servo.max_velocity, limits, q0=state[0])
    arm_goal = track_goals(scene, servo, state, fps, n_sub) if cfg.mode == "track" else None
    scan = ContactScan(b)
    C = scan.ncap
    link_ids = [m.body(n).id for n in LINKS]
    obj_ids = [o["body"] for o in b.objects]
    L, O = len(link_ids), len(obj_ids)
    q_log, ctrl_log = np.empty((T, 6)), np.empty((T, 6))
    lp, lq = np.empty((T, L, 3)), np.empty((T, L, 4))
    op, oq = np.empty((T, O, 3)), np.empty((T, O, 4))
    x = {"x_pen_fc": np.zeros(T), "x_pen_obj": np.zeros(T), "x_touch": np.zeros((T, C, 2), bool),
         "x_fc_force": np.zeros((T, C, 2)), "x_arm_obj": np.zeros(T), "x_flex": np.zeros(T),
         "x_finger": np.zeros(T), "x_ncon": np.zeros(T, np.int32), "x_actuator_force": np.zeros((T, 6))}
    finite = True
    t0 = time.time()
    ctrl = state[0].copy()
    for k in range(T):
        # --- log frame k (t_k) ---
        q_log[k] = d.qpos[b.qadr]
        ctrl_log[k] = ctrl
        lp[k], lq[k] = d.xpos[link_ids], d.xquat[link_ids]
        if O:
            op[k], oq[k] = d.xpos[obj_ids], d.xquat[obj_ids]
        x["x_flex"][k] = d.qpos[b.flex_qadr] if b.flex_qadr is not None else 0.0
        x["x_finger"][k] = mdl.finger_angle(b, d)
        x["x_ncon"][k] = d.ncon
        x["x_actuator_force"][k] = d.actuator_force[b.act]
        if k == T - 1:
            break
        # --- advance to t_(k+1) ---
        scan.reset()
        if cfg.mode == "kinematic":
            mdl.set_arm_state(b, d, state[k + 1])
            mujoco.mj_forward(m, d)
            ctrl = state[k + 1]
            continue
        for s in range(n_sub):
            tt = (k + s / n_sub) / fps
            ctrl = stream.at(tt)
            if arm_goal is not None:
                ctrl = ctrl.copy()
                ctrl[:5] = arm_goal[k * n_sub + s]
            d.ctrl[b.act] = ctrl
            mujoco.mj_step(m, d)
            scan.scan(d)
        x["x_pen_fc"][k + 1], x["x_pen_obj"][k + 1] = scan.pen_fc, scan.pen_obj
        x["x_touch"][k + 1], x["x_fc_force"][k + 1], x["x_arm_obj"][k + 1] = scan.touch, scan.mean_force(), scan.arm_obj
        if not np.isfinite(d.qpos).all():
            finite = False
            q_log[k + 1:] = np.nan
            break
        if progress and k % 30 == 0:
            progress(k, T)
    wall = time.time() - t0
    frame = np.arange(lo, lo + T)
    meta = poselog.meta("mujoco", cfg.mode, episode, scene.hash, dt=dt, substeps=n_sub, seed=cfg.seed,
                        perturbation=cfg.perturb.as_dict(), physical=cfg.mode != "kinematic",
                        servo=servo.to_config()["model"], dead_time_s=servo.dead_time, physics=_phys_meta(b),
                        caps=[o["name"] for o in b.objects if o["kind"] == "cap"], wall_s=round(wall, 2),
                        finite=finite, scene_layers=[p.name for p in scene.layers],
                        start_arm_table_penetration_mm=round(start_pen * 1000, 3), placement=cfg.placement,
                        scene_overrides=getattr(scene, "overrides", {}),
                        cap_start={o["name"]: np.round(o["pos0"], 5).tolist() for o in b.objects})
    log = poselog.make(fps, frame, np.nan_to_num(q_log), ctrl_log, LINKS, lp, lq, [o["name"] for o in b.objects],
                       op if O else None, oq if O else None, meta, **x)
    return Result(log, b, wall, (T - 1) / fps, finite)


def _arm_table_penetration(b, d) -> float:
    """Deepest arm-table contact (m) in the current state (mj_forward done)."""
    n = d.ncon
    if not n:
        return 0.0
    g1, g2, dist = d.contact.geom1[:n], d.contact.geom2[:n], d.contact.dist[:n]
    t = (g1 == b.table_geom) | (g2 == b.table_geom)
    other = np.where(g1 == b.table_geom, g2, g1)
    arm = np.isin(other, b.arm_geoms)
    k = t & arm
    return float(max(0.0, -dist[k].min())) if k.any() else 0.0


def _phys_meta(b) -> dict:
    p = b.physics
    return {"dt": p.dt, "integrator": p.integrator, "cone": p.cone, "impratio": p.impratio, "iterations": p.iterations,
            "timeconst": p.timeconst, "solimp": list(p.obj_solimp), "condim": p.condim, "fingers": p.fingers,
            "flex": b.info["flex"], "cap_shape": p.cap_shape, "table_z": b.info["table_z"], "materials": b.info["materials"]}
