"""Identify the servo model (servo.py) from the dataset: recorded action in, recorded state out.

    ./robot real2sim mujoco servo-fit [--iters N] [--workers W]   -> config/<ds>.servo.json

METHOD (SIMPLER / halloween-bot style: replay the real commands open-loop, match the
real response). The arm alone (contacts off: nothing in the data touches the arm
except the cap in the jaws) is driven by the recorded ACTION through servo.GoalStream
and compared with observation.state at every frame:

    loss_j = RMSE over frames of (sim q_j - real q_j), in URDF degrees
    train  episodes 0 + 1 (1386 frames, 46 s), held out: episode 2 (782 frames)
    Jaw    only on FREE frames: a frame is excluded while the real jaw is blocked by a
           cap (scene picks: close start .. drop end, plus 5 frames): with no cap in
           this model a blocked frame measures the cap, not the servo. The holds
           are fitted separately (grip_hold.py) for the flex and the torque limit.

Per joint: kp, kd, damping, armature, frictionloss (log-parameterised, bounded).
Common to all joints: the dead time (one sync_write carries every goal). The effort
clamps are not fitted from tracking (ep0 + ep1 never need more than 2.5 N.m): the
arm's is the STS3215 full-duty stall (servo.BAM_STALL = 3.41 N.m; the held-out ep2
needs up to 3.16 N.m on Pitch), the gripper's comes from its holds (grip_hold.py).

OPTIMISER. The joints are only weakly coupled, so each evaluation (one simulation of
all six joints with every joint's candidate) scores all six at once, and a separable
(1+lambda) evolution strategy keeps, per joint, the best of the incumbent and lambda
Gaussian candidates, adapting each joint's step size (x1.5 on success, x0.82 on
failure). Candidates run in a process pool. The dead time is a 1-D outer search over
the grid dead_grid (s), refined by a parabola through the best three.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import episodes as eplib
from .. import paths
from ..scene import load as load_scene
from ..units import URDF_JOINTS
from . import model as mdl
from .servo import BAM_MAX_VELOCITY, BAM_STALL, GoalStream, JointServo, ServoModel, firmware_limits_rad

PARAMS = ("kp", "kd", "damping", "armature", "frictionloss")
# log-space bounds (SI). kp 2..40 N.m/rad brackets every published STS3215 value
# (BAM 9.34 at P=16, MJCF 17.8 at P=32); armature 1e-4..0.2 kg m^2 (BAM 0.026, mjlab 0.1).
BOUNDS = {"kp": (2.0, 40.0), "kd": (1e-3, 3.0), "damping": (1e-3, 5.0), "armature": (1e-4, 0.2),
          "frictionloss": (1e-3, 0.5)}
START = {"kp": 8.0, "kd": 0.05, "damping": 0.8, "armature": 0.01, "frictionloss": 0.052}  # joint_units best fit
HOLD_PAD = 5  # frames after a drop still excluded from the Jaw loss (the jaw re-opens off the cap)


# --- data -------------------------------------------------------------------------

def episode_data(scene, episode: int) -> dict:
    """{'goal', 'state' (T, 6) URDF rad, 'jaw_free' (T,) bool} of one recorded episode."""
    u = scene.units()
    ep = eplib.load(scene.ds, include_excluded=True)[episode]
    free = np.ones(len(ep), bool)
    for pk in scene.episode(episode).get("picks", []):
        a = min(pk["close"][0], pk.get("first_close", pk["close"])[0])
        free[a: pk["drop"][1] + 1 + HOLD_PAD] = False
    return {"goal": u.to_urdf(ep.action), "state": u.to_urdf(ep.state), "jaw_free": free, "fps": ep.fps}


# --- simulation (arm only) ----------------------------------------------------------

class ArmSim:
    """The arm-only model (no contacts, no objects), re-parameterised in place per run."""

    def __init__(self, scene, substeps: int | None = None):
        import mujoco

        self.mj = mujoco
        self.scene = scene
        substeps = substeps or mdl.Physics().substeps
        phys = mdl.Physics(substeps=substeps, contacts=False, fingers="hull", flex=False)
        self.b = mdl.build(scene, None, ServoModel.from_scene(scene), phys, cameras=(), objects=False)
        self.m = self.b.model
        self.d = mujoco.MjData(self.m)
        self.limits = firmware_limits_rad(scene.units())

    def set_servo(self, servo: ServoModel) -> None:
        m, b = self.m, self.b
        for i, n in enumerate(URDF_JOINTS):
            s, a, v = servo.joint(n), b.act[i], b.dadr[i]
            m.actuator_gainprm[a, 0] = s.kp
            m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = -s.kp, -s.kd
            m.actuator_forcerange[a] = (-s.effort, s.effort)
            m.dof_damping[v], m.dof_armature[v], m.dof_frictionloss[v] = s.damping, s.armature, s.frictionloss

    def run(self, servo: ServoModel, data: dict) -> np.ndarray:
        """Sim joint angles at every frame (T, 6): state at frame k is sampled at t_k,
        after which the goal stream (dead time, slew) drives the substeps to t_k+1."""
        mj, m, d, b = self.mj, self.m, self.d, self.b
        self.set_servo(servo)
        mj.mj_resetData(m, d)
        q0 = data["state"][0]
        mdl.set_arm_state(b, d, q0)
        mj.mj_forward(m, d)
        stream = GoalStream(data["goal"], data["fps"], servo.dead_time, servo.max_velocity, self.limits, q0=q0)
        n_sub = b.physics.substeps
        dt = m.opt.timestep
        T = len(data["goal"])
        out = np.empty((T, 6))
        ctrl = d.ctrl
        act = b.act
        t = 0.0
        for k in range(T):
            out[k] = d.qpos[b.qadr]
            for _ in range(n_sub):
                ctrl[act] = stream.at(t)
                mj.mj_step(m, d)
                t += dt
            if not np.isfinite(d.qpos[b.qadr[0]]):
                out[k:] = np.nan
                break
        return out


def rmse_deg(sim: np.ndarray, data: dict) -> np.ndarray:
    """(6,) per-joint RMSE in degrees; the Jaw only over free frames."""
    e = np.degrees(sim - data["state"])
    r = np.sqrt(np.nanmean(e ** 2, axis=0))
    r[5] = np.sqrt(np.nanmean(e[data["jaw_free"], 5] ** 2))
    return np.where(np.isfinite(r), r, 1e3)


def lag_frames(sim: np.ndarray, real: np.ndarray, goal: np.ndarray, max_lag: float = 10.0) -> tuple:
    """(sim lag, real lag) behind the goal per joint: the fractional shift minimising RMSE."""
    t = np.arange(len(goal))
    res = []
    for x in (sim, real):
        lags = []
        for j in range(6):
            best = (np.inf, 0.0)
            for L in np.arange(0.0, max_lag + 1e-9, 0.1):
                gi = np.interp(t - L, t, goal[:, j])
                r = np.sqrt(np.mean((x[10:, j] - gi[10:]) ** 2))
                best = min(best, (r, L))
            lags.append(best[1])
        res.append(np.array(lags))
    return tuple(res)


# --- parameter vectors ----------------------------------------------------------------

def _to_theta(servo: ServoModel) -> np.ndarray:
    """(6, 5) log-parameters, clipped into BOUNDS."""
    th = np.empty((6, len(PARAMS)))
    for i, n in enumerate(URDF_JOINTS):
        s = servo.joint(n)
        for k, p in enumerate(PARAMS):
            lo, hi = BOUNDS[p]
            th[i, k] = math.log(min(max(getattr(s, p), lo), hi))
    return th


def _from_theta(base: ServoModel, th: np.ndarray, dead_time: float) -> ServoModel:
    js = {}
    for i, n in enumerate(URDF_JOINTS):
        kw = {}
        for k, p in enumerate(PARAMS):
            lo, hi = BOUNDS[p]
            kw[p] = float(min(max(math.exp(th[i, k]), lo), hi))
        js[n] = base.joint(n).replace(**kw)
    return base.replace(joints=js, dead_time=float(dead_time))


# --- worker pool (each process builds its own model once) ----------------------------

_W: dict = {}


def _init_worker(ds: str, episodes_: tuple, substeps: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sc = load_scene(ds)
    _W["sim"] = ArmSim(sc, substeps)
    _W["data"] = [episode_data(sc, e) for e in episodes_]


def _eval(servo: ServoModel) -> np.ndarray:
    """Per-joint RMSE (deg) over the pool's episodes, frame-weighted."""
    sq, n = np.zeros(6), 0
    for data in _W["data"]:
        r = rmse_deg(_W["sim"].run(servo, data), data)
        T = len(data["goal"])
        sq += r ** 2 * T
        n += T
    return np.sqrt(sq / n)


def es_fit(base: ServoModel, dead_time: float, pool, iters: int = 30, lam: int = 16, sigma0: float = 0.35,
           seed: int = 0, mask=None, log=print) -> tuple:
    """Separable (1+lambda)-ES over the per-joint log-parameters at a fixed dead time.
    mask (6, 5) bool: which (joint, PARAMS) entries move (default all).
    Returns (servo, per-joint train RMSE)."""
    rng = np.random.default_rng(seed)
    mask = np.ones((6, len(PARAMS)), bool) if mask is None else np.asarray(mask, bool)
    live = mask.any(1)
    th = _to_theta(base)
    f = pool.submit(_eval, _from_theta(base, th, dead_time)).result()
    sig = np.full(6, sigma0)
    for it in range(iters):
        cands = [th + mask * sig[:, None] * rng.standard_normal(th.shape) for _ in range(lam)]
        S = np.array(list(pool.map(_eval, [_from_theta(base, c, dead_time) for c in cands])))  # (lam, 6)
        best = S.argmin(0)
        for j in np.flatnonzero(live):
            if S[best[j], j] < f[j]:
                th[j], f[j] = cands[best[j]][j], S[best[j], j]
                sig[j] = min(sig[j] * 1.5, 1.0)
            else:
                sig[j] = max(sig[j] * 0.82, 0.01)
        # the kept rows came from different candidates: re-score them together
        f = pool.submit(_eval, _from_theta(base, th, dead_time)).result()
        log(f"    it {it + 1:2d}  rmse " + " ".join(f"{v:5.2f}" for v in f) + "  sigma "
            + " ".join(f"{v:.2f}" for v in sig[live]))
    return _from_theta(base, th, dead_time), f


SHARED = ("kp", "kd", "damping", "armature")  # one STS3215 model: the motor constants are common


def _shared_from_vec(base: ServoModel, x: np.ndarray, dead_time: float, joints=URDF_JOINTS[:5]) -> ServoModel:
    """x = log of SHARED (4) then log frictionloss per joint in `joints`: the same servo on
    those joints, with a per-joint Coulomb term (gear friction grows with the load)."""
    js = dict(base.joints)
    for i, n in enumerate(joints):
        kw = {p: float(np.clip(math.exp(x[k]), *BOUNDS[p])) for k, p in enumerate(SHARED)}
        kw["frictionloss"] = float(np.clip(math.exp(x[len(SHARED) + i]), *BOUNDS["frictionloss"]))
        js[n] = base.joint(n).replace(**kw)
    return base.replace(joints=js, dead_time=float(dead_time))


def shared_fit(base: ServoModel, dead_time: float, pool, iters: int = 40, lam: int = 16, sigma0: float = 0.3,
               seed: int = 2, joints=URDF_JOINTS[:5], log=print) -> tuple:
    """(1+lambda)-ES on the shared model of `joints`; loss = sum of their train RMSEs."""
    rng = np.random.default_rng(seed)
    idx = [URDF_JOINTS.index(n) for n in joints]
    r = base.joint(joints[0])
    x = np.log([min(max(getattr(r, p), BOUNDS[p][0]), BOUNDS[p][1]) for p in SHARED]
               + [min(max(base.joint(n).frictionloss, BOUNDS["frictionloss"][0]), BOUNDS["frictionloss"][1])
                  for n in joints])
    fx = pool.submit(_eval, _shared_from_vec(base, x, dead_time, joints)).result()
    sig = sigma0
    for it in range(iters):
        cands = [x + sig * rng.standard_normal(x.shape) for _ in range(lam)]
        S = np.array(list(pool.map(_eval, [_shared_from_vec(base, c, dead_time, joints) for c in cands])))
        k = int(S[:, idx].sum(1).argmin())
        if S[k, idx].sum() < fx[idx].sum():
            x, fx, sig = cands[k], S[k], min(sig * 1.5, 1.0)
        else:
            sig = max(sig * 0.82, 0.01)
        log(f"    shared it {it + 1:2d}  rmse " + " ".join(f"{v:5.2f}" for v in fx) + f"  sigma {sig:.3f}")
    return _shared_from_vec(base, x, dead_time, joints), fx


def evaluate(sc, servo: ServoModel, episodes_=(0, 1, 2), train=(0, 1), substeps: int | None = None) -> dict:
    """Per-episode RMSE, bias, max error, lags and peak speeds of a servo model."""
    sim = ArmSim(sc, substeps)
    out = {}
    for e in episodes_:
        data = episode_data(sc, e)
        q = sim.run(servo, data)
        ls, lr = lag_frames(q, data["state"], data["goal"])
        v = lambda x: np.degrees(np.abs(np.gradient(x[:, :5], axis=0)) * data["fps"]).max(0)  # noqa: E731
        out[e] = {"split": "train" if e in train else "held-out", "rmse_deg": rmse_deg(q, data).round(3).tolist(),
                  "bias_deg": np.degrees(np.nanmean(q - data["state"], 0)).round(3).tolist(),
                  "max_abs_deg": np.degrees(np.nanmax(np.abs(q - data["state"]), 0)).round(2).tolist(),
                  "lag_sim_frames": ls.round(2).tolist(), "lag_real_frames": lr.round(2).tolist(),
                  "peak_speed_sim_deg_s": v(q).round(0).tolist(), "peak_speed_real_deg_s": v(data["state"]).round(0).tolist()}
    return out


def identify(ds: str | None = None, iters: int = 25, workers: int = 8, train=(0, 1), test=(2,), substeps: int | None = None,
             dead_grid=(0.0, 0.0111, 0.0222, 0.0333, 0.0444, 0.0556), log=print) -> dict:
    """The whole identification, in order:

      1  gripper holds (grip_hold) at the start kp: the gripper's torque limit; the arm clamp
         is the STS3215's full-duty stall (servo.BAM_STALL, 3.41 N.m)
      2  dead-time grid, per-joint ES at the best three dead times, parabola, final ES:
         the PER-JOINT model (a diagnostic: an unloaded joint only fixes ratios of kp,
         damping, armature and friction -- the Jaw came out at kp 3.8 -- so it is not used)
      3  the SHARED model on the five arm joints: one kp, kd, damping, armature (the
         same STS3215) plus per-joint frictionloss, from the per-joint model's
         gravity-loaded mean
      4  the Jaw: kp fixed to the shared value (same servo); kd, damping, armature,
         frictionloss fitted on its free frames
      5  gripper holds again with the final Jaw kp: flex stiffness and torque limit
      6  evaluation on train and held-out episodes
    Returns {'servo', 'per_joint', 'hold', 'report'}."""
    from . import grip_hold

    substeps = substeps or mdl.Physics().substeps  # fit at the replay's own timestep
    sc = load_scene(ds)
    t0 = time.time()
    hold0 = grip_hold.analyse(sc, START["kp"])
    eff = {n: BAM_STALL for n in URDF_JOINTS}
    eff["Jaw"] = hold0["effort_limit"]
    base = ServoModel.from_scene(sc)
    base = base.replace(max_velocity=BAM_MAX_VELOCITY, joints={
        n: JointServo(START["kp"], START["kd"], START["damping"], START["armature"], START["frictionloss"], eff[n],
                      None, None, "fit start") for n in URDF_JOINTS})
    results = {}
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(sc.ds, tuple(train), substeps)) as pool:
        coarse = {dt: pool.submit(_eval, base.replace(dead_time=dt)).result() for dt in dead_grid}
        for dt, r in coarse.items():
            log(f"  dead {dt * 1000:5.1f} ms at the start point: rmse " + " ".join(f"{v:5.2f}" for v in r))
        for dt in sorted(coarse, key=lambda k: coarse[k][:5].mean())[:3]:
            log(f"  per-joint ES at dead time {dt * 1000:.1f} ms")
            results[dt] = es_fit(base, dt, pool, iters=max(iters * 3 // 5, 8), log=log)
        xs = np.array(sorted(results))
        ys = np.array([results[x][1][:5].mean() for x in xs])
        dt_star = float(xs[np.argmin(ys)])
        a, bq, _ = np.polyfit(xs, ys, 2)
        if a > 0:
            dt_star = float(np.clip(-bq / (2 * a), xs.min(), xs.max()))
        prev = results[min(results, key=lambda k: results[k][1][:5].mean())][0]
        log(f"  per-joint ES at the parabola's dead time {dt_star * 1000:.1f} ms")
        per_joint, _ = es_fit(prev, dt_star, pool, iters=max(iters * 2 // 5, 8), sigma0=0.15, seed=1, log=log)
        st = {p: float(np.mean([getattr(per_joint.joint(n), p) for n in ("Pitch", "Elbow", "Wrist_Pitch")]))
              for p in SHARED}
        shared0 = per_joint.replace(joints={n: per_joint.joint(n).replace(**st) for n in URDF_JOINTS})
        log("  shared arm model")
        shared, _ = shared_fit(shared0, dt_star, pool, iters=int(1.6 * iters), log=log)
        kp = shared.joint("Rotation").kp
        log("  Jaw at the shared kp")
        jaw0 = shared.with_joint("Jaw", kp=kp, kd=shared.joint("Rotation").kd, damping=shared.joint("Rotation").damping,
                                 armature=shared.joint("Rotation").armature)
        m = np.zeros((6, len(PARAMS)), bool)
        m[5, 1:] = True
        servo, _ = es_fit(jaw0, dt_star, pool, iters=max(iters * 4 // 5, 8), sigma0=0.3, seed=3, mask=m, log=log)
    hold = grip_hold.analyse(sc, servo.joint("Jaw").kp)
    js = {n: servo.joint(n).replace(effort=BAM_STALL) for n in URDF_JOINTS}
    I_f = _finger_inertia(sc)
    js["Jaw"] = servo.joint("Jaw").replace(effort=hold["effort_limit"], flex_stiffness=hold["flex_stiffness"],
                                           flex_damping=grip_hold.flex_damping(hold["flex_stiffness"], I_f))
    servo = servo.replace(joints=js)
    fit_s = time.time() - t0
    report = {"train": list(train), "test": list(test), "fit_seconds": round(fit_s, 1), "dead_time_s": round(dt_star, 5),
              "substeps": substeps, "dt": 1.0 / (30 * substeps),
              "dead_time_grid": {f"{k * 1000:.1f}ms": v.round(3).tolist() for k, v in coarse.items()},
              "es_by_dead_time": {f"{k * 1000:.1f}ms": results[k][1].round(3).tolist() for k in sorted(results)},
              "episodes": evaluate(sc, servo, tuple(train) + tuple(test), train, substeps),
              "per_joint_model": {"episodes": evaluate(sc, per_joint, tuple(train) + tuple(test), train, substeps),
                                  "params": {n: {p: round(getattr(per_joint.joint(n), p), 5) for p in PARAMS}
                                             for n in URDF_JOINTS}},
              "finger_inertia_kgm2": I_f}
    return {"servo": servo, "per_joint": per_joint, "hold": hold, "report": report}


def _finger_inertia(sc) -> float:
    """The moving jaw's inertia about the Jaw axis (kg m^2), from the MJCF body inertial."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(paths.MJCF))
    b = m.body("jaw").id
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = d.xipos[b] - d.xanchor[m.joint("Jaw").id]
    ax = d.xaxis[m.joint("Jaw").id]
    R = d.ximat[b].reshape(3, 3)
    I = R @ np.diag(m.body_inertia[b]) @ R.T
    perp = r - (r @ ax) * ax
    return float(ax @ I @ ax + m.body_mass[b] * (perp @ perp))


def kv_crosscheck(ds: str | None = None, episode: int = 0) -> list[dict]:
    """The understand phase's 'kv discrepancy', measured: mjlab's servo constants with kv
    as JOINT damping (outside the force clamp) vs as the ACTUATOR velocity bias
    (inside it), each under Euler and implicitfast, at dt 1/480 and 1/900 s, no dead time."""
    sc = load_scene(ds)
    data = episode_data(sc, episode)
    rows = []
    for substeps in (16, 30):
        for integ in ("euler", "implicitfast"):
            sim = ArmSim(sc, substeps)
            sim.m.opt.integrator = {"euler": sim.mj.mjtIntegrator.mjINT_EULER,
                                    "implicitfast": sim.mj.mjtIntegrator.mjINT_IMPLICITFAST}[integ]
            for where in ("joint", "actuator"):
                for force in (1.5, 3.0):
                    js = {n: JointServo(17.8, 2.0 if where == "actuator" else 0.0, 2.0 if where == "joint" else 0.0,
                                        0.1, 0.052, force) for n in URDF_JOINTS}
                    sv = ServoModel(js, 0.0, None)
                    q = sim.run(sv, data)
                    r = rmse_deg(q, data)
                    ls, lr = lag_frames(q, data["state"], data["goal"])
                    v = np.degrees(np.abs(np.gradient(q[:, :5], axis=0)) * data["fps"]).max()
                    rows.append({"dt": 1 / (30 * substeps), "integrator": integ, "kv_as": where, "forcerange": force,
                                 "rmse_deg": r[:5].round(2).tolist(), "mean_arm": round(float(r[:5].mean()), 2),
                                 "lag_sim": round(float(ls[:5].mean()), 2), "lag_real": round(float(lr[:5].mean()), 2),
                                 "max_speed_deg_s": round(float(v), 1)})
    return rows


def write_servo_json(res: dict, kv_rows: list | None = None, ds: str | None = None, path=None) -> str:
    """config/<ds>.servo.json from identify()'s result: {'actuator': ...}, every group sourced."""
    from . import grip_hold

    sc = load_scene(ds)
    p = path or paths.config_path(sc.ds, "servo")
    servo, rep, hold = res["servo"], res["report"], res["hold"]
    ep = rep["episodes"]
    tr = [e for e in ep if ep[e]["split"] == "train"]
    te = [e for e in ep if ep[e]["split"] == "held-out"]
    fmt = lambda es, i: "/".join(f"{ep[e]['rmse_deg'][i]:.2f}" for e in es)  # noqa: E731
    src = {}
    for i, n in enumerate(URDF_JOINTS):
        how = ("shared STS3215 kp/kd/damping/armature fitted on the 5 arm joints, frictionloss per joint"
               if n != "Jaw" else "kp = the shared arm kp (same servo; unloaded, the Jaw only fixes ratios), "
                                  "kd/damping/armature/frictionloss fitted on its free frames")
        src[n] = (f"fitted: servo_fit.identify, action -> observation.state on episodes {rep['train']}, contacts off; "
                  f"{how}; RMSE train {fmt(tr, i)} deg, held-out ep{'/'.join(map(str, rep['test']))} {fmt(te, i)} deg; "
                  + ("effort_limit fitted: grip_hold (the clamp seen on the ep2 A' and B holds)"
                     if n == "Jaw" else "effort_limit read: STS3215 full-duty stall servo.BAM_STALL (Rhoban BAM m1)"))
    f = hold["fit_pct"]
    src["Jaw_flex"] = (f"fitted: grip_hold hinge fit over {len(hold['plateaus'])} squeeze plateaus: encoder = "
                       f"{f['theta_c']:.2f} % - {f['s']:.3f} x min(err, {f['e_sat']:.2f} %), residual rms "
                       f"{f['rms']:.2f} % (max {f['max_abs']:.2f}); stiffness = kp / {f['s']:.3f}; damping ASSUMED "
                       "zeta 0.5 of the ~120 Hz flex mode")
    for e in ep:
        ep[e] = {"source": f"measured: servo_fit.evaluate ep{e} ({ep[e]['split']}), deg; Jaw on free frames", **ep[e]}
    pj = rep["per_joint_model"]
    for e in pj["episodes"]:
        pj["episodes"][e] = {"source": f"measured: per-joint model on ep{e}", **pj["episodes"][e]}
    for n in pj["params"]:
        pj["params"][n] = {"source": "fitted: per-joint ES (diagnostic, not used)", **pj["params"][n]}
    gap = grip_hold.contact_gap_mm(sc, hold["theta_c_rad"])
    top = {
        "source": (f"fitted: MUJOCO servo_fit.identify + grip_hold on {sc.ds} (mujoco 3.11, dt 1/{30 * rep['substeps']} s, "
                   "hard joint dry friction); dead time "
                   f"{servo.dead_time * 1000:.1f} ms common to all joints; scene hash {sc.hash} at fit time"),
        "stall_torque": {"source": ("read: Rhoban BAM STS3215 7.4 V m1 full-duty stall 0.97 x 7.4 V x 1.1776 N.m/A "
                                    "/ 2.4787 ohm; MEASURED: the recorded ep2 motion needs up to 3.16 N.m (Pitch, inverse "
                                    "dynamics), above the vendor ratings 1.91 (7.4 V) and 2.94 N.m (12 V). The gripper's "
                                    "identified clamp is only 33 % of it, not the 50 % Max_Torque_Limit: the firmware's "
                                    "overload protection (Overload_Torque 25) is the likely reason; unverified"),
                         "value": round(BAM_STALL, 4), "range": [1.91, 3.52],
                         "gripper_clamp_fraction": round(hold["effort_limit"] / BAM_STALL, 3)},
        "metrics": {"source": "measured: servo_fit.evaluate (arm-only model, action -> state)",
                    "fit_seconds": rep["fit_seconds"],
                    "dead_time_grid_rmse_deg": {"source": "measured: per-joint RMSE at the fit start point",
                                                **rep["dead_time_grid"]},
                    "per_joint_es_rmse_deg": {"source": "fitted: per-joint ES train RMSE at each dead time",
                                              **rep["es_by_dead_time"]},
                    "episodes": ep},
        "gripper_hold": {"source": "measured: grip_hold.plateaus on the used episodes; fitted: hinge model",
                         "theta_c_pct": round(f["theta_c"], 3), "theta_c_deg": round(math.degrees(hold["theta_c_rad"]), 3),
                         "slope_kp_over_k": round(f["s"], 4), "e_sat_pct": round(f["e_sat"], 3),
                         "rms_pct": round(f["rms"], 3), "max_abs_pct": round(f["max_abs"], 3),
                         "contact_gap_mm_at_theta_c": round(gap, 2),
                         "_gap": "the free finger-mesh gap at theta_c under the scene's gripper map: the cap width "
                                 "the hold data imply (a cross-check for CALIB's cap diameter)",
                         "plateaus": [{"source": "measured", **{k: (round(v, 3) if isinstance(v, float) else v)
                                                                for k, v in pl.items()}} for pl in hold["plateaus"]]},
        "per_joint_model": {"source": "fitted: per-joint ES (6 x 5 parameters); diagnostic only", **pj},
    }
    if kv_rows:
        top["kv_crosscheck"] = {"source": "measured: servo_fit.kv_crosscheck, mjlab constants kp 17.8 kv 2.0 "
                                          "armature 0.1 on ep0, no dead time",
                                "rows": [{"source": "measured", **r} for r in kv_rows]}
    cfg = {"_doc": "Identified STS3215 servo model (real2sim/mujoco/servo.py; SI units). Written by "
                   "./robot real2sim mujoco servo-fit; merged over <ds>.json and <ds>.calib.json by scene.load.",
           "actuator": servo.to_config(src, **top)}
    # never land an invalid layer: every track's scene.load would fail on it
    from ..scene import deep_merge, validate

    merged = {}
    for layer in ("", "calib"):
        lp = paths.config_path(sc.ds, layer)
        if lp.exists():
            merged = deep_merge(merged, json.loads(lp.read_text()))
    problems = validate(deep_merge(merged, json.loads(json.dumps(cfg))))
    if problems:
        raise ValueError("servo.json would make the scene invalid:\n  " + "\n  ".join(problems))
    p.write_text(json.dumps(cfg, indent=1) + "\n")
    return str(p)
