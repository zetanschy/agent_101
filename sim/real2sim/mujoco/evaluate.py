"""Score replays against the real episode: per-pick outcomes, physics honesty, fidelity.

    m = episode_metrics(scene, log)          # one replay's numbers (dict, JSON-ready)
    ens = ensemble(scene, episode, mode, K)  # K perturbed replays -> success rates
    ./robot real2sim mujoco eval             # every used episode -> mujoco/eval.json

PER PICK (scene picks: inclusive episode-local frame ranges of the REAL events)
    grasped      some frame from the real close start on has BOTH fingers on the cap
    lifted       the cap centre rose >= LIFT_MM above its resting height while grasped
    carried      grasped on >= 90 % of the frames from the real lift end to the real
                 release start (the transport)
    slip_mm      how far the cap moved in the gripper frame over that transport
    release      sim separation frame (last both-finger frame of the pick) and sim DROP
                 frame (the cap centre DROP_MM from where it was held, in the gripper
                 frame) vs the real drop (the wrist camera's teal fraction falls: scene
                 picks.drop[0]); drop_err_frames = sim - real
    in_mug       metrics.cap_in_mug at the episode's last frame
    jaw_hold     the sim ENCODER (Jaw horn) vs observation.state over the real hold
                 (real lift start .. release start), degrees and percent
    grip_force   finger-cap normal force during the transport (N, per finger)
    penetration  max / p99 finger-cap penetration over the frames the cap is touched

PER EPISODE: joint RMSE vs observation.state (lag 0 and best lag), grasp-site
fingertip error, outcome agreement (the set of caps in the mug vs the real set),
mug displacement, the largest force any non-finger arm geom put on an object (spurious
contact), all-object penetration, the table clearance of the arm at reset (a scene
that puts the real rest pose inside the table is not feasible), speed, and the caps'
pixel error against CALIB's real teal detections (observations/detections.npz) in
both cameras.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .. import episodes as eplib
from .. import metrics, paths
from ..transforms import quat_to_mat

LIFT_MM = 20.0
DROP_MM = 15.0
CARRY_FRAC = 0.9


def _cap_index(log, cap_id):
    names = list(log["objects"])
    caps = [n for n in names if n.startswith("cap_")]
    return names.index(f"cap_{cap_id}"), caps.index(f"cap_{cap_id}")


def _in_gripper(log, oi) -> np.ndarray:
    """(T, 3) the object's CENTRE in the gripper frame (m)."""
    links = list(log["links"])
    g = links.index("gripper")
    R = quat_to_mat(log["links_quat"][:, g])
    return np.einsum("tji,tj->ti", R, log["obj_pos"][:, oi] - log["links_pos"][:, g])


def pick_metrics(scene, log: dict, pk: dict, units) -> dict:
    oi, ci = _cap_index(log, pk["cap"])
    T = len(log["q"])
    f0 = int(log["frame"][0])
    loc = lambda f: int(np.clip(f - f0, 0, T - 1))  # noqa: E731  episode frame -> log row
    dims = scene.cap_dims()
    pert = log["meta"].get("perturbation") or {}
    dims.update({k: pert[f"cap_{k}"] for k in ("diameter", "height") if pert.get(f"cap_{k}") is not None})
    Rz = quat_to_mat(log["obj_quat"][:, oi])[:, :, 2]
    centre = log["obj_pos"][:, oi] + Rz * dims["height"] / 2
    rest = centre[0, 2]
    touch = log["x_touch"][:, ci]
    both = touch.all(1)
    a = loc(min(pk["close"][0], pk.get("first_close", pk["close"])[0]))
    lift_end, rel0, drop0 = loc(pk["lift"][1]), loc(pk["release"][0]), loc(pk["drop"][0])
    end = loc(pk["drop"][1] + 45)
    win = slice(a, end + 1)
    grasped = bool(both[win].any())
    held_h = np.where(both, centre[:, 2] - rest, -np.inf)
    lift_mm = float(np.max(held_h[win]) * 1000) if grasped else 0.0
    carry = both[lift_end:rel0 + 1]
    carried = bool(len(carry) and carry.mean() >= CARRY_FRAC)
    g = _in_gripper(log, oi)
    slip = float(np.linalg.norm(g[lift_end:rel0 + 1] - g[lift_end], axis=1).max() * 1000) if carried else None
    sep = drop = None
    if grasped:
        idx = np.flatnonzero(both[win]) + a
        sep = int(idx[-1])
        held = g[sep]
        far = np.flatnonzero(np.linalg.norm(g[sep:] - held, axis=1) > DROP_MM / 1000)
        drop = int(sep + far[0]) if len(far) else None
    mi = list(log["objects"]).index("mug")
    in_mug = bool(metrics.cap_in_mug(log["obj_pos"][-1, oi], log["obj_quat"][-1, oi], log["obj_pos"][-1, mi],
                                     log["obj_quat"][-1, mi], dims, scene.mug_dims()))
    hold = slice(loc(pk["lift"][0]), rel0 + 1)
    ep = eplib.load(scene.ds, include_excluded=True)[int(log["meta"]["episode"])]
    real_jaw = units.to_urdf(ep.state[f0:f0 + T])[:, 5]
    jaw_sim, jaw_real = float(np.degrees(log["q"][hold, 5]).mean()), float(np.degrees(real_jaw[hold]).mean())
    b = units.grip_b_deg_per_pct
    force = log["x_fc_force"][lift_end:rel0 + 1, ci]
    pen = log["x_pen_fc"][touch.any(1)]
    ps = metrics.penetration_summary(pen)
    # the grasp site (mjlab's fingertip midpoint) at the real blocked frame: sim vs FK(state)
    from ..kinematics import default as kin

    kb = loc(pk["close"][1])
    site_s, site_r = kin().grasp_site(log["q"][kb]), kin().grasp_site(units.to_urdf(ep.state[f0 + kb]))
    return {"cap": pk["cap"], "grasped": grasped, "lifted": bool(lift_mm >= LIFT_MM), "lift_mm": round(lift_mm, 1),
            "carried": carried, "slip_mm": None if slip is None else round(slip, 2),
            "sep_frame": None if sep is None else sep + f0, "drop_frame_sim": None if drop is None else drop + f0,
            "drop_frame_real": pk["drop"][0], "drop_err_frames": None if drop is None else drop + f0 - pk["drop"][0],
            "in_mug": in_mug, "jaw_hold_sim_deg": round(jaw_sim, 2), "jaw_hold_real_deg": round(jaw_real, 2),
            "jaw_hold_err_deg": round(jaw_sim - jaw_real, 2), "jaw_hold_err_pct": round((jaw_sim - jaw_real) / b, 2),
            "grip_force_N_median": np.round(np.median(force, 0), 2).tolist() if len(force) else None,
            "grip_force_N_max": np.round(force.max(0), 2).tolist() if len(force) else None,
            "pen_fc_max_mm": round(ps["max_mm"], 3), "pen_fc_p99_mm": round(ps["p99_mm"], 3),
            "fingertip_err_at_grasp_mm": round(float(np.linalg.norm(site_s - site_r) * 1000), 2),
            "fingertip_dz_at_grasp_mm": round(float((site_s[2] - site_r[2]) * 1000), 2)}


PHASES = {"rest": "rest", "held": "carried", "in_mug": "in_mug"}  # CALIB label state -> phase name


def _at(a: np.ndarray, t: float) -> np.ndarray:
    """a[t] for a fractional row t, linear between rows, clamped to the log."""
    t = float(np.clip(t, 0, len(a) - 1))
    i = min(int(np.floor(t)), len(a) - 2) if len(a) > 1 else 0
    w = t - i
    return a[i] if len(a) == 1 else (1 - w) * a[i] + w * a[i + 1]


def _quat_at(q: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0, len(q) - 1))
    i = min(int(np.floor(t)), len(q) - 2) if len(q) > 1 else 0
    if len(q) == 1:
        return q[0]
    q0, q1, w = q[i], q[i + 1], t - i
    q1 = q1 if q0 @ q1 >= 0 else -q1
    v = (1 - w) * q0 + w * q1
    return v / np.linalg.norm(v)


def cap_pixel_errors(scene, log: dict, latency: bool = True) -> dict:
    """Sim cap vs CALIB's real teal blobs, per camera, per cap, per phase, in the camera's
    distorted pixels: rest (the blobs CALIB labels 'rest' for that cap), carried
    ('held') and in_mug (front only: CALIB's blobs inside the rim after a drop carry no
    identity, so per frame they are paired with the sim caps whose real drop is over,
    closest pairs first, one blob per cap).

    The sim point is the cap's bounding-cylinder centre, projected with the scene camera
    (the wrist camera from the SIM gripper pose). latency=True evaluates the sim at the
    image's exposure time, row - cameras.<cam>.latency.frames (CALIB: images trail
    observation.state by 2.27 front / 1.42 wrist frames), linearly between frames.
    Identity comes from CALIB's labels (gated 12 px front / 90 px wrist around the
    CALIBRATED prediction, never the sim's), so a phase CALIB could not label is
    reported with its count: the front view sees a carried cap under the gripper."""
    from ..transforms import pose_to_T

    obs = paths.out_dir(scene.ds, "observations")
    if not (obs / "detections.npz").exists() or not (obs / "labels.npz").exists():
        return {"source": "none: no observations/detections.npz + labels.npz (./robot real2sim calib observe)"}
    with np.load(obs / "detections.npz", allow_pickle=False) as z:
        teal = z["teal"]
    with np.load(obs / "labels.npz", allow_pickle=False) as z:
        lab, lab_hash = z["labels"], str(z["scene_hash"])
    e = int(log["meta"]["episode"])
    f0, T = int(log["frame"][0]), len(log["q"])
    h = (log["meta"].get("perturbation") or {}).get("cap_height") or scene.cap_dims()["height"]
    names = [str(n) for n in log["objects"]]
    caps = [n[4:] for n in names if n.startswith("cap_")]
    centre = {c: log["obj_pos"][:, names.index(f"cap_{c}")]
              + quat_to_mat(log["obj_quat"][:, names.index(f"cap_{c}")])[:, :, 2] * h / 2 for c in caps}
    drop_end = {pk["cap"]: pk["drop"][1] for pk in scene.episode(e).get("picks", [])}
    g = list(log["links"]).index("gripper")
    out = {"source": "measured: sim cap centres projected with the scene cameras vs CALIB's labelled teal blobs "
                     "(observations/detections.npz + labels.npz)", "labels_scene_hash": lab_hash,
           "latency_corrected": latency}
    for cam_i, cam_name in enumerate(("front", "grip")):
        cam = scene.camera(cam_name)
        lag = float(scene["cameras"][cam_name].get("latency", {}).get("frames", 0.0)) if latency else 0.0
        sel = (teal["cam"] == cam_i) & (teal["episode"] == e) & (teal["frame"] >= f0) & (teal["frame"] < f0 + T)
        errs = {c: {p: [] for p in PHASES.values()} for c in caps}
        in_mug = {}  # frame -> [(pixel distance, blob, cap)] for the pairing below
        for j in np.flatnonzero(sel):
            st = str(lab["state"][j])
            if st not in PHASES or (st == "in_mug" and cam_name != "front"):
                continue
            fr = int(teal["frame"][j])
            t = fr - f0 - lag  # the sim at the image's exposure
            Tw = None if cam.parent == "world" else pose_to_T(_at(log["links_pos"][:, g], t), _quat_at(log["links_quat"][:, g], t))
            uv_real = np.array([teal["u"][j], teal["v"][j]], dtype=float)
            cand = ([c for c in caps if fr > drop_end.get(c, 10**9)] if st == "in_mug"
                    else [str(lab["cap"][j])] if str(lab["cap"][j]) in caps else [])
            for c in cand:
                uv = cam.project(_at(centre[c], t), Tw)
                if not np.isfinite(uv).all():
                    continue
                d = float(np.hypot(*(uv - uv_real)))
                if st == "in_mug":
                    in_mug.setdefault(fr, []).append((d, j, c))
                else:
                    errs[c][PHASES[st]].append(d)
        for pairs in in_mug.values():
            used_b, used_c = set(), set()
            for d, j, c in sorted(pairs):
                if j not in used_b and c not in used_c:
                    errs[c]["in_mug"].append(d)
                    used_b.add(j)
                    used_c.add(c)
        out[cam_name] = {f"cap_{c}": {p: ({"median_px": round(float(np.median(v)), 1),
                                           "p90_px": round(float(np.percentile(v, 90)), 1), "n": len(v)}
                                          if v else {"n": 0}) for p, v in errs[c].items()} for c in caps}
    return out


def table_clearance_mm(built, log: dict) -> float:
    """Lowest arm collision point above the table (mm) over the joint trajectory log['q']
    (every 3rd frame), by FK of the model's collision meshes. Negative = inside the table."""
    import mujoco

    m = built.model
    d = mujoco.MjData(m)
    arm = {m.body(n).id for n in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")}
    gs = [gi for gi in range(m.ngeom) if m.geom_contype[gi] and m.geom_bodyid[gi] in arm
          and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH]
    V = {gi: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[gi]]: m.mesh_vertadr[m.geom_dataid[gi]]
                         + m.mesh_vertnum[m.geom_dataid[gi]]] for gi in gs}
    tx, ty = built.info.get("table_tilt", (0.0, 0.0))
    low = np.inf
    for k in range(0, len(log["q"]), 3):
        d.qpos[built.qadr] = log["q"][k]
        mujoco.mj_kinematics(m, d)
        for gi in gs:
            w = V[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]
            low = min(low, float((w[:, 2] - tx * w[:, 0] - ty * w[:, 1]).min()))
    return float((low - built.info["table_z"]) * 1000)


def recorded_clearance_mm(scene, built, episode: int) -> float:
    """The same over the RECORDED state (observation.state through the scene units): if
    negative, the scene itself (table height + joint offsets) puts the real arm inside
    the table, so no replay of it can be physically faithful."""
    u = scene.units()
    ep = eplib.load(scene.ds, include_excluded=True)[episode]
    return table_clearance_mm(built, {"q": u.to_urdf(ep.state)})


def episode_metrics(scene, log: dict, built=None, pixels: bool = True) -> dict:
    u = scene.units()
    e = int(log["meta"]["episode"])
    ep = eplib.load(scene.ds, include_excluded=True)[e]
    f0, T = int(log["frame"][0]), len(log["q"])
    real = ep.state[f0:f0 + T]
    jr = metrics.joint_rmse_deg(log["q"], real, u)
    fe = metrics.fingertip_error_mm(log["q"], real, u)
    picks = [pick_metrics(scene, log, pk, u) for pk in scene.episode(e).get("picks", [])
             if f0 <= pk["close"][0] and pk["drop"][1] < f0 + T]
    names = list(log["objects"])
    mi = names.index("mug") if "mug" in names else None
    sim_in = sorted(p["cap"] for p in picks if p["in_mug"])
    real_in = sorted(scene.episode(e)["outcome"]["in_mug"])
    touched = log["x_touch"].any(-1).any(-1) if log["x_touch"].size else np.zeros(T, bool)
    out = {
        "episode": e, "mode": log["meta"]["mode"], "seed": log["meta"].get("seed"),
        "perturbation": log["meta"].get("perturbation"), "scene_hash": log["meta"]["scene_hash"],
        "physical": log["meta"].get("physical", True), "finite": log["meta"].get("finite", True),
        "picks": picks, "success": bool(picks) and all(p["in_mug"] for p in picks),
        "outcome": {"sim_in_mug": sim_in, "real_in_mug": real_in, "agree": sim_in == real_in},
        "joint_rmse_deg": np.round(jr["rmse"], 3).tolist(), "joint_rmse_at_lag_deg": np.round(jr["rmse_at_lag"], 3).tolist(),
        "lag_frames": jr["lag"], "joint_max_abs_deg": np.round(jr["max_abs"], 2).tolist(),
        "fingertip_err_mm": {"mean": round(fe["mean"], 2), "p95": round(fe["p95"], 2), "max": round(fe["max"], 2)},
        "pen_fc": {k: round(v, 3) if isinstance(v, float) else v
                   for k, v in metrics.penetration_summary(log["x_pen_fc"][touched]).items()},
        "pen_obj_max_mm": round(float(log["x_pen_obj"].max() * 1000), 3),
        "arm_object_force_max_N": round(float(log["x_arm_obj"].max()), 3),
        # servo torque actually applied (sampled at the frames) vs the model's clamps
        "actuator_force_max_Nm": np.round(np.abs(log["x_actuator_force"]).max(0), 3).tolist(),
        "effort_limit_Nm": [round(float(v), 3) for v in built.model.actuator_forcerange[built.act, 1]]
        if built is not None else None,
        "mug_moved_mm": round(float(np.linalg.norm(log["obj_pos"][-1, mi, :2] - log["obj_pos"][0, mi, :2]) * 1000), 2)
        if mi is not None else None,
        "wall_s": log["meta"].get("wall_s"), "sim_s": round((T - 1) / log["fps"], 2),
        "x_real_time": round((T - 1) / log["fps"] / max(log["meta"].get("wall_s") or 1e-9, 1e-9), 2),
    }
    if built is not None:
        out["arm_table_clearance_mm"] = {"sim_min": round(table_clearance_mm(built, log), 2),
                                         "recorded_min": round(recorded_clearance_mm(scene, built, e), 2)}
    if pixels:
        out["cap_pixels"] = cap_pixel_errors(scene, log)
        out["cap_pixels_no_latency"] = cap_pixel_errors(scene, log, latency=False)
    return out


def scene_checks(scene) -> dict:
    """Feasibility and consistency of the SCENE (not of a replay), for eval.json:

    arm_table_clearance_mm  lowest collision point of the recorded arm (observation.state
                            through the scene units) above the table, per episode;
                            negative = the scene puts the real arm inside the table
    cap_width_vs_hold       the free finger gap (contact band) at the jaw angle where the
                            real holds say the fingers first touch the cap (grip_hold
                            theta_c, an ENCODER percentage, mapped with the scene's
                            gripper map) vs the scene's cap diameter. A compliant jaw
                            releases the cap when the horn passes theta_c, so a cap
                            narrower than this gap lets go early and a wider one never
                            lets go on ep0's +2 % release."""
    from ..kinematics import default as kin
    from . import grip_hold
    from . import model as mdl

    out = {"source": "computed: evaluate.scene_checks", "arm_table_clearance_mm": {}}
    for e in scene.used_episodes():
        b = mdl.build(scene, e, cameras=())
        out["arm_table_clearance_mm"][str(e)] = round(recorded_clearance_mm(scene, b, e), 2)
    gh = scene.actuator().get("gripper_hold")
    if gh:
        u = scene.units()
        th = float(u.jaw_rad(gh["theta_c_pct"]))
        gap = kin().jaw_gap(th) * 1000
        out["cap_width_vs_hold"] = {"theta_c_pct": gh["theta_c_pct"], "theta_c_deg": round(float(np.degrees(th)), 3),
                                    "gap_at_theta_c_mm": round(gap, 2),
                                    "scene_cap_diameter_mm": round(scene.cap_dims()["diameter"] * 1000, 2),
                                    "diff_mm": round(scene.cap_dims()["diameter"] * 1000 - gap, 2)}
    return out


OPTIONAL_KEYS = ("table.tilt",)  # keys a scene may lack that --set may add (model.table_top)


def apply_overrides(sc, items):
    """--set PATH=VALUE (Python literals) on the merged scene, for sensitivity runs: a new
    Scene with its own hash, the changes kept in .overrides and in every log's meta."""
    import ast
    import copy

    from ..scene import Scene

    if not items:
        return sc
    data = copy.deepcopy(dict(sc))
    done = {}
    for it in items:
        key, _, val = (it.partition("=") if isinstance(it, str) else (it[0], None, it[1]))
        v = ast.literal_eval(val) if isinstance(it, str) else val
        d = data
        parts = key.split(".")
        for p in parts[:-1]:
            d = d[int(p)] if isinstance(d, list) else d[p]
        if parts[-1] not in d and not isinstance(d, list) and key not in OPTIONAL_KEYS:
            raise KeyError(f"--set {key}: no such scene key")
        done[key] = {"was": d[int(parts[-1])] if isinstance(d, list) else d.get(parts[-1]), "now": v}
        if isinstance(d, list):
            d[int(parts[-1])] = v
        else:
            d[parts[-1]] = v
    out = Scene(data, sc.ds, sc.layers, sc.origin)
    out.overrides = done
    return out



# --- ensembles ------------------------------------------------------------------------

def draw_perturbation(scene, episode: int, rng, sigma: dict | None = None) -> "object":
    """Scene uncertainty -> one Perturbation: cap xy ~ N(0, cap sigma), table z ~ N(0,
    table sigma) (clipped to its range), cap diameter ~ N(d, sigma[0]) and height ~
    N(h, sigma[1]) when the scene fits them (CALIB: objects.cap.sigma = [D, h]; the
    diameter clipped to diameter_range), else diameter ~ U(diameter_range) (the
    provisional plausible range, 23-32 mm: drawing from it on the fitted scene would
    swamp the fitted 0.89 mm), frictions ~ U(MATERIAL_RANGES) as scale factors, cap mass
    ~ U(2, 3) g. `sigma` replaces the scene's uncertainties for a what-if ensemble:
    {'cap_xy': m, 'table_z': m, 'cap_d': (lo, hi) m, 'cap_h': m}."""
    from .model import MATERIALS, Perturbation

    sg = sigma or {}
    ep = scene.episode(episode)
    dxy = {c["id"]: tuple(rng.normal(0.0, float(sg.get("cap_xy", c.get("sigma", 0.005))), 2)) for c in ep["caps"]}
    t = scene["table"]
    dz = float(rng.normal(0.0, float(sg.get("table_z", t.get("sigma", 0.002)))))
    if "range" in t and "table_z" not in sg:
        dz = float(np.clip(scene.table_z() + dz, *t["range"]) - scene.table_z())
    cap = scene["objects"]["cap"]
    s = cap.get("sigma")
    s_d, s_h = (tuple(s) + (None,))[:2] if isinstance(s, (list, tuple)) else (s, None)
    if "cap_d" in sg:
        D = float(rng.uniform(*sg["cap_d"]))
    elif s_d is not None:
        D = float(rng.normal(cap["diameter"], float(s_d)))
        D = float(np.clip(D, *cap["diameter_range"])) if "diameter_range" in cap else D
    elif "diameter_range" in cap:
        D = float(rng.uniform(*cap["diameter_range"]))
    else:
        D = float(rng.normal(cap["diameter"], 0.0005))
    s_h = sg.get("cap_h", s_h)
    H = float(rng.normal(cap["height"], float(s_h))) if s_h else None
    fs = {k: float(rng.uniform(*MATERIAL_RANGES[k])) / MATERIALS[k][0] for k in MATERIAL_RANGES}
    return Perturbation(cap_dxy=dxy, table_dz=dz, cap_diameter=D, cap_height=H, cap_mass=float(rng.uniform(0.002, 0.003)),
                        friction_scale=fs)


# ESTIMATED dry-friction ranges (handbook values for these material pairs; see model.MATERIALS)
MATERIAL_RANGES = {"cap": (0.2, 0.4), "finger": (0.3, 0.5), "mat": (0.4, 0.9), "mug": (0.2, 0.4)}


def _run_one(args):
    ds, episode, mode, seed, servo_cfg, phys_kw, pert, overrides, placement, sigma = args
    from ..scene import load
    from . import model as mdl
    from .replay import Config, run
    from .servo import ServoModel

    sc = apply_overrides(load(ds), list(overrides.items()))
    servo = ServoModel.from_config(servo_cfg) if servo_cfg else ServoModel.from_scene(sc)
    p = pert if pert is not None else draw_perturbation(sc, episode, np.random.default_rng(seed), sigma)
    cfg = Config(mode=mode, physics=mdl.Physics(**phys_kw), perturb=p, seed=seed, placement=placement)
    try:
        res = run(sc, episode, cfg, servo, warn=lambda s: None)
    except Exception as e:  # one broken seed must not lose the ensemble: it counts as a failure
        return {"episode": episode, "mode": mode, "seed": seed, "perturbation": p.as_dict(), "error": repr(e),
                "success": False, "outcome": {"agree": False},
                "picks": [{"cap": pk["cap"], "grasped": False, "lifted": False, "carried": False, "in_mug": False,
                           "drop_err_frames": None, "jaw_hold_err_deg": float("nan"), "pen_fc_max_mm": 0.0}
                          for pk in sc.episode(episode)["picks"]],
                "pen_fc": {"max_mm": 0.0, "p99_mm": 0.0}, "mug_moved_mm": 0.0}, None
    return episode_metrics(sc, res.log, res.built, pixels=False), res.log


def ensemble(scene, episode: int, mode: str, K: int, workers: int = 6, servo_cfg: dict | None = None,
             phys_kw: dict | None = None, seed0: int = 1000, write_logs: str | None = None, log=print,
             placement: str = "config", sigma: dict | None = None) -> dict:
    """K perturbed replays in a process pool. Success rates per pick and per episode.
    The scene's --set overrides (scene.overrides) and the placement carry over."""
    from concurrent.futures import ProcessPoolExecutor

    from .. import poselog

    ov = {k: v["now"] for k, v in getattr(scene, "overrides", {}).items()}
    args = [(scene.ds, episode, mode, seed0 + i, servo_cfg, phys_kw or {}, None, ov, placement, sigma)
            for i in range(K)]
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, (m, lg) in enumerate(pool.map(_run_one, args)):
            rows.append(m)
            if write_logs and lg is not None:
                poselog.write(f"{write_logs}/ep{episode}_{mode}_seed{seed0 + i}.npz", lg)
            log(f"    seed {seed0 + i}: success {m['success']} picks "
                + " ".join(f"{p['cap']}:{'G' if p['grasped'] else '-'}{'L' if p['lifted'] else '-'}"
                           f"{'C' if p['carried'] else '-'}{'M' if p['in_mug'] else '-'}" for p in m["picks"]))
    caps = [p["cap"] for p in rows[0]["picks"]]
    per_pick = {}
    for j, c in enumerate(caps):
        ps = [r["picks"][j] for r in rows]
        de = [p["drop_err_frames"] for p in ps if p["drop_err_frames"] is not None]
        per_pick[c] = {k: round(float(np.mean([p[k] for p in ps])), 3) for k in ("grasped", "lifted", "carried", "in_mug")}
        per_pick[c]["drop_err_frames_median"] = float(np.median(de)) if de else None
        per_pick[c]["jaw_hold_err_deg_median"] = float(np.nanmedian([p["jaw_hold_err_deg"] for p in ps]))
        per_pick[c]["pen_fc_max_mm"] = float(max(p["pen_fc_max_mm"] for p in ps))
    return {"n": K, "mode": mode, "seed0": seed0, "wall_s": round(time.time() - t0, 1), "placement": placement,
            "sigma_override": sigma or {},
            "scene_overrides": getattr(scene, "overrides", {}),
            "success_rate": round(float(np.mean([r["success"] for r in rows])), 3),
            "outcome_agreement": round(float(np.mean([r["outcome"]["agree"] for r in rows])), 3),
            "per_pick": per_pick,
            "pen_fc_max_mm": float(max(r["pen_fc"]["max_mm"] for r in rows)),
            "pen_fc_p99_mm_median": float(np.median([r["pen_fc"]["p99_mm"] for r in rows])),
            "mug_moved_mm_max": float(max(r["mug_moved_mm"] for r in rows)),
            "errors": sum("error" in r for r in rows),
            "seeds": [{"seed": r["seed"], "success": r["success"], "perturbation": r["perturbation"], "error": r.get("error"),
                       "picks": [{k: p[k] for k in ("cap", "grasped", "lifted", "carried", "in_mug", "drop_err_frames")}
                                 for p in r["picks"]]} for r in rows]}


def write_json(path, obj) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)
