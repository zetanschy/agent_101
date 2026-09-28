"""eval.json for an Isaac replay: fidelity against the dataset, and physics honesty.

    ./robot real2sim isaac eval <run dir>... [--no-penetration] [--ensemble-stride N]

Host python3 (no Kit). Reads <run>/poselog.npz (+ ensemble/env_*.npz) and run.json and
rebuilds the scene the run used (scene config + the run's --set overrides); the scene
hash must match the log's, else the config changed after the run and eval says so.

SHARED KEYS. Every key the MuJoCo track's evaluate.episode_metrics writes is computed BY
that function on this pose log (the replay logs its x_touch / x_fc_force / x_pen_fc /
x_pen_obj / x_arm_obj with the same definitions, contact.py), so the two engines'
numbers mean the same thing: per pick grasped / lifted / carried / slip / separation and
drop frames vs the real drop / in_mug / the Jaw ENCODER over the real hold vs
observation.state / grip force / finger-cap penetration; per episode joint RMSE at lag
0 and best lag, fingertip error, outcome agreement, mug displacement, spurious arm-
object force, and the caps' pixel error in both cameras against CALIB's detections.

ISAAC KEYS (under "isaac"):
    penetration_mm    offline (penetration.py: logged poses + meshes, exact geometry) for
                      fingers-cap, cap-table, cap-mug, fingers-table, fingers-mug; and
                      PhysX's own finger-cap separations, any physics step and at the
                      logged instant
    picks             the finger angle next to the encoder over the hold, the servo torque
                      at the horn vs its limit, finger forces vs limit / lever arm, and the
                      cap's rotation in the gripper frame
    table_contact     force of every robot link on the table (the bench-geometry test)
    link_frames_fk_mm logged PhysX link poses vs real2sim.kinematics FK

ENSEMBLE (under "ensemble", runs with --num-envs > 1): mujoco.evaluate.ensemble's block,
same keys and statistics, over the perturbed envs 1..K-1 (env 0 is "nominal"); each row
carries its drawn scene and, with --perturb mujoco, its MuJoCo seed (ensemble_summary).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .. import episodes as eplib
from .. import kinematics, metrics, objects, poselog, scene as r2s_scene, units as U
from ..transforms import quat_to_mat
from . import params as P


def scene_of_run(info: dict, ds=None):
    sc = r2s_scene.load(ds or info.get("dataset"))
    items = [f"{k}={v['now']!r}" for k, v in (info.get("scene_overrides") or {}).items()]
    return P.scene_with_overrides(sc, items)[0]


def _q_geom(lg) -> np.ndarray:
    """Joint angles that place the links: the arm, and the FINGER for the Jaw."""
    q = np.array(lg["q"], dtype=float)
    if "x_finger" in lg:
        q[:, U.GRIPPER] = lg["x_finger"]
    return q


def isaac_picks(lg, sc, ep_cfg) -> list[dict]:
    """Per pick, over the real hold (real lift start .. release start, the MuJoCo track's
    window): encoder vs finger angle, servo torque at the horn, finger forces, rotation."""
    fr = lg["frame"]
    f0, T = int(fr[0]), len(fr)
    loc = lambda f: int(np.clip(f - f0, 0, T - 1))  # noqa: E731
    objs = list(lg["objects"])
    caps = [n for n in objs if n.startswith("cap_")]
    links = list(lg["meta"].get("contact_links", []))
    servo = lg["meta"]["servo"]
    eff = float(servo["joints"]["Jaw"]["effort_limit"]) if "joints" in servo else float(servo["effort_limit"]["gripper"])
    li_g, li_j = list(lg["links"]).index("gripper"), list(lg["links"]).index("jaw")
    out = []
    for pk in ep_cfg.get("picks", []):
        name = f"cap_{pk['cap']}"
        if name not in caps or not (f0 <= pk["close"][0] and pk["release"][0] < f0 + T):
            continue
        j, c = objs.index(name), caps.index(name)
        w = slice(loc(pk["lift"][0]), loc(pk["release"][0]) + 1)
        centre = objects.cap_centre(lg["obj_pos"][w, j], lg["obj_quat"][w, j], sc.cap_dims())
        R = quat_to_mat(lg["links_quat"][w, li_j])
        v = centre - lg["links_pos"][w, li_j]
        lever = np.linalg.norm(v - np.einsum("ti,ti->t", v, R[:, :, 2])[:, None] * R[:, :, 2], axis=1)
        Rg = quat_to_mat(lg["links_quat"][w, li_g])
        Rc = np.einsum("tji,tjk->tik", Rg, quat_to_mat(lg["obj_quat"][w, j]))
        rot = [math.degrees(math.acos(np.clip((np.trace(Rc[0].T @ r) - 1) / 2, -1, 1))) for r in Rc]
        rec = {"cap": pk["cap"], "hold_frames": [int(fr[w][0]), int(fr[w][-1])],
               "jaw_encoder_deg": float(np.degrees(lg["q"][w, U.GRIPPER]).mean()),
               "jaw_finger_deg": float(np.degrees(lg.get("x_finger", lg["q"][:, U.GRIPPER])[w]).mean()),
               "servo_torque_Nm": {"median": float(np.median(np.abs(lg["x_tau"][w, U.GRIPPER]))),
                                   "max": float(np.abs(lg["x_tau"][w, U.GRIPPER]).max()), "limit": eff},
               "lever_arm_mm": float(np.median(lever) * 1000),
               "squeeze_bound_N": float(eff / np.median(lever)),
               "cap_rotation_in_gripper_deg": float(max(rot))}
        if "x_robot_max_force" in lg and links:
            fmax = lg["x_robot_max_force"][w][:, [links.index("gripper"), links.index("jaw")], c]
            rec["finger_force_max_N"] = np.round(fmax.max(0), 2).tolist()  # fixed, moving; any physics step
        out.append(rec)
    return out


def isaac_extras(lg, sc, kin, penetration: bool = True) -> dict:
    ep_cfg = sc.episode(int(lg["meta"]["episode"]))
    out = {"picks": isaac_picks(lg, sc, ep_cfg)}
    pen = {}
    if "x_pen_fc" in lg:
        touched = lg["x_touch"].any(-1).any(-1) if lg["x_touch"].size else np.zeros(len(lg["q"]), bool)
        pen["physx_finger_cap_any_step"] = metrics.penetration_summary(lg["x_pen_fc"][touched])
    if "x_robot_last_sep" in lg:
        links = list(lg["meta"]["contact_links"])
        C = len([n for n in lg["objects"] if str(n).startswith("cap_")])
        s = lg["x_robot_last_sep"][:, [links.index("gripper"), links.index("jaw")], :C]
        pen["physx_finger_cap_logged_instant"] = metrics.penetration_summary(-s[s < 0])
    if penetration:
        from . import penetration as PEN

        pr = lg["meta"].get("perturbation") or {}
        geo = dict(lg)
        off = PEN.run(geo, sc, kin, cap_scale=pr.get("cap_scale", 1.0), table_dz=pr.get("table_dz", 0.0),
                      cap_zscale=pr.get("cap_zscale", 1.0))
        for k, v in off.items():
            v = np.asarray(v).ravel()
            pen[f"offline_{k}"] = {**metrics.penetration_summary(v[v > 0]), "frames_over_1mm": int((v > 1e-3).sum())}
    out["penetration_mm"] = pen
    if "x_robot_max_force" in lg:
        links = list(lg["meta"]["contact_links"])
        ft = lg["x_robot_max_force"][:, :, -1]  # the last contact filter is the table
        fr = lg["frame"]
        out["table_contact"] = {
            l: {"max_N": round(float(ft[:, i].max()), 3), "frames_gt_0.5N": int((ft[:, i] > 0.5).sum()),
                "first_frames": [int(x) for x in fr[ft[:, i] > 0.5][:3]]}
            for i, l in enumerate(links) if ft[:, i].max() > 0}
    idx = np.arange(0, len(lg["frame"]), 10)
    fk_pos, _ = kin.link_poses_batch(_q_geom(lg)[idx], tuple(lg["links"]))
    out["link_frames_fk_mm"] = float(np.linalg.norm(fk_pos - lg["links_pos"][idx], axis=-1).max() * 1000)
    ep = eplib.load(sc.ds, include_excluded=True)[int(lg["meta"]["episode"])]
    real_jaw = np.degrees(sc.units().to_urdf(ep.state[lg["frame"]])[:, U.GRIPPER])
    finger = np.degrees(lg["x_finger"]) if "x_finger" in lg else np.degrees(lg["q"][:, U.GRIPPER])
    out["jaw_finger_rmse_deg"] = float(np.sqrt(np.mean((finger - real_jaw) ** 2)))
    return out


def evaluate_log(lg, sc, kin, penetration: bool = True, pixels: bool = True) -> dict:
    from ..mujoco.evaluate import episode_metrics

    shared = episode_metrics(sc, lg, built=None, pixels=pixels)
    shared["joint_names"] = list(U.URDF_JOINTS)
    shared["isaac"] = isaac_extras(lg, sc, kin, penetration)
    return shared


def evaluate(run: Path, penetration: bool = True, ensemble_stride: int = 1) -> dict:
    run = Path(run)
    info = json.loads((run / "run.json").read_text())
    lg = poselog.read(run / "poselog.npz")
    sc = scene_of_run(info)
    kin = kinematics.default()
    res = {"run": str(run), "engine": "isaac", "mode": info["mode"], "episode": info["episode"],
           "scene_hash": sc.hash, "log_scene_hash": lg["meta"]["scene_hash"],
           "scene_hash_matches": sc.hash == lg["meta"]["scene_hash"], "scene_layers": info.get("scene_layers"),
           "scene_overrides": info.get("scene_overrides"), "placement": info.get("placement"),
           "hz": info.get("hz"), "flex": info.get("flex"), "timings": info.get("timings"),
           "nominal": evaluate_log(lg, sc, kin, penetration)}
    ens = sorted((run / "ensemble").glob("env_*.npz"))[::max(1, ensemble_stride)]
    if ens:
        rows = [evaluate_log(poselog.read(p), sc, kin, penetration, pixels=False) | {"env": p.stem} for p in ens]
        res["ensemble"] = ensemble_summary(rows, info)
    (run / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    return res


PICK_KEYS = ("cap", "grasped", "lifted", "carried", "in_mug", "drop_err_frames", "jaw_hold_err_deg", "slip_mm",
             "pen_fc_max_mm")


def ensemble_summary(rows: list[dict], info: dict) -> dict:
    """The MuJoCo track's ensemble block (mujoco.evaluate.ensemble: same keys, same
    statistics) over the PERTURBED envs 1..K-1 only; env 0, the nominal replay, is
    eval.json's "nominal". With --perturb mujoco every env carries its MuJoCo seed
    (perturbation.mujoco_seed), so the two engines' rows pair up seed by seed."""
    caps = [pk["cap"] for pk in rows[0]["picks"]]
    per_pick = {}
    for j, c in enumerate(caps):
        ps = [r["picks"][j] for r in rows]
        de = [p["drop_err_frames"] for p in ps if p["drop_err_frames"] is not None]
        per_pick[c] = {k: round(float(np.mean([p[k] for p in ps])), 3) for k in ("grasped", "lifted", "carried", "in_mug")}
        per_pick[c]["drop_err_frames_median"] = float(np.median(de)) if de else None
        per_pick[c]["jaw_hold_err_deg_median"] = float(np.nanmedian([p["jaw_hold_err_deg"] for p in ps]))
        per_pick[c]["pen_fc_max_mm"] = float(max(p["pen_fc_max_mm"] for p in ps))
    off = [r["isaac"]["penetration_mm"].get("offline_finger_cap", {}).get("max_mm", 0.0) for r in rows]
    pert = [r.get("perturbation") or {} for r in rows]
    return {
        "n": len(rows), "mode": info.get("mode"), "draws": info["args"].get("perturb"),
        "seed0": min((p["mujoco_seed"] for p in pert if "mujoco_seed" in p), default=info["args"].get("seed")),
        "loop_s": (info.get("timings") or {}).get("loop_s"), "placement": info.get("placement"),
        "scene_overrides": info.get("scene_overrides") or {},
        "success_rate": round(float(np.mean([r["success"] for r in rows])), 3),
        "outcome_agreement": round(float(np.mean([r["outcome"]["agree"] for r in rows])), 3),
        "per_pick": per_pick,
        "pen_fc_max_mm": float(max(r["pen_fc"]["max_mm"] for r in rows)),
        "pen_fc_p99_mm_median": float(np.median([r["pen_fc"]["p99_mm"] for r in rows])),
        "offline_finger_cap_max_mm": float(max(off)), "offline_finger_cap_max_mm_median": float(np.median(off)),
        "mug_moved_mm_max": float(max(r["mug_moved_mm"] or 0.0 for r in rows)),
        "errors": int(sum(not r["finite"] for r in rows)),
        "seeds": [{"env": r["env"], "seed": p.get("mujoco_seed"), "success": r["success"], "finite": r["finite"],
                   "perturbation": p, "picks": [{k: pk[k] for k in PICK_KEYS} for pk in r["picks"]]}
                  for r, p in zip(rows, pert)],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--no-penetration", action="store_true")
    p.add_argument("--ensemble-stride", type=int, default=1)
    a = p.parse_args(argv)
    for run in a.runs:
        r = evaluate(run, not a.no_penetration, a.ensemble_stride)
        n = r["nominal"]
        rm = n["joint_rmse_deg"]
        pen = n["isaac"]["penetration_mm"].get("offline_finger_cap", {}).get("max_mm", float("nan"))
        print(f"{run.name}: joint RMSE {' '.join(f'{v:.2f}' for v in rm)} deg (lag {n['lag_frames']}), "
              f"tip {n['fingertip_err_mm']['mean']:.1f} mm, in mug sim {n['outcome']['sim_in_mug']} real "
              f"{n['outcome']['real_in_mug']}, finger-cap pen PhysX {n['pen_fc']['max_mm']:.2f} / offline {pen:.2f} mm"
              + (f", ensemble success {r['ensemble']['success_rate']} (n={r['ensemble']['n']} perturbed) per pick "
               + " ".join(f"{c}:{v['in_mug']}" for c, v in r['ensemble']['per_pick'].items()) if "ensemble" in r else ""))


if __name__ == "__main__":
    main()
