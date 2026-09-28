"""Command line of the MuJoCo track (run.sh calls `python -m real2sim.mujoco.cli <cmd>`).

    servo-fit [--iters N] [--workers W] [--dry-run]
    replay --episode N [--mode action|track|kinematic] [--seed S] [--ensemble K] [--render] [--gui]
           [--frames A:B] [--table-dz M] [--cap-diameter M] [--every N] [--tag T]
    eval [--episodes 0,1,2] [--modes action,track] [--ensemble K] [--workers W] [--render] [--tag T]
    render --log LOG.npz [--every N] [--no-real]
    view --log LOG.npz
    info
  replay/eval, what-if diagnostics (all recorded in every log's meta and in eval.json; each
  needs --tag, so the untagged logs and eval.json stay the nominal replays):
    --placement config|grasp   cap xy from the scene (default) or the antipodal grasp (model.grasp_placements)
    --set PATH=VALUE           override a merged-scene value (evaluate.apply_overrides), repeatable
    --sigma cap_xy=M,table_z=M,cap_d=LO:HI,cap_h=M   ensemble uncertainties instead of the scene's
    --cap-shape cylinder|taper, --impratio, --substeps, --condim, --fingers, --no-flex   physics variants

Outputs under sim/outputs/real2sim/<ds>/mujoco/: logs/ (pose logs), video/ (mp4),
eval[_<tag>].json, servo_fit.json. --seed 0 is the scene as configured; S > 0 draws a
perturbation from the scene's uncertainties (evaluate.draw_perturbation).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .. import paths, poselog
from ..scene import load as load_scene


def _out(ds, *parts, create=True) -> Path:
    p = paths.out_dir(ds, "mujoco", *parts[:-1], create=create) / parts[-1] if parts else paths.out_dir(ds, "mujoco")
    return p


def _say(*a):
    print(*a, flush=True)


def _scene(a):
    """The merged scene, with --set overrides applied."""
    from .evaluate import apply_overrides

    return apply_overrides(load_scene(a.ds), getattr(a, "set", None) or [])


def cmd_servo_fit(a) -> int:
    from . import servo_fit as F
    from .servo import describe

    t0 = time.time()
    _say("kv cross-check (mjlab constants, ep0) ...")
    kv = F.kv_crosscheck(a.ds)
    for r in kv:
        _say(f"  dt {r['dt']:.5f} {r['integrator']:12s} kv as {r['kv_as']:8s} force {r['forcerange']}: mean arm RMSE "
             f"{r['mean_arm']:.2f} deg, lag {r['lag_sim']:.1f} (real {r['lag_real']:.1f}) frames, peak "
             f"{r['max_speed_deg_s']:.0f} deg/s")
    _say(f"identify (ES, {a.workers} workers, {a.iters} iterations) ...")
    res = F.identify(a.ds, iters=a.iters, workers=a.workers, log=_say if a.verbose else (lambda s: None))
    _say(describe(res["servo"]))
    for e, v in res["report"]["episodes"].items():
        _say(f"  ep{e} {v['split']:8s} RMSE " + " ".join(f"{x:.2f}" for x in v["rmse_deg"]) + " deg")
    h = res["hold"]
    _say(f"  gripper holds: theta_c {h['fit_pct']['theta_c']:.2f} %, kp/k {h['fit_pct']['s']:.3f}, "
         f"e_sat {h['fit_pct']['e_sat']:.2f} % -> flex {h['flex_stiffness']:.2f} N.m/rad, "
         f"gripper clamp {h['effort_limit']:.3f} N.m")
    rep = _out(a.ds, "servo_fit.json")
    from .evaluate import write_json

    write_json(rep, {"report": res["report"], "hold": res["hold"], "kv_crosscheck": kv})
    if a.dry_run:
        _say(f"dry run: report {rep}; config not written")
        return 0
    p = F.write_servo_json(res, kv, a.ds)
    _say(f"wrote {p}  (report {rep}; {time.time() - t0:.0f} s)")
    return 0


def _sigma(a) -> dict | None:
    """--sigma cap_xy=0.003,table_z=0.002,cap_d=0.026:0.028 (m) -> draw_perturbation's sigma."""
    if not getattr(a, "sigma", None):
        return None
    out = {}
    for item in a.sigma.split(","):
        k, _, v = item.partition("=")
        out[k] = tuple(float(x) for x in v.split(":")) if ":" in v else float(v)
    return out


def _perturb(a, sc, episode):
    from . import model as mdl
    from .evaluate import draw_perturbation

    p = draw_perturbation(sc, episode, np.random.default_rng(a.seed), _sigma(a)) if a.seed else mdl.Perturbation()
    kw = {}
    if a.table_dz is not None:
        kw["table_dz"] = a.table_dz
    if a.cap_diameter is not None:
        kw["cap_diameter"] = a.cap_diameter
    if kw:
        from dataclasses import replace

        p = replace(p, **kw)
    return p


def _physics(a):
    from . import model as mdl

    kw = {}
    for k in ("impratio", "substeps", "condim", "fingers", "cap_shape"):
        v = getattr(a, k, None)
        if v is not None:
            kw[k] = v
    if getattr(a, "no_flex", False):
        kw["flex"] = False
    return mdl.Physics(**kw)


WHATIF_OPTS = ("set", "sigma", "impratio", "substeps", "condim", "fingers", "cap_shape", "no_flex", "table_dz",
               "cap_diameter", "frames")


def _need_tag(a) -> None:
    """The untagged logs and eval.json are THE nominal replays other tracks read (the Blender
    track renders logs/ep0_action_seed0.npz): a what-if run must not overwrite them."""
    used = [k for k in WHATIF_OPTS if getattr(a, k, None) not in (None, False, [], "")]
    if getattr(a, "placement", "config") != "config":
        used.append("placement")
    if used and not a.tag:
        raise SystemExit(f"error: a what-if run ({', '.join('--' + k.replace('_', '-') for k in used)}) needs --tag T; "
                         "the untagged outputs are the nominal replays")


def cmd_replay(a) -> int:
    from . import evaluate as ev
    from .replay import Config, run

    _need_tag(a)
    sc = _scene(a)
    sfx = f"_{a.tag}" if a.tag else ""
    if a.ensemble:
        _say(f"ensemble: episode {a.episode}, mode {a.mode}, {a.ensemble} seeds")
        res = ev.ensemble(sc, a.episode, a.mode, a.ensemble, workers=a.workers, phys_kw=_phys_kw(a), log=_say,
                          write_logs=str(_out(a.ds, "logs", *([a.tag] if a.tag else []), "x").parent),
                          placement=a.placement, sigma=_sigma(a))
        p = _out(a.ds, f"ensemble_ep{a.episode}_{a.mode}{sfx}.json")
        ev.write_json(p, res)
        _say(f"success rate {res['success_rate']:.2f}, per pick {json.dumps(res['per_pick'])}\nwrote {p}")
        return 0
    frames = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    cfg = Config(mode=a.mode, physics=_physics(a), perturb=_perturb(a, sc, a.episode), seed=a.seed, frames=frames,
                 placement=a.placement)
    _say(f"scene {sc.ds} hash {sc.hash} layers {[p.name for p in sc.layers]}")
    res = run(sc, a.episode, cfg)
    log = res.log
    tag = f"ep{a.episode}_{a.mode}_seed{a.seed}{sfx}"
    lp = _out(a.ds, "logs", f"{tag}.npz")
    poselog.write(lp, log)
    m = ev.episode_metrics(sc, log, res.built)
    ev.write_json(_out(a.ds, "logs", f"{tag}.metrics.json"), m)
    _say(f"{a.mode}: {res.sim_s:.1f} s simulated in {res.wall_s:.1f} s ({res.sim_s / max(res.wall_s, 1e-9):.1f}x real time)")
    _print_metrics(m)
    _say(f"pose log {lp}")
    if a.render:
        from .render import render_log

        v = render_log(sc, log, str(_out(a.ds, "video", f"{tag}.mp4")), res.built, every=a.every)
        _say(f"video {v['path']} ({v['frames']} frames, {v['seconds']} s)")
    if a.gui:
        from .viewer import play_log

        play_log(res.built, log)
    return 0


def _phys_kw(a) -> dict:
    p = _physics(a)
    from dataclasses import asdict

    return {k: v for k, v in asdict(p).items() if k in ("impratio", "substeps", "condim", "fingers", "flex", "cap_shape")}


def _print_metrics(m: dict) -> None:
    _say(f"  success {m['success']}  outcome sim {m['outcome']['sim_in_mug']} real {m['outcome']['real_in_mug']}")
    for p in m["picks"]:
        _say(f"  cap {p['cap']}: grasped {p['grasped']} lifted {p['lifted']} ({p['lift_mm']} mm) carried {p['carried']} "
             f"slip {p['slip_mm']} mm | drop sim {p['drop_frame_sim']} real {p['drop_frame_real']} "
             f"(err {p['drop_err_frames']}) | in mug {p['in_mug']} | jaw hold sim {p['jaw_hold_sim_deg']} real "
             f"{p['jaw_hold_real_deg']} deg | grip {p['grip_force_N_median']} N | pen max {p['pen_fc_max_mm']} "
             f"p99 {p['pen_fc_p99_mm']} mm | fingertip at grasp {p['fingertip_err_at_grasp_mm']} mm "
             f"(dz {p['fingertip_dz_at_grasp_mm']})")
    _say(f"  joint RMSE deg {m['joint_rmse_deg']} (at lag {m['lag_frames']}: {m['joint_rmse_at_lag_deg']})")
    _say(f"  fingertip err mm {m['fingertip_err_mm']}  pen_fc {m['pen_fc']}  mug moved {m['mug_moved_mm']} mm  "
         f"arm-object force max {m['arm_object_force_max_N']} N")
    if "arm_table_clearance_mm" in m:
        _say(f"  arm-table clearance mm {m['arm_table_clearance_mm']}")
    cp = m.get("cap_pixels", {})
    for cam in ("front", "grip"):
        if cam in cp:
            _say(f"  cap px {cam} (median, n; latency-corrected): " + "  ".join(
                f"{k} " + " ".join(f"{ph} {v['median_px'] if v['n'] else '-'} ({v['n']})" for ph, v in ps.items())
                for k, ps in cp[cam].items()))


def cmd_eval(a) -> int:
    from . import evaluate as ev
    from .replay import Config, run
    from .servo import ServoModel, describe

    _need_tag(a)
    sc = _scene(a)
    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else sc.used_episodes()
    modes = a.modes.split(",")
    servo = ServoModel.from_scene(sc)
    out = {"scene": sc.ds, "scene_hash": sc.hash, "scene_layers": [p.name for p in sc.layers],
           "servo": describe(servo), "physics": _phys_kw(a), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "git": poselog.git_describe(), "scene_checks": ev.scene_checks(sc), "episodes": {}}
    _say(json.dumps(out["scene_checks"]))
    t0 = time.time()
    for e in eps:
        out["episodes"][str(e)] = {}
        for mode in modes:
            _say(f"episode {e} {mode}: nominal")
            res = run(sc, e, Config(mode=mode, physics=_physics(a), seed=0, placement=a.placement), servo)
            tag = f"ep{e}_{mode}_seed0{'_' + a.tag if a.tag else ''}"
            poselog.write(_out(a.ds, "logs", f"{tag}.npz"), res.log)
            m = ev.episode_metrics(sc, res.log, res.built)
            _print_metrics(m)
            entry = {"nominal": m}
            if a.render:
                from .render import render_log

                entry["video"] = render_log(sc, res.log, str(_out(a.ds, "video", f"{tag}.mp4")), res.built, every=a.every)
            if a.ensemble and mode != "kinematic":
                _say(f"episode {e} {mode}: ensemble of {a.ensemble}")
                logdir = _out(a.ds, "logs", *([a.tag] if a.tag else []), "x").parent
                entry["ensemble"] = ev.ensemble(sc, e, mode, a.ensemble, workers=a.workers, phys_kw=_phys_kw(a), log=_say,
                                                write_logs=str(logdir), placement=a.placement, sigma=_sigma(a))
                _say(f"  success rate {entry['ensemble']['success_rate']}")
            out["episodes"][str(e)][mode] = entry
    out["wall_s"] = round(time.time() - t0, 1)
    out["summary"] = _summary(out)
    out["placement"], out["scene_overrides"], out["sigma_override"] = a.placement, getattr(sc, "overrides", {}), _sigma(a)
    p = _out(a.ds, f"eval{'_' + a.tag if a.tag else ''}.json")
    ev.write_json(p, out)
    _say(json.dumps(out["summary"], indent=1))
    _say(f"wrote {p}")
    return 0


def _summary(out: dict) -> dict:
    s = {}
    for e, modes in out["episodes"].items():
        for mode, entry in modes.items():
            n = entry["nominal"]
            row = {"success": n["success"], "picks": {p["cap"]: {k: p[k] for k in ("grasped", "lifted", "carried", "in_mug",
                                                                                   "drop_err_frames", "jaw_hold_err_deg")}
                                                      for p in n["picks"]},
                   "joint_rmse_deg": n["joint_rmse_deg"], "fingertip_err_mm_mean": n["fingertip_err_mm"]["mean"],
                   "pen_fc_max_mm": n["pen_fc"]["max_mm"], "pen_fc_p99_mm": n["pen_fc"]["p99_mm"],
                   "x_real_time": n["x_real_time"]}
            if "ensemble" in entry:
                row["ensemble_success_rate"] = entry["ensemble"]["success_rate"]
                row["ensemble_n"] = entry["ensemble"]["n"]
            s[f"ep{e}_{mode}"] = row
    return s


def cmd_render(a) -> int:
    from .evaluate import apply_overrides
    from .render import render_log

    log = poselog.read(a.log)
    sc = apply_overrides(load_scene(a.ds), [(k, v["now"]) for k, v in log["meta"].get("scene_overrides", {}).items()])
    out = Path(a.out) if a.out else _out(a.ds, "video", Path(a.log).stem + ".mp4")
    v = render_log(sc, log, str(out), every=a.every, real=not a.no_real)
    _say(f"video {v['path']} ({v['frames']} frames, {v['seconds']} s)")
    return 0


def cmd_view(a) -> int:
    from . import model as mdl
    from .viewer import play_log

    from .evaluate import apply_overrides

    log = poselog.read(a.log)
    sc = apply_overrides(load_scene(a.ds), [(k, v["now"]) for k, v in log["meta"].get("scene_overrides", {}).items()])
    b = mdl.build(sc, int(log["meta"]["episode"]), perturb=mdl.Perturbation.from_dict(log["meta"].get("perturbation")),
                  placement=log["meta"].get("placement", "config"))
    play_log(b, log)
    return 0


def cmd_info(a) -> int:
    from . import model as mdl
    from .evaluate import recorded_clearance_mm
    from .servo import ServoModel, describe

    sc = load_scene(a.ds)
    _say(f"scene {sc.ds} hash {sc.hash}; layers {[p.name for p in sc.layers]}")
    _say(describe(ServoModel.from_scene(sc)))
    ph = mdl.Physics()
    _say(f"physics: dt {ph.dt:.6f} s ({ph.substeps} substeps/frame), {ph.integrator}, {ph.cone} impratio {ph.impratio}, "
         f"object solref ({ph.timeconst:.5f}, {ph.obj_dampratio}) solimp {ph.obj_solimp}, condim {ph.condim}, "
         f"fingers {ph.fingers}")
    for e in sc.used_episodes():
        b = mdl.build(sc, e)
        _say(f"episode {e}: table z {b.info['table_z'] * 1000:.1f} mm; recorded-state arm clearance above the table "
             f"{recorded_clearance_mm(sc, b, e):+.1f} mm (negative = the scene puts the real arm inside the table)")
    from .evaluate import scene_checks

    cw = scene_checks(sc).get("cap_width_vs_hold")
    if cw:
        _say(f"cap width vs the holds: fingers first touch the real caps at {cw['theta_c_pct']} % = "
             f"{cw['theta_c_deg']} deg, where the finger gap is {cw['gap_at_theta_c_mm']} mm; scene cap "
             f"{cw['scene_cap_diameter_mm']} mm ({cw['diff_mm']:+.2f} mm)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="./robot real2sim mujoco", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", default=None, help="dataset name (default $R2S_DATASET / the default)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("servo-fit", help="identify the servo model -> config/<ds>.servo.json")
    s.add_argument("--iters", type=int, default=25)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--dry-run", action="store_true", help="write only the report, not the config")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_servo_fit)

    def physics_args(p):
        p.add_argument("--placement", choices=("config", "grasp"), default="config",
                       help="cap xy: the scene's (default) or the antipodal grasp from the recorded joints (diagnostic)")
        p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                       help="override a merged-scene value for a sensitivity run, e.g. table.z=0.0147 "
                            "or objects.cap.height=0.016 (recorded in the log meta)")
        p.add_argument("--sigma", help="what-if ensemble uncertainties replacing the scene's: "
                                       "cap_xy=M,table_z=M,cap_d=LO:HI (m)")
        p.add_argument("--impratio", type=float)
        p.add_argument("--substeps", type=int)
        p.add_argument("--condim", type=int)
        p.add_argument("--fingers", choices=("coacd", "hull"))
        p.add_argument("--cap-shape", choices=("cylinder", "taper"),
                       help="cap collision: the scene's mid-height cylinder (default) or CALIB's fitted frustum")
        p.add_argument("--no-flex", action="store_true", help="rigid jaw (no series compliance), for comparison")

    s = sub.add_parser("replay", help="replay one episode")
    s.add_argument("--episode", type=int, required=True)
    s.add_argument("--mode", default="action", choices=("action", "track", "kinematic"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ensemble", type=int, default=0)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--render", action="store_true")
    s.add_argument("--every", type=int, default=1, help="render every Nth frame")
    s.add_argument("--gui", action="store_true", help="play the result in mujoco.viewer (needs DISPLAY)")
    s.add_argument("--frames", help="A:B episode-local window")
    s.add_argument("--table-dz", type=float, help="shift the table (m), an experiment on top of the scene")
    s.add_argument("--cap-diameter", type=float, help="override the cap diameter (m), an experiment")
    s.add_argument("--tag", default="", help="suffix of the log names (required for what-if options)")
    physics_args(s)
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("eval", help="all episodes (+ ensembles) -> mujoco/eval.json")
    s.add_argument("--episodes")
    s.add_argument("--modes", default="action,track")
    s.add_argument("--ensemble", type=int, default=20)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--render", action="store_true")
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--tag", default="", help="write eval_<tag>.json (e.g. a --set / --placement diagnostic)")
    physics_args(s)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("render", help="mp4 from a pose log")
    s.add_argument("--log", required=True)
    s.add_argument("--out")
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--no-real", action="store_true")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("view", help="play a pose log in mujoco.viewer (needs DISPLAY)")
    s.add_argument("--log", required=True)
    s.set_defaults(fn=cmd_view)

    s = sub.add_parser("info", help="servo model, physics, scene feasibility")
    s.set_defaults(fn=cmd_info)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except RuntimeError as e:
        if "no display" in str(e):  # the viewer, headless: say so, do not dump a traceback
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
