"""Collect both stacks' measured numbers into one table (compare/summary.{json,md}).

PHYSICS. Both engines replay the same scene (config layers + scene hash), the same
identified servo, the same replay modes and the same metrics code: Isaac imports
real2sim.mujoco.evaluate.episode_metrics, and its ensembles call the MuJoCo track's
draw_perturbation seed for seed. So a row here is the same measurement in two engines.

  nominal     1 run per episode and mode, the scene as fitted
  stress      ensemble, uncertainties as CALIB wrote them (table z sigma 5.5 mm = the
              leave-one-episode-out spread, which one fold's table/cap-height trade dominates)
  consistent  ensemble with table z sigma 1.2 mm: the spread of the three folds that agree
              (final 23.99, holdout 26.19, loeo_0 23.69 mm; loeo_1 found 13.5 mm with an 18 mm
              cap). Perturbing the table alone by 5.5 mm while the arm offsets it was fitted
              with stay fixed makes geometry no calibration fold produced.

RENDERING. Every renderer draws LOOK's 24 sample timesteps x 2 cameras with the arm at the
recorded joints (kinematic) and is scored by the same scorer (look/imagemetrics.py):
PSNR / SSIM / mean-colour Delta E per labelled region. Blender's default look adds two fits
of its own (materials.py RENDER_FIT, PER_CAMERA); 'blender eevee shared-look' drops them, so
it and Isaac RTX render identical material inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import paths, scene as scene_mod

MODES = ("action", "track")
ENSEMBLES = {"stress": {"mujoco": "eval.json", "isaac": "ens16"},
             "consistent": {"mujoco": "eval_consistent_table.json", "isaac": "ens16_consistent_table"}}
RENDERERS = [  # (label, directory under outputs/<ds>/, scored images inside it)
    ("Blender EEVEE", "blender/samples_kinematic_eevee", "samples"),
    ("Blender EEVEE shared-look", "blender/samples_kinematic_eevee_sharedlook", "samples"),
    ("Blender Cycles", "blender/samples_kinematic_cycles", "samples"),
    ("Isaac RTX real-time", "isaac/render/samples_kinematic", "samples_rt"),
    ("Isaac RTX path-traced", "isaac/render/samples_kinematic", "samples_pt"),
]
REGIONS = {"front": ("all", "background", "arm", "fingers", "cap", "mug"),
           "grip": ("all", "background", "fingers", "cap", "mug")}
PICK_KEYS = ("in_mug", "drop_err_frames", "jaw_hold_err_deg", "grip_force_N_median", "slip_mm", "pen_fc_max_mm",
             "fingertip_err_at_grasp_mm")


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _r(x, n=2):
    """Rounded scalar; a per-finger pair (grip_force_N_*) is shown as its mean."""
    if isinstance(x, (list, tuple)):
        x = float(np.mean(x)) if len(x) else None
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def _episode_row(nom: dict) -> dict:
    rm = nom.get("joint_rmse_deg") or []
    fe = nom.get("fingertip_err_mm") or {}
    return {"scene_hash": nom.get("scene_hash"), "success": nom.get("success"),
            "picks": {p["cap"]: {k: p.get(k) for k in PICK_KEYS} for p in nom.get("picks", [])},
            "arm_joint_rmse_deg": _r(np.mean(rm[:5])) if len(rm) >= 5 else None,
            "jaw_rmse_deg": _r(rm[5]) if len(rm) > 5 else None,
            "fingertip_err_mm_mean": _r(fe.get("mean") if isinstance(fe, dict) else fe),
            "pen_fc_max_mm": _r((nom.get("pen_fc") or {}).get("max_mm"), 3),
            "pen_fc_p99_mm": _r((nom.get("pen_fc") or {}).get("p99_mm"), 3),
            "mug_moved_mm": _r(nom.get("mug_moved_mm")), "x_real_time": _r(nom.get("x_real_time")),
            "wall_s": _r(nom.get("wall_s"), 1)}


def _ens_row(ens: dict | None, scene_hash: str | None) -> dict | None:
    if not ens:
        return None
    return {"success_rate": _r(ens.get("success_rate"), 3), "n": ens.get("n"), "scene_hash": scene_hash,
            "pen_fc_max_mm": _r(ens.get("pen_fc_max_mm"), 3), "wall_s": _r(ens.get("wall_s") or ens.get("loop_s"), 1)}


def physics(ds: str, eps) -> dict:
    out = {"mujoco": {}, "isaac": {}}
    d = paths.out_dir(ds)
    mj = {k: _load(d / "mujoco" / v["mujoco"]) for k, v in ENSEMBLES.items()}
    for e in eps:
        for m in MODES:
            key = f"ep{e}_{m}"
            base = mj["stress"]
            if base and str(e) in base["episodes"] and m in base["episodes"][str(e)]:
                rec = base["episodes"][str(e)][m]
                row = _episode_row(rec["nominal"])
                row["ensembles"] = {k: _ens_row((mj[k] or {}).get("episodes", {}).get(str(e), {}).get(m, {}).get("ensemble"),
                                                (mj[k] or {}).get("scene_hash")) for k in ENSEMBLES}
                out["mujoco"][key] = row
            nom = _load(d / "isaac" / key / "eval.json")
            if nom:
                row = _episode_row(nom["nominal"])
                row["ensembles"] = {}
                for k, v in ENSEMBLES.items():
                    ev = _load(d / "isaac" / f"{key}_{v['isaac']}" / "eval.json")
                    row["ensembles"][k] = _ens_row((ev or {}).get("ensemble"), (ev or {}).get("scene_hash"))
                t = nom.get("timings") or {}
                row["ms_per_frame"] = t.get("ms_per_frame")
                out["isaac"][key] = row
    return out


def _score(render_dir: Path, ds: str) -> dict | None:
    """The shared scorer's summary for a directory of sample PNGs (cached next to them)."""
    if not render_dir.exists() or not any(render_dir.glob("*.png")):
        return None
    for name in ("score.json", "look_score.json"):
        for p in (render_dir / name, render_dir.parent / name):
            s = _load(p)
            if s and "summary" in s and p.stat().st_mtime >= max(f.stat().st_mtime for f in render_dir.glob("*.png")):
                return s
    from ..look.imagemetrics import score
    s = score(render_dir, ds, erode_px=2)
    (render_dir / "look_score.json").write_text(json.dumps(s, indent=1))
    return s


def _render_hash_and_speed(base: Path, sub: str) -> tuple[str | None, dict]:
    """Scene hash the renderer used and its per-image speed, from its manifests."""
    if sub == "samples":  # Blender: one manifest for the whole sample run
        m = _load(base / "manifest.json") or {}
        per = {}
        for ep in (m.get("timing") or {}).values():
            for cam, c in ((ep.get("blender") or {}).get("cameras") or {}).items():
                per.setdefault(cam, []).append(c.get("mean_s_after_first"))
        h = m.get("scene_hash")
        if isinstance(h, dict):  # one render job per episode, each with its own hash
            hs = set(h.values())
            h = hs.pop() if len(hs) == 1 else "mixed:" + ",".join(sorted(hs))
        return h, {c: _r(np.mean([x for x in v if x is not None]), 3) for c, v in per.items()}
    mode = sub.split("_")[-1]  # Isaac: one manifest per episode and mode
    ms = [_load(p) for p in sorted(base.glob(f"manifest_ep*_{mode}.json"))]
    ms = [m for m in ms if m]
    if not ms:
        return None, {}
    n_img = sum(len(m.get("cameras") or {}) * len(m.get("frames", []) or [0] * 8) for m in ms)
    busy = sum(float(m.get("wall_s", 0)) - float(m.get("boot_s", 0)) for m in ms)
    return ms[0].get("scene_hash"), {"all": _r(busy / max(n_img, 1), 3), "boot_s": _r(np.mean([m.get("boot_s", 0) for m in ms]), 1)}


def rendering(ds: str) -> dict:
    out = {}
    d = paths.out_dir(ds)
    for label, rel, sub in RENDERERS:
        base = d / rel
        s = _score(base / sub, ds)
        if not s:
            continue
        h, speed = _render_hash_and_speed(base, sub)
        out[label] = {"scene_hash": h, "s_per_image": speed, "cameras": {
            cam: {reg: {k: _r((s["summary"].get(cam) or {}).get(reg, {}).get(k), 3 if k == "ssim" else 2)
                        for k in ("psnr", "ssim", "delta_e_mean_colour")}
                  for reg in regs if (s["summary"].get(cam) or {}).get(reg)}
            for cam, regs in REGIONS.items()}}
    return out


def _stale(h, current):
    return "" if h in (None, current) else " STALE"


def markdown(res: dict) -> str:
    cur = res["scene_hash"]
    L = [f"# Isaac Sim vs MuJoCo + Blender: {res['dataset']}", "",
         f"Current scene hash `{cur}`. Rows measured on another scene say STALE.", ""]
    ph = res["physics"]
    for m in MODES:
        L += [f"## Physics, nominal replay, {m} mode", "",
              "| pick | engine | in mug | drop − real (frames) | jaw hold − real (°) | squeeze (N) | slip (mm) | pen. max (mm) | fingertip at grasp (mm) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for key in sorted({k for eng in ph.values() for k in eng if k.endswith(m)}):
            for eng in ("mujoco", "isaac"):
                row = ph[eng].get(key)
                if not row:
                    continue
                for cap, p in row["picks"].items():
                    L.append(f"| {key.split('_')[0]} {cap} | {eng}{_stale(row['scene_hash'], cur)} | "
                             f"{'yes' if p['in_mug'] else '**no**'} | {p['drop_err_frames'] if p['drop_err_frames'] is not None else '—'} | "
                             f"{_r(p['jaw_hold_err_deg'])} | {_r(p['grip_force_N_median'], 1)} | {_r(p['slip_mm'])} | "
                             f"{_r(p['pen_fc_max_mm'])} | {_r(p['fingertip_err_at_grasp_mm'], 1)} |")
        L.append("")
    L += ["## Physics, per episode", "",
          "| episode | engine | success | arm joint RMSE (°) | fingertip mean (mm) | pen. max / p99 (mm) | × real time | stress ensemble | consistent ensemble |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted({k for eng in ph.values() for k in eng}):
        for eng in ("mujoco", "isaac"):
            r = ph[eng].get(key)
            if not r:
                continue
            ens = {k: (f"{r['ensembles'][k]['success_rate']:.2f} (n={r['ensembles'][k]['n']}){_stale(r['ensembles'][k]['scene_hash'], cur)}"
                       if r["ensembles"].get(k) else "—") for k in ENSEMBLES}
            L.append(f"| {key} | {eng}{_stale(r['scene_hash'], cur)} | {'yes' if r['success'] else 'no'} | {r['arm_joint_rmse_deg']} | "
                     f"{r['fingertip_err_mm_mean']} | {r['pen_fc_max_mm']} / {r['pen_fc_p99_mm']} | {r['x_real_time']} | "
                     f"{ens['stress']} | {ens['consistent']} |")
    L += ["", "## Rendering, LOOK's 24 sample timesteps (PSNR dB / SSIM / ΔE)", ""]
    rd = res["rendering"]
    for cam, regs in REGIONS.items():
        L += [f"**{cam}**", "", "| renderer | s/image | " + " | ".join(regs) + " |", "|---|---|" + "---|" * len(regs)]
        for label, r in rd.items():
            c = r["cameras"].get(cam) or {}
            sp = r["s_per_image"].get(cam, r["s_per_image"].get("all"))
            L.append(f"| {label}{_stale(r['scene_hash'], cur)} | {sp} | " + " | ".join(
                f"{c[g]['psnr']} / {c[g]['ssim']} / {c[g]['delta_e_mean_colour']}" if g in c else "—" for g in regs) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="real2sim compare summary")
    ap.add_argument("--ds")
    a = ap.parse_args(argv)
    ds = paths.dataset(a.ds)
    sc = scene_mod.load(ds)
    res = {"dataset": ds, "scene_hash": sc.hash, "physics": physics(ds, sc.used_episodes()), "rendering": rendering(ds)}
    out = paths.out_dir(ds, "compare", create=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    md = markdown(res)
    (out / "summary.md").write_text(md)
    print(md)
    print(f"-> {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
