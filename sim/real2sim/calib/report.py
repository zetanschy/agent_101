"""Overlays, contact sheets, report.json, the calib.json layer and the README numbers.

    ./robot real2sim calib report [--result <fit.pkl>]

OVERLAYS draw the fitted model on real frames, both cameras at the same timestep:
    magenta  the model arm / fingers (white-PLA silhouette, FK of observation.state
             through the fitted offsets and gripper map)
    green    the real white-PLA mask outline (masks.white)
    cyan     each resting cap: front = the up-facing end circle; wrist = the cylinder's
             silhouette hull (both at the fitted centre, D, h, table z)
    yellow   the mug rim circle (fitted centre, R, H) and a tick toward the handle
    orange   the wrist lens centre (front view) / the grasp site (both views)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import chain, objfit, params as P
from .render import WHITE
from .. import camera as C, episodes as E, paths
from ..kinematics import GRASP_SITE_POS

COL = {"model": (255, 0, 255), "real": (0, 255, 0), "cap": (0, 255, 255), "rim": (255, 255, 0), "lens": (255, 140, 0)}


def _poly(img, uv, col, closed=True, thick=1):
    import cv2

    uv = uv[np.isfinite(uv).all(1)]
    if len(uv) >= 2:
        cv2.polylines(img, [np.round(uv * 4).astype(np.int32)], closed, col, thick, cv2.LINE_AA, shift=2)


def overlay(model: P.Model, row: int, renderer, scene, cams=("front", "grip"), real_outline=True) -> dict:
    """{cam: RGB uint8 image} with the model drawn on the real frame of `row`."""
    import cv2

    from . import masks
    from .silhouette import grip_real_mask

    z = E.npz(scene.ds)
    state = z["state"][row].astype(float)
    ep = int(z["episode_index"][row])
    frame = int(z["frame_index"][row])
    q = model.q(state)
    Tl = chain.link_T(q[None])
    Tg = Tl["gripper"][0]
    out = {}
    for cam_name in cams:
        cam = model.camera(cam_name)
        img = np.ascontiguousarray(E.frames(scene.ds, cam_name)[row]).copy()
        Tp = None if cam.parent == "world" else Tl[cam.parent][0]
        r = renderer.render(cam_name, q, model.T_parent_cam(cam_name))
        mx, my = C.remap_maps(cam, renderer.specs[cam_name][1])
        mm = cv2.remap((r.cls == WHITE).astype(np.uint8), mx, my, cv2.INTER_NEAREST)
        if real_outline:
            real = masks.white(img, cam_name) if cam_name == "front" else grip_real_mask(img)[0]
            cnt, _ = cv2.findContours(real.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnt, -1, COL["real"], 1)
        cnt, _ = cv2.findContours(mm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, COL["model"], 1)
        cfg = scene.episode(ep)
        picked = {pk["cap"] for pk in cfg.get("picks", []) if frame > pk["lift"][0]}
        _, h = model.cap_dims()
        for c in cfg.get("caps", []):
            if c["id"] in picked or not model.has(f"cap.{ep}.{c['id']}"):
                continue
            xy = model.cap_xy(ep, c["id"])
            if cam_name == "front":
                Pc = objfit.circle_pts(xy, model.table_z(xy) + h, objfit.cap_top_diameter(model, c["up"]) / 2)
                _poly(img, cam.project(Pc), COL["cap"])
            else:
                uv = cam.project(objfit.cylinder_world(model, xy, c["up"]), Tp)
                uv = uv[np.isfinite(uv).all(1)]
                if len(uv) >= 3 and np.abs(uv).max() < 5000:
                    _poly(img, objfit.convex_hull(uv), COL["cap"])
        if model.has(f"mug.{ep}"):
            R, H = model.mug_dims()
            xy, yaw = model.mug(ep)
            rim = objfit.circle_pts(xy, model.table_z(xy) + H, R)
            uv = cam.project(rim, Tp)
            if np.isfinite(uv).all() and np.abs(uv).max() < 5000:
                _poly(img, uv, COL["rim"])
                a = cam.project(np.array([[*xy, model.table_z(xy) + H], [*(xy + (R + 0.02) * np.array([np.cos(yaw), np.sin(yaw)])),
                                                                     model.table_z(xy) + H]]), Tp)
                _poly(img, a, COL["rim"], closed=False, thick=2)
        site = Tg[:3, :3] @ np.array(GRASP_SITE_POS) + Tg[:3, 3]
        pts = [site] if cam_name == "grip" else [site, (Tg @ model.T_parent_cam("grip"))[:3, 3]]
        for k, p in enumerate(pts):
            uv = cam.project(p[None], Tp)[0]
            if np.isfinite(uv).all():
                cv2.circle(img, tuple(np.round(uv).astype(int)), 3 if k == 0 else 5, COL["lens"], -1 if k == 0 else 1)
        cv2.putText(img, f"ep{ep} f{frame} {cam_name}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        out[cam_name] = img
    return out


def contact_sheet(model, rows, renderer, scene, path, cols=4, scale=0.5):
    """Front and wrist overlays of each row side by side, `cols` pairs per line, saved JPEG."""
    import cv2

    tiles = []
    for row in rows:
        o = overlay(model, row, renderer, scene)
        pair = np.concatenate([o["front"], o["grip"]], 1)
        tiles.append(cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def key_rows(scene, ep: int, n: int = 12) -> list:
    """Global rows spread over an episode's phases: rest, each approach, close, carry, drop."""
    eps = E.load(scene.ds)
    e = eps[ep]
    cfg = scene.episode(ep)
    ks = [5]
    for pk in cfg["picks"]:
        c0 = pk.get("first_close", pk["close"])[0]
        ks += [max(0, c0 - 40), max(0, c0 - 15), pk["close"][1] + 2, (pk["lift"][1] + pk["release"][0]) // 2, pk["drop"][0]]
    ks += [len(e) - 5]
    ks = sorted(set(int(np.clip(k, 0, len(e) - 1)) for k in ks))
    if len(ks) > n:
        ks = [ks[i] for i in np.linspace(0, len(ks) - 1, n).astype(int)]
    return [e.start + k for k in ks]


# --- report.json, gates, the calib.json layer -------------------------------------------

# Held-out acceptance (episode 2 scored by the fit on episodes 0 + 1). The config is
# written only if every gate passes. Each bound is set against a number measured
# before this calibration existed, so passing means "better than what we had":
GATES = {
    # fit6r, the understand-phase front pose, scored 0.676 on held-out episode 2
    "front_arm_iou": (">=", 0.70),
    # the klip_camera.py pose + its fx 487 pinhole scored 0.67 (mujoco report §5)
    "grip_finger_iou": (">=", 0.93),
    # a resting cap located by the wrist camera alone vs the front camera's ray through
    # its blob (the understand phase measured 2.1 mm and 15 mm for two candidate poses)
    "cap_wrist_to_front_ray_mm_max": ("<=", 12.0),
    # the held-out mug, placed by the front camera alone, seen by the wrist camera
    "mug_rim_wrist_px_median": ("<=", 15.0),
    # fraction of near-table frames with the FK fingertips > 1 mm under the table
    "clearance_below_1mm_frac": ("<=", 0.25),
}

OBJ_PREFIX = ("cap.0.", "cap.1.", "cap.2.", "cap.3.", "mug.0", "mug.1", "mug.2", "mug.3")
# Parameter sets that are degenerate with each other in THIS data (measured in the fit
# log: moving one along the valley costs < 1 px anywhere in the image), so only their
# combined image mapping is determined -- listed with the sigmas.
DEGENERATE = {
    "front.c ~ front.rot": "a principal-point shift is a camera rotation to first order: 31 px of cy is a 1.8 deg "
                           "tilt, and the two differ by f*d*y^2 <= 2.9 px at the image edge, where k1/k2 absorb it",
    "grip.c ~ grip.rot": "same for the wrist camera; the fingers are 3-12 cm away, so depth does not break it",
    "grip.f ~ grip.t (dolly)": "the finger silhouettes alone scale f with the lens-to-finger distance; the caps and the "
                               "rim seen from 5-20 cm (known motion, FK) pin it",
}


def _gate(value, rule):
    op, bound = rule
    return bool(value >= bound) if op == ">=" else bool(value <= bound)


def gates(ev: dict) -> dict:
    caps = ev.get("caps", {})
    ray = [v["wrist_only_top_to_front_ray_mm"] for v in caps.values() if "wrist_only_top_to_front_ray_mm" in v]
    vals = {
        "front_arm_iou": ev["front_arm"]["iou_mean"],
        "grip_finger_iou": ev["grip_fingers"]["iou_mean"],
        "cap_wrist_to_front_ray_mm_max": max(ray) if ray else float("inf"),
        "mug_rim_wrist_px_median": ev.get("mug_front_to_wrist_rim_px", {}).get("median") or float("inf"),
        "clearance_below_1mm_frac": ev["physical"]["clearance_mm"]["below_table_frac"],
    }
    return {k: {"value": round(float(v), 4), "rule": f"{GATES[k][0]} {GATES[k][1]}", "pass": _gate(v, GATES[k])}
            for k, v in vals.items()}


def global_names(layout) -> list:
    return [n for n in layout.names() if not n.startswith(OBJ_PREFIX) and n != "table.tilt"]


def spread(fits: dict, names) -> dict:
    """Leave-one-episode-out spread: the std, per entry, over the fits trained on two of
    the three episodes (holdout = 0+1, loeo_1 = 0+2, loeo_0 = 1+2); zero if fewer than 2."""
    use = [f for k, f in fits.items() if k == "holdout" or k.startswith("loeo")]
    out = {}
    for n in names:
        vals = np.array([f.model().get(n) for f in use])
        out[n] = vals.std(0) if len(vals) > 1 else np.zeros(np.shape(vals[0]))
    return out


def weakly_determined(final, spr: dict) -> list:
    """[(block, why)]: prior-dominated (posterior sigma > half the prior sigma), or
    unstable across the leave-one-episode-out folds (spread > 3x the Jacobian sigma)."""
    L = final.layout
    out = []
    for b in L.blocks:
        if not b.free or b.name.startswith(OBJ_PREFIX):
            continue
        sj = np.asarray(final.sigma[b.name], float)
        if b.sigma is not None:
            r = sj / b.sigma
            for i in np.nonzero(r > 0.5)[0]:
                out.append((f"{b.name}[{i}]", f"prior-dominated: posterior {sj[i]:.4g} vs prior {b.sigma[i]:.4g}"))
        s = np.asarray(spr.get(b.name, np.zeros(b.n)), float)
        for i in np.nonzero(s > 3 * np.maximum(sj, 1e-12))[0]:
            out.append((f"{b.name}[{i}]", f"unstable across folds: LOEO std {s[i]:.4g} vs Jacobian {sj[i]:.4g}"))
    return out


def _fmt(v, nd=5):
    return [round(float(x), nd) for x in np.atleast_1d(v)]


def pose_sigma(sig_rot, sig_t) -> list:
    """[rx ry rz (deg, about the camera's own axes), tx ty tz (m)]."""
    return _fmt(np.degrees(sig_rot), 4) + _fmt(sig_t, 5)


def calib_layer(scene, final, sig: dict, evidence: dict) -> dict:
    """The config/<ds>.calib.json layer: ONLY the fitted keys, each group with source + sigma."""
    m = final.model()
    ev = evidence
    src = lambda what: f"fitted: {what} (real2sim.calib, report.json {ev['report']})"  # noqa: E731
    lay = {"_doc": "Dataset-only joint calibration (sim/real2sim/calib). Every value was fitted on this dataset's "
                   "frames + recorded joints; see report.json for the residuals, held-out scores and LOEO spreads.",
           "_calib": {"git": ev["git"], "scene_hash_before": scene.hash, "train": final.train,
                      "heldout": ev["heldout_summary"]}}
    off, grip = m.get("robot.off"), m.get("robot.grip")
    lay["robot"] = {
        "joint_offsets": {"source": src("front arm silhouettes + the wrist camera's views of the world, all five joints"),
                          "sigma": _fmt(sig["robot.off"], 3), "deg": _fmt(off, 3)},
        "gripper_map": {"source": src("moving-jaw silhouettes in the wrist camera at 1.3-36 % openings"),
                        "sigma": _fmt(sig["robot.grip"], 4), "a_deg": round(float(grip[0]), 4),
                        "b_deg_per_pct": round(float(grip[1]), 5)},
    }
    tz = float(m.get("table.z")[0])
    st = float(sig["table.z"][0])
    lay["table"] = {"source": src("resting caps' bottoms seen by the wrist camera, with the fingertip clearance "
                                  "and grasp constraints"), "sigma": round(st, 5), "z": round(tz, 5),
                    "range": [round(tz - 3 * st, 5), round(tz + 3 * st, 5)],
                    "_fingertips": ev["fingertips"]}
    cams = {}
    for cam in ("front", "grip"):
        c = m.camera(cam)
        cams[cam] = {
            "intrinsics": {"source": src(ev["cam_src"][cam]["K"]), "sigma": _fmt(np.r_[sig[f"{cam}.f"], sig[f"{cam}.c"]], 3),
                           "K": _fmt(c.K, 4)},
            "distortion": {"source": src(ev["cam_src"][cam]["dist"]), "sigma": _fmt(sig[f"{cam}.k"], 5),
                           "dist": _fmt(c.dist, 6)},
            "pose": {"source": src(ev["cam_src"][cam]["pose"]), "sigma": pose_sigma(sig[f"{cam}.rot"], sig[f"{cam}.t"]),
                     "_sigma_order": "rotation deg about the camera x y z axes, then translation m x y z",
                     "parent": c.parent, "T_parent_cam": np.round(c.T_parent_cam, 9).tolist()},
            "latency": {"source": src("images trail observation.state; moving-arm / moving-jaw silhouettes"),
                        "sigma": round(float(sig[f"{cam}.lag"][0]), 3), "frames": round(float(m.get(f"{cam}.lag")[0]), 3),
                        "_use": "compare a sim frame of state time t with the real image of row t + frames"},
        }
    lay["cameras"] = cams
    D, h = m.cap_dims()
    lip = float(m.get("cap.lip")[0])
    sd, sh = sig["cap.dims"]
    lay["objects"] = {
        # `diameter` is what the engines build their straight-cylinder cap from, so it is the
        # RIGID body the fingers squeeze (the closed end, D), not the mid-height mean D + lip/2.
        # The thin tamper-band lip yields under the fingers: measured on the gripper-hold data,
        # the real encoder settles at the fingers' zero-force contact, a 29.4-29.6 mm gap, and
        # ep0's cap falls there (mujoco/README.md). A 30.82 mm (mean) cylinder held all six picks
        # 0.5-1.2 deg too open in both engines and never released ep0; D = 29.32 mm brought the
        # MuJoCo holds to -0.53..+0.14 deg and released ep0 (both engines' sens_cap_closed_end runs).
        "cap": {"source": src("front blob radius (up-facing end) + wrist silhouettes of the whole cylinder + the "
                              "blocked-jaw gap"), "sigma": [round(float(sd), 5), round(float(sh), 5)],
                "diameter": round(D, 5), "height": round(h, 5),
                "diameter_closed_end": round(D, 5), "diameter_open_end": round(D + lip, 5),
                "diameter_mean": round(D + lip / 2, 5),
                "_note": "diameter = the rigid closed-end body the fingers grip (the lip is a thin tamper band, "
                         "diameter_open_end, that yields); the core cap mesh is a straight cylinder of it"},
        "mug": {"source": src("front rim ring + red-wall crescent, wrist rim at release"),
                "sigma": [round(float(2 * sig["mug.dims"][0]), 5), round(float(sig["mug.dims"][1]), 5)],
                "outer_diameter": round(2 * (m.mug_dims()[0] + 0.0015), 5), "height": round(m.mug_dims()[1], 5),
                "rim_radius": round(m.mug_dims()[0], 5)},
    }
    eps = {}
    for e in final.train:
        base = scene.episode(e)
        caps = []
        for c in base["caps"]:
            n = f"cap.{e}.{c['id']}"
            cc = dict(c)
            cc["source"] = src("front blob + wrist approach silhouettes + grasp physics")
            cc["sigma"] = round(float(np.max(sig[n])), 5)
            cc["xy"] = _fmt(m.get(n), 5)
            caps.append(cc)
        xy, yaw = m.mug(e)
        smug = sig[f"mug.{e}"]
        eps[str(e)] = {"caps": caps, "mug": {"source": src("front rim ring + handle blob, wrist rim at release"),
                                             "sigma": round(float(max(smug[0], smug[1])), 5),
                                             "sigma_yaw_deg": round(float(np.degrees(smug[2])), 2),
                                             "xy": _fmt(xy, 5), "handle_yaw_deg": round(float(np.degrees(yaw)), 2)}}
    lay["episodes"] = eps
    return lay


def build(ds, out, folds, quick=False, wall=None, verbose=print) -> dict:
    """Aggregate the folds in `out`: report.json, overlays, contact sheets, gates, config."""
    import os

    from . import fit as FIT
    from .render import ArmRenderer
    from .. import scene as S

    out = Path(out)
    scene = S.load(ds)
    fits = {k: FIT.FitResult.load(out / k / "fit.pkl") for k in folds}
    evals = {k: json.loads((out / k / "eval.json").read_text()) for k in folds}
    final = fits.get("final") or fits["holdout"]  # quick mode has the hold-out fold only
    m = final.model()
    names = global_names(final.layout)
    spr = spread({k: v for k, v in fits.items()}, names)
    sig = {}
    for b in final.layout.blocks:
        s = np.asarray(final.sigma.get(b.name, np.zeros(b.n)), float)
        sig[b.name] = np.maximum(s, spr.get(b.name, np.zeros(b.n)))
    ho = evals["holdout"]["eval"]
    g = gates(ho)
    phys_final = FIT.physfit.Physical(scene, E.load(scene.ds), list(final.train)).report(m)
    weak = weakly_determined(final, spr)
    git = __import__("real2sim.poselog", fromlist=["x"]).git_describe()
    report = {
        "dataset": scene.ds, "git": git, "quick": quick, "wall_s": wall, "scene_hash_before": scene.hash,
        "folds": {k: {"train": evals[k]["train"], "test": evals[k]["test"], "timings": evals[k]["timings"],
                      "blocks": evals[k]["blocks"]} for k in folds},
        "heldout": {k: evals[k].get("eval") for k in folds if evals[k].get("eval")},
        "gates": g,
        "final": {"params": {n: _fmt(m.get(n), 6) for n in final.layout.names()},
                  "sigma_jacobian": {n: _fmt(final.sigma[n], 6) for n in final.layout.names()},
                  "loeo_std": {n: _fmt(spr[n], 6) for n in names},
                  "sigma_used": {n: _fmt(sig[n], 6) for n in final.layout.names()},
                  "chi2_reduced": float(final.sigma["_chi2"][0]),
                  "physical": phys_final, "log": final.log},
        "loeo_params": {k: {n: _fmt(f.model().get(n), 6) for n in names} for k, f in fits.items()},
        "weakly_determined": [{"param": p, "why": w} for p, w in weak],
        "degenerate": DEGENERATE,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=FIT._jsonable))
    (out / "summary.md").write_text(markdown(json.loads((out / "report.json").read_text())))
    # --- overlays: held-out episode 2 from the hold-out fit, objects from the front only
    ren = None
    try:
        hm = fits["holdout"].model()
        v = hm.v.copy()
        sl = hm.L.slices()
        for n, val in ho.get("objects_front_only", {}).items():
            v[sl[n]] = val
        hm = P.Model(hm.L, v, hm.meta)
        ren = ArmRenderer(FIT.render_specs(hm.camera("front")))
        rows = key_rows(scene, 2, 12)
        contact_sheet(hm, rows, ren, scene, out / "overlays" / "heldout_ep2.jpg", cols=3, scale=0.5)
        import cv2

        for row in rows[1::3]:
            o = overlay(hm, row, ren, scene)
            cv2.imwrite(str(out / "overlays" / f"heldout_ep2_row{row}.png"),
                        cv2.cvtColor(np.concatenate([o["front"], o["grip"]], 1), cv2.COLOR_RGB2BGR))
        for e in final.train:
            contact_sheet(m, key_rows(scene, e, 12), ren, scene, out / "overlays" / f"final_ep{e}.jpg", cols=3, scale=0.5)
    finally:
        if ren is not None:
            ren.close()
    summary = {"gates": g, "passed": all(v["pass"] for v in g.values()), "report": str(out / "report.json")}
    if not quick and summary["passed"]:
        ev = {"report": str((out / "report.json").relative_to(paths.REPO)),
              "git": git, "heldout_summary": {k: v["value"] for k, v in g.items()},
              "fingertips": {"min_clearance_mm": round(phys_final["clearance_mm"]["min"], 2),
                             "p1_clearance_mm": round(phys_final["clearance_mm"]["p1"], 2),
                             "_why": "FK fingertips vs this table plane over the train frames near the table"},
              "cam_src": {"front": {"K": "arm silhouettes + multi-view objects, weak prior at the trusted C270 K",
                                    "dist": "k1 k2 (p1 p2 k3 held at 0: not supported on held-out)",
                                    "pose": "arm silhouettes (FK of the recorded joints) + objects seen by both cameras"},
                          "grip": {"K": "finger silhouettes + resting caps and the mug rim from 5-20 cm",
                                   "dist": "k1 k2 (p1 p2 k3 held at 0: not supported on held-out)",
                                   "pose": "on the gripper link: finger silhouettes + multi-view objects"}}}
        layer = calib_layer(scene, final, sig, ev)
        # validate the merged scene BEFORE writing: base <- this layer <- servo (if any)
        merged = S.deep_merge(json.loads(paths.config_path(ds).read_text()), layer)
        sp = paths.config_path(ds, "servo")
        if sp.exists():
            merged = S.deep_merge(merged, json.loads(sp.read_text()))
        problems = S.validate(merged)
        if problems:
            raise ValueError("calib layer would make the scene invalid:\n  " + "\n  ".join(problems))
        p = paths.config_path(ds, "calib")
        p.write_text(json.dumps(layer, indent=1) + "\n")
        summary["config"] = str(p)
        new = S.load(ds)
        summary["scene_hash_after"] = new.hash
        verbose(f"calib: wrote {p} (scene hash {scene.hash} -> {new.hash})")
    elif not quick:
        verbose("calib: held-out gates FAILED, config not written: " + json.dumps({k: v for k, v in g.items() if not v["pass"]}))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def markdown(rep: dict) -> str:
    """The numbers of a report.json as markdown tables (written next to it as summary.md)."""
    fp, sj, su = rep["final"]["params"], rep["final"]["sigma_jacobian"], rep["final"]["sigma_used"]
    lo = rep["final"]["loeo_std"]
    L = []
    row = lambda *c: L.append("| " + " | ".join(str(x) for x in c) + " |")  # noqa: E731
    fmt = lambda v, k=1.0, nd=2: ", ".join(f"{x * k:.{nd}f}" for x in v)  # noqa: E731
    L += ["## Final calibration (fit on episodes " + ", ".join(map(str, rep["folds"].get("final", rep["folds"]["holdout"])["train"])) + ")", "",
          "| parameter | value | sigma used (max of Jacobian and LOEO std) | LOEO std |", "|---|---|---|---|"]
    show = [("front.f", 1, 1, "px"), ("front.c", 1, 1, "px"), ("front.k", 1, 4, ""), ("front.t", 1000, 1, "mm"),
            ("front.lag", 1, 2, "frames"), ("grip.f", 1, 1, "px"), ("grip.c", 1, 1, "px"), ("grip.k", 1, 4, ""),
            ("grip.t", 1000, 1, "mm (gripper frame)"), ("grip.lag", 1, 2, "frames"), ("robot.off", 1, 2, "deg"),
            ("robot.grip", 1, 4, "a deg, b deg/%"), ("table.z", 1000, 2, "mm"), ("cap.dims", 1000, 2, "mm (D closed end, h)"),
            ("cap.lip", 1000, 2, "mm"), ("mug.dims", 1000, 1, "mm (rim radius, rim height)")]
    for n, k, nd, unit in show:
        if n in fp:
            row(f"{n} ({unit})" if unit else n, fmt(fp[n], k, nd), fmt(su[n], k, nd), fmt(lo.get(n, [0] * len(fp[n])), k, nd))
    L += ["", "## Held-out gates (hold-out fold: fit 0 + 1, scored on 2)", "", "| gate | value | rule | pass |", "|---|---|---|---|"]
    for g, v in rep["gates"].items():
        row(g, v["value"], v["rule"], v["pass"])
    L += ["", "## Held-out scores per fold", "",
          "| fold | test | front IoU | front chamfer m->r / r->m px | wrist IoU | wrist chamfer px | cap ray mm (per cap) | "
          "cap dz mm | front->wrist cap edge px | mug rim px | fingertips < table-1mm |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, ev in rep["heldout"].items():
        caps = ev.get("caps", {})
        row(k, ev["test"], f"{ev['front_arm']['iou_mean']:.3f}",
            f"{ev['front_arm']['chamfer_model_to_real_px']['median']:.2f} / {ev['front_arm']['chamfer_real_to_model_px']['median']:.2f}",
            f"{ev['grip_fingers']['iou_mean']:.3f}",
            f"{ev['grip_fingers']['chamfer_model_to_real_px']['median']:.2f} / {ev['grip_fingers']['chamfer_real_to_model_px']['median']:.2f}",
            ", ".join(f"{c} {v.get('wrist_only_top_to_front_ray_mm', float('nan')):.1f}" for c, v in caps.items()),
            ", ".join(f"{v.get('wrist_only_dz_mm', float('nan')):+.1f}" for v in caps.values()),
            ", ".join(f"{v['front_to_wrist_edge_px']['median']:.1f}" for v in caps.values()),
            f"{ev['mug_front_to_wrist_rim_px']['median']:.1f}" if ev["mug_front_to_wrist_rim_px"]["median"] is not None else "-",
            f"{ev['physical']['clearance_mm']['below_table_frac']:.3f} (min {ev['physical']['clearance_mm']['min']:.1f} mm)")
    ph = rep["final"]["physical"]
    L += ["", "## Physics of the final fit (train frames)", "",
          f"fingertip clearance over the table: min {ph['clearance_mm']['min']:.2f} mm, p1 {ph['clearance_mm']['p1']:.2f}, "
          f"p5 {ph['clearance_mm']['p5']:.2f}, median {ph['clearance_mm']['median']:.2f}; below -1 mm: "
          f"{ph['clearance_mm']['below_table_frac']:.3f} of near-table frames", "",
          "| pick | cap in gripper mm (x close, y across, z along) | closing offset mm | jaw deg | gap mm | gap - cap mm | "
          "grasp site vs cap xy mm | grasp site above table mm |", "|---|---|---|---|---|---|---|---|"]
    for p in ph["picks"]:
        row(f"ep{p['episode']} {p['cap']}", p["cap_in_gripper_mm"], p["closing_offset_mm"], p["jaw_deg"], p["gap_mm"],
            p.get("gap_minus_cap_mm", "-"), p["grasp_site_vs_cap_xy_mm"], p["grasp_site_height_above_table_mm"])
    L += ["", "## Weakly determined", ""] + [f"- {w['param']}: {w['why']}" for w in rep["weakly_determined"]]
    L += ["", "## Degenerate combinations", ""] + [f"- {k}: {v}" for k, v in rep["degenerate"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import argparse
    import os
    import sys

    from . import pipeline

    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser(description="real2sim CALIB: rebuild the report from saved folds")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ds")
    a = ap.parse_args()
    ds = paths.dataset(a.ds)
    out = pipeline.out_dir(ds, a.quick)
    folds = [k for k in (pipeline.FOLDS_QUICK if a.quick else pipeline.FOLDS_FULL) if (out / k / "eval.json").exists()]
    print(json.dumps(build(ds, out, folds, quick=a.quick), indent=1))
    sys.stdout.flush()
    os._exit(0)
