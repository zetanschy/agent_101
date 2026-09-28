"""Held-out and cross-view metrics of a fitted calibration (RULE 3: numbers, not claims).

evaluate(result, test) scores a FitResult on episodes it was NOT fitted on:

PER VIEW (global parameters frozen, nothing refitted):
    front arm   silhouette IoU (care region) and chamfer px, both directions:
                model->real |SDF| at model edge points, real->model point-to-line
    grip        the same for the fixed finger + moving jaw (the gripper map at work)
OBJECTS of a held-out episode need poses, so they are fitted from ONE view and scored
in the OTHER (cross-view), with every global parameter frozen:
    front-only  cap xy from the front blobs, mug from the front rim + handle
                -> wrist-view cap silhouette error of those caps (px) at the approach
                   frames, and the grasp / drop physics (mm)
    wrist-only  cap xy AND height from the wrist silhouettes alone (a free z per cap:
                wrist-only triangulation, bottom at table + dz)
                -> 3D distance of its top-face centre to the FRONT camera's ray through
                   the front blob centroid (mm), its front reprojection error (px), and
                   dz itself (mm: how far the table plane is off at that cap)
PHYSICAL: fingertip-to-table clearance over all frames, grasp-site vs cap xy at every
grasp, the closing-direction offset of the cap in the jaws, the blocked gap vs D.
"""

from __future__ import annotations

import numpy as np

from . import fit as FIT, objfit, params as P, silhouette as SIL
from .render import ArmRenderer
from .. import camera as C, episodes as E, scene as S


def _sil_metrics(block, model, renderer, lens=None) -> dict:
    block.associate(model, renderer, lens)
    r = block.residuals(model) * SIL.SIGMA_PX[block.cam]
    n = len(block.f_m)
    m2d = np.abs(r[:n] / np.maximum(block.w_m, 1e-9))
    d2m = np.abs(r[n:] / np.maximum(block.w_d, 1e-9))
    iou = block.iou(model, renderer)
    return {"frames": len(block.frames), "iou_mean": float(iou.mean()), "iou_min": float(iou.min()),
            "chamfer_model_to_real_px": {"median": float(np.median(m2d)), "mean": float(np.mean(np.clip(m2d, 0, 40)))},
            "chamfer_real_to_model_px": {"median": float(np.median(d2m)), "mean": float(np.mean(np.clip(d2m, 0, 40)))},
            "per_frame_iou": iou.round(4).tolist()}


def _with_capz(layout: P.Layout, ep_ids, scene) -> P.Layout:
    L = layout.copy()
    for e in ep_ids:
        for c in scene.episode(e)["caps"]:
            L.add(f"capz.{e}.{c['id']}", [0.0], 1e-3, free=False, lo=[-0.03], hi=[0.03], note="wrist-only cap height offset")
    return L


def evaluate(result, test, cfg=None, verbose=print) -> dict:
    cfg = cfg or FIT.Config()
    rng = np.random.default_rng(cfg.seed + 100)
    scene = S.load()
    eps = E.load(scene.ds)
    model = result.model()
    renderer = ArmRenderer(FIT.render_specs(model.camera("front")))
    d = FIT.build_data(scene, eps, list(test), cfg, model, renderer, rng)
    out = {"test": list(test), "train": result.train}
    out["front_arm"] = _sil_metrics(d.front_sil, model, renderer, FIT.lens_disks(model, d.front_sil))
    out["grip_fingers"] = _sil_metrics(d.grip_sil, model, renderer)
    say = verbose or (lambda *_: None)
    say(f"  held-out front arm IoU {out['front_arm']['iou_mean']:.3f}, grip IoU {out['grip_fingers']['iou_mean']:.3f}")

    # --- front-only objects of the held-out episodes
    L = result.layout.rebase(result.full)
    L.only_free(FIT.object_blocks(test))
    Lf, full_f, _ = FIT.solve(L, L.full0(), result.meta,
                              [("front_caps", d.front_caps.residuals, 1.0), ("front_mug", d.front_mug.residuals, 1.0)],
                              200, None, "front-only")
    mf = P.Model(Lf, full_f, result.meta)
    d.grip_caps = objfit.GripCaps(FIT.grip_cap_frames(scene, eps, mf, test, max(cfg.n_gripcap, 10), rng))
    per_cap = {}
    # wrist-view error of the front-derived caps
    polys = d.grip_caps.polygons(mf)
    for fr, poly in zip(d.grip_caps.frames, polys):
        key = f"{fr.ep}.{fr.cap}"
        dist = np.abs(objfit.polygon_sdf(fr.pts, poly)) if poly is not None else np.full(len(fr.pts), np.nan)
        per_cap.setdefault(key, {"front_to_wrist_edge_px": []})["front_to_wrist_edge_px"].append(dist)
    for key, v in per_cap.items():
        a = np.concatenate(v["front_to_wrist_edge_px"])
        v["front_to_wrist_edge_px"] = {"median": float(np.nanmedian(a)), "rms": float(np.sqrt(np.nanmean(a ** 2))),
                                       "frames": len(v["front_to_wrist_edge_px"])}

    # --- wrist-only 3D caps (xy + height), scored against the front rays
    Lw = _with_capz(result.layout.rebase(result.full), test, scene)
    Lw.only_free([f"cap.{e}." for e in test] + [f"capz.{e}." for e in test])
    Lw, full_w, _ = FIT.solve(Lw, Lw.full0(), result.meta, [("grip_caps", d.grip_caps.residuals, 1.0)], 200, None, "wrist-only")
    mw = P.Model(Lw, full_w, result.meta)
    front = model.camera("front")
    _, h = model.cap_dims()
    for it in d.front_caps.items:
        key = f"{it.ep}.{it.cap}"
        if not mw.has(f"capz.{it.ep}.{it.cap}") or key not in per_cap:
            continue
        xy = mw.cap_xy(it.ep, it.cap)
        dz = float(mw.get(f"capz.{it.ep}.{it.cap}")[0])
        top = np.array([*xy, mw.table_z(xy) + dz + h])
        o, ray = front.pixel_to_ray(np.array([it.u, it.v]))
        v = top - o
        perp = np.linalg.norm(v - (v @ ray) * ray)
        # front reprojection of the wrist-only cap (top circle, as the front block models it)
        Pc = objfit.circle_pts(xy, mw.table_z(xy) + dz + h, objfit.cap_top_diameter(mw, it.up) / 2)
        c, A = objfit.poly_centroid_area(front.project(Pc))
        xy_f = mf.cap_xy(it.ep, it.cap)
        per_cap[key].update({
            "wrist_only_top_to_front_ray_mm": round(float(perp * 1000), 2),
            "wrist_only_front_reproj_px": round(float(np.hypot(c[0] - it.u, c[1] - it.v)), 2),
            "wrist_only_dz_mm": round(dz * 1000, 2),
            "wrist_vs_front_xy_mm": round(float(np.linalg.norm(xy - xy_f) * 1000), 2),
            "front_xy_mm": (xy_f * 1000).round(1).tolist(), "wrist_xy_mm": (xy * 1000).round(1).tolist()})
    out["caps"] = per_cap
    # --- the mug of the held-out episodes: front-only centre/yaw, scored in the wrist view
    rim_frames = FIT.wrist_rim_frames(scene, eps, list(test), 6, rng)
    wr = objfit.WristRim(rim_frames)
    wr.associate(mf, renderer, search=60.0)
    rr = wr.residuals(mf) * objfit.SIG_WRIST_RIM
    rr = rr / np.concatenate([np.full(len(f.th), np.sqrt(30.0 / len(f.th))) for f in rim_frames if f.th is not None and len(f.th)]) if len(rr) else rr
    out["mug_front_to_wrist_rim_px"] = {"points": int(len(rr)), "frames": len(rim_frames),
                                        "median": float(np.median(np.abs(rr))) if len(rr) else None,
                                        "rms": float(np.sqrt(np.mean(rr ** 2))) if len(rr) else None}
    # --- physics of the held-out episodes, with the front-only objects
    phys = FIT.physfit.Physical(scene, eps, list(test))
    out["physical"] = phys.report(mf)
    rim = []
    for it in d.front_mug.items:
        xy, yaw = mf.mug(it.ep)
        rim.append({"episode": it.ep, "xy_mm": (xy * 1000).round(1).tolist(), "handle_yaw_deg": round(float(np.degrees(yaw)), 1)})
    out["mug"] = rim
    out["objects_front_only"] = {n: mf.get(n).tolist() for n in FIT.object_names(Lf, test)}
    renderer.close()
    return out

