"""Per-frame REAL observations of the dataset, reusable by every track's sim-vs-real metrics.

    ./robot real2sim calib observe          # detect (all frames, ~1 min) + label
    obs = real2sim.calib.observations.load()
    obs.teal("front", rows=range(0, 488))   # structured array of teal blobs, one per row
    obs.teal("grip", cap="A", episode=1)    # the blobs identified as that cap
    obs.rim(rows)                           # front mug-rim circle per frame
    obs.summary("front")                    # white-mask area/centroid/bbox per frame
    obs.mask("grip", row)                   # the real white-PLA (finger) mask, bool
    obs.jaw()                               # {row, jaw_deg, iou, state_pct, state_jaw_deg}

Everything is in the camera's own distorted pixels, (0, 0) = centre of the top-left
pixel (camera.py). Rows are global dataset rows (episodes.npz); `episode` and
`frame` columns give the episode-local index.

FILES (sim/outputs/real2sim/<ds>/observations/):
    detections.npz   teal blobs (both cameras), red blobs and the rim circle (front),
                     mask summaries; image-only, no calibration involved
    masks_<cam>.npz  the white-PLA masks, bit-packed along x (np.unpackbits(..., axis=-1))
    labels.npz       cap identity + state of every teal blob, from the SCENE (calibration,
                     pick schedule): re-run `label` after the calibration changes
    jaw.npz          per row the Jaw angle (deg) the wrist camera sees, its template IoU,
                     and the gripper-map angle of the recorded state, at the row and at
                     the image's exposure (row - the wrist latency) (jaw_angles)
    README.md        these columns with units

TEAL BLOB columns: row, episode, frame, cam (0 front, 1 grip), u, v (area centroid),
area (px), r_eq = sqrt(area / pi), bbox x y w h, border (touches the frame),
ell_u, ell_v, ell_a, ell_b, ell_deg (cv2.fitEllipse on the blob's outer contour
without the points that touch the frame or the white fingers: full axes in px; NaN
if fewer than 12 such points), occluded (fraction of that contour dropped).
LABELS: cap ('A'..; '' unknown), state 'rest' | 'held' | 'in_mug' | 'moving' | '?'.

RIM (front, one per row): cx, cy, r (px) maximising the median grey level on the
circle (180 samples) within +-5 px / +-4 px of the episode's rim (config rim_px),
1 px grid refined to 0.1 px; score (that median grey level); occluded = fraction of
the config rim circle whose grey level departs from the episode's median ring
profile by more than 40 (the arm or a held cap over the rim).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import masks
from .. import paths

TEAL_DTYPE = [("row", "i4"), ("episode", "i2"), ("frame", "i4"), ("cam", "i1"), ("u", "f4"), ("v", "f4"),
              ("area", "i4"), ("r_eq", "f4"), ("bx", "i2"), ("by", "i2"), ("bw", "i2"), ("bh", "i2"),
              ("border", "?"), ("ell_u", "f4"), ("ell_v", "f4"), ("ell_a", "f4"), ("ell_b", "f4"),
              ("ell_deg", "f4"), ("occluded", "f4")]
RED_DTYPE = [("row", "i4"), ("u", "f4"), ("v", "f4"), ("area", "i4"), ("bx", "i2"), ("by", "i2"), ("bw", "i2"),
             ("bh", "i2")]
MIN_AREA = {"front": 25, "grip": 150}  # px; a resting cap is ~500 px front, ~15000 px grip
MAX_BLOBS = {"front": 12, "grip": 6}
CAMS = ("front", "grip")


def out_dir(ds=None, create=False):
    return paths.out_dir(ds, "observations", create=create)


# --- detection (image only) ------------------------------------------------------------

def _ellipse(comp_mask, white_near, W, H):
    """Ellipse through the blob's outer contour, minus points on the frame or the fingers."""
    import cv2

    cnt, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnt:
        return (np.nan,) * 5, 1.0
    c = max(cnt, key=len)[:, 0, :]
    ok = (c[:, 0] > 1) & (c[:, 0] < W - 2) & (c[:, 1] > 1) & (c[:, 1] < H - 2)
    if white_near is not None:
        ok &= ~white_near[c[:, 1], c[:, 0]]
    occ = 1.0 - ok.mean()
    if ok.sum() < 12:
        return (np.nan,) * 5, float(occ)
    (eu, ev), (a, b), ang = cv2.fitEllipse(c[ok].astype(np.float32))
    return (eu, ev, a, b, ang), float(occ)


def _teal_blobs(img, cam, row, ep, frame, white=None):
    import cv2

    t = masks.teal(img, cam)
    H, W = t.shape
    near = cv2.dilate(white.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 if white is not None else None
    out = []
    for area, cen, (x, y, w, h), comp in masks.components(t, MIN_AREA[cam])[: MAX_BLOBS[cam]]:
        ell, occ = _ellipse(comp, near, W, H)
        border = x <= 0 or y <= 0 or x + w >= W or y + h >= H
        out.append((row, ep, frame, CAMS.index(cam), cen[0], cen[1], area, np.sqrt(area / np.pi), x, y, w, h,
                    border, *ell, occ))
    return out


def rim_search(gray, c0, r0, dxy=5.0, dr=4.0):
    """(cx, cy, r, score): the circle with the brightest median ring near (c0, r0);
    1 px grid over +-dxy / +-dr, then 0.25 px, then 0.1 px around the best."""
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    cs, sn = np.cos(th), np.sin(th)

    def best(cxs, cys, rs):
        C = np.stack(np.meshgrid(cxs, cys, rs, indexing="ij"), -1).reshape(-1, 3)
        u = C[:, 0:1] + C[:, 2:3] * cs
        v = C[:, 1:2] + C[:, 2:3] * sn
        val = masks.bilinear(gray, np.stack([u, v], -1), fill=np.nan)
        sc = np.nanmedian(val, axis=1)
        sc[np.isnan(val).mean(1) > 0.3] = -1
        k = int(np.argmax(sc))
        return C[k], float(sc[k])

    (cx, cy, r), s = best(np.arange(c0[0] - dxy, c0[0] + dxy + 1e-9, 1.0), np.arange(c0[1] - dxy, c0[1] + dxy + 1e-9, 1.0),
                          np.arange(r0 - dr, r0 + dr + 1e-9, 1.0))
    for f in (np.arange(-0.75, 0.76, 0.25), np.arange(-0.2, 0.21, 0.1)):
        (cx, cy, r), s = best(cx + f, cy + f, r + f)
    return cx, cy, r, s


def ring_profile(gray, cx, cy, r, n=90) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return masks.bilinear(gray, np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], -1), fill=np.nan)


def _detect_chunk(args):
    """Worker: rows [a, b) of both cameras (opens the memmaps itself)."""
    import cv2

    ds, a, b, rim0 = args
    from .. import episodes

    F, G = episodes.frames(ds, "front"), episodes.frames(ds, "grip")
    z = episodes.npz(ds)
    teal, red, rim, summ, prof = [], [], [], [], []
    wm_f = np.zeros((b - a, F.shape[1], (F.shape[2] + 7) // 8), np.uint8)
    wm_g = np.zeros_like(wm_f)
    for k, row in enumerate(range(a, b)):
        ep, frame = int(z["episode_index"][row]), int(z["frame_index"][row])
        f, g = np.asarray(F[row]), np.asarray(G[row])
        wf = masks.white(f, "front")
        from .silhouette import grip_real_mask

        wg, _ = grip_real_mask(g)
        wm_f[k], wm_g[k] = np.packbits(wf, axis=-1), np.packbits(wg, axis=-1)
        teal += _teal_blobs(f, "front", row, ep, frame)
        teal += _teal_blobs(g, "grip", row, ep, frame, white=wg)
        for area, cen, (x, y, w, h), _ in masks.components(masks.red(f), 80)[:4]:
            red.append((row, cen[0], cen[1], area, x, y, w, h))
        gray = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0.8)
        c0 = rim0.get(ep)
        if c0 is not None:
            cx, cy, r, s = rim_search(gray, c0[:2], c0[2])
            rim.append((row, cx, cy, r, s, np.nan))
            prof.append(ring_profile(gray, c0[0], c0[1], c0[2]))
        else:
            rim.append((row, np.nan, np.nan, np.nan, np.nan, np.nan))
            prof.append(np.full(90, np.nan))
        ys, xs = np.nonzero(wf)
        yg, xg = np.nonzero(wg)
        top = g[: g.shape[0] * 2 // 5].max(2)
        summ.append((row, len(xs), xs.mean() if len(xs) else np.nan, ys.mean() if len(xs) else np.nan,
                     xs.min() if len(xs) else -1, ys.min() if len(xs) else -1, xs.max() if len(xs) else -1,
                     ys.max() if len(xs) else -1, len(xg), float((top > 100).mean())))
    return teal, red, rim, summ, wm_f, wm_g, np.array(prof), a


def detect(ds=None, workers: int | None = None, verbose=True) -> dict:
    """Run the image-only detections over every row and write detections.npz + masks."""
    import time

    from .. import episodes, scene as scene_mod

    ds = paths.dataset(ds)
    sc = scene_mod.load(ds)
    rim0 = {int(k): e["mug"]["rim_px"] for k, e in sc["episodes"].items() if "mug" in e and "rim_px" in e["mug"]}
    N = len(episodes.npz(ds)["index"])
    workers = workers or min(12, os.cpu_count() or 4)
    edges = np.linspace(0, N, workers * 3 + 1).astype(int)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_detect_chunk, [(ds, int(a), int(b), rim0) for a, b in zip(edges[:-1], edges[1:])]))
    parts.sort(key=lambda p: p[-1])
    teal = np.array([t for p in parts for t in p[0]], dtype=TEAL_DTYPE)
    red = np.array([t for p in parts for t in p[1]], dtype=RED_DTYPE)
    rim = np.array([t for p in parts for t in p[2]], dtype=[("row", "i4"), ("cx", "f4"), ("cy", "f4"), ("r", "f4"),
                                                             ("score", "f4"), ("occluded", "f4")])
    # occlusion: the fraction of the episode's rim circle (config rim_px) whose grey level
    # departs from that episode's median ring profile by > 40 (the arm or the held cap
    # over the rim; the steel itself is white-ish, so a white-mask test cannot tell)
    prof = np.concatenate([p[6] for p in parts])
    z = episodes.npz(ds)
    for e, a, b in zip(z["episodes"], z["ep_from"], z["ep_to"]):
        med = np.nanmedian(prof[a:b], axis=0)
        rim["occluded"][a:b] = (np.abs(prof[a:b] - med) > 40).mean(1)
    summ = np.array([t for p in parts for t in p[3]], dtype=[
        ("row", "i4"), ("front_area", "i4"), ("front_u", "f4"), ("front_v", "f4"), ("front_x0", "i2"), ("front_y0", "i2"),
        ("front_x1", "i2"), ("front_y1", "i2"), ("grip_area", "i4"), ("grip_top_bright", "f4")])
    od = out_dir(ds, create=True)
    np.savez_compressed(od / "detections.npz", teal=teal, red=red, rim=rim, summary=summ,
                        meta=json.dumps({"dataset": ds, "rows": int(N), "thresholds": {
                            "white": masks.WHITE, "teal": masks.TEAL_HSV, "red": masks.RED_HSV,
                            "min_area": MIN_AREA}, "rim_init_px": rim0, "seconds": round(time.time() - t0, 1)}))
    np.savez_compressed(od / "masks_front.npz", packed=np.concatenate([p[4] for p in parts]), width=640)
    np.savez_compressed(od / "masks_grip.npz", packed=np.concatenate([p[5] for p in parts]), width=640)
    (od / "README.md").write_text("# Real observations (real2sim.calib.observations)\n\n```\n" + __doc__.strip() + "\n```\n")
    if verbose:
        print(f"detections: {len(teal)} teal blobs, {len(red)} red blobs, {N} rim fits, {time.time() - t0:.1f} s -> {od}")
    return {"teal": teal, "red": red, "rim": rim, "summary": summ}


# --- identity (needs the scene) --------------------------------------------------------

def _pick_schedule(ep_cfg):
    """[(cap, close_first, lift_first, drop_last)] in episode order."""
    out = []
    for pk in ep_cfg.get("picks", []):
        first = pk.get("first_close", pk["close"])[0]
        out.append((pk["cap"], first, pk["lift"][0], pk["drop"][1]))
    return out


def label(ds=None, scene=None, verbose=True) -> np.ndarray:
    """Cap identity + state for every teal blob, from the scene's calibration.

    A cap is 'rest' from the episode start until its pick's lift, then 'held' until the
    last drop frame, then 'in_mug'. Each blob gets the cap whose predicted image
    position is nearest (front: the resting centre / the grasp site while held; grip:
    the same through the wrist camera at that frame's FK), within 12 px (front) or 90 px
    (grip); a blob inside the rim circle after a drop is 'in_mug'. Unmatched blobs keep
    cap '' and state '?' (mat glare, reflections in the mug)."""
    from .. import episodes, kinematics, objects, scene as scene_mod

    ds = paths.dataset(ds)
    sc = scene or scene_mod.load(ds)
    det = load(ds, labels=False)
    teal = det.detections["teal"]
    units = sc.units()
    cams = {c: sc.camera(c) for c in CAMS}
    kin = kinematics.default()
    z = episodes.npz(ds)
    cap_lab = np.full(len(teal), "", dtype="<U2")
    state = np.full(len(teal), "?", dtype="<U7")
    D, h = sc.cap_dims()["diameter"], sc.cap_dims()["height"]
    rim = {int(r["row"]): r for r in det.detections["rim"]}
    for key, ep_cfg in sc["episodes"].items():
        e = int(key)
        sel_ep = np.nonzero(teal["episode"] == e)[0]
        if not len(sel_ep):
            continue
        a = int(z["ep_from"][list(z["episodes"]).index(e)])
        sched = {c: (lift, drop) for c, _, lift, drop in _pick_schedule(ep_cfg)}
        rest_c = {}
        for c in ep_cfg.get("caps", []):
            pos, quat = objects.cap_pose(sc, e, c["id"])
            rest_c[c["id"]] = objects.cap_centre(pos, quat, sc.cap_dims())
        for frame in np.unique(teal["frame"][sel_ep]):
            idx = sel_ep[teal["frame"][sel_ep] == frame]
            row = a + int(frame)
            q = units.to_urdf(z["state"][row])
            T_g = kin.link_T(q, "gripper")
            site = kin.grasp_site(q)
            for cam_i, cam in enumerate(CAMS):
                ii = idx[teal["cam"][idx] == cam_i]
                if not len(ii):
                    continue
                Tp = None if cams[cam].parent == "world" else T_g
                pred, st = {}, {}
                for cid, P in rest_c.items():
                    lift, drop = sched.get(cid, (10**9, 10**9))
                    if frame < lift:
                        pred[cid], st[cid] = P, "rest"
                    elif frame <= drop:
                        pred[cid], st[cid] = site, "held"
                    else:
                        st[cid] = "in_mug"
                uv = {cid: cams[cam].project(P[None], Tp)[0] for cid, P in pred.items()}
                tol = 12.0 if cam == "front" else 90.0
                taken = set()
                for j in ii[np.argsort(-teal["area"][ii])]:
                    b = np.array([teal["u"][j], teal["v"][j]])
                    best, bd = None, tol
                    for cid, p in uv.items():
                        d = np.linalg.norm(b - p)
                        if cid not in taken and np.isfinite(d) and d < bd:
                            best, bd = cid, d
                    if best is not None:
                        cap_lab[j], state[j] = best, st[best]
                        taken.add(best)
                    elif cam == "front" and row in rim and np.isfinite(rim[row]["cx"]):
                        rr = rim[row]
                        if np.hypot(b[0] - rr["cx"], b[1] - rr["cy"]) < rr["r"] and any(s == "in_mug" for s in st.values()):
                            state[j] = "in_mug"
    lab = np.zeros(len(teal), dtype=[("cap", "<U2"), ("state", "<U7")])
    lab["cap"], lab["state"] = cap_lab, state
    od = out_dir(ds, create=True)
    np.savez_compressed(od / "labels.npz", labels=lab, scene_hash=sc.hash)
    if verbose:
        for cam_i, cam in enumerate(CAMS):
            m = teal["cam"] == cam_i
            print(f"labels {cam}: " + ", ".join(f"{s} {int(((state == s) & m).sum())}" for s in ("rest", "held", "in_mug", "?")))
    return lab


# --- the jaw angle the wrist camera sees (needs the scene) ------------------------------

JAW_GRID_DEG = np.arange(-16.0, 46.01, 0.5)
JAW_DS = 4  # masks compared at 160 x 120


def jaw_angles(ds=None, scene=None, verbose=True) -> dict:
    """Per row, the Jaw angle (deg, URDF) whose rendered moving jaw best matches the real
    wrist-view white mask: the camera rides on the gripper, so the jaw's image depends
    on the jaw angle alone. Templates: the calibrated model rendered every 0.5 deg over
    -16..46 deg; per frame the template maximising IoU inside the jaw's region (the
    union of all templates' jaw pixels, minus the fixed finger), refined by a parabola.
    Frames with a bright background (a white desk merges with the jaw) or a held cap
    covering the jaw score a low IoU: use `iou` to filter (> 0.8 is a clean view)."""
    import cv2

    from .fit import render_specs
    from .render import WHITE, ArmRenderer
    from .timing import clock
    from .. import camera as C, scene as scene_mod

    ds = paths.dataset(ds)
    sc = scene or scene_mod.load(ds)
    cam = sc.camera("grip")
    ren = ArmRenderer(render_specs(sc.camera("front")))
    maps = C.remap_maps(cam, ren.specs["grip"][1])
    units = sc.units()
    temps, fixed = [], None
    for deg in JAW_GRID_DEG:
        q = np.zeros(6)
        q[5] = np.radians(deg)
        r = ren.render("grip", q, cam.T_parent_cam)
        jaw = (r.cls == WHITE) & (r.link == 6)
        fix = (r.cls == WHITE) & (r.link != 6)
        temps.append(cv2.remap(jaw.astype(np.uint8), *maps, cv2.INTER_NEAREST))
        fixed = cv2.remap(fix.astype(np.uint8), *maps, cv2.INTER_NEAREST) if fixed is None else fixed
    ren.close()
    down = lambda a: cv2.resize(a.astype(np.float32), (a.shape[1] // JAW_DS, a.shape[0] // JAW_DS),  # noqa: E731
                                interpolation=cv2.INTER_AREA)
    T = np.stack([down(t) for t in temps])  # (A, h, w)
    roi = (T.max(0) > 0.5) & (down(fixed) < 0.1)
    Tf = T[:, roi]  # (A, P)
    with np.load(out_dir(ds) / "masks_grip.npz") as z:
        packed, width = z["packed"], int(z["width"])
    N = len(packed)
    R = np.empty((N, roi.sum()), np.float32)
    for i in range(N):
        R[i] = down(np.unpackbits(packed[i], axis=-1)[:, :width])[roi]
    inter = R @ Tf.T
    union = R.sum(1, keepdims=True) + Tf.sum(1)[None] - inter
    iou = inter / np.maximum(union, 1e-6)
    k = iou.argmax(1)
    best = JAW_GRID_DEG[k].astype(float)
    inner = (k > 0) & (k < len(JAW_GRID_DEG) - 1)
    a, b, c = (iou[np.arange(N), np.clip(k + d, 0, len(JAW_GRID_DEG) - 1)] for d in (-1, 0, 1))
    den = a - 2 * b + c
    best[inner & (den < 0)] += (0.5 * (a - c) / np.where(den < 0, den, -1))[inner & (den < 0)] * 0.5
    z = __import__("real2sim.episodes", fromlist=["x"]).npz(ds)
    pct = z["state"][:, 5].astype(float)
    lag = float(sc["cameras"]["grip"].get("latency", {}).get("frames", 0.0))
    pct_exp = clock(ds).at(np.arange(N), lag)[:, 5]  # the state when the image was exposed
    out = {"row": np.arange(N), "jaw_deg": best, "iou": iou[np.arange(N), k].astype(np.float32),
           "state_pct": pct, "state_jaw_deg": np.degrees(units.jaw_rad(pct)),
           "exposure_jaw_deg": np.degrees(units.jaw_rad(pct_exp)), "latency_frames": lag, "scene_hash": sc.hash}
    np.savez_compressed(out_dir(ds, create=True) / "jaw.npz", **out)
    if verbose:
        ok = out["iou"] > 0.8
        d = (best - out["exposure_jaw_deg"])[ok]
        print(f"jaw: {ok.sum()}/{N} clean frames (IoU > 0.8)" + (
            f"; observed - gripper-map jaw: median {np.median(d):+.2f}, p5..p95 {np.percentile(d, 5):+.2f}.."
            f"{np.percentile(d, 95):+.2f} deg" if ok.any() else " -- is the scene calibrated?"))
    return out


# --- loading ------------------------------------------------------------------------

@dataclass
class Observations:
    ds: str
    detections: dict
    labels: np.ndarray | None

    def teal(self, cam: str, rows=None, cap: str | None = None, episode: int | None = None, state: str | None = None):
        t = self.detections["teal"]
        m = t["cam"] == CAMS.index(cam)
        if rows is not None:
            m &= np.isin(t["row"], np.asarray(list(rows)))
        if episode is not None:
            m &= t["episode"] == episode
        if cap is not None or state is not None:
            if self.labels is None:
                raise ValueError("no labels.npz: run `./robot real2sim calib observe`")
            if cap is not None:
                m &= self.labels["cap"] == cap
            if state is not None:
                m &= self.labels["state"] == state
        out = t[m]
        if self.labels is not None:
            import numpy.lib.recfunctions as rf

            out = rf.merge_arrays([out, self.labels[m]], flatten=True, usemask=False)
        return out

    def rim(self, rows=None):
        r = self.detections["rim"]
        return r if rows is None else r[np.isin(r["row"], np.asarray(list(rows)))]

    def summary(self, cam: str | None = None):
        return self.detections["summary"]

    def jaw(self) -> dict:
        with np.load(out_dir(self.ds) / "jaw.npz") as z:
            return {k: z[k] for k in z.files}

    def mask(self, cam: str, row: int) -> np.ndarray:
        if not hasattr(self, "_masks"):
            self._masks = {}
        if cam not in self._masks:
            with np.load(out_dir(self.ds) / f"masks_{cam}.npz") as z:
                self._masks[cam] = (z["packed"], int(z["width"]))
        p, w = self._masks[cam]
        return np.unpackbits(p[row], axis=-1)[:, :w].astype(bool)


def load(ds=None, labels: bool = True) -> Observations:
    ds = paths.dataset(ds)
    od = out_dir(ds)
    if not (od / "detections.npz").exists():
        raise FileNotFoundError(f"{od}/detections.npz missing: run ./robot real2sim calib observe")
    with np.load(od / "detections.npz") as z:
        det = {k: z[k] for k in ("teal", "red", "rim", "summary")}
        det["meta"] = json.loads(str(z["meta"]))
    lab = None
    if labels and (od / "labels.npz").exists():
        with np.load(od / "labels.npz") as z:
            lab = z["labels"]
    return Observations(ds, det, lab)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="real2sim CALIB: per-frame real observations")
    ap.add_argument("what", nargs="?", default="all", choices=("all", "detect", "label", "jaw"))
    ap.add_argument("--ds")
    a = ap.parse_args()
    if a.what in ("all", "detect"):
        detect(a.ds)
    if a.what in ("all", "label"):
        label(a.ds)
    if a.what in ("all", "label", "jaw"):
        jaw_angles(a.ds)
