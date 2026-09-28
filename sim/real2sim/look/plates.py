"""Front-camera background plates: what the overhead C270 sees with no arm in the way.

PER EPISODE. The per-pixel MEDIAN over every 3rd frame (163-300 frames an episode)
of the front frames. Masked out: the FK arm (LabelRenderer, dilated 12 px for fit6r's
1-3 px error and the cables) and the caps (teal, dilated 12 px, which also takes a
cap's own small shadow). The mug stays in: it did not move in any episode (rim
within 1 px, episodes report). A cap's resting spot is filled by the frames after
its pick. Pixels the arm never uncovers are invalid; the robot base is one. The
median is taken in 8-bit sRGB, which commutes with the monotone decode, through a
sort of (pixels, frames) uint16 with a 999 sentinel for masked samples. That takes
1-2 s an episode, where np.nanmedian takes 20-80 s.

WHITE BALANCE DRIFT (MEASURED). The C270 runs auto white balance. A static mat
patch has an R std of 1.2-2.3 levels on a mean of 14-18 within episodes 1-2, the
G std is 0.3-0.6, and the episodes differ in colour (ep0 16/16/16, ep3 14/18/20).
So plates and frames carry per-channel GAINS: the ratio of linear sums over
well-exposed background pixels (8-bit 12-200), outliers dropped, to a reference plate. Episode
gains are normalised to a geometric mean of 1 over the four episodes. That average
camera state is the REFERENCE every LOOK colour is quoted in, and
frame_gain = frame vs its episode plate x the episode gain.

CLEAN PLATE. The median across the gain-normalised episode plates. At each pixel it
skips the episodes whose mug (model hull dilated 8 px) or mug shadow covers it.
Shadows are found in two passes: episode plate / first-pass clean below 0.80,
within 260 px of the mug and connected to it. The mug sits somewhere different in
each episode, so all but the robot's footprint is covered.
"""

from __future__ import annotations

import time

import numpy as np

from .. import episodes as E
from .. import objects
from . import colour
from .segment import LabelRenderer, dilate

STRIDE = 3
ROBOT_DILATE, TEAL_DILATE = 12, 12
GAIN_RANGE = (12, 200)  # 8-bit reference values used for gains: above the noise, below the glare
GAIN_SUB = 3  # per-frame gains on every 3rd pixel each way: ~30 k background pixels, 4x faster
SHADOW_RATIO, SHADOW_REACH = 0.80, 260


def masked_median_u8(stack, valid, chunk_rows: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median over axis 0 of uint8 (N, H, W, C) where valid (N, H, W).
    Returns (median float32 (H, W, C) in 8-bit units, NaN where no sample; count (H, W))."""
    N, H, W, C = stack.shape
    med = np.full((H, W, C), np.nan, np.float32)
    cnt = valid.sum(0).astype(np.int32)
    for r0 in range(0, H, chunk_rows):
        r1 = min(H, r0 + chunk_rows)
        a = stack[:, r0:r1].astype(np.uint16)
        a[~valid[:, r0:r1]] = 999
        a = np.ascontiguousarray(np.moveaxis(a, 0, -1)).reshape(-1, N)
        a.sort(axis=1)
        c = np.repeat(cnt[r0:r1].reshape(-1), C)
        lo, hi = np.maximum((c - 1) // 2, 0), np.maximum(c // 2, 0)
        rows = np.arange(len(a))
        m = 0.5 * (a[rows, lo].astype(np.float32) + a[rows, hi])
        m[c == 0] = np.nan
        med[r0:r1] = m.reshape(r1 - r0, W, C)
    return med, cnt


def well_exposed(ref_lin) -> np.ndarray:
    """Pixels whose reference is inside GAIN_RANGE (8-bit): above the noise floor, below the glare."""
    r8 = colour.linear_to_srgb(np.nan_to_num(ref_lin)).max(-1)
    return (r8 >= GAIN_RANGE[0]) & (r8 <= GAIN_RANGE[1])


def gain(img_lin, ref_lin, use) -> np.ndarray:
    """Per-channel sum(img) / sum(ref) over the pixels `use` (see well_exposed), after
    dropping pixels whose luminance ratio is off by more than exp(0.4), such as
    shadows, stray cables and glints. A ratio of SUMS, not a median of ratios: on a
    mat at 8-bit 16-20, every per-pixel ratio falls on one of a few quantised values
    and a median sticks to one of them (build-phase probe: episode 0 gave identical
    gains on every frame)."""
    if use.sum() < 500:
        return np.ones(3, np.float32)
    a, b = img_lin[use].astype(np.float64), ref_lin[use].astype(np.float64)
    lr = np.log(np.maximum(colour.luminance(a), 1e-6) / np.maximum(colour.luminance(b), 1e-6))
    keep = np.abs(lr - np.median(lr)) < 0.4
    return (a[keep].sum(0) / np.maximum(b[keep].sum(0), 1e-9)).astype(np.float32)


def mug_hull(scene, cam, ep: int) -> np.ndarray:
    """Model silhouette of the mug (body + handle triangles, projected) in `cam`."""
    import cv2

    from ..transforms import quat_to_mat

    pos, quat = objects.mug_pose(scene, ep)
    R = quat_to_mat(quat)
    m = np.zeros((cam.height, cam.width), np.uint8)
    for part in objects.mug_parts(scene.mug_dims()).values():
        uv = cam.project(np.asarray(part.vertices) @ R.T + pos)
        tri = uv[np.asarray(part.faces)]
        for t in tri[np.isfinite(tri).all((1, 2))]:
            cv2.fillConvexPoly(m, np.rint(t * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    return m > 0


def build(scene, renderer: LabelRenderer | None = None, log=print) -> dict:
    """All plates for scene.ds. Returns the arrays (see module doc) and a summary."""
    import cv2

    t0 = time.time()
    cam = scene.camera("front")
    units = scene.units()
    rnd = renderer or LabelRenderer(scene, cams=("front",))
    rnd.set_mug(None)  # the mug is part of the background here
    eps = E.load(scene.ds, include_excluded=True)
    H, W = cam.height, cam.width
    frame_rel = np.ones((max(ep.stop for ep in eps.values()), 3), np.float32)  # frame vs its episode plate
    lin, valid, counts, hulls, n_used = {}, {}, {}, {}, {}
    for e, ep in eps.items():
        idx = np.arange(0, len(ep), STRIDE)
        stack = np.empty((len(idx), H, W, 3), np.uint8)
        ok = np.empty((len(idx), H, W), bool)
        fr = ep.frames("front")
        for j, k in enumerate(idx):
            im = np.asarray(fr[k])
            robot = rnd.parts(units.to_urdf(ep.state[k]), "front") > 0
            stack[j] = im
            ok[j] = ~(dilate(robot, ROBOT_DILATE) | dilate(colour.teal(im, "front"), TEAL_DILATE))
        med, counts[e] = masked_median_u8(stack, ok)
        valid[e] = counts[e] > 0
        lin[e] = colour.srgb_to_linear(np.nan_to_num(med) / 255.0)
        use = valid[e] & well_exposed(lin[e])
        sub = (slice(None, None, GAIN_SUB), slice(None, None, GAIN_SUB))
        rel = np.array([gain(colour.srgb_to_linear(stack[j][sub]), lin[e][sub], (ok[j] & use)[sub])
                        for j in range(len(idx))])
        near = np.clip(np.searchsorted(idx, np.arange(len(ep))), 0, len(idx) - 1)  # unmeasured: next measured
        frame_rel[ep.start:ep.stop] = rel[near]
        hulls[e], n_used[e] = mug_hull(scene, cam, e), len(idx)
        del stack, ok
        log(f"  plate ep{e}: {len(idx)} frames, valid {valid[e].mean():.3f}  ({time.time() - t0:.1f} s)")
    # episode gains against a joint reference (iterated: reference = median of the normalised plates)
    common = np.logical_and.reduce([valid[e] & ~dilate(hulls[e], 40) for e in eps])
    g = {e: np.ones(3, np.float32) for e in eps}
    for _ in range(3):
        ref = np.median(np.stack([lin[e] / g[e] for e in eps]), 0)
        use = common & well_exposed(ref)
        g = {e: gain(lin[e], ref, use) for e in eps}
        gm = np.exp(np.mean([np.log(v) for v in g.values()], 0))
        g = {e: (v / gm).astype(np.float32) for e, v in g.items()}
    norm = {e: lin[e] / g[e] for e in eps}

    def combine(excl):
        st = np.stack([np.where((valid[e] & ~excl[e])[..., None], norm[e], np.nan) for e in eps])
        out = np.full(st.shape[1:], np.nan, np.float32)
        n = np.isfinite(st[..., 0]).sum(0)
        s = np.sort(np.where(np.isfinite(st), st, np.inf), axis=0)  # NaNs last
        for c in range(1, len(eps) + 1):  # median of the c finite values
            m = n == c
            out[m] = 0.5 * (s[(c - 1) // 2][m] + s[c // 2][m])
        return out

    zone = {e: dilate(hulls[e], 8) for e in eps}
    clean0 = combine(zone)
    y0 = colour.luminance(np.nan_to_num(clean0))
    shadows = {}
    for e in eps:
        ratio = colour.luminance(norm[e]) / np.maximum(y0, 1e-6)
        ys, xs = np.nonzero(hulls[e])
        reach = np.zeros((H, W), np.uint8)
        cv2.circle(reach, (int(xs.mean()), int(ys.mean())), SHADOW_REACH, 1, -1)
        cand = (ratio < SHADOW_RATIO) & valid[e] & np.isfinite(clean0[..., 0]) & (reach > 0) & ~hulls[e]
        n, lab = cv2.connectedComponents((cand | zone[e]).astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(lab[zone[e]])] = True
        keep[0] = False
        shadows[e] = keep[lab] & cand
    clean = combine({e: zone[e] | dilate(shadows[e], 6) for e in eps})
    clean_valid = np.isfinite(clean[..., 0])
    yc = np.maximum(colour.luminance(np.nan_to_num(clean)), 1e-6)
    ratio = {e: np.where(valid[e] & clean_valid, colour.luminance(norm[e]) / yc, np.nan).astype(np.float32)
             for e in eps}
    frame_gain = frame_rel.copy()
    for e, ep in eps.items():
        frame_gain[ep.start:ep.stop] *= g[e]
    fg = {e: frame_gain[ep.start:ep.stop] for e, ep in eps.items()}
    summary = {
        "frames_per_episode": n_used,
        "valid_fraction": {e: round(float(valid[e].mean()), 4) for e in eps},
        "clean_valid_fraction": round(float(clean_valid.mean()), 4),
        "gain_episode": {e: np.round(g[e], 4).tolist() for e in eps},
        "frame_gain_std": {e: np.round(fg[e].std(0), 4).tolist() for e in eps},
        "frame_gain_p5_p95": {e: np.round(np.percentile(fg[e], [5, 95], axis=0), 3).tolist() for e in eps},
        "shadow_px": {e: int(shadows[e].sum()) for e in eps},
        "seconds": round(time.time() - t0, 1),
    }
    if renderer is None:
        rnd.close()
    return {"episodes": sorted(eps), "plate_lin": lin, "valid": valid, "count": counts, "gain_episode": g,
            "norm_lin": norm, "clean_lin": clean, "clean_valid": clean_valid, "mug_hull": hulls,
            "shadow": shadows, "shadow_ratio": ratio, "frame_gain": frame_gain, "summary": summary}
