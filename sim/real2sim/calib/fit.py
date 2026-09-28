"""The staged, joint, dataset-only calibration: robust least squares over every view.

    ./robot real2sim calib fit [--quick] [--train 0,1] [--test 2]

STAGES (each a scipy least_squares, method trf, loss soft_l1 on sigma-normalised
residuals, f_scale 1; silhouette blocks re-rendered and re-associated between
solves, see silhouette.py):

    A  front   camera pose, the 5 joint offsets and the front latency from the arm
               silhouettes; K held at the trusted values, no distortion (MEASURED: on
               held-out ep2 the arm alone cannot tell K or any distortion model apart --
               IoU 0.737-0.750 for every variant while f wanders 930-990 px along the
               f/distance valley).
    B  grip    wrist mount pose + gripper map from the finger silhouettes, starting from
               the pinhole the klip pose was fitted with (fx 487, no distortion) so the
               start already draws the fingers where they are; then + K, k1, k2 and the
               wrist latency (a far start lands the finger ICP in local minima: three
               starts gave three mounts 7-9 mm apart).
    C  objects cap centres, cap D/h/lip, table z, mug centres/yaw/R/H with the cameras
               held: front cap blobs, front rim + handle + red wall, physics.
    D  joint   everything above at once plus the wrist views of the world (resting caps
               during each approach, the mug rim at each release), with the weak priors
               of params.py (+ a 5 % square-pixel prior on the wrist camera); wrist
               frames the first joint solve cannot explain are dropped (reject_outliers).
Then the Gauss-Newton covariance of D (res.jac, scaled by the reduced chi^2 when
above 1) gives every parameter a sigma, and the prior-vs-posterior ratio says which
ones the prior, not the data, decided.

HOLD-OUT (fit_holdout): fit on the train episodes; the test episode's objects are
then fitted from ONE view with every global parameter frozen and scored in the
OTHER view (evaluate.py). Leave-one-episode-out repeats that for each episode.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import chain, masks, objfit, params as P, physfit, silhouette as SIL
from .render import ArmRenderer
from .. import camera as C, episodes as E, paths, scene as S

LOSS = "soft_l1"
GRIP_ASPECT_SIGMA = 0.05  # fy/fx - 1: a weak square-pixel prior (not from any calibration)
KLIP_PINHOLE = (487.1, 487.1, 332.3, 287.1)  # READ mjlab klip_camera.py: the K the klip pose was fitted with


@dataclass
class Config:
    quick: bool = False
    n_front: int = 30  # silhouette frames per episode
    n_grip: int = 24
    n_gripcap: int = 12  # wrist frames per cap
    n_rim: int = 6  # wrist frames per pick over the mug
    outer: int = 3  # ICP re-associations per silhouette stage
    nfev: int = 60
    seed: int = 0
    # 'fixed' (default): the trusted C270 K. MEASURED: freeing it (even with a 0.5 % / 3 px
    # prior) moves cy 38 px and f 30 px along a valley where the image mapping changes by
    # < 3 px (degenerate with the camera tilt and k2), and the held-out episode scores the
    # same (front IoU 0.7486 free vs 0.7506 fixed; cap cross-view 5.4-6.4 vs 5.1-10.0 mm).
    front_K: str = "fixed"  # | 'prior': fx fy cx cy free with the trusted-K prior (params.py)
    front_dist: str = "k12"  # 'none' | 'k12' | 'k5'
    grip_dist: str = "k12"  # 'k12' | 'k5'
    tilt: bool = False  # free the table's slope (dz/dx, dz/dy) in the joint stage

    @classmethod
    def make(cls, quick: bool) -> "Config":
        return cls(quick=True, n_front=14, n_grip=12, n_gripcap=6, n_rim=4, outer=2, nfev=25) if quick else cls()


# --- data ---------------------------------------------------------------------------

def joint_speed(state) -> np.ndarray:
    s = np.asarray(state, dtype=float)
    v = np.gradient(s, axis=0) * 30.0
    return np.abs(v[:, :5]).max(1)


def spread_pick(cand, n, rng) -> list:
    """n indices from sorted `cand`, one per equal-width bin of the list (spread in time)."""
    cand = np.asarray(sorted(cand))
    if len(cand) <= n:
        return list(cand)
    bins = np.array_split(cand, n)
    return [int(rng.choice(b)) for b in bins]


def fresh(frames_img, row) -> bool:
    """False if the camera re-sent the previous row's image (an old exposure: the front
    camera ran at an effective 15 fps in bursts, episodes report)."""
    return row == 0 or not np.array_equal(np.asarray(frames_img[row]), np.asarray(frames_img[row - 1]))


def front_rows(ep, n, rng, frames_img=None) -> list:
    """n slow frames (arm < 8 deg/s) + n/2 moving ones (20-70 deg/s, fresh images only):
    the slow ones fix the geometry, the moving ones the camera latency."""
    sp = joint_speed(ep.state)
    slow = spread_pick(np.nonzero(sp < 8.0)[0], n, rng)
    mov = np.nonzero((sp > 20.0) & (sp < 70.0))[0]
    mov = mov[(mov > 4) & (mov < len(ep) - 4)]
    if frames_img is not None:
        mov = np.array([k for k in mov if fresh(frames_img, ep.start + k)], int)
    return sorted(ep.start + k for k in slow + spread_pick(mov, n // 2, rng))


def grip_rows(ep, n, summary, rng) -> list:
    """Dark-background wrist frames: n at rest-ish gripper spread over closed / holding /
    open, + n/3 with the gripper moving (10-80 %/s) for the latency."""
    g = ep.state[:, 5].astype(float)
    gv = np.abs(np.gradient(g)) * 30.0
    dark = summary["grip_top_bright"][ep.start:ep.stop] < 0.02
    ok = np.nonzero(dark & (gv < 3.0))[0]
    out = []
    for lo, hi, k in ((0, 3, n // 3), (10, 20, n // 3), (20, 60, n - 2 * (n // 3))):
        c = ok[(g[ok] >= lo) & (g[ok] < hi)]
        out += spread_pick(c, k, rng)
    mv = np.nonzero(dark & (gv > 10.0) & (gv < 80.0))[0]
    mv = mv[(mv > 4) & (mv < len(ep) - 4)]
    out += spread_pick(mv, n // 3, rng)
    return sorted(ep.start + k for k in out)


def front_cap_obs(scene, eps, det, ep_ids) -> list:
    """Resting caps: blobs within 6 px of the config seed pixel, from the episode start
    until the gripper first closes on that cap, kept while the blob area stays within
    15 % of its first-30-frame median (the arm has not reached it yet)."""
    teal = det["teal"]
    out = []
    for e in ep_ids:
        ep, cfg = eps[e], scene.episode(e)
        firsts = {pk["cap"]: pk.get("first_close", pk["close"])[0] for pk in cfg["picks"]}
        for c in cfg["caps"]:
            px = np.array(c["px"], float)
            m = (teal["cam"] == 0) & (teal["episode"] == e) & (teal["frame"] < firsts[c["id"]] - 5)
            t = teal[m]
            t = t[np.hypot(t["u"] - px[0], t["v"] - px[1]) < 6.0]
            if len(t) < 5:
                continue
            a0 = np.median(t["area"][t["frame"] < t["frame"].min() + 30])
            t = t[np.abs(t["area"] - a0) < 0.15 * a0]
            out.append(objfit.FrontCapObs(e, c["id"], c["up"], float(np.mean(t["u"])), float(np.mean(t["v"])),
                                          float(np.mean(t["r_eq"])), int(len(t))))
    return out


def front_mug_obs(scene, eps, det, ep_ids) -> list:
    """Per episode: the median unoccluded rim circle, and the handle = the SECOND
    largest red blob near the mug (the largest is the body's outer-wall crescent)."""
    rim, red = det["rim"], det["red"]
    out = []
    for e in ep_ids:
        ep = eps[e]
        r = rim[(rim["row"] >= ep.start) & (rim["row"] < ep.stop) & (rim["occluded"] < 0.05)]
        if len(r) < 10:
            continue
        cx, cy, rr = float(np.median(r["cx"])), float(np.median(r["cy"])), float(np.median(r["r"]))
        hs = []
        for row in r["row"][:: max(1, len(r) // 60)]:
            b = red[red["row"] == row]
            b = b[np.hypot(b["u"] - cx, b["v"] - cy) < 2.5 * rr]
            if len(b) >= 2:
                b = np.sort(b, order="area")[::-1]
                h = b[1]
                if 200 < h["area"] < 1500 and 1.05 * rr < np.hypot(h["u"] - cx, h["v"] - cy) < 2.0 * rr:
                    hs.append((h["u"], h["v"]))
        huv = tuple(np.median(np.array(hs), 0)) if len(hs) >= 5 else None
        out.append(objfit.FrontMugObs(e, cx, cy, rr, huv, int(len(r))))
    return out


def grip_cap_frames(scene, eps, model, ep_ids, n_per_cap, rng) -> list:
    """Wrist frames of each resting cap during its approach: the teal blob nearest the
    cap's predicted wrist-image position (model at the current stage), within 150 px,
    and MUTUALLY nearest: no other cap still on the mat is predicted closer to that
    blob (MEASURED: without this, 1 in 4 episode-1 frames took a neighbouring cap,
    100 px off). Blob area >= 1500 px, clear of the frame by 8 px (a cut blob's
    outline is the frame's), arm slower than 30 deg/s (motion blur; the latency itself
    is fitted), and at least 8 frames before the jaw first closes."""
    G = E.frames(scene.ds, "grip")
    cam = model.camera("grip")
    _, h = model.cap_dims()
    out = []
    for e in ep_ids:
        ep, cfg = eps[e], scene.episode(e)
        ups = {c["id"]: c["up"] for c in cfg["caps"]}
        lifted = {pk["cap"]: pk["lift"][0] for pk in cfg["picks"]}
        centres = {c["id"]: np.array([*model.cap_xy(e, c["id"]), model.table_z(model.cap_xy(e, c["id"])) + h / 2])
                   for c in cfg["caps"]}
        prev_drop = 0
        sp = joint_speed(ep.state)
        for pk in cfg["picks"]:
            cid = pk["cap"]
            first = pk.get("first_close", pk["close"])[0]
            ks = np.arange(prev_drop + 5, first - 8)  # the descending fingers can nudge it later
            ks = ks[sp[ks] < 30.0]  # latency is modelled (timing.py); faster frames are blurred
            prev_drop = pk["drop"][1]
            if not len(ks):
                continue
            Tg = chain.link_T(model.q(ep.state[ks]))["gripper"]
            cands = []
            for k, T in zip(ks, Tg):
                on_mat = [c for c in centres if lifted.get(c, 10**9) > k]
                pred = {c: cam.project(centres[c][None], T)[0] for c in on_mat}
                p = pred[cid]
                if not (np.isfinite(p).all() and -40 < p[0] < 680 and -40 < p[1] < 520):
                    continue
                img = np.asarray(G[ep.start + k])
                comps = masks.components(masks.teal(img, "grip"), 1500)
                if not comps:
                    continue
                j = int(np.argmin([np.hypot(*(c[1] - p)) for c in comps]))
                b = comps[j][1]
                x, y, w, h_ = comps[j][2]
                edge = x < 8 or y < 8 or x + w > img.shape[1] - 8 or y + h_ > img.shape[0] - 8
                d_own = np.hypot(*(b - p))
                d_other = min([np.hypot(*(b - q)) for c, q in pred.items() if c != cid and np.isfinite(q).all()],
                              default=np.inf)
                if d_own < 150 and d_own < d_other and not edge:
                    cands.append((k, b))
            if not cands:
                continue
            for i in spread_pick(np.arange(len(cands)), n_per_cap, rng):
                k, buv = cands[i]
                f = objfit.grip_cap_frame(ep.start + k, e, cid, ups[cid], ep.state[k], np.asarray(G[ep.start + k]), buv)
                if f is not None:
                    out.append(f)
    return out


def wrist_rim_frames(scene, eps, ep_ids, n_per_pick, rng) -> list:
    """Wrist frames over the mug: each pick's release range and the 10 frames before it,
    arm slower than 40 deg/s (the latency is fitted), spread over the range."""
    G = E.frames(scene.ds, "grip")
    out = []
    for e in ep_ids:
        ep, cfg = eps[e], scene.episode(e)
        sp = joint_speed(ep.state)
        for pk in cfg["picks"]:
            ks = np.arange(max(0, pk["release"][0] - 10), pk["drop"][1] + 1)
            ks = ks[sp[ks] < 40.0]
            for k in spread_pick(ks, n_per_pick, rng):
                out.append(objfit.wrist_rim_frame(ep.start + k, e, ep.state[k], np.asarray(G[ep.start + k])))
    return out


@dataclass
class Data:
    ep_ids: list
    front_sil: SIL.SilhouetteBlock
    grip_sil: SIL.SilhouetteBlock
    front_caps: objfit.FrontCaps
    front_mug: objfit.FrontMug
    phys: physfit.Physical
    grip_caps: objfit.GripCaps | None = None
    wrist_rim: objfit.WristRim | None = None
    front_body: objfit.FrontBody | None = None
    lens_dontcare: dict = field(default_factory=dict)


def build_data(scene, eps, ep_ids, cfg: Config, model, renderer: ArmRenderer, rng) -> Data:
    import cv2

    from .observations import load as load_obs

    obs = load_obs(scene.ds, labels=False)
    det = obs.detections
    F, G = E.frames(scene.ds, "front"), E.frames(scene.ds, "grip")
    # front don't-care: static white where the model robot is not (initial calibration)
    front = model.camera("front")
    maps = C.remap_maps(front, renderer.specs["front"][1])
    robot, rows_by_ep = {}, {}
    for e in ep_ids:
        ep = eps[e]
        rr = list(range(ep.start, ep.stop, max(1, len(ep) // 60)))
        rows_by_ep[e] = rr
        q = model.q(ep.state[np.array(rr) - ep.start])
        for r, qq in zip(rr, q):
            rd = renderer.render("front", qq, model.T_parent_cam("front"))
            m = cv2.remap((rd.cls > 0).astype(np.uint8), *maps, cv2.INTER_NEAREST)
            robot[r] = cv2.dilate(m, np.ones((31, 31), np.uint8)) > 0
    dc = SIL.front_dontcare(F, rows_by_ep, robot)
    fr_rows, gr_rows = [], []
    for e in ep_ids:
        fr_rows += front_rows(eps[e], cfg.n_front, rng, F)
        gr_rows += grip_rows(eps[e], cfg.n_grip, det["summary"], rng)
    ep_of = {r: e for e in ep_ids for r in range(eps[e].start, eps[e].stop)}
    state = E.npz(scene.ds)["state"].astype(float)
    front_sil = SIL.SilhouetteBlock("front", [SIL.make_frame("front", r, np.asarray(F[r]), dc[ep_of[r]]) for r in fr_rows],
                                    ds=scene.ds)
    grip_sil = SIL.SilhouetteBlock("grip", [SIL.make_frame("grip", r, np.asarray(G[r])) for r in gr_rows],
                                   sdf_clamp=120.0, max_assoc=40.0, ds=scene.ds)
    mugs = front_mug_obs(scene, eps, det, ep_ids)
    body = objfit.FrontBody([objfit.front_body_obs(F, it.ep, list(range(eps[it.ep].start, eps[it.ep].start + 30, 3)),
                                                   (it.cx, it.cy, it.r)) for it in mugs])
    return Data(list(ep_ids), front_sil, grip_sil, objfit.FrontCaps(front_cap_obs(scene, eps, det, ep_ids)),
                objfit.FrontMug(mugs), physfit.Physical(scene, eps, ep_ids), front_body=body)


def render_specs(front) -> dict:
    """The fixed pinhole renders the silhouettes are lifted from (camera.py RenderSpec).
    They only need to COVER every ray the real image can have while the lens is being
    fitted, at about the real pixel density:
        front  the trusted K with a +0.3 k1 margin (render_spec, 40 px margin)
        grip   f' 640 px over 1340 x 1070: normalised +-1.05 x +-0.84, which holds the
               klip pinhole start (fx 487, +-0.66 x -0.59..+0.40) and a k1 -0.7 barrel
               lens of fx 700 (corners at ~+-0.9 x +-0.7) alike."""
    return {"front": ("world", C.render_spec(front.replace(dist=(0.3, 0, 0, 0, 0)), margin=40)),
            "grip": ("gripper", C.RenderSpec(1340, 1070, (640.0, 640.0, 669.5, 534.5)))}


def lens_disks(model, block: SIL.SilhouetteBlock, radius: float = 28.0) -> dict:
    """{row: [(u, v, r)]}: the wrist lens (webcam body + LED) in the front view."""
    front = model.camera("front")
    Tg = chain.link_T(model.q(block.states(model)))["gripper"]
    c = (Tg @ model.T_parent_cam("grip"))[:, :3, 3]
    uv = front.project(c)
    return {fr.row: [(float(u), float(v), radius)] for fr, (u, v) in zip(block.frames, uv)}


# --- solving --------------------------------------------------------------------------

def residual_fn(layout, base, meta, terms):
    """terms: [(name, fn(model) -> r, weight)]; returns f(xs) for least_squares."""
    sl = layout.slices()

    def fun(xs):
        full = layout.from_solver(xs, base=base)
        m = P.Model(layout, full, meta)
        out = [w * fn(m) for _, fn, w in terms]
        out.append(layout.prior_residuals(full))
        if layout.block("grip.f").free:
            fx, fy = full[sl["grip.f"]]
            out.append(np.array([(fy / fx - 1.0) / GRIP_ASPECT_SIGMA]))
        return np.concatenate(out)

    return fun


def pin_terms(layout, pins: dict):
    """Tiny-sigma pulls that hold individual entries (e.g. p1 p2 k3) at their x0."""
    sl = layout.slices()

    def fn(m):
        return np.concatenate([(m.v[sl[name]][list(idx)] - layout.block(name).x0[list(idx)]) / 1e-6
                               for name, idx in pins.items()]) if pins else np.zeros(0)

    return fn


_FUN = None  # the residual function forked Jacobian workers evaluate (set per solve)


def _jac_cols(args):
    x, f0, cols, h = args
    out = np.empty((len(f0), len(cols)))
    for k, j in enumerate(cols):
        xj = x.copy()
        xj[j] += h[j]
        out[:, k] = (_FUN(xj) - f0) / h[j]
    return out


class ParallelJac:
    """Forward-difference Jacobian, columns spread over forked workers.

    scipy's '2-point' rule (h = diff_step * max(1, |x|), sign of x) evaluated in
    parallel: the workers inherit the residual function and its data by fork, so
    nothing is pickled but x. The last fun(x) is cached, as least_squares always asks
    for jac(x) right after fun(x)."""

    def __init__(self, fun, lo, hi, workers: int, diff_step: float = 1e-3):
        import multiprocessing as mp

        global _FUN
        _FUN = fun
        self.fun, self.lo, self.hi, self.ds = fun, lo, hi, diff_step
        self.pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
        self.workers = workers
        self._last = (None, None)

    def f(self, x):
        v = self.fun(x)
        self._last = (x.copy(), v)
        return v

    def __call__(self, x):
        x0, f0 = self._last
        if x0 is None or not np.array_equal(x0, x):
            f0 = self.fun(x)
        h = self.ds * np.where(x >= 0, 1.0, -1.0) * np.maximum(1.0, np.abs(x))
        h = np.where((x + h > self.hi) | (x + h < self.lo), -h, h)  # stay inside the bounds
        cols = [c for c in np.array_split(np.arange(len(x)), max(1, self.workers * 2)) if len(c)]
        if self.pool is None:
            parts = [_jac_cols((x, f0, c, h)) for c in cols]
        else:
            parts = self.pool.map(_jac_cols, [(x, f0, c, h) for c in cols])
        return np.concatenate(parts, axis=1)

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()


WORKERS = int(os.environ.get("R2S_CALIB_WORKERS", min(12, os.cpu_count() or 4)))


def solve(layout, full, meta, terms, nfev, verbose=None, tag=""):
    from scipy.optimize import least_squares

    layout = layout.rebase(full)
    full = layout.full0()
    fun = residual_fn(layout, full, meta, terms)
    t = time.time()
    lo, hi = layout.bounds()
    x0 = np.clip(layout.to_solver(full), lo + 1e-9, hi - 1e-9)
    jac = ParallelJac(fun, lo, hi, WORKERS)
    try:
        res = least_squares(jac.f, x0, jac=jac, method="trf", loss=LOSS, f_scale=1.0, max_nfev=nfev,
                            bounds=(lo, hi), x_scale=1.0)
    finally:
        jac.close()
    full = layout.from_solver(res.x, base=full)
    if verbose:
        verbose(f"    {tag}: cost {res.cost:.1f} ({len(res.fun)} res, {res.nfev} evals, {time.time() - t:.1f} s)")
    return layout, full, res


@dataclass
class FitResult:
    layout: P.Layout
    full: np.ndarray
    meta: dict
    train: list
    data: Data
    res: object
    sigma: dict
    timings: dict
    log: list

    def model(self) -> P.Model:
        return P.Model(self.layout, self.full, self.meta)

    def save(self, path):
        """Pickle everything but the observation blocks (they are rebuilt from the data)."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump({"layout": self.layout, "full": self.full, "meta": self.meta, "train": self.train,
                         "sigma": self.sigma, "timings": self.timings, "log": self.log,
                         "cost": float(self.res.cost) if self.res is not None else None}, f)

    @classmethod
    def load(cls, path) -> "FitResult":
        import pickle

        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["layout"], d["full"], d["meta"], d["train"], None, None, d["sigma"], d["timings"], d["log"])


def _front_terms(d: Data):
    return [("front_sil", d.front_sil.residuals, 1.0)]


def _grip_terms(d: Data):
    return [("grip_sil", d.grip_sil.residuals, 1.0)]


def _object_terms(d: Data):
    t = [("front_caps", d.front_caps.residuals, 1.0), ("front_mug", d.front_mug.residuals, 1.0),
         ("phys", d.phys.residuals, 1.0)]
    if d.front_body is not None and d.front_body.items:
        t.append(("front_body", d.front_body.residuals, 1.0))
    if d.grip_caps is not None and d.grip_caps.frames:
        t.append(("grip_caps", d.grip_caps.residuals, 1.0))
    if d.wrist_rim is not None and d.wrist_rim.frames:
        t.append(("wrist_rim", d.wrist_rim.residuals, 1.0))
    return t


GRIPCAP_OUTLIER_PX = 10.0  # median edge distance of a wrist cap frame after the first joint solve
RIM_OUTLIER_PX = 15.0


def reject_outliers(data, model, say):
    """Drop wrist frames the first joint solve cannot explain: a cap blob merged with a
    reflection or smeared by motion, a rim frame whose ridges locked onto the steel's
    reflections. The rest of the frame set then decides without them."""
    if data.grip_caps is not None and data.grip_caps.frames:
        med = []
        for fr, poly in zip(data.grip_caps.frames, data.grip_caps.polygons(model)):
            d = np.abs(objfit.polygon_sdf(fr.pts, poly)) if poly is not None else np.array([1e3])
            med.append(np.median(d))
        med = np.array(med)
        # a frame is an outlier against ITS CAP's other frames: if every frame of a cap is
        # off, the cap's position is what is wrong, and the frames must stay to fix it
        caps = np.array([f"{f.ep}.{f.cap}" for f in data.grip_caps.frames])
        typical = {c: np.median(med[caps == c]) for c in set(caps)}
        keep = [m_ < max(GRIPCAP_OUTLIER_PX, 3.0 * typical[c]) for m_, c in zip(med, caps)]
        n0 = len(keep)
        data.grip_caps = objfit.GripCaps([f for f, k in zip(data.grip_caps.frames, keep) if k])
        say(f"    outliers: wrist cap frames {n0} -> {len(data.grip_caps.frames)}")
    if data.wrist_rim is not None and data.wrist_rim.frames:
        uvs = data.wrist_rim._rim_uv(model)
        med = []
        for fr, uv in zip(data.wrist_rim.frames, uvs):
            if fr.th is None or len(fr.th) < 8:
                med.append(np.inf)
                continue
            med.append(np.median(np.abs(np.einsum("ni,ni->n", fr.n, fr.uv - uv[fr.th]))))
        med = np.array(med)
        # as for the caps: relative to the block's own typical frame, so a mug that starts
        # far off keeps its frames (MEASURED: an absolute 15 px cut dropped all 18 rim frames
        # of one fold, and its table then sank 17 mm under the fingertips)
        typical = np.median(med[np.isfinite(med)]) if np.isfinite(med).any() else np.inf
        keep = list(med < max(RIM_OUTLIER_PX, 3.0 * typical))
        n0 = len(keep)
        data.wrist_rim = objfit.WristRim([f for f, k in zip(data.wrist_rim.frames, keep) if k])
        say(f"    outliers: wrist rim frames {n0} -> {len(data.wrist_rim.frames)}")


def object_names(layout, ep_ids) -> list:
    return [n for n in layout.names() if any(n.startswith(p) for p in object_blocks(ep_ids))]


def object_blocks(ep_ids) -> list:
    """Name prefixes of the per-episode object blocks of these episodes."""
    return [f"cap.{e}." for e in ep_ids] + [f"mug.{e}" for e in ep_ids]


def fit(ds=None, train=(0, 1, 2), cfg: Config | None = None, verbose=print) -> FitResult:
    cfg = cfg or Config()
    t_all = time.time()
    rng = np.random.default_rng(cfg.seed)
    scene = S.load(ds)
    eps = E.load(scene.ds)
    layout, meta = P.from_scene(scene)
    log, timings = [], {}

    def say(msg):
        log.append(msg)
        if verbose:
            verbose(msg)

    # initial guesses (RULE 1): K/dist from the config, no front distortion
    layout.block("front.k").x0 = np.zeros(5)
    layout.block("front.k").mu = np.zeros(5)
    klip = layout.block("grip.t").x0.copy()
    full = layout.full0()
    model = P.Model(layout, full, meta)
    front = model.camera("front")
    renderer = ArmRenderer(render_specs(front))
    t = time.time()
    data = build_data(scene, eps, list(train), cfg, model, renderer, rng)
    timings["data"] = time.time() - t
    say(f"data: {len(data.front_sil.frames)} front + {len(data.grip_sil.frames)} wrist silhouette frames, "
        f"{len(data.front_caps.items)} front caps, {len(data.front_mug.items)} mugs ({timings['data']:.0f} s)")
    pins_front = pin_terms(layout, {"front.k": (0, 1, 2, 3, 4)})

    # ---- A: front pose + offsets
    t = time.time()
    layout.only_free(["front.rot", "front.t", "robot.off", "front.lag"])
    for it in range(cfg.outer + 1):
        m = P.Model(layout, full, meta)
        data.front_sil.associate(m, renderer, lens_disks(m, data.front_sil))
        layout, full, res = solve(layout, full, meta, _front_terms(data), cfg.nfev, say, f"A{it}")
    timings["A"] = time.time() - t

    # ---- B: wrist mount + gripper map. Start from the pinhole the klip pose was fitted
    # with (klip_camera.py: 66.606 deg hfov = fx 487.1, principal point (332.3, 287.1),
    # no distortion), where the pose already draws the fingers where they are, and free
    # the lens only after the pose has settled: a far start (the pose with the ~680 px
    # ChArUco guess) lands the finger ICP in local minima (MEASURED: three starts, three
    # different mounts 7-9 mm apart).
    t = time.time()
    fb = full.copy()
    sl = layout.slices()
    fb[sl["grip.f"]], fb[sl["grip.c"]], fb[sl["grip.k"]] = KLIP_PINHOLE[:2], KLIP_PINHOLE[2:], 0.0
    layout = layout.rebase(fb)
    full = layout.full0()
    layout.only_free(["grip.rot", "grip.t", "robot.grip"])
    layout.block("grip.t").lo = layout.block("grip.t").hi = None
    for it in range(cfg.outer + 1):
        data.grip_sil.associate(P.Model(layout, full, meta), renderer)
        layout, full, res = solve(layout, full, meta, _grip_terms(data), cfg.nfev, say, f"B pose {it}")
    layout.only_free(["grip.rot", "grip.t", "robot.grip", "grip.f", "grip.c", "grip.k", "grip.lag"])
    pins_grip = pin_terms(layout, {"grip.k": (2, 3, 4)})
    for it in range(cfg.outer + 1):
        data.grip_sil.associate(P.Model(layout, full, meta), renderer)
        layout, full, res = solve(layout, full, meta, _grip_terms(data) + [("pin", pins_grip, 1.0)], cfg.nfev, say, f"B+K {it}")
    timings["B"] = time.time() - t

    # ---- C: objects with cameras held (front view + physics; the wrist caps join in D,
    # once the wrist lens is free to move along its dolly-zoom valley)
    t = time.time()
    layout.only_free(["table.z", "cap.dims", "cap.lip", "mug.dims"] + object_blocks(train))
    layout, full, res = solve(layout, full, meta, _object_terms(data), cfg.nfev * 2, say, "C (front + physics)")
    m = P.Model(layout, full, meta)
    data.grip_caps = objfit.GripCaps(grip_cap_frames(scene, eps, m, train, cfg.n_gripcap, rng), scene.ds)
    data.wrist_rim = objfit.WristRim(wrist_rim_frames(scene, eps, train, cfg.n_rim, rng), scene.ds)
    say(f"  C: {len(data.grip_caps.frames)} wrist cap frames, {len(data.wrist_rim.frames)} wrist rim frames")
    timings["C"] = time.time() - t

    # ---- D: joint
    t = time.time()
    free = ["front.rot", "front.t", "front.lag", "grip.", "robot.", "table.z", "cap.dims", "cap.lip", "mug.dims"]
    free += ["table.tilt"] if cfg.tilt else []
    free += ["front.f", "front.c"] if cfg.front_K == "prior" else []
    free += ["front.k"] if cfg.front_dist != "none" else []
    layout.only_free(free + object_blocks(train))
    pin = {"k12": (2, 3, 4), "k5": (), "none": (0, 1, 2, 3, 4)}
    pins = pin_terms(layout, {k: v for k, v in (("front.k", pin[cfg.front_dist]), ("grip.k", pin[cfg.grip_dist])) if v})
    for it in range(cfg.outer):
        m = P.Model(layout, full, meta)
        data.front_sil.associate(m, renderer, lens_disks(m, data.front_sil))
        data.grip_sil.associate(m, renderer)
        data.wrist_rim.associate(m, renderer, search=60.0 if it == 0 else objfit.RIM_SEARCH_PX)
        say(f"    D{it}: wrist rim points {data.wrist_rim.n_points()}")
        terms = _front_terms(data) + _grip_terms(data) + _object_terms(data) + [("pin", pins, 1.0)]
        layout, full, res = solve(layout, full, meta, terms, cfg.nfev, say, f"D{it}")
        if it == 0:
            reject_outliers(data, P.Model(layout, full, meta), say)
    timings["D"] = time.time() - t
    sigma = covariance(layout, full, res)
    m = P.Model(layout, full, meta)
    summ = {}
    for name, fn, _ in terms:
        if name == "pin":
            continue
        r = fn(m)
        summ[name] = {"n": int(len(r)), "median_sigma": float(np.median(np.abs(r))) if len(r) else None,
                      "rms_sigma": float(np.sqrt(np.mean(r ** 2))) if len(r) else None}
    say("  residuals (in sigmas): " + json.dumps({k: (v["n"], round(v["median_sigma"] or 0, 2)) for k, v in summ.items()}))
    timings["blocks"] = summ
    timings["total"] = time.time() - t_all
    renderer.close()
    return FitResult(layout, full, meta, list(train), data, res, sigma, timings, log)


def covariance(layout, full, res) -> dict:
    """{block: sigma (value units)} from the Gauss-Newton J^T J of the last solve,
    scaled by the reduced chi^2 when that exceeds 1; fixed blocks get 0."""
    J = res.jac
    m, n = J.shape
    chi2 = 2 * res.cost / max(m - n, 1)
    try:
        cov = np.linalg.pinv(J.T @ J) * max(chi2, 1.0)
    except np.linalg.LinAlgError:
        cov = np.full((n, n), np.nan)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None)) * layout.steps()[layout.free_mask()]
    out, i = {}, 0
    for b in layout.blocks:
        if b.free:
            out[b.name] = sd[i:i + b.n]
            i += b.n
        else:
            out[b.name] = np.zeros(b.n)
    out["_chi2"] = np.array([chi2])
    return out


def job(train, test, out_dir, quick=False, ds=None, overrides=None) -> dict:
    """One fold: fit on `train`, score on `test` (evaluate.py) and scan both cameras'
    latency on the train episodes; writes fit.pkl, eval.json and log.txt to out_dir."""
    from pathlib import Path

    from . import evaluate as EV

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.make(quick)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    logf = open(out / "log.txt", "w")

    def say(msg):
        logf.write(msg + "\n")
        logf.flush()

    r = fit(ds, list(train), cfg, verbose=say)
    r.save(out / "fit.pkl")
    res = {"train": list(train), "test": list(test), "config": cfg.__dict__, "timings": {
        k: v for k, v in r.timings.items() if k != "blocks"}, "blocks": r.timings.get("blocks")}
    if test:
        t = time.time()
        res["eval"] = EV.evaluate(r, list(test), cfg, verbose=say)
        res["timings"]["eval"] = time.time() - t
    (out / "eval.json").write_text(json.dumps(res, indent=1, default=_jsonable))
    logf.close()
    return res


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


if __name__ == "__main__":
    import argparse

    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser(description="real2sim CALIB: one fit (+ held-out evaluation)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train", default="0,1,2")
    ap.add_argument("--test", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds")
    a = ap.parse_args()
    job([int(x) for x in a.train.split(",")], [int(x) for x in a.test.split(",") if x], a.out, a.quick, a.ds)
    sys.stdout.flush()
    os._exit(0)  # EGL contexts error out in interpreter teardown (harmless, but noisy)
