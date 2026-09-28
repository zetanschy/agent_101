"""The table plane as a texture: observed radiance, de-lit albedo, and where each texel came from.

GRID (`Grid`). The plane z = scene.table_z() in the base frame, x -0.90..0.70 by y
-1.10..0.25 m. That spans the front view (x -0.31..0.41, y -0.50..0.04 at table
height; cameras report) and the wrist camera's reach at carry height (0.53-0.66 m out
on -x, past y -0.55; episodes report). Row r, col c centre = (x0 + (c + .5) res,
y1 - (r + .5) res). Image up is base +y, image right is base +x, and UV (0, 0) sits
at (x0, y0).

FRONT (1 mm). The clean plate (plates.py), rectified: every texel centre is projected
through the scene's front camera, distortion included, and the plate sampled
bilinearly there. At the table the camera resolves 1.09 mm/px (cameras report), so
1 mm texels keep every pixel.

WRIST (2 mm). Every 4th wrist frame whose camera moves under 0.30 m/s (no motion
blur) and looks down, projected onto the plane through FK(observation.state) and the
scene's grip camera, again texel -> pixel. Masked out: where the fingers ever appear
(GripFingerPrior, any opening, dilated 6 px), teal (+10 px), red (+20 px) and the
mug model (+30 px). Each sample is weighted cos(incidence) / distance^2, and faded
over the image's outer 40 px, where the lens model is extrapolated. The wrist camera
runs its own auto exposure and white balance and saturates colour more than the
C270, so every frame goes into the FRONT reference state through
colour.grip_to_front. Its white balance comes from the static finger core in the
same frame, sigma is global (materials.grip_colour_model), and its exposure is the
trimmed luminance ratio of sums against the front texture over the plain mat both
see. Frames that see no front-covered mat take the exposure of their nearest
measured neighbour. The mosaic is a robust mean: pass 1 gives the weighted mean and
spread, pass 2 drops samples beyond 3 sigma (moving shadows, glints).

MERGE (1 mm). Front where the front sees, feathered over 10 mm into the wrist mosaic,
then the wrist mosaic. Holes up to 30 mm from observed texels get a pyramid fill;
everything further out gets the plain mat's median radiance. Nothing ever sees under
the robot base. `source`: 1 front, 2 wrist, 0 filled.

ALBEDO. albedo = rho_PLA (L - specular) / L_PLA, per channel, as in materials.py:
L_PLA is the lit horizontal white PLA in the same front reference state, and
rho_PLA = 0.80 (assumed). The specular is the gloss fit's GGX lobe for the front
camera's viewpoint (lighting.py), refitted per channel here. It is removed from
front texels only. Where it exceeds 40 % of L (inside the glare), the texel takes the
plain-mat median albedo times its own fine detail (Y / blur(Y, 15 mm)). The wrist texels see the mat from many angles,
and their robust mean already drops most glints. What remains on the plain mat is a
smooth gradient: plain-mat albedo p5/p50/p95 of 0.004 / 0.007 / 0.014 in G across
the front view. A second, broader GGX lobe does not explain it (build-phase probe: rms
0.01007 against 0.01008 with one lobe). So it is a diffuse irradiance variation
that the distant-source model lacks, such as a nearer edge of the window or panel.
It is left in the albedo as baked diffuse light, which is view-independent and
therefore right for both cameras.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from .. import episodes as E
from ..kinematics import default as default_kinematics
from . import colour
from .segment import LABELS, PART_TO_LABEL, dilate

EXTENT = (-0.90, 0.70, -1.10, 0.25)  # x0, x1, y0, y1 (m), base frame
RES_FRONT, RES_WRIST = 0.001, 0.002
WRIST_STRIDE, WRIST_MAX_SPEED = 4, 0.30  # frames; m/s of the camera centre
BORDER_FADE_PX = 40
FEATHER_M = 0.010
SPEC_REPLACE = 0.4  # specular share of L above which the glare texel takes the plain-mat albedo


@dataclass(frozen=True)
class Grid:
    x0: float
    x1: float
    y0: float
    y1: float
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        return int(round((self.y1 - self.y0) / self.res)), int(round((self.x1 - self.x0) / self.res))

    def xy(self, r0=0, r1=None, c0=0, c1=None) -> tuple[np.ndarray, np.ndarray]:
        R, Cn = self.shape
        r = np.arange(r0, R if r1 is None else r1)
        c = np.arange(c0, Cn if c1 is None else c1)
        return np.meshgrid(self.x0 + (c + 0.5) * self.res, self.y1 - (r + 0.5) * self.res)

    def rc(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Continuous (row, col) of points (x, y); texel centres are integers."""
        return (self.y1 - np.asarray(y)) / self.res - 0.5, (np.asarray(x) - self.x0) / self.res - 0.5

    def to_json(self, z: float) -> dict:
        return {**asdict(self), "z": z, "shape": list(self.shape),
                "layout": "row r, col c centre = (x0 + (c+.5) res, y1 - (r+.5) res); UV (0,0) at (x0, y0)"}


def rectify(img_lin, valid, cam, grid: Grid, z: float, T_world_parent=None) -> tuple[np.ndarray, np.ndarray]:
    """A camera image resampled onto the table grid: (texels (R, C, 3) float32, valid (R, C))."""
    import cv2

    X, Y = grid.xy()
    uv = cam.project(np.stack([X, Y, np.full_like(X, z)], -1), T_world_parent).astype(np.float32)
    mx, my = np.nan_to_num(uv[..., 0], nan=-1e4), np.nan_to_num(uv[..., 1], nan=-1e4)
    tex = cv2.remap(np.ascontiguousarray(np.nan_to_num(img_lin).astype(np.float32)), mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ok = cv2.remap(np.asarray(valid, np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                   borderValue=0) > 0.999
    tex[~ok] = np.nan
    return tex, ok


def fill(tex, ok, constant, reach_texels: int = 30, levels: int = 5) -> np.ndarray:
    """Fill texels where ~ok: within `reach_texels` of observed ones by normalised
    convolution over a coarsening pyramid (a seamless continuation), beyond that with
    `constant` (nothing was observed there; look.json says what it is)."""
    import cv2

    out = np.where(ok[..., None], tex, 0).astype(np.float32)
    near = ~ok & (cv2.distanceTransform((~ok).astype(np.uint8), cv2.DIST_L2, 5) <= reach_texels)
    todo = near.copy()
    src = np.where(ok[..., None], tex, 0).astype(np.float32)
    for lv in range(1, levels + 1):
        if not todo.any():
            break
        s = 2 ** lv
        small = (max(out.shape[1] // s, 1), max(out.shape[0] // s, 1))
        w = cv2.GaussianBlur(cv2.resize(ok.astype(np.float32), small, interpolation=cv2.INTER_AREA), (5, 5), 1.0)
        v = cv2.GaussianBlur(cv2.resize(src, small, interpolation=cv2.INTER_AREA), (5, 5), 1.0)
        big_w = cv2.resize(w, out.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        big_v = cv2.resize(v, out.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        got = todo & (big_w > 1e-3)
        out[got] = big_v[got] / big_w[got, None]
        todo &= ~got
    out[~ok & ~(near & ~todo)] = np.asarray(constant, np.float32)
    return out


def sample(img, uv, interp=None) -> np.ndarray:
    """img at float pixel positions uv (N, 2) via cv2.remap: (N, C) or (N,). The maps
    are folded into rows of 4096 (cv2.remap wants both sides < 32767)."""
    import cv2

    n = len(uv)
    w = 4096
    rows = -(-n // w)
    m = np.full((rows * w, 2), -1e4, np.float32)
    m[:n] = uv
    m = m.reshape(rows, w, 2)
    out = cv2.remap(np.ascontiguousarray(img), m[..., 0], m[..., 1], cv2.INTER_LINEAR if interp is None else interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(rows * w, *out.shape[2:])[:n]


def finger_zone(prior) -> np.ndarray:
    """Every pixel a finger occupies at ANY opening in the frames the prior holds."""
    return dilate(prior.frequency() > 0.02, 6)


def _grow(m, r: int) -> np.ndarray:
    """Square dilation: a generous safety margin, and far cheaper than a disk at r 20-30."""
    import cv2

    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def wrist_exclusions(img, h, zone, mug_model) -> np.ndarray:
    """Wrist pixels that must not be projected onto the proxies: the finger zone, teal
    (+10 px), red (+20 px) and the mug model (+30 px)."""
    return zone | _grow(colour.teal(img, "grip", h), 10) | _grow(colour.red(img, h), 20) | _grow(mug_model, 30)


def robust_mean(cache, n: int, scale: dict) -> tuple[np.ndarray, np.ndarray]:
    """Two-pass weighted mean of per-frame samples [(key, idx, x (N, 3), w (N,))] onto n
    bins. Each frame's x is multiplied by scale[key], and frames without a scale are
    skipped. Pass 2 drops samples farther than max(3 sigma, 0.2 mean + 0.003) from pass 1.
    Returns (weight (n,), mean (n, 3))."""
    def acc(reject=None):
        sw, s1, s2 = np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3))
        for key, idx, val, w in cache:
            if key not in scale:
                continue
            x = val.astype(np.float64) * scale[key]
            ww = w.astype(np.float64)
            if reject is not None:
                mu, sd = reject
                dev = np.abs(x - mu[idx]).max(1)
                ww = np.where(dev <= np.maximum(3 * sd[idx].max(1), 0.2 * mu[idx].max(1) + 0.003), ww, 0.0)
            sw += np.bincount(idx, ww, n)
            for c in range(3):
                s1[:, c] += np.bincount(idx, ww * x[:, c], n)
                s2[:, c] += np.bincount(idx, ww * x[:, c] ** 2, n)
        mu = s1 / np.maximum(sw, 1e-12)[:, None]
        return sw, mu, np.sqrt(np.maximum(s2 / np.maximum(sw, 1e-12)[:, None] - mu ** 2, 0))

    _, mu, sd = acc()
    sw, mu, _ = acc((mu, sd))
    return sw, mu


def lum_ratio(ref, x) -> float | None:
    """sum(Y ref) / sum(Y x) over paired samples (N, 3), dropping pairs whose log ratio
    is more than 0.4 from the median (moving shadows, glints)."""
    yr, yx = ref @ colour.LUMA, x @ colour.LUMA
    ok = (yr > 0) & (yx > 0)
    if ok.sum() < 300:
        return None
    lr = np.log(yr[ok] / yx[ok])
    keep = np.abs(lr - np.median(lr)) < 0.4
    return float(yr[ok][keep].sum() / yx[ok][keep].sum())


def wrist_mosaic(scene, renderer, prior, front_ref, grid: Grid, L_pla, sigma: float, log=print) -> dict:
    """Wrist frames on the table plane in the front reference state (see module doc).
    front_ref: (tex (R, C, 3), ok (R, C)) of the front texture resampled to `grid`.
    Each frame goes through colour.grip_to_front: white balance from its static finger
    core, saturation 1/sigma, and an exposure from its plain-mat luminance against
    front_ref."""
    import cv2

    t0 = time.time()
    cam, tz, units = scene.camera("grip"), scene.table_z(), scene.units()
    kin = default_kinematics()
    H, W = cam.height, cam.width
    zone, core = finger_zone(prior), prior.static_core()
    eps = E.load(scene.ds, include_excluded=True)
    fref, fok = front_ref
    plain = fok & (colour.linear_to_srgb(np.nan_to_num(fref)).max(-1) < 60)
    uv_b = np.concatenate([np.stack([np.linspace(0, W - 1, 24), np.full(24, v)], 1) for v in (0, H - 1)] +
                          [np.stack([np.full(18, u), np.linspace(0, H - 1, 18)], 1) for u in (0, W - 1)])
    edge = np.minimum.reduce([np.arange(W)[None, :].repeat(H, 0), (W - 1 - np.arange(W))[None, :].repeat(H, 0),
                              np.arange(H)[:, None].repeat(W, 1), (H - 1 - np.arange(H))[:, None].repeat(W, 1)])
    fade = np.clip(edge / BORDER_FADE_PX, 0, 1).astype(np.float32)
    R, Cn = grid.shape
    cache, measured, meta = [], {}, []  # measured[(e, k)] = (exposure or None, white balance)
    for e, ep in eps.items():
        Q = units.to_urdf(ep.state)
        centres = np.array([cam.T_world_cam(kin.link_T(q, "gripper"))[:3, 3] for q in Q])
        speed = np.linalg.norm(np.gradient(centres, axis=0), axis=1) * ep.fps
        renderer.set_mug(mug_pose_or_none(scene, e))
        fr = ep.frames("grip")
        for k in range(0, len(ep), WRIST_STRIDE):
            Tg = kin.link_T(Q[k], "gripper")
            Twc = cam.T_world_cam(Tg)
            if speed[k] > WRIST_MAX_SPEED or Twc[2, 2] > -0.1:
                continue
            P = cam.unproject_to_plane(uv_b, z=tz, T_world_parent=Tg)
            P = P[np.isfinite(P).all(1)]
            if len(P) < 4:
                continue
            (ra, ca), (rb, cb) = grid.rc(P[:, 0].min(), P[:, 1].max()), grid.rc(P[:, 0].max(), P[:, 1].min())
            r0, r1 = int(np.clip(np.floor(ra), 0, R)), int(np.clip(np.ceil(rb) + 1, 0, R))
            c0, c1 = int(np.clip(np.floor(ca), 0, Cn)), int(np.clip(np.ceil(cb) + 1, 0, Cn))
            if r1 <= r0 or c1 <= c0:
                continue
            X, Y = grid.xy(r0, r1, c0, c1)
            Pt = np.stack([X.ravel(), Y.ravel(), np.full(X.size, tz)], 1)
            uv = cam.project(Pt, Tg)
            inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
            if inside.sum() < 50:
                continue
            img = np.asarray(fr[k])
            lin = colour.srgb_to_linear(img)
            wb = colour.grip_white_balance(img, core, L_pla, lin)
            if wb is None:
                continue
            mug = PART_TO_LABEL[renderer.parts(Q[k], "grip")] == LABELS["mug"]
            bad = wrist_exclusions(img, colour.hsv(img), zone, mug)
            u = uv[inside]
            val = colour.grip_to_front(sample(lin, u), wb, 1.0, sigma, L_pla)
            nb = sample(bad.astype(np.uint8), u, cv2.INTER_NEAREST) == 0
            fd = sample(fade, u)
            rr, cc = np.divmod(np.flatnonzero(inside), c1 - c0)
            flat = (rr + r0) * Cn + (cc + c0)
            v = Pt[inside] - Twc[:3, 3]
            dist = np.linalg.norm(v, axis=1)
            w = (-v[:, 2] / dist) / dist ** 2 * fd
            keep = nb & (w > 0)
            if keep.sum() < 50:
                continue
            flat, val, w = flat[keep], val[keep], w[keep]
            ov = plain.ravel()[flat]
            measured[(e, k)] = (lum_ratio(fref.reshape(-1, 3)[flat[ov]].astype(np.float64), val[ov].astype(np.float64)), wb)
            cache.append(((e, k), flat.astype(np.int64), val.astype(np.float16), w.astype(np.float32)))
            meta.append({"episode": e, "frame": int(k), "texels": int(len(flat)), "cam_height": float(Twc[2, 3] - tz)})
    renderer.set_mug(None)
    # frames without a measured exposure take their nearest measured neighbour in the episode
    final = {}
    for e in eps:
        ks = sorted(k for (ee, k) in measured if ee == e)
        have = [k for k in ks if measured[(e, k)][0] is not None]
        for k in ks:
            s = measured[(e, k)][0]
            if s is None and have:
                s = measured[(e, min(have, key=lambda h: abs(h - k)))][0]
            if s is not None:
                final[(e, k)] = s
    sw, mu = robust_mean(cache, R * Cn, final)
    ok = sw > 0
    tex = np.where(ok[:, None], mu, np.nan).reshape(R, Cn, 3).astype(np.float32)
    s_arr = np.array(list(final.values())) if final else np.ones(1)
    wb_arr = np.array([g[1] for g in measured.values()])
    n_meas = int(sum(g[0] is not None for g in measured.values()))
    log(f"  wrist mosaic: {len(cache)} frames, {ok.mean():.3f} of the grid, exposure measured on {n_meas}; "
        f"exposure p5/50/95 {np.round(np.percentile(s_arr, [5, 50, 95]), 3).tolist()} ({time.time() - t0:.1f} s)")
    model = {f"{e}:{k}": {"white_balance": np.round(measured[(e, k)][1], 4).tolist(), "exposure": round(float(v), 4),
                          "exposure_measured": measured[(e, k)][0] is not None} for (e, k), v in final.items()}
    return {"tex": tex, "ok": ok.reshape(R, Cn), "weight": sw.reshape(R, Cn).astype(np.float32),
            "model": model, "frames": meta,
            "summary": {"frames": len(cache), "frames_with_measured_exposure": n_meas,
                        "coverage_fraction": round(float(ok.mean()), 4),
                        "exposure_p5_p50_p95": np.round(np.percentile(s_arr, [5, 50, 95]), 4).tolist(),
                        "white_balance_p5_p50_p95": np.round(np.percentile(wb_arr, [5, 50, 95], axis=0), 4).tolist(),
                        "seconds": round(time.time() - t0, 1)}}


def mug_pose_or_none(scene, e):
    from .. import objects

    return objects.mug_pose(scene, e) if "mug" in scene.episode(e) else None


def specular_field(scene, light, grid: Grid, a_ggx: float, step: int = 4) -> np.ndarray:
    """The gloss model's GGX lobe S(p) for the FRONT camera over the grid (computed on
    every `step`-th texel and upsampled: it is smooth)."""
    import cv2

    from .lighting import specular_lobe

    R, Cn = grid.shape
    X, Y = grid.xy()
    Xs, Ys = X[::step, ::step], Y[::step, ::step]
    P = np.stack([Xs.ravel(), Ys.ravel(), np.full(Xs.size, scene.table_z())], 1)
    C = scene.camera("front").T_world_cam()[:3, 3]
    S = specular_lobe(P, C, np.array(light["direction_to_light"]), np.radians(light["angular_radius_deg"]), a_ggx)
    return cv2.resize(S.reshape(Xs.shape).astype(np.float32), (Cn, R), interpolation=cv2.INTER_LINEAR)


def build(scene, plates, light, gloss, renderer, prior, L_pla, sigma: float, log=print) -> dict:
    """Front rectification + wrist mosaic + merge (radiance, front reference state)."""
    import cv2

    t0 = time.time()
    tz = scene.table_z()
    fine, coarse = Grid(*EXTENT, RES_FRONT), Grid(*EXTENT, RES_WRIST)
    front, fok = rectify(plates["clean_lin"], plates["clean_valid"], scene.camera("front"), fine, tz)
    # the front texture on the wrist grid: the reference for each wrist frame's exposure
    fc = cv2.resize(np.nan_to_num(front), coarse.shape[::-1], interpolation=cv2.INTER_AREA)
    fcok = cv2.resize(fok.astype(np.float32), coarse.shape[::-1], interpolation=cv2.INTER_AREA) > 0.999
    wm = wrist_mosaic(scene, renderer, prior, (np.where(fcok[..., None], fc, np.nan), fcok), coarse, L_pla, sigma, log)
    wt = cv2.resize(np.nan_to_num(wm["tex"]), fine.shape[::-1], interpolation=cv2.INTER_LINEAR)
    wok = cv2.resize(wm["ok"].astype(np.float32), fine.shape[::-1], interpolation=cv2.INTER_LINEAR) > 0.999
    inside = cv2.distanceTransform(fok.astype(np.uint8), cv2.DIST_L2, 5) * RES_FRONT
    a = np.clip(inside / FEATHER_M, 0, 1)[..., None]
    a = np.where(wok[..., None], a, 1.0)
    merged = np.where(fok[..., None], a * np.nan_to_num(front) + (1 - a) * wt, wt)
    ok = fok | wok
    source = np.where(fok, 1, np.where(wok, 2, 0)).astype(np.uint8)
    plain_front = fok & (colour.linear_to_srgb(np.nan_to_num(front)).max(-1) < 40)
    mat_const = np.median(front[plain_front], axis=0)
    merged = fill(merged, ok, mat_const)
    # per-channel specular for the front view: L = k_d + k_s S on plain front mat
    a_ggx = float(np.median([g["alpha_ggx"] for g in gloss.values()]))
    S = specular_field(scene, light, fine, a_ggx)
    u8 = colour.linear_to_srgb(np.nan_to_num(front))
    hsv = colour.hsv(u8)
    fluor = (hsv[..., 0] > 20) & (hsv[..., 0] < 45) & (hsv[..., 1] > 80)
    plain = fok & ~fluor & (u8.max(-1) < 250)
    A = np.stack([np.ones(plain.sum()), S[plain]], 1)
    k = np.linalg.lstsq(A, front[plain].astype(np.float64), rcond=None)[0]  # (2, 3): k_d, k_s per channel
    spec = np.where(fok[..., None], S[..., None] * k[1][None, None, :], 0.0).astype(np.float32)
    log(f"  table: front {fok.mean():.3f}, wrist {(~fok & wok).mean():.3f}, filled {(~ok).mean():.3f} of the grid; "
        f"k_d {np.round(k[0], 5).tolist()} k_s {np.round(k[1], 4).tolist()} ({time.time() - t0:.1f} s)")
    return {"fine": fine, "coarse": coarse, "z": tz, "radiance": merged.astype(np.float32), "source": source,
            "specular": spec, "k_diffuse": k[0].tolist(), "k_specular": k[1].tolist(), "alpha_ggx": a_ggx,
            "front_ok": fok, "wrist": wm, "fluor": fluor & fok,
            "summary": {"grid_fine": fine.to_json(tz), "grid_wrist": coarse.to_json(tz),
                        "fraction_front": round(float(fok.mean()), 4),
                        "fraction_wrist_only": round(float((~fok & wok).mean()), 4),
                        "fraction_filled": round(float((~ok).mean()), 4),
                        "k_diffuse_lin": np.round(k[0], 6).tolist(), "k_specular": np.round(k[1], 5).tolist(),
                        "alpha_ggx": a_ggx, "wrist": wm["summary"], "seconds": round(time.time() - t0, 1)}}


def albedo(tab, L_pla, rho_pla: float) -> tuple[np.ndarray, dict]:
    """De-lit albedo texture (R, C, 3) from the merged radiance (see module doc). Inside
    the glare the texel keeps its fine detail, Y / blur(Y, 15 mm) clipped to 0.5-2, on
    the plain-mat median albedo, so scratches and marks survive the glare's removal."""
    import cv2

    L = tab["radiance"]
    spec = tab["specular"]
    alb = rho_pla * np.clip(L - spec, 0, None) / np.asarray(L_pla, np.float32)
    front = tab["source"] == 1
    glare = (spec.max(-1) > SPEC_REPLACE * np.maximum(L.max(-1), 1e-6)) & front
    plain = front & ~glare & ~tab["fluor"]
    mat = np.median(alb[plain], axis=0)
    Y = colour.luminance(L)
    detail = np.clip(Y / np.maximum(cv2.GaussianBlur(Y, (0, 0), 0.015 / RES_FRONT), 1e-6), 0.5, 2.0)
    alb[glare] = mat[None, :] * detail[glare][:, None]
    wr = tab["source"] == 2
    g = alb[plain][:, 1]
    return np.clip(alb, 0, 1).astype(np.float32), {
        "mat_median": np.round(mat, 5).tolist(),
        "mat_plain_p5_p50_p95_green": np.round(np.percentile(g, [5, 50, 95]), 5).tolist(),
        "glare_texels_fraction_of_front": round(float(glare.sum() / max(front.sum(), 1)), 4),
        "fluorescent_median": np.round(np.median(alb[tab["fluor"]], 0), 4).tolist() if tab["fluor"].any() else None,
        "wrist_region_p50_p90": np.round(np.percentile(alb[wr], [50, 90], axis=0), 4).tolist() if wr.any() else None}
