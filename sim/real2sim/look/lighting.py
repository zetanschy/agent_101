"""The key light and the ambient level from the mug's shadows; the mat's gloss from its glare.

SHADOWS (MEASURED, plates.py). Every episode plate shows the mug's shadow on the mat
as a capsule of 9-11 k px, running toward the image's lower left. The shadow ratio
(episode plate / clean plate, luminance) has no arm, cap or white balance in it. It
is fitted with a DISTANT disk source:

    ratio(p) = a_e + (1 - a_e) vis(p)
    vis(p)   the visible fraction of the source (direction az, el; angular radius
             beta) from table point p past the mug. The mug is a vertical cylinder of
             the scene's diameter and height, plus its handle as a 10 mm post from
             z_bottom to z_top, 51.5 mm out along the handle yaw. Penumbra wedge,
             which is smooth in every parameter, so least squares gets clean
             gradients: delta = the ray's closest horizontal approach to the axis
             within the cylinder's height, minus its radius; w = beta * s (s = the
             distance along the ray); vis = clip(1/2 + delta / 2w, 0, 1),
             multiplied over the occluders.
    a_e      ambient / (ambient + key) on the table: the umbra depth, per episode.

    free:    az, el, log beta, a_0..a_3. A 7.5 deg grid over az and el (a quarter of the points) comes first,
             then scipy least_squares (soft L1) over about 33 k points, subsampled
             4 px within 280 px of each mug. Plain mat only: the clean plate below 60
             in 8-bit, which drops the glare corner and the fluorescent marks.

WHY DISTANT (build-phase probe, same data). The four shadows point 56, 57, 69 and 73
deg, which looks like parallax from a nearby lamp. But a finite-distance lamp (fitted
with the mug height free) fits WORSE the closer it is: rms 0.095 at 0.8 m, 0.080 at
1.8 m, 0.078 at 2.7 m and 0.0745 at 9 m. The spread in direction comes from the handle posts and the mug-pose
error (8 mm), not from parallax, so the key is at least ~2.5 m away: a window or
ceiling panel rather than a desk lamp. The angular radius is large (see look.json).
Freeing the mug height as well sends it to the lower bound of 0.8 x 100 mm. Shadow
length and elevation are degenerate, so the height stays at the scene's value, and
`mug_height_if_free` in the result is reported as a diagnostic only.

GLARE (a check on the fit, and the gloss). The black mat is a glossy dielectric. Its
specular reflection of the source is the bright top-right corner of every front
frame, where 8-bit values reach 188-200 against the mat's 18. The shadow-fitted source
predicts WHERE that reflection must be: the table points whose mirrored view ray
falls inside the source's cone. The plates' luminance is fitted with

    L(p) = k_d + k_s S(p; alpha_ggx)       (the key is distant: diffuse is flat)

S being the GGX lobe (alpha_ggx = Blender roughness^2), averaged over the source's
cone, divided by 4 cos(theta_v). alpha_ggx comes from a 1-D search, k_d and k_s from
linear least squares on plain-mat pixels. In episodes 1 and 3 sharp dark shapes
cross the glare: something in the room shows in the mirror. So the fit runs on
episodes 0 and 2, where the glare is unbroken.
"""

from __future__ import annotations

import numpy as np

HANDLE_POST_RADIUS = 0.010  # m, a crude stand-in for the handle loop (estimated)
POINT_STRIDE_PX, POINT_REACH_PX = 4, 280
GGX_GRID = np.round(np.geomspace(0.004, 0.6, 22), 5)


def _sunflower(n: int) -> np.ndarray:
    """n points spread evenly over the unit disk (Vogel spiral)."""
    i = np.arange(n) + 0.5
    r, t = np.sqrt(i / n), i * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def direction(az, el) -> np.ndarray:
    """Unit vector from the scene TOWARDS the source (base frame)."""
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def cone_samples(d, beta, n: int = 64) -> np.ndarray:
    """(n, 3) unit directions spread over the source's cone (axis d, half-angle beta)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    a = np.cross(d, [0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    s = _sunflower(n) * np.tan(beta)
    v = d + s[:, :1] * a + s[:, 1:] * b
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def occluders(scene, ep: int, height=None) -> list[tuple]:
    """[(centre xy, radius, z0, z1)] of the mug body and the handle post, base frame."""
    m, d, tz = scene.episode(ep)["mug"], scene.mug_dims(), scene.table_z()
    c = np.asarray(m["xy"], float)
    yaw = np.radians(m["handle_yaw_deg"])
    R, h = d["outer_diameter"] / 2, d["handle"]
    post = c + (R + h["protrusion"] / 2) * np.array([np.cos(yaw), np.sin(yaw)])
    H = d["height"] if height is None else height
    return [(c, R, tz, tz + H), (post, HANDLE_POST_RADIUS, tz + h["z_bottom"], tz + min(h["z_top"], H))]


def _cylinder_vis(P, d, beta, c, R, z0, z1) -> np.ndarray:
    """Penumbra-wedge visibility (N,) of a distant source (unit direction d, angular
    radius beta) from points P past one vertical cylinder (axis xy c, radius R, z0..z1)."""
    dxy2 = max(float(d[0] ** 2 + d[1] ** 2), 1e-12)
    rel = P[:, :2] - c
    s = -(rel @ d[:2]) / dxy2  # closest horizontal approach along the ray, unconstrained
    s0, s1 = (z0 - P[:, 2]) / d[2], (z1 - P[:, 2]) / d[2]
    s = np.clip(s, np.maximum(s0, 0.0), np.maximum(s1, 0.0))
    h = np.linalg.norm(rel + s[:, None] * d[:2], axis=1)
    w = np.maximum(beta * s, 1e-4)
    return np.where(s1 > 0, np.clip(0.5 + (h - R) / (2 * w), 0.0, 1.0), 1.0)


def visibility(P, d, beta, occ) -> np.ndarray:
    vis = np.ones(len(P))
    for c, R, z0, z1 in occ:
        vis *= _cylinder_vis(P, d, beta, c, R, z0, z1)
    return vis


def shadow_points(plates, scene) -> list[dict]:
    """Observed shadow ratios near each episode's mug, as table points. Kept: plain mat
    (clean plate below 60 in 8-bit), ratio in (0, 1.3), and no mug. The mug's own
    pixels are the model hull, or wherever the episode plate is 15 levels brighter than
    the clean plate, dilated 6 px."""
    from . import colour
    from .segment import brighter, dilate

    cam, tz = scene.camera("front"), scene.table_z()
    clean_u8 = colour.linear_to_srgb(np.nan_to_num(plates["clean_lin"]))
    out = []
    for e in plates["episodes"]:
        hull = plates["mug_hull"][e]
        body = hull | brighter(colour.linear_to_srgb(plates["norm_lin"][e]), clean_u8, 15)
        ys, xs = np.nonzero(hull)
        gy, gx = np.mgrid[0:cam.height:POINT_STRIDE_PX, 0:cam.width:POINT_STRIDE_PX]
        near = (gx - xs.mean()) ** 2 + (gy - ys.mean()) ** 2 < POINT_REACH_PX ** 2
        r = plates["shadow_ratio"][e][gy, gx]
        keep = near & np.isfinite(r) & (r > 0) & (r < 1.3) & (clean_u8.max(-1)[gy, gx] < 60) \
            & ~dilate(body, 6)[gy, gx]
        uv = np.stack([gx[keep], gy[keep]], 1).astype(float)
        out.append({"episode": int(e), "P": cam.unproject_to_plane(uv, z=tz), "ratio": r[keep].astype(float)})
    return out


def _ambient_ls(ratio, vis) -> float:
    """Closed-form a in ratio = vis + a (1 - vis)."""
    w = 1 - vis
    return float(np.clip((w * (ratio - vis)).sum() / max((w * w).sum(), 1e-9), 0, 1))


def fit_key_light(plates, scene, log=print) -> dict:
    from scipy.optimize import least_squares

    obs = shadow_points(plates, scene)
    occ = {o["episode"]: occluders(scene, o["episode"]) for o in obs}

    sub = [dict(o, P=o["P"][::4], ratio=o["ratio"][::4]) for o in obs]  # the grid only needs a sketch

    def cost(d, beta):
        tot = 0.0
        for o in sub:
            vis = visibility(o["P"], d, beta, occ[o["episode"]])
            a = _ambient_ls(o["ratio"], vis)
            tot += ((vis + a * (1 - vis) - o["ratio"]) ** 2).sum()
        return tot

    grid = [(cost(direction(az, el), 0.1), az, el) for az in np.radians(np.arange(-180, 180, 7.5))
            for el in np.radians(np.arange(10, 90, 7.5))]
    _, az0, el0 = min(grid)

    def resid(th, height=None):
        d, beta = direction(th[0], th[1]), np.exp(th[2])
        return np.concatenate([visibility(o["P"], d, beta, occluders(scene, o["episode"], height) if height
                                          else occ[o["episode"]]) * (1 - a) + a - o["ratio"]
                               for o, a in zip(obs, th[3:3 + len(obs)])])

    lo = [-2 * np.pi, np.radians(5), np.log(0.002)] + [0.0] * len(obs)
    hi = [2 * np.pi, np.radians(89.5), np.log(0.5)] + [1.0] * len(obs)
    x0 = np.r_[az0, el0, np.log(0.1), [0.5] * len(obs)]
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.1, max_nfev=300, bounds=(lo, hi))
    H0 = scene.mug_dims()["height"]
    free_h = least_squares(lambda t: resid(t[:-1], t[-1]), np.r_[sol.x, H0], loss="soft_l1", f_scale=0.1,
                           max_nfev=200, bounds=(lo + [0.6 * H0], hi + [1.4 * H0]))
    az, el, beta, a_e = sol.x[0], sol.x[1], float(np.exp(sol.x[2])), sol.x[3:]
    az = (az + np.pi) % (2 * np.pi) - np.pi
    r = resid(sol.x)
    per_ep, i0 = {}, 0
    for o, a in zip(obs, a_e):
        n = len(o["ratio"])
        ri, oi = r[i0:i0 + n], o["ratio"]
        so, sp = oi < 0.8, (ri + oi) < 0.8
        per_ep[o["episode"]] = {"points": n, "rms": round(float(np.sqrt(np.mean(ri ** 2))), 4),
                                "shadow_iou_at_0.8": round(float((so & sp).sum() / max((so | sp).sum(), 1)), 4),
                                "ambient_fraction": round(float(a), 4)}
        i0 += n
    a_med = float(np.median(a_e))
    res = {
        "type": "distant disk source",
        "azimuth_deg": round(float(np.degrees(az)), 2), "elevation_deg": round(float(np.degrees(el)), 2),
        "direction_to_light": np.round(direction(az, el), 5).tolist(),
        "angular_radius_deg": round(float(np.degrees(beta)), 3),
        "ambient_fraction": {"median": round(a_med, 4), "per_episode": np.round(a_e, 4).tolist()},
        "ambient_over_key_horizontal": round(a_med / (1 - a_med), 4),
        "grid_start_deg": [round(float(np.degrees(az0)), 1), round(float(np.degrees(el0)), 1)],
        "mug_height_if_free": round(float(free_h.x[-1]), 4),
        "elevation_if_height_free_deg": round(float(np.degrees(free_h.x[1])), 2),
        "rms": round(float(np.sqrt(np.mean(r ** 2))), 4), "n_points": int(len(r)), "nfev": int(sol.nfev),
        "per_episode": per_ep,
    }
    log(f"  key light: az {res['azimuth_deg']} el {res['elevation_deg']} deg, radius {res['angular_radius_deg']} deg, "
        f"ambient fraction {np.round(a_e, 3).tolist()}, rms {res['rms']}")
    return res


# --- glare: gloss of the mat --------------------------------------------------------

def ggx_D(cos_h, a):
    c2 = np.clip(cos_h, 0, 1) ** 2
    return a * a / (np.pi * (c2 * (a * a - 1) + 1) ** 2)


def specular_lobe(P, C, d, beta, a_ggx, n: int = 64) -> np.ndarray:
    """GGX lobe of the horizontal table at points P, seen from camera centre C, averaged
    over the source's cone: mean_s D(h_s) / (4 cos theta_v)."""
    v = C - P
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out = np.zeros(len(P))
    for l in cone_samples(d, beta, n):
        h = v + l
        out += ggx_D(h[:, 2] / np.linalg.norm(h, axis=1), a_ggx)
    return out / n / (4 * np.clip(v[:, 2], 1e-3, None))


def fit_gloss(plates, scene, light: dict, episodes=(0, 2), log=print) -> dict:
    from . import colour
    from .segment import dilate

    cam, tz = scene.camera("front"), scene.table_z()
    C = cam.T_world_cam()[:3, 3]
    d, beta = np.array(light["direction_to_light"]), np.radians(light["angular_radius_deg"])
    gy, gx = np.mgrid[0:cam.height:4, 0:cam.width:4]
    P = cam.unproject_to_plane(np.stack([gx.ravel(), gy.ravel()], 1).astype(float), z=tz)
    lobes = {a: specular_lobe(P, C, d, beta, a) for a in GGX_GRID}
    out = {}
    for e in episodes:
        lin = plates["norm_lin"][e]
        Y = colour.luminance(lin)[gy, gx].ravel()
        u8 = colour.linear_to_srgb(lin)[gy, gx].reshape(-1, 3)
        hsv = colour.hsv(u8.reshape(1, -1, 3)).reshape(-1, 3)
        fluor = (hsv[:, 0] > 20) & (hsv[:, 0] < 45) & (hsv[:, 1] > 80)  # the yellow-green marks
        bad = dilate(plates["mug_hull"][e], 25) | dilate(plates["shadow"][e], 8) | ~plates["valid"][e]
        ok = ~bad[gy, gx].ravel() & ~fluor & np.isfinite(Y) & (u8.max(1) < 250)
        fits = []
        for a in GGX_GRID:
            A = np.stack([np.ones(ok.sum()), lobes[a][ok]], 1)
            coef, *_ = np.linalg.lstsq(A, Y[ok], rcond=None)
            fits.append((float(np.sqrt(np.mean((A @ coef - Y[ok]) ** 2))), a, coef))
        rms, a, coef = min(fits, key=lambda f: f[0])
        rms_flat = float(np.std(Y[ok]))
        top = ok & (lobes[a] >= np.percentile(lobes[a][ok], 97))
        out[int(e)] = {"alpha_ggx": float(a), "roughness": round(float(np.sqrt(a)), 4),
                       "k_diffuse_lin": round(float(coef[0]), 6), "k_specular": round(float(coef[1]), 6),
                       "rms_lin": round(rms, 6), "rms_flat_lin": round(rms_flat, 6),
                       "glare_top3pct_pred_vs_obs_lin": [round(float(coef[0] + coef[1] * lobes[a][top].mean()), 5),
                                                         round(float(Y[top].mean()), 5)],
                       "pixels": int(ok.sum()), "rms_by_alpha": {float(f[1]): round(f[0], 6) for f in fits}}
        log(f"  gloss ep{e}: alpha_ggx {a} (roughness {np.sqrt(a):.3f}), rms {rms:.5f} vs {rms_flat:.5f} flat")
    return out
