"""The far field: what the wrist camera sees past the mat, on proxy geometry; an environment map.

At carry and over-mug poses the wrist camera sits 0.26-0.31 m above the table, tilted
35-45 deg outward. Its upper rays reach the table 0.53-0.66 m out on the -x side
(episodes report), and there it sees a white desk, a black round object, a pole,
cables and colourful clutter. None of that is ever in the front view. The PROXY is
the simplest geometry that places those pixels plausibly:

    disc      the table plane z = table_z, inside radius R of the axis. Its texture is
              table.py's, which covers the rectangle.
    cylinder  a vertical wall at radius R = 0.80 m around (x, y) = (-0.10, -0.30), from
              table_z to table_z + 0.60 m. Rays that leave the disc above the table
              land on it. It is textured here from the same wrist frames, masks and
              wrist->front colour model as the mosaic, as a robust mean.

Where it breaks. Anything standing on the desk inside R (the round object and pole
are 0.55-0.8 m out) is smeared along the plane. Anything beyond R is flattened onto
the wall. The texture is exact only when seen from where the wrist camera was, so
it serves the wrist view, reflections and ambient light, and the front camera never
sees it. It is also only as sharp as the grip camera's calibration: with the
provisional wrist pose (2-3 cm uncertain) things land several cm off. Re-run after
CALIB.

ENVIRONMENT MAP. An equirectangular map centred at (-0.10, -0.30, table_z + 0.10),
the height of a cap in the mug. Each direction is traced to the proxy (disc, wall,
top), and table / cylinder radiance is looked up. Directions the wrist camera never
saw, which is most of the upper hemisphere, the ceiling and the room behind the
robot, get one uniform radiance. That radiance is chosen so the map's own
horizontal irradiance equals the ambient share of the light fit. The key source is
NOT in the map; it is an analytic light (look.json lights.key), so it is never
counted twice.

UNITS. Stored values are SCENE radiance: total horizontal irradiance at the table =
1, so a horizontal Lambertian face of albedo rho has radiance rho / pi. Camera-linear
front-reference values are converted by rho_PLA / (pi L_PLA), per channel.
"""

from __future__ import annotations

import time

import numpy as np

from .. import episodes as E
from ..kinematics import default as default_kinematics
from . import colour
from .segment import LABELS, PART_TO_LABEL
from .table import (BORDER_FADE_PX, WRIST_MAX_SPEED, WRIST_STRIDE, mug_pose_or_none, finger_zone, robust_mean, sample,
                    wrist_exclusions)

CENTRE_XY = (-0.10, -0.30)
RADIUS, HEIGHT = 0.80, 0.60
RES_CYL = 0.004  # m per texel along the wall and up it
ENV_SIZE = (1024, 512)
ENV_CENTRE_DZ = 0.10


def cylinder_grid(tz: float):
    n_phi, n_z = int(round(2 * np.pi * RADIUS / RES_CYL)), int(round(HEIGHT / RES_CYL))
    phi = -np.pi + (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    z = tz + HEIGHT - (np.arange(n_z) + 0.5) * RES_CYL  # row 0 = top
    return phi, z


def cylinder(scene, renderer, prior, wrist_model: dict, L_pla, sigma: float, log=print) -> dict:
    """Cylinder texture (n_z, n_phi, 3), front-reference camera-linear, robust mean. The
    frames are the wrist mosaic's (wrist_model: white balance + exposure per frame)."""
    import cv2

    t0 = time.time()
    cam, tz, units = scene.camera("grip"), scene.table_z(), scene.units()
    kin = default_kinematics()
    H, W = cam.height, cam.width
    phi, zc = cylinder_grid(tz)
    P_all = np.stack([CENTRE_XY[0] + RADIUS * np.cos(phi)[None, :].repeat(len(zc), 0),
                      CENTRE_XY[1] + RADIUS * np.sin(phi)[None, :].repeat(len(zc), 0),
                      zc[:, None].repeat(len(phi), 1)], -1).reshape(-1, 3)
    zone = finger_zone(prior)
    model = {tuple(map(int, k.split(":"))): v for k, v in wrist_model.items()}
    edge = np.minimum.reduce([np.arange(W)[None, :].repeat(H, 0), (W - 1 - np.arange(W))[None, :].repeat(H, 0),
                              np.arange(H)[:, None].repeat(W, 1), (H - 1 - np.arange(H))[:, None].repeat(W, 1)])
    fade = np.clip(edge / BORDER_FADE_PX, 0, 1).astype(np.float32)
    cache = []
    for e, ep in E.load(scene.ds, include_excluded=True).items():
        renderer.set_mug(mug_pose_or_none(scene, e))
        fr = ep.frames("grip")
        Q = units.to_urdf(ep.state)
        centres = np.array([cam.T_world_cam(kin.link_T(q, "gripper"))[:3, 3] for q in Q])
        speed = np.linalg.norm(np.gradient(centres, axis=0), axis=1) * ep.fps
        for k in range(0, len(ep), WRIST_STRIDE):
            if (e, k) not in model or speed[k] > WRIST_MAX_SPEED:
                continue
            Tg = kin.link_T(Q[k], "gripper")
            Twc = cam.T_world_cam(Tg)
            # quick reject: does any ray of the top image row leave the disc before reaching the table?
            top = cam.unproject_to_plane(np.stack([np.linspace(0, W - 1, 9), np.zeros(9)], 1), z=tz, T_world_parent=Tg)
            far = ~np.isfinite(top[:, 0]) | (np.hypot(top[:, 0] - CENTRE_XY[0], top[:, 1] - CENTRE_XY[1]) > RADIUS)
            if not far.any():
                continue
            v = P_all - Twc[:3, 3]
            front_ = v @ Twc[:3, 2] > 0.05
            uv = np.full((len(P_all), 2), np.nan)
            uv[front_] = cam.project(P_all[front_], Tg)
            inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
            if inside.sum() < 50:
                continue
            img = np.asarray(fr[k])
            mug = PART_TO_LABEL[renderer.parts(Q[k], "grip")] == LABELS["mug"]
            bad = wrist_exclusions(img, colour.hsv(img), zone, mug)
            u = uv[inside]
            val = sample(colour.srgb_to_linear(img), u)
            ok = sample(bad.astype(np.uint8), u, cv2.INTER_NEAREST) == 0
            fd = sample(fade, u)
            idx = np.flatnonzero(inside)
            dist = np.linalg.norm(v[idx], axis=1)
            w = fd / dist ** 2
            keep = ok & (w > 0)
            if keep.sum() < 50:
                continue
            m = model[(e, k)]
            cache.append(((e, k), idx[keep].astype(np.int64),
                          colour.grip_to_front(val[keep], m["white_balance"], m["exposure"], sigma, L_pla), w[keep]))
    renderer.set_mug(None)
    sw, mu = robust_mean(cache, len(P_all), {key: 1.0 for key, *_ in cache})
    shape = (len(zc), len(phi))
    ok = (sw > 0).reshape(shape)
    tex = np.where(ok[..., None], mu.reshape(*shape, 3), np.nan).astype(np.float32)
    log(f"  backdrop cylinder: {len(cache)} frames, {ok.mean():.3f} of the wall seen ({time.time() - t0:.1f} s)")
    return {"tex": tex, "ok": ok, "phi": phi, "z": zc,
            "summary": {"frames": len(cache), "coverage_fraction": round(float(ok.mean()), 4),
                        "seen_azimuth_deg": _seen_range(phi, ok), "seconds": round(time.time() - t0, 1)}}


def _seen_range(phi, ok):
    cols = ok.any(0)
    if not cols.any():
        return None
    a = np.degrees(phi[cols])
    return [round(float(a.min()), 1), round(float(a.max()), 1)]


def env_map(scene, table_tex, table_grid, cyl, to_scene, ambient_fraction: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Equirect (H, W, 3) scene radiance + (H, W) observed mask (see module doc)."""
    import cv2

    tz = scene.table_z()
    Wd, Hd = ENV_SIZE
    phi = -np.pi + (np.arange(Wd) + 0.5) * 2 * np.pi / Wd
    theta = (np.arange(Hd) + 0.5) * np.pi / Hd  # from +z
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    d = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
    c = np.array([CENTRE_XY[0], CENTRE_XY[1], tz + ENV_CENTRE_DZ])
    out = np.full((Hd, Wd, 3), np.nan, np.float32)
    # the wall: solve |c_xy + t d_xy - axis| = R (c is on the axis)
    dxy = np.hypot(d[..., 0], d[..., 1])
    t_wall = np.where(dxy > 1e-9, RADIUS / np.maximum(dxy, 1e-9), np.inf)
    z_wall = c[2] + t_wall * d[..., 2]
    t_table = np.where(d[..., 2] < -1e-9, -ENV_CENTRE_DZ / np.minimum(d[..., 2], -1e-9), np.inf)
    hit_table = t_table < t_wall
    hit_wall = ~hit_table & (z_wall >= tz) & (z_wall <= tz + HEIGHT)
    # table lookups
    p = c + np.where(hit_table, t_table, 0.0)[..., None] * d
    r, cc = table_grid.rc(p[..., 0], p[..., 1])
    tt = cv2.remap(np.nan_to_num(table_tex).astype(np.float32), cc.astype(np.float32), r.astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    out[hit_table] = tt[hit_table]
    # wall lookups
    col = (PH + np.pi) / (2 * np.pi) * len(cyl["phi"]) - 0.5
    row = (tz + HEIGHT - z_wall) / RES_CYL - 0.5
    wt = cv2.remap(np.nan_to_num(cyl["tex"], nan=-1).astype(np.float32), col.astype(np.float32), row.astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    wok = cv2.remap(cyl["ok"].astype(np.float32), col.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP) > 0.999
    use = hit_wall & wok
    out[use] = wt[use]
    out *= np.asarray(to_scene, np.float32)
    seen = np.isfinite(out[..., 0]) & (out[..., 0] >= 0)
    # uniform fill so that the map's own horizontal irradiance (upper hemisphere) = ambient
    dOmega = (2 * np.pi / Wd) * (np.pi / Hd) * np.sin(TH)
    up = d[..., 2] > 0
    w_cos = np.where(up, d[..., 2], 0) * dOmega
    Y = colour.luminance(np.nan_to_num(out))
    have = float((Y * w_cos)[seen].sum())
    free = float(w_cos[~seen].sum())
    fill_level = max((ambient_fraction - have) / max(free, 1e-9), 0.0)
    out[~seen] = fill_level
    info = {"size": [Wd, Hd], "centre": np.round(c, 4).tolist(), "observed_fraction": round(float(seen.mean()), 4),
            "observed_fraction_upper_hemisphere": round(float(seen[up].mean()), 4),
            "fill_radiance": round(fill_level, 5), "horizontal_irradiance_from_observed": round(have, 5),
            "layout": "col = azimuth phi from -pi (left) to +pi, phi = atan2(y - cy, x - cx); row 0 = +z (up)"}
    return out, seen, info
