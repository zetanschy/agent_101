"""Numbers that say how close a replay is to the real episode. Small and engine-agnostic.

Every fidelity claim in real2sim carries one of these, computed against the dataset:

    joint_rmse_deg      sim joints vs observation.state, per joint, at lag 0 and best lag
    fingertip_error_mm  FK point distance between sim and real joints (default: grasp_site)
    gripper_events      close/open transitions from a gripper % series (grasp/release frames)
    cap_in_mug          whether a cap's centre is inside the mug's cavity
    penetration_summary max / p99 of contact penetration depths (target <= 1 mm)
    masked_psnr, masked_ssim, silhouette_iou   image agreement

Joint errors are in URDF degrees: the gripper row is the JAW angle (via the scene's
gripper map), so all six rows share a unit. Lag convention: lag k > 0 means the sim
trails the real arm by k frames (sim[t + k] is compared with real[t]).
"""

from __future__ import annotations

import numpy as np


def joint_rmse_deg(sim_q, real_state, units=None, max_lag: int = 10) -> dict:
    """sim_q (T, 6) URDF rad; real_state (T, 6) lerobot units (observation.state),
    converted with `units` (pass scene.units(); default: zero offsets, tick_physical).
    Returns degrees: {rmse (6,), lag, rmse_at_lag (6,), mean_arm, mean_arm_at_lag,
    max_abs (6,)}. The best lag minimises the mean over the 5 arm joints."""
    from .units import Units

    sim = np.degrees(np.asarray(sim_q, dtype=float))
    real = np.degrees((units or Units()).to_urdf(real_state))
    if sim.shape != real.shape:
        raise ValueError(f"sim {sim.shape} vs real {real.shape}")
    T = len(sim)

    def rmse(k):
        a, b = (sim[k:], real[:T - k]) if k >= 0 else (sim[:T + k], real[-k:])
        return np.sqrt(np.mean((a - b) ** 2, axis=0))

    lags = range(-max_lag, max_lag + 1)
    scores = {k: rmse(k) for k in lags}
    best = min(lags, key=lambda k: scores[k][:5].mean())
    return {"rmse": scores[0], "lag": int(best), "rmse_at_lag": scores[best],
            "mean_arm": float(scores[0][:5].mean()), "mean_arm_at_lag": float(scores[best][:5].mean()),
            "max_abs": np.abs(sim - real).max(0)}


def fingertip_error_mm(sim_q, real_state, units=None, kin=None, point: str = "grasp_site") -> dict:
    """Per-frame distance (mm) between the same FK point under the sim joints (URDF
    rad) and the real ones (lerobot units, through `units`). point: 'grasp_site'
    (mjlab GRASP_SITE_POS) or 'mid' (the jaw-following fingertip midpoint)."""
    from .kinematics import default
    from .units import Units

    kin = kin or default()
    real_q = (units or Units()).to_urdf(real_state)
    f = kin.grasp_site if point == "grasp_site" else (lambda q: kin.fingertips(q)["mid"])
    d = np.array([np.linalg.norm(f(a) - f(b)) for a, b in zip(np.asarray(sim_q), np.asarray(real_q))]) * 1000
    return {"per_frame": d, "mean": float(d.mean()), "p95": float(np.percentile(d, 95)), "max": float(d.max())}


def gripper_events(pct, min_step: float = 3.0, still: float = 0.25, gap: int = 2) -> list[dict]:
    """Close/open transitions in a gripper % series (state or action).

    A frame is MOVING when the 3-frame-median-smoothed value changes by more than
    `still` %/frame; moving runs of one sign separated by <= `gap` still frames are
    merged; a run whose net change is at least `min_step` % is an event. Returns
    [{kind: 'close'|'open', start, stop (inclusive frames), before, after (%)}].
    Measured on the real data: holds sit at 14-16 % after a close from ~25-36 %, and a
    release is only +1.4..+2 % in episode 0 (use min_step ~1 to see it on the state)."""
    x = np.asarray(pct, dtype=float)
    if len(x) < 3:
        return []
    s = x.copy()
    s[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    v = np.diff(s)
    sign = np.where(v > still, 1, np.where(v < -still, -1, 0))
    runs, i = [], 0
    while i < len(sign):
        if sign[i] == 0:
            i += 1
            continue
        j, sg = i, sign[i]
        while True:
            k = j + 1
            while k < len(sign) and sign[k] == 0 and k - j <= gap:
                k += 1
            if k < len(sign) and sign[k] == sg:
                j = k
            else:
                break
        runs.append((i, j + 1, sg))  # diff index i covers frames i..i+1
        i = j + 1
    out = []
    for a, b, sg in runs:
        delta = s[b] - s[a]
        if abs(delta) >= min_step:
            out.append({"kind": "close" if delta < 0 else "open", "start": int(a), "stop": int(b),
                        "before": float(s[a]), "after": float(s[b])})
    return out


def cap_in_mug(cap_pos, cap_quat, mug_pos, mug_quat, cap_dims: dict, mug_dims: dict, margin: float = 0.0):
    """True where the cap's centre is inside the mug cavity (radius < inner radius -
    margin, floor <= z <= rim). Poses are ORIGINS per objects.py; arrays broadcast (T,)."""
    from .objects import cap_centre
    from .transforms import quat_to_mat

    c = cap_centre(cap_pos, cap_quat, cap_dims)
    R = quat_to_mat(mug_quat)
    local = np.einsum("...ji,...j->...i", R, c - np.asarray(mug_pos, dtype=float))
    ri = mug_dims["outer_diameter"] / 2 - mug_dims["wall"]
    r = np.hypot(local[..., 0], local[..., 1])
    return (r < ri - margin) & (local[..., 2] >= mug_dims["floor"] - 1e-4) & (local[..., 2] <= mug_dims["height"])


def penetration_summary(depth_m) -> dict:
    """{max_mm, p99_mm, n} of penetration depths (m, positive = interpenetrating)."""
    d = np.asarray(depth_m, dtype=float).ravel()
    d = d[np.isfinite(d)]
    if not len(d):
        return {"max_mm": 0.0, "p99_mm": 0.0, "n": 0}
    return {"max_mm": float(d.max() * 1000), "p99_mm": float(np.percentile(d, 99) * 1000), "n": int(len(d))}


def _mask(mask, shape):
    return np.ones(shape[:2], bool) if mask is None else np.asarray(mask, bool)


def masked_psnr(a, b, mask=None, peak: float = 255.0) -> float:
    """PSNR (dB) over the masked pixels, all channels."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = _mask(mask, a.shape)
    mse = np.mean((a[m] - b[m]) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(peak ** 2 / mse))


def _gauss_filter(x, sigma: float, radius: int):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (radius, radius)
        xp = np.pad(x, pad, mode="reflect")
        n = x.shape[axis]
        x = sum(w * np.take(xp, np.arange(i, i + n), axis=axis) for i, w in enumerate(k))
    return x


def masked_ssim(a, b, mask=None, peak: float = 255.0, sigma: float = 1.5, radius: int = 5,
                k1: float = 0.01, k2: float = 0.03) -> float:
    """Mean SSIM (Wang et al. 2004, Gaussian window sigma 1.5, 11x11) over the masked
    pixels, averaged over channels. The window statistics use all pixels; the mask
    selects which SSIM-map values are averaged."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    m = _mask(mask, a.shape)
    c1, c2 = (k1 * peak) ** 2, (k2 * peak) ** 2
    vals = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch], b[..., ch]
        mx, my = _gauss_filter(x, sigma, radius), _gauss_filter(y, sigma, radius)
        sxx = _gauss_filter(x * x, sigma, radius) - mx * mx
        syy = _gauss_filter(y * y, sigma, radius) - my * my
        sxy = _gauss_filter(x * y, sigma, radius) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(s[m].mean())
    return float(np.mean(vals))


def silhouette_iou(a, b) -> float:
    """|a & b| / |a | b| of two boolean masks; NaN when both are empty."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = np.logical_or(a, b).sum()
    return float("nan") if u == 0 else float(np.logical_and(a, b).sum() / u)
