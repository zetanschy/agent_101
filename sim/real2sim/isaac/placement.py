"""Where the objects stand at reset, from the scene config or from the recorded grasp.

Placing an object once at reset is not teleporting (RULE 2); the question is only
which ESTIMATE of the real position to place it at. Two are offered, and eval.json
records which one a run used:

    config  the scene's cap xy (objects.cap_pose): the front-camera detection,
            CALIB-fitted from both views once calib.json exists. The default.
    grasp   where a rigid cap of the scene's diameter must have stood for BOTH real
            fingers to touch it at the recorded blocked-jaw frame of its pick: the
            recorded joints (FK) plus "the cap was between the fingers". Computed by
            grasp_centre(); a diagnostic that separates placement error from contact
            physics, not a fit to the outcome.

WHY THE FINGERTIP MIDPOINT IS NOT A PLACEMENT. MEASURED at ep0 frame 231 (the jaw
blocked at 6.6 deg): the two fingers' cross-sections at the cap's height are small
rounded pads, 30 mm apart along a closing line tilted about 25 deg from base x, and
the FK fingertip midpoint (kinematics.fingertips 'mid') lies 3.2 mm off that line.
A 29 mm cap placed there was pushed 9.5 mm forward, out of the grasp, as the jaw
closed (Isaac grasp probe g0). The fixed finger also ends 2.1 mm higher than the
moving jaw at that pose, so how much of the cap's side both fingers can reach depends
on the table height to the millimetre.
"""

from __future__ import annotations

import functools

import numpy as np

from ..kinematics import FIXED_FINGER_MESH, MOVING_JAW_MESH
from .geometry import sample_surface


@functools.lru_cache(maxsize=4)
def _finger_points(kin, n: int = 30000):
    """Surface samples of the two finger meshes in their BODY frames: (fixed on
    'gripper', moving on 'jaw')."""
    out = []
    for name in (FIXED_FINGER_MESH, MOVING_JAW_MESH):
        g = next(m for m in kin.meshes(False) if m["mesh"] == name)
        out.append(sample_surface(g["verts"], g["faces"], n, seed=1))
    return tuple(out)


def finger_points_world(kin, q) -> tuple[np.ndarray, np.ndarray]:
    pf, pj = _finger_points(kin)
    Tg, Tj = kin.link_T(q, "gripper"), kin.link_T(q, "jaw")
    return pf @ Tg[:3, :3].T + Tg[:3, 3], pj @ Tj[:3, :3].T + Tj[:3, 3]


def pad_gap(kin, q, z_lo: float, z_hi: float) -> tuple[float, np.ndarray, np.ndarray]:
    """(free gap m, point on the fixed pad, point on the moving pad): the closest pair
    between the two fingers' surfaces within the height band z_lo..z_hi, in the table
    plane (the cap is a vertical cylinder, closed-up or open-up)."""
    from scipy.spatial import cKDTree

    pf, pj = finger_points_world(kin, q)
    pf = pf[(pf[:, 2] >= z_lo) & (pf[:, 2] <= z_hi)]
    pj = pj[(pj[:, 2] >= z_lo) & (pj[:, 2] <= z_hi)]
    if not len(pf) or not len(pj):
        return float("nan"), np.full(2, np.nan), np.full(2, np.nan)
    d, i = cKDTree(pf[:, :2]).query(pj[:, :2], k=1)
    k = int(np.argmin(d))
    return float(d[k]), pf[i[k], :2], pj[k, :2]


def grasp_centre(kin, q, diameter: float, z_lo: float, z_hi: float, tol: float = 5e-5) -> dict:
    """Where a rigid vertical cylinder of `diameter` (spanning z_lo..z_hi) is held
    ANTIPODALLY by the two finger pads at the recorded arm pose q: the jaw angle at
    which the pads' free gap equals the diameter (bisection from the recorded, blocked
    q[5]), and the midpoint of the closest pad pair there.

    Antipodal because a cylinder touching both pads on a chord shorter than its
    diameter is squeezed sideways out of the grasp (MEASURED, module doc); the stable
    hold has opposed normals, so its centre is on the closing line. Returns {xy,
    jaw_rad, jaw_recorded_rad, residual_m (|gap - diameter|), line_deg}."""
    q = np.array(q, dtype=float)

    def gap_at(jaw):
        qq = q.copy()
        qq[5] = jaw
        return pad_gap(kin, qq, z_lo, z_hi)

    g0 = gap_at(q[5])[0]
    if not np.isfinite(g0):
        raise ValueError("no finger surface inside the cap's height band at this pose")
    # the gap grows with the jaw angle: bracket the diameter, then bisect
    lo, hi = (q[5], q[5] + np.radians(30.0)) if g0 < diameter else (np.radians(-11.5), q[5])
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        g = gap_at(mid)[0]
        if not np.isfinite(g) or abs(g - diameter) < tol:
            break
        lo, hi = (mid, hi) if g < diameter else (lo, mid)
    gap, a, b = gap_at(mid)
    u = (a - b) / np.linalg.norm(a - b)
    return {"xy": ((a + b) / 2).tolist(), "jaw_rad": float(mid), "jaw_recorded_rad": float(q[5]),
            "residual_m": float(abs(gap - diameter)), "line_deg": float(np.degrees(np.arctan2(u[1], u[0])))}
