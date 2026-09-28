"""The gripper's blocked-jaw data: series compliance and torque limit of the real gripper.

While the real jaws hold a cap, the encoder (observation.state) sits still while the
goal (action) is more closed; the gap between them is the only record of the squeeze
(joint_units report section 5). Six of the seven holds of the dataset are SQUEEZE
PLATEAUS at different commanded errors, and their encoder angle falls LINEARLY with
the error (MEASURED, see `analyse`):

    ep1 A 15.79 % at err 1.91 % | ep2 A 15.66 at 1.70 | ep0 A 14.80 at 4.1 |
    ep1 B 14.54 at 4.58 | ep1 C 14.21 at 4.5 | ep2 A' 14.0 at 5.5 | ep2 B 14.08 at 10.9 (!)

A rigid cap between rigid fingers would pin the encoder at one angle whatever the
squeeze. A spring k in series with the P servo gives exactly a line: at equilibrium
kp (theta - u) = k (theta_c - theta), i.e.

    theta = theta_c - s * min(err, e_sat),   s = kp / k,   e_sat = effort / kp

theta_c is the zero-force contact angle (the cap's width through the finger geometry
and the gripper map), s the compliance ratio and e_sat the error at which the
gripper's torque clamp takes over. ep2 B's hold at 10.9 % error sits far above the
line: the clamp. Fitting all seven (least squares, hinge model) gives the flex
stiffness k = kp / s and the gripper torque limit effort = kp * e_sat, both in the
units of the servo model (N.m/rad, N.m at the horn), with kp from servo_fit.

The clamp comes out at ~1.1 N.m, a third of the STS3215 full-duty stall (servo.
BAM_STALL), not the half that lerobot's Max_Torque_Limit = 50 % would give; the
gripper's Overload_Torque = 25 protection is the likely cause (UNVERIFIED: the
firmware's semantics are not documented well enough to model it). The number used is
the measured one, whatever sets it.

The same number is a check on the ep0 release (episodes report: the cap drops while
the encoder climbs 14.9 -> 16.6 %): the fingers let go when the horn passes theta_c.
"""

from __future__ import annotations

import math

import numpy as np

from .. import episodes as eplib
from ..units import GRIPPER

MIN_LEN = 10  # frames: a plateau must hold still at least this long
STILL_PCT = 0.15  # %: max frame-to-frame encoder change on a plateau
MIN_SQUEEZE_PCT = 1.0  # % the goal must be past the encoder
SHUT_PCT = 3.0  # % above which the jaw is on a cap, not shut on nothing


def plateaus(state_pct, action_pct, min_len: int = MIN_LEN) -> list[dict]:
    """Squeeze plateaus of one episode: runs where the encoder is still, above the shut
    stop, with the goal at least MIN_SQUEEZE_PCT more closed. Medians per run."""
    s, a = np.asarray(state_pct, float), np.asarray(action_pct, float)
    ok = (s > SHUT_PCT) & (s - a > MIN_SQUEEZE_PCT)
    still = np.r_[False, np.abs(np.diff(s)) <= STILL_PCT] & ok
    out, i = [], 0
    while i < len(s):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(s) and still[j + 1] and abs(s[j + 1] - s[i]) <= 2 * STILL_PCT:
            j += 1
        if j - i + 1 >= min_len:
            out.append({"start": int(i), "stop": int(j), "state_pct": float(np.median(s[i:j + 1])),
                        "err_pct": float(np.median(s[i:j + 1] - a[i:j + 1]))})
        i = j + 1
    return out


def hinge_fit(err, theta) -> dict:
    """theta = c - s * min(err, e_sat): exhaustive over e_sat on a fine grid, linear LSQ
    for (c, s) at each; units follow the inputs."""
    err, theta = np.asarray(err, float), np.asarray(theta, float)
    best = None
    for es in np.linspace(max(err.min(), 1e-6), err.max() * 1.5, 3000):
        x = np.minimum(err, es)
        A = np.stack([np.ones_like(x), -x], 1)
        (c, s), *_ = np.linalg.lstsq(A, theta, rcond=None)
        r = theta - (c - s * x)
        sse = float(r @ r)
        if best is None or sse < best[0] - 1e-15:
            best = (sse, c, s, es, r)
    sse, c, s, es, r = best
    return {"theta_c": float(c), "s": float(s), "e_sat": float(es), "rms": math.sqrt(sse / len(err)),
            "max_abs": float(np.abs(r).max()), "residuals": r.tolist(),
            "saturated": [bool(e > es) for e in err]}


def analyse(scene, kp_jaw: float) -> dict:
    """Plateaus of every used episode, the hinge fit (in % and in URDF rad), and the
    implied flex stiffness and torque limit for a Jaw kp (N.m/rad)."""
    u = scene.units()
    rows = []
    for e, ep in eplib.load(scene.ds).items():
        for p in plateaus(ep.state[:, GRIPPER], ep.action[:, GRIPPER]):
            p["episode"] = e
            rows.append(p)
    err = np.array([r["err_pct"] for r in rows])
    st = np.array([r["state_pct"] for r in rows])
    f = hinge_fit(err, st)
    b = math.radians(u.grip_b_deg_per_pct)  # rad per %
    stiff = kp_jaw / f["s"]
    effort = kp_jaw * f["e_sat"] * b
    return {"plateaus": rows, "fit_pct": f, "theta_c_rad": float(u.jaw_rad(f["theta_c"])),
            "flex_stiffness": float(stiff), "effort_limit": float(effort), "kp_jaw": float(kp_jaw)}


def flex_damping(stiffness: float, finger_inertia: float, ratio: float = 0.5) -> float:
    """ASSUMED damping ratio of the flex mode (~120 Hz, invisible at 30 fps): c = 2 zeta sqrt(k I)."""
    return 2 * ratio * math.sqrt(stiffness * finger_inertia)


def contact_gap_mm(scene, theta_c_rad: float) -> float:
    """Free gap between the finger meshes (contact band) at the zero-force contact angle:
    what the hold data say the cap's width is, under the scene's gripper map."""
    from ..kinematics import default

    return default().jaw_gap(theta_c_rad) * 1000
