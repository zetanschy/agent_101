"""Soft physical constraints: what the real episodes prove about the geometry.

Every real pick succeeded and nothing went through the table, so a correct
calibration must make these true (all residuals are one-sided hinges or Gaussian
pulls in mm / px, divided by the sigma shown):

    clearance   over every frame, the lowest point of the gripper, jaw and wrist links
                (support points of their meshes) is at most 1 mm below the table:
                max(0, (table - 1 mm) - z_min) / 0.25 mm -- the firmest term: the real arm
                cannot be in the table (the task's feasibility rule), and it does come
                down to it (the operator lowers the tips to the mat).
    between     at each pick's closure (2 frames after the close range, the cap still on
                the table and the jaw blocked on it) the cap's resting centre, in the
                gripper frame, is within 6 mm of the middle of the finger gap in the
                closing direction (the closing jaw can nudge it that far), within +-8 mm
                of the finger centre plane across the fingers, and between the fingertip
                and 20 mm above it along them (hinges, sigma 1 mm).
    blocked     at the same frame the free gap between the finger meshes in the
                fingertip contact band (kinematics.jaw_gap, 70-80 mm from the Jaw axis)
                at the jaw angle the gripper map gives for the recorded state equals the
                cap's mid-height diameter (sigma 1.5 mm: the contact height along the
                6 deg finger taper and PLA flex are not modelled).
    reach       at the same frame the lowest fingertip point is at least 2 mm below the
                cap's top (the fingers overlap the cap vertically, or nothing is held;
                hinge, sigma 1 mm). With `clearance` this brackets the table from both sides.
    drop        over each pick's drop frames the grasp site is above the rim (hinge,
                sigma 3 mm) and inside the opening by at least the cap radius
                (hinge, sigma 2 mm).

GRIPPER FRAME (MEASURED on the MJCF): x is the closing direction (the fixed tip at
x = -9.8 mm, the moving tip at +11.5 mm at Jaw 0), y across the fingers, the fingers
point to -z (tips at z = -104.4 mm).
"""

from __future__ import annotations

import functools

import numpy as np

from . import chain
from ..kinematics import FIXED_FINGER_MESH, GRASP_SITE_POS, default as kin_default

SIG_CLEAR, SIG_HINGE, SIG_GAP, SIG_DROP_Z, SIG_DROP_R = 0.00025, 0.001, 0.0015, 0.003, 0.002
REACH_MARGIN = 0.002  # at a closure the lowest fingertip is at least 2 mm below the cap's top
# the resting cap may be nudged while the jaw closes (MEASURED: the wrist view shows caps
# shifted ~5 mm between the last approach frame and the closed grasp, episode 1 cap B),
# so its REST centre need only be within 6 mm of the gap's middle
BETWEEN_TOL = 0.006
CLEAR_TOL = 0.001
WALL = 0.003  # assumed mug wall (config objects.mug.wall)


@functools.lru_cache(maxsize=None)
def low_points(n_dir: int = 600) -> dict:
    """{link: (N, 3)} support points of every visual mesh on the links that can reach
    the table (gripper, jaw, wrist): for n_dir Fibonacci directions, each link's vertex
    furthest along it. The lowest point of a rigid link in ANY pose is one of these
    (to within the direction spacing: ~0.1 mm on these 10 cm parts), at ~100 points per
    link instead of the ~4000 hull vertices (MEASURED: 2.0 s -> ~0.05 s per evaluation)."""
    i = np.arange(n_dir) + 0.5
    phi, th = np.arccos(1 - 2 * i / n_dir), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    out = {}
    for g in kin_default().meshes(visual=True):
        if g["link"] in ("gripper", "jaw", "wrist"):
            out.setdefault(g["link"], []).append(g["verts"])
    res = {}
    for k, v in out.items():
        v = np.concatenate(v)
        res[k] = v[np.unique(np.argmax(v @ dirs.T, axis=0))]
    return res


@functools.lru_cache(maxsize=None)
def gap_table():
    """(jaw rad (N,), band gap m (N,)) on a 0.25 deg grid, -12..30 deg (MEASURED meshes)."""
    kin = kin_default()
    ang = np.radians(np.arange(-12.0, 30.01, 0.25))
    return ang, np.array([kin.jaw_gap(a) for a in ang])


@functools.lru_cache(maxsize=None)
def fixed_inner_x() -> float:
    """x (gripper frame, m) of the fixed finger's inner face in the fingertip band:
    the 95th percentile of its vertices' x within 12 mm of the tip (the face the
    cap is pressed against; the finger's own tip rounding is below that)."""
    kin = kin_default()
    g = next(m for m in kin.meshes(visual=False) if m["mesh"] == FIXED_FINGER_MESH)
    v = g["verts"]
    zt = v[:, 2].min()
    band = v[(v[:, 2] < zt + 0.012)]
    return float(np.percentile(band[:, 0], 95))


class Physical:
    def __init__(self, scene, episodes: dict, eps_used, clearance_step: int = 2):
        """episodes: {i: real2sim.episodes.Episode}; eps_used: the episodes to constrain."""
        self.scene = scene
        rows_state = []
        for e in eps_used:
            ep = episodes[e]
            rows_state.append(ep.state[::clearance_step])
        self.clear_states = np.concatenate(rows_state).astype(float)
        self.low = low_points()
        # only frames that come near the table can violate the clearance: keep those whose
        # lowest point is under 60 mm at zero offsets (the fitted offsets move it by < 15 mm,
        # and the table is below 40 mm on every estimate)
        from .params import Model  # noqa: F401  (type only)
        z0 = self._zmin(self.clear_states, np.zeros(5), (-13.26, 1.336))
        self.clear_states = self.clear_states[z0 < 0.06]
        self.picks = []  # (ep, cap, up, state at closure, drop states (k, 6))
        for e in eps_used:
            ep, cfg = episodes[e], scene.episode(e)
            ups = {c["id"]: c["up"] for c in cfg["caps"]}
            for pk in cfg["picks"]:
                # 'between' needs the cap where it RESTED: the first closure (a regrasp
                # moves it: episode 1 cap C was tilted by its first close, 690-697);
                # 'blocked' needs the final hold
                k_first = min(pk.get("first_close", pk["close"])[1] + 2, len(ep) - 1)
                k = min(pk["close"][1] + 2, len(ep) - 1)
                a, b = pk["drop"]
                self.picks.append((e, pk["cap"], ups[pk["cap"]], ep.state[k].astype(float),
                                   ep.state[a:b + 1].astype(float), ep.state[k_first].astype(float)))

    def _zmin(self, states, off_deg, grip):
        q = np.empty_like(states)
        q[:, :5] = np.radians(states[:, :5] + off_deg)
        q[:, 5] = np.radians(grip[0] + grip[1] * states[:, 5])
        poses = chain.link_T(q)
        zmin = np.full(len(states), np.inf)
        for link, P in self.low.items():
            T = poses[link]
            zmin = np.minimum(zmin, (np.einsum("tj,nj->tn", T[:, 2, :3], P) + T[:, 2, 3:4]).min(1))
        return zmin

    def clearance(self, model):
        """Per frame: min over the low points of (z - table height under that point),
        plus the table's height at the origin, i.e. comparable with model.table_z()."""
        poses = chain.link_T(model.q(self.clear_states))
        tx, ty = model.get("table.tilt") if model.has("table.tilt") else (0.0, 0.0)
        # height above the tilted plane is a linear function of the world point, so only
        # the row (-tx, -ty, 1) of each link rotation is needed (3x cheaper than full points)
        a = np.array([-tx, -ty, 1.0])
        zmin = np.full(len(self.clear_states), np.inf)
        for link, P in self.low.items():
            T = poses[link]
            row = np.einsum("k,tkj->tj", a, T[:, :3, :3])
            off = T[:, :3, 3] @ a
            zmin = np.minimum(zmin, (row @ P.T).min(1) + off)
        return zmin

    def residuals(self, model) -> np.ndarray:
        tz = model.table_z()
        zmin = self.clearance(model)
        r = [np.maximum(0.0, (tz - CLEAR_TOL) - zmin) / SIG_CLEAR]
        D, h = model.cap_dims()
        lip = float(model.get("cap.lip")[0]) if model.has("cap.lip") else 0.0
        ang, gap = gap_table()
        xf = fixed_inner_x()
        R_rim, H = model.mug_dims()
        out = []
        for e, cap, up, s_close, s_drop, s_first in self.picks:
            q = model.q(s_close)
            g = float(np.interp(q[5], ang, gap))  # the final hold: blocked on the cap
            q1 = model.q(s_first)
            Tg = chain.link_T(q1[None])["gripper"][0]
            xy = model.cap_xy(e, cap)
            c = np.array([*xy, model.table_z(xy) + h / 2])
            cg = Tg[:3, :3].T @ (c - Tg[:3, 3])
            g1 = float(np.interp(q1[5], ang, gap))
            tip_z = GRASP_SITE_POS[2]
            # reach: to grasp the cap the fingers must overlap it vertically
            zlow = min((chain.transform(chain.link_T(q[None])[link][0], P)[:, 2].min()) for link, P in self.low.items()
                       if link != "wrist")
            out.append(max(0.0, zlow - (model.table_z(xy) + h - REACH_MARGIN)) / SIG_HINGE)
            out += [max(0.0, abs(cg[0] - (xf + g1 / 2)) - BETWEEN_TOL) / SIG_HINGE,
                    max(0.0, abs(cg[1]) - 0.008) / SIG_HINGE,
                    max(0.0, tip_z - cg[2]) / SIG_HINGE, max(0.0, cg[2] - (tip_z + 0.020)) / SIG_HINGE,
                    (g - (D + lip / 2)) / SIG_GAP]
            if model.has(f"mug.{e}"):
                mxy, _ = model.mug(e)
                rim_z = model.table_z(mxy) + H
                T = chain.link_T(model.q(s_drop))["gripper"]
                site = np.einsum("tij,j->ti", T[:, :3, :3], np.array(GRASP_SITE_POS)) + T[:, :3, 3]
                rad = np.linalg.norm(site[:, :2] - mxy, axis=1)
                lim = (R_rim - WALL) - (D + lip) / 2
                out += list(np.maximum(0.0, rim_z - site[:, 2]) / SIG_DROP_Z)
                out += list(np.maximum(0.0, rad - lim) / SIG_DROP_R)
        return np.concatenate([np.concatenate(r), np.array(out)])

    def report(self, model) -> dict:
        """Numbers for the report: clearance distribution and grasp geometry per pick."""
        tz = model.table_z()
        zmin = self.clearance(model)
        D, h = model.cap_dims()
        ang, gap = gap_table()
        xf = fixed_inner_x()
        picks = []
        for e, cap, up, s_close, s_drop, s_first in self.picks:
            q = model.q(s_close)
            g = float(np.interp(q[5], ang, gap))
            q1 = model.q(s_first)
            T = chain.link_T(q1[None])
            Tg = T["gripper"][0]
            xy = model.cap_xy(e, cap)
            c = np.array([*xy, model.table_z(xy) + h / 2])
            cg = Tg[:3, :3].T @ (c - Tg[:3, 3])
            site = Tg[:3, :3] @ np.array(GRASP_SITE_POS) + Tg[:3, 3]
            g1 = float(np.interp(q1[5], ang, gap))
            picks.append({"episode": e, "cap": cap, "cap_in_gripper_mm": (cg * 1000).round(2).tolist(),
                          "closing_offset_mm": round(float((cg[0] - (xf + g1 / 2)) * 1000), 2),
                          "jaw_deg": round(float(np.degrees(q[5])), 2), "gap_mm": round(g * 1000, 2),
                          "gap_minus_cap_mm": round(float((g - (D + (float(model.get("cap.lip")[0]) if model.has("cap.lip") else 0) / 2)) * 1000), 2),
                          "grasp_site_vs_cap_xy_mm": round(float(np.linalg.norm(site[:2] - xy) * 1000), 2),
                          "grasp_site_height_above_table_mm": round(float((site[2] - model.table_z(xy)) * 1000), 2)})
        return {"clearance_mm": {"min": float((zmin - tz).min() * 1000), "p1": float(np.percentile(zmin - tz, 1) * 1000),
                                 "p5": float(np.percentile(zmin - tz, 5) * 1000),
                                 "median": float(np.median(zmin - tz) * 1000),
                                 "below_table_frac": float(((zmin - tz) < -CLEAR_TOL).mean())},
                "picks": picks}
