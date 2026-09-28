"""The caps and the mug as least-squares residuals, in both cameras.

CAP MODEL. A cylinder of diameter D and height h standing on the table at its
resting centre (x, y): bottom at table_z(x, y), top at + h. Every cap is the same
type. The two ends may differ (a tamper-band lip): D is the CLOSED end, D + lip the
open end (block cap.lip, fixed at 0 unless freed). Up-facing end: the closed end for
up='closed' caps, the open end for up='open' caps.

    front  the up-facing end's circle projected: its polygon centroid and equivalent
           radius vs the blob's area centroid and r_eq, averaged over the frames the
           cap rests untouched (sigma 0.3 px centroid, 0.5 px radius: the teal
           threshold sits at ~43 % of the edge ramp and the edge is ~2 px wide).
           The shaded side wall is not teal in this view (episodes report), so the
           blob is the up-facing end only.
    grip   the whole cylinder's silhouette (both end circles and three side rings,
           48 points each, projected through the wrist camera at that frame's FK) as a
           convex polygon; every VALID real outline point of the cap's teal blob must
           lie on it: r = signed distance to the polygon (px). Valid = not on the frame
           border, not within 4 px of the white fingers (occlusion edges are not the
           cap's edge). One direction only on purpose: occluded parts of the cap put no
           demand on the model.

MUG MODEL. Axis at (x, y), rim circle of radius R (the bright ring's radius) at
table_z + H; handle along the mug's +x, rotated by the yaw about z.
    front  rim: the per-episode median of the rim circle fits over unoccluded frames
           (observations.py) vs the projected rim polygon's centroid and mean radius;
           handle: the image direction from the rim centre to the handle blob's
           centroid vs the projected direction of a point 12 mm beyond the wall at
           mid-height (sigma 3 deg; insensitive to the assumed handle shape).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import chain, masks

N_CIRCLE = 48
SIDE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIG_FRONT_C, SIG_FRONT_R = 0.3, 0.5  # px
SIG_GRIP_EDGE = 2.0  # px, the wrist blob edge (defocus ~4 px ramp)
SIG_RIM_C, SIG_RIM_R = 0.5, 0.5
SIG_HANDLE_DEG = 3.0

_TH = np.linspace(0, 2 * np.pi, N_CIRCLE, endpoint=False)
_CIRC = np.stack([np.cos(_TH), np.sin(_TH)], 1)


def circle_pts(c_xy, z, radius) -> np.ndarray:
    return np.concatenate([c_xy + radius * _CIRC, np.full((N_CIRCLE, 1), z)], 1)


def poly_centroid_area(uv) -> tuple[np.ndarray, float]:
    x, y = uv[:, 0], uv[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = cr.sum() / 2
    if abs(A) < 1e-9:
        return uv.mean(0), 0.0
    return np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6 * A), abs(A)


def convex_hull(uv) -> np.ndarray:
    """CCW hull vertices (in image coords with v down, 'CCW' as computed on (u, v))."""
    from scipy.spatial import ConvexHull

    return uv[ConvexHull(uv).vertices]


def polygon_sdf(pts, poly) -> np.ndarray:
    """Signed distance of points to a convex polygon (px, positive outside): the max over
    edges of the signed line distance (exact inside and near edge interiors)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    e = b - a
    n = np.stack([e[:, 1], -e[:, 0]], 1)  # outward for scipy's CCW order in (u, v)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.einsum("mkj,kj->mk", pts[:, None, :] - a[None], n).max(1)


# --- caps ---------------------------------------------------------------------------

def cap_top_diameter(model, up: str) -> float:
    D, _ = model.cap_dims()
    lip = float(model.get("cap.lip")[0]) if model.has("cap.lip") else 0.0
    return D + lip if up == "open" else D


@dataclass
class FrontCapObs:
    ep: int
    cap: str
    up: str
    u: float
    v: float
    r_eq: float
    n: int


class FrontCaps:
    def __init__(self, items: list):
        self.items = items

    def residuals(self, model) -> np.ndarray:
        cam = model.camera("front")
        _, h = model.cap_dims()
        out = []
        for it in self.items:
            xy = model.cap_xy(it.ep, it.cap)
            P = circle_pts(xy, model.table_z(xy) + h, cap_top_diameter(model, it.up) / 2)
            c, A = poly_centroid_area(cam.project(P))
            out += [(c[0] - it.u) / SIG_FRONT_C, (c[1] - it.v) / SIG_FRONT_C,
                    (np.sqrt(A / np.pi) - it.r_eq) / SIG_FRONT_R]
        return np.array(out)



def cylinder_world(model, xy, up, dz_name: str | None = None) -> np.ndarray:
    """Side-wall sample points of a resting cap; dz_name: an optional per-cap height
    offset block (evaluate.py's wrist-only triangulation frees it)."""
    D, h = model.cap_dims()
    lip = float(model.get("cap.lip")[0]) if model.has("cap.lip") else 0.0
    z0 = model.table_z(xy) + (float(model.get(dz_name)[0]) if dz_name and model.has(dz_name) else 0.0)
    pts = []
    for f in SIDE_LEVELS:
        # radius along the height: the open end (bottom for up='closed', top for 'open') is D + lip
        frac_open = (1 - f) if up == "closed" else f
        pts.append(circle_pts(xy, z0 + f * h, (D + lip * frac_open) / 2))
    return np.concatenate(pts)


@dataclass
class GripCapFrame:
    row: int
    ep: int
    cap: str
    up: str
    state: np.ndarray  # (6,) lerobot at the image time
    pts: np.ndarray  # (M, 2) valid outline points of the blob


def grip_cap_frame(row, ep, cap, up, state, img, blob_uv, max_pts=160):
    """Outline points of the teal component nearest blob_uv, minus border/finger edges."""
    import cv2

    from .silhouette import grip_real_mask

    t = masks.teal(img, "grip")
    fingers, _ = grip_real_mask(img)
    comps = masks.components(t, 150)
    if not comps:
        return None
    k = int(np.argmin([np.hypot(*(c[1] - blob_uv)) for c in comps]))
    comp = comps[k][3]
    cnt, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnt, key=len)[:, 0, :].astype(float)
    H, W = comp.shape
    near = cv2.dilate(fingers.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ci = c.astype(int)
    ok = (ci[:, 0] > 2) & (ci[:, 0] < W - 3) & (ci[:, 1] > 2) & (ci[:, 1] < H - 3) & ~near[ci[:, 1], ci[:, 0]]
    c = c[ok]
    if len(c) < 30:
        return None
    if len(c) > max_pts:
        c = c[np.linspace(0, len(c) - 1, max_pts).astype(int)]
    return GripCapFrame(int(row), int(ep), cap, up, np.asarray(state, float), c)


class GripCaps:
    def __init__(self, frames: list, ds=None):
        from .timing import clock

        self.frames = frames
        self.rows = np.array([f.row for f in frames], int)
        self.clock = clock(ds)

    def states(self, model):
        return self.clock.at(self.rows, model.lag("grip"))

    def polygons(self, model):
        """Per frame the projected cylinder hull (or None), one batched projection."""
        cam = model.camera("grip")
        if not len(self.rows):
            return []
        Tc = chain.link_T(model.q(self.states(model)))["gripper"] @ model.T_parent_cam("grip")  # (F, 4, 4)
        cyl = {}
        P = np.stack([cyl.setdefault((fr.ep, fr.cap), cylinder_world(model, model.cap_xy(fr.ep, fr.cap), fr.up,
                                                                     f"capz.{fr.ep}.{fr.cap}")) for fr in self.frames])
        pc = np.einsum("fji,fnj->fni", Tc[:, :3, :3], P - Tc[:, None, :3, 3])  # world -> camera
        uv = cam.project_cam(pc)
        out = []
        for u in uv:
            u = u[np.isfinite(u).all(1)]
            out.append(convex_hull(u) if len(u) >= 3 else None)
        return out

    def residuals(self, model) -> np.ndarray:
        out = []
        for fr, poly in zip(self.frames, self.polygons(model)):
            w = np.sqrt(60.0 / len(fr.pts))  # each frame counts as ~60 edge points
            if poly is None:
                out.append(np.full(len(fr.pts), 50.0 * w / SIG_GRIP_EDGE))
            else:
                out.append(np.clip(polygon_sdf(fr.pts, poly), -60, 60) * w / SIG_GRIP_EDGE)
        return np.concatenate(out) if out else np.zeros(0)


# --- mug ----------------------------------------------------------------------------

@dataclass
class FrontMugObs:
    ep: int
    cx: float
    cy: float
    r: float
    handle_uv: tuple | None
    n: int


class FrontMug:
    def __init__(self, items: list):
        self.items = items

    def residuals(self, model) -> np.ndarray:
        cam = model.camera("front")
        R, H = model.mug_dims()
        out = []
        for it in self.items:
            xy, yaw = model.mug(it.ep)
            P = circle_pts(xy, model.table_z(xy) + H, R)
            uv = cam.project(P)
            c = uv.mean(0)
            r = np.linalg.norm(uv - c, axis=1).mean()
            out += [(c[0] - it.cx) / SIG_RIM_C, (c[1] - it.cy) / SIG_RIM_C, (r - it.r) / SIG_RIM_R]
            if it.handle_uv is not None:
                ph = np.array([*(xy + (R + 0.012) * np.array([np.cos(yaw), np.sin(yaw)])), model.table_z(xy) + 0.5 * H])
                pc = np.array([*xy, model.table_z(xy) + 0.5 * H])
                a, b = cam.project(np.stack([ph, pc]))
                ang_m = np.arctan2(a[1] - b[1], a[0] - b[0])
                ang_o = np.arctan2(it.handle_uv[1] - it.cy, it.handle_uv[0] - it.cx)
                out.append(np.degrees(np.angle(np.exp(1j * (ang_m - ang_o)))) / SIG_HANDLE_DEG)
        return np.array(out)


# --- the mug rim in the wrist view ------------------------------------------------------

SIG_WRIST_RIM = 2.0  # px, the rim's bright edge in the wrist view (defocus, reflections)
RIM_SEARCH_PX = 28.0  # px along the normal; the first association of a far start uses 60
N_RIM = 72


@dataclass
class WristRimFrame:
    row: int
    ep: int
    state: np.ndarray
    gray: np.ndarray  # (H, W) float32, blurred
    block: np.ndarray  # (H, W) bool: fingers / teal, where no rim point may be taken
    th: np.ndarray = None  # (K,) rim angles associated
    uv: np.ndarray = None  # (K, 2) ridge pixels
    n: np.ndarray = None  # (K, 2) image normals


def wrist_rim_frame(row, ep, state, img):
    """The frame's blurred grey image and its teal mask; the fingers are added to the
    blocking mask from the MODEL at association (the real white mask cannot be used
    here: over the mug the steel interior is white too and joins the fingers)."""
    import cv2

    teal = cv2.dilate(masks.teal(img, "grip").astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    gray = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
    return WristRimFrame(int(row), int(ep), np.asarray(state, float), gray, teal)


class WristRim:
    """The mug rim seen by the wrist camera while a cap is carried over it and let go.

    ICP-like: `associate` projects N_RIM rim points (fixed angles on the rim circle) at
    the current model and searches +-RIM_SEARCH_PX along each image normal for the
    brightest ridge (the rolled steel rim catches the key light; MEASURED on the
    release frames, e.g. episode 0 frame 362). A ridge must stand 25 grey levels above
    its profile's median and its segment must not cross the fingers or the held cap.
    `residuals` = n . (ridge - project(rim point)) / SIG_WRIST_RIM."""

    def __init__(self, frames: list, ds=None):
        from .timing import clock

        self.frames = frames
        self.rows = np.array([f.row for f in frames], int)
        self.clock = clock(ds)
        self._th = np.linspace(0, 2 * np.pi, N_RIM, endpoint=False)

    def states(self, model):
        return self.clock.at(self.rows, model.lag("grip"))

    def _rim_uv(self, model):
        cam = model.camera("grip")
        R, H = model.mug_dims()
        if not len(self.rows):
            return []
        Tg = chain.link_T(model.q(self.states(model)))["gripper"]
        out = []
        for i, fr in enumerate(self.frames):
            xy, _ = model.mug(fr.ep)
            P = np.stack([xy[0] + R * np.cos(self._th), xy[1] + R * np.sin(self._th),
                          np.full(N_RIM, model.table_z(xy) + H)], 1)
            out.append(cam.project(P, Tg[i]))
        return out

    def associate(self, model, renderer=None, search: float = RIM_SEARCH_PX, min_prom: float = 15.0):
        """Along each rim point's image normal (inside -> outside, +-search px) take the
        OUTERMOST ridge: walking in from the outside, the first local maximum that
        stands `min_prom` above the outside's median. Reflections inside the steel cup
        are brighter than the rim at times; the rim is always the outermost bright
        line (MEASURED on the release frames of episodes 0 and 1)."""
        import cv2

        t = np.arange(-search, search + 0.5, 1.0)
        cam = model.camera("grip")
        q = model.q(self.states(model)) if len(self.rows) else np.zeros((0, 6))
        maps = None
        for i, (fr, uv) in enumerate(zip(self.frames, self._rim_uv(model))):
            H, W = fr.gray.shape
            block = fr.block
            if renderer is not None:  # the model fingers (render.WHITE and the servo/bracket)
                from ..camera import remap_maps

                if maps is None:
                    maps = remap_maps(cam, renderer.specs["grip"][1])
                r = renderer.render("grip", q[i], model.T_parent_cam("grip"))
                fm = cv2.remap((r.cls > 0).astype(np.uint8), *maps, cv2.INTER_NEAREST)
                block = block | (cv2.dilate(fm, np.ones((15, 15), np.uint8)) > 0)
            c = np.nanmean(uv, axis=0)
            tang = np.roll(uv, -1, 0) - np.roll(uv, 1, 0)
            n = np.stack([tang[:, 1], -tang[:, 0]], 1)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            n *= np.sign(np.einsum("ni,ni->n", n, uv - c))[:, None]  # outward
            keep_th, keep_uv, keep_n = [], [], []
            for j in range(N_RIM):
                if not np.isfinite(uv[j]).all():
                    continue
                seg = uv[j] + t[:, None] * n[j]
                if (seg[:, 0] < 2).any() or (seg[:, 0] > W - 3).any() or (seg[:, 1] < 2).any() or (seg[:, 1] > H - 3).any():
                    continue
                iu, iv = seg[:, 0].round().astype(int), seg[:, 1].round().astype(int)
                prof = masks.bilinear(fr.gray, seg)
                prof[block[iv, iu]] = np.nan  # fingers / held cap: no rim evidence there
                out_part = prof[-8:]
                if np.isnan(out_part).any():
                    continue
                bg = np.median(out_part)
                k = None
                for i in range(len(t) - 2, 0, -1):  # from the outside in
                    if np.isnan(prof[i - 1:i + 2]).any():
                        break  # an occluder between the outside and here: stop
                    if prof[i] - bg > min_prom and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]:
                        k = i
                        break
                if k is None:
                    continue
                a, b, cc = prof[k - 1], prof[k], prof[k + 1]
                dk = 0.5 * (a - cc) / (a - 2 * b + cc) if (a - 2 * b + cc) < 0 else 0.0
                keep_th.append(j)
                keep_uv.append(uv[j] + (t[k] + dk) * n[j])
                keep_n.append(n[j])
            fr.th = np.array(keep_th, int)
            fr.uv = np.array(keep_uv).reshape(-1, 2)
            fr.n = np.array(keep_n).reshape(-1, 2)

    def residuals(self, model) -> np.ndarray:
        out = []
        for fr, uv in zip(self.frames, self._rim_uv(model)):
            if fr.th is None or not len(fr.th):
                continue
            w = np.sqrt(30.0 / len(fr.th))
            d = np.einsum("ni,ni->n", fr.n, fr.uv - np.nan_to_num(uv[fr.th], nan=1e3))
            out.append(np.clip(d, -60, 60) * w / SIG_WRIST_RIM)
        return np.concatenate(out) if out else np.zeros(0)

    def n_points(self) -> int:
        return int(sum(len(f.th) for f in self.frames if f.th is not None))


# --- the mug's outer wall in the front view --------------------------------------------

SIG_BODY_PX = 2.0  # the wall's bottom may be a dark enamel band, indistinguishable from the mat


@dataclass
class FrontBodyObs:
    ep: int
    pts: np.ndarray  # (M, 2) outer edge of the red wall crescent, beyond the rim ring


def front_body_obs(frames_img, ep, rows, rim_cxcyr) -> FrontBodyObs | None:
    """The red wall's outer edge, from the median of the episode's first frames (the mug
    never moved: rim within 1 px all episode, episodes report): the body blob's
    outline points more than 4 px outside the rim circle (inside, the rim itself
    occludes the wall) and not on the frame border."""
    import cv2

    img = np.median(np.stack([np.asarray(frames_img[r]) for r in rows]), 0).astype(np.uint8)
    comps = masks.components(masks.red(img), 300)
    if not comps:
        return None
    cx, cy, rr = rim_cxcyr
    comps = [c for c in comps if np.hypot(c[1][0] - cx, c[1][1] - cy) < 2.0 * rr]
    if not comps:
        return None
    body = comps[0][3]  # the largest red blob near the rim: the wall crescent (the handle is smaller)
    cnt, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnt, key=len)[:, 0, :].astype(float)
    H, W = body.shape
    d = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    ok = (d > rr + 4) & (c[:, 0] > 2) & (c[:, 0] < W - 3) & (c[:, 1] > 2) & (c[:, 1] < H - 3)
    return FrontBodyObs(int(ep), c[ok][::2]) if ok.sum() > 10 else None


class FrontBody:
    """Outer-wall points must lie on the projected cylinder hull: rim circle (R) at
    table + H and the bottom circle (R + half the 3 mm wall) on the table. This is what
    measures H in the front view (the bottom circle is displaced toward the nadir by
    perspective, ~27 px per 100 mm here)."""

    def __init__(self, items: list):
        self.items = [it for it in items if it is not None]

    def residuals(self, model) -> np.ndarray:
        cam = model.camera("front")
        R, H = model.mug_dims()
        out = []
        for it in self.items:
            xy, _ = model.mug(it.ep)
            tz = model.table_z(xy)
            uv = cam.project(np.concatenate([circle_pts(xy, tz + H, R + 0.0015), circle_pts(xy, tz, R + 0.0015)]))
            w = np.sqrt(40.0 / len(it.pts))
            out.append(np.clip(polygon_sdf(it.pts, convex_hull(uv)), -30, 30) * w / SIG_BODY_PX)
        return np.concatenate(out) if out else np.zeros(0)
