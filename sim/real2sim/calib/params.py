"""The calibration's unknowns as one flat vector: named blocks, units, priors, scene I/O.

Everything the replay's geometry depends on, in the URDF base frame (real2sim/__init__.py):

    front.rot  front.t       overhead C270 pose: rotation increment (rad, applied on the
                             initial camera-to-world rotation: R = R0 @ exp(rot)) and
                             centre (m) in the base frame
    front.f    front.c       fx, fy, cx, cy (px)
    front.k                  k1, k2, p1, p2, k3 (OpenCV)
    grip.*                   the wrist KWC-500: the same, pose in the `gripper` link frame
    robot.off                zero offsets of the 5 arm joints (deg): q = radians(v + off)
    robot.grip               gripper map (a deg, b deg/%): Jaw = radians(a + b pct)
    table.z  table.tilt      table top z (m) and slope (dz/dx, dz/dy)
    cap.dims                 cap diameter, height (m); every cap is one type
    mug.dims                 mug rim radius (outer, m), rim height above the table (m)
    cap.<ep>.<id>            one cap's resting centre xy (m)
    mug.<ep>                 mug axis xy (m) and handle yaw (rad; handle along the
                             mug's +x, objects.py convention)

A solver sees x = (value - x0) / step over the FREE entries only: steps are chosen so
one unit moves an image point by roughly a pixel (1 mrad, 1 mm, 1 px, 0.01 for the
distortion coefficients, 0.1 deg for joint offsets), which keeps the finite
differences and the trust region well scaled. Priors are Gaussian, in value units,
and only ever WEAK regularisers where the data cannot constrain (RULE 1): they are
listed with every result, and fit.py reports which parameters their prior, not the
data, decided.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..camera import Camera
from ..transforms import make_T
from ..units import Units

CAMS = ("front", "grip")


def expm_so3(w) -> np.ndarray:
    """Rotation vector (3,) -> 3x3 (Rodrigues)."""
    w = np.asarray(w, dtype=float)
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


@dataclass
class Block:
    name: str
    x0: np.ndarray  # initial value (value units)
    step: np.ndarray  # solver unit
    free: bool = True
    mu: np.ndarray | None = None  # prior mean (value units); None = no prior
    sigma: np.ndarray | None = None
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.x0)


@dataclass
class Layout:
    blocks: list = field(default_factory=list)
    R0: dict = field(default_factory=dict)  # {cam: initial camera-to-parent rotation}

    # --- construction --------------------------------------------------------------
    def add(self, name, x0, step, **kw):
        x0 = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
        step = np.broadcast_to(np.asarray(step, dtype=float), x0.shape).copy()
        for k in ("mu", "sigma", "lo", "hi"):
            if kw.get(k) is not None:
                kw[k] = np.broadcast_to(np.asarray(kw[k], dtype=float), x0.shape).copy()
        self.blocks.append(Block(name, x0, step, **kw))
        return self

    def block(self, name) -> Block:
        return next(b for b in self.blocks if b.name == name)

    def names(self, prefix=""):
        return [b.name for b in self.blocks if b.name.startswith(prefix)]

    def set_free(self, patterns, free=True):
        """Free/fix every block whose name starts with any of `patterns`."""
        for b in self.blocks:
            if any(b.name.startswith(p) for p in patterns):
                b.free = free
        return self

    def only_free(self, patterns):
        for b in self.blocks:
            b.free = any(b.name.startswith(p) for p in patterns)
        return self

    def copy(self) -> "Layout":
        return copy.deepcopy(self)

    # --- vectors -------------------------------------------------------------------
    def slices(self) -> dict:
        out, i = {}, 0
        for b in self.blocks:
            out[b.name] = slice(i, i + b.n)
            i += b.n
        return out

    def full0(self) -> np.ndarray:
        return np.concatenate([b.x0 for b in self.blocks])

    def free_mask(self) -> np.ndarray:
        return np.concatenate([np.full(b.n, b.free) for b in self.blocks])

    def steps(self) -> np.ndarray:
        return np.concatenate([b.step for b in self.blocks])

    def to_solver(self, full) -> np.ndarray:
        m = self.free_mask()
        return ((np.asarray(full) - self.full0()) / self.steps())[m]

    def from_solver(self, xs, base=None) -> np.ndarray:
        full = (self.full0() if base is None else np.asarray(base, dtype=float)).copy()
        m = self.free_mask()
        full[m] = self.full0()[m] + np.asarray(xs) * self.steps()[m]
        return full

    def bounds(self):
        lo, hi = [], []
        for b in self.blocks:
            if not b.free:
                continue
            lo.append((b.lo - b.x0) / b.step if b.lo is not None else np.full(b.n, -np.inf))
            hi.append((b.hi - b.x0) / b.step if b.hi is not None else np.full(b.n, np.inf))
        return (np.concatenate(lo), np.concatenate(hi)) if lo else (np.zeros(0), np.zeros(0))

    def prior_residuals(self, full) -> np.ndarray:
        """(value - mu) / sigma for every FREE entry with a prior."""
        sl, r = self.slices(), []
        for b in self.blocks:
            if b.free and b.mu is not None:
                r.append((full[sl[b.name]] - b.mu) / b.sigma)
        return np.concatenate(r) if r else np.zeros(0)

    def rebase(self, full) -> "Layout":
        """A copy whose x0 is `full` (so later stages start where this one ended);
        rotation increments are folded into R0 so they restart at zero."""
        new = self.copy()
        sl = self.slices()
        for b in new.blocks:
            b.x0 = np.asarray(full[sl[b.name]], dtype=float).copy()
        for cam in CAMS:
            if f"{cam}.rot" in sl:
                new.R0[cam] = self.R0[cam] @ expm_so3(full[sl[f"{cam}.rot"]])
                new.block(f"{cam}.rot").x0 = np.zeros(3)  # (no rotation block carries a prior)
        return new


class Model:
    """The scene a full parameter vector describes (read-only accessors)."""

    def __init__(self, layout: Layout, full, meta: dict):
        self.L, self.v, self.meta = layout, np.asarray(full, dtype=float), meta
        self._sl = layout.slices()

    def get(self, name) -> np.ndarray:
        return self.v[self._sl[name]]

    def has(self, name) -> bool:
        return name in self._sl

    # cameras
    def T_parent_cam(self, cam) -> np.ndarray:
        return make_T(self.L.R0[cam] @ expm_so3(self.get(f"{cam}.rot")), self.get(f"{cam}.t"))

    def lag(self, cam) -> float:
        """Frames the camera's image trails observation.state (timing.py)."""
        return float(self.get(f"{cam}.lag")[0]) if self.has(f"{cam}.lag") else 0.0

    def camera(self, cam) -> Camera:
        m = self.meta["cameras"][cam]
        return Camera(cam, m["width"], m["height"], (*self.get(f"{cam}.f"), *self.get(f"{cam}.c")),
                      tuple(self.get(f"{cam}.k")), m["parent"], self.T_parent_cam(cam))

    # robot
    def units(self) -> Units:
        a, b = self.get("robot.grip")
        return Units(tuple(float(x) for x in self.get("robot.off")), float(a), float(b))

    def q(self, state) -> np.ndarray:
        """lerobot state/action (..., 6) -> URDF radians under these offsets + gripper map."""
        s = np.asarray(state, dtype=float)
        q = np.empty_like(s)
        q[..., :5] = np.radians(s[..., :5] + self.get("robot.off"))
        a, b = self.get("robot.grip")
        q[..., 5] = np.radians(a + b * s[..., 5])
        return q

    # scene
    def table_z(self, xy=None) -> np.ndarray | float:
        z0 = float(self.get("table.z")[0])
        if xy is None:
            return z0
        tx, ty = self.get("table.tilt")
        xy = np.asarray(xy, dtype=float)
        return z0 + tx * xy[..., 0] + ty * xy[..., 1]

    def cap_dims(self) -> tuple[float, float]:
        D, h = self.get("cap.dims")
        return float(D), float(h)

    def mug_dims(self) -> tuple[float, float]:
        R, H = self.get("mug.dims")
        return float(R), float(H)

    def cap_xy(self, ep, cid) -> np.ndarray:
        return self.get(f"cap.{ep}.{cid}")

    def mug(self, ep) -> tuple[np.ndarray, float]:
        v = self.get(f"mug.{ep}")
        return v[:2], float(v[2])


def from_scene(scene, episodes=None) -> tuple[Layout, dict]:
    """Initial layout from the merged scene (provisional config + any existing layers).

    Initial values only (RULE 1): cameras.json / extrinsics.json / ChArUco / fit6r
    numbers enter here as the starting point and, for the front intrinsics, as the
    weak prior the task allows ("K with a weak prior at the trusted values")."""
    L = Layout()
    meta = {"cameras": {}, "episodes": {}}
    for cam in CAMS:
        c = scene.camera(cam)
        meta["cameras"][cam] = {"width": c.width, "height": c.height, "parent": c.parent}
        L.R0[cam] = c.T_parent_cam[:3, :3].copy()
        L.add(f"{cam}.rot", np.zeros(3), 1e-3, note="rad, on R0")
        L.add(f"{cam}.t", c.T_parent_cam[:3, 3], 1e-3, note="m")
        L.add(f"{cam}.f", c.K[:2], 1.0, note="px")
        L.add(f"{cam}.c", c.K[2:], 1.0, note="px")
        L.add(f"{cam}.k", c.dist, 0.01, note="k1 k2 p1 p2 k3")
        # latency (frames): starts at the episodes report's front-image-vs-state estimate
        # (2-3 frames, weak r = 0.6) for both cameras; free only where moving frames are fitted
        L.add(f"{cam}.lag", [2.0], 0.1, free=False, lo=[-2.0], hi=[8.0], note="frames")
    # the front K prior: trusted (two independent solves agree, cameras report §2),
    # sigma 2 % on f and 10 px on the principal point -- weak, the data decide
    fr = scene.camera("front")
    # MEASURED in the variant study (report.json 'degenerate'): with a 20 px / 10 px prior
    # the joint fit moved cy 31-55 px and f 20-55 px along the K-rotation-distortion
    # valley with no held-out gain, so the prior carries the trusted solve's own
    # uncertainty (0.5 % on f, 3 px on the principal point) and the data move it when
    # they are sure, not when they are compensating something else
    L.block("front.f").mu, L.block("front.f").sigma = np.array(fr.K[:2]), np.full(2, 5.0)
    L.block("front.c").mu, L.block("front.c").sigma = np.array(fr.K[2:]), np.full(2, 3.0)
    # distortion: shrink toward zero only (the stored front coefficients are suspect)
    for cam, s in (("front", (0.5, 2.0, 0.02, 0.02, 5.0)), ("grip", (0.5, 0.5, 0.02, 0.02, 0.5))):
        L.block(f"{cam}.k").mu, L.block(f"{cam}.k").sigma = np.zeros(5), np.array(s)
    L.block("front.t").lo = L.block("front.t").x0 - 0.2
    L.block("front.t").hi = L.block("front.t").x0 + 0.2
    L.block("grip.t").lo = L.block("grip.t").x0 - 0.06
    L.block("grip.t").hi = L.block("grip.t").x0 + 0.06
    r = scene["robot"]
    L.add("robot.off", r["joint_offsets"]["deg"], 0.1, mu=np.zeros(5), sigma=np.full(5, 5.0),
          lo=np.full(5, -12.0), hi=np.full(5, 12.0), note="deg")
    gm = r["gripper_map"]
    L.add("robot.grip", [gm["a_deg"], gm["b_deg_per_pct"]], [0.1, 0.002], lo=[-30.0, 0.8], hi=[5.0, 1.8],
          note="a deg, b deg/%")
    L.add("table.z", [scene.table_z()], 1e-3, lo=[-0.02], hi=[0.06], note="m")
    L.add("table.tilt", [0.0, 0.0], 1e-3, free=False, mu=np.zeros(2), sigma=np.full(2, 0.02), note="dz/dx dz/dy")
    cd = scene.cap_dims()
    # h: a weak prior (13 +- 2 mm, a 28 mm PET closure; episodes report) -- the front view
    # cannot see it and the wrist silhouettes trade it against the table height (MEASURED:
    # without the rim constraint one fold slid to h 19 mm with the table 17 mm lower)
    L.add("cap.dims", [cd["diameter"], cd["height"]], 1e-3, lo=[0.018, 0.005], hi=[0.036, 0.025],
          mu=[cd["diameter"], 0.013], sigma=[1.0, 0.002], note="D, h m")
    L.add("cap.lip", [0.0], 1e-3, lo=[-0.004], hi=[0.006], note="open-end diameter minus D, m")
    md = scene.mug_dims()
    L.add("mug.dims", [md["outer_diameter"] / 2, md["height"]], 1e-3, lo=[0.03, 0.06], hi=[0.05, 0.14],
          note="rim outer radius, rim height m")
    for key, ep in scene["episodes"].items():
        e = int(key)
        if episodes is not None and e not in episodes:
            continue
        meta["episodes"][e] = {"caps": [c["id"] for c in ep.get("caps", [])], "use": ep["use"]}
        for c in ep.get("caps", []):
            L.add(f"cap.{e}.{c['id']}", c["xy"], 1e-3, note="m")
        if "mug" in ep:
            L.add(f"mug.{e}", [*ep["mug"]["xy"], np.radians(ep["mug"]["handle_yaw_deg"])], [1e-3, 1e-3, 1e-2],
                  note="x y m, yaw rad")
    return L, meta
