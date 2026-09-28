"""Forward kinematics of the SO-ARM101, through MuJoCo, on mjlab's so101_calib.xml.

The same MJCF every engine loads (paths.MJCF), with the `base` body at the world
origin, so poses come out in the URDF base frame that real2sim uses everywhere.
mujoco is imported lazily; this runs unchanged on mujoco 3.6 (host python3, Isaac's
45pysaac) and 3.11 (mjlab env). Joint limits are irrelevant here (mj_kinematics does
not clamp), so real poses beyond the URDF limits are fine.

    kin = Kinematics()                      # or kinematics.default()
    kin.link_poses(q)        {link: (pos (3,), quat wxyz (4,))}, q = 6 URDF radians
    kin.link_poses_batch(Q)  (T, L, 3), (T, L, 4) for Q (T, 6), links in LINKS order
    kin.grasp_site(q)        the fingertip midpoint mjlab steers by (GRASP_SITE_POS)
    kin.fingertips(q)        {'fixed', 'moving', 'mid'} world points
    kin.jaw_gap(jaw_rad)     free gap between the finger meshes in the contact band
    kin.meshes() / kin.world_vertices(q)     the visual meshes, for silhouettes

GRASP SITE vs FINGERTIPS. grasp_site() is mjlab's constant GRASP_SITE_POS
(so101_constants.py:74-83), the point the understand-phase fingertip errors were
measured at, so metrics default to it. fingertips() follows the jaw: each tip is the
mean of the 40 vertices of that finger mesh reaching furthest along the gripper's -z
(the finger direction), taken with the Jaw at mjlab's HOME angle (-0.15 rad); the
fixed tip (wrist_roll_follower_so101_v1) is stored in the gripper frame, the moving
tip (moving_jaw_so101_v1) in the jaw frame. MEASURED: their midpoint at HOME is
(-5.2, -0.2, -103.8) mm in the gripper frame, 2.2 mm from GRASP_SITE_POS (mjlab's
"40 lowest vertices" recipe does not reproduce its own constant either: -5.5 mm).

JAW GAP. What an object between the fingers feels is the free gap across the inner
faces where the tips squeeze it, not the distance between tip centroids. jaw_gap
samples both finger surfaces (area-weighted, seeded) and returns the smallest
distance between them among points 70-80 mm from the Jaw axis (the fingertip
contact band of the joint_units report, which measured 22.8 / 24.2 / 25.8 mm at
Jaw 5.5 / 6.6 / 7.9 deg and first touch at -11.5 deg). It is a free-space width: a
cap of diameter D blocks the jaw at jaw_rad_for_gap(D).
"""

from __future__ import annotations

import functools

import numpy as np

from . import paths
from .transforms import inv_T, make_T, quat_to_mat
from .units import URDF_JOINTS

LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")
# mjlab so101_constants.py:83, in the gripper body frame (READ; MEASURED there as the
# 40-lowest-vertex fingertip midpoint at HOME).
GRASP_SITE_POS = (-0.0074, 0.0, -0.1035)
# mjlab so101_constants.py:128-135, the pose GRASP_SITE_POS was measured at.
HOME = (-0.2585, -0.5715, 1.0236, 1.1098, -0.1473, -0.1500)
FIXED_FINGER_MESH = "wrist_roll_follower_so101_v1"  # on `gripper`
MOVING_JAW_MESH = "moving_jaw_so101_v1"  # on `jaw`
CONTACT_BAND = (0.070, 0.080)  # m from the Jaw axis (joint_units report §5)
TIP_REGION = (0.060, 0.120)  # the distal fingers, clear of the hinge
TIP_VERTICES = 40


class Kinematics:
    def __init__(self, mjcf=None):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf or paths.MJCF))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.body_id = {n: m.body(n).id for n in LINKS}
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in URDF_JOINTS])

    # --- poses ---------------------------------------------------------------------
    def _set(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,):
            raise ValueError(f"q must be 6 URDF radians ({', '.join(URDF_JOINTS)}), got shape {q.shape}")
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qadr] = q
        self._mj.mj_kinematics(self.model, self.data)

    def link_poses(self, q) -> dict:
        """{link: (pos (3,), quat wxyz (4,))} in the URDF base frame, for LINKS."""
        self._set(q)
        d = self.data
        return {n: (d.xpos[i].copy(), d.xquat[i].copy()) for n, i in self.body_id.items()}

    def link_T(self, q, link: str) -> np.ndarray:
        self._set(q)
        i = self.body_id[link]
        return make_T(self.data.xmat[i].reshape(3, 3), self.data.xpos[i])

    def link_poses_batch(self, Q, links=LINKS) -> tuple[np.ndarray, np.ndarray]:
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        ids = [self.body_id[n] for n in links]
        pos, quat = np.empty((len(Q), len(ids), 3)), np.empty((len(Q), len(ids), 4))
        for t, q in enumerate(Q):
            self._set(q)
            pos[t], quat[t] = self.data.xpos[ids], self.data.xquat[ids]
        return pos, quat

    def point(self, q, link: str, p_local) -> np.ndarray:
        """A point fixed in `link`'s frame, in the world."""
        T = self.link_T(q, link)
        return T[:3, :3] @ np.asarray(p_local, dtype=float) + T[:3, 3]

    def grasp_site(self, q) -> np.ndarray:
        return self.point(q, "gripper", GRASP_SITE_POS)

    def fingertips(self, q) -> dict:
        """World fingertip points {'fixed', 'moving', 'mid'} (see module doc)."""
        fixed_g, jaw_j = self._tips_local()
        self._set(q)
        out = {}
        for key, link, p in (("fixed", "gripper", fixed_g), ("moving", "jaw", jaw_j)):
            i = self.body_id[link]
            out[key] = self.data.xmat[i].reshape(3, 3) @ p + self.data.xpos[i]
        out["mid"] = 0.5 * (out["fixed"] + out["moving"])
        return out

    def fingertips_batch(self, Q) -> dict:
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        res = {k: np.empty((len(Q), 3)) for k in ("fixed", "moving", "mid")}
        for t, q in enumerate(Q):
            for k, v in self.fingertips(q).items():
                res[k][t] = v
        return res

    # --- meshes --------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def meshes(self, visual: bool = True) -> tuple:
        """Every mesh geom as dicts {geom, body, link, mesh, verts (V,3) in the BODY
        frame, faces (F,3), rgba}. visual=True: the group-1 visual copies (what a
        camera sees); False: the collision copies. Same meshes either way."""
        m, mj = self.model, self._mj
        out = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH or (m.geom_contype[g] == 0) != visual:
                continue
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            f = m.mesh_face[m.mesh_faceadr[k]: m.mesh_faceadr[k] + m.mesh_facenum[k]]
            R = quat_to_mat(m.geom_quat[g])
            b = int(m.geom_bodyid[g])
            out.append({"geom": g, "body": b, "link": m.body(b).name, "mesh": m.mesh(k).name,
                        "verts": v @ R.T + m.geom_pos[g], "faces": f.copy(), "rgba": m.geom_rgba[g].copy()})
        return tuple(out)

    def world_vertices(self, q, visual: bool = True) -> list:
        """[(mesh dict, verts (V,3) in the world)] at joint vector q."""
        self._set(q)
        d = self.data
        return [(g, g["verts"] @ d.xmat[g["body"]].reshape(3, 3).T + d.xpos[g["body"]]) for g in self.meshes(visual)]

    def _mesh(self, name: str) -> dict:
        return next(g for g in self.meshes(False) if g["mesh"] == name)

    # --- fingers ---------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def _tips_local(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.array(HOME)
        T_g = self.link_T(q, "gripper")
        d, tips = self.data, []
        for name in (FIXED_FINGER_MESH, MOVING_JAW_MESH):
            g = self._mesh(name)
            R, p = d.xmat[g["body"]].reshape(3, 3), d.xpos[g["body"]]
            w = g["verts"] @ R.T + p  # world
            along = (w - T_g[:3, 3]) @ T_g[:3, 2]  # gripper z; the fingers point to -z
            tip_w = w[np.argsort(along)[:TIP_VERTICES]].mean(0)
            tips.append(R.T @ (tip_w - p))
        return tips[0], tips[1]

    @functools.lru_cache(maxsize=None)
    def _finger_samples(self, n: int = 20000, seed: int = 0):
        """Surface samples of both fingers, fixed in the gripper frame and jaw in the jaw frame."""
        rng = np.random.default_rng(seed)
        out = []
        for name in (FIXED_FINGER_MESH, MOVING_JAW_MESH):
            g = self._mesh(name)
            tri = g["verts"][g["faces"]]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            idx = rng.choice(len(tri), n, p=area / area.sum())
            r1, r2 = rng.random(n), rng.random(n)
            s = np.sqrt(r1)
            out.append((1 - s)[:, None] * tri[idx, 0] + (s * (1 - r2))[:, None] * tri[idx, 1]
                       + (s * r2)[:, None] * tri[idx, 2])
        # jaw joint frame in the gripper frame at Jaw = 0; the Jaw turns about its local z
        self._set(np.zeros(6))
        T_gj0 = inv_T(self.link_T(np.zeros(6), "gripper")) @ self.link_T(np.zeros(6), "jaw")
        return out[0], out[1], T_gj0

    def _jaw_points_in_gripper(self, jaw_rad: float) -> np.ndarray:
        _, pj, T_gj0 = self._finger_samples()
        c, s = np.cos(jaw_rad), np.sin(jaw_rad)
        T = T_gj0 @ make_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))
        return pj @ T[:3, :3].T + T[:3, 3]

    def _radial(self, P) -> np.ndarray:
        _, _, T_gj0 = self._finger_samples()
        a, o = T_gj0[:3, 2], T_gj0[:3, 3]
        v = P - o
        return np.linalg.norm(v - np.outer(v @ a, a), axis=1)

    def jaw_gap(self, jaw_rad: float, band=CONTACT_BAND) -> float:
        """Smallest distance (m) between the finger surfaces within `band` (m from the
        Jaw axis) at this jaw angle. band=None uses the whole fingers."""
        pf, _, _ = self._finger_samples()
        pj = self._jaw_points_in_gripper(float(jaw_rad))
        if band is not None:
            pf, pj = pf[(self._radial(pf) >= band[0]) & (self._radial(pf) <= band[1])], \
                pj[(self._radial(pj) >= band[0]) & (self._radial(pj) <= band[1])]
        return _min_dist(pf, pj)

    @functools.lru_cache(maxsize=None)
    def _gap_table(self, band=CONTACT_BAND):
        ang = np.radians(np.arange(-15.0, 45.01, 0.5))
        return ang, np.array([self.jaw_gap(a, band) for a in ang])

    def jaw_rad_for_gap(self, gap: float, band=CONTACT_BAND) -> float:
        """The jaw angle at which the band gap equals `gap` (m): where a rigid object of
        that width blocks the closing jaw. Interpolated on a 0.5 deg table."""
        ang, g = self._gap_table(tuple(band) if band is not None else None)
        k = int(np.argmax(g > 1e-4))  # gap is flat ~0 while the fingers overlap
        return float(np.interp(gap, g[k:], ang[k:]))

    def jaw_touch_rad(self, threshold: float = 3e-4) -> float:
        """Closing, the first jaw angle where the distal finger region (TIP_REGION) closes
        to `threshold`. Near the hinge the two parts sit 0.5 mm apart at EVERY angle, so
        the whole-finger gap says nothing; 0.3 mm is the surface-sampling resolution."""
        lo, hi = np.radians(-20.0), np.radians(0.0)
        for _ in range(14):  # 20 deg / 2^14 = 0.001 deg
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.jaw_gap(mid, band=TIP_REGION) < threshold else (lo, mid)
        return 0.5 * (lo + hi)


def _min_dist(A, B) -> float:
    """Smallest distance between two point sets: scipy's KD-tree when present (every
    interpreter here has scipy except Blender's), else chunked brute force."""
    if not len(A) or not len(B):
        return float("nan")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return float(np.sqrt(min(((A[i:i + 128, None, :] - B[None]) ** 2).sum(-1).min()
                                 for i in range(0, len(A), 128))))
    return float(cKDTree(B).query(A, k=1)[0].min())


@functools.lru_cache(maxsize=None)
def default() -> Kinematics:
    """A shared Kinematics on paths.MJCF (compiles once per process, ~0.1 s)."""
    return Kinematics()


def link_poses(q) -> dict:
    return default().link_poses(q)


def grasp_site(q) -> np.ndarray:
    return default().grasp_site(q)


def fingertips(q) -> dict:
    return default().fingertips(q)
