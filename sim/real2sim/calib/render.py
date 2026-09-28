"""MuJoCo EGL renders of the arm model, for silhouettes: class, link and depth per pixel.

The model is the core's MJCF (paths.MJCF, the arm every engine uses) plus mjlab's
printed klip_support bracket at its mount pose (READ from klip_camera.py, below),
seen through one pinhole camera per real camera. Distorted cameras are rendered as
the core prescribes: an oversized pinhole (camera.render_spec) whose pixels are
later related to the real image through the lens model, never by warping images
inside the least squares.

Per pixel a render gives:
    cls    0 background, 1 white PLA (the printed parts: the MJCF's rgba 1 .82 .12),
           2 black (the STS3215 servos, rgba .1 .1 .1), 3 bracket (don't care: its
           colour and the webcam body that sits on it are not modelled)
    link   index into kinematics.LINKS of the body the pixel's geom belongs to
    depth  metres along the camera's optical axis (OpenCV z)

`outline_points` turns the white silhouette's boundary into 3D points fixed in
their links, so a residual can re-project them under new parameters without
re-rendering (the ICP step of fit.py). At a boundary pixel the silhouette edge
belongs to whichever surface is in front: the white part itself against the
background, or the occluder (servo, bracket) where that hides the white part.

MEASURED here: 674x520 + 807x620 seg+depth pairs take 6.4 ms (RTX 3060, EGL).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .. import paths
from ..camera import CV_TO_GL, Camera, RenderSpec, to_mujoco
from ..kinematics import LINKS
from ..transforms import mat_to_quat
from ..units import URDF_JOINTS

# mjlab klip_camera.py _MOUNT_POS / _MOUNT_RPY_DEG (READ): gripper -> klip_support,
# extrinsic xyz euler (scipy "xyz"), bolted to the two wrist holes.
KLIP_MOUNT_POS = (-0.01455, 0.08710, -0.01039)
KLIP_MOUNT_RPY_DEG = (-179.755, -34.920, -90.061)
KLIP_STL = paths.MJLAB_SO101 / "xmls" / "assets" / "klip_support.stl"

BG, WHITE, BLACK, DONTCARE = 0, 1, 2, 3


def _rpy_extrinsic_quat(rpy_deg) -> np.ndarray:
    r, p, y = np.radians(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return mat_to_quat(Rz @ Ry @ Rx)


@dataclass
class Render:
    cam: str
    spec: RenderSpec
    cls: np.ndarray  # (H', W') uint8
    link: np.ndarray  # (H', W') int8, -1 background
    depth: np.ndarray  # (H', W') float32 m


class ArmRenderer:
    """One compiled model with a camera per entry of `specs` ({name: (parent, RenderSpec)})."""

    def __init__(self, specs: dict, mjcf=None):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self._mj = mujoco
        spec = mujoco.MjSpec.from_file(str(mjcf or paths.MJCF))
        grip = next(b for b in spec.bodies if b.name == "gripper")
        spec.add_mesh(name="klip_support", file=str(KLIP_STL))
        mount = grip.add_body(name="klip_support", pos=KLIP_MOUNT_POS, quat=_rpy_extrinsic_quat(KLIP_MOUNT_RPY_DEG))
        mount.add_geom(name="klip_support_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="klip_support",
                       group=1, contype=0, conaffinity=0, rgba=(0.45, 0.45, 0.47, 1.0))
        self.specs = dict(specs)
        for name, (parent, rs) in self.specs.items():
            body = spec.worldbody if parent == "world" else next(b for b in spec.bodies if b.name == parent)
            pin = Camera(name, rs.width, rs.height, rs.K, parent=parent)
            a = to_mujoco(pin)
            c = body.add_camera(name=name, pos=a["pos"], quat=a["quat"])
            c.resolution, c.sensor_size = a["resolution"], a["sensorsize"]
            c.focal_pixel, c.principal_pixel = a["focalpixel"], a["principalpixel"]
        spec.visual.global_.offwidth = max(rs.width for _, rs in self.specs.values()) + 8
        spec.visual.global_.offheight = max(rs.height for _, rs in self.specs.values()) + 8
        spec.visual.map.znear = 0.001  # x extent: the fingers are 3-10 cm from the wrist lens
        m = self.model = spec.compile()
        self.data = mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in URDF_JOINTS])
        self.cam_id = {n: m.camera(n).id for n in self.specs}
        # per-geom class and link, visual copies only (group 1; the collision copies are
        # group 0 at the same place and would z-fight)
        self.geom_cls = np.zeros(m.ngeom, np.uint8)
        self.geom_link = np.full(m.ngeom, -1, np.int8)
        body_link = {m.body(n).id: i for i, n in enumerate(LINKS)}
        body_link[m.body("klip_support").id] = LINKS.index("gripper")
        for g in range(m.ngeom):
            rgba = m.geom_rgba[g]
            name = m.geom(g).name
            self.geom_cls[g] = DONTCARE if name.startswith("klip") else (WHITE if rgba[0] > 0.9 and rgba[1] > 0.7 else BLACK)
            self.geom_link[g] = body_link.get(int(m.geom_bodyid[g]), -1)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1
        self._rnd = {n: mujoco.Renderer(m, rs.height, rs.width) for n, (_, rs) in self.specs.items()}

    def close(self):
        for r in self._rnd.values():
            r.close()

    def render(self, name: str, q, T_parent_cam) -> Render:
        """Render camera `name` at joint vector q (6 URDF rad) with pose T_parent_cam (OpenCV)."""
        mj, m, d = self._mj, self.model, self.data
        d.qpos[:] = 0.0
        d.qpos[self.qadr] = q
        cid = self.cam_id[name]
        T = np.asarray(T_parent_cam, dtype=float)
        m.cam_pos[cid] = T[:3, 3]
        m.cam_quat[cid] = mat_to_quat(T[:3, :3] @ CV_TO_GL)
        mj.mj_kinematics(m, d)
        mj.mj_camlight(m, d)
        r = self._rnd[name]
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=name, scene_option=self.opt)
        seg = r.render()
        r.disable_segmentation_rendering()
        r.enable_depth_rendering()
        r.update_scene(d, camera=name, scene_option=self.opt)
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        gid = seg[..., 0]
        ok = (seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
        g = np.where(ok, gid, 0)
        cls = np.where(ok, self.geom_cls[g], BG).astype(np.uint8)
        link = np.where(ok, self.geom_link[g], -1).astype(np.int8)
        return Render(name, self.specs[name][1], cls, link, np.where(ok, depth, np.inf).astype(np.float32))


def outline_points(r: Render, step: int = 2, border: int = 3):
    """3D points on the white silhouette's boundary, in their links' frames.

    Returns (link (N,) int, p_cam (N, 3) camera-frame metres, uv (N, 2) pinhole pixels).
    The caller maps p_cam to the link frame with that frame's pose (fit.py), because
    only it knows T_world_cam. Points within `border` px of the render edge are dropped
    (the silhouette continues outside the image; that edge is not an edge)."""
    import cv2

    fx, fy, cx, cy = r.spec.K
    wm = (r.cls == WHITE).astype(np.uint8)
    inner = cv2.erode(wm, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
    v, u = np.nonzero(wm & ~inner)
    H, W = wm.shape
    keep = (u >= border) & (u < W - border) & (v >= border) & (v < H - border)
    u, v = u[keep][::step], v[keep][::step]
    # the edge belongs to the nearer of the pixel and its nearest non-white 4-neighbour
    best_u, best_v, best_d = u.copy(), v.copy(), r.depth[v, u].copy()
    for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        uu, vv = u + du, v + dv
        nb = (r.cls[vv, uu] != WHITE) & (r.cls[vv, uu] != BG)
        closer = nb & (r.depth[vv, uu] < best_d - 0.002)
        best_u, best_v, best_d = np.where(closer, uu, best_u), np.where(closer, vv, best_v), np.where(closer, r.depth[vv, uu], best_d)
    link = r.link[best_v, best_u].astype(int)
    z = best_d.astype(float)
    p = np.stack([(best_u - cx) / fx * z, (best_v - cy) / fy * z, z], 1)
    ok = np.isfinite(z) & (link >= 0)
    return link[ok], p[ok], np.stack([u, v], 1)[ok].astype(float)
