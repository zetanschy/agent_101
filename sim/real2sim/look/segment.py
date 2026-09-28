"""Label maps of the real frames: arm, fingers, cap, mug, background (codes in look/__init__).

Two sources of evidence, combined so that neither alone decides a boundary:

MODEL. `LabelRenderer` renders the FK arm (MuJoCo EGL segmentation, host mujoco 3.6)
and the mug model at the scene pose, in both cameras. Each camera is drawn as the
core prescribes: an oversized pinhole (camera.render_spec), then remapped into the real
distorted pixels (camera.remap_maps, nearest-neighbour for labels). The model is the
core MJCF. On top of it goes mjlab's printed klip_support bracket (READ:
klip_camera.py _MOUNT_POS / _MOUNT_RPY_DEG, extrinsic xyz, the convention it
documents), plus the KWC-500 body as a 30 x 26 x 16 mm box on the bracket's hidden
face, centred on the 19 mm bore (READ: sim/sim_agent101/assets/objects.py BODY_W/D/L,
BORE_CENTRE, CAMERA_SIDE -1). This black box on the gripper shows plainly in every
front frame. Timing: about 9 ms per camera per frame, including the remap.

IMAGE. The model is only as good as the calibration: fit6r for the front camera, an
unfinished wrist calibration for the grip camera. So each boundary is re-decided
from the image inside a band around the model silhouette (`refine`). The inner
eroded region is trusted. A band pixel joins the object only if it carries
evidence and connects to that inner region.
    front arm   evidence = at least 20 levels BRIGHTER in max(R,G,B) than the
                episode's background plate. PLA and cables sit at 150-200 against
                the mat's 18. Darker pixels are the arm's own shadow, and those stay
                background, because a renderer must reproduce them there. Black
                servos on the black mat carry no evidence, so their edges follow
                the model.
    front mug   static per episode, so it is decided once, from the episode plate
                (mug present) against the clean plate (mug absent). Evidence is
                red enamel, or at least 15 levels brighter (the steel interior).
                The mug's own shadow is darker and so is excluded.
    grip        the fingers are rigid in the wrist image apart from the moving jaw,
                so their prior comes from the DATA (`GripFingerPrior`): how often
                each pixel is white PLA in the frames whose background is the black
                mat, at the nearest gripper openings. That prior works whatever the
                wrist calibration. The mug comes from the image alone (grip_mug): the
                red enamel anchors the mug-coloured components.
    caps        teal (colour.teal) in both views. A teal blob counts as CAP if it
                touches the robot (a held cap), or lies outside the mug. Otherwise it
                is a dropped cap seen through the opening, or its reflection on the
                steel: MUG.

The grip-camera finger and mug renders are kept too, for QA. Their IoU against the
image-derived fingers and mug measures the current wrist calibration, and
samples/index.json reports it.
"""

from __future__ import annotations

import os

import numpy as np

from .. import objects, paths
from ..camera import remap_maps, render_spec, to_mujoco
from ..kinematics import FIXED_FINGER_MESH, MOVING_JAW_MESH
from ..transforms import mat_to_quat
from ..units import URDF_JOINTS
from . import LABELS, colour

# Fine part codes of the model render; PART_TO_LABEL folds them into LABELS.
PART = {"printed": 1, "fingers": 2, "mug": 4, "servo": 5, "mount": 6}
PART_TO_LABEL = np.array([0, 1, 2, 0, 4, 1, 1, 0], np.uint8)

KLIP_STL = paths.MJLAB_SO101 / "xmls" / "assets" / "klip_support.stl"
KLIP_MOUNT_POS = (-0.01455, 0.08710, -0.01039)  # READ: mjlab klip_camera.py _MOUNT_POS
KLIP_MOUNT_RPY_DEG = (-179.755, -34.920, -90.061)  # READ: _MOUNT_RPY_DEG, extrinsic xyz
KWC500_BODY_HALF = (0.015, 0.013, 0.008)  # READ: objects.py BODY_W, BODY_D, BODY_L / 2
KWC500_BODY_POS = (0.017, -0.0175, -0.008)  # READ: BORE_CENTRE, glued below the plate (CAMERA_SIDE -1)

# Band widths (px). Front: fit6r puts arm silhouettes 1-3 px off (IoU 0.75 train)
# and cables stand up to ~5 px proud of the model. Grip: the data prior is blurred
# by the jaw-angle binning, about 1 % of opening = 1.3 deg, a few px at the tips.
BAND = {"front_robot": (2, 6), "front_mug": (3, 8), "grip_fingers": (3, 8)}
FRONT_ARM_BRIGHTER = 20  # levels in max(R,G,B) over the episode plate
FRONT_MUG_BRIGHTER = 15  # levels over the clean plate (steel interior)


def _rpy_extrinsic_quat(rpy_deg) -> np.ndarray:
    r, p, y = np.radians(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return mat_to_quat(Rz @ Ry @ Rx)


class LabelRenderer:
    """Model part maps of the FK arm and the mug, in the real (distorted) cameras."""

    def __init__(self, scene, cams=("front", "grip")):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self._mj = mujoco
        self.scene = scene
        self.cams = {c: scene.camera(c) for c in cams}
        self.specs = {c: render_spec(cam) for c, cam in self.cams.items()}
        spec = mujoco.MjSpec.from_file(str(paths.MJCF))
        spec.visual.global_.offwidth = max(s.width for s in self.specs.values())
        spec.visual.global_.offheight = max(s.height for s in self.specs.values())
        bodies = {b.name: b for b in spec.bodies}
        spec.add_mesh(name="klip_support", file=str(KLIP_STL))
        mount = bodies["gripper"].add_body(name="klip_support", pos=KLIP_MOUNT_POS,
                                           quat=_rpy_extrinsic_quat(KLIP_MOUNT_RPY_DEG))
        mount.add_geom(name="look_klip", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="klip_support",
                       group=1, contype=0, conaffinity=0)
        mount.add_geom(name="look_kwc500", type=mujoco.mjtGeom.mjGEOM_BOX, size=KWC500_BODY_HALF,
                       pos=KWC500_BODY_POS, group=1, contype=0, conaffinity=0)
        mug = spec.worldbody.add_body(name="look_mug", mocap=True)
        for name, part in objects.mug_parts(scene.mug_dims()).items():
            spec.add_mesh(name=f"look_mug_{name}", uservert=np.asarray(part.vertices, np.float32).ravel().tolist(),
                          userface=np.asarray(part.faces, np.int32).ravel().tolist())
            mug.add_geom(name=f"look_mug_{name}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"look_mug_{name}",
                         group=1, contype=0, conaffinity=0)
        for c, cam in self.cams.items():
            a = to_mujoco(self.specs[c].camera(cam))
            parent = spec.worldbody if cam.parent == "world" else bodies[cam.parent]
            mc = parent.add_camera(name=f"look_{c}", pos=a["pos"], quat=a["quat"])
            mc.resolution, mc.sensor_size = a["resolution"], a["sensorsize"]
            mc.focal_pixel, mc.principal_pixel = a["focalpixel"], a["principalpixel"]
        self.model = m = spec.compile()
        self.data = mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in URDF_JOINTS])
        part = np.zeros(m.ngeom + 1, np.uint8)
        for g in range(m.ngeom):
            if m.geom_group[g] != 1:
                continue  # collision copies (group 0) never render
            name = m.geom(g).name
            mesh = m.mesh(m.geom_dataid[g]).name if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else ""
            if name.startswith("look_mug"):
                part[g] = PART["mug"]
            elif name in ("look_klip", "look_kwc500"):
                part[g] = PART["mount"]
            elif mesh in (FIXED_FINGER_MESH, MOVING_JAW_MESH):
                part[g] = PART["fingers"]
            elif mesh.startswith("sts3215"):
                part[g] = PART["servo"]
            else:
                part[g] = PART["printed"]
        self.part_of_geom = part
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1
        self._renderers = {}
        self.maps = {c: remap_maps(cam, self.specs[c]) for c, cam in self.cams.items()}
        self._mug_id = m.body("look_mug").mocapid[0]
        self.set_mug(None)

    def set_mug(self, pose) -> None:
        """Place the mug model at (pos, quat) for an episode, or park it far away (None)."""
        pos, quat = pose if pose is not None else ((0.0, 0.0, -10.0), (1.0, 0.0, 0.0, 0.0))
        self.data.mocap_pos[self._mug_id] = pos
        self.data.mocap_quat[self._mug_id] = quat

    def _renderer(self, cam, mode: str):
        key = (cam, mode)
        if key not in self._renderers:
            s = self.specs[cam]
            r = self._mj.Renderer(self.model, s.height, s.width)
            (r.enable_segmentation_rendering if mode == "seg" else r.enable_depth_rendering)()
            self._renderers[key] = r
        return self._renderers[key]

    def _pose(self, q):
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[self.qadr] = np.asarray(q, dtype=float)
        self._mj.mj_forward(self.model, d)

    def _to_real(self, img, cam, interp):
        import cv2

        mx, my = self.maps[cam]
        return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def parts(self, q, cam: str) -> np.ndarray:
        """(H, W) uint8 PART codes in `cam`'s real distorted pixels, arm at URDF joints q."""
        import cv2

        mj, m = self._mj, self.model
        self._pose(q)
        r = self._renderer(cam, "seg")
        r.update_scene(self.data, camera=f"look_{cam}", scene_option=self.opt)
        seg = r.render()
        gid, typ = seg[..., 0], seg[..., 1]
        ok = (typ == int(mj.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
        pin = np.where(ok, self.part_of_geom[np.clip(gid, 0, m.ngeom)], 0).astype(np.uint8)
        return self._to_real(pin, cam, cv2.INTER_NEAREST)

    def normals_world(self, q, cam: str, T_world_parent=None) -> np.ndarray:
        """(H, W, 3) float32 world-frame surface normals of the model (0 where empty),
        from the depth render: back-projected through the pinhole spec, differentiated,
        oriented towards the camera. Used to pick horizontal faces for the white reference."""
        import cv2

        self._pose(q)
        r = self._renderer(cam, "depth")
        r.update_scene(self.data, camera=f"look_{cam}", scene_option=self.opt)
        z = r.render().astype(np.float64)
        fx, fy, cx, cy = self.specs[cam].K
        v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]]
        P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
        n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        n *= -np.sign((n * P).sum(-1, keepdims=True))  # face the camera
        far = z >= 0.99 * self.model.vis.map.zfar * self.model.stat.extent  # nothing drawn: the far plane
        n[far] = 0.0
        cam_obj = self.cams[cam]
        R = cam_obj.T_world_cam(T_world_parent)[:3, :3]
        nw = (n @ R.T).astype(np.float32)
        return np.stack([self._to_real(np.ascontiguousarray(nw[..., k]), cam, cv2.INTER_NEAREST)
                         for k in range(3)], -1)

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


# --- image refinement -----------------------------------------------------------------

def _disk(r: int):
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(m, r: int) -> np.ndarray:
    import cv2

    return cv2.dilate(np.asarray(m, np.uint8), _disk(r)) > 0 if r > 0 else np.asarray(m, bool)


def erode(m, r: int) -> np.ndarray:
    import cv2

    return cv2.erode(np.asarray(m, np.uint8), _disk(r)) > 0 if r > 0 else np.asarray(m, bool)


def refine(model_mask, evidence, r_in: int, r_out: int) -> np.ndarray:
    """model eroded by r_in, plus band pixels (within r_out of the model) that carry
    evidence and are 8-connected to that eroded core."""
    import cv2

    core = erode(model_mask, r_in)
    cand = core | (dilate(model_mask, r_out) & np.asarray(evidence, bool))
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[core])] = True
    keep[0] = False
    return keep[lab]


def split_by_nearest(mask, parts) -> np.ndarray:
    """Give every pixel of `mask` the LABEL of the nearest model robot pixel
    (arm vs fingers), so a refined silhouette keeps the model's part split."""
    import cv2

    lab = PART_TO_LABEL[parts]
    robot = (lab == LABELS["arm"]) | (lab == LABELS["fingers"])
    out = np.zeros(mask.shape, np.uint8)
    if not robot.any():
        out[mask] = LABELS["arm"]
        return out
    _, idx = cv2.distanceTransformWithLabels((~robot).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(robot)
    near = np.zeros(idx.max() + 1, np.uint8)
    near[idx[ys, xs]] = lab[ys, xs]
    out[mask] = near[idx[mask]]
    return out


def brighter(img, ref, levels: int) -> np.ndarray:
    return (np.asarray(img, np.int16).max(2) - np.asarray(ref, np.int16).max(2)) >= levels


def front_mug_mask(parts_mug, ep_plate_u8, clean_u8) -> np.ndarray:
    """The mug in the front view, once per episode: model silhouette refined by the
    plate (with mug) vs the clean plate (without)."""
    ev = colour.red(ep_plate_u8) | brighter(ep_plate_u8, clean_u8, FRONT_MUG_BRIGHTER)
    return refine(parts_mug, ev, *BAND["front_mug"])


def assign_caps(labels, teal_mask, mug_mask, touch_px: int = 6) -> np.ndarray:
    """Teal blobs: CAP where they touch the robot or lie outside the mug, else MUG."""
    import cv2

    robot = dilate((labels == LABELS["arm"]) | (labels == LABELS["fingers"]), touch_px)
    n, lab = cv2.connectedComponents(teal_mask.astype(np.uint8), connectivity=8)
    out = labels.copy()
    for i in range(1, n):
        blob = lab == i
        inside_mug = mug_mask[blob].mean() > 0.5
        out[blob] = LABELS["cap"] if (robot[blob].any() or not inside_mug) else LABELS["mug"]
    return out


def front_labels(img, parts, ep_plate_u8, mug_mask) -> tuple[np.ndarray, dict]:
    """Semantic labels of one front frame. parts: LabelRenderer.parts(q, 'front') with
    the mug parked (the mug is `mug_mask`, fixed per episode)."""
    lab_model = PART_TO_LABEL[parts]
    robot_model = (lab_model == LABELS["arm"]) | (lab_model == LABELS["fingers"])
    robot = refine(robot_model, brighter(img, ep_plate_u8, FRONT_ARM_BRIGHTER), *BAND["front_robot"])
    out = np.zeros(robot.shape, np.uint8)
    out[mug_mask] = LABELS["mug"]
    rl = split_by_nearest(robot, parts)
    out[robot] = rl[robot]
    out = assign_caps(out, colour.teal(img, "front"), mug_mask)
    iou = float((robot & robot_model).sum() / max((robot | robot_model).sum(), 1))
    return out, {"robot_model_vs_refined_iou": round(iou, 4)}


class GripFingerPrior:
    """Where the fingers sit in the wrist image, measured from the wrist frames alone.

    The fixed finger is rigid in the camera frame. The moving jaw is a function of
    the jaw angle only, i.e. of observation.state's gripper %. So in the frames whose
    background is the black mat, white PLA is exactly the fingers. Those frames are
    the ones whose top 35 % of rows has a 90th-percentile max(R,G,B) below 45: 62 % of
    frames. A median test would pass 99 %, desk and steel included. For a query opening, the prior is the per-pixel white
    frequency over the `k` such frames nearest in gripper %.
    """

    DARK_TOP_ROWS = 0.35
    DARK_P90 = 45

    def __init__(self, frames, gripper_pct, stride: int = 2):
        H, W = frames.shape[1:3]
        self.shape = (H, W)
        rows = np.arange(0, len(frames), stride)
        white, pct, top = [], [], int(self.DARK_TOP_ROWS * H)
        for i in rows:
            im = np.asarray(frames[i])
            if np.percentile(im[:top].max(2), 90) >= self.DARK_P90:
                continue
            white.append(np.packbits(colour.white_pla(im, "grip"), axis=-1))
            pct.append(float(gripper_pct[i]))
        self.white = np.array(white)
        self.pct = np.array(pct)
        self.n_frames, self.n_scanned = len(pct), len(rows)

    def frequency(self) -> np.ndarray:
        """White frequency over every dark-background frame, all openings."""
        return np.unpackbits(self.white, axis=-1)[..., : self.shape[1]].mean(0)

    def static_core(self) -> np.ndarray:
        """Pixels that are white PLA in 95 % of those frames whatever the opening: the
        fixed finger and the gripper body under the lens, 27.7 k px. It is the wrist
        camera's in-frame WHITE REFERENCE (colour.grip_white_balance)."""
        return erode(self.frequency() > 0.95, 3)

    def prior(self, pct: float, k: int = 15) -> tuple[np.ndarray, float]:
        """(white frequency (H, W) float32, max |gripper % difference| of the k used)."""
        near = np.argsort(np.abs(self.pct - pct))[:k]
        w = np.unpackbits(self.white[near], axis=-1)[..., : self.shape[1]].astype(np.float32)
        return w.mean(0), float(np.abs(self.pct[near] - pct).max())


def grip_mug(img, h, fingers, model) -> tuple[np.ndarray, int]:
    """The mug in a wrist frame, from the IMAGE: mug-coloured pixels (red enamel, steel
    grey S < 70 and 55 < V <= 225, teal inside) outside the fingers, closed 4 px and
    opened 3 px, then split into connected components. Kept are the components that
    touch red enamel (dilated 8 px), which anchors the mug: those holding at least 10 %
    of the best one's red. With no red in view, the largest component of at least
    5 k px that meets the model silhouette dilated 150 px is kept instead. The kept
    region is filled (dark reflections inside the opening stay mug).
    Calibration-independent by design. Build-phase probe: with the provisional wrist
    pose the MODEL silhouette sat up to 90 px off and 20 % too large, and aligning it
    to this evidence still failed. Returns (mask, red anchor px)."""
    import cv2

    v, sat = h[..., 2].astype(np.int16), h[..., 1].astype(np.int16)
    red = colour.red(img, h) & ~fingers
    cand = (red | ((sat < 70) & (v > 55) & (v <= 225)) | colour.teal(img, "grip", h)) & ~fingers
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, _disk(4))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _disk(3))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n <= 1:
        return np.zeros(fingers.shape, bool), 0
    anchor = np.bincount(lab[dilate(red, 8)].ravel(), minlength=n).astype(float)
    anchor[0] = 0
    if red.sum() > 800 and anchor.max() > 200:
        keep = anchor > 0.1 * anchor.max()
    else:
        area = st[:, cv2.CC_STAT_AREA].astype(float)
        area[0] = 0
        near = np.bincount(lab[dilate(model, 150)].ravel(), minlength=n) > 0 if model.any() else np.zeros(n, bool)
        area[~near] = 0
        keep = np.zeros(n, bool)
        keep[int(np.argmax(area))] = area.max() >= 5000
    m = keep[lab].astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(m.shape, np.uint8)
    cv2.drawContours(filled, cnt, -1, 1, thickness=-1)
    return (filled > 0) & ~fingers, int(red.sum())


def grip_labels(img, parts, finger_prior, prior_spread) -> tuple[np.ndarray, dict]:
    """Semantic labels of one wrist frame. parts: LabelRenderer.parts(q, 'grip') with
    the episode's mug placed; finger_prior: GripFingerPrior.prior(state gripper %)."""
    h = colour.hsv(img)
    prior = finger_prior > 0.5
    fingers = refine(prior, colour.white_pla(img, "grip"), *BAND["grip_fingers"])
    lab_model = PART_TO_LABEL[parts]
    mug_model = lab_model == LABELS["mug"]
    mug, red_px = grip_mug(img, h, fingers, mug_model)
    out = np.zeros(prior.shape, np.uint8)
    out[mug] = LABELS["mug"]
    out[fingers] = LABELS["fingers"]
    out = assign_caps(out, colour.teal(img, "grip", h), mug)
    fm = lab_model == LABELS["fingers"]
    iou = lambda a, b: round(float((a & b).sum() / max((a | b).sum(), 1)), 4) if (a | b).any() else None  # noqa: E731
    qa = {"fingers_model_vs_data_iou": iou(fm, fingers), "finger_prior_spread_pct": round(prior_spread, 2),
          "mug_model_vs_image_iou": iou(mug_model, mug), "mug_px": int(mug.sum()), "mug_red_anchor_px": red_px}
    return out, qa
