"""Arm and finger silhouettes as least-squares residuals (the ICP blocks of the fit).

Real side, per frame: the white-PLA mask (masks.white), its signed distance field
(SDF, px, negative inside) and its outline points with outward normals. Model side:
the white silhouette of the FK arm rendered by render.ArmRenderer at the current
parameters, whose boundary pixels are lifted to 3D points FIXED IN THEIR LINKS.

Two residuals per frame, both in pixels and both smooth in the parameters while the
point sets are held fixed (fit.py re-renders and re-associates between solves):

    model -> real   r_i = SDF_real(project(P_i))           a model edge point must lie
                                                           on a real edge
    real -> model   r_j = n_j . (u_j - project(P_nn(j)))   a real edge point must have a
                                                           model edge through it (point-
                                                           to-line, n = model normal)

The second keeps the model from sliding to cover only part of the real outline;
associations further than MAX_ASSOC_PX are dropped (cables, glare, the unmodelled
webcam body). Both are divided by the camera's pixel sigma and robustified by the
solver's soft_l1, and every frame is weighted to the same effective number of
points so that a frame with a long outline does not dominate.

DON'T CARE. Front: pixels white in > 60 % of an episode's frames that are not the
robot base (the mug's steel, the glare corner) and a 28 px disk around the wrist
lens (the webcam body and its always-on LED are not modelled). Model edge points
whose surface is the klip bracket are dropped for the same reason. Grip: the teal
cap (dilated 6 px) and every white blob that does not touch the image's bottom
rows (the fingers always do; the white desk seen at carry height never does).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import chain, masks
from .render import BG, DONTCARE, WHITE, ArmRenderer, outline_points
from ..kinematics import LINKS

SIGMA_PX = {"front": 1.5, "grip": 2.5}  # inlier edge noise (px): blur + mask threshold
MAX_ASSOC_PX = {"front": 12.0, "grip": 25.0}
SDF_CLAMP = 40.0
N_REF = 150  # effective points per frame and direction
BORDER = 4  # px: real outline points this close to the image edge are the frame, not the arm


def real_sdf(mask) -> np.ndarray:
    """SDF of a mask with the image border REPLICATED 32 px, so a region leaving the
    frame gets no artificial edge along the border."""
    import cv2

    pad = 32
    m = cv2.copyMakeBorder(np.asarray(mask, np.uint8), pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return masks.sdf(m > 0)[pad:-pad, pad:-pad]


@dataclass
class Frame:
    cam: str
    row: int  # global dataset row
    sdf: np.ndarray  # (H, W) float32
    care: np.ndarray  # (H, W) bool
    real_uv: np.ndarray  # (M, 2)
    real_n: np.ndarray  # (M, 2)
    # model state (associate)
    m_link: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    m_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    m_ok: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    d_real: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    d_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    d_n: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    w_m: float = 1.0
    w_d: float = 1.0


def front_dontcare(frames_img, rows_by_ep: dict, robot_masks: dict, frac: float = 0.6) -> dict:
    """{episode: (H, W) bool}: pixels white in more than `frac` of the frames in which the
    MODEL says no robot is there (robot_masks {row: (H, W) bool}, the rendered arm at the
    initial calibration, dilated by the caller). That finds the mug's steel and the glare
    corner without eating the arm near the base, where the arm sits in most frames."""
    import cv2

    out = {}
    for ep, rows in rows_by_ep.items():
        rows = [r for r in rows if r in robot_masks]
        white = np.zeros(frames_img.shape[1:3], np.float32)
        seen = np.zeros_like(white)
        for r in rows:
            free = ~robot_masks[r]
            white += masks.white(frames_img[r], "front") & free
            seen += free
        st = (white / np.maximum(seen, 1) > frac) & (seen >= 5)
        out[ep] = cv2.dilate(st.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return out


def grip_real_mask(img):
    """White fingers of a wrist frame and the care mask (see module doc)."""
    import cv2

    w = masks.white(img, "grip")
    H = w.shape[0]
    n, lab, st, _ = cv2.connectedComponentsWithStats(w.astype(np.uint8), connectivity=8)
    touch = np.zeros(n, bool)
    touch[np.unique(lab[H - 6:])] = True
    touch[0] = False
    fingers = touch[lab]
    other = w & ~fingers
    teal = cv2.dilate(masks.teal(img, "grip").astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    care = ~teal & ~(cv2.dilate(other.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    return fingers, care


def make_frame(cam: str, row: int, img, dontcare=None, step: int = 2) -> Frame:
    if cam == "front":
        m = masks.white(img, "front")
        care = np.ones(m.shape, bool) if dontcare is None else ~dontcare
    else:
        m, care = grip_real_mask(img)
    H, W = m.shape
    uv, n = masks.outline(m, step=step)
    iu, iv = uv[:, 0].astype(int), uv[:, 1].astype(int)
    ok = care[iv, iu] & (iu >= BORDER) & (iu < W - BORDER) & (iv >= BORDER) & (iv < H - BORDER)
    return Frame(cam, int(row), real_sdf(m), care, uv[ok], n[ok])


class SilhouetteBlock:
    """The frames of one camera and their model association."""

    def __init__(self, cam: str, frames: list, sdf_clamp: float = SDF_CLAMP, max_assoc: float | None = None,
                 ds: str | None = None):
        from .timing import clock

        self.cam, self.frames = cam, frames
        self.rows = np.array([f.row for f in frames], int)
        self.clock = clock(ds)
        self.sdf_clamp = sdf_clamp  # a far-off start needs a longer reach than the refinement
        self.max_assoc = max_assoc or MAX_ASSOC_PX[cam]

    # --- geometry helpers ------------------------------------------------------------
    def states(self, model) -> np.ndarray:
        """(T, 6) lerobot state at each frame's exposure (row - the camera's lag)."""
        return self.clock.at(self.rows, model.lag(self.cam))

    def _poses(self, model):
        return chain.link_T(model.q(self.states(model)))  # {link: (T, 4, 4)}

    def _T_world_cam(self, model, poses) -> np.ndarray:
        T = model.T_parent_cam(self.cam)
        if model.meta["cameras"][self.cam]["parent"] == "world":
            return np.broadcast_to(T, (len(self.frames), 4, 4))
        return poses[model.meta["cameras"][self.cam]["parent"]] @ T

    # --- association (renders) -----------------------------------------------------------
    def associate(self, model, renderer: ArmRenderer, dontcare_px: dict | None = None):
        """Render every frame at `model`, lift the white outline to link-local 3D points
        and fix the real->model correspondences. dontcare_px: {row: [(u, v, r)]} extra
        model-side don't-care disks (the wrist lens in the front view)."""
        import cv2
        from scipy.spatial import cKDTree

        cam = model.camera(self.cam)
        poses = self._poses(model)
        Twc = self._T_world_cam(model, poses)
        link_stack = np.stack([poses[n] for n in LINKS], 1)  # (T, L, 4, 4)
        q = model.q(self.states(model))
        H, W = cam.height, cam.width
        from ..camera import remap_maps

        maps = remap_maps(cam, renderer.specs[self.cam][1])
        for i, fr in enumerate(self.frames):
            r = renderer.render(self.cam, q[i], model.T_parent_cam(self.cam))
            link, pc, uv_pin = outline_points(r)
            # drop edges whose surface is the bracket (unmodelled webcam around it)
            cls_at = r.cls[uv_pin[:, 1].astype(int), uv_pin[:, 0].astype(int)]
            keep = cls_at != DONTCARE
            vv, uu = uv_pin[:, 1].astype(int), uv_pin[:, 0].astype(int)
            near_bracket = np.zeros(len(uu), bool)
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
                a, b = np.clip(uu + du, 0, r.cls.shape[1] - 1), np.clip(vv + dv, 0, r.cls.shape[0] - 1)
                near_bracket |= r.cls[b, a] == DONTCARE
            keep &= ~near_bracket
            link, pc = link[keep], pc[keep]
            pw = pc @ Twc[i][:3, :3].T + Twc[i][:3, 3]
            Tl = link_stack[i, link]
            local = np.einsum("nji,nj->ni", Tl[:, :3, :3], pw - Tl[:, :3, 3])
            uv = cam.project_cam(pc)  # the render camera IS the current camera
            inside = np.isfinite(uv).all(1) & (uv[:, 0] > BORDER) & (uv[:, 0] < W - 1 - BORDER) \
                & (uv[:, 1] > BORDER) & (uv[:, 1] < H - 1 - BORDER)
            iu = np.clip(np.nan_to_num(uv[:, 0]).round().astype(int), 0, W - 1)
            iv = np.clip(np.nan_to_num(uv[:, 1]).round().astype(int), 0, H - 1)
            ok = inside & fr.care[iv, iu]
            if dontcare_px and fr.row in dontcare_px:
                for (cu, cv, rad) in dontcare_px[fr.row]:
                    ok &= (uv[:, 0] - cu) ** 2 + (uv[:, 1] - cv) ** 2 > rad**2
            fr.m_link, fr.m_local, fr.m_ok = link[ok], local[ok], np.ones(ok.sum(), bool)
            # real -> model: nearest projected model edge point, with the model normal
            if ok.sum() < 10 or len(fr.real_uv) < 10:
                fr.d_real, fr.d_idx, fr.d_n = np.zeros((0, 2)), np.zeros(0, int), np.zeros((0, 2))
            else:
                muv = uv[ok]
                # model normal from the distorted-image model mask (render remapped)
                mm = cv2.remap((r.cls == WHITE).astype(np.uint8), maps[0], maps[1], cv2.INTER_NEAREST)
                sm = cv2.GaussianBlur(mm.astype(np.float32), (7, 7), 1.5)
                gx, gy = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
                d, j = cKDTree(muv).query(fr.real_uv, k=1)
                sel = d < self.max_assoc
                ju = np.clip(muv[j[sel], 0].round().astype(int), 0, W - 1)
                jv = np.clip(muv[j[sel], 1].round().astype(int), 0, H - 1)
                n = -np.stack([gx[jv, ju], gy[jv, ju]], 1)
                nn = np.linalg.norm(n, axis=1)
                good = nn > 1e-4
                fr.d_real = fr.real_uv[sel][good]
                fr.d_idx = j[sel][good]
                fr.d_n = n[good] / nn[good, None]
                fr.mm = mm
            fr.w_m = np.sqrt(N_REF / max(len(fr.m_link), N_REF // 4))
            fr.w_d = np.sqrt(N_REF / max(len(fr.d_idx), N_REF // 4))
        self._stack()

    def _stack(self):
        """Flatten per-frame point sets for vectorised residuals."""
        f_m, l_m, p_m, w_m = [], [], [], []
        f_d, i_d, u_d, n_d, w_d = [], [], [], [], []
        off = 0
        for i, fr in enumerate(self.frames):
            k = len(fr.m_link)
            f_m.append(np.full(k, i))
            l_m.append(fr.m_link)
            p_m.append(fr.m_local)
            w_m.append(np.full(k, fr.w_m))
            f_d.append(np.full(len(fr.d_idx), i))
            i_d.append(fr.d_idx + off)
            u_d.append(fr.d_real)
            n_d.append(fr.d_n)
            w_d.append(np.full(len(fr.d_idx), fr.w_d))
            off += k
        cat = lambda a, s: np.concatenate(a) if a else np.zeros(s)  # noqa: E731
        self.f_m, self.l_m, self.p_m, self.w_m = cat(f_m, 0).astype(int), cat(l_m, 0).astype(int), cat(p_m, (0, 3)), cat(w_m, 0)
        self.f_d, self.i_d, self.u_d, self.n_d, self.w_d = cat(f_d, 0).astype(int), cat(i_d, 0).astype(int), cat(u_d, (0, 2)), cat(n_d, (0, 2)), cat(w_d, 0)

    # --- residuals ---------------------------------------------------------------------
    def project(self, model) -> np.ndarray:
        """Current pixels of every model edge point (N, 2)."""
        poses = self._poses(model)
        Twc = self._T_world_cam(model, poses)
        link_stack = np.stack([poses[n] for n in LINKS], 1)
        Tl = link_stack[self.f_m, self.l_m]
        pw = np.einsum("nij,nj->ni", Tl[:, :3, :3], self.p_m) + Tl[:, :3, 3]
        Tc = Twc[self.f_m]
        pc = np.einsum("nji,nj->ni", Tc[:, :3, :3], pw - Tc[:, :3, 3])
        return model.camera(self.cam).project_cam(pc)

    def residuals(self, model) -> np.ndarray:
        uv = self.project(model)
        s = SIGMA_PX[self.cam]
        if getattr(self, "_sdf_stack", None) is None:
            self._sdf_stack = np.stack([fr.sdf for fr in self.frames])
        r_m = np.clip(masks.bilinear_stack(self._sdf_stack, self.f_m, uv, fill=self.sdf_clamp),
                      -self.sdf_clamp, self.sdf_clamp)
        r_m = np.nan_to_num(r_m, nan=self.sdf_clamp) * self.w_m / s
        r_d = np.einsum("ni,ni->n", self.n_d, self.u_d - np.nan_to_num(uv[self.i_d], nan=1e3)) * self.w_d / s
        return np.concatenate([r_m, r_d])

    def iou(self, model, renderer: ArmRenderer) -> np.ndarray:
        """Per-frame IoU of model vs real white masks inside the care region."""
        import cv2
        from ..camera import remap_maps

        cam = model.camera(self.cam)
        q = model.q(self.states(model))
        mx, my = remap_maps(cam, renderer.specs[self.cam][1])
        out = []
        for i, fr in enumerate(self.frames):
            r = renderer.render(self.cam, q[i], model.T_parent_cam(self.cam))
            mm = cv2.remap((r.cls == WHITE).astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
            real = fr.sdf < 0
            a, b = real & fr.care, mm & fr.care
            out.append((a & b).sum() / max((a | b).sum(), 1))
        return np.array(out)
