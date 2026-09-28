"""MuJoCo EGL renders of both cameras, warped through the real lens distortion, from pose logs.

    r = Renderer(built)                        # one mujoco.Renderer per camera, EGL
    img = r.render(data, "grip")                # (480, 640, 3) uint8, distorted like the KWC-500
    render_log(scene, log, out_mp4)             # real | sim, front over grip, one row per frame

A FIRST LOOK, not the photoreal path (LOOK/Blender owns appearance). MuJoCo's rasteriser
has no reflections (the steel mug interior) and no lens model, so each camera renders
its oversized pinhole (camera.render_spec: front 638 x 484, grip 771 x 584) and a
cached bilinear warp (the same map as camera.remap_maps, applied with precomputed
indices, ~10 ms per frame without cv2) turns it into the real camera's distorted
640 x 480 image. The wrist camera rides on `gripper`, so its pose comes from the
simulated arm of the frame being drawn, like the real one.

Rendering reads POSE LOGS (real2sim.poselog), not a live simulation: the log's joint
angles, flex deflection and object poses are written into an MjData and only
kinematics runs, so any engine's log (or the real episode's) renders the same way.
Only geom group 2 (visual meshes) is drawn: collision pieces stay invisible.
"""

from __future__ import annotations

import os

import numpy as np

from .. import camera as C
from . import model as mdl


class FastRemap:
    """camera.remap with the 4 bilinear taps precomputed once (pure numpy)."""

    def __init__(self, cam: C.Camera, spec: C.RenderSpec):
        mx, my = C.remap_maps(cam, spec)
        W, H = spec.width, spec.height
        x0, y0 = np.floor(mx).astype(np.int64), np.floor(my).astype(np.int64)
        wx, wy = (mx - x0).astype(np.float32), (my - y0).astype(np.float32)
        ok = (mx > -1) & (mx < W) & (my > -1) & (my < H)
        self.idx, self.w = [], []
        for dy, dx, w in ((0, 0, (1 - wx) * (1 - wy)), (0, 1, wx * (1 - wy)), (1, 0, (1 - wx) * wy), (1, 1, wx * wy)):
            xi, yi = x0 + dx, y0 + dy
            inside = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            self.idx.append((np.clip(yi, 0, H - 1) * W + np.clip(xi, 0, W - 1)).ravel())
            self.w.append((w * inside).ravel()[:, None])
        self.shape = (cam.height, cam.width)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        flat = img.reshape(-1, img.shape[-1]).astype(np.float32)
        out = sum(w * flat[i] for i, w in zip(self.idx, self.w))
        return np.clip(out + 0.5, 0, 255).astype(np.uint8).reshape(*self.shape, -1)


class Renderer:
    def __init__(self, b: mdl.Built, cams=("front", "grip")):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self.mj, self.b = mujoco, b
        self.cams = {}
        for name in cams:
            spec, cam = b.cameras[name]["spec"], b.cameras[name]["cam"]
            r = mujoco.Renderer(b.model, spec.height, spec.width)
            opt = mujoco.MjvOption()
            opt.geomgroup[:] = 0
            opt.geomgroup[mdl.VISUAL_GROUP] = 1
            self.cams[name] = (r, opt, FastRemap(cam, spec))

    def render(self, data, name: str, distort: bool = True) -> np.ndarray:
        r, opt, remap = self.cams[name]
        r.update_scene(data, camera=name, scene_option=opt)
        img = r.render()
        return remap(img) if distort else img.copy()

    def close(self):
        for r, _, _ in self.cams.values():
            r.close()


def set_from_log(b: mdl.Built, data, log: dict, k: int) -> None:
    """Pose the model at row k of a pose log (arm joints, flex, object origins)."""
    import mujoco

    data.qpos[b.qadr] = log["q"][k]
    if b.flex_qadr is not None and "x_flex" in log:
        data.qpos[b.flex_qadr] = log["x_flex"][k]
    names = list(log["objects"])
    for o in b.objects:
        if o["name"] in names:
            i = names.index(o["name"])
            p, q = log["obj_pos"][k, i], log["obj_quat"][k, i]
            if np.isfinite(p).all():
                data.qpos[o["qadr"]:o["qadr"] + 3] = p
                data.qpos[o["qadr"] + 3:o["qadr"] + 7] = q
    mujoco.mj_kinematics(b.model, data)
    mujoco.mj_camlight(b.model, data)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a small label into the top-left corner (PIL's default font; PIL ships with imageio)."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.ascontiguousarray(img))
    dr = ImageDraw.Draw(im)
    dr.rectangle([2, 2, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    dr.text((5, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def render_log(scene, log: dict, out_mp4: str, built: mdl.Built | None = None, every: int = 1,
               real: bool = True, cams=("front", "grip"), fps: float | None = None, progress=None) -> dict:
    """Write an mp4: per frame, for each camera a row [real | sim] (sim only if real=False).
    The real image beside sim frame k is row k + the camera's fitted latency, rounded
    (cameras.<cam>.latency.frames: 2.27 front, 1.42 wrist): a real image shows the arm
    that many frames before its row's observation.state, and each label names the row.
    Returns {'frames', 'seconds', 'path', 'real_offset_frames'}."""
    import time

    import imageio.v2 as imageio
    import mujoco

    from .. import episodes as eplib

    ep_i = int(log["meta"]["episode"])
    # the same objects and table the log was simulated with (its perturbation)
    b = built or mdl.build(scene, ep_i, perturb=mdl.Perturbation.from_dict(log["meta"].get("perturbation")),
                           placement=log["meta"].get("placement", "config"))
    d = mujoco.MjData(b.model)
    rnd = Renderer(b, cams)
    ep = eplib.load(scene.ds, include_excluded=True)[ep_i] if real else None
    frames = {c: ep.frames(c) for c in cams} if real else {}
    shift = {c: int(round(float(scene["cameras"][c].get("latency", {}).get("frames", 0.0)))) for c in cams}
    t0 = time.time()
    w = imageio.get_writer(out_mp4, fps=fps or log["fps"] / every, codec="libx264", quality=7,
                           macro_block_size=16, ffmpeg_log_level="error")
    n = 0
    for k in range(0, len(log["q"]), every):
        set_from_log(b, d, log, k)
        rows = []
        for c in cams:
            sim = _label(rnd.render(d, c), f"sim {c} f{int(log['frame'][k])}")
            if real:
                kr = int(np.clip(int(log["frame"][k]) + shift[c], 0, len(frames[c]) - 1))
                rows.append(np.concatenate([_label(np.asarray(frames[c][kr]), f"real {c} f{kr}"), sim], 1))
            else:
                rows.append(sim)
        w.append_data(np.concatenate(rows, 0))
        n += 1
        if progress and n % 60 == 0:
            progress(k, len(log["q"]))
    w.close()
    rnd.close()
    return {"frames": n, "seconds": round(time.time() - t0, 1), "path": str(out_mp4), "real_offset_frames": shift}
