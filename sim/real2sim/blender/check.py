"""`./robot real2sim blender check`: Blender's geometry against LOOK's MuJoCo label renderer.

    check [--episode 0] [--engine eevee|cycles] [--ds N]

Renders LOOK's sample frames of one episode in bl_render.py's SILHOUETTE mode (every
material an emission of its part code, no lights, one sample, no pixel filter), warps
them through the same lens remap as the beauty pass, and compares them part by part
with look/segment.LabelRenderer, which draws the FK arm and the mug with MuJoCo's own
camera (camera.to_mujoco, verified to < 0.5 px by the core render test). Both sides use
the scene's cameras and the recorded joints at the camera's time t = frame - latency
(joints interpolated linearly between rows; the job's link poses are nlerped, which at
these slow frames differs by far less than a pixel).

So a disagreement here is the renderer's own error (camera conversion, meshes, link
poses, remap), not the calibration: what the calibration leaves against the REAL
images is LOOK's sample QA (front robot model-vs-refined IoU, wrist fingers
model-vs-data IoU; samples/index.json). Caps are Blender-only (LOOK's label renderer
has none) and are left out of the comparison.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .. import objects as OBJ
from .. import paths
from ..scene import load as load_scene
from . import camera_model as CM
from .cli import run_blender
from .job import JobSpec, Source, build_job, camera_latency, sample_frames
from .post import write_json

PARTS = {"printed": 1, "fingers": 2, "mug": 4, "servo": 5, "mount": 6}


def joints_at(ds, episode: int, scene, frames, lag: float) -> np.ndarray:
    """URDF joint vectors (N, 6) at episode time frame - lag, linear between rows."""
    from .. import poselog

    log = poselog.from_dataset(ds, episode, units=scene.units())
    fr = np.asarray(log["frame"], float)
    t = np.clip(np.asarray(frames, float) - lag, fr[0], fr[-1])
    q = np.asarray(log["q"], float)
    return np.stack([np.interp(t, fr, q[:, j]) for j in range(q.shape[1])], 1)


def main(argv=None) -> int:
    import cv2

    ap = argparse.ArgumentParser(prog="./robot real2sim blender check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ds")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    a = ap.parse_args(argv)
    ds = paths.dataset(a.ds)
    scene = load_scene(ds)
    samples = sample_frames(ds, a.episode)
    frames = sorted(s["frame"] for s in samples)
    out = paths.out_dir(ds, "blender", f"check_ep{a.episode}_{a.engine}", create=True)
    t0 = time.time()
    job = build_job(JobSpec(ds, a.episode, Source.kinematic(), a.engine, frames, ("front", "grip"), {}),
                    out / "_job", log_fn=print)
    run_blender(out / "_job", out / "_pinhole", ("front", "grip"), None, mode="silhouette")

    from ..look.segment import LabelRenderer

    lr = LabelRenderer(scene)
    mug = next((o for o in OBJ.episode_objects(scene, a.episode) if o["name"] == "mug"), None)
    lr.set_mug((mug["pos"], mug["quat"]) if mug else None)
    res = {"dataset": ds, "episode": a.episode, "engine": a.engine, "scene_hash": job["scene_hash"],
           "frames": frames, "cameras": {}}
    for cam in ("front", "grip"):
        c = job["cameras"][cam]
        maps = CM.remap_maps(scene.camera(cam), CM.RenderSpec(c["pinhole"]["width"], c["pinhole"]["height"],
                                                             tuple(c["pinhole"]["K"])))
        lag, _ = camera_latency(scene, cam)
        Q = joints_at(ds, a.episode, scene, frames, lag)
        inter, union, disagree, npx = {k: 0 for k in PARTS}, {k: 0 for k in PARTS}, 0, 0
        for f, (fr, q) in enumerate(zip(frames, Q)):
            pin = CM.read_exr(out / "_pinhole" / cam / f"{c['frame_to_render'][f]:05d}.exr")[..., 0]
            code = np.rint(pin * 10).astype(np.uint8)
            bl = cv2.remap(code, maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mj = lr.parts(q, cam)
            keep = bl != 3  # caps: Blender only
            for k, v in PARTS.items():
                inter[k] += int(((bl == v) & (mj == v) & keep).sum())
                union[k] += int((((bl == v) | (mj == v)) & keep).sum())
            fg = ((bl > 0) | (mj > 0)) & keep
            disagree += int((fg & (bl != mj)).sum())
            npx += int(fg.sum())
        res["cameras"][cam] = {"iou": {k: round(inter[k] / union[k], 4) if union[k] else None for k in PARTS},
                               "foreground_disagreement": round(disagree / max(npx, 1), 4)}
        print(f"  {cam:5s} IoU " + "  ".join(f"{k} {v}" for k, v in res["cameras"][cam]["iou"].items())
              + f"   foreground pixels that disagree {res['cameras'][cam]['foreground_disagreement']:.2%}")
    for d in [*(out / "_pinhole").glob("*/*.exr"), *(out / "_job").glob("tex_*.npy")]:
        d.unlink()  # the EXRs are consumed; the textures are rebuilt by any re-run (cli.py)
    res["seconds"] = round(time.time() - t0, 1)
    write_json(out / "check.json", res)
    print(f"check done in {res['seconds']} s -> {out / 'check.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
