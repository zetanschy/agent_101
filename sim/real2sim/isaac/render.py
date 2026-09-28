"""RTX frames -> the real cameras' distorted 640x480 images -> mp4.

Isaac 4.5 renders square-pixel pinholes only (and approximates OpenCV distortion by a
radial f-theta fit at best, prior-art report §5), so each camera is rendered as the
oversized pinhole camera.render_spec(cam, square=True) and every frame is warped into
the real camera with camera.remap(img, camera.remap_maps(cam, spec)): the real K
(fx != fy on the wrist camera) and the real (k1, k2, p1, p2, k3). The encoder is the
host's ffmpeg (libx264, yuv420p, CRF 18) fed raw RGB on stdin, so no frames touch disk.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .. import camera as cam_mod

FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"


class VideoOut:
    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int = 18):
        self.path = Path(path)
        self.proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", str(self.path)], stdin=subprocess.PIPE)
        self.n = 0

    def write(self, rgb: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self) -> int:
        self.proc.stdin.close()
        return self.proc.wait()


class CameraRecorder:
    """One VideoOut per camera; add(name, pinhole_rgb) warps and encodes a frame."""

    def __init__(self, scene, names, out_dir: Path, fps: int):
        self.cams, self.maps, self.specs, self.videos = {}, {}, {}, {}
        for n in names:
            cam = scene.camera(n)
            spec = cam_mod.render_spec(cam, square=True)
            self.cams[n], self.specs[n] = cam, spec
            self.maps[n] = cam_mod.remap_maps(cam, spec)
            self.videos[n] = VideoOut(Path(out_dir) / f"{n}.mp4", cam.width, cam.height, fps)

    def add(self, name: str, pinhole_rgb: np.ndarray) -> np.ndarray:
        spec = self.specs[name]
        if pinhole_rgb.shape[:2] != (spec.height, spec.width):
            raise ValueError(f"{name}: render is {pinhole_rgb.shape[:2]}, expected {(spec.height, spec.width)}")
        img = cam_mod.remap(np.ascontiguousarray(pinhole_rgb[..., :3]), self.maps[name])
        self.videos[name].write(img)
        return img

    def close(self) -> dict:
        return {n: {"path": str(v.path), "frames": v.n, "rc": v.close()} for n, v in self.videos.items()}
