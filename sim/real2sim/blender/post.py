"""Host side of a render: stream Blender's pinhole EXRs into camera frames, PNGs and mp4s.

    Consumer(job, job_dir, out_dir, responses, ...)   one per render
    .frame_done(cam, index)    called for every R2S_FRAME line Blender prints
    .close()                   flush videos, write the side-by-side, return the stats

Output frames are written IN ORDER per camera as soon as the unique render each one
needs exists (job.json cameras.<cam>.frame_to_render; indices are rendered in
first-use order, so a frame never waits for a later pose). An EXR is deleted once
its last output frame is written, unless keep_pinhole: an episode's pinholes would
otherwise be ~2 GB. Videos are H.264 (libx264, crf 18, yuv420p, the dataset's 30 fps)
through the host ffmpeg; the side-by-side is a 2 x 2 grid (real | render, front over
wrist) captioned with the source label, built after both passes from the render
videos and the decoded real frames (frames/<cam>.npy).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .. import episodes as E
from .camera_model import read_exr

FFMPEG = "ffmpeg"
CRF = 18


def ffmpeg_writer(path: Path, w: int, h: int, fps: int, crf: int = CRF) -> subprocess.Popen:
    cmd = [FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


class Consumer:
    def __init__(self, job: dict, job_dir: Path, pin_dir: Path, out_dir: Path, responses: dict, samples: dict,
                 fps: int, row0: int, video: bool = True, png_all: bool = False, keep_pinhole: bool = False):
        self.job, self.pin, self.out = job, Path(pin_dir), Path(out_dir)
        self.resp, self.samples, self.fps, self.row0 = responses, samples, fps, row0
        self.video, self.png_all, self.keep = video, png_all, keep_pinhole
        self.frames = job["frames"]
        self.next = {c: 0 for c in job["cameras"]}
        self.done = {c: -1 for c in job["cameras"]}
        self.writers = {}
        self.last_use = {}
        for c, cc in job["cameras"].items():
            last = {}
            for f, i in enumerate(cc["frame_to_render"]):
                last[i] = f
            self.last_use[c] = last
        self.post_s = {c: 0.0 for c in job["cameras"]}
        (self.out / "samples").mkdir(parents=True, exist_ok=True)
        if png_all:
            for c in job["cameras"]:
                (self.out / "frames" / c).mkdir(parents=True, exist_ok=True)

    def _writer(self, cam):
        if cam not in self.writers:
            self.writers[cam] = ffmpeg_writer(self.out / f"{cam}.mp4", 640, 480, self.fps)
        return self.writers[cam]

    def frame_done(self, cam: str, index: int) -> None:
        import cv2

        self.done[cam] = max(self.done[cam], index)
        f2r = self.job["cameras"][cam]["frame_to_render"]
        cache = {}
        while self.next[cam] < len(f2r) and f2r[self.next[cam]] <= self.done[cam]:
            f = self.next[cam]
            i = f2r[f]
            t = time.time()
            if i not in cache:
                cache = {i: read_exr(self.pin / cam / f"{i:05d}.exr")}
            fr = int(self.frames[f])
            img = self.resp[cam](cache[i], self.job["episode"], fr, self.row0 + fr)
            if self.video:
                self._writer(cam).stdin.write(np.ascontiguousarray(img).tobytes())
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if fr in self.samples:
                cv2.imwrite(str(self.out / "samples" / f"{self.samples[fr]}_{cam}.png"), bgr)
            if self.png_all:
                cv2.imwrite(str(self.out / "frames" / cam / f"{fr:05d}.png"), bgr)
            if not self.keep and self.last_use[cam].get(i) == f:
                (self.pin / cam / f"{i:05d}.exr").unlink(missing_ok=True)
            self.post_s[cam] += time.time() - t
            self.next[cam] += 1

    def close(self, label: str) -> dict:
        for w in self.writers.values():
            w.stdin.close()
            w.wait()
        stats = {"post_s_per_frame": {c: round(self.post_s[c] / max(len(self.frames), 1), 4) for c in self.post_s},
                 "frames_written": dict(self.next)}
        if self.video and len(self.frames) > 1 and all((self.out / f"{c}.mp4").exists() for c in self.job["cameras"]):
            t = time.time()
            side_by_side(self.out, self.job, self.row0, self.fps, label)
            stats["side_by_side_s"] = round(time.time() - t, 1)
        return stats


def _caption(img, text, scale=0.5):
    import cv2

    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def side_by_side(out: Path, job: dict, row0: int, fps: int, label: str) -> Path:
    """side_by_side.mp4: [real front | render front] over [real wrist | render wrist],
    1280 x 960, each tile captioned; the source label (e.g. 'kinematic: ... objects
    static') on the render tiles, so a kinematic replay cannot pass for a physical one."""
    import cv2

    cams = [c for c in ("front", "grip") if c in job["cameras"]]
    caps = {c: cv2.VideoCapture(str(out / f"{c}.mp4")) for c in cams}
    real = {c: E.frames(job["dataset"], c) for c in cams}
    path = out / "side_by_side.mp4"
    w = ffmpeg_writer(path, 1280, 480 * len(cams), fps)
    short = label if len(label) < 70 else label[:67] + "..."
    for fr in job["frames"]:
        rows = []
        for c in cams:
            ok, bgr = caps[c].read()
            ren = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else np.zeros((480, 640, 3), np.uint8)
            r = np.array(real[c][row0 + int(fr)])
            _caption(r, f"real {c}  ep{job['episode']} frame {fr}")
            _caption(ren, f"render {c} ({job['engine']})  {short}", 0.42)
            rows.append(np.concatenate([r, ren], 1))
        w.stdin.write(np.ascontiguousarray(np.concatenate(rows, 0)).tobytes())
    w.stdin.close()
    w.wait()
    for cap in caps.values():
        cap.release()
    return path


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
