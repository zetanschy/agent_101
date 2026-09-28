"""Real | MuJoCo + Blender | Isaac RTX for one episode, front camera over wrist camera.

The two sim columns are each track's render of its own NOMINAL action-mode replay on the
current scene: Blender renders the MuJoCo pose log (blender/ep<N>_mujoco-ep<N>_action_seed0_
eevee/), Isaac renders its own replay (isaac/render/ep<N>_ep<N>_action/rt/). The real column
is the dataset frame shifted by CALIB's camera latency (cameras.<cam>.latency.frames: the
real image of row t + lag shows the arm state of row t), the same alignment every track's
side-by-side uses, so all three columns show the same instant.

Decoding and encoding go through ffmpeg pipes (host ffmpeg 6.1; cv2 cannot read AV1 here).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from .. import episodes, paths, scene as scene_mod

W, H = 640, 480


def _frames(mp4: Path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    p.wait()


def _label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2
    img = np.array(img, copy=True)  # memmap rows and ffmpeg buffers are read-only; cv2 draws in place
    cv2.rectangle(img, (0, 0), (8 + 11 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def triptych(ds: str, episode: int, out: Path | None = None) -> Path:
    sc = scene_mod.load(ds)
    d = paths.out_dir(ds)
    ep = episodes.load(ds)[episode]
    cols = {"MuJoCo + Blender EEVEE": d / "blender" / f"ep{episode}_mujoco-ep{episode}_action_seed0_eevee",
            "Isaac RTX real-time": d / "isaac" / "render" / f"ep{episode}_ep{episode}_action" / "rt"}
    for k, v in cols.items():
        for cam in ("front", "grip"):
            if not (v / f"{cam}.mp4").exists():
                raise SystemExit(f"missing {v / f'{cam}.mp4'} ({k}); render it first (compare/README in real2sim/README.md)")
    lag = {cam: int(round(float(((sc["cameras"][cam].get("latency") or {}).get("frames", 0))))) for cam in ("front", "grip")}
    real = {cam: episodes.frames(ds, cam) for cam in ("front", "grip")}
    streams = {(k, cam): _frames(v / f"{cam}.mp4") for k, v in cols.items() for cam in ("front", "grip")}
    out = out or paths.out_dir(ds, "compare", create=True) / f"ep{episode}_triptych.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * W}x{2 * H}",
                            "-r", str(ep.fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
                           stdin=subprocess.PIPE)
    n = len(ep)
    for t in range(n):
        rows = []
        for cam in ("front", "grip"):
            r = real[cam][ep.start + min(t + lag[cam], n - 1)]
            tiles = [_label(r, f"real {cam} (ep{episode} t={t})")]
            for k in cols:
                f = next(streams[(k, cam)], None)
                tiles.append(_label(f if f is not None else np.zeros((H, W, 3), np.uint8), k))
            rows.append(np.concatenate(tiles, axis=1))
        enc.stdin.write(np.concatenate(rows, axis=0).tobytes())
    enc.stdin.close()
    enc.wait()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="real2sim compare video")
    ap.add_argument("--ds")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    p = triptych(paths.dataset(a.ds), a.episode, Path(a.out) if a.out else None)
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
