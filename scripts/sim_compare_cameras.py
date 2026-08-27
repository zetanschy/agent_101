#!/usr/bin/env python3
"""Put the sim render next to a live capture, so the camera extrinsics can be tuned.

Intrinsics are pinned to the real lenses. Where the cameras SIT is not: the overhead
height is back-solved from one frame and the wrist offset is eyeballed. This is the
loop for fixing that.

    ./robot sim-play                 # render the sim cameras
    ./robot sim-compare-cameras      # capture real, write sim/outputs/compare_*.png

Then edit OVERHEAD_POS / OVERHEAD_YAW_DEG / WRIST_POS / WRIST_PITCH_DEG at the top of
sim/sim_agent101/tasks/push_t_env_cfg.py, re-run sim-play, and look again. Match the
framing, not the pixels -- the sim arm is yellow and the real one is white, and the
real mat has clutter the sim does not.

Aim for: the same things at the same edges. If the real overhead puts the arm base at
the bottom edge and the sim puts it in the middle, move the camera along +x.
"""

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SIM_OUT = ROOT / "sim" / "outputs"
PAIRS = {"front": ("front_c270.png", 0), "grip": ("grip_kwc500.png", 2)}
WARMUP_FRAMES = 40


def grab(index, size=(640, 480)):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    frame = None
    for _ in range(WARMUP_FRAMES):
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.02)
    cap.release()
    return frame


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--front", type=int, default=None, help="/dev/video index of the overhead C270")
    p.add_argument("--grip", type=int, default=None, help="/dev/video index of the wrist KWC-500")
    p.add_argument("--grid", action="store_true", help="overlay a thirds grid to compare framing")
    args = p.parse_args()

    missing = [n for n, (f, _) in PAIRS.items() if not (SIM_OUT / f).exists()]
    if missing:
        print(f"no sim render for {missing} -- run ./robot sim-play first", file=sys.stderr)
        return 1

    overrides = {"front": args.front, "grip": args.grip}
    written = []
    for name, (fname, default_dev) in PAIRS.items():
        dev = overrides[name] if overrides[name] is not None else default_dev
        real = grab(dev)
        if real is None:
            print(f"{name}: cannot open /dev/video{dev}, skipping", file=sys.stderr)
            continue
        sim = cv2.imread(str(SIM_OUT / fname))
        if sim.shape != real.shape:
            real = cv2.resize(real, (sim.shape[1], sim.shape[0]))
        panels = [label(real, f"REAL  {name}  /dev/video{dev}"), label(sim, f"SIM   {name}")]
        if args.grid:
            for panel in panels:
                h, w = panel.shape[:2]
                for i in (1, 2):
                    cv2.line(panel, (w * i // 3, 0), (w * i // 3, h), (0, 200, 0), 1)
                    cv2.line(panel, (0, h * i // 3), (w, h * i // 3), (0, 200, 0), 1)
        pair = np.hstack(panels)
        out = SIM_OUT / f"compare_{name}.png"
        cv2.imwrite(str(out), pair)
        written.append(out.relative_to(ROOT))
        print(f"{name}: {out.relative_to(ROOT)}")

    if written:
        print("\nMatch the framing, then edit the extrinsics in "
              "sim/sim_agent101/tasks/push_t_env_cfg.py and re-run ./robot sim-play.")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
