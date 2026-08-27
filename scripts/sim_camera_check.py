#!/usr/bin/env python3
"""Verify, against the real cameras, the assumptions sim_agent101/cameras.py rests on.

Two claims are checked, both on live capture:

  1. Native aspect. Grab the same static scene at 4:3 and at the sensor's native 16:9
     and template-match one inside the other. If 4:3 is a centred horizontal crop of
     16:9 (same vertical FOV), the sim must NARROW the horizontal FOV at 640x480
     rather than widen it. This is the assumption that decides ~11-15 degrees of view.

  2. Identity. Prints which /dev/video node answers to each name, so a replugged USB
     tree cannot silently swap the overhead and wrist cameras in the sim config.

Nothing here measures absolute FOV -- that needs a calibration target. See
scripts/sim_calibrate_cameras.py.

    ./robot sim-camera-check [--save DIR]
"""

import argparse
import pathlib
import sys
import time

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "sim"))
from sim_agent101.cameras import CAMERAS, CROP_4X3  # noqa: E402

# (name, 4:3 mode, native 16:9 mode)
MODES = {"front": ((640, 480), (1280, 720)), "grip": ((640, 480), (1920, 1080))}
WARMUP_FRAMES = 40  # auto-exposure and white balance need this long or frames come back black


def grab(index: int, size):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    frame = None
    for _ in range(WARMUP_FRAMES):
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.02)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if frame is None:
        raise SystemExit(f"/dev/video{index}: no frame at {size}")
    if got != size:
        print(f"    ! asked for {size}, camera gave {got}")
    return frame


def check(name, index, save):
    small, native = MODES[name]
    print(f"\n{name}  ({CAMERAS[name].model})  /dev/video{index}")
    a = grab(index, small)
    b = grab(index, native)
    if save:
        cv2.imwrite(str(save / f"{name}_4x3.png"), a)
        cv2.imwrite(str(save / f"{name}_native.png"), b)

    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # H1 "same hFOV": scale native to the 4:3 width; it should sit inside the 4:3 frame.
    s1 = ga.shape[1] / gb.shape[1]
    b1 = cv2.resize(gb, (ga.shape[1], round(gb.shape[0] * s1)))
    h1 = cv2.minMaxLoc(cv2.matchTemplate(ga, b1, cv2.TM_CCOEFF_NORMED))[1]
    # H2 "same vFOV": scale native to the 4:3 height; the 4:3 frame should sit inside it.
    s2 = ga.shape[0] / gb.shape[0]
    b2 = cv2.resize(gb, (round(gb.shape[1] * s2), ga.shape[0]))
    _, h2, _, loc = cv2.minMaxLoc(cv2.matchTemplate(b2, ga, cv2.TM_CCOEFF_NORMED))

    covered = ga.shape[1] / b2.shape[1]
    centred = (b2.shape[1] - ga.shape[1]) // 2
    print(f"    same-hFOV (vertical crop)   correlation {h1:.3f}")
    print(f"    same-vFOV (horizontal crop) correlation {h2:.3f}   <- expected winner")
    print(f"    4:3 covers {covered:.3f} of the native width (expected {CROP_4X3}), "
          f"offset x={loc[0]} (centred would be {centred})")

    ok = h2 > h1 and abs(covered - CROP_4X3) < 0.02 and abs(loc[0] - centred) <= 8
    print(f"    {'OK' if ok else 'MISMATCH -- cameras.py assumes a centred horizontal crop'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--front", type=int, default=None, help="/dev/video index of the overhead C270")
    p.add_argument("--grip", type=int, default=None, help="/dev/video index of the wrist KWC-500")
    p.add_argument("--save", type=pathlib.Path, default=None, help="write the captured frames here")
    args = p.parse_args()

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
    idx = {"front": args.front, "grip": args.grip}
    for name, default in (("front", 0), ("grip", 2)):
        if idx[name] is None:
            idx[name] = default
            print(f"note: --{name} not given, assuming /dev/video{default}")

    results = [check(name, idx[name], args.save) for name in ("front", "grip")]
    print()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
