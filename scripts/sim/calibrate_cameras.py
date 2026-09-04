#!/usr/bin/env python3
"""Replace the spec-derived camera intrinsics with real ones, from a checkerboard.

sim_agent101/cameras.py anchors focal length on the manufacturer's diagonal-FOV
number, which is a spec sheet and not this unit. This measures it. Print a
checkerboard (any size -- pass --cols/--rows as the number of INNER corners, and
--square as the printed square size in mm), hold it in front of one camera, and
move it around: different distances, different tilts, corners of the frame. Tilt
matters most -- a stack of head-on views cannot separate focal length from distance.

    ./robot sim-calibrate --camera front
    ./robot sim-calibrate --camera grip --cols 9 --rows 6 --square 25

Keys while running: SPACE captures the current view if a board is detected,
c calibrates and writes, q quits without writing. With --headless it captures
automatically, one frame per second, whenever the board is visible.

Writes fx/fy/cx/cy and the 5 distortion coefficients into
sim/sim_agent101/config/cameras.json with source="calibrated". Isaac Lab builds
its camera straight from those via PinholeCameraCfg.from_intrinsic_matrix, so the
sim picks the new values up with no code change. Distortion is recorded but NOT
applied by the pinhole sim camera -- it is there so you can undistort the real
frames instead, which is the side that should move.
"""

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]

def _load(name):
    """Import a sim_agent101 submodule without executing the package __init__.

    sim_agent101/__init__.py pulls in isaaclab to register the Gym envs, and these
    calibration tools run on the host where isaaclab does not exist. cameras.py and
    kinematics.py are deliberately dependency-free, so load them by path.
    """
    import importlib.util
    path = pathlib.Path(__file__).resolve().parents[2] / "sim" / "sim_agent101" / f"{name}.py"
    import sys as _sys
    spec = importlib.util.spec_from_file_location(f"_sim_agent101_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass looks the module up in sys.modules while the
    # class body is being processed, and blows up if it is not there yet.
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

_cameras = _load("cameras")
CAMERAS, CONFIG, default_config = _cameras.CAMERAS, _cameras.CONFIG, _cameras.default_config

MIN_VIEWS = 8
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_points(cols, rows, square_mm):
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts * (square_mm / 1000.0)


def calibrate(objp_list, imgp_list, size):
    rms, k, dist, _, _ = cv2.calibrateCamera(objp_list, imgp_list, size, None, None)
    return rms, k, dist


def write_config(name, k, dist, size, rms, views):
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else default_config(*size)
    cfg["cameras"][name] = {
        "width": size[0], "height": size[1],
        "fx": round(float(k[0, 0]), 3), "fy": round(float(k[1, 1]), 3),
        "cx": round(float(k[0, 2]), 3), "cy": round(float(k[1, 2]), 3),
        "distortion": [round(float(x), 6) for x in dist.ravel()[:5]],
        "hfov_deg": round(float(np.degrees(2 * np.arctan(size[0] / (2 * k[0, 0])))), 3),
        "vfov_deg": round(float(np.degrees(2 * np.arctan(size[1] / (2 * k[1, 1])))), 3),
        "source": "calibrated",
        "model": CAMERAS[name].model,
        "note": f"cv2.calibrateCamera, {views} views, RMS reprojection {rms:.4f} px",
    }
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg["cameras"][name]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", required=True, choices=sorted(CAMERAS), help="which camera to calibrate")
    p.add_argument("--device", type=int, default=None, help="/dev/video index (default: front=0, grip=2)")
    p.add_argument("--cols", type=int, default=9, help="inner corners across (default 9, i.e. a 10-square edge)")
    p.add_argument("--rows", type=int, default=6, help="inner corners down (default 6)")
    p.add_argument("--square", type=float, default=25.0, help="printed square size in mm (default 25)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--headless", action="store_true", help="no window; auto-capture 1 view/sec")
    p.add_argument("--views", type=int, default=15, help="target number of views in headless mode")
    args = p.parse_args()

    device = args.device if args.device is not None else {"front": 0, "grip": 2}[args.camera]
    size = (args.width, args.height)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open /dev/video{device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    objp = board_points(args.cols, args.rows, args.square)
    objp_list, imgp_list = [], []
    print(f"{args.camera} ({CAMERAS[args.camera].model}) on /dev/video{device} at {size[0]}x{size[1]}")
    print(f"board: {args.cols}x{args.rows} inner corners, {args.square} mm squares")
    print("SPACE capture   c calibrate+write   q quit" if not args.headless
          else f"headless: auto-capturing up to {args.views} views")

    last_auto = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (args.cols, args.rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
            )
            key = -1
            if not args.headless:
                vis = frame.copy()
                if found:
                    cv2.drawChessboardCorners(vis, (args.cols, args.rows), corners, found)
                cv2.putText(vis, f"{len(imgp_list)} views  {'BOARD' if found else 'no board'}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if found else (0, 0, 255), 2)
                cv2.imshow(f"calibrate {args.camera}", vis)
                key = cv2.waitKey(1) & 0xFF

            take = found and ((key == 32) or (args.headless and time.time() - last_auto > 1.0))
            if take:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                objp_list.append(objp.copy())
                imgp_list.append(refined)
                last_auto = time.time()
                print(f"  captured view {len(imgp_list)}")

            done_headless = args.headless and len(imgp_list) >= args.views
            if key == ord("c") or done_headless:
                if len(imgp_list) < MIN_VIEWS:
                    print(f"  need at least {MIN_VIEWS} views, have {len(imgp_list)}")
                    if done_headless:
                        break
                    continue
                rms, k, dist = calibrate(objp_list, imgp_list, size)
                out = write_config(args.camera, k, dist, size, rms, len(imgp_list))
                print(f"\nRMS reprojection error {rms:.4f} px over {len(imgp_list)} views")
                print(f"  fx={out['fx']} fy={out['fy']} cx={out['cx']} cy={out['cy']}")
                print(f"  hFOV={out['hfov_deg']} deg  vFOV={out['vfov_deg']} deg")
                spec = default_config(*size)["cameras"][args.camera]
                print(f"  spec said fx={spec['fx']} hFOV={spec['hfov_deg']} -> "
                      f"{100 * (out['fx'] / spec['fx'] - 1):+.1f}% focal length")
                print(f"wrote {CONFIG}")
                if rms > 1.0:
                    print("  ! RMS above 1 px: add views with more tilt and more of the frame covered")
                break
            if key == ord("q"):
                print("quit without writing")
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
