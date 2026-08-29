#!/usr/bin/env python3
"""Measure where the two cameras actually are, in the robot's base frame.

Intrinsics come from scripts/sim_calibrate_cameras.py. This is the other half: the
POSE of each camera, which the sim currently estimates (OVERHEAD_POS was back-solved
from one frame's field of view, and the wrist camera's direction is a hand-set angle).

The board goes on the MAT, not on the arm. A ChArUco board big enough to detect
reliably from the overhead camera -- 25 mm squares at 0.63 mm/pixel -- is 60-80 mm
across at minimum, and the SO-ARM101's jaw is 22 mm wide. On the mat it can be A4,
both cameras can see it, and it sits exactly where the task happens.

    ./robot calib-board                     # print this first
    ./robot calib-capture                   # ~15 arm poses looking at the board
    ./robot calib-solve                     # writes config/extrinsics.json

How it works:
  wrist   eye-in-hand. Each pose gives base->gripper from forward kinematics and
          wristcam->board from solvePnP. cv2.calibrateHandEye solves gripper->wristcam,
          and base->board comes out with it.
  top     no hand-eye needed. Once base->board is known, one overhead frame gives
          overhead->board by solvePnP, so base->overhead = base->board * inv(overhead->board).

Accuracy is bounded by the arm, not the vision: printed links and serial-bus servos
with backlash mean a few millimetres of FK error. Spread the poses over the whole
workspace rather than clustering them, so that averages down instead of biasing.
Always record MEASURED joint angles, never commanded -- under gravity the arm settles
below its target, which in sim is up to 9 mm at full extension.
"""

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

def _load(name):
    """Import a sim_agent101 submodule without executing the package __init__.

    sim_agent101/__init__.py pulls in isaaclab to register the Gym envs, and these
    calibration tools run on the host where isaaclab does not exist. cameras.py and
    kinematics.py are deliberately dependency-free, so load them by path.
    """
    import importlib.util
    path = pathlib.Path(__file__).resolve().parent.parent / "sim" / "sim_agent101" / f"{name}.py"
    import sys as _sys
    spec = importlib.util.spec_from_file_location(f"_sim_agent101_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass looks the module up in sys.modules while the
    # class body is being processed, and blows up if it is not there yet.
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

_cameras = _load("cameras")
_kin = _load("kinematics")
CAMERAS, load_intrinsics = _cameras.CAMERAS, _cameras.load
CHAIN, fk_gripper = _kin.CHAIN, _kin.fk_gripper

SESSION = ROOT / "sim" / "outputs" / "calib"
OUT = ROOT / "sim" / "sim_agent101" / "config" / "extrinsics.json"
DICT = cv2.aruco.DICT_4X4_100
DEV = {"front": 0, "grip": 2}


def make_board(cols: int, rows: int, square_mm: float, marker_mm: float):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard((cols, rows), square_mm / 1000.0, marker_mm / 1000.0, d), d


def cmd_board(a) -> int:
    board, _ = make_board(a.cols, a.rows, a.square, a.marker)
    # 10 px/mm renders crisply on any home printer
    img = board.generateImage((int(a.cols * a.square * 10), int(a.rows * a.square * 10)), marginSize=40)
    SESSION.mkdir(parents=True, exist_ok=True)
    p = SESSION / f"charuco_{a.cols}x{a.rows}_{a.square:g}mm.png"
    cv2.imwrite(str(p), img)
    print(f"wrote {p.relative_to(ROOT)}")
    print(f"  {a.cols}x{a.rows} squares, {a.square} mm square, {a.marker} mm marker")
    print(f"  {a.cols*a.square:.0f} x {a.rows*a.square:.0f} mm -- print at 100% scale, NO 'fit to page',")
    print("  then MEASURE a square with calipers and pass the real value to --square.")
    print("  Tape it flat on the mat: any bow becomes pose error.")
    return 0


def grab(index: int, size=(640, 480), warmup: int = 40):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    f = None
    for _ in range(warmup):
        ok, x = cap.read()
        if ok:
            f = x
        time.sleep(0.02)
    cap.release()
    if f is None:
        raise SystemExit(f"/dev/video{index}: no frame")
    return f


def read_joints():
    """Measured joint angles, in radians, in CHAIN order."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    import os
    cfg = SO101FollowerConfig(
        port=os.environ.get("ROBOT_PORT", "/dev/ttyACM1"),
        id=os.environ.get("ROBOT_ID", "zetans_follower"),
        cameras={},
        use_degrees=True,
    )
    robot = SO101Follower(cfg)
    robot.connect()
    obs = robot.get_observation()
    robot.disconnect()
    return np.array([np.deg2rad(float(obs[f"{n}.pos"])) for n in CHAIN])


def cmd_capture(a) -> int:
    SESSION.mkdir(parents=True, exist_ok=True)
    board, adict = make_board(a.cols, a.rows, a.square, a.marker)
    det = cv2.aruco.CharucoDetector(board)
    n = len(list(SESSION.glob("pose_*.json")))
    print(f"capturing into {SESSION.relative_to(ROOT)} (already have {n})")
    print("Move the arm so the WRIST camera sees the board from a new angle, then ENTER.")
    print("Vary rotation, not just position -- pure translation makes the solve degenerate.")
    print("Type q then ENTER to stop.\n")
    while True:
        if input(f"[{n} captured] ENTER to grab, q to stop: ").strip().lower() == "q":
            break
        try:
            q = read_joints()
        except Exception as e:  # noqa: BLE001
            print(f"  cannot read joints: {e}")
            continue
        wrist = grab(DEV["grip"])
        top = grab(DEV["front"])
        cc, ci, _, _ = det.detectBoard(cv2.cvtColor(wrist, cv2.COLOR_BGR2GRAY))
        k = 0 if ci is None else len(ci)
        if k < a.min_corners:
            print(f"  only {k} charuco corners in the wrist view (need {a.min_corners}) -- skipped")
            continue
        cv2.imwrite(str(SESSION / f"wrist_{n:03d}.png"), wrist)
        cv2.imwrite(str(SESSION / f"top_{n:03d}.png"), top)
        (SESSION / f"pose_{n:03d}.json").write_text(json.dumps({
            "joints_rad": q.tolist(), "joint_names": CHAIN, "charuco_corners": k}))
        print(f"  captured #{n}: {k} corners, joints(deg) {np.round(np.rad2deg(q),1).tolist()}")
        n += 1
    print(f"\n{n} poses. {'Run ./robot calib-solve' if n >= 8 else 'Need at least 8.'}")
    return 0


def _pose_from_board(gray, det, board, K, D):
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None or len(ci) < 6:
        return None
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ci, board, K, D, None, None)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def cmd_solve(a) -> int:
    board, _ = make_board(a.cols, a.rows, a.square, a.marker)
    det = cv2.aruco.CharucoDetector(board)
    intr = load_intrinsics()
    if any(intr[n]["source"] != "calibrated" for n in ("front", "grip")):
        print("WARNING: camera intrinsics are spec-derived, not calibrated. Extrinsics")
        print("         inherit that error -- run ./robot sim-calibrate first.\n")

    def KD(name):
        c = intr[name]
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
        return K, np.array(c["distortion"], dtype=float)

    poses = sorted(SESSION.glob("pose_*.json"))
    if len(poses) < 8:
        raise SystemExit(f"need >= 8 poses, found {len(poses)} in {SESSION}")
    Kw, Dw = KD("grip")
    R_g2b, t_g2b, R_b2c, t_b2c = [], [], [], []
    used = 0
    for p in poses:
        meta = json.loads(p.read_text())
        img = cv2.imread(str(p.with_name(p.name.replace("pose_", "wrist_")).with_suffix(".png")))
        if img is None:
            continue
        T_cam_board = _pose_from_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), det, board, Kw, Dw)
        if T_cam_board is None:
            print(f"  {p.name}: board not resolvable, skipped")
            continue
        T_base_grip = fk_gripper(np.array(meta["joints_rad"]))
        R_g2b.append(T_base_grip[:3, :3]); t_g2b.append(T_base_grip[:3, 3])
        R_b2c.append(T_cam_board[:3, :3]); t_b2c.append(T_cam_board[:3, 3])
        used += 1
    print(f"using {used}/{len(poses)} poses")
    if used < 8:
        raise SystemExit("too few usable poses")

    R_gc, t_gc = cv2.calibrateHandEye(R_g2b, t_g2b, R_b2c, t_b2c, method=cv2.CALIB_HAND_EYE_PARK)
    T_grip_cam = np.eye(4); T_grip_cam[:3, :3] = R_gc; T_grip_cam[:3, 3] = t_gc.ravel()
    print("\nwrist camera, in the gripper frame:")
    print(f"  xyz mm {np.round(T_grip_cam[:3,3]*1000,2)}")

    # base->board from every pose; spread across them is the honest error estimate
    boards = [ (np.array([[*r[0],ti[0]],[*r[1],ti[1]],[*r[2],ti[2]],[0,0,0,1]]))
               for r, ti in zip(R_b2c, t_b2c, strict=True) ]
    ests = []
    for i in range(used):
        T_base_grip = np.eye(4); T_base_grip[:3,:3]=R_g2b[i]; T_base_grip[:3,3]=t_g2b[i]
        ests.append(T_base_grip @ T_grip_cam @ boards[i])
    P = np.array([e[:3, 3] for e in ests])
    T_base_board = ests[0].copy(); T_base_board[:3, 3] = P.mean(0)
    print(f"\nboard origin in the base frame: {np.round(P.mean(0)*1000,2)} mm")
    print(f"  spread across poses: {np.round(P.std(0)*1000,2)} mm  <- this IS your accuracy")

    result = {"wrist_cam_in_gripper": T_grip_cam.tolist(),
              "board_in_base": T_base_board.tolist(),
              "poses_used": used,
              "board_spread_mm": (P.std(0) * 1000).tolist()}

    tops = sorted(SESSION.glob("top_*.png"))
    Kt, Dt = KD("front")
    seen = []
    for f in tops:
        T = _pose_from_board(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY), det, board, Kt, Dt)
        if T is not None:
            seen.append(T_base_board @ np.linalg.inv(T))
    if seen:
        Q = np.array([s[:3, 3] for s in seen])
        T_base_top = seen[0].copy(); T_base_top[:3, 3] = Q.mean(0)
        print(f"\noverhead camera in the base frame ({len(seen)} frames saw the board):")
        print(f"  xyz mm {np.round(Q.mean(0)*1000,2)}   spread {np.round(Q.std(0)*1000,2)} mm")
        result["top_cam_in_base"] = T_base_top.tolist()
    else:
        print("\nno overhead frame resolved the board -- is it inside the top camera's view?")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("board", cmd_board), ("capture", cmd_capture), ("solve", cmd_solve)):
        s = sub.add_parser(name)
        s.add_argument("--cols", type=int, default=8)
        s.add_argument("--rows", type=int, default=11)
        s.add_argument("--square", type=float, default=25.0, help="square size in mm, MEASURED after printing")
        s.add_argument("--marker", type=float, default=18.0, help="aruco marker size in mm")
        if name == "capture":
            s.add_argument("--min-corners", type=int, default=12)
        s.set_defaults(fn=fn)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
