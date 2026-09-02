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
import os
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
URDF_TO_LEROBOT = _kin.URDF_TO_LEROBOT
JOINTS_FILE = ROOT / "sim" / "outputs" / "calib" / "joints.json"

SESSION = ROOT / "sim" / "outputs" / "calib"
OUT = ROOT / "sim" / "sim_agent101" / "config" / "extrinsics.json"
DICT = cv2.aruco.DICT_4X4_100
# /dev/videoN differs between host and container: compose pins the cameras to fixed
# nodes (CAM_*_INDEX in .env, 4 and 6), while on the host they enumerate as 0 and 2.
# capture runs in the container, so the env vars win there and the host values are
# only the fallback.
DEV = {"front": int(os.environ.get("CAM_FRONT_INDEX", 0)),
       "grip": int(os.environ.get("CAM_GRIP_INDEX", 2))}


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


def cmd_teleop(a) -> int:
    """Drive the follower from the leader and publish its measured joints.

    Runs INSIDE the container: it needs lerobot and both serial buses. The follower
    bus can only be opened once, so capture cannot read joints itself while this is
    running -- instead this writes them to a file about 30 times a second and capture
    reads that. The repo root is mounted into the container, so both see it.

    Splitting it this way also dodges the container's HEADLESS OpenCV, which cannot
    open the preview window capture needs.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    follower = SO101Follower(SO101FollowerConfig(
        port=os.environ.get("ROBOT_PORT", "/dev/ttyACM1"),
        id=os.environ.get("ROBOT_ID", "zetans_follower"), cameras={}, use_degrees=True))
    leader = SO101Leader(SO101LeaderConfig(
        port=os.environ.get("TELEOP_PORT", "/dev/ttyACM0"),
        id=os.environ.get("TELEOP_ID", "zetans_leader")))
    follower.connect()
    leader.connect()
    JOINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"teleop running. Move the LEADER; joints published to "
          f"{JOINTS_FILE.relative_to(ROOT)}. Ctrl-C to stop.", flush=True)
    try:
        while True:
            follower.send_action(leader.get_action())
            obs = follower.get_observation()
            JOINTS_FILE.write_text(json.dumps({
                "t": time.time(),
                "joints_deg": {u: float(obs[f"{lr}.pos"]) for u, lr in
                               ((u, URDF_TO_LEROBOT[u]) for u in CHAIN)}}))
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            leader.disconnect(); follower.disconnect()
        except Exception:  # noqa: BLE001
            pass
    return 0


def _joints_published(max_age_s: float = 2.0) -> np.ndarray:
    """Latest joints from the teleop publisher, in radians, in CHAIN order."""
    if not JOINTS_FILE.exists():
        raise RuntimeError("no joints.json -- start './robot calib-teleop' in another terminal")
    d = json.loads(JOINTS_FILE.read_text())
    age = time.time() - d["t"]
    if age > max_age_s:
        raise RuntimeError(f"joints.json is {age:.0f}s stale -- is calib-teleop still running?")
    return np.array([np.deg2rad(d["joints_deg"][n]) for n in CHAIN])


def _open(index: int, size=(640, 480)):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def cmd_capture(a) -> int:
    """Live view of the wrist camera; SPACE captures, q quits.

    Runs on the HOST: the container's OpenCV is headless, so cv2.imshow raises there.
    Joint angles come from the container instead, one shell-out per capture.
    """
    SESSION.mkdir(parents=True, exist_ok=True)
    board, _ = make_board(a.cols, a.rows, a.square, a.marker)
    det = cv2.aruco.CharucoDetector(board)
    n = len(list(SESSION.glob("pose_*.json")))

    wrist = _open(DEV["grip"])
    top = _open(DEV["front"])
    print(f"capturing into {SESSION.relative_to(ROOT)} (already have {n})")
    print("SPACE capture   q quit.  Move the arm so the WRIST camera sees the board")
    print("from a NEW ANGLE each time -- vary rotation, not just position.\n")
    try:
        while True:
            ok, frame = wrist.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cc, ci, _, _ = det.detectBoard(gray)
            k = 0 if ci is None else len(ci)
            vis = frame.copy()
            if k:
                cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci, (0, 255, 0))
            good = k >= a.min_corners
            colour = (0, 220, 0) if good else (0, 0, 255)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(vis, f"{k} corners (need {a.min_corners})   captured {n}   "
                             f"{'SPACE to grab' if good else 'board not visible enough'}",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)
            cv2.imshow("calib-capture  |  wrist camera", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                if not good:
                    print(f"  only {k} corners -- not captured")
                    continue
                try:
                    q = _joints_published()
                except Exception as e:  # noqa: BLE001
                    print(f"  {e}")
                    continue
                ok_t, tframe = top.read()
                cv2.imwrite(str(SESSION / f"wrist_{n:03d}.png"), frame)
                if ok_t:
                    cv2.imwrite(str(SESSION / f"top_{n:03d}.png"), tframe)
                (SESSION / f"pose_{n:03d}.json").write_text(json.dumps({
                    "joints_rad": q.tolist(), "joint_names": CHAIN, "charuco_corners": k}))
                print(f"  #{n}: {k} corners, joints(deg) {np.round(np.rad2deg(q),1).tolist()}")
                n += 1
    finally:
        wrist.release(); top.release()
        cv2.destroyAllWindows()
    print(f"\n{n} poses. {'Run ./robot calib-solve' if n >= 8 else 'Need at least 8.'}")
    return 0


def _pose_from_board(gray, det, board, K, D):
    """Board pose in the camera frame, or None.

    cv2.aruco.estimatePoseCharucoBoard is gone in OpenCV 4.11; the current route is
    matchImagePoints to lift the detected corners into board coordinates, then a
    plain solvePnP.
    """
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None or len(ci) < 6:
        return None
    obj, img = board.matchImagePoints(cc, ci)
    if obj is None or len(obj) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def _rodrigues_chain(p):
    """(rvec, tvec) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(p[:3], float))[0]
    T[:3, 3] = p[3:6]
    return T


def _bundle_adjust(obs, T_grip_cam0, T_base_board0, K, D, fit_offsets=True):
    """Refine the calibration against every corner, in pixels.

    calibrateHandEye consumes per-pose solvePnP results, and at 475 mm each of those
    is a noisy depth estimate; the closed form cannot tell a 48-corner pose from a
    12-corner one and weights them alike. This never forms those intermediates -- it
    optimises gripper->cam and base->board directly against the corner observations,
    where the measurement noise actually lives.

    With fit_offsets it also solves five joint-angle offsets. Error between lerobot's
    calibrated zero and the URDF's zero is otherwise unmodelled and gets absorbed into
    the camera pose, which is exactly the kind of systematic bias that makes a
    calibration look precise and be wrong.

    obs: list of (joints_rad, object_points Nx3, image_points Nx2)
    """
    from scipy.optimize import least_squares

    def pack(Tg, Tb, off):
        return np.concatenate([cv2.Rodrigues(Tg[:3, :3])[0].ravel(), Tg[:3, 3],
                               cv2.Rodrigues(Tb[:3, :3])[0].ravel(), Tb[:3, 3],
                               off])

    n_off = len(CHAIN) if fit_offsets else 0
    x0 = pack(T_grip_cam0, T_base_board0, np.zeros(n_off))

    def residuals(x):
        Tg = _rodrigues_chain(x[0:6])
        Tb = _rodrigues_chain(x[6:12])
        off = x[12:12 + n_off] if n_off else np.zeros(len(CHAIN))
        Tg_inv = np.linalg.inv(Tg)
        out = []
        for q, objp, imgp in obs:
            T_base_grip = fk_gripper(q + off)
            # board point -> base -> gripper -> camera
            M = Tg_inv @ np.linalg.inv(T_base_grip) @ Tb
            rvec = cv2.Rodrigues(M[:3, :3])[0]
            proj, _ = cv2.projectPoints(objp, rvec, M[:3, 3], K, D)
            out.append((proj.reshape(-1, 2) - imgp.reshape(-1, 2)).ravel())
        return np.concatenate(out)

    r0 = residuals(x0)
    res = least_squares(residuals, x0, method="lm", max_nfev=400)
    rms0 = np.sqrt((r0 ** 2).mean())
    rms1 = np.sqrt((res.fun ** 2).mean())
    off = res.x[12:12 + n_off] if n_off else np.zeros(len(CHAIN))
    return _rodrigues_chain(res.x[0:6]), _rodrigues_chain(res.x[6:12]), off, rms0, rms1


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
    obs = []          # (joints, object points, image points) for the bundle adjuster
    used = 0
    for p in poses:
        meta = json.loads(p.read_text())
        img = cv2.imread(str(p.with_name(p.name.replace("pose_", "wrist_")).with_suffix(".png")))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T_cam_board = _pose_from_board(gray, det, board, Kw, Dw)
        if T_cam_board is None:
            print(f"  {p.name}: board not resolvable, skipped")
            continue
        cc, ci, _, _ = det.detectBoard(gray)
        objp, imgp = board.matchImagePoints(cc, ci)
        if objp is not None and len(objp) >= a.min_corners_solve:
            obs.append((np.array(meta["joints_rad"]), objp.reshape(-1, 3), imgp.reshape(-1, 2)))
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
              "board_spread_mm": (P.std(0) * 1000).tolist(),
              "method": "calibrateHandEye (closed form)"}

    if not a.no_refine and len(obs) >= 6:
        print(f"\nrefining against {sum(len(o[1]) for o in obs)} corner observations "
              f"from {len(obs)} poses{'' if a.no_joint_offsets else ', plus 5 joint offsets'}")
        Tg, Tb, off, rms0, rms1 = _bundle_adjust(
            obs, T_grip_cam, T_base_board, Kw, Dw, fit_offsets=not a.no_joint_offsets)
        print(f"  reprojection rms {rms0:.3f} -> {rms1:.3f} px")
        moved = np.linalg.norm(Tg[:3, 3] - T_grip_cam[:3, 3]) * 1000
        print(f"  wrist camera moved {moved:.1f} mm from the closed-form estimate")
        print(f"  wrist cam in gripper: xyz mm {np.round(Tg[:3,3]*1000,2)}")
        print(f"  board in base:        xyz mm {np.round(Tb[:3,3]*1000,2)}")
        if not a.no_joint_offsets:
            print(f"  joint offsets (deg):  {np.round(np.rad2deg(off),2).tolist()}")
            print("    large values here are real: lerobot's calibrated zero is not the")
            print("    URDF's zero, and that bias was previously absorbed into the camera pose.")
        T_grip_cam, T_base_board = Tg, Tb
        result.update({"wrist_cam_in_gripper": Tg.tolist(),
                       "board_in_base": Tb.tolist(),
                       "joint_offsets_rad": off.tolist(),
                       "joint_names": CHAIN,
                       "reproj_rms_px": float(rms1),
                       "reproj_rms_px_before_refine": float(rms0),
                       "method": "bundle adjustment on corner reprojection"})
    else:
        print("\nskipping refinement" if a.no_refine else
              f"\ntoo few usable poses to refine ({len(obs)})")

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
    for name, fn in (("board", cmd_board), ("capture", cmd_capture),
                     ("solve", cmd_solve), ("teleop", cmd_teleop)):
        s = sub.add_parser(name)
        s.add_argument("--cols", type=int, default=7)
        s.add_argument("--rows", type=int, default=10)
        s.add_argument("--square", type=float, default=25.0, help="square size in mm, MEASURED after printing")
        s.add_argument("--marker", type=float, default=18.0, help="aruco marker size in mm")
        if name == "capture":
            s.add_argument("--min-corners", type=int, default=30,
                           help="reject a pose with fewer charuco corners; 30 of 54 keeps "
                                "the far, sparse views out that dominated the error at 12")
        if name == "solve":
            s.add_argument("--min-corners-solve", type=int, default=12,
                           help="minimum corners for a pose to enter the refinement")
            s.add_argument("--no-refine", action="store_true",
                           help="closed-form hand-eye only, no bundle adjustment")
            s.add_argument("--no-joint-offsets", action="store_true",
                           help="do not solve joint-angle offsets during refinement")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
