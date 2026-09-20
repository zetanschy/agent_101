#!/usr/bin/env python3
"""Teleoperate the Isaac push-T scene and record a LeRobot dataset.

    ./robot sim-record --episodes 20 --task "Push the T onto the goal"
    ./robot sim-record --episodes 20 --no-dr        # plain scene, no randomization

The sim is the follower: your real SO-101 LEADER drives it, exactly as
./robot sim-teleop does, and every frame is written to a LeRobot dataset with the
same schema our real recordings use, so the two can be merged and co-trained.

Between episodes the T block, the goal, and (unless --no-dr) lighting, camera pose,
friction, mass, actuator gains and the robot's shade are all re-randomized.

CONTROLS, typed in this terminal:
    n / RIGHT   end this episode and KEEP it
    r / LEFT    throw this episode away and redo it
    q / ESC     finalize the dataset and quit

Why this can be one process at all: importing lerobot after Kit starts fails,
because Kit prepends its pip_prebundle and that ships a botocore too old for
lerobot's import chain. Binding the good botocore first, at the very top of this
file, fixes it -- see the import block below. Earlier notes in this repo claiming
a sidecar-and-convert design is required are wrong.
"""

# ruff: noqa: E402
# FIRST. Before isaaclab, before anything that starts Kit. Kit prepends
# isaacsim/extscache/omni.kit.pip_archive-*/pip_prebundle to sys.path, and its
# botocore 1.34 shadows site-packages' 1.42. lerobot -> accelerate -> boto3 ->
# s3transfer wants DEFAULT_CHECKSUM_ALGORITHM, which 1.34 does not have, and the
# import dies with a message that has nothing to do with lerobot. Importing the
# good one here binds it in sys.modules before Kit gets a chance to shadow it.
import botocore.httpchecksum  # noqa: F401

import argparse
import contextlib
import json
import math
import os
import pathlib
import select
import sys
import termios
import time
import traceback
import tty

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--repo-id", default=None, help="dataset id (default $HF_USER/sim_push_t)")
parser.add_argument("--task", default="Push the T block onto the goal outline",
                    help="task string stored on every frame; must match the real dataset's")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--episode-time-s", type=float, default=30.0)
parser.add_argument("--fps", type=int, default=30, help="must match the real dataset (30)")
parser.add_argument("--no-dr", action="store_true", help="plain scene instead of the randomized one")
parser.add_argument("--source", choices=("auto", "direct", "file"), default="auto",
                    help="where leader angles come from; see scripts/sim/teleop.py")
parser.add_argument("--port", default=None)
parser.add_argument("--id", default=None)
parser.add_argument("--resume", action="store_true", help="append to an existing dataset")
parser.add_argument("--profile", action="store_true",
                    help="print where each frame's time goes -- use it when the rate warning fires")
parser.add_argument("--dry-run", action="store_true", help="run the loop but write nothing, to time the sim alone")
parser.add_argument("--writer-procs", type=int, default=0)
parser.add_argument("--writer-threads", type=int, default=4)
# "performance", not the env's usual "quality". Recording has a hard 33.3 ms budget
# and the operator needs the viewport open to aim -- a simulated task has nothing
# real to look at -- so the window is a third render on top of the two cameras.
# Measured with the arm moving and both cameras read: quality 33.3 ms (exactly at
# the limit, before the window), performance 25.9 ms. The visual difference does not
# reach the 640x480 camera images; the frame rate very much does.
parser.add_argument("--render-mode", default="performance",
                    choices=("performance", "balanced", "quality"))
# The operator has to see the scene -- a simulated task has nothing real to look at
# -- but Isaac's viewport is a THIRD full render on top of the two cameras and costs
# about 10 ms a frame, which is the difference between 26 and 36 Hz. The camera
# images are already in hand every frame, so show those instead: near-free, and it
# is literally what the policy will see. --viewport opens Isaac's window as well.
parser.add_argument("--no-preview", action="store_true", help="no camera preview window")
parser.add_argument("--viewport", action="store_true", help="also open Isaac's own window (slow)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = not args.viewport
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import sim_agent101  # noqa: F401  (registers the envs)
from sim_agent101.assets.so101 import JOINT_NAMES
from sim_agent101.kinematics import URDF_TO_LEROBOT, gripper_deg_to_pct, lerobot_to_urdf_deg

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOINTS_FILE = ROOT / "sim" / "outputs" / "calib" / "joints.json"

# The feature order our real datasets use. Also the URDF/articulation order, which
# was checked at runtime -- but the mapping below is by NAME, so it stays correct
# even if the articulation reorders.
URDF_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
LEROBOT_JOINTS = [f"{URDF_TO_LEROBOT[n]}.pos" for n in URDF_ORDER]

# All eight keys, not just is_depth_map. lerobot's merge path compares feature dicts
# after stripping only six video ENCODER keys; everything else must match, and the
# real datasets carry these. Omit them and aggregate_datasets refuses to merge with
# a message about incompatible features rather than about missing metadata.
VIDEO_INFO = {
    "is_depth_map": False,
    "video.height": 480, "video.width": 640, "video.channels": 3,
    "video.codec": "av1", "video.pix_fmt": "yuv420p", "video.fps": 30,
    "has_audio": False,
}


def camera_feature() -> dict:
    return {"dtype": "video", "shape": (480, 640, 3),
            "names": ["height", "width", "channels"], "info": dict(VIDEO_INFO)}


class Keys:
    """Single keypresses from this terminal, without blocking the control loop."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.saved = None

    def __enter__(self):
        if self.fd is not None:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self) -> str | None:
        if self.fd is None:
            return None
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":                      # arrow keys arrive as an escape sequence
            rest = ""
            while select.select([sys.stdin], [], [], 0)[0]:
                rest += sys.stdin.read(1)
            if rest.endswith("C"):
                return "n"                    # right
            if rest.endswith("D"):
                return "r"                    # left
            return "q"                        # bare ESC
        return ch.lower()


def open_leader(port, ident):
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    leader = SO101Leader(SO101LeaderConfig(port=port, id=ident, use_degrees=True))
    leader.connect()
    return leader


def port_is_openable(port) -> bool:
    try:
        import serial

        serial.Serial(port, 1000000, timeout=0.1).close()
        return True
    except Exception:  # noqa: BLE001
        return False


_last_good = None


def read_file(max_age_s: float = 2.0) -> dict:
    global _last_good

    if not JOINTS_FILE.exists():
        raise RuntimeError(f"no {JOINTS_FILE.relative_to(ROOT)} -- run './robot leader-publish --no-follower'")
    try:
        d = json.loads(JOINTS_FILE.read_text())
    except json.JSONDecodeError:
        if _last_good is None:
            raise
        d = _last_good
    _last_good = d
    if time.time() - d["t"] > max_age_s:
        raise RuntimeError("joints.json is stale -- is the publisher still running?")
    return dict(d["joints_deg"])


def main() -> int:
    task_id = "Agent101-So101-Push-T" if args.no_dr else "Agent101-So101-Push-T-DR"
    repo_id = args.repo_id or f"{os.environ.get('HF_USER', 'soarm101')}/sim_push_t"
    port = args.port or os.environ.get("TELEOP_PORT", "/dev/ttyACM0")
    ident = args.id or os.environ.get("TELEOP_ID", "zetans_leader")

    source = args.source
    leader = None
    if source == "auto":
        source = "direct" if port_is_openable(port) else "file"
    if source == "direct":
        leader = open_leader(port, ident)
        print(f"leader on {port} (id {ident}), read directly")
    else:
        read_file()
        print(f"reading {JOINTS_FILE.relative_to(ROOT)} (./robot leader-publish --no-follower)")

    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
    env_cfg.sim.render.rendering_mode = args.render_mode
    env = gym.make(task_id, cfg=env_cfg)
    step_dt = env.unwrapped.step_dt
    if abs(1.0 / step_dt - args.fps) > 0.5:
        raise SystemExit(f"env runs at {1/step_dt:.2f} Hz but the dataset wants {args.fps}; "
                         f"fix decimation/sim.dt rather than recording dilated data")
    env.reset()
    robot = env.unwrapped.scene["robot"]
    cams = {"front": env.unwrapped.scene["camera_front"], "grip": env.unwrapped.scene["camera_grip"]}
    term = env.unwrapped.action_manager.get_term("joint_positions")
    order = list(term._joint_names)
    jidx = {n: i for i, n in enumerate(robot.data.joint_names)}

    # Widen the model's joint limits to cover what the real arm can actually reach.
    # The USD stops Elbow at +90 deg; the real leader sits at +96.5 in a normal
    # push-T pose. Left alone, the sim clamps and the recorded ACTION no longer
    # matches the resulting STATE -- a dataset that quietly lies about the
    # action-to-state map on every frame near the limit. The commands come from a
    # real arm, so by construction they are reachable and widening is safe.
    REAL_RANGE_DEG = {"Rotation": (-110, 110), "Pitch": (-108, 108), "Elbow": (-100, 100),
                      "Wrist_Pitch": (-95, 106), "Wrist_Roll": (-168, 168), "Jaw": (-10, 100)}
    def widen_limits(report: bool = False) -> None:
        lim = robot.data.joint_pos_limits.clone()
        changed = []
        for name, (lo, hi) in REAL_RANGE_DEG.items():
            i = jidx[name]
            was = (math.degrees(float(lim[0, i, 0])), math.degrees(float(lim[0, i, 1])))
            lim[0, i, 0] = math.radians(min(lo, was[0]))
            lim[0, i, 1] = math.radians(max(hi, was[1]))
            now = (math.degrees(float(lim[0, i, 0])), math.degrees(float(lim[0, i, 1])))
            if now != was:
                changed.append(f"{name} {was[0]:+.0f}/{was[1]:+.0f} -> {now[0]:+.0f}/{now[1]:+.0f}")
        robot.write_joint_position_limit_to_sim(lim)
        if report and changed:
            print("widened joint limits to the real arm's range: " + "; ".join(changed))

    # After EVERY reset, not once at startup: a reset restores the limits from the
    # asset's spawn configuration, so widening only at the start silently reverts on
    # episode 1. Caught by loading a recorded dataset back and seeing elbow_flex
    # action reach +96.5 while the state stopped dead at +89.6 -- the exact
    # action-does-not-match-state corruption the widening exists to prevent.
    widen_limits(report=True)

    def to_lerobot(vec_rad) -> np.ndarray:
        """Articulation-order radians -> the real dataset's units.

        Five arm joints in DEGREES, gripper in PERCENT 0-100. lerobot hardcodes the
        gripper to RANGE_0_100 regardless of use_degrees, so degrees here would be a
        units bug that looks entirely plausible in the statistics.
        """
        out = []
        for name in URDF_ORDER:
            d = math.degrees(float(vec_rad[jidx[name]]))
            out.append(gripper_deg_to_pct(d) if name == "Jaw" else d)
        return np.asarray(out, dtype=np.float32)

    root = pathlib.Path(os.environ.get("HF_LEROBOT_HOME",
                                       pathlib.Path.home() / ".cache/huggingface/lerobot")) / repo_id
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": LEROBOT_JOINTS},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": LEROBOT_JOINTS},
        "observation.images.front": camera_feature(),
        "observation.images.grip": camera_feature(),
    }
    if args.resume and root.exists():
        ds = LeRobotDataset(repo_id, root=root)
        ds.start_image_writer(args.writer_procs, args.writer_threads)
        print(f"appending to {root} ({ds.meta.total_episodes} episodes already)")
    else:
        if root.exists():
            raise SystemExit(f"{root} exists; pass --resume to append or delete it")
        ds = LeRobotDataset.create(
            repo_id=repo_id, root=root, fps=args.fps, features=features,
            robot_type="so_follower",       # what the real datasets record, not the .env alias
            use_videos=True, image_writer_processes=args.writer_procs,
            image_writer_threads=args.writer_threads,
        )
        print(f"created {root}")

    # An explicit units marker. lerobot's merge guesses degrees-vs-normalized by
    # asking whether any action exceeds +/-100.5 -- and push-T is a small planar task
    # that may never get there, so a perfectly correct degrees dataset can be
    # misclassified and refused. Recording the answer removes the guess.
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "units.json").write_text(json.dumps(
        {"joints": "degrees", "gripper": "percent_0_100",
         "note": "written by scripts/sim/record.py; lerobot's joint_units() heuristic "
                 "misreads small-motion tasks, prefer this"}, indent=2) + "\n")

    print(f"\n{task_id}   {1/step_dt:.1f} Hz   {args.episodes} episodes of up to "
          f"{args.episode_time_s:.0f}s\n  n/right = keep   r/left = redo   q/esc = finish\n")

    preview = None
    if not args.no_preview:
        import cv2

        # Half size, every other frame. At full 640x480 x2 every frame this cost
        # 8.7 ms of a 33.3 ms budget, which is most of the headroom; like this it is
        # about 1 ms and still perfectly aimable at ~10 Hz.
        def preview(imgs, ep, frame, sat):
            if frame % 3:
                return
            side = np.hstack([imgs["front"][::2, ::2, ::-1], imgs["grip"][::2, ::2, ::-1]])
            side = np.ascontiguousarray(side)
            bad = sum(sat.values())
            cv2.rectangle(side, (0, 0), (side.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(side, f"ep {ep}/{args.episodes}  f{frame}  n=keep r=redo q=quit"
                              f"{'  SATURATED' if bad else ''}",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("sim-record  (left: overhead, right: wrist)", side)
            cv2.waitKey(1)

    saved = 0
    stop = False
    with Keys() as keys:
        while saved < args.episodes and not stop:
            env.reset()
            widen_limits()
            limits = robot.data.joint_pos_limits[0]
            frames, sat = 0, {n: 0 for n in order}
            prof = {"grab": 0.0, "leader": 0.0, "step": 0.0, "add": 0.0}
            t_ep = time.time()
            periods = []
            verdict = None
            while verdict is None:
                tick = time.time()
                # State and images at time t, BEFORE the action is applied -- the
                # order lerobot's own recorder uses. .cpu().numpy() is load-bearing:
                # output["rgb"] is a view onto a buffer overwritten every update.
                _t0 = time.perf_counter()
                state = to_lerobot(robot.data.joint_pos[0])
                imgs = {k: c.data.output["rgb"][0, ..., :3].cpu().numpy().copy() for k, c in cams.items()}
                _t1 = time.perf_counter()
                if preview is not None:
                    preview(imgs, saved + 1, frames, sat)
                try:
                    deg = lerobot_to_urdf_deg(
                        {k[:-4]: float(v) for k, v in leader.get_action().items() if k.endswith(".pos")}
                    ) if source == "direct" else read_file()
                except RuntimeError as exc:
                    print(f"\n  leader read failed: {exc}"); verdict = "abort"; break
                actions = torch.zeros((1, len(order)), device=env.unwrapped.device)
                for i, name in enumerate(order):
                    rad = math.radians(deg[name])
                    lo, hi = limits[jidx[name]].tolist()
                    if not (lo <= rad <= hi):
                        sat[name] += 1
                    actions[0, i] = max(lo, min(hi, rad))
                action = to_lerobot(actions[0])
                _t2 = time.perf_counter()
                env.step(actions)
                _t3 = time.perf_counter()
                if not args.dry_run:
                    ds.add_frame({"action": action, "observation.state": state,
                                  "observation.images.front": imgs["front"],
                                  "observation.images.grip": imgs["grip"],
                                  "task": args.task})
                _t4 = time.perf_counter()
                prof["grab"] += _t1 - _t0; prof["leader"] += _t2 - _t1
                prof["step"] += _t3 - _t2; prof["add"] += _t4 - _t3
                frames += 1
                k = keys.get()
                if k in ("n",):
                    verdict = "keep"
                elif k in ("r",):
                    verdict = "redo"
                elif k in ("q", "\x03"):
                    verdict = "quit"
                elif time.time() - t_ep >= args.episode_time_s:
                    verdict = "keep"
                periods.append(time.time() - tick)
                time.sleep(max(step_dt - (time.time() - tick), 0.0))

            hz = 1.0 / max(np.median(periods), 1e-6) if periods else 0.0
            if args.profile and frames:
                print("    per frame ms: " + "  ".join(f"{k} {v/frames*1000:5.1f}" for k, v in prof.items()))
            bad = [f"{n}x{c}" for n, c in sat.items() if c]
            note = f"  saturated: {' '.join(bad)}" if bad else ""
            if args.dry_run:
                ds.clear_episode_buffer()
                print(f"  DRY RUN: {frames} frames, {hz:.1f} Hz (nothing written)")
                saved += 1
                continue
            if verdict in ("redo", "abort"):
                ds.clear_episode_buffer()
                print(f"  episode discarded ({frames} frames, {hz:.1f} Hz){note}")
                if verdict == "abort":
                    stop = True
                continue
            ds.save_episode()
            # finalize THEN reopen, every episode. finalize-and-never-reopen keeps
            # only the last episode while info.json still claims them all; never
            # finalizing at all leaves the parquet without a footer and the whole
            # dataset unreadable. This way a crash costs the episode in progress
            # and nothing else.
            ds.finalize()
            ds = LeRobotDataset(repo_id, root=root)
            ds.start_image_writer(args.writer_procs, args.writer_threads)
            saved += 1
            flag = "" if hz >= args.fps - 1.0 else "   <-- BELOW RATE, data is time-dilated"
            print(f"  episode {saved}/{args.episodes} saved: {frames} frames, {hz:.1f} Hz{flag}{note}")
            if verdict == "quit":
                stop = True

    if preview is not None:
        import cv2

        cv2.destroyAllWindows()
    print(f"\n{saved} episodes -> {root}")
    if leader is not None:
        with contextlib.suppress(Exception):
            leader.disconnect()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        code = 0
    except SystemExit as exc:
        # os._exit in the finally runs before Python would print this, so the
        # message has to be printed here or the script exits 1 in silence.
        if exc.code not in (0, None):
            print(f"\n{exc}", file=sys.stderr)
        code = 0 if exc.code in (0, None) else 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
