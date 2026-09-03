#!/usr/bin/env python3
"""Drive the Isaac push-T scene from the real SO-101 leader arm.

    ./robot sim-teleop                    # leader -> sim, GUI open

One command: ./robot sim-teleop starts the leader publisher in the container,
runs this against it, and stops it again. If you already have one running (from
./robot leader-publish, or ./robot calib-teleop during a calibration) it is
consumed as-is rather than started twice -- the serial bus only opens once.

This is the workshop's lerobot_agent.py idea (thirdparty/sim2real_so101), fitted
to this repo's split: there, lerobot and Isaac share one process and one Python.
Here they may not, so the leader angles can arrive two ways and the rest of the
loop does not care which:

  direct  the leader's serial port, opened from this process. Needs the invoking
          user in the `dialout` group -- `sudo usermod -aG dialout $USER`, then
          log out and back in. Lowest latency, and what the workshop does.

  file    sim/outputs/calib/joints.json, written ~30 Hz by `./robot leader-publish`
          inside the container. Needs no group change, because Docker owns the
          serial device. Costs a file round-trip, and mirrors the FOLLOWER's
          measured angles rather than the leader's command -- which is arguably
          the better sim2real check, since it includes the real servos' lag.

Units. lerobot is asked for DEGREES and this converts straight to radians -- no
normalised -100..100 range anywhere. The workshop maps -100..100 onto per-joint
USD ranges instead, which is the same arithmetic done twice and only matches if
the ranges agree; asking for degrees skips the question. See kinematics.py, whose
FK on these same values was checked against Isaac to 0.000 mm.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

# Do NOT import lerobot here, or anywhere, unless the direct source is really
# going to be used. Isaac Sim 4.5's Python and lerobot 0.4.3 do not co-exist in
# this env, and they fail differently depending on the order:
#
#   lerobot first   binds pydantic 2.13 from ~/.local/lib, and then isaaclab_tasks
#                   itself fails to import -- FieldInfo.from_annotated_attribute()
#                   got an unexpected keyword argument '_source'
#   isaaclab first  Kit prepends its pip_prebundle, whose botocore is too old:
#                   cannot import name 'DEFAULT_CHECKSUM_ALGORITHM'
#
# So the single-process design the workshop uses is not available here as
# installed, and the file source is not a workaround for a missing group -- it is
# the path that works. Probing the port with pyserial below costs no imports.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Agent101-So101-Push-T")
parser.add_argument("--source", choices=("auto", "direct", "file"), default="auto",
                    help="where leader angles come from (default auto: direct, else file)")
parser.add_argument("--fps", type=int, default=30, help="control rate")
parser.add_argument("--port", default=None, help="leader serial port (default TELEOP_PORT)")
parser.add_argument("--id", default=None, help="leader calibration id (default TELEOP_ID)")
# --headless comes from AppLauncher, and its default (window open) is what teleop
# wants; pass --headless only for a smoke test.
parser.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = until closed)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim_agent101  # noqa: E402,F401  (registers the envs)
from sim_agent101.kinematics import LEROBOT_TO_URDF, lerobot_to_urdf_deg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
JOINTS_FILE = ROOT / "sim" / "outputs" / "calib" / "joints.json"
URDF_TO_LEROBOT = {v: k for k, v in LEROBOT_TO_URDF.items()}


def port_is_openable(port) -> bool:
    """Can this process open the leader's serial port at all? No lerobot needed."""
    try:
        import serial

        serial.Serial(port, 1000000, timeout=0.1).close()
        return True
    except Exception:  # noqa: BLE001
        return False


def open_leader(port, ident):
    """Connect to the leader over serial. See the import note at the top."""
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    leader = SO101Leader(SO101LeaderConfig(port=port, id=ident, use_degrees=True))
    leader.connect()
    return leader


_last_good = None


def read_file(max_age_s: float = 2.0):
    """Latest angles from the container publisher, as {lerobot name: degrees}.

    The publisher renames into place so reads are atomic, but an older publisher
    (or a different filesystem) may still truncate-and-write. Falling back to the
    last good sample for one tick is better than dropping teleop on a torn read.
    """
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
    age = time.time() - d["t"]
    if age > max_age_s:
        raise RuntimeError(f"{JOINTS_FILE.name} is {age:.0f}s stale -- is the publisher running?")
    # already URDF-named and already true degrees, gripper included
    return dict(d["joints_deg"])


def main() -> int:
    import math

    port = args.port or os.environ.get("TELEOP_PORT", "/dev/ttyACM0")
    ident = args.id or os.environ.get("TELEOP_ID", "zetans_leader")

    leader, source = None, args.source
    if source == "auto":
        source = "direct" if port_is_openable(port) else "file"
        if source == "file":
            print(f"\n{port} is not openable by this user -- using the file source.")
            print("  (for the direct path: sudo usermod -aG dialout $USER, then log back in")
            print("   -- though see the import note at the top of this file first)")
    if source == "direct":
        leader = open_leader(port, ident)
        print(f"\nleader on {port} (id {ident}), read directly")
    else:
        read_file()  # fail now, not mid-loop, if the publisher is not running
        print(f"\nreading {JOINTS_FILE.relative_to(ROOT)} (from ./robot leader-publish)")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    # The action vector is ordered by the ARTICULATION, not by the joint_names in
    # the config: JointPositionActionCfg leaves preserve_order False, so asking the
    # term what it actually resolved is the only safe way to fill the tensor.
    term = env.unwrapped.action_manager.get_term("joint_positions")
    order = list(term._joint_names)
    print(f"action order: {order}")
    limits = env.unwrapped.scene["robot"].data.soft_joint_pos_limits[0]
    idx = {n: i for i, n in enumerate(env.unwrapped.scene["robot"].data.joint_names)}

    actions = torch.zeros((1, len(order)), device=env.unwrapped.device)
    warned, period, n = set(), 1.0 / args.fps, 0
    started = time.time()
    print(f"\nteleop running at {args.fps} Hz. Move the leader. Ctrl-C or close the window to stop.\n")
    try:
        while app.is_running():
            tick = time.time()
            if args.seconds and tick - started > args.seconds:
                break
            if source == "direct":
                raw = {k[:-4]: float(v) for k, v in leader.get_action().items() if k.endswith(".pos")}
                deg = lerobot_to_urdf_deg(raw)   # the gripper is a percent, not degrees
            else:
                deg = read_file()
            for i, urdf_name in enumerate(order):
                rad = math.radians(deg[urdf_name])
                lo, hi = limits[idx[urdf_name]].tolist()
                if not (lo <= rad <= hi) and urdf_name not in warned:
                    warned.add(urdf_name)
                    print(f"  ! {urdf_name} commanded {math.degrees(rad):+.1f} deg, outside the "
                          f"model's [{math.degrees(lo):+.1f}, {math.degrees(hi):+.1f}] -- clamping")
                actions[0, i] = max(lo, min(hi, rad))
            env.step(actions)
            n += 1
            time.sleep(max(period - (time.time() - tick), 0.0))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        # Did the sim actually FOLLOW? A loop that runs cleanly while the arm sits
        # still looks identical to a working one, so report the last commanded
        # angle against what the articulation reached.
        if n:
            got = env.unwrapped.scene["robot"].data.joint_pos[0]
            print(f"\n{n} steps at {args.fps} Hz. last commanded vs reached (deg):")
            for i, urdf_name in enumerate(order):
                c = math.degrees(float(actions[0, i]))
                g = math.degrees(float(got[idx[urdf_name]]))
                print(f"  {urdf_name:12} cmd {c:+8.2f}   got {g:+8.2f}   err {g - c:+6.2f}")
        if leader is not None:
            try:
                leader.disconnect()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit hangs in app.close(); force the exit like sim_play does.
        os._exit(code)
