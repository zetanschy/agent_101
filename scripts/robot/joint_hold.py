#!/usr/bin/env python3
"""Hold ONE joint at a known reading while you bolt its link back on.

    ./robot joint-hold --joint wrist_flex --degrees 0
    ./robot joint-hold --joint wrist_flex --degrees 0 --free   # shaft loose instead

For reassembling a joint whose horn slipped. The point is not to get the link on at
exactly the right angle -- you cannot. The coupling between a servo and its link is
discrete (a splined horn indexes in whole teeth, a bolted one in whole holes), so
whatever you do you land within half a step of the truth, and no amount of unbolting and
trying again changes that. Trial and error here is not perfectionism, it is a loop with
no exit.

The way out is to do it ONCE against a reference and finish in software:

    1. ./robot joint-hold --joint wrist_flex --degrees 0
       The servo goes to 0.00 and holds. That is now a known angle you can see.
    2. Bolt the link on as close as you can get it to its true zero. Nearest tooth is
       fine. Tighten.
    3. Ctrl-C, then ./robot joint-check --sweep wrist_flex
       The midpoint is what is left over -- half a tooth at worst.
    4. ./robot joint-offset --joint wrist_flex --degrees <that> --apply
       Exact, and the frame every checkpoint was trained in is preserved.

TORQUE IS LEFT ON WHEN THIS EXITS, deliberately: you are mid-assembly and the arm going
limp with a wrist half-bolted is how the next thing breaks. The other joints are held at
wherever they already were; only the named one is commanded anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MAX_RES = 4095


def env(key: str, default: str) -> str:
    return os.environ.get(key, default) or default


def limits(motor: str) -> tuple[float, float] | None:
    """That joint's own calibrated limits, in degrees, from the versioned file."""
    path = (ROOT / "calibration" / "robots" / env("ROBOT_TYPE", "so101_follower")
            / f"{env('ROBOT_ID', 'zetans_follower')}.json")
    if not path.exists():
        return None
    cal = json.loads(path.read_text()).get(motor)
    if not cal:
        return None
    span = (int(cal["range_max"]) - int(cal["range_min"])) / 2 * 360 / MAX_RES
    return -span, span


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--joint", required=True)
    p.add_argument("--degrees", type=float, default=0.0,
                   help="the reading to hold (default 0, the middle of the joint's range)")
    p.add_argument("--free", action="store_true",
                   help="release this joint instead of holding it, so the shaft turns freely")
    p.add_argument("--seconds", type=float, default=2.0, help="ramp time to the target")
    args = p.parse_args(argv)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(
        SO101FollowerConfig(
            port=env("ROBOT_PORT", "/dev/ttyACM1"),
            id=env("ROBOT_ID", "zetans_follower"),
            cameras={},
            use_degrees=True,
            # Mid-assembly, an arm that goes limp on exit is how the next thing breaks.
            disable_torque_on_disconnect=False,
        )
    )
    robot.connect()
    key = f"{args.joint}.pos"
    if key not in robot.action_features:
        print(f"{args.joint} is not a joint: "
              f"{', '.join(k[:-4] for k in robot.action_features)}", flush=True)
        robot.disconnect()
        return 2

    try:
        if args.free:
            robot.bus.disable_torque(motors=[args.joint], num_retry=3)
            print(f"{args.joint} released -- the shaft turns freely. Ctrl-C to re-hold it.",
                  flush=True)
            while True:
                print(f"\r  reading {float(robot.get_observation()[key]):8.2f}", end="", flush=True)
                time.sleep(0.1)

        bounds = limits(args.joint)
        target = args.degrees
        if bounds and not (bounds[0] <= target <= bounds[1]):
            print(f"{target:.2f} is outside {args.joint}'s calibrated range "
                  f"[{bounds[0]:.1f}, {bounds[1]:.1f}]", flush=True)
            robot.disconnect()
            return 2

        start = {k: float(v) for k, v in robot.get_observation().items()
                 if k in robot.action_features}
        begin = start[key]
        steps = max(int(args.seconds * 30), 1)
        print(f"\n{args.joint}: {begin:.2f} -> {target:.2f} over {args.seconds:g}s, "
              "everything else holds where it is.", flush=True)
        for step in range(steps + 1):
            t = step / steps
            command = dict(start)
            command[key] = begin * (1 - t) + target * t
            robot.send_action(command)
            time.sleep(1 / 30)

        print(f"holding {args.joint} at {target:.2f}. Bolt the link on at its true angle,\n"
              "nearest tooth is fine -- the leftover is measured afterwards. Ctrl-C when done.\n",
              flush=True)
        held = dict(start)
        held[key] = target
        while True:
            robot.send_action(held)
            print(f"\r  reading {float(robot.get_observation()[key]):8.2f}   target {target:8.2f}",
                  end="", flush=True)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print(flush=True)
    finally:
        if args.free:
            try:
                robot.bus.enable_torque(motors=[args.joint], num_retry=3)
            except Exception as exc:  # noqa: BLE001
                print(f"could not re-hold {args.joint}: {exc}", flush=True)
        robot.disconnect()

    print(f"\ntorque left ON. Next:\n"
          f"  ./robot joint-check --sweep {args.joint}      measure what is left over\n"
          f"  ./robot joint-offset --joint {args.joint} --degrees <midpoint> --apply", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
