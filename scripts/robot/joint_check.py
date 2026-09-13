#!/usr/bin/env python3
"""Read both arms and say where they disagree. Moves nothing, commands nothing.

    ./robot joint-check                 # one reading of follower, leader and the delta
    ./robot joint-check --watch         # keep printing while you move them by hand
    ./robot joint-check --follower-only # when the leader is not plugged in

WHAT THIS IS FOR. A collision during teleop does not decalibrate an encoder -- a
magnetic absolute encoder does not forget. What a collision does is move the METAL: the
horn slips on the spline, and from then on the link sits a few degrees away from where
the servo thinks it is. The calibration file is untouched (this repo version-controls
it, so `git status calibration/` proves that in a second), and the arm is simply no
longer where its numbers say.

WHY THAT MATTERS MORE THAN IT SOUNDS. Every checkpoint here was trained on ABSOLUTE
joint targets, so what a policy relies on is one mapping:

    physical pose  ->  reported degrees

A slipped horn breaks that mapping for one joint, and the policy then drives that joint
to the wrong physical place while both the log and the clamp look perfectly normal.
Restoring the mapping is the whole job -- and there are two ways to do it, one of which
keeps every trained model and one of which throws them away. See the README section
this prints a pointer to.

READ-ONLY, DELIBERATELY. It connects (which lerobot uses to push the stored calibration
INTO the servos, and which makes a servo hold where it already is -- it does not jump)
and then only reads. No action is ever sent, so this is safe to run on an arm you do
not trust yet.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def env(key: str, default: str) -> str:
    return os.environ.get(key, default) or default


def home_pose() -> dict[str, float]:
    """The recorded home pose, in degrees: a PHYSICAL reference that predates the crash."""
    path = pathlib.Path(env("HOME_POSE_FILE", str(ROOT / "config" / "home_pose.json")))
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.loads(path.read_text()).items()}


def calibration(kind: str, robot_type: str, robot_id: str) -> dict:
    """The versioned calibration, so the printout can show each joint's own range."""
    path = ROOT / "calibration" / kind / robot_type / f"{robot_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def sweep(follower, teleop, args) -> int:
    """Measure one joint's slip against its own mechanical stops.

    THE MIDPOINT IS THE MEASUREMENT. lerobot stores each joint's range in counts and
    reports degrees as (position - mid) * 360 / 4095 with mid halfway between them, so
    the middle of the PHYSICAL travel reads 0.00 by construction -- that is what the
    calibration recorded. A horn that slipped by D moves every reading by D, so sweeping
    stop to stop and halving gives D directly, with no second arm to pose by eye and no
    guess about which end you are nearer.

    Only the joint under test is freed. The rest keep holding, because an SO-101 with
    all six released falls onto the table.
    """
    motor = args.sweep
    joints = [k[:-4] for k in follower.action_features if k.endswith(".pos")]
    if motor not in joints:
        print(f"{motor} is not a joint: {', '.join(joints)}", flush=True)
        return 2

    span_deg = None
    cal = calibration("robots", env("ROBOT_TYPE", "so101_follower"),
                      env("ROBOT_ID", "zetans_follower")).get(motor)
    if cal:
        span_deg = (int(cal["range_max"]) - int(cal["range_min"])) / 2 * 360 / 4095

    print(f"\nfreeing {motor} only -- the other joints keep holding.", flush=True)
    try:
        follower.bus.disable_torque(motors=[motor], num_retry=3)
    except Exception as exc:  # noqa: BLE001
        print(f"could not free {motor}: {exc}", flush=True)
        return 2

    print("Move it BY HAND to one mechanical stop, then to the other, slowly.\n"
          "Ctrl-C when you have touched both.\n", flush=True)
    lo = hi = None
    try:
        while True:
            value = float(follower.get_observation()[f"{motor}.pos"])
            lo = value if lo is None else min(lo, value)
            hi = value if hi is None else max(hi, value)
            mid = (lo + hi) / 2
            print(f"\r  now {value:8.2f}   seen [{lo:8.2f}, {hi:8.2f}]   midpoint {mid:+7.2f}",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print(flush=True)
        try:
            follower.bus.enable_torque(motors=[motor], num_retry=3)
        except Exception as exc:  # noqa: BLE001
            print(f"could not re-hold {motor}: {exc}", flush=True)

    if lo is None:
        return 1
    mid = (lo + hi) / 2
    travelled = hi - lo
    print(f"\n  swept  [{lo:.2f}, {hi:.2f}]  = {travelled:.1f} deg of travel")
    if span_deg:
        print(f"  stored +/-{span_deg:.2f} deg = {2 * span_deg:.1f} deg of travel")
        if travelled < 2 * span_deg - 10:
            print(f"  ...you did not reach both stops ({2 * span_deg - travelled:.0f} deg short);"
                  " the midpoint below is not trustworthy yet.")
    print(f"\n  MIDPOINT {mid:+.2f} deg  -- it should read 0.00\n")
    if abs(mid) > 1.0:
        print(f"  So {motor} reads {mid:+.2f} deg high at every pose. To restore the frame:\n"
              f"      ./robot joint-offset --joint {motor} --degrees {mid:.2f}\n"
              f"      ./robot joint-offset --joint {motor} --degrees {mid:.2f} --apply\n"
              "  Sweep again afterwards: the midpoint should come back ~0.", flush=True)
    else:
        print(f"  Within a degree of zero: {motor}'s zero is where it always was, and the\n"
              "  problem is somewhere else -- a bent link, or the other arm.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--watch", action="store_true", help="keep printing until Ctrl-C")
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--follower-only", action="store_true")
    p.add_argument("--tolerance", type=float, default=3.0,
                   help="degrees of leader/follower disagreement to flag (default 3)")
    p.add_argument("--sweep", metavar="JOINT", default=None,
                   help="measure ONE joint's slip against its own hard stops: frees that "
                        "joint (and only that one) so you can move it by hand, tracks the "
                        "extremes, and reports the midpoint -- which should read 0")
    args = p.parse_args(argv)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(
        SO101FollowerConfig(
            port=env("ROBOT_PORT", "/dev/ttyACM1"),
            id=env("ROBOT_ID", "zetans_follower"),
            cameras={},  # no cameras: this is about joints, and it keeps the USB quiet
            use_degrees=True,
        )
    )
    follower.connect()
    teleop = None
    # A sweep measures one joint against its own stops, so the leader is not part of it
    # -- and requiring it to be plugged in would only add a way for the measurement to
    # fail before it starts.
    if not args.follower_only and not args.sweep:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        teleop = SO101Leader(
            SO101LeaderConfig(
                port=env("TELEOP_PORT", "/dev/ttyACM0"),
                id=env("TELEOP_ID", "zetans_leader"),
                use_degrees=True,
            )
        )
        teleop.connect()
        # Torque off so you can pose it by hand: this is a measurement, not a handover.
        try:
            teleop.bus.disable_torque(num_retry=3)
        except Exception as exc:  # noqa: BLE001 - a stiff leader is annoying, not fatal
            print(f"could not release the leader ({exc}); pose it anyway if it gives", flush=True)

    if args.sweep:
        return sweep(follower, teleop, args)

    home = home_pose()
    follower_cal = calibration("robots", env("ROBOT_TYPE", "so101_follower"),
                               env("ROBOT_ID", "zetans_follower"))

    def line(name: str, f: float, l: float | None) -> str:
        home_delta = f" home{f - home[name]:+7.1f}" if name in home else " " * 12
        span = ""
        if name in follower_cal:
            c = follower_cal[name]
            span = f"  counts[{c.get('range_min')},{c.get('range_max')}] homing={c.get('homing_offset')}"
        if l is None:
            return f"  {name:16s} {f:8.2f}{home_delta}{span}"
        delta = f - l
        flag = "  <-- OFFSET" if abs(delta) > args.tolerance else ""
        return f"  {name:16s} follower {f:8.2f}   leader {l:8.2f}   delta {delta:+7.2f}{flag}"

    try:
        while True:
            raw = follower.get_observation()
            joints = [k[:-4] for k in follower.action_features if k.endswith(".pos")]
            lead = teleop.get_action() if teleop is not None else {}
            print(f"\n--- {time.strftime('%H:%M:%S')} ---", flush=True)
            for motor in joints:
                f = float(raw[f"{motor}.pos"])
                l = float(lead[f"{motor}.pos"]) if f"{motor}.pos" in lead else None
                print(line(motor, f, l), flush=True)
            if teleop is not None:
                print("\n  `delta` only means something when BOTH arms are in the same physical\n"
                      "  pose -- hold them against the same reference (both folded to their\n"
                      "  mechanical stop is the easiest) and read the joint that disagrees.",
                      flush=True)
            elif home:
                print("\n  `home` is how far each joint is from the recorded home pose. Put the\n"
                      "  arm physically at home and the joint that does not read ~0 is the one\n"
                      "  that moved.", flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()
        if teleop is not None:
            teleop.disconnect()

    print("\nA slipped joint is METAL, not calibration: see 'A joint slipped' in README.md\n"
          "before changing anything -- one of the two fixes keeps your trained models and\n"
          "the other silently invalidates them.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
