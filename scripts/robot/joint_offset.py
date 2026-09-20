#!/usr/bin/env python3
"""Re-align ONE joint's zero after its horn slipped, without recalibrating the arm.

    ./robot joint-offset --joint wrist_flex --degrees 12.5        # dry run, shows the patch
    ./robot joint-offset --joint wrist_flex --degrees 12.5 --apply

`--degrees` is what the joint READS MINUS what it should read, at one physical pose:
the number `./robot joint-check` prints as `delta` when both arms are held the same.
Positive means the follower reads high.

WHY THIS IS NOT A RECALIBRATION. On feetech motors, by their own driver's comment:

    Present_Position = Actual_Position - Homing_Offset          feetech.py:281
    degrees          = (Present_Position - mid) * 360 / 4095    mid = (min+max)/2

A slipped horn changes Actual_Position for a given physical pose, and nothing else.
Adding the same shift to Homing_Offset cancels it exactly, and the arm reports the
degrees it always did for the pose it is really in -- which is the only thing every
checkpoint trained on absolute joint targets depends on.

AND THE RANGE STAYS CORRECT, which is the part that is easy to get wrong. range_min and
range_max were recorded from Present_Position, i.e. AFTER the offset, so shifting the
offset shifts them with it: the physical extremes keep mapping to the same stored
counts. The safety clamp evals/rig.py computes from that range is therefore unchanged,
in degrees and in physical space. (An earlier version of the README claimed otherwise.
It was wrong.)

WHAT THIS DOES NOT FIX. The arm's geometry still disagrees with the URDF by the slip,
so anything that reasons about links rather than joint readings -- the sim2real work,
camera extrinsics solved against the arm -- stays off until the horn is reseated.
Reseating is the better repair for exactly that reason; this is the one that gets you
collecting data again tonight.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# STS3215: 4096 positions, and lerobot uses resolution - 1 everywhere it converts.
MAX_RES = 4095


def env(key: str, default: str) -> str:
    return os.environ.get(key, default) or default


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--joint", required=True, help="e.g. wrist_flex")
    p.add_argument("--degrees", type=float, required=True,
                   help="reading MINUS truth, in degrees (joint-check's `delta`)")
    p.add_argument("--arm", choices=("follower", "leader"), default="follower")
    p.add_argument("--apply", action="store_true", help="write it; otherwise dry run")
    args = p.parse_args(argv)

    if args.arm == "follower":
        path = (ROOT / "calibration" / "robots" / env("ROBOT_TYPE", "so101_follower")
                / f"{env('ROBOT_ID', 'zetans_follower')}.json")
    else:
        path = (ROOT / "calibration" / "teleoperators" / env("TELEOP_TYPE", "so101_leader")
                / f"{env('TELEOP_ID', 'zetans_leader')}.json")
    if not path.exists():
        raise SystemExit(f"no calibration at {path}")

    data = json.loads(path.read_text())
    if args.joint not in data:
        raise SystemExit(f"{args.joint} is not in {path.name}: {', '.join(data)}")

    counts = round(args.degrees * MAX_RES / 360)
    entry = data[args.joint]
    before = int(entry["homing_offset"])
    after = before + counts  # Present = Actual - Homing, so +counts lowers the reading

    span = int(entry["range_max"]) - int(entry["range_min"])
    print(f"file    : {path.relative_to(ROOT)}")
    print(f"joint   : {args.joint}  (id {entry.get('id')})")
    print(f"delta   : {args.degrees:+.2f} deg  ->  {counts:+d} counts  (at {MAX_RES / 360:.3f} counts/deg)")
    print(f"homing  : {before}  ->  {after}")
    print(f"range   : [{entry['range_min']}, {entry['range_max']}] unchanged "
          f"(recorded after the offset, so it shifts with it)")
    print(f"clamp   : +/-{span / 2 * 360 / MAX_RES:.2f} deg, unchanged -- it comes from the span")

    if not args.apply:
        print("\ndry run. Re-run with --apply to write it.")
        return 0

    entry["homing_offset"] = after
    path.write_text(json.dumps(data, indent=4) + "\n")
    print(f"\nwritten. Now:\n"
          f"  1. ./robot joint-check      -- connecting pushes the new offset into the servo;\n"
          f"     `delta` for {args.joint} should be ~0. If it DOUBLED, the sign was backwards:\n"
          f"     re-run with --degrees {-args.degrees:+.2f} twice to undo and correct.\n"
          f"  2. ./robot home             -- it should reach the pose physically now.\n"
          f"  3. git diff calibration/    -- commit it, so the frame your models were\n"
          f"     trained in stays version-controlled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
