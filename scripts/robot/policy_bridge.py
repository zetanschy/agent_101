#!/usr/bin/env python3
"""Serial-bus half of `./robot sim-policy --real`: the only thing that touches motors.

    ./robot policy-bridge            # publish state, send NOTHING (the default)
    ./robot policy-bridge --engage   # ...and actually drive the follower

Isaac cannot run this. Isaac Sim is a native conda install and the arm lives behind
lerobot in a container, so the two halves talk through two files, atomically renamed,
exactly as ./robot leader-publish and sim-teleop already do:

    sim/outputs/policy/targets.json   written by the simulator: what to command
    sim/outputs/policy/state.json     written here: where the arm actually is

The state file is published WHETHER OR NOT anything is being commanded, because that
is what draws the ghost arm in the viewport -- so the honest way to try a checkpoint
is to run this with no --engage, move the goal around, and watch how far the ghost
would have had to go before you let it.

SAFETY, in the order it matters:

  * DISENGAGED BY DEFAULT. Without --engage this process opens the bus, reads it, and
    never writes. The simulator's own --engage is a second, independent switch: a
    target frame that does not say engaged is not sent even here.
  * RATE LIMIT. Commands move at most MAX_DEG_PER_S per joint. A policy trained in a
    simulation with a massless wrist can ask for a step change; the real arm answering
    one at full torque is how a gripper meets a table.
  * WATCHDOG. Targets older than STALE_S stop the commanding -- if the simulator dies
    mid-reach, the arm holds where it is instead of finishing a stale instruction.
  * TORQUE OFF on the way out, including on Ctrl-C.

Units are the other half of not breaking anything. The policy thinks in URDF joints
and radians, the bus wants lerobot names, degrees for the arm, and PERCENT for the
gripper -- kinematics.urdf_deg_to_lerobot is the one place that conversion lives.
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
IPC = ROOT / "sim" / "outputs" / "policy"
TARGETS, STATE = IPC / "targets.json", IPC / "state.json"

# A target frame older than this is not acted on. Three control periods at 30 Hz:
# long enough to ride out a slow frame in the simulator, short enough that a crash
# stops the arm before it has gone anywhere.
STALE_S = 0.1
# Degrees per second, per joint. The SO-101 can move far faster; this is a leash for
# a policy whose sim had no table in it.
MAX_DEG_PER_S = 60.0
HZ = 30.0


def _load(name: str):
    """Import one module out of sim/sim_agent101 without importing the package.

    The package's __init__ pulls in isaaclab, which does not exist in this container.
    Same trick calibrate_extrinsics.py uses, for the same reason.
    """
    import importlib.util

    path = ROOT / "sim" / "sim_agent101" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kin = _load("kinematics")
URDF_TO_LEROBOT = _kin.URDF_TO_LEROBOT
urdf_deg_to_lerobot, lerobot_to_urdf_deg = _kin.urdf_deg_to_lerobot, _kin.lerobot_to_urdf_deg


def write_atomic(path: pathlib.Path, payload: dict) -> None:
    """Rename, never write in place: a reader polling at 30 Hz will otherwise catch
    the file in the empty gap between truncate and write and take a JSONDecodeError.
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def read_targets() -> tuple[dict | None, bool, str]:
    """(joints_deg, engaged, why-not) from the simulator's target file."""
    try:
        d = json.loads(TARGETS.read_text())
    except (OSError, json.JSONDecodeError):
        return None, False, "no targets file yet"
    age = time.time() - float(d.get("t", 0.0))
    if age > STALE_S:
        return None, False, f"targets {age:.1f}s stale"
    return dict(d.get("joints_deg", {})), bool(d.get("engaged", False)), ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engage", action="store_true", help="actually drive the follower (default: read only)")
    p.add_argument("--max-deg-per-s", type=float, default=MAX_DEG_PER_S, help=f"rate limit (default {MAX_DEG_PER_S})")
    p.add_argument("--hz", type=float, default=HZ)
    a = p.parse_args()

    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(SO101FollowerConfig(
        port=os.environ.get("ROBOT_PORT", "/dev/ttyACM1"),
        id=os.environ.get("ROBOT_ID", "zetans_follower"), cameras={}, use_degrees=True))
    follower.connect()
    IPC.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / a.hz
    max_step = a.max_deg_per_s * dt
    commanded: dict[str, float] | None = None      # last thing SENT, for the rate limit
    said = ""
    print(f"policy-bridge up. {'DRIVING THE ARM' if a.engage else 'read-only (no --engage)'}; "
          f"targets <- {TARGETS.relative_to(ROOT)}, state -> {STATE.relative_to(ROOT)}", flush=True)
    try:
        while True:
            started = time.time()

            # Publish where the arm is, always: this is what draws the ghost.
            obs = follower.get_observation()
            measured = lerobot_to_urdf_deg({n: obs[f"{n}.pos"] for n in URDF_TO_LEROBOT.values()
                                            if f"{n}.pos" in obs})
            write_atomic(STATE, {"t": time.time(), "joints_deg": measured})

            targets, engaged, why = read_targets()
            allowed = a.engage and engaged and targets is not None
            note = why if why else ("" if allowed else "not engaged")
            if note != said:
                print(f"  [{'commanding' if allowed else 'holding'}] {note}", flush=True)
                said = note

            if allowed:
                # Rate limit from the last COMMAND, not from the measurement: limiting
                # against where the arm got to turns a servo lagging under load into a
                # command that creeps, which looks like the policy going slack.
                if commanded is None:
                    commanded = dict(measured)
                # Start from the measurement so joints the policy does NOT drive --
                # the gripper -- are commanded to hold where they are. send_action
                # with a partial dict is not something to find out about on hardware.
                nxt = dict(measured)
                for joint, want in targets.items():
                    if joint not in commanded:
                        continue
                    delta = max(-max_step, min(max_step, want - commanded[joint]))
                    nxt[joint] = commanded[joint] + delta
                follower.send_action({f"{k}.pos": v for k, v in urdf_deg_to_lerobot(nxt).items()})
                commanded = nxt
            else:
                # Not commanding: forget the command history so re-engaging ramps from
                # where the arm IS, not from a target it never reached.
                commanded = None

            remaining = dt - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nstopping, torque off", flush=True)
    finally:
        follower.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
