#!/usr/bin/env python3
"""Persistent inference worker: load the policy + connect the robot ONCE, then
run/stop the rollout loop on stdin commands without reloading. Lets the web UI
'Load' a model separately from Start/Stop so testing doesn't pay the (slow) pi05
load cost on every run.

Takes the same args as `lerobot-rollout` (built by webui/app.py). Duration is
forced to 0 (infinite) — start/stop is controlled here via the shutdown event.

Protocol (one command per stdin line):
    run   -> start the rollout control loop (prints RUN_START / RUN_STOP)
    stop  -> stop the loop, keep the model resident
    quit  -> tear down and exit
Markers printed to stdout for the backend: LOADING, MODEL_LOADED, RUN_START,
RUN_STOP, UNLOADED.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

# Importing the rollout CLI module runs its module-level registration imports
# (cameras/robots/teleoperators), which populate the draccus choice registries so
# --robot.type=so101_follower / --robot.cameras / etc. parse. main() is
# __main__-guarded, so this import has no side effect beyond registration.
import lerobot.scripts.lerobot_rollout  # noqa: F401
from lerobot.configs import parser
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy

_holder = {}
HOME_FILE = Path(os.environ.get("HOME_POSE_FILE", "/workspace/config/home_pose.json"))


@parser.wrap()
def _capture(cfg: RolloutConfig):
    _holder["cfg"] = cfg


def _positions(rw, motors):
    obs = rw.get_observation()
    return {m: float(obs[f"{m}.pos"]) for m in motors}


def do_home(ctx):
    """Move the robot to the saved home pose using the worker's own connection
    (so it works while a model is loaded — no unload needed). Closed-loop, same
    as webui/home.py."""
    rw = ctx.hardware.robot_wrapper
    obs = rw.get_observation()
    motors = sorted(k[:-4] for k in obs if k.endswith(".pos"))
    if HOME_FILE.exists():
        saved = json.loads(HOME_FILE.read_text())
        target = {m: float(saved.get(m, 0.0)) for m in motors}
    else:
        target = {m: 0.0 for m in motors}

    print("HOME_START", flush=True)
    start = _positions(rw, motors)
    for i in range(1, 61):                       # coarse approach
        f = i / 60
        rw.send_action({f"{m}.pos": start[m] + (target[m] - start[m]) * f for m in motors})
        time.sleep(0.033)
    cmd = dict(target)                            # closed-loop integral correction
    worst = 999.0
    for _ in range(40):
        err = {m: target[m] - v for m, v in _positions(rw, motors).items()}
        worst = max(abs(v) for v in err.values())
        if worst < 1.5:
            break
        for m in motors:
            lo, hi = target[m] - 45.0, target[m] + 45.0
            cmd[m] = max(lo, min(hi, cmd[m] + 0.6 * err[m]))
        rw.send_action({f"{m}.pos": cmd[m] for m in motors})
        time.sleep(0.12)
    print(f"HOME_DONE (residual {worst:.1f} deg)", flush=True)


def set_steps(ctx, n):
    ctx.policy.policy.config.num_inference_steps = int(n)   # pi05 reads this per inference
    print(f"STEPS_SET {n}", flush=True)


def main():
    _capture()  # parses sys.argv (the rollout args) into a RolloutConfig
    cfg = _holder["cfg"]
    cfg.duration = 0          # run until 'stop'
    cfg.display_data = False  # no rerun window in the worker

    shutdown = threading.Event()
    shutdown.set()            # start stopped

    print("LOADING", flush=True)
    ctx = build_rollout_context(cfg, shutdown)   # <-- loads policy + connects robot (slow, once)
    strategy = create_strategy(cfg.strategy)
    strategy.setup(ctx)
    print("MODEL_LOADED", flush=True)

    run_thread = None
    try:
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "run":
                if run_thread and run_thread.is_alive():
                    continue

                def _go():
                    print("RUN_START", flush=True)
                    try:
                        strategy.run(ctx)      # blocks until shutdown set (or resets each run)
                    except Exception as e:     # noqa: BLE001 - keep the worker alive on a run error
                        print(f"RUN_ERROR: {e}", flush=True)
                    finally:
                        print("RUN_STOP", flush=True)

                shutdown.clear()
                run_thread = threading.Thread(target=_go, daemon=True)
                run_thread.start()
            elif cmd == "stop":
                shutdown.set()
            elif cmd == "home":
                # Ensure any running loop is stopped and wound down first, then home.
                shutdown.set()
                if run_thread and run_thread.is_alive():
                    run_thread.join(timeout=20)
                try:
                    do_home(ctx)
                except Exception as e:  # noqa: BLE001
                    print(f"HOME_ERROR: {e}", flush=True)
            elif cmd.startswith("steps "):
                try:
                    set_steps(ctx, int(cmd.split()[1]))    # live — no reload
                except Exception as e:  # noqa: BLE001
                    print(f"STEPS_ERROR: {e}", flush=True)
            elif cmd == "quit":
                shutdown.set()
                break
    finally:
        shutdown.set()
        if run_thread and run_thread.is_alive():
            run_thread.join(timeout=10)
        try:
            strategy.teardown(ctx)
        except Exception as e:  # noqa: BLE001
            print(f"teardown note: {e}", flush=True)
        print("UNLOADED", flush=True)


if __name__ == "__main__":
    main()
