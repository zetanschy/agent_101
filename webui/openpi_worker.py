#!/usr/bin/env python3
"""Persistent openpi (JAX) inference worker for the web UI.

Same contract as webui/infer_worker.py, so app.py drives either stack identically:

    stdin commands   run | stop | home | actions <n> | quit
    stdout markers   LOADING, MODEL_LOADED, RUN_START, RUN_STOP, HOME_DONE,
                     ACTION_STEPS_SET, UNLOADED

Why a separate worker rather than a branch inside infer_worker.py: openpi needs
JAX-on-GPU with CPU torch, lerobot pi05 needs CUDA torch, and those cannot share one
image. This one runs from agent101/openpi; infer_worker.py runs from agent101/lerobot.

The policy/camera/observation logic is imported from scripts/evaluate_openpi.py so
there is exactly one definition of how a frame becomes an openpi observation — that
script stays the standalone/headless entry point.
"""

import argparse
import importlib.util
import os
import pathlib
import statistics
import sys
import threading
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_eval_module():
    """Import scripts/evaluate_openpi.py without making it a package."""
    spec = importlib.util.spec_from_file_location("evaluate_openpi", ROOT / "scripts" / "evaluate_openpi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Same location and override as infer_worker.py, so both stacks home to one pose.
HOME_FILE = pathlib.Path(os.environ.get("HOME_POSE_FILE", "/workspace/config/home_pose.json"))


def _positions(robot, motors) -> dict:
    obs = robot.get_observation()
    return {m: float(obs[f"{m}.pos"]) for m in motors}


def do_home(robot) -> None:
    """Move to the saved home pose over the worker's OWN connection.

    Deliberately not a subprocess call to webui/home.py: this worker already holds
    /dev/ttyACM*, and a second process opening the same bus fails. Mirrors
    infer_worker.do_home — coarse ramp, then closed-loop integral correction.
    """
    import json

    obs = robot.get_observation()
    motors = sorted(k[:-4] for k in obs if k.endswith(".pos"))
    saved = json.loads(HOME_FILE.read_text()) if HOME_FILE.exists() else {}
    target = {m: float(saved.get(m, 0.0)) for m in motors}

    print("HOME_START", flush=True)
    start = _positions(robot, motors)
    for i in range(1, 61):
        f = i / 60
        robot.send_action({f"{m}.pos": start[m] + (target[m] - start[m]) * f for m in motors})
        time.sleep(0.033)
    cmd, worst = dict(target), 999.0
    for _ in range(40):
        err = {m: target[m] - v for m, v in _positions(robot, motors).items()}
        worst = max(abs(v) for v in err.values())
        if worst < 1.5:
            break
        for m in motors:
            lo, hi = target[m] - 45.0, target[m] + 45.0
            cmd[m] = max(lo, min(hi, cmd[m] + 0.6 * err[m]))
        robot.send_action({f"{m}.pos": cmd[m] for m in motors})
        time.sleep(0.12)
    print(f"HOME_DONE (residual {worst:.1f} deg)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--actions", type=int, default=15)
    p.add_argument("--units", choices=("normalized", "degrees"), default="degrees")
    p.add_argument("--fps", type=int, default=None)
    # sync  : predict, then execute the window. The arm holds still while it thinks.
    # async : predict the NEXT chunk in a background thread while the current one is
    #         still executing, so the arm never pauses — openpi's equivalent of
    #         lerobot's rtc. Same trade as rtc: the observation is one window stale.
    p.add_argument("--mode", choices=("sync", "async"), default="async")
    args = p.parse_args()

    ev = _load_eval_module()
    from openpi.policies import policy_config
    from openpi.training import config as pi0_config

    fps = args.fps or int(ev.env("CAM_FPS", "30"))
    config_name = args.config or ev.infer_config(args.policy)

    print("LOADING", flush=True)
    cfg = pi0_config.get_config(config_name)
    policy = policy_config.create_trained_policy(cfg, args.policy)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(
        SO101FollowerConfig(
            port=ev.env("ROBOT_PORT", "/dev/ttyACM1"),
            id=ev.env("ROBOT_ID", "zetans_follower"),
            cameras=ev.cameras(fps),
            use_degrees=(args.units == "degrees"),
        )
    )
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("robot did not connect")
    print(f"config={config_name} horizon={cfg.model.action_horizon} units={args.units} "
          f"mode={args.mode} fps={fps}", flush=True)

    # Warm up here, not on the first Start. JAX traces and compiles the model on the
    # first infer() — tens of seconds — and doing that inside the control loop stalls
    # the arm at step 0. One throwaway inference on a real observation moves the whole
    # cost into Load. No action is sent, so the robot does not move.
    t = time.perf_counter()
    policy.infer(ev.build_observation(robot, robot.get_observation(), args.task))
    print(f"WARMUP_DONE {(time.perf_counter() - t) * 1000:.0f} ms (jit compile)", flush=True)

    print("MODEL_LOADED", flush=True)

    # `actions` is mutable so the UI can retune the open-loop window live, the same
    # way lerobot's n_action_steps is live in the other worker.
    state = {"actions": args.actions}
    shutdown = threading.Event()
    shutdown.set()
    run_thread: threading.Thread | None = None

    def loop():
        import concurrent.futures

        from lerobot.utils.robot_utils import precise_sleep

        print("RUN_START", flush=True)
        chunk, idx = None, 0
        latencies: list[float] = []
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        pending: concurrent.futures.Future | None = None

        def grab_obs():
            """Read the robot. MAIN THREAD ONLY.

            get_observation() talks to the feetech bus, which is not thread-safe: doing
            it from the inference thread while the loop is sending actions collides on
            the serial port ("Failed to sync write 'Goal_Position' ... Port is in use!").
            So the observation is captured here and only the model call is offloaded.
            """
            return ev.build_observation(robot, robot.get_observation(), args.task)

        def infer_only(obs):
            """Pure compute — safe on a worker thread, touches no hardware."""
            t = time.perf_counter()
            out = policy.infer(obs)
            return out["actions"], time.perf_counter() - t

        def predict():
            return infer_only(grab_obs())

        def record(dt: float, n: int, window: int) -> None:
            latencies.append(dt)
            print(f"inference {dt * 1000:.0f} ms (mean {statistics.mean(latencies) * 1000:.0f}) "
                  f"chunk={n} using={window} mode={args.mode}", flush=True)

        try:
            while not shutdown.is_set():
                tick = time.perf_counter()
                window = min(state["actions"], len(chunk)) if chunk is not None else 0

                if chunk is None:
                    # First chunk has to block; the jit cost was already paid at load.
                    chunk, dt = predict()
                    idx = 0
                    record(dt, len(chunk), min(state["actions"], len(chunk)))
                elif idx >= window:
                    if pending is not None:
                        chunk, dt = pending.result()   # normally already finished
                        pending = None
                    else:
                        chunk, dt = predict()
                    idx = 0
                    record(dt, len(chunk), min(state["actions"], len(chunk)))
                else:
                    # In async mode, kick the next inference one action into the window
                    # so it has (window-1) ticks to finish — at 30fps and window 15
                    # that is ~470ms of cover, more than an inference typically needs.
                    if args.mode == "async" and pending is None and idx >= 1:
                        # Observation read HERE on the main thread; only the model call
                        # is offloaded, so nothing else touches the serial bus.
                        pending = pool.submit(infer_only, grab_obs())
                    action = chunk[idx]
                    robot.send_action(
                        {n: float(action[i]) for i, n in enumerate(robot.action_features) if i < len(action)}
                    )
                    idx += 1
                precise_sleep(max(1.0 / fps - (time.perf_counter() - tick), 0.0))
        except Exception as e:  # noqa: BLE001 - keep the worker alive for another run
            print(f"RUN_ERROR: {e}", flush=True)
        finally:
            if pending is not None:
                pending.cancel()
            pool.shutdown(wait=False)
            print("RUN_STOP", flush=True)

    try:
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "run":
                if run_thread and run_thread.is_alive():
                    continue
                shutdown.clear()
                run_thread = threading.Thread(target=loop, daemon=True)
                run_thread.start()
            elif cmd == "stop":
                shutdown.set()
            elif cmd.startswith("actions "):
                try:
                    state["actions"] = int(cmd.split()[1])
                    print(f"ACTION_STEPS_SET {state['actions']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"ACTION_STEPS_ERROR: {e}", flush=True)
            elif cmd == "home":
                shutdown.set()
                if run_thread and run_thread.is_alive():
                    run_thread.join(timeout=20)
                try:
                    do_home(robot)
                except Exception as e:  # noqa: BLE001
                    print(f"HOME_ERROR: {e}", flush=True)
            elif cmd == "quit":
                shutdown.set()
                break
    finally:
        shutdown.set()
        if run_thread and run_thread.is_alive():
            run_thread.join(timeout=20)
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        print("UNLOADED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
