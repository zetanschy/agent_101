#!/usr/bin/env python3
"""Persistent openpi (JAX) inference worker for the web UI.

Same contract as webui/infer_worker.py, so app.py drives either stack identically:

    stdin commands   run | stop | home | actions <n> | quit
    stdout markers   LOADING, MODEL_LOADED, RUN_START, RUN_STOP, HOME_DONE,
                     ACTION_STEPS_SET, UNLOADED

Why a separate worker rather than a branch inside infer_worker.py: openpi needs
JAX-on-GPU with CPU torch, lerobot pi05 needs CUDA torch, and those cannot share one
image. This one runs from agent101/openpi; infer_worker.py runs from agent101/lerobot.

The policy/camera/observation logic is imported from scripts/openpi/evaluate.py so
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

# Fraction of TOTAL VRAM, not free — see .env (XLA_MEM_FRACTION), which compose
# passes in. This default only applies to a bare `python webui/openpi_worker.py`.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_eval_module():
    """Import scripts/openpi/evaluate.py without making it a package."""
    spec = importlib.util.spec_from_file_location(
        "openpi_evaluate", ROOT / "scripts" / "openpi" / "evaluate.py")
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
    #         still executing, so the arm never pauses. The observation is one window
    #         stale, and the new chunk is spliced in cold — the arm jumps at the seam.
    # rtc   : async, plus real-time chunking (arXiv:2506.07339). The new chunk is
    #         generated knowing exactly which actions will have been executed by the
    #         time it lands, so the seam is continuous by construction.
    p.add_argument("--mode", choices=("sync", "async", "rtc"), default="async")
    p.add_argument("--rtc-schedule", choices=("zeros", "ones", "linear", "exp"), default="exp",
                   help="how the prefix constraint decays past the frozen actions")
    p.add_argument("--rtc-max-guidance", type=float, default=5.0, help="beta_max")
    p.add_argument("--rtc-jacobian", choices=("identity", "full"), default="identity",
                   help="identity matches the lerobot reference and is free; full is the true VJP (~2x slower)")
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

    # In rtc mode every inference is guided by the tail of the chunk still executing.
    # The chunker only does the bookkeeping; the guidance itself is in the sampler.
    chunker = None
    if args.mode == "rtc":
        from openpi.policies.rtc import RealTimeChunker

        chunker = RealTimeChunker(
            policy,
            execution_horizon=args.actions,
            prefix_attention_schedule=args.rtc_schedule,
            max_guidance_weight=args.rtc_max_guidance,
            jacobian=args.rtc_jacobian,
        )
        print(f"rtc: schedule={args.rtc_schedule} beta_max={args.rtc_max_guidance} "
              f"jacobian={args.rtc_jacobian}", flush=True)

    # Warm up here, not on the first Start. JAX traces and compiles the model on the
    # first infer() — tens of seconds — and doing that inside the control loop stalls
    # the arm at step 0. One throwaway inference on a real observation moves the whole
    # cost into Load. No action is sent, so the robot does not move.
    t = time.perf_counter()
    warm_obs = ev.build_observation(robot, robot.get_observation(), args.task)
    policy.infer(warm_obs)
    if chunker is not None:
        # The guided sampler is a SECOND trace (the prefix changes the argument tree),
        # so compile it here too or the first replan of the run eats the compile
        # mid-motion. The first call only exists to give the chunker a prefix.
        chunker.infer(warm_obs, prefix_start=0, inference_delay=1)
        chunker.infer(warm_obs, prefix_start=1, inference_delay=1)
        chunker.reset()
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
        # `end` is the index at which this chunk is retired. With rtc the next chunk
        # does not start at 0: the first `delay` of its actions were already executed
        # off the previous chunk, and the sampler was told as much.
        chunk, idx, start, end = None, 0, 0, 0
        delay = 0
        latencies: list[float] = []
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        pending: concurrent.futures.Future | None = None
        if chunker is not None:
            # A chunk left over from the previous run would pin this one to actions
            # nobody is executing.
            chunker.reset()

        def grab_obs():
            """Read the robot. MAIN THREAD ONLY.

            get_observation() talks to the feetech bus, which is not thread-safe: doing
            it from the inference thread while the loop is sending actions collides on
            the serial port ("Failed to sync write 'Goal_Position' ... Port is in use!").
            So the observation is captured here and only the model call is offloaded.
            """
            return ev.build_observation(robot, robot.get_observation(), args.task)

        def infer_only(obs, prefix_start: int, inference_delay: int):
            """Pure compute — safe on a worker thread, touches no hardware."""
            t = time.perf_counter()
            if chunker is None:
                out = policy.infer(obs)
            else:
                out = chunker.infer(obs, prefix_start=prefix_start, inference_delay=inference_delay,
                                    execution_horizon=state["actions"])
            return out["actions"], time.perf_counter() - t

        def predict(prefix_start: int, inference_delay: int):
            return infer_only(grab_obs(), prefix_start, inference_delay)

        def record(dt: float, n: int, window: int) -> None:
            latencies.append(dt)
            rtc = f" d={chunker.last_delay} s={chunker.last_horizon}" if chunker is not None else ""
            print(f"inference {dt * 1000:.0f} ms (mean {statistics.mean(latencies) * 1000:.0f}) "
                  f"chunk={n} using={window} mode={args.mode}{rtc}", flush=True)

        try:
            while not shutdown.is_set():
                tick = time.perf_counter()
                if chunk is not None and pending is None:
                    # The window stays live-tunable from the UI, but only until an
                    # inference is in flight: moving `end` after that would falsify
                    # the delay the sampler was given.
                    end = min(start + state["actions"], len(chunk))

                if chunk is None:
                    # First chunk has to block; the jit cost was already paid at load.
                    # Nothing has executed yet, so there is no prefix to honour.
                    chunk, dt = predict(0, 0)
                    idx = start = 0
                    end = min(start + state["actions"], len(chunk))
                    record(dt, len(chunk), end - idx)
                elif idx >= end:
                    if pending is not None:
                        chunk, dt = pending.result()   # normally already finished
                        pending = None
                        # `delay` actions of this chunk are already behind us — that
                        # is precisely what the sampler was told to reproduce.
                        idx = delay
                    else:
                        # sync: the arm holds still, so nothing gets executed while we
                        # think and the new chunk starts at 0 with no frozen prefix.
                        chunk, dt = predict(end, 0)
                        idx = 0
                    start = idx
                    end = min(start + state["actions"], len(chunk))
                    record(dt, len(chunk), end - idx)
                else:
                    # Kick the next inference early in the window so it has most of
                    # the window to finish — at 30fps and window 15 that is ~470ms of
                    # cover, more than an inference typically needs. In rtc mode a new
                    # chunk starts at idx = delay, so this fires on its first tick and
                    # the cover is the whole window.
                    if args.mode in ("async", "rtc") and pending is None and idx >= 1:
                        # The chunk is retired at a fixed index, not on arrival, so the
                        # number of actions executed between this observation and the
                        # swap is exactly end - idx — known now, and still exact if the
                        # inference overruns (the loop just stalls at `end`). That is
                        # the one number RTC cannot afford to guess.
                        delay = min(end - idx, len(chunk) - 1) if chunker is not None else 0
                        # Observation read HERE on the main thread; only the model call
                        # is offloaded, so nothing else touches the serial bus.
                        pending = pool.submit(infer_only, grab_obs(), idx, delay)
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
