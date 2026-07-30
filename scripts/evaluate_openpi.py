#!/usr/bin/env python3
"""Run an openpi (JAX) checkpoint on the SO-ARM101 and report inference latency.

This is the setup that worked in April 2026, recovered and parameterised: lerobot
drives the hardware, openpi supplies the policy. Kept as a REFERENCE — it is the
yardstick for whether a lerobot pi05 problem is the weights or the execution.

    ./robot openpi-eval --policy /checkpoints/pi05lora_caps_pick_place_soarm_21000 \
        --task "Pick the cap and place it into the red mug"
    ./robot openpi-eval --policy <dir> --task "..." --actions 15 --duration 60
    ./robot openpi-eval --policy <dir> --task "..." --dry-run   # no motion, latency only

Camera mapping: the dataset has front+grip, and openpi's Soarm101Inputs names its
second view `observation/images/side`, so the grip frame goes into that slot —
matching how these checkpoints were trained.

--actions is the parameter that matters. The original ran 15 actions per predicted
chunk (0.5s at 30fps) before re-planning; lerobot's equivalent is n_action_steps,
which defaults to 50 (1.7s). Everything else here mirrors the original loop.
"""

import argparse
import os

# Must precede any JAX import: without it JAX grabs almost all VRAM up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import statistics
import time

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# lerobot 0.6.1 moved this: robots.so101_follower -> robots.so_follower (the same
# rename that makes datasets report robot_type "so_follower"). The class names are
# unchanged, so the original script's import path is the only thing that broke.
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep
from openpi.policies import policy_config
from openpi.training import config as pi0_config

STATE_DIM = 6


def env(key: str, default: str) -> str:
    return os.environ.get(key, default) or default


def index_or_path(value: str) -> int | str:
    return value if value.startswith("/dev/") else int(value)


def cameras(fps: int) -> dict[str, OpenCVCameraConfig]:
    """front + grip, from .env — the same indices the rest of the repo uses."""
    w, h = int(env("CAM_WIDTH", "640")), int(env("CAM_HEIGHT", "480"))
    fourcc = env("CAM_FOURCC", "MJPG") or None
    return {
        name: OpenCVCameraConfig(
            index_or_path=index_or_path(idx), fps=fps, width=w, height=h, fourcc=fourcc
        )
        for name, idx in (
            ("front", env("CAM_FRONT_INDEX", "4")),
            ("grip", env("CAM_GRIP_INDEX", "6")),
        )
    }


def build_observation(robot: SO101Follower, raw: dict, prompt: str) -> dict:
    """lerobot get_observation() -> openpi / Soarm101Inputs keys (slash-separated)."""
    state = np.asarray([float(raw[k]) for k in robot.action_features], dtype=np.float32)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"expected {STATE_DIM} joints, robot reported {state.shape[0]}")
    missing = [k for k in ("front", "grip") if k not in raw]
    if missing:
        raise KeyError(f"observation missing camera(s) {missing}; check CAM_*_INDEX in .env")
    return {
        "prompt": prompt,
        "observation/state": state,
        "observation/images/front": raw["front"],
        # grip fills openpi's "side" slot — see the module docstring.
        "observation/images/side": raw["grip"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="checkpoint dir (orbax) or hf repo id")
    p.add_argument("--task", required=True, help="prompt; must match the training task string")
    p.add_argument("--config", default="pi05_soarm101_lora", help="openpi TrainConfig name")
    p.add_argument("--actions", type=int, default=15, help="actions executed per chunk (default 15)")
    p.add_argument("--horizon", type=int, default=None, help="override model action_horizon")
    p.add_argument("--fps", type=int, default=None, help="control rate (default CAM_FPS from .env)")
    p.add_argument("--duration", type=float, default=60.0, help="seconds; 0 = until Ctrl-C")
    p.add_argument("--dry-run", action="store_true", help="predict but never send actions")
    # Joint units. lerobot 0.6.1 added use_degrees and defaults it to True, but these
    # openpi checkpoints were trained on datasets recorded before that, in normalized
    # RANGE_M100_100 — visible in the norm stats, which clamp at exactly ±100, whereas
    # a 0.6.1-recorded dataset reaches -166. Feeding degrees to a normalized-trained
    # policy puts the state out of distribution AND misreads its actions, so default
    # to what the checkpoint expects rather than to lerobot's current default.
    p.add_argument("--units", choices=("normalized", "degrees"), default="normalized",
                   help="joint units the policy was trained in (default normalized)")
    args = p.parse_args()

    fps = args.fps or int(env("CAM_FPS", "30"))

    cfg = pi0_config.get_config(args.config)
    if args.horizon is not None:
        import dataclasses

        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, action_horizon=args.horizon))
    print(f"config={args.config}  action_horizon={cfg.model.action_horizon}  fps={fps}")
    print(f"loading {args.policy} ...", flush=True)
    t0 = time.perf_counter()
    policy = policy_config.create_trained_policy(cfg, args.policy)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    robot = SO101Follower(
        SO101FollowerConfig(
            port=env("ROBOT_PORT", "/dev/ttyACM1"),
            id=env("ROBOT_ID", "zetans_follower"),
            cameras=cameras(fps),
            use_degrees=(args.units == "degrees"),
        )
    )
    print(f"joint units: {args.units}"
          f"{'  (lerobot 0.6.1 default is degrees — overridden)' if args.units == 'normalized' else ''}")
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("robot did not connect")
    print(f"robot connected; executing {args.actions} actions per chunk"
          f"{' (DRY RUN — no motion)' if args.dry_run else ''}", flush=True)

    step, action_index, chunk = 0, 0, None
    latencies: list[float] = []
    started = time.perf_counter()
    try:
        while args.duration <= 0 or (time.perf_counter() - started) < args.duration:
            tick = time.perf_counter()
            # Re-plan when the executed window is used up. Note this tick sends no
            # action — same as the original loop, so a chunk costs 1 + N ticks.
            if chunk is None or action_index >= min(args.actions, len(chunk)):
                obs = build_observation(robot, robot.get_observation(), args.task)
                t_pred = time.perf_counter()
                out = policy.infer(obs)
                dt = time.perf_counter() - t_pred
                latencies.append(dt)
                chunk, action_index = out["actions"], 0
                print(f"step {step:5d}  predicted {len(chunk):3d} actions, using "
                      f"{min(args.actions, len(chunk)):3d}  |  {dt * 1000:6.1f} ms "
                      f"(mean {statistics.mean(latencies) * 1000:6.1f})", flush=True)
            else:
                action = chunk[action_index]
                if not args.dry_run:
                    robot.send_action(
                        {name: float(action[i]) for i, name in enumerate(robot.action_features)
                         if i < len(action)}
                    )
                action_index += 1
                step += 1
            precise_sleep(max(1.0 / fps - (time.perf_counter() - tick), 0.0))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001 - never mask the latency report
            pass

    if latencies:
        ms = sorted(x * 1000 for x in latencies)
        print("\ninference latency (ms) over", len(ms), "predictions")
        print(f"  mean {statistics.mean(ms):7.1f}   p50 {ms[len(ms) // 2]:7.1f}   "
              f"p95 {ms[min(len(ms) - 1, int(0.95 * len(ms)))]:7.1f}   "
              f"min {ms[0]:7.1f}   max {ms[-1]:7.1f}")
        budget = 1000.0 * args.actions / fps
        print(f"  chunk of {args.actions} actions = {budget:.0f} ms of motion; "
              f"{'inference fits inside it' if statistics.mean(ms) < budget else 'INFERENCE EXCEEDS IT — the arm stalls each chunk'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
