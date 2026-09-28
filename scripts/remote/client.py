#!/usr/bin/env python3
"""Drive this arm from a policy running on another machine.

    ./robot infer-remote --server gpubox:8080 --policy molmoact2 \
        --checkpoint allenai/MolmoAct2-SO100_101 --task "pick up the lemon"

The robot half. It owns the bus and the cameras, streams observations to the server
and executes the action chunks that come back; the GPU box runs scripts/remote/serve.py.
The wire is lerobot's own async-inference protocol, so either half can be swapped for
a stock lerobot one.

WHY THE ARM DOES NOT STUTTER over a link with real latency. The policy returns a
CHUNK, not an action, and the client keeps a queue of it. `--chunk-threshold` is the
fraction of that queue remaining at which it asks for the next one, so a new chunk is
in flight while the old one is still executing. At 0.5 the request goes out with half
the chunk left -- 0.5 s of cover at 30 fps with a 30-action chunk -- which is the
budget the round trip and the inference have to fit inside. Lower it and the arm runs
further into a stale plan; raise it and the server is queried more often than it can
answer. Watch `--debug-queue`.

CALIBRATION IS THE ROBOT'S, NOT THE SERVER'S. The follower is built from this repo's
.env and the version-controlled calibration/ directory, exactly as every other worker
here builds it, so the joint values on the wire mean the same thing they mean in a
recorded dataset. A server told to expect a differently calibrated arm is a silent
frame error, not an exception.

THE LINK IS UNAUTHENTICATED. See the note in serve.py: the intended way across
machines is an SSH tunnel, which turns --server into 127.0.0.1 and needs no open port
on the GPU box.

    ssh -N -L 8080:127.0.0.1:8080 gpubox &
    ./robot infer-remote --server 127.0.0.1:8080 ...
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="127.0.0.1:8080", help="host:port of the policy server")
    p.add_argument("--policy", default="molmoact2", help="policy type the server should load")
    p.add_argument("--checkpoint", default="allenai/MolmoAct2-SO100_101",
                   help="what the SERVER loads; a Hub id or a path on that machine")
    p.add_argument("--task", default="", help="natural-language instruction for a VLA")
    p.add_argument("--actions-per-chunk", type=int, default=30,
                   help="actions requested per inference; the server truncates to this")
    p.add_argument("--chunk-threshold", type=float, default=0.5,
                   help="re-query when this fraction of the queue remains (see the docstring)")
    p.add_argument("--fps", type=int, default=None, help="control rate (default CAM_FPS)")
    p.add_argument("--aggregate", default="weighted_average",
                   help="how a new chunk is blended with the tail of the old one")
    p.add_argument("--debug-queue", action="store_true", help="plot the action queue size")
    a = p.parse_args()

    fps = a.fps or int(env("CAM_FPS", "30"))

    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import async_client
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    def cam(index_key: str, name: str):
        idx = env(index_key, "")
        if not idx:
            return None
        return OpenCVCameraConfig(
            index_or_path=int(idx) if idx.isdigit() else pathlib.Path(idx),
            fps=fps, width=int(env("CAM_WIDTH", "640")), height=int(env("CAM_HEIGHT", "480")),
            fourcc=env("CAM_FOURCC", "MJPG"))

    cameras = {n: c for n, c in (("front", cam("CAM_FRONT_INDEX", "front")),
                                 ("grip", cam("CAM_GRIP_INDEX", "grip"))) if c is not None}
    if not cameras:
        print("no cameras configured — set CAM_FRONT_INDEX / CAM_GRIP_INDEX in .env",
              file=sys.stderr)
        return 1

    if env("ROBOT_TYPE", "") == "real2sim":
        # `./robot infer-remote --sim ENGINE`: the live sim (sim/real2sim/live) instead of
        # the arm. Same cameras by name, same degrees; ROBOT_PORT is the sim's socket.
        from lerobot_robot_real2sim import Real2SimConfig

        robot = Real2SimConfig(port=env("ROBOT_PORT", ""), id=env("ROBOT_ID", "sim"), cameras=cameras)
    else:
        robot = SO101FollowerConfig(
            port=env("ROBOT_PORT", "/dev/ttyACM1"),
            id=env("ROBOT_ID", "zetans_follower"),
            cameras=cameras,
            # Degrees, matching how this bench records and how every other worker here
            # drives the arm. A policy trained on normalized units needs this flipped,
            # and that is a property of the CHECKPOINT, not of the transport.
            use_degrees=True,
        )

    cfg = RobotClientConfig(
        policy_type=a.policy,
        pretrained_name_or_path=a.checkpoint,
        robot=robot,
        actions_per_chunk=a.actions_per_chunk,
        task=a.task,
        server_address=a.server,
        chunk_size_threshold=a.chunk_threshold,
        fps=fps,
        aggregate_fn_name=a.aggregate,
        debug_visualize_queue_size=a.debug_queue,
    )

    print(f"server   : {a.server}", flush=True)
    print(f"policy   : {a.policy}  checkpoint={a.checkpoint}", flush=True)
    print(f"cameras  : {', '.join(cameras)}  fps={fps}", flush=True)
    print(f"task     : {a.task or '(none)'}", flush=True)
    async_client(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
