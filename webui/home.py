#!/usr/bin/env python3
"""Move the SO-ARM101 follower slowly to its calibrated-zero (neutral) pose.
Reads current joint positions and interpolates each to 0 over ~2s. Used by the
web UI 'Move Home' button and `./robot home`. Requires calibration to be present.
"""
import os
import time

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

cfg = SOFollowerRobotConfig(
    port=os.environ.get("ROBOT_PORT", "/dev/ttyACM1"),
    id=os.environ.get("ROBOT_ID", "zetans_follower"),
    cameras={},          # home needs no vision — motors only (faster connect)
    use_degrees=True,    # 0 deg per joint == calibrated neutral
)

robot = SOFollower(cfg)
print("connecting follower...", flush=True)
robot.connect()
try:
    obs = robot.get_observation()
    motors = sorted(k[:-4] for k in obs if k.endswith(".pos"))
    cur = {m: float(obs[f"{m}.pos"]) for m in motors}
    print("current:", {m: round(cur[m], 1) for m in motors}, flush=True)

    N = 60  # interpolation steps over ~2s (slow + safe)
    for i in range(1, N + 1):
        f = i / N
        robot.send_action({f"{m}.pos": cur[m] * (1.0 - f) for m in motors})
        time.sleep(0.03)

    print("HOME_DONE", flush=True)
finally:
    robot.disconnect()
