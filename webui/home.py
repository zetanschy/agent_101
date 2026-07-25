#!/usr/bin/env python3
"""Move the SO-ARM101 follower to a saved HOME pose — or capture one.

  python webui/home.py            go to the saved home (config/home_pose.json);
                                  falls back to calibrated-zero if none saved yet
  python webui/home.py --set      free the arm, you position it by hand, press
                                  ENTER, and that pose is saved as home

Homing is closed-loop: it reads the joint positions and integrates the error so
it actually reaches the target (the servos' low P gain + I=0 otherwise leave a
few-degree steady-state offset under gravity).
"""
import json
import os
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

HOME_FILE = Path(os.environ.get("HOME_POSE_FILE", "/workspace/config/home_pose.json"))
TOL_DEG, GAIN, MAX_CORR = 1.5, 0.6, 45.0

cfg = SOFollowerRobotConfig(
    port=os.environ.get("ROBOT_PORT", "/dev/ttyACM1"),
    id=os.environ.get("ROBOT_ID", "zetans_follower"),
    cameras={},
    use_degrees=True,
    disable_torque_on_disconnect=False,
)


def positions(robot, motors):
    obs = robot.get_observation()
    return {m: float(obs[f"{m}.pos"]) for m in motors}


robot = SOFollower(cfg)
print("connecting follower...", flush=True)
robot.connect()
try:
    motors = sorted(k[:-4] for k in robot.get_observation() if k.endswith(".pos"))

    # ---------- capture mode ----------
    if "--set" in sys.argv:
        try:
            robot.bus.disable_torque()
        except Exception as e:
            print(f"torque-off note: {e}", flush=True)
        print("SET_HOME: the arm is now FREE — move it by hand to your desired home pose.", flush=True)
        input("Press ENTER to capture that pose as home... ")
        pose = {m: round(v, 2) for m, v in positions(robot, motors).items()}
        HOME_FILE.parent.mkdir(parents=True, exist_ok=True)
        HOME_FILE.write_text(json.dumps(pose, indent=2))
        print(f"HOME_SET saved to {HOME_FILE}: {pose}", flush=True)

    # ---------- go-home mode ----------
    else:
        if HOME_FILE.exists():
            saved = json.loads(HOME_FILE.read_text())
            target = {m: float(saved.get(m, 0.0)) for m in motors}
            print(f"home target (from {HOME_FILE.name}):",
                  {m: round(target[m], 1) for m in motors}, flush=True)
        else:
            target = {m: 0.0 for m in motors}
            print("no saved home yet — using calibrated zero. "
                  "Define your own with:  ./robot home --set", flush=True)

        try:
            robot.bus.enable_torque()
        except Exception as e:
            print(f"torque enable note: {e}", flush=True)

        start = positions(robot, motors)
        print("start:", {m: round(start[m], 1) for m in motors}, flush=True)

        # phase 1: coarse open-loop approach
        N = 60
        for i in range(1, N + 1):
            f = i / N
            robot.send_action({f"{m}.pos": start[m] + (target[m] - start[m]) * f for m in motors})
            time.sleep(0.033)

        # phase 2: closed-loop integral correction to actually reach target
        cmd = dict(target)
        worst = 999.0
        for _ in range(40):
            err = {m: target[m] - v for m, v in positions(robot, motors).items()}
            worst = max(abs(v) for v in err.values())
            if worst < TOL_DEG:
                break
            for m in motors:
                lo, hi = target[m] - MAX_CORR, target[m] + MAX_CORR
                cmd[m] = max(lo, min(hi, cmd[m] + GAIN * err[m]))
            robot.send_action({f"{m}.pos": cmd[m] for m in motors})
            time.sleep(0.12)

        end = {m: round(v, 1) for m, v in positions(robot, motors).items()}
        print("end:  ", end, flush=True)
        print(f"HOME_DONE (residual {worst:.1f} deg)" if worst < TOL_DEG
              else f"HOME_PARTIAL (residual {worst:.1f} deg)", flush=True)
finally:
    try:
        robot.disconnect()
    except Exception as e:
        print(f"disconnect note: {e}", flush=True)
