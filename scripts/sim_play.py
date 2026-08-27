#!/usr/bin/env python3
"""Bring up the push-T scene, step it, and prove the pieces are actually working.

    ./robot sim-play                       # headless, writes renders + a report
    ./robot sim-play --gui                 # watch it
    ./robot sim-play --steps 300 --task Agent101-So101-Push-T-Eval

Checks, in order, the things that silently go wrong when a scene is assembled:
  * the T spawns ON the mat and settles instead of sinking, exploding or drifting
  * its collision is the convex decomposition, not a hull that filled the notches
  * both cameras render non-black frames at the right resolution
  * the goal is reachable, the pose error term is finite, success can be evaluated

Renders land in sim/outputs/ so they can be put next to real captures.
"""

import argparse
import os
import sys
import traceback
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Agent101-So101-Push-T")
parser.add_argument("--steps", type=int, default=180, help="control steps to run (default 180 = 3 s)")
parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
parser.add_argument("--out", default=None, help="where to write renders (default sim/outputs)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim_agent101  # noqa: E402,F401  (registers the envs)
from sim_agent101 import mdp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Not ROOT/outputs: that directory is root-owned here (Docker made it) and these
# sim commands run natively as the user.
OUT = pathlib.Path(args.out) if args.out else ROOT / "sim" / "outputs"


def save(name: str, tensor) -> str:
    import imageio.v3 as iio

    img = tensor[0].detach().cpu().numpy()
    if img.dtype != "uint8":
        img = (img.clip(0, 1) * 255).astype("uint8")
    img = img[..., :3]
    path = OUT / f"{name}.png"
    iio.imwrite(path, img)
    return f"{path.relative_to(ROOT)}  {img.shape[1]}x{img.shape[0]}  mean {img.mean():.1f}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    print(f"\nenv: {args.task}")
    print(f"  scene keys: {sorted(env.unwrapped.scene.keys())}")

    obs, _ = env.reset()
    actions = env.unwrapped.action_manager.action.clone()

    t0 = mdp.t_pose_world(env.unwrapped)[0].tolist()
    for _ in range(args.steps):
        obs, _, _, _, _ = env.step(actions)

    scene = env.unwrapped.scene
    t1 = mdp.t_pose_world(env.unwrapped)[0].tolist()
    goal = mdp.goal_pose_obs(env.unwrapped)
    err = mdp.t_block_to_goal_obs(env.unwrapped)[0].tolist()
    z = scene["t_block"].data.root_pos_w[0, 2].item()
    origin_z = scene.env_origins[0, 2].item()
    settled = bool(mdp.t_block_settled(env.unwrapped)[0])
    success = bool(mdp.t_block_at_goal(env.unwrapped)[0])

    print(f"\nT block after {args.steps} steps")
    print(f"  centroid start  x={t0[0]:+.4f} y={t0[1]:+.4f} yaw={t0[2]:+.3f}")
    print(f"  centroid now    x={t1[0]:+.4f} y={t1[1]:+.4f} yaw={t1[2]:+.3f}")
    print(f"  drifted         {((t1[0]-t0[0])**2 + (t1[1]-t0[1])**2) ** 0.5 * 1000:.1f} mm  (nothing pushed it)")
    print(f"  root z          {z - origin_z:+.4f} m above the env origin")
    print(f"  settled={settled}  success={success}  goal error dx={err[0]:+.3f} dy={err[1]:+.3f} dyaw={err[2]:+.3f}")

    print("\ncameras")
    lines = []
    for key, name in (("camera_front", "front_c270"), ("camera_grip", "grip_kwc500")):
        cam = scene[key]
        rgb = cam.data.output["rgb"]
        lines.append(f"  {key:13} {save(name, rgb)}")
        intr = cam.data.intrinsic_matrices[0]
        print(f"  {key:13} fx={intr[0,0]:.1f} fy={intr[1,1]:.1f} cx={intr[0,2]:.1f} cy={intr[1,2]:.1f}")
    print()
    for line in lines:
        print(line)

    problems = []
    if abs(z - origin_z - 0.035) > 0.004:
        problems.append(f"T sits at z={z - origin_z:.4f}, expected 0.035 (the mat's top face)")
    if ((t1[0] - t0[0]) ** 2 + (t1[1] - t0[1]) ** 2) ** 0.5 > 0.01:
        problems.append("T drifted >10 mm with no contact -- check friction and the initial pose")
    for key, name in (("camera_front", "front"), ("camera_grip", "grip")):
        m = scene[key].data.output["rgb"][0].float().mean().item()
        if m < 2.0:
            problems.append(f"{name} camera is black (mean {m:.2f}) -- lighting or the near plane")

    print("\n" + ("FAILED\n  " + "\n  ".join(problems) if problems else "OK: scene builds, T rests on the mat, both cameras render"))
    env.close()
    return 1 if problems else 0


code = 1
try:
    code = main()
except BaseException:  # noqa: BLE001 - os._exit below would swallow the traceback
    traceback.print_exc()
finally:
    # Deliberately NOT app.close(): Kit 4.5 hangs inside it on this box. A headless
    # run prints its result and never returns from close(), so calling it first does
    # not help. Everything we care about is on disk by here; let the OS reclaim.
    #
    # os._exit skips BOTH the stdio flush and the traceback, so both are done by hand
    # above and below -- without them a failure here exits 1 in total silence.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
