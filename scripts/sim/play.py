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
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Agent101-So101-Push-T")
parser.add_argument("--steps", type=int, default=180, help="control steps to run (default 180 = 3 s)")
parser.add_argument("--pose", action="store_true",
                    help="open the window with physics OFF, to drag prims and read their transform")
parser.add_argument("--gui", action="store_true",
                    help="open the Isaac Sim window and hold it open after the checks")
parser.add_argument("--out", default=None, help="where to write renders (default sim/outputs)")
# Six lerobot joint angles in DEGREES, lerobot's own order, exactly as
# robot.get_observation() reports them. Comparing a sim render against a real
# frame is meaningless unless the arm is in the same pose in both -- otherwise
# the jaws are open in one and shut in the other and the camera gets the blame.
parser.add_argument("--joints", default=None,
                    help="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper "
                         "exactly as lerobot reports them (arm in degrees, gripper 0-100)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.gui = args.gui or args.pose   # posing needs a window to drag in
if args.joints:
    # The step loop drives the arm from the action manager, which would undo the
    # pose immediately. Rendering a given pose and running the settle test are
    # mutually exclusive.
    args.steps = 0
args.headless = not args.gui
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim_agent101  # noqa: E402,F401  (registers the envs)
from sim_agent101 import mdp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
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



def _report_mount_transform(path=None, quiet: bool = False) -> None:
    """Print the mount's TOTAL pose, ready to paste into assets/objects.py.

    The subtlety: KLIP_MOUNT_* is baked into klip_support.usd, so the geometry
    already carries it and the prim itself sits at identity. Dragging in the viewport
    changes the prim, so Isaac's property panel shows only the delta you added -- not
    the pose. Pasting the panel value straight in loses the bake.

    This composes the two, so what it prints is the whole thing.
    """
    try:
        import numpy as np
        import isaaclab.sim as sim_utils
        from pxr import UsdGeom

        from sim_agent101.assets.objects import KLIP_MOUNT_POS, KLIP_MOUNT_ROT
    except Exception:
        return
    prims = sim_utils.find_matching_prims("/World/envs/env_0/Robot/gripper/klip_support")
    if not prims:
        return

    def qmat(q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    m = UsdGeom.Xformable(prims[0]).GetLocalTransformation()
    t_p = np.array(m.ExtractTranslation())
    q = m.ExtractRotationQuat()
    i = q.GetImaginary()
    R_p = qmat([q.GetReal(), i[0], i[1], i[2]])
    R_b = qmat(np.array(KLIP_MOUNT_ROT))
    R = R_p @ R_b
    t = t_p + R_p @ np.array(KLIP_MOUNT_POS)
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    lines = [
        "TOTAL mount pose (prim drag composed with the baked pose):",
        f"    KLIP_MOUNT_POS = ({t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f})",
        f"    KLIP_MOUNT_ROLL_DEG = {roll:.3f}",
        f"    KLIP_MOUNT_PITCH_DEG = {pitch:.3f}",
        f"    KLIP_MOUNT_YAW_DEG = {yaw:.3f}",
    ]
    if np.allclose(t_p, 0, atol=1e-9):
        lines.append("  (prim not moved -- this is just the baked pose)")
    if path is not None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text("\n".join(lines) + "\n")
    if not quiet:
        print("\n" + "\n".join(lines))
        print("  paste into sim/sim_agent101/assets/objects.py, then ./robot sim-assets")
        if path is not None:
            print(f"  also written to {path}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    print(f"\nenv: {args.task}")
    print(f"  scene keys: {sorted(env.unwrapped.scene.keys())}")

    obs, _ = env.reset()
    actions = env.unwrapped.action_manager.action.clone()

    if args.joints:
        import math

        import torch

        from sim_agent101.assets.so101 import JOINT_NAMES
        from sim_agent101.kinematics import lerobot_to_urdf_deg

        order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        deg = [float(v) for v in args.joints.split(",")]
        if len(deg) != len(order):
            raise SystemExit(f"--joints needs {len(order)} values, got {len(deg)}")
        # via lerobot_to_urdf_deg: the sixth value is the gripper, which lerobot
        # reports as 0-100 percent, not degrees, whatever use_degrees says.
        want = {k: math.radians(v)
                for k, v in lerobot_to_urdf_deg(dict(zip(order, deg, strict=True))).items()}
        robot = env.unwrapped.scene["robot"]
        idx, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
        q = robot.data.joint_pos.clone()
        for i, nm in zip(idx, names, strict=True):
            q[:, i] = want[nm]
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        dt = env.unwrapped.physics_dt
        for _ in range(60):
            robot.set_joint_position_target(q)
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=True)
            env.unwrapped.scene.update(dt)
        got = robot.data.joint_pos[0]
        print("\nposed to the real arm's joints (deg):")
        for i, nm in zip(idx, names, strict=True):
            print(f"  {nm:12} want {math.degrees(want[nm]):+8.3f}   got {math.degrees(float(got[i])):+8.3f}")

    if args.pose:
        # Straight to the window. The checks below step physics 180 times, which in
        # pose mode is both pointless and a silent wait with nothing on screen.
        pose_file = OUT / "mount_pose.txt"
        print(f"\nPOSE MODE: physics is off. Drag klip_support, then close the window."
              f"\n  The pose is rewritten to {pose_file.relative_to(ROOT)} about once a"
              f"\n  second, so Ctrl-C is fine too -- Kit swallows SIGINT and hard-exits,"
              f"\n  which is why nothing used to print on the way out.")
        last = 0.0
        try:
            while app.is_running():
                app.update()
                now = time.time()
                if now - last > 1.0:
                    _report_mount_transform(path=pose_file, quiet=True)
                    last = now
        except KeyboardInterrupt:
            print("\ninterrupted")
        _report_mount_transform(path=pose_file)
        return 0

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

    if args.gui:
        # Keep stepping so the window stays live and the episode keeps resetting --
        # each reset re-randomises the T and the goal, which is the point of looking.
        # Without this the checks finish and the window vanishes before you see it.
        print("\nwindow open: close it (or Ctrl-C here) to quit."
              "\n  mouse: left-drag orbit, middle-drag pan, scroll zoom")
        stepping = True   # --pose returns before this; plain --gui runs physics
        try:
            while app.is_running():
                if stepping:
                    try:
                        env.step(actions)
                    except RuntimeError as e:
                        # Moving a prim that lives INSIDE the robot articulation (which
                        # klip_support does) makes PhysX rebuild it and invalidates the
                        # tensor view. Stepping again throws. Dropping to render-only
                        # keeps the window alive so the drag can be finished, instead of
                        # taking Isaac down mid-adjustment.
                        if "invalidated" not in str(e):
                            raise
                        stepping = False
                        print("\nphysics view invalidated (a prim moved) -- physics stopped, "
                              "window still live. Keep posing; the transform is printed on exit.")
                else:
                    app.update()
        except KeyboardInterrupt:
            print("\ninterrupted")
    env.close()
    return 1 if problems else 0


code = 1
try:
    code = main()
except KeyboardInterrupt:
    print("\ninterrupted")
except BaseException:  # noqa: BLE001 - os._exit below would swallow the traceback
    traceback.print_exc()
finally:
    # In the finally, not at the end of main(): Ctrl-C is the normal way to leave pose
    # mode, and reporting only on a clean return meant it printed nothing at all.
    try:
        _report_mount_transform()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
    # Deliberately NOT app.close(): Kit 4.5 hangs inside it on this box. A headless
    # run prints its result and never returns from close(), so calling it first does
    # not help. Everything we care about is on disk by here; let the OS reclaim.
    #
    # os._exit skips BOTH the stdio flush and the traceback, so both are done by hand
    # above and below -- without them a failure here exits 1 in total silence.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
