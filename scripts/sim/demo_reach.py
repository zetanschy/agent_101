#!/usr/bin/env python3
"""One arm, one goal, and you move the goal: a demo for a trained reach checkpoint.

    ./robot sim-demo                             # window opens, you drive the goal
    ./robot sim-demo --auto                      # the goal walks a slow circle itself
    ./robot sim-demo --auto --headless --steps 600   # the same, as numbers

TWO WAYS TO MOVE IT, and they work at the same time:

  * DRAG. /World/GoalHandle is an orange ball, and it IS the goal -- pick it in the
    stage tree or the viewport and drag it with the translate gizmo while the sim
    runs. The policy chases wherever you leave it.
  * KEYS, with the viewport focused:
        W / S   goal away from you / toward you      (world x)
        A / D   goal left / right                    (world y)
        Q / E   goal up / down                       (world z)
        R       goal back to the middle of the box
        L       clear a stuck key (Isaac Lab's own binding)

The goal is CLAMPED to the box the policy was trained on (REACH_* in
tasks/reach_env_cfg.py). Drag the ball past the edge and it stops at the edge rather
than wandering somewhere the arm was never taught to go and looking broken.

Nothing about the policy changes here. This overrides the command term's buffer
instead of letting it resample every 4 s -- the checkpoint is the same one
./robot sim-policy scores.
"""

import argparse
import math
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Agent101-So101-Reach-Play")
parser.add_argument("--checkpoint", default=None, help="path to a .pt (default: newest run, newest checkpoint)")
parser.add_argument("--experiment", default="so101_reach", help="which sim/outputs/rsl_rl/<dir> to search")
parser.add_argument("--load-run", default=".*", help="run directory regex, newest match wins")
parser.add_argument("--auto", action="store_true", help="move the goal on a slow circle instead of by hand")
parser.add_argument("--speed", type=float, default=0.12, help="metres per second, for the keys (default 0.12)")
parser.add_argument("--period", type=float, default=8.0, help="seconds per lap, for --auto")
parser.add_argument("--steps", type=int, default=None, help="stop after this many control steps")
# --headless and --device come from AppLauncher; declaring either twice is an
# argparse conflict at import. Add them BEFORE parsing, or the parse below rejects
# the very flags this script documents.
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless and not args.auto:
    # Nothing can move the goal without a window, and a demo that never moves is a
    # bug report waiting to happen.
    print("[sim-demo] headless: no keyboard and no gizmo, so --auto is implied", flush=True)
    args.auto = True
    args.steps = args.steps or 600
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

import sim_agent101  # noqa: E402,F401  (registers the envs)
from sim_agent101.tasks.reach_env_cfg import CMD_POS_X, CMD_POS_Y, CMD_POS_Z  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGS = ROOT / "sim" / "outputs" / "rsl_rl"
HANDLE = "/World/GoalHandle"

# The command term works in the ROBOT BASE frame, so the box and everything derived
# from it are kept there too -- one conversion, at the edge, where the handle lives.
BOX = (CMD_POS_X, CMD_POS_Y, CMD_POS_Z)
CENTRE = np.array([sum(r) / 2 for r in BOX])
HALF = np.array([(hi - lo) / 2 for lo, hi in BOX])


def clamp_b(p: np.ndarray) -> np.ndarray:
    return np.clip(p, CENTRE - HALF, CENTRE + HALF)


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # The two changes that turn an episodic task into a demo. Without the first the
    # goal jumps somewhere random every 4 s while you are still dragging it; without
    # the second the episode times out and resets the arm mid-demonstration.
    env_cfg.commands.ee_pose.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = 1.0e6
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    env_cfg.sim.device = args.device

    ckpt = (pathlib.Path(args.checkpoint) if args.checkpoint
            else pathlib.Path(get_checkpoint_path(str(LOGS / args.experiment), args.load_run, "model_.*.pt")))
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}. Train one first: ./robot sim-train")

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[sim-demo] {ckpt.relative_to(ROOT) if ckpt.is_relative_to(ROOT) else ckpt}", flush=True)

    inner = env.unwrapped
    robot = inner.scene["robot"]
    term = inner.command_manager.get_term("ee_pose")
    dt = inner.step_dt
    device = inner.device

    def to_world(p_b: np.ndarray) -> np.ndarray:
        t = torch.tensor(p_b, dtype=torch.float32, device=device).unsqueeze(0)
        p_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, t)
        return p_w[0].cpu().numpy()

    def to_base(p_w: np.ndarray) -> np.ndarray:
        t = torch.tensor(p_w, dtype=torch.float32, device=device).unsqueeze(0)
        p_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, t)
        return p_b[0].cpu().numpy()

    # The handle. Visual only -- no rigid or collision props -- so it can be dragged
    # while physics runs without becoming part of the physics.
    ball = sim_utils.SphereCfg(
        radius=0.015,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.95, 0.35, 0.08), emissive_color=(0.45, 0.14, 0.02), roughness=0.5
        ),
    )
    goal_b = CENTRE.copy()
    ball.func(HANDLE, ball, translation=tuple(float(v) for v in to_world(goal_b)))
    handle = inner.sim.stage.GetPrimAtPath(HANDLE)
    xform = UsdGeom.Xformable(handle)

    # The prim's OWN translate op, not UsdGeom.XformCommonAPI. The spawner writes an
    # orient op alongside the translate, and XformCommonAPI refuses any stack it did
    # not author -- "Could not determine xform ops for incompatible xformable". It
    # says so as a warning and does nothing, so the ball stays at spawn, the loop
    # reads it back, decides a human moved it, and snaps the goal there. The demo
    # then flickers between the circle and the centre and looks like a policy fault.
    translate_op = next((op for op in xform.GetOrderedXformOps()
                         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    _vec = Gf.Vec3d if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f

    def read_handle_w() -> np.ndarray:
        t = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return np.array([t[0], t[1], t[2]])

    def write_handle_w(p_w: np.ndarray) -> None:
        translate_op.Set(_vec(*(float(v) for v in p_w)))

    keyboard = None
    recentre = [False]
    if not args.headless and not args.auto:
        from isaaclab.devices import Se3Keyboard

        # pos_sensitivity 1.0: advance() then returns a unit vector per held key and
        # the metres-per-second is ours to choose, rather than being buried in the
        # device's default scaling.
        keyboard = Se3Keyboard(pos_sensitivity=1.0)
        # R recentres. The callback fires on Kit's thread, so it sets a flag and the
        # loop acts on it, rather than reaching into the command buffer from there.
        keyboard.add_callback("R", lambda: recentre.__setitem__(0, True))
        print(keyboard, flush=True)

    print("[sim-demo] drag /World/GoalHandle, or use W/S A/D Q/E. Ctrl-C to stop.", flush=True)
    written_w = to_world(goal_b)
    obs, _ = env.get_observations()
    step = 0
    try:
        while app.is_running():
            # 1. where does the user want the goal? A drag wins over the keys: if the
            #    ball is not where we last put it, a human moved it.
            current_w = read_handle_w()
            if recentre[0]:
                goal_b, recentre[0] = CENTRE.copy(), False
            elif np.linalg.norm(current_w - written_w) > 1e-4:
                goal_b = clamp_b(to_base(current_w))
            elif args.auto:
                a = 2 * math.pi * (step * dt) / args.period
                goal_b = CENTRE + 0.7 * HALF * np.array([math.cos(a), math.sin(a), math.sin(2 * a)])
            elif keyboard is not None:
                delta, _ = keyboard.advance()          # nonzero while a key is held
                if np.any(delta[:3]):
                    goal_b = clamp_b(to_base(to_world(goal_b) + delta[:3] * args.speed * dt))

            # 2. put the ball exactly where the command is, so what you see is what
            #    the policy is being asked for -- including after a clamp.
            written_w = to_world(goal_b)
            write_handle_w(written_w)

            # 3. overwrite the command. The observation is rebuilt inside env.step()
            #    after this, so the policy acts on the new goal one control step later
            #    -- 33 ms, which is below anything a hand can notice.
            term.pose_command_b[:, :3] = torch.tensor(goal_b, dtype=torch.float32, device=device)
            term.pose_command_b[:, 3] = 1.0
            term.pose_command_b[:, 4:] = 0.0

            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))

            step += 1
            if step % 15 == 0:
                err = term.metrics["position_error"][0].item() * 1000.0
                print(f"\r[sim-demo] goal (base) {goal_b[0]:+.3f} {goal_b[1]:+.3f} {goal_b[2]:+.3f} m"
                      f"   error {err:6.1f} mm", end="", flush=True)
            if args.steps is not None and step >= args.steps:
                break
    except KeyboardInterrupt:
        pass
    print(flush=True)
    env.close()


if __name__ == "__main__":
    main()
    # flush=True on every print above, and this comment for the same reason as in
    # train.py: Kit ends the process in here without draining python's stdout.
    app.close()
