#!/usr/bin/env python3
"""Run a trained checkpoint: watch it, score it, or drive its goal by hand.

    ./robot sim-policy                        # newest so101_reach checkpoint, windowed
    ./robot sim-policy --headless --steps 600 # just the numbers
    ./robot sim-policy --goal manual          # ONE arm, and you move the goal
    ./robot sim-policy --goal auto            # the goal walks a slow circle itself
    ./robot sim-policy --checkpoint sim/outputs/rsl_rl/so101_reach/<run>/model_400.pt
    ./robot sim-policy --export               # also write policy.pt / policy.onnx
    ./robot sim-policy --real --goal manual   # ...with the REAL arm following along

WINDOWED by default, the opposite of sim-train: the point of this command is to look
at the thing. A reach policy can reach a low mean error by flinging the arm through
the target and settling behind it, and no scalar in the training log says so -- the
viewport does.

--goal picks WHO CHOOSES THE TARGET. The policy is identical in all three; only the
command term's buffer is fed differently.

    task    (default) the environment's own sampler, a new pose every 4 s. This is
            the setting to score a checkpoint in, because it is the distribution the
            policy was trained on.
    manual  one arm, and /World/GoalHandle -- an orange ball that IS the goal. Drag
            it with the translate gizmo while the sim runs, or use the keys:
                W / S   goal away from you / toward you      (world x)
                A / D   goal left / right                    (world y)
                Q / E   goal up / down                       (world z)
                R       goal back to the middle of the box
                L       clear a stuck key (Isaac Lab's own binding)
    auto    the same single arm, with the goal walking a circle by itself. Works
            headless, which is how the manual path gets tested without hands.

manual and auto force ONE environment, disable the 4 s resampling (or the goal jumps
away while you are still dragging it) and stretch the episode so a time-out cannot
reset the arm mid-demonstration. The goal is clamped to the box the policy was
trained on: drag past the edge and it stops at the edge, rather than asking for a
pose the arm was never taught and letting the checkpoint look broken when it is the
request that is wrong.

--real puts the real SO-101 in the loop. The scene gains a GHOST arm: the solid one
is what the policy commanded, the translucent cyan one is where the real robot
actually is, so the sim2real gap is a thing you watch rather than a number you infer.
Targets go out through scripts/robot/policy_bridge.py, which ./robot starts in the
container for you and which is the only process here that touches a motor.

NOTHING MOVES without --engage. --real alone runs the policy, draws both arms and
prints the per-joint difference while the bridge stays read-only. Add --engage and
the arm follows, rate-limited, watchdogged, and with torque released on the way out.
Before you do, know what this checkpoint does not know: it was trained with a
massless wrist (the mount and the webcam are visual-only in the sim), with no bench
collision geometry, and with five joints. It can ask for a pose that puts the gripper
through the table.

Prints mean and 90th-percentile position error over the run, in millimetres, which is
the number to compare against the arm's own repeatability before believing a
checkpoint is good enough to deploy.
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
parser.add_argument("--goal", choices=("task", "manual", "auto"), default="task",
                    help="who picks the target: the env's sampler, you, or a circle")
parser.add_argument("--speed", type=float, default=0.12, help="--goal manual: metres per second, for the keys")
parser.add_argument("--period", type=float, default=8.0, help="--goal auto: seconds per lap")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--steps", type=int, default=None, help="control steps to run (default: until the window closes)")
# --headless and --device come from AppLauncher below, so they are NOT declared here:
# declaring one twice is an argparse conflict at import, before anything runs.
parser.add_argument("--export", action="store_true", help="write policy.pt and policy.onnx beside the checkpoint")
parser.add_argument("--real-time", action="store_true", help="pace the loop at the task's control rate")
parser.add_argument("--real", action="store_true",
                    help="put the real arm in the loop: ghost arm from its encoders, targets to the bridge")
parser.add_argument("--engage", action="store_true",
                    help="with --real: actually command the arm. Without it nothing moves.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless and args.goal == "manual":
    # No window means no gizmo and no keyboard, and a demo that cannot move is a bug
    # report waiting to happen.
    print("[sim-policy] headless: nothing can drive the goal by hand, using --goal auto", flush=True)
    args.goal = "auto"
if args.headless and args.steps is None:
    # 20 s at 30 Hz. A headless run has no window to close, so without this it never
    # ends.
    args.steps = 600
if args.real:
    # The ghost arm lives in its own task, and a real arm can only follow one policy.
    args.task = "Agent101-So101-Reach-Real"
    args.num_envs = 1
    # Wall-clock pacing is not optional once hardware is following: free-running, the
    # simulator would hand the bridge targets faster than the bus can carry them.
    args.real_time = True
if args.goal == "manual" and not args.real_time:
    # Free-running, the sim goes as fast as the renderer allows and the arm answers a
    # drag in a fraction of the time it takes to make it. Pace it.
    args.real_time = True
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

import sim_agent101  # noqa: E402,F401  (registers the envs)
from sim_agent101 import mdp  # noqa: E402
from sim_agent101.tasks.reach_env_cfg import ARM_JOINTS, CMD_POS_X, CMD_POS_Y, CMD_POS_Z, EE_BODY  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGS = ROOT / "sim" / "outputs" / "rsl_rl"
HANDLE = "/World/GoalHandle"
IPC = ROOT / "sim" / "outputs" / "policy"
TARGETS, STATE = IPC / "targets.json", IPC / "state.json"

# The command term works in the ROBOT BASE frame, so the box and everything derived
# from it stay there too -- one conversion, at the edge, where the handle lives.
BOX = (CMD_POS_X, CMD_POS_Y, CMD_POS_Z)
CENTRE = np.array([sum(r) / 2 for r in BOX])
HALF = np.array([(hi - lo) / 2 for lo, hi in BOX])


class RealArm:
    """The real robot, as seen from inside the simulator.

    Two directions, both through files the bridge in the container also holds open:
    what the policy wants goes out to targets.json, where the arm actually is comes
    back in state.json and gets written straight onto the ghost articulation.

    Deliberately NOT a controller. It never decides to move anything -- it reports
    what the policy already commanded, and refuses to mark a frame engaged unless the
    operator asked for that on the command line.
    """

    def __init__(self, env, engage: bool) -> None:
        self.inner = env.unwrapped
        self.robot = self.inner.scene["robot"]
        self.ghost = self.inner.scene["ghost"]
        self.engage = engage
        # URDF joint names in the ARTICULATION's own order, which is not the order
        # anything else in this repo lists them in. Zipping the two by position is
        # how a policy ends up commanding the elbow with a wrist angle.
        self.names = list(self.robot.joint_names)
        self.ghost_names = list(self.ghost.joint_names)
        self.state_age = float("inf")
        self.measured: dict[str, float] = {}
        IPC.mkdir(parents=True, exist_ok=True)
        for stale in (TARGETS, STATE):
            stale.unlink(missing_ok=True)   # never inherit a previous session's frame
        print(f"[sim-policy] real arm: {'ENGAGED, the arm will move' if engage else 'read-only, nothing will move'}",
              flush=True)

    def publish_targets(self) -> None:
        """What the policy is asking the joints to do, in URDF degrees."""
        # joint_pos_target, not the raw action: the action term has already applied
        # its scale and its offset from the rest pose, and the target is what the
        # simulated servo is actually chasing.
        target = self.robot.data.joint_pos_target[0].cpu().numpy()
        # ARM_JOINTS, not every joint with a lerobot name. The Jaw has an actuator but
        # no action term, so its target sits at 0 rad rather than at the rest pose --
        # publishing it would command the real gripper shut on the first frame, which
        # is the sort of thing you only find out with a finger in the way.
        payload = {"t": time.time(), "engaged": bool(self.engage),
                   "joints_deg": {n: math.degrees(float(v)) for n, v in zip(self.names, target)
                                  if n in ARM_JOINTS}}
        tmp = TARGETS.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, TARGETS)            # atomic: the reader polls this at 30 Hz

    def pull_state(self) -> None:
        """Snap the ghost onto the real arm's measured angles."""
        try:
            d = json.loads(STATE.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.state_age = time.time() - float(d.get("t", 0.0))
        self.measured = dict(d.get("joints_deg", {}))
        if self.state_age > 1.0 or not self.measured:
            return
        pos = torch.tensor(
            [[math.radians(self.measured.get(n, 0.0)) for n in self.ghost_names]],
            dtype=torch.float32, device=self.inner.device)
        # write_joint_state_to_sim, not a drive target: the ghost has no gains and no
        # collisions, so it is placed, not simulated.
        self.ghost.write_joint_state_to_sim(pos, torch.zeros_like(pos))

    def report(self) -> str:
        if not self.measured:
            return "real arm: no state yet -- is the bridge running?"
        if self.state_age > 1.0:
            return f"real arm: state {self.state_age:.1f}s STALE"
        target = self.robot.data.joint_pos_target[0].cpu().numpy()
        worst, at = 0.0, ""
        for n, v in zip(self.names, target):
            if n not in self.measured:
                continue
            gap = abs(math.degrees(float(v)) - self.measured[n])
            if gap > worst:
                worst, at = gap, n
        return f"real arm: worst joint gap {worst:5.1f} deg at {at:<12s}"


class GoalControl:
    """Feeds the command term by hand: a draggable ball, the keys, or a circle.

    Only built for --goal manual/auto. With --goal task none of this exists and the
    environment samples its own targets exactly as it did during training.
    """

    def __init__(self, env, mode: str) -> None:
        self.mode = mode
        self.inner = env.unwrapped
        self.robot = self.inner.scene["robot"]
        self.term = self.inner.command_manager.get_term("ee_pose")
        self.device = self.inner.device
        self.dt = self.inner.step_dt
        self.goal_b = CENTRE.copy()
        self.recentre = [False]
        self.elapsed = 0.0

        # The handle. Visual only -- no rigid or collision props -- so it can be
        # dragged while physics runs without becoming part of the physics.
        ball = sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.35, 0.08), emissive_color=(0.45, 0.14, 0.02), roughness=0.5
            ),
        )
        ball.func(HANDLE, ball, translation=tuple(float(v) for v in self.to_world(self.goal_b)))
        prim = self.inner.sim.stage.GetPrimAtPath(HANDLE)
        self.xform = UsdGeom.Xformable(prim)

        # The prim's OWN translate op, not UsdGeom.XformCommonAPI. The spawner authors
        # an orient op beside the translate, and XformCommonAPI refuses a stack it did
        # not author -- "Could not determine xform ops for incompatible xformable". It
        # says so as a warning and does nothing, so the ball stays at spawn, the loop
        # reads it back, concludes a human moved it, and snaps the goal there. The
        # demo then flickers between the circle and the centre and looks like a policy
        # fault when it is a USD one.
        self.op = next((o for o in self.xform.GetOrderedXformOps()
                        if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None) or self.xform.AddTranslateOp()
        self.vec = Gf.Vec3d if self.op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f
        self.written_w = self.to_world(self.goal_b)

        self.keyboard = None
        if mode == "manual":
            from isaaclab.devices import Se3Keyboard

            # pos_sensitivity 1.0: advance() then returns a unit vector per held key
            # and the metres-per-second is ours to pick, rather than being buried in
            # the device's default scaling.
            self.keyboard = Se3Keyboard(pos_sensitivity=1.0)
            self.keyboard.add_callback("R", lambda: self.recentre.__setitem__(0, True))
            print(self.keyboard, flush=True)
            print(f"[sim-policy] drag {HANDLE}, or use W/S A/D Q/E, R to recentre.", flush=True)

    def to_world(self, p_b: np.ndarray) -> np.ndarray:
        t = torch.tensor(p_b, dtype=torch.float32, device=self.device).unsqueeze(0)
        p_w, _ = combine_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, t)
        return p_w[0].cpu().numpy()

    def to_base(self, p_w: np.ndarray) -> np.ndarray:
        t = torch.tensor(p_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        p_b, _ = subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, t)
        return p_b[0].cpu().numpy()

    def step(self) -> None:
        """Decide where the goal is, show it, and write it into the command."""
        # A drag wins over the keys: if the ball is not where we last put it, a human
        # moved it.
        current_w = np.array(self.xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
        if self.recentre[0]:
            self.goal_b, self.recentre[0] = CENTRE.copy(), False
        elif np.linalg.norm(current_w - self.written_w) > 1e-4:
            self.goal_b = np.clip(self.to_base(current_w), CENTRE - HALF, CENTRE + HALF)
        elif self.mode == "auto":
            a = 2 * math.pi * self.elapsed / args.period
            self.goal_b = CENTRE + 0.7 * HALF * np.array([math.cos(a), math.sin(a), math.sin(2 * a)])
        elif self.keyboard is not None:
            delta, _ = self.keyboard.advance()          # nonzero while a key is held
            if np.any(delta[:3]):
                moved = self.to_world(self.goal_b) + delta[:3] * args.speed * self.dt
                self.goal_b = np.clip(self.to_base(moved), CENTRE - HALF, CENTRE + HALF)
        self.elapsed += self.dt

        # Put the ball exactly where the command is, so what you see is what the
        # policy is being asked for -- including after a clamp.
        self.written_w = self.to_world(self.goal_b)
        self.op.Set(self.vec(*(float(v) for v in self.written_w)))

        # Overwrite the command. The observation is rebuilt inside env.step() after
        # this, so the policy acts on the new goal one control step later -- 33 ms,
        # below anything a hand notices.
        self.term.pose_command_b[:, :3] = torch.tensor(self.goal_b, dtype=torch.float32, device=self.device)
        self.term.pose_command_b[:, 3] = 1.0
        self.term.pose_command_b[:, 4:] = 0.0


def main() -> None:
    num_envs = 1 if args.goal != "task" else args.num_envs
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    if args.goal != "task":
        # The two changes that turn an episodic task into a demonstration.
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
    print(f"[sim-policy] {ckpt.relative_to(ROOT) if ckpt.is_relative_to(ROOT) else ckpt}", flush=True)

    if args.export:
        # runner.alg.POLICY, not runner.alg.actor_critic -- rsl_rl renamed it, and
        # Isaac Lab's own play.py still reaches for the old name, so copying that
        # script is how you get an AttributeError three minutes into a run.
        out = ckpt.parent / "exported"
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=str(out), filename="policy.pt")
        export_policy_as_onnx(runner.alg.policy, normalizer=runner.obs_normalizer,
                              path=str(out), filename="policy.onnx")
        print(f"[sim-policy] exported to {out}", flush=True)

    goal = GoalControl(env, args.goal) if args.goal != "task" else None
    real = RealArm(env, args.engage) if args.real else None

    # Scored with the task's own reward term, so "how far off is it" here means
    # exactly what it meant during training.
    asset_cfg = SceneEntityCfg("robot", body_names=[EE_BODY])
    asset_cfg.resolve(env.unwrapped.scene)
    errors = []

    dt = env.unwrapped.step_dt      # the CONTROL step, not sim.dt: one policy action
    obs, _ = env.get_observations()
    step = 0
    try:
        while app.is_running():
            started = time.time()
            if goal is not None:
                goal.step()
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
                errors.append(mdp.position_command_error(env.unwrapped, "ee_pose", asset_cfg).mean().item())
                if real is not None:
                    # INSIDE the inference_mode block, and it has to be: the ghost is
                    # moved with write_joint_state_to_sim, which updates the
                    # articulation's joint_acc in place. Those buffers were created
                    # under inference mode by the step above, and torch refuses an
                    # in-place write to an inference tensor from outside it.
                    #
                    # After the step, so the target published is the one the policy
                    # just produced rather than the previous frame's.
                    real.publish_targets()
                    real.pull_state()
            step += 1
            if step % 15 == 0 and (goal is not None or real is not None):
                line = ""
                if goal is not None:
                    line += (f"goal (base) {goal.goal_b[0]:+.3f} {goal.goal_b[1]:+.3f} "
                             f"{goal.goal_b[2]:+.3f} m   error {errors[-1] * 1000:6.1f} mm")
                if real is not None:
                    line += ("   " if line else "") + real.report()
                print(f"\r[sim-policy] {line}", end="", flush=True)
            if args.steps is not None and step >= args.steps:
                break
            if args.real_time:
                remaining = dt - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass

    if errors:
        e = torch.tensor(errors) * 1000.0
        print(f"\n[sim-policy] {step} steps, {env_cfg.scene.num_envs} envs, --goal {args.goal}", flush=True)
        print(f"[sim-policy] position error  mean {e.mean():.1f} mm   "
              f"p90 {e.quantile(0.9):.1f} mm   worst {e.max():.1f} mm", flush=True)
        # With --goal task the target moves every 4 s, so the first steps after each
        # resample are travel, not error: a mean well above the p90 means it is
        # chasing, not that it cannot get there. With manual or auto the target is
        # moving continuously, so the whole distribution shifts up -- compare those
        # numbers against each other, not against a task run.
    env.close()


if __name__ == "__main__":
    main()
    # Every print above passes flush=True, and that is not decoration: Kit tears the
    # process down inside app.close() without giving Python's stdout buffer a chance
    # to drain. Redirect this script to a file without flushing and the Kit log
    # arrives (carb writes at the fd) while every line this script printed is gone.
    app.close()
