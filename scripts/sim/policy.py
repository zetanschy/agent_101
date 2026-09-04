#!/usr/bin/env python3
"""Run a trained checkpoint and report how well it actually tracks.

    ./robot sim-policy                        # newest so101_reach checkpoint, windowed
    ./robot sim-policy --headless --steps 600 # just the numbers
    ./robot sim-policy --checkpoint sim/outputs/rsl_rl/so101_reach/<run>/model_400.pt
    ./robot sim-policy --export               # also write policy.pt / policy.onnx

WINDOWED by default, the opposite of sim-train: the point of this command is to
look at the thing. A reach policy can reach a low mean error by flinging the arm
through the target and settling behind it, and no scalar in the training log says
so -- the viewport does.

Prints mean and 90th-percentile position error over the run, in millimetres, which
is the number to compare against the arm's own repeatability before believing a
checkpoint is good enough to deploy.
"""

import argparse
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Agent101-So101-Reach-Play")
parser.add_argument("--checkpoint", default=None, help="path to a .pt (default: newest run, newest checkpoint)")
parser.add_argument("--experiment", default="so101_reach", help="which sim/outputs/rsl_rl/<dir> to search")
parser.add_argument("--load-run", default=".*", help="run directory regex, newest match wins")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--steps", type=int, default=None, help="control steps to run (default: until the window closes)")
# --headless and --device come from AppLauncher below, so they are NOT declared here:
# declaring one twice is an argparse conflict at import, before anything runs.
parser.add_argument("--export", action="store_true", help="write policy.pt and policy.onnx beside the checkpoint")
parser.add_argument("--real-time", action="store_true", help="pace the loop at the task's control rate")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless and args.steps is None:
    # 20 s at 30 Hz. A headless run has no window to close, so without this it
    # never ends.
    args.steps = 600
app = AppLauncher(args).app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402

import sim_agent101  # noqa: E402,F401  (registers the envs)
from sim_agent101 import mdp  # noqa: E402
from sim_agent101.tasks.reach_env_cfg import EE_BODY  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGS = ROOT / "sim" / "outputs" / "rsl_rl"


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    env_cfg.sim.device = args.device

    ckpt = (pathlib.Path(args.checkpoint) if args.checkpoint
            else pathlib.Path(get_checkpoint_path(str(LOGS / args.experiment), args.load_run, "model_.*.pt")))
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")

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

    # Scored with the task's own reward term, so "how far off is it" here means
    # exactly what it meant during training.
    asset_cfg = SceneEntityCfg("robot", body_names=[EE_BODY])
    asset_cfg.resolve(env.unwrapped.scene)
    errors = []

    dt = env.unwrapped.step_dt      # the CONTROL step, not sim.dt: one policy action
    obs, _ = env.get_observations()
    step = 0
    while app.is_running():
        started = time.time()
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
            errors.append(mdp.position_command_error(env.unwrapped, "ee_pose", asset_cfg).mean().item())
        step += 1
        if args.steps is not None and step >= args.steps:
            break
        if args.real_time:
            remaining = dt - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)

    if errors:
        e = torch.tensor(errors) * 1000.0
        print(f"[sim-policy] {step} steps, {env_cfg.scene.num_envs} envs", flush=True)
        print(f"[sim-policy] position error  mean {e.mean():.1f} mm   "
              f"p90 {e.quantile(0.9):.1f} mm   worst {e.max():.1f} mm", flush=True)
        # The commanded target moves every 4 s, so the first steps after each
        # resample are travel, not error. A mean well above the p90 of a settled
        # policy usually means it is chasing, not that it cannot get there.
    env.close()


if __name__ == "__main__":
    main()
    # Every print above passes flush=True, and that is not decoration: Kit tears the
    # process down inside app.close() without giving Python's stdout buffer a chance
    # to drain. Redirect this script to a file without flushing and the Kit log
    # arrives (carb writes at the fd) while every line this script printed is gone.
    app.close()
