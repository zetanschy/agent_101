#!/usr/bin/env python3
"""Train a policy on one of this package's RL tasks, with rsl_rl PPO.

    ./robot sim-train                                  # reach, headless, 4096 envs
    ./robot sim-train --max-iterations 200             # a short run
    ./robot sim-train --task Agent101-So101-Reach-DR   # randomized actuator gains
    ./robot sim-train --gui --num-envs 64              # watch it learn (much slower)
    ./robot sim-train --resume                         # continue the newest run

Deliberately NOT Isaac Lab's own scripts/reinforcement_learning/rsl_rl/train.py.
That one is a hydra entry point: it rewrites sys.argv, imports a cli_args module
that has to sit beside it, and reads task configs out of a hydra store. Everything
it does that matters here is the twenty lines below, and doing it directly means
./robot sim-train takes the same kind of flags as every other command in this repo.

Logs and checkpoints go to sim/outputs/rsl_rl/<experiment>/<timestamp>/:

    tensorboard --logdir sim/outputs/rsl_rl

WANDB IS ON when credentials exist, the same rule ./robot train follows: a
WANDB_API_KEY in .env.local or a `wandb login` in ~/.netrc, project from
WANDB_PROJECT. No credentials and it says so and logs to tensorboard alone, rather
than failing a training run over a logger. --no-wandb and --wandb-offline override.

HEADLESS BY DEFAULT, and that is not a detail -- rendering 4096 arms is most of the
cost of training them. --gui is for looking at a scene, not for training in.
"""

import argparse
import os
import pathlib
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Agent101-So101-Reach")
parser.add_argument("--num-envs", type=int, default=None, help="override the task's env count")
parser.add_argument("--max-iterations", type=int, default=None, help="override the agent's iteration count")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--gui", action="store_true", help="open the Isaac Sim window (slow; for looking, not training)")
parser.add_argument("--resume", action="store_true", help="load the newest checkpoint and keep going")
parser.add_argument("--load-run", default=".*", help="with --resume: which run directory (regex, newest match wins)")
parser.add_argument("--checkpoint", default="model_.*.pt", help="with --resume: which checkpoint file (regex)")
parser.add_argument("--name", default=None, help="name this run; suffixes the log dir and names the wandb run")
parser.add_argument("--wandb", action="store_true", help="force wandb on (fails fast if there are no credentials)")
parser.add_argument("--no-wandb", action="store_true", help="force wandb off; tensorboard only")
parser.add_argument("--wandb-offline", action="store_true", help="log to disk for a later `wandb sync`")
parser.add_argument("--wandb-project", default=None, help="default: $WANDB_PROJECT, else agent_101")
parser.add_argument("--wandb-entity", default=None, help="team/org, if not your personal account")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402

import sim_agent101  # noqa: E402,F401  (registers the envs)

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Not ROOT/outputs: that directory is root-owned here (Docker made it) and the sim
# commands run natively as the user.
LOGS = ROOT / "sim" / "outputs" / "rsl_rl"

# TF32 on. These are 64x64 MLPs, so this is nearly free either way, but it is what
# Isaac Lab's own trainer does and there is no reason for the two to differ.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _have_wandb_creds() -> bool:
    """A key in the environment, or a `wandb login` netrc entry. Same test as train.sh.

    .env.local is where the key lives and scripts/sim.sh exports it, so by the time
    this runs it is an ordinary environment variable.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return "api.wandb.ai" in (pathlib.Path.home() / ".netrc").read_text(errors="ignore")
    except OSError:
        return False


def configure_logging(agent_cfg) -> str:
    """Point the runner at wandb or tensorboard. Returns what to tell the user.

    ON BY DEFAULT when credentials exist, off (with a note) when they do not --
    ./robot train behaves the same way, and the reason is that a missing logger
    credential should not cost you a training run you have already started.
    """
    if args.no_wandb:
        agent_cfg.logger = "tensorboard"
        return "wandb: off (--no-wandb)"

    offline = args.wandb_offline
    if not (args.wandb or offline or _have_wandb_creds()):
        agent_cfg.logger = "tensorboard"
        return ("wandb: off (no credentials). Either\n"
                "  echo 'WANDB_API_KEY=<key>' >> .env.local   (from wandb.ai/authorize)\n"
                "  wandb login                                (caches ~/.netrc)\n"
                "or pass --wandb-offline to log to disk.")
    if not offline and not _have_wandb_creds():
        # Only reachable via an explicit --wandb: asking for it and not having it is
        # worth failing on, before an hour of training goes somewhere it cannot log.
        raise SystemExit("wandb: --wandb given but no WANDB_API_KEY and no netrc entry")

    agent_cfg.logger = "wandb"
    agent_cfg.wandb_project = (args.wandb_project or os.environ.get("WANDB_PROJECT") or "agent_101")
    if offline:
        # Runs land in ./wandb/ for a later `wandb sync`. No network, no credentials.
        os.environ["WANDB_MODE"] = "offline"
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    if entity:
        # WANDB_USERNAME, not WANDB_ENTITY: rsl_rl's WandbSummaryWriter reads the
        # former and passes it to wandb.init(entity=...). Set only the variable this
        # repo documents and the run lands in your personal account instead of the
        # team, silently and with no error to notice.
        os.environ["WANDB_USERNAME"] = entity
    return (f"wandb: project={agent_cfg.wandb_project}"
            f"{f' entity={entity}' if entity else ''}{' (offline)' if offline else ''}")


def finish_logging(agent_cfg) -> None:
    """Close the wandb run, explicitly.

    The same failure as the flush=True on every print below, and it costs more: Kit
    ends the process inside app.close() without running atexit handlers, and wandb's
    atexit hook is what finalizes a run. Without this the run directory appears, the
    config is written, and the transaction log stays at ZERO BYTES -- an offline run
    that syncs nothing and an online run that shows up crashed with no metrics. It
    looks like wandb is working right up until you go and read the run.
    """
    if getattr(agent_cfg, "logger", None) != "wandb":
        return
    import wandb

    if wandb.run is not None:
        wandb.finish()


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.name:
        agent_cfg.run_name = args.name
    logging_note = configure_logging(agent_cfg)
    # The env seed has to be set from the agent's, not left to default: some of the
    # randomization happens while the environment is being built, before anything
    # the runner does could seed it.
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device

    run_root = LOGS / agent_cfg.experiment_name
    run_dir = run_root / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                          + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""))

    # Resolved BEFORE the new run directory is created, or ".*" matches the empty
    # directory this run is about to make and finds no checkpoint in it.
    resume_path = None
    if args.resume:
        resume_path = get_checkpoint_path(str(run_root), args.load_run, args.checkpoint)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path:
        print(f"[sim-train] resuming from {resume_path}", flush=True)
        runner.load(resume_path)

    # The exact configs this run used, next to its checkpoints. A checkpoint whose
    # env cfg you cannot reconstruct is a checkpoint you cannot evaluate.
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)
    dump_pickle(str(run_dir / "params" / "env.pkl"), env_cfg)
    dump_pickle(str(run_dir / "params" / "agent.pkl"), agent_cfg)

    print(f"[sim-train] task {args.task}", flush=True)
    print(f"[sim-train] {env_cfg.scene.num_envs} envs x {agent_cfg.num_steps_per_env} steps "
          f"= {env_cfg.scene.num_envs * agent_cfg.num_steps_per_env} transitions/iteration, "
          f"{agent_cfg.max_iterations} iterations", flush=True)
    print(f"[sim-train] logging to {run_dir.relative_to(ROOT)}", flush=True)
    # The wandb run takes its name from the log directory's basename, so --name shows
    # up in both places or neither.
    print(f"[sim-train] {logging_note}", flush=True)
    print_dict(agent_cfg.to_dict(), nesting=1)

    # finally, so a Ctrl-C partway through an hour of training still leaves a synced
    # run behind rather than an empty one.
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()
        finish_logging(agent_cfg)


if __name__ == "__main__":
    main()
    # Every print above passes flush=True, and that is not decoration: Kit tears the
    # process down inside app.close() without giving Python's stdout buffer a chance
    # to drain. Redirect this script to a file without flushing and the Kit log
    # arrives (carb writes at the fd) while every line this script printed is gone.
    app.close()
