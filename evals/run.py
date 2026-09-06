"""Run one Inspect Robots eval on the SO-ARM101: an LLM agent, or a lerobot VLA.

    ./robot eval --policy agent  --model openai/gpt-6-astra    # the LLM agent
    ./robot eval-openpi --policy openpi                        # the working pi0.5
    ./robot eval --policy agent --dry-run                      # no motion, no arm

The openpi policy runs in the OTHER image: openpi pins jax and its own lerobot, so it
has a container of its own (docker-compose.openpi.yml), which is why it has its own
./robot verb. The task, the rig config and this runner are shared across both.

Both policies get the SAME Task object out of evals/tasks.py and the SAME embodiment
out of evals/rig.py, which is the only reason the two scores can be put beside each
other. Everything that differs between the two runs is the policy.

THE ARM MOVES. This drives the real follower through lerobot at up to 30 Hz, with the
gripper live, on whatever instruction the task carries. Three layers stand between a
model's output and the motors, and they are not redundant:

  * ClampApprover, from Inspect Robots, clips the action to the declared space before
    the embodiment sees it.
  * The embodiment's own clamp, inside step(), clips again to SOArmConfig.joint_low /
    joint_high -- which evals/rig.py computes from this arm's calibration file rather
    than accepting the +/-180 degree placeholders.
  * lerobot's max_relative_target, a per-step slew limit in motor counts, which turns
    a step change into a ramp.

--dry-run exercises the whole chain -- registry, compatibility check, task, policy,
API credentials -- against the dependency-free CubePick mock world instead of the arm.
Run it once before every hardware session; it costs nothing and catches the failures
that are annoying to discover with a powered arm in front of you.
"""

from __future__ import annotations

import argparse
import os
import sys

from inspect_robots import eval as robot_eval
from inspect_robots.approver import ClampApprover

from evals import rig, tasks
from evals.openpi_policy import DEFAULT_CHECKPOINT as OPENPI_CHECKPOINT

# The lerobot-format checkpoint webui/app.py names as its default. NOT the model the
# comparison is against -- see --policy openpi, which loads the orbax checkpoint that
# actually works on this bench.
LEROBOT_CHECKPOINT = "zetanschy/pi05_lora_cap_tu_cup"


def _grade(record, scene) -> None:
  """Ask the operator for the verdict, before the scorer reads it.

  operator_scorer() scores `record.operator_judgement`; this hook is what puts a
  value there. One without the other silently scores every trial zero, so they are
  wired together here rather than left to the caller.
  """
  setup = (scene.metadata or {}).get("operator_setup", "")
  print(f"\n--- trial finished: {scene.id} ({setup}) ---")
  record.operator_judgement = input("Outcome? [y/n/partial/skip]: ").strip()
  record.operator_note = input("Note (optional): ").strip()


def build_policy(args):
  """The policy under test. Both arms of the comparison are constructed here."""
  if args.policy == "openpi":
    from evals.openpi_policy import OpenPiPolicy

    return OpenPiPolicy(args.checkpoint or OPENPI_CHECKPOINT,
                        cameras=tuple(args.cameras),
                        replan_interval=args.actions)

  if args.policy == "agent":
    from inspect_robots_agent.policy import LLMAgentPolicy

    # Model strings are OpenRouter-style provider/model. openai/* resolves against
    # OPENAI_API_KEY directly; anything unknown falls through to OPENROUTER_API_KEY.
    return LLMAgentPolicy(model=args.model, effort=args.effort)

    # NOTE: the agent's whole tool surface is built from the embodiment's declared
    # spaces at bind time, and EmbodimentInfo.docs goes into its system prompt
    # verbatim. What inspect-robots-so101 declares is what the model gets told about
    # this arm; there is nothing to write here to improve its prompt.

  from inspect_robots_so101 import LeRobotPolicy, LeRobotPolicyConfig

  return LeRobotPolicy(
    LeRobotPolicyConfig(
      pretrained_path=args.checkpoint or LEROBOT_CHECKPOINT,
      policy_type=args.policy_type,
      device=args.device,
      cameras=tuple(args.cameras),
      # MUST match the embodiment: the embodiment commands policy output verbatim
      # after the clamp, and Inspect Robots' compat check compares state keys, not
      # units, so a mismatch here is silent and drives the arm in the wrong units.
      use_degrees=True,
    )
  )


def use_degrees(args) -> bool:
  """Joint units for this run, decided by the policy under test.

  The openpi pi0.5 checkpoints on this bench are normalized (+/-100); the agent reads
  whatever bounds the embodiment declares, so degrees is the friendlier choice there.
  """
  if args.policy == "openpi":
    from evals.openpi_policy import USE_DEGREES

    return USE_DEGREES
  return True


def build_embodiment(args):
  """The arm, or the mock world under --dry-run."""
  if args.dry_run:
    if args.policy in ("lerobot", "openpi"):
      # CubePick is a 2-D eef-delta world and the lerobot policy declares the 6-D
      # joint contract, so pairing them fails the compatibility check for a reason
      # that says nothing about this rig. The adapter ships the right tool for
      # checking that side without motion.
      raise SystemExit(
        f"--dry-run pairs the 2-D CubePick mock world, which the 6-D {args.policy} "
        "policy cannot bind to. To check that path without moving the arm, run:\n"
        "    ./robot shell -c 'inspect-robots-so101-preflight --dry-run'"
      )
    from inspect_robots.mock import CubePickEmbodiment

    return CubePickEmbodiment()
  from inspect_robots_so101 import SOArmEmbodiment

  # The units follow the POLICY, because they are a property of what it was trained
  # on and not a preference. Getting this wrong is silent: see evals/openpi_policy.py.
  # Settling waits for the arm to arrive before observing. Right for an LLM agent
  # taking one action at a time; wrong for a chunked VLA, whose cadence it changes --
  # which is why inspect-robots-so101 ships it off. Keyed off the policy, like units.
  # For openpi, match webui/openpi_worker.py exactly: no settling, and NO SLEW LIMIT.
  # lerobot's max_relative_target clamps against the measured position, so under
  # grasping load it turns a lagging servo into a creeping command -- see rig.py. No
  # slew limit also means no homing (SOArmConfig couples them), so home by hand.
  is_vla = args.policy == "openpi"
  return SOArmEmbodiment(
    rig.so_arm_config(
      cameras=tuple(args.cameras),
      use_degrees=use_degrees(args),
      settle_tolerance=None if is_vla else 2.0,
      slew_limit=None if is_vla else 10.0,
    )
  )


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--policy", choices=("agent", "lerobot", "openpi"), default="agent")
  p.add_argument("--model", default=os.environ.get("INSPECT_ROBOTS_MODEL", "openai/gpt-6-astra"),
                 help="agent only: provider/model (default openai/gpt-6-astra)")
  p.add_argument("--effort", default="medium", help="agent only: reasoning effort")
  p.add_argument("--checkpoint", default=None,
                 help="lerobot: Hub id or path; openpi: orbax checkpoint dir")
  p.add_argument("--policy-type", default="pi05", help="lerobot only: policy class")
  p.add_argument("--device", default="cuda")
  p.add_argument("--cameras", nargs="+", default=["front", "grip"])
  p.add_argument("--actions", type=int, default=15,
                 help="openpi only: actions executed per chunk before re-planning "
                      "(webui/openpi_worker.py's default, and what it was tuned with)")
  p.add_argument("--layouts", type=int, default=5)
  p.add_argument("--epochs", type=int, default=1)
  p.add_argument("--seconds", type=float, default=tasks.DEFAULT_SECONDS)
  p.add_argument("--log-dir", default="outputs/evals")
  p.add_argument("--dry-run", action="store_true",
                 help="run against the CubePick mock world; the arm is never opened")
  args = p.parse_args(argv)

  task = tasks.pick_and_place(
    layouts=args.layouts, seconds=args.seconds, epochs=args.epochs
  )
  policy = build_policy(args)
  embodiment = build_embodiment(args)

  print(f"task     : {task.name}  ({len(task.scenes)} scenes x {args.epochs} epochs)")
  shown = args.model if args.policy == "agent" else (
    args.checkpoint or (OPENPI_CHECKPOINT if args.policy == "openpi" else LEROBOT_CHECKPOINT)
  )
  print(f"policy   : {args.policy} ({shown})")
  print(f"units    : {'degrees' if use_degrees(args) else 'normalized (+/-100)'}")
  if args.policy == "openpi":
    print(f"replan   : every {args.actions} actions   settling: off   slew limit: none")
    print("NOTE: no slew limit means no auto-homing. Run `./robot home` between "
          "trials so each starts from the same pose.")
  print(f"embodiment: {embodiment.info.name}"
        f"{'  [DRY RUN - mock world, arm not opened]' if args.dry_run else ''}")
  if not args.dry_run:
    cfg = rig.so_arm_config(cameras=tuple(args.cameras), with_cameras=False,
                            use_degrees=use_degrees(args))
    print(f"port     : {cfg.port}   calibration id: {cfg.robot_id}")
    print(f"clamp    : {[round(v, 1) for v in cfg.joint_low]}")
    print(f"           {[round(v, 1) for v in cfg.joint_high]}")

  try:
    (log,) = robot_eval(
      task,
      policy,
      embodiment,
      approver=ClampApprover(embodiment.info.action_space),
      before_scoring=_grade,
      log_dir=args.log_dir,
    )
  finally:
    embodiment.close()

  print(f"\nstatus: {log.status}   scenes: {log.results.total_scenes}   "
        f"trials: {log.results.total_trials}")
  for name, value in sorted(log.results.metrics.items()):
    print(f"  {name}: {value:.4g}")
  print(f"log: {args.log_dir}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
