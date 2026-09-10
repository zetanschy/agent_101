"""Run one Inspect Robots eval on the SO-ARM101: an LLM agent, or a lerobot VLA.

    ./robot eval --policy agent  --model openai/gpt-6-astra    # the LLM agent
    ./robot eval-openpi --policy openpi                        # the working pi0.5
    ./robot eval-openpi --task cap-quadrants                   # quadrant cap, random cup
    ./robot eval-openpi --mode sync                            # the pre-RTC code path
    ./robot eval --policy agent --dry-run                      # no motion, no arm

--task picks the benchmark out of evals/tasks.py; --layouts/--epochs default to
whatever that benchmark declares (cap-to-cup 5 x 1, cap-quadrants 2 x 5).

--mode is how the openpi chunks are executed, and it defaults to `rtc` because that is
the configuration webui/openpi_worker.py drives this checkpoint with. `sync` is the
code path every eval log before this change was recorded in -- same controller, same
act(), same blocking inference -- so those scores stay comparable. See evals/rtc.py.

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
from evals.openpi_policy import RTC_JACOBIAN, RTC_MAX_GUIDANCE, RTC_SCHEDULE

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

    # rtc=True only builds the chunker; whether inference OVERLAPS execution is the
    # controller's call (build_controller). Both come off --mode so they cannot
    # disagree -- a chunker with no overlap would promise a delay of zero, and an
    # overlap with no chunker is the cold splice --mode async exists to show.
    return OpenPiPolicy(args.checkpoint or OPENPI_CHECKPOINT,
                        cameras=tuple(args.cameras),
                        replan_interval=args.actions,
                        rtc=args.mode == "rtc",
                        rtc_schedule=args.rtc_schedule,
                        rtc_max_guidance=args.rtc_max_guidance,
                        rtc_jacobian=args.rtc_jacobian,
                        rtc_horizon=args.rtc_horizon)

  if args.policy == "agent":
    from inspect_robots_agent.policy import LLMAgentPolicy

    # Model strings are OpenRouter-style provider/model. openai/* resolves against
    # OPENAI_API_KEY directly; anything unknown falls through to OPENROUTER_API_KEY.
    kwargs = {
      "model": args.model,
      "effort": args.effort,
      "max_llm_calls": args.decisions,
    }
    wire = args.wire or ("responses" if args.model.startswith("openai/") else None)
    if wire:
      kwargs["wire"] = wire
    return LLMAgentPolicy(**kwargs)

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


def build_controller(args):
  """The execution loop, or None to let eval() build its own.

  None means DefaultController(policy.config.replan_interval): infer, wait, play the
  window, repeat. That is `--mode sync`, and it is returned unwrapped rather than
  reimplemented so the synchronous path stays the one the earlier logs used.
  """
  if args.policy != "openpi" or args.mode == "sync":
    return None
  from evals.rtc import AsyncChunkController

  return AsyncChunkController(args.actions, rtc=args.mode == "rtc")


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
  # Recent OpenAI reasoning models REJECT function tools on the Chat Completions wire
  # ("Function tools with reasoning_effort are not supported ... use /v1/responses or
  # set reasoning_effort to 'none'"). This policy is nothing but function tools, so
  # openai/* defaults to the Responses wire rather than to the plugin's `chat`, which
  # keeps reasoning on. It needs a direct OpenAI endpoint, i.e. OPENAI_API_KEY -- a
  # model routed through OpenRouter should be given --wire chat and --effort none.
  p.add_argument("--wire", default=None,
                 help="agent only: chat | responses | messages | interactions | "
                      "gemini-live (default: responses for openai/*, else the "
                      "plugin's own default)")
  p.add_argument("--checkpoint", default=None,
                 help="lerobot: Hub id or path; openpi: orbax checkpoint dir")
  p.add_argument("--policy-type", default="pi05", help="lerobot only: policy class")
  p.add_argument("--device", default="cuda")
  p.add_argument("--cameras", nargs="+", default=["front", "grip"])
  p.add_argument("--actions", type=int, default=15,
                 help="openpi only: actions executed per chunk before re-planning "
                      "(webui/openpi_worker.py's default, and what it was tuned with)")
  p.add_argument("--mode", choices=("rtc", "async", "sync"), default="rtc",
                 help="openpi only: rtc = overlap inference with execution and pin "
                      "each new chunk to the actions committed while it was sampled "
                      "(default, matches webui/openpi_worker.py --mode rtc); async = "
                      "overlap without the pin, which jumps at every seam; sync = the "
                      "arm holds still through every inference (the pre-RTC path)")
  p.add_argument("--task", choices=tuple(tasks.TASKS), default="cap-to-cup",
                 help="cap-to-cup: 5 hand-set layouts, both objects pinned. "
                      "cap-quadrants: cap in one lower table quadrant, cup re-drawn "
                      "at random before every trial (default 2 x 5 = 10 trials)")
  # Unset means "whatever this task declares", which is not the same number for the
  # two of them -- cap-to-cup is 5 x 1, cap-quadrants 2 x 5. Only what the operator
  # actually typed is forwarded to the builder.
  # RTC tuning. The defaults are openpi's own (schedule exp, beta_max 5.0 -- PI's
  # kinetix eval value -- and the identity Jacobian, which is what the LeRobot
  # reference computes and what scripts/openpi/rtc_parity.py pins us to). They are
  # flags rather than constants so a hardware session can A/B them: the other openpi
  # RTC implementation in the wild defaults to beta_max 10.0 (LeRobot's) and the true
  # VJP, and which is better on THIS arm is a measurement nobody has taken.
  p.add_argument("--rtc-schedule", choices=("exp", "linear", "ones", "zeros"),
                 default=RTC_SCHEDULE,
                 help="rtc only: how the prefix constraint decays past the frozen actions")
  p.add_argument("--rtc-max-guidance", type=float, default=RTC_MAX_GUIDANCE,
                 help="rtc only: beta_max, the clamp on the guidance weight")
  p.add_argument("--rtc-jacobian", choices=("identity", "full"),
                 default=RTC_JACOBIAN,
                 help="rtc only: identity is free and matches the LeRobot reference; "
                      "full is the true VJP through the action expert, ~2x per step")
  p.add_argument("--rtc-horizon", type=int, default=None,
                 help="rtc only: release the prefix constraint EARLY, at this index. "
                      "Default is the whole overlap with the retiring chunk, which is "
                      "PI's own value (action_horizon - actions) and tracks --actions")
  p.add_argument("--layouts", type=int, default=None,
                 help="scenes to run (default: all the task declares)")
  p.add_argument("--epochs", type=int, default=None,
                 help="trials per scene (default: the task's own)")
  p.add_argument("--seconds", type=float, default=tasks.DEFAULT_SECONDS,
                 help="VLA horizon, converted to steps at the declared control_hz")
  # The agent's real budget. max_llm_calls is enforced AND written into its system
  # prompt, so the model plans against it; it governs runtime and the bill alike, at a
  # measured ~8 s and ~$0.06 per decision (the per-call cost rises through a trial as
  # the conversation accumulates images). The plugin's own default is 100, which on
  # this rig is roughly 13 minutes and $6 a trial.
  p.add_argument("--decisions", type=int, default=40,
                 help="agent only: LLM call budget per trial (default 40, ~5 min, ~$2.5)")
  # Nothing in Inspect Robots bounds a trial in TIME -- every other limit is a count.
  # A degraded provider can stall a decision for 6 minutes (120 s timeout, 3 retries)
  # without erroring, so an unattended run needs a clock. See evals/deadline.py.
  p.add_argument("--max-minutes", type=float, default=10.0,
                 help="wall-clock budget per trial; 0 disables (default 10)")
  p.add_argument("--log-dir", default="outputs/evals")
  p.add_argument("--no-frames", action="store_true",
                 help="do not store camera frames (no video in the HTML report)")
  p.add_argument("--rerun", metavar="PATH", default=None,
                 help="also stream to a Rerun .rrd recording at PATH")
  p.add_argument("--dry-run", action="store_true",
                 help="run against the CubePick mock world; the arm is never opened")
  args = p.parse_args(argv)

  # The horizon follows the policy, because "seconds" means different things to the
  # two. eval() turns max_seconds into steps at the embodiment's declared 30 Hz, which
  # the chunked VLA really does run at; the agent measured 5.5 steps/s, so a seconds
  # budget silently became 2.7x the wall clock and truncated the first trial. For the
  # agent the horizon is derived from its decision budget instead, sized so the
  # DECISION limit is what ends the trial -- that is the one it is told about.
  horizon = (
    {"steps": args.decisions * tasks.STEPS_PER_DECISION}
    if args.policy == "agent"
    else {"seconds": args.seconds}
  )
  requested = {
    key: value
    for key, value in (("layouts", args.layouts), ("epochs", args.epochs))
    if value is not None
  }
  task = tasks.TASKS[args.task](**requested, **horizon)
  epochs = task.epoch_spec.count
  policy = build_policy(args)
  controller = build_controller(args)
  if args.max_minutes:
    from evals.deadline import Deadline

    policy = Deadline(policy, args.max_minutes)
  embodiment = build_embodiment(args)

  trials = len(task.scenes) * epochs
  print(f"task     : {task.name}  ({len(task.scenes)} scenes x {epochs} epochs "
        f"= {trials} trial(s))")
  # The operator sets every scene up by hand and the framework has no pre-trial hook,
  # so the setups are listed HERE, before the arm moves, rather than only echoed at
  # the verdict prompt once the trial is already over.
  for scene in task.scenes:
    print(f"  {scene.id}: {(scene.metadata or {}).get('operator_setup', '')}")
  shown = args.model if args.policy == "agent" else (
    args.checkpoint or (OPENPI_CHECKPOINT if args.policy == "openpi" else LEROBOT_CHECKPOINT)
  )
  print(f"policy   : {args.policy} ({shown})")
  if args.policy == "agent":
    wire = args.wire or ("responses" if args.model.startswith("openai/") else "default")
    print(f"wire     : {wire}   effort: {args.effort}")
    # Measured on the first Astra trial: 6.0 s mean LLM latency plus the arm executing
    # each move, ~8 s per decision, $0.064 per decision and rising within a trial.
    print(f"budget   : {args.decisions} decisions/trial "
          f"(~{args.decisions * 8 // 60} min, ~${args.decisions * 0.064:.2f}) "
          f"x {trials} trial(s)")
  if args.max_minutes:
    print(f"deadline : {args.max_minutes:g} min/trial "
          f"(worst case {args.max_minutes * trials:g} min for {trials} trial(s))")
  print(f"units    : {'degrees' if use_degrees(args) else 'normalized (+/-100)'}")
  if args.policy == "openpi":
    modes = {
      "rtc": "overlapped, seam pinned to the committed actions (arXiv:2506.07339)",
      "async": "overlapped, seam spliced cold",
      "sync": "arm holds still through every inference",
    }
    print(f"mode     : {args.mode} -- {modes[args.mode]}")
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

  # Frames are what make the HTML report visual: `inspect-robots view` turns them
  # into a per-trial composite MP4 (one playhead across both cameras) when ffmpeg is
  # present, which it is in both images. Without them the report is text and scores.
  sinks = None
  if args.rerun:
    # Passing sinks REPLACES the default JsonLogSink rather than adding to it, so the
    # JSON log has to be listed explicitly or the canonical record is lost.
    from inspect_robots.logging import JsonLogSink, RerunSink

    sinks = [JsonLogSink(args.log_dir), RerunSink(args.rerun)]

  try:
    (log,) = robot_eval(
      task,
      policy,
      embodiment,
      approver=ClampApprover(embodiment.info.action_space),
      controller=controller,
      before_scoring=_grade,
      log_dir=args.log_dir,
      store_frames=not args.no_frames,
      sinks=sinks,
    )
  finally:
    embodiment.close()
    # The openpi policy owns an inference thread; a Ctrl-C between trials would
    # otherwise leave it holding a chunk nobody will collect.
    close = getattr(policy, "close", None)
    if callable(close):
      close()

  print(f"\nstatus: {log.status}   scenes: {log.results.total_scenes}   "
        f"trials: {log.results.total_trials}")
  for name, value in sorted(log.results.metrics.items()):
    print(f"  {name}: {value:.4g}")
  print(f"log: {args.log_dir}")
  if not args.no_frames:
    print("video + transcript report:  ./robot eval-view")
  if args.rerun:
    print(f"rerun recording: {args.rerun}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
