#!/usr/bin/env python3
"""DAgger for the openpi (JAX) checkpoint: run the policy, take over by hand, record it.

    ./robot dagger --dataset zetanschy/rollout_cap_to_cup_dagger
    ./robot dagger --dataset ... --corrections-only      # only your windows, one per episode
    ./robot dagger --dataset ... --mode sync             # no overlap, no RTC

    space  pause / resume the policy       c  take over / hand back
    enter  task complete: save the episode and pause for the reset
    esc    end the session

    The protocol, from le101's HIL guide: watch, pause when failure is imminent, take
    over, recover the arm to an in-distribution state, correct, hand back, repeat --
    then `enter` when the task is done. Intervene as often as you like within one
    episode; the trajectory stays continuous.

WHY THIS EXISTS when le101 already ships one. `lerobot-rollout --strategy.type=dagger`
is the same idea and better integrated -- but it loads the policy through lerobot's
factory, and the checkpoint that works on this bench is an openpi ORBAX directory the
factory cannot read. That is the wall evals/openpi_policy.py was written against. So
this is lerobot's strategy reimplemented around openpi's JAX policy: the same three
phases, the same keys, and above all the same `intervention` column, so the dataset it
writes trains exactly like one recorded by `lerobot-rollout`.

WHAT GETS RECORDED, where le101's docs and its code disagree. Its HIL data-collection
guide describes one episode per ATTEMPT, both segments recorded, the episode continuing
across every handoff ("no reset required just because intervention happened"), ending
when the task is done -- and then fine-tuning on the combined dataset. Its
DAggerStrategyConfig defaults to the opposite: `record_autonomous=False`, only the
correction windows, each its own episode.

This follows the DOCUMENTED protocol, because the continuous trajectory is the point:
the autonomous segment is the state the policy actually reached, and the human's
recovery from it is only informative attached to it. `--corrections-only` gives you the
code's default instead, which is the stricter DAgger reading -- an autonomous frame
carries the POLICY's own action, so those frames teach the model what it already
believes. Nothing here weights the two; `intervention` is recorded so a training run
can, if you decide it should.

NO FRAMES DURING A PAUSE, also from the protocol: pausing is for aiming, not for data.

WHAT REAL-TIME CHUNKING ADDS TO THE PROBLEM. The policy does not emit one action, it
emits a 50-action plan, and RTC pins each new plan to the actions the arm has already
committed (scripts/openpi/chunk_loop.py). A human takeover invalidates both halves of
that: the arm is somewhere the plan never predicted, and the prefix the sampler would
agree with describes a trajectory nobody is executing any more. So resuming resets the
chunker AND the schedule, and the first chunk after a correction is blocking and
unguided, exactly like an episode's first. Skipping that reset would make the policy's
first post-correction second an interpolation between your hand and a stale plan --
wrong, and almost invisible in a recording.

ONE BUS, ONE THREAD. Follower and leader are separate ports, but a feetech bus is not
thread-safe, so every read and write happens on this loop and only the model call is
offloaded. Same rule as webui/openpi_worker.py, same reason: a concurrent sync write
fails with "Port is in use!".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import enum
import importlib.util
import os
import pathlib
import statistics
import sys
import time

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# le101's own DAgger adds exactly this column and lerobot-train reads it. Keeping the
# name, dtype and shape identical is what makes the two datasets interchangeable.
INTERVENTION_FEATURE = {"dtype": "bool", "shape": (1,), "names": None}


def _load(name: str, relative: str):
    """Import one of scripts/openpi/*.py by path, registered in sys.modules.

    scripts/ is not a package and scripts/openpi/ cannot go on sys.path (a directory
    named openpi there would shadow the installed one). Registration is not optional:
    @dataclass resolves its own module out of sys.modules, so chunk_loop.py dies on its
    first dataclass without it.
    """
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- handover, transcribed from le101's control_utils ------------------------------
#
# lerobot/common/control_utils.py has these three and they are pure ramps -- but that
# module opens with `from lerobot.policies import ...`, which in the openpi image dies
# on a transformers mismatch. Transcribed rather than imported, same semantics.


def teleop_supports_feedback(teleop) -> bool:
    """True when the teleop can be DRIVEN, i.e. it is actuated (the SO-101 leader is)."""
    return (
        bool(getattr(teleop, "feedback_features", None))
        and hasattr(teleop, "disable_torque")
        and hasattr(teleop, "enable_torque")
    )


def teleop_smooth_move_to(teleop, target: dict, duration_s: float = 2.0, fps: int = 30) -> None:
    """Drive an actuated teleop to `target`, so your hand meets the follower's pose.

    This is the whole reason a handover does not jerk: the leader is brought to where
    the follower already is, instead of the follower snapping to wherever you left the
    leader. On an arm running without a slew limit that difference is not cosmetic.
    """
    teleop.enable_torque()
    current = teleop.get_action()
    steps = max(int(duration_s * fps), 1)
    for step in range(steps + 1):
        t = step / steps
        interp = {
            k: current[k] * (1 - t) + target[k] * t if k in target else current[k] for k in current
        }
        teleop.send_feedback(interp)
        time.sleep(1 / fps)


def follower_smooth_move_to(robot, current: dict, target: dict,
                            duration_s: float = 1.0, fps: int = 30) -> None:
    """Slide the follower to `target`, for a teleop that cannot be driven."""
    steps = max(int(duration_s * fps), 1)
    for step in range(steps + 1):
        t = step / steps
        interp = {
            k: current[k] * (1 - t) + target[k] * t if k in target else current[k] for k in current
        }
        robot.send_action(interp)
        time.sleep(1 / fps)


class Phase(enum.Enum):
    """The three observable states, named as in le101's DAgger strategy."""

    AUTONOMOUS = "autonomous"  # the policy is driving
    PAUSED = "paused"          # holding position, leader aligned, waiting for you
    CORRECTING = "correcting"  # you are driving, and it is being recorded


# Valid (phase, event) -> phase, copied from le101's _DAGGER_TRANSITIONS. Note what is
# ABSENT: (CORRECTING, "pause_resume"). You cannot hand the arm back to the policy while
# your hand is still on the leader -- stop the correction first. An earlier version of
# this file allowed it, which would have resumed the policy mid-motion.
TRANSITIONS: dict[tuple[Phase, str], Phase] = {
    (Phase.AUTONOMOUS, "pause_resume"): Phase.PAUSED,
    (Phase.PAUSED, "pause_resume"): Phase.AUTONOMOUS,
    (Phase.PAUSED, "correction"): Phase.CORRECTING,
    (Phase.CORRECTING, "correction"): Phase.PAUSED,
}

# Which key raises which event. le101 binds these per device (keyboard or pedal); this
# is the keyboard half, with its defaults.
EVENTS = {"space": "pause_resume", "c": "correction"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--policy", default="/checkpoints/openpi_pi05_lora_cap_to_cup_200",
                   help="orbax checkpoint directory (the one the web UI drives)")
    p.add_argument("--config", default=None,
                   help="TrainConfig name; inferred from the checkpoint's assets/ otherwise")
    p.add_argument("--task", default="Put the cap into the red cup",
                   help="the instruction, and the dataset's single_task. Keep it verbatim: "
                        "a VLA only owes you the string it was fine-tuned on")
    p.add_argument("--dataset", required=True,
                   help="dataset repo_id, e.g. you/rollout_cap_to_cup_dagger (lerobot's "
                        "rollout convention prefixes the name with rollout_)")
    p.add_argument("--root", default=None, help="write the dataset here instead of the HF cache")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset")
    p.add_argument("--corrections-only", action="store_true",
                   help="record ONLY your correction windows, each as its own episode "
                        "(le101's code default). The default here is the protocol its HIL "
                        "docs describe: one episode per attempt, both segments recorded")
    p.add_argument("--mode", choices=("rtc", "async", "sync"), default="rtc",
                   help="how chunks are executed while autonomous (default rtc)")
    p.add_argument("--actions", type=int, default=15, help="actions executed per chunk")
    p.add_argument("--units", choices=("degrees", "normalized"), default="degrees")
    p.add_argument("--fps", type=int, default=None, help="control rate (default CAM_FPS)")
    p.add_argument("--push", action="store_true", help="push the dataset to the hub at the end")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ev = _load("openpi_evaluate", "scripts/openpi/evaluate.py")
    chunk_loop = _load("openpi_chunk_loop", "scripts/openpi/chunk_loop.py")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
    from lerobot.utils.keyboard_input import create_key_listener
    from lerobot.utils.robot_utils import precise_sleep

    from openpi.policies import policy_config
    from openpi.training import config as pi0_config

    fps = args.fps or int(ev.env("CAM_FPS", "30"))
    config_name = args.config or ev.infer_config(args.policy)
    print(f"loading {args.policy}  (config {config_name})", flush=True)
    policy = policy_config.create_trained_policy(pi0_config.get_config(config_name), args.policy)

    chunker = None
    if args.mode == "rtc":
        from openpi.policies.rtc import RealTimeChunker

        # No prefix_attention_horizon: it defaults to the whole leftover, which is PI's
        # own value and tracks --actions on its own. See openpi/policies/rtc.py.
        chunker = RealTimeChunker(policy)

    robot = SO101Follower(
        SO101FollowerConfig(
            port=ev.env("ROBOT_PORT", "/dev/ttyACM1"),
            id=ev.env("ROBOT_ID", "zetans_follower"),
            cameras=ev.cameras(fps),
            use_degrees=(args.units == "degrees"),
        )
    )
    # The leader's units MUST match the follower's: its action dict is sent to the
    # follower verbatim during a correction, and nothing downstream would notice a
    # normalized value commanded as degrees except the arm.
    teleop = SO101Leader(
        SO101LeaderConfig(
            port=ev.env("TELEOP_PORT", "/dev/ttyACM0"),
            id=ev.env("TELEOP_ID", "zetans_leader"),
            use_degrees=(args.units == "degrees"),
        )
    )
    robot.connect()
    teleop.connect()
    actuated = teleop_supports_feedback(teleop)
    print(f"follower {robot.config.port} | leader {teleop.config.port} "
          f"({'actuated: the handover is driven' if actuated else 'not actuated: the follower slides to you'})",
          flush=True)

    # The schema: the robot's own features, plus the one column that makes this DAgger
    # rather than a recording.
    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
        **hw_to_dataset_features(robot.action_features, ACTION),
    }
    features["intervention"] = dict(INTERVENTION_FEATURE)
    if args.resume:
        dataset = LeRobotDataset.resume(args.dataset, root=args.root)
    else:
        dataset = LeRobotDataset.create(
            args.dataset, fps=fps, features=features, root=args.root,
            robot_type=robot.name, use_videos=True,
        )
    print(f"dataset  : {args.dataset} ({len(features)} features, incl. intervention)", flush=True)

    # Warm up BOTH traces before the arm is live: the guided sampler is a second trace,
    # and compiling it at the first replan would freeze the arm mid-motion. No action is
    # sent from here.
    warm = ev.build_observation(robot, robot.get_observation(), args.task)
    started = time.perf_counter()
    policy.infer(warm)
    if chunker is not None:
        chunker.infer(warm, prefix_start=0, inference_delay=0)
        chunker.infer(warm, prefix_start=1, inference_delay=1)
        chunker.reset()
    print(f"warmup   : {time.perf_counter() - started:.0f} s (jit compile)", flush=True)

    schedule = chunk_loop.ChunkSchedule(
        args.actions, overlap=args.mode in ("async", "rtc"), rtc=chunker is not None
    )
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi")
    pending: concurrent.futures.Future | None = None
    chunk: np.ndarray | None = None
    promised = 0
    latencies: list[float] = []

    state = {"phase": Phase.PAUSED, "stop": False, "cut": False}
    recording = False
    frames = interventions = 0

    def dispatch(key: str) -> None:
        """Keyboard events, on the listener thread: only flags are set here.

        Transitions go through TRANSITIONS, so an event the table does not define is
        ignored with a hint rather than forced -- the same refusal le101 gets from
        looking its own table up.
        """
        if key == "esc":
            state["stop"] = True
            return
        if key == "enter":
            state["cut"] = True
            return
        event = EVENTS.get(key)
        if event is None:
            return
        phase = state["phase"]
        nxt = TRANSITIONS.get((phase, event))
        if nxt is None:
            print(f"\n{event} does not apply while {phase.value}"
                  + ("  (stop the correction with c first)"
                     if phase is Phase.CORRECTING else "  (pause with space first)"),
                  flush=True)
            return
        state["phase"] = nxt
        print(f"\n-> {nxt.value}", flush=True)

    listener = create_key_listener(
        dispatch, controls_help="space=pause/resume, c=correct, enter=cut episode, esc=quit"
    )
    if listener is None:
        robot.disconnect()
        teleop.disconnect()
        raise SystemExit(
            "no keyboard backend available: this needs an interactive terminal. "
            "`./robot dagger` runs docker compose run, which gives one."
        )

    def infer(obs, prefix_start: int, inference_delay: int):
        """Pure compute, safe on the worker thread: it touches no serial port."""
        t0 = time.perf_counter()
        if chunker is None:
            out = policy.infer(obs)
        else:
            out = chunker.infer(obs, prefix_start=prefix_start, inference_delay=inference_delay)
        return np.asarray(out["actions"], dtype=np.float64), time.perf_counter() - t0

    def to_action(vector) -> dict:
        return {name: float(vector[i]) for i, name in enumerate(robot.action_features)
                if i < len(vector)}

    def pose(raw: dict) -> dict:
        """The follower's current pose, in action key space."""
        return {name: float(raw[name]) for name in robot.action_features if name in raw}

    def drop_pending() -> None:
        """Discard an in-flight chunk. Called whenever a human takes the arm."""
        nonlocal pending
        if pending is not None and not pending.cancel():
            try:
                pending.result()
            except Exception:  # noqa: BLE001 - a discarded chunk's failure is nobody's
                pass
        pending = None

    def record(raw: dict, act: dict, intervention: bool) -> None:
        nonlocal frames, interventions, recording
        dataset.add_frame({
            **build_dataset_frame(features, raw, prefix=OBS_STR),
            **build_dataset_frame(features, act, prefix=ACTION),
            "task": args.task,
            "intervention": np.array([intervention], dtype=bool),
        })
        recording = True
        frames += 1
        interventions += int(intervention)

    def close_episode(reason: str) -> None:
        nonlocal recording
        if not recording:
            return
        dataset.save_episode()
        recording = False
        print(f"episode saved ({reason}): {frames} frames total, {interventions} corrections",
              flush=True)

    print("\nready. space=pause/resume, c=correct (from paused), enter=cut, esc=quit\n", flush=True)
    last_action: dict | None = None
    previous = state["phase"]
    try:
        while not state["stop"]:
            tick = time.perf_counter()
            phase = state["phase"]
            raw = robot.get_observation()

            # --- transitions, resolved before acting on the new phase --------------
            if phase is not previous:
                # The transition table, and the torque with it, is le101's
                # DAggerStrategy._handle_phase_change translated to this engine. The
                # torque dance is not decoration: teleop_smooth_move_to LEAVES THE
                # LEADER POWERED (it enables torque to drive it), so a correction that
                # began without releasing it would have you fighting its own motors.
                if phase is Phase.PAUSED and previous is Phase.AUTONOMOUS:
                    drop_pending()
                    # The follower's MEASURED pose, not the last command. The one
                    # deliberate departure from le101, which drives the leader to
                    # `prev_action`: under gravity a loaded arm sits below where it was
                    # told to be, and the leader should meet the arm, not the command.
                    last_action = pose(raw)
                    if actuated:
                        teleop_smooth_move_to(teleop, last_action, fps=fps)  # enables torque
                elif phase is Phase.CORRECTING:
                    if actuated:
                        # Let go of the leader so a human can actually move it.
                        teleop.disable_torque()
                    else:
                        follower_smooth_move_to(robot, pose(raw), teleop.get_action(), fps=fps)
                elif phase is Phase.PAUSED and previous is Phase.CORRECTING:
                    if actuated:
                        # Re-lock it, so it holds where you left it instead of sagging.
                        teleop.enable_torque()
                elif phase is Phase.AUTONOMOUS:
                    # The arm moved under a human hand: the plan and its prefix both
                    # describe a trajectory nobody is executing. Start clean -- this is
                    # their `engine.reset(); interpolator.reset()` on resume, and for
                    # RTC it is load-bearing rather than hygiene.
                    drop_pending()
                    chunk = None
                    schedule.reset()
                    if chunker is not None:
                        chunker.reset()
                    if actuated:
                        # Release the leader before the policy drives, as they do.
                        teleop.disable_torque()
                    if args.corrections_only:
                        close_episode("correction ended")
                if phase is Phase.CORRECTING:
                    print("recording your correction (c to stop)", flush=True)
                previous = phase

            if state["cut"]:
                state["cut"] = False
                close_episode("task complete")
                # Step 6 of their protocol: the episode ends PAUSED, with the leader
                # aligned, so you can reposition the arm for the next attempt.
                state["phase"] = Phase.PAUSED

            # --- act ---------------------------------------------------------------
            if phase is Phase.CORRECTING:
                act = teleop.get_action()
                robot.send_action(act)
                last_action = act
                record(raw, act, intervention=True)

            elif phase is Phase.PAUSED:
                if last_action is not None:
                    robot.send_action(last_action)

            else:  # AUTONOMOUS
                plan = schedule.plan(in_flight=pending is not None)
                if plan.action == "infer":
                    obs = ev.build_observation(robot, raw, args.task)
                    chunk, dt = infer(obs, plan.request.prefix_start, plan.request.inference_delay)
                    schedule.adopt(len(chunk), delay=plan.request.inference_delay)
                    latencies.append(dt)
                elif plan.action == "collect":
                    chunk, dt = pending.result()
                    pending = None
                    schedule.adopt(len(chunk), delay=promised)
                    latencies.append(dt)
                else:
                    if plan.request is not None:
                        promised = plan.request.inference_delay
                        # Observation read on THIS thread; only the model call is offloaded.
                        obs = ev.build_observation(robot, raw, args.task)
                        pending = pool.submit(infer, obs, plan.request.prefix_start, promised)
                    act = to_action(chunk[schedule.take()])
                    robot.send_action(act)
                    last_action = act
                    if not args.corrections_only:
                        record(raw, act, intervention=False)

            precise_sleep(max(1.0 / fps - (time.perf_counter() - tick), 0.0))
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        close_episode("session ended")
        drop_pending()
        pool.shutdown(wait=False)
        listener.stop()
        robot.disconnect()
        teleop.disconnect()

    if latencies:
        print(f"\ninference: {statistics.mean(latencies) * 1000:.0f} ms mean over "
              f"{len(latencies)} calls", flush=True)
    print(f"recorded : {frames} frames, {interventions} of them corrections, "
          f"{dataset.num_episodes} episode(s)", flush=True)
    if args.push:
        print("pushing to the hub ...", flush=True)
        dataset.push_to_hub()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
