#!/usr/bin/env python3
"""DAgger for the openpi (JAX) checkpoint: run the policy, take over by hand, record it.

    ./robot dagger --dataset zetanschy/rollout_cap_to_cup_dagger
    ./robot dagger --dataset ... --display_data          # the live rerun view, as in record
    ./robot dagger --dataset ... --corrections-only      # only your windows, one per episode
    ./robot dagger --dataset ... --mode sync             # no overlap, no RTC

    space  pause / resume the policy       tab    take over / hand back
    right  task complete: save the episode  left   the attempt failed: throw it away
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

WHEN YOUR OWN CORRECTION GOES WRONG. Keep correcting. Recovering the arm from a mess
you made is the same data as recovering it from one the policy made -- the protocol's
step 3 is "teleoperate the robot back to a good state", and it does not care whose fault
the state is. Only when the attempt is spoiled outright (the cap on the floor, the cup
knocked over) does the episode stop being worth keeping, and then `left` drops the whole
thing.

Deliberately NOT offered: undoing just the last correction. The buffer could be
truncated -- it is a dict of lists -- but the episode would then jump from the frame
before your correction to wherever the arm physically is now, and a trajectory with a
teleport in it is worse than no trajectory. An episode is continuous or it is nothing,
which is why lerobot's own tools drop whole episodes too (lerobot-edit-dataset's
delete_episodes) and never a segment.

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
import threading
import time

import numpy as np

# Only applies to a bare `python scripts/openpi/dagger.py`; compose sets it. 0.65 of a
# 12 GB card is 7373 MiB against a measured 6939 MiB peak -- 0.75 does not fit beside a
# desktop session and the run OOMs after the checkpoint is already loaded.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")

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


# A feetech write can come back garbled -- "Incorrect status packet!" -- and lerobot's
# SOLeader.enable_torque() calls the bus with num_retry=0, so one bad packet raises. It
# happened on the first real session, during the handover, and took the whole run with
# it. The bus itself accepts retries; this is how they get asked for.
TORQUE_RETRIES = 3


def set_leader_torque(teleop, enabled: bool) -> bool:
    """Enable or disable the leader's torque, retrying a garbled bus. False if it failed."""
    try:
        if enabled:
            teleop.bus.enable_torque(num_retry=TORQUE_RETRIES)
        else:
            teleop.bus.disable_torque(num_retry=TORQUE_RETRIES)
        return True
    except Exception as exc:  # noqa: BLE001 - a bus that will not answer, whatever it raises
        print(f"\nleader torque {'on' if enabled else 'off'} failed: {exc}", flush=True)
        return False


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
    set_leader_torque(teleop, True)
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

# Which key raises which event, copied from le101's DAggerKeyboardConfig defaults --
# `space` and `tab`, not the "c" its docstring happens to use as a format example.
# dagger_test.py parses those defaults out of the submodule and compares, so a rebind
# upstream shows up as a failing test rather than as a key that does nothing.
EVENTS = {"space": "pause_resume", "tab": "correction"}

# Episode control, and these ARE lerobot's: apply_recording_control in
# lerobot/utils/keyboard_input.py maps right -> exit_early (end and keep), left ->
# rerecord_episode (throw it away and do it again), esc -> stop_recording. Every
# `./robot record` session already uses them, and a failed attempt on this rig is the
# same gesture as a failed attempt there.
#
# DAgger's own keys (space, tab) come from DAggerKeyboardConfig above; these three come
# from the recording loop. Two sources, because the two things are separate: one is
# about who drives, the other about what is kept.
RECORDING_KEYS = {"right": "save", "left": "discard", "esc": "stop"}


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
    # The same live view `./robot record` gets, through le101's own helpers, so the
    # panel layout is the one you already read during teleop.
    p.add_argument("--display_data", action="store_true",
                   help="live rerun view of both cameras, the state and the action")
    p.add_argument("--display_mode", choices=("rerun", "foxglove"), default="rerun")
    p.add_argument("--display_ip", default=None,
                   help="stream to a viewer running elsewhere instead of spawning one "
                        "(rerun: its host; needed when the container has no X11)")
    p.add_argument("--display_port", type=int, default=None)
    p.add_argument("--display_compressed", action="store_true",
                   help="compress images before logging: less bandwidth, more CPU on the "
                        "control thread")
    p.add_argument("--push", action="store_true", help="push the dataset to the hub at the end")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ev = _load("openpi_evaluate", "scripts/openpi/evaluate.py")
    chunk_loop = _load("openpi_chunk_loop", "scripts/openpi/chunk_loop.py")

    from lerobot.datasets import VideoEncodingManager
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
    from lerobot.utils.keyboard_input import create_key_listener
    from lerobot.utils.robot_utils import precise_sleep

    from openpi.policies import policy_config
    from openpi.training import config as pi0_config

    if args.display_data:
        from lerobot.utils.visualization_utils import (
            init_visualization,
            log_visualization_data,
            shutdown_visualization,
        )

        # CHECKED BEFORE THE CHECKPOINT LOADS. le101 defers this to init_rerun's
        # require_package, which here would fire after a minute of JIT compile with the
        # arms already connected. The openpi image did not ship rerun-sdk until the
        # Dockerfile gained it, so an older image needs a rebuild.
        backend = {"rerun": "rerun", "foxglove": "foxglove_websocket"}[args.display_mode]
        try:
            importlib.import_module(backend)
        except ImportError:
            raise SystemExit(
                f"--display_data needs {backend} inside the openpi image, and this one "
                "does not have it. Rebuild with `./robot openpi-build`, or drop "
                "--display_data (the session runs fine without a viewer)."
            ) from None

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
            # ASYNC IMAGE WRITING IS NOT OPTIONAL AT 30 Hz. With both of these left at
            # 0, LeRobotDataset starts no writer and add_frame() encodes the PNGs on
            # the calling thread -- which here is the control loop, whose whole budget
            # is 33 ms. lerobot's own record passes threads_per_camera * cameras, and
            # its default per camera is 4.
            image_writer_processes=0,
            image_writer_threads=4 * max(len(robot.cameras), 1),
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

    state = {"phase": Phase.PAUSED, "stop": False, "save": False, "discard": False}
    recording = False
    frames = interventions = 0        # kept, i.e. saved episodes only
    ep_frames = ep_interventions = 0  # the episode being recorded right now

    def dispatch(key: str) -> None:
        """Keyboard events, on the listener thread: only flags are set here.

        Transitions go through TRANSITIONS, so an event the table does not define is
        ignored with a hint rather than forced -- the same refusal le101 gets from
        looking its own table up.
        """
        control = RECORDING_KEYS.get(key)
        if control == "stop":
            state["stop"] = True
            return
        if control in ("save", "discard"):
            state[control] = True
            return
        event = EVENTS.get(key)
        if event is None:
            return
        phase = state["phase"]
        nxt = TRANSITIONS.get((phase, event))
        if nxt is None:
            print(f"\n{event} does not apply while {phase.value}"
                  + ("  (hand back with tab first)"
                     if phase is Phase.CORRECTING else "  (pause with space first)"),
                  flush=True)
            return
        state["phase"] = nxt
        print(f"\n-> {nxt.value}", flush=True)

    listener = create_key_listener(
        dispatch, controls_help="space=pause/resume, tab=take over, right=save episode, left=discard it, esc=quit"
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
        nonlocal ep_frames, ep_interventions, recording
        dataset.add_frame({
            **build_dataset_frame(features, raw, prefix=OBS_STR),
            **build_dataset_frame(features, act, prefix=ACTION),
            "task": args.task,
            "intervention": np.array([intervention], dtype=bool),
        })
        recording = True
        ep_frames += 1
        ep_interventions += int(intervention)

    def discard_episode() -> None:
        """Throw the current episode away, the way left-arrow does in lerobot-record.

        A failed attempt is worse than no attempt: it teaches the policy a trajectory
        that did not work. clear_episode_buffer() drops the buffered frames and their
        images, so nothing of it reaches the parquet.
        """
        nonlocal recording, ep_frames, ep_interventions
        if not recording:
            print("\nnothing recorded yet, nothing to discard", flush=True)
            return
        dataset.clear_episode_buffer()
        print(f"\nepisode discarded: {ep_frames} frames dropped "
              f"({ep_interventions} corrections). Reset the scene and go again.", flush=True)
        recording = False
        ep_frames = ep_interventions = 0

    # Half a second. Below this an "episode" is an artefact of where a keypress landed,
    # not an attempt: the first real session ended with a 1-frame episode saved on the
    # way out, which is 33 ms of video and a row of noise in the parquet.
    min_frames = max(int(fps * 0.5), 2)

    def close_episode(reason: str) -> None:
        """Keep what has been recorded so far as one episode, if there is one."""
        nonlocal recording, frames, interventions, ep_frames, ep_interventions
        if not recording:
            return
        if ep_frames < min_frames:
            print(f"\nepisode dropped ({reason}): {ep_frames} frames is under "
                  f"{min_frames}, too short to be an attempt", flush=True)
            discard_episode()
            return
        dataset.save_episode()
        frames += ep_frames
        interventions += ep_interventions
        print(f"episode {dataset.num_episodes - 1} saved ({reason}): {ep_frames} frames, "
              f"{ep_interventions} of them corrections", flush=True)
        recording = False
        ep_frames = ep_interventions = 0

    display = args.display_data
    if display:
        # Same session name lerobot uses, so a viewer already open on it just works.
        init_visualization(args.display_mode, session_name="lerobot_control_loop",
                           ip=args.display_ip, port=args.display_port)
        if args.display_ip:
            print(f"display  : {args.display_mode} -> {args.display_ip}:{args.display_port}",
                  flush=True)
        else:
            # rerun spawns the viewer INSIDE the container here, which it warns is
            # unsupported -- and a viewer that dies on startup took a whole session
            # with it once: the log calls kept going to a sink with nobody behind it
            # and shutdown_rerun() hung the exit. Guarded below, but say so.
            print(f"display  : {args.display_mode} -> viewer spawned in the container "
                  "(rerun calls this unsupported; --display_ip streams to one on the "
                  "host instead)", flush=True)

    print("\nready. space=pause/resume, tab=take over (from paused), "
          "right=save episode, left=discard it, esc=quit\n", flush=True)
    last_action: dict | None = None
    previous = state["phase"]
    # WITHOUT THIS THE SESSION WRITES AN UNREADABLE DATASET. VideoEncodingManager's
    # __exit__ calls dataset.finalize(), which flushes pending videos and writes the
    # parquet and the metadata -- le101's DAgger wraps both of its recording loops in
    # it for the same reason. It also cleans a half-written episode when the body
    # raises, which on a session ended by Ctrl-C is the normal case. Verified the hard
    # way: a dataset recorded without it cannot be read back at all; LeRobotDataset
    # falls through to the Hub looking for what should have been on disk.
    with VideoEncodingManager(dataset):
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
                            # THE HANDOVER MAY FAIL AND THE SESSION MUST NOT. A garbled
                            # feetech reply here used to raise out of the loop and end a
                            # run; now it leaves you paused with the arm holding, and you
                            # can press space to try again or just move the leader by hand
                            # (the follower is not driven from it until you press tab).
                            try:
                                teleop_smooth_move_to(teleop, last_action, fps=fps)
                            except Exception as exc:  # noqa: BLE001 - bus, not logic
                                print(f"\nhandover failed: {exc}\n"
                                      "the arm is holding and you are still PAUSED. Check the "
                                      "leader's power and its bus, then space to retry.",
                                      flush=True)
                    elif phase is Phase.CORRECTING:
                        if actuated:
                            # Let go of the leader so a human can actually move it.
                            set_leader_torque(teleop, False)
                        else:
                            follower_smooth_move_to(robot, pose(raw), teleop.get_action(), fps=fps)
                    elif phase is Phase.PAUSED and previous is Phase.CORRECTING:
                        if actuated:
                            # Re-lock it, so it holds where you left it instead of sagging.
                            set_leader_torque(teleop, True)
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
                            set_leader_torque(teleop, False)
                        if args.corrections_only:
                            close_episode("correction ended")
                    if phase is Phase.CORRECTING:
                        print("recording your correction (c to stop)", flush=True)
                    previous = phase

                if state["save"] or state["discard"]:
                    if state["save"]:
                        close_episode("task complete")
                    else:
                        discard_episode()
                    state["save"] = state["discard"] = False
                    # Step 6 of their protocol: the episode ends PAUSED, with the leader
                    # aligned, so you can reposition the arm for the next attempt.
                    state["phase"] = Phase.PAUSED
                    # RE-READ IT. `phase` was taken at the top of the tick, so without
                    # this the act() below still runs the OLD phase and records one more
                    # autonomous frame into the episode that was just discarded -- which
                    # is exactly the stray 1-frame episode the first session ended with.
                    phase = state["phase"]

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

                if display:
                    # `raw` is the observation this tick acted on, and last_action is what
                    # was sent for it -- during a pause nothing new was, hence the None.
                    #
                    # A VIEW IS NEVER WORTH THE SESSION. If logging fails, the view is
                    # switched off and the arm keeps going: the data being collected is the
                    # point, and a dead viewer is not a reason to lose an episode.
                    try:
                        log_visualization_data(
                            args.display_mode,
                            observation=raw,
                            action=None if phase is Phase.PAUSED else last_action,
                            compress_images=args.display_compressed,
                        )
                    except Exception as exc:  # noqa: BLE001 - any backend failure, same answer
                        display = False
                        print(f"\ndisplay off: {type(exc).__name__}: {exc} "
                              "(the session continues without it)", flush=True)

                precise_sleep(max(1.0 / fps - (time.perf_counter() - tick), 0.0))
        except KeyboardInterrupt:
            print("\ninterrupted", flush=True)
        finally:
            close_episode("session ended")
            drop_pending()
            pool.shutdown(wait=False)
            listener.stop()
            if args.display_data:
                # IN A THREAD, WITH A DEADLINE. shutdown_rerun() flushes, and flushing to a
                # viewer that died takes forever -- which is how a Ctrl-C once left the
                # process alive, holding 8 GB of VRAM, until it was killed by hand. The
                # arms matter more than a clean flush, so this gets three seconds.
                closer = threading.Thread(
                    target=lambda: shutdown_visualization(args.display_mode), daemon=True
                )
                closer.start()
                closer.join(timeout=3.0)
                if closer.is_alive():
                    print("display: shutdown did not return in 3 s, leaving it", flush=True)
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
