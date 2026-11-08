#!/usr/bin/env python3
"""Persistent worker for the mjlab BC student (a .ts export) on the SO-ARM101.

    python webui/student_worker.py --policy sim/outputs/mjlab/students/bc_2800.ts

Same protocol as webui/openpi_worker.py -- run / eval / stop / home / quit on stdin,
markers on stdout -- so the control panel and the Eval page drive it unchanged. Below
the protocol it is a plain 30 Hz closed loop: read the joints and both cameras, one
~2 ms CPU forward pass (scripts/openpi/student_policy.py), send the new joint target.
No chunks and no task string; `actions N` is accepted and ignored.

HOME is the twin's rest pose from the student's contract, not config/home_pose.json:
every training episode started at rest, and the bench's home is ~10 deg away from it
in wrist roll.

THE GATE: on the real arm, a student loads only if the live twin has scored that exact
file (sha256) over >= GATE_TRIALS trials at >= GATE_SUCCESS: `./robot sim-eval --policy
/workspace/<student>.ts --episodes 20`. The live twin renders as the real camera sees;
bc_2800 scored 13/20 in mjlab, 0/20 there, and on the arm swept into the mug.

SAFETY: the target moves at most what the env allowed, 0.1 rad per frame on the arm
(172 deg/s: the teacher moves that fast), and stays inside the joint limits (Wrist_Roll
inside the operator's range). Keep a hand on Stop.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
import threading
import time

import numpy as np
import openpi_worker as ow

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE_TRIALS, GATE_SUCCESS = 20, 0.25


def live_twin_score(policy: str) -> tuple[int, int, str] | None:
    """(successes, trials, run dir) of the best live-twin sim-eval run of this exact file."""
    sha = hashlib.sha256(pathlib.Path(policy).read_bytes()).hexdigest()
    runs = []
    for run in (ROOT / "outputs" / "sim_eval").glob("*/run.json"):
        trials = run.parent / "trials.jsonl"
        if json.loads(run.read_text()).get("policy_sha256") != sha or not trials.exists():
            continue
        rows = [json.loads(x) for x in trials.read_text().splitlines() if x.strip()]
        k, n = sum(r["success"] for r in rows), len(rows)
        runs.append((n >= GATE_TRIALS, k / max(n, 1), k, n, run.parent.name))
    return max(runs)[2:] if runs else None  # complete runs first, then the best rate


def _student_policy():
    spec = importlib.util.spec_from_file_location(
        "student_policy", ROOT / "scripts" / "openpi" / "student_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StudentPolicy


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="a student.py .ts export")
    p.add_argument("--fps", type=int, default=None)
    args = p.parse_args()

    ev = ow._load_eval_module()
    fps = args.fps or int(ev.env("CAM_FPS", "30"))
    print("LOADING", flush=True)
    if ev.env("ROBOT_TYPE", "") != "real2sim":
        score = live_twin_score(args.policy)
        if score is None or score[1] < GATE_TRIALS or score[0] / score[1] < GATE_SUCCESS:
            have = f"{score[0]}/{score[1]} in outputs/sim_eval/{score[2]}" if score else "no live-twin run of this file"
            raise SystemExit(f"GATE: not on the arm until the live twin scores {args.policy} at >= "
                             f"{GATE_SUCCESS:.0%} over >= {GATE_TRIALS} trials ({have}). Run: ./robot sim-eval "
                             f"--policy {args.policy} --episodes {GATE_TRIALS} --name <name>")
        print(f"gate: live twin {score[0]}/{score[1]} (outputs/sim_eval/{score[2]})", flush=True)
    policy = _student_policy()(args.policy)
    if policy.contract["fps"] != fps:
        raise SystemExit(f"the student acts at {policy.contract['fps']} Hz, the cameras run at {fps}")
    robot = ow.connect_robot(ev, fps)
    policy(robot.get_observation())  # warm-up; nothing is sent
    rest = {f"{j}.pos": float(v) for j, v in
            zip(policy.joints, policy.to_lerobot(np.asarray(policy.contract["rest"])))}
    ow.HOME_FILE = pathlib.Path("/tmp/student_rest_pose.json")
    ow.HOME_FILE.write_text(json.dumps({k[:-4]: v for k, v in rest.items()}))
    print(f"student {args.policy}: home = the twin's rest pose "
          f"{ {k: round(v, 1) for k, v in rest.items()} }", flush=True)

    preview = None
    try:
        preview = ow.Preview(robot, sim=ev.env("ROBOT_TYPE", "") == "real2sim")
        preview.start()
    except Exception as e:  # noqa: BLE001 - inference works without the Eval page's view
        print(f"no eval preview: {e}", flush=True)
    print("MODEL_LOADED", flush=True)

    state = {"trial": None}
    shutdown = threading.Event()
    shutdown.set()
    run_thread: threading.Thread | None = None

    def loop():
        from lerobot.utils.robot_utils import precise_sleep

        print("RUN_START", flush=True)
        trial = state["trial"]
        if trial is not None:
            trial.update(t0=time.perf_counter(), reason="stopped")
            if preview is not None:
                preview.record(trial["video"])
        policy.reset()  # the target restarts where the arm is
        lat: list[float] = []
        try:
            while not shutdown.is_set():
                tick = time.perf_counter()
                if trial is not None and tick - trial["t0"] >= trial["timeout"]:
                    trial["reason"] = "timeout"
                    break
                robot.send_action(policy(robot.get_observation()))
                lat.append(time.perf_counter() - tick)
                if len(lat) % (5 * fps) == 0:
                    print(f"student {len(lat)} steps, read+infer+send {statistics.mean(lat) * 1000:.0f} ms",
                          flush=True)
                precise_sleep(max(1.0 / fps - (time.perf_counter() - tick), 0.0))
        except Exception as e:  # noqa: BLE001 - keep the worker alive for another run
            print(f"RUN_ERROR: {e}", flush=True)
        finally:
            if trial is not None:
                if preview is not None:
                    preview.record(None)
                print("EVAL_DONE " + json.dumps({
                    "duration": round(time.perf_counter() - trial["t0"], 2), "reason": trial["reason"],
                    "inferences": len(lat),
                    "latency_ms_mean": round(statistics.mean(lat) * 1000) if lat else None,
                    "latency_ms_p95": round(float(np.percentile(lat, 95)) * 1000) if lat else None,
                    "video": trial["video"]}), flush=True)
                state["trial"] = None
            print("RUN_STOP", flush=True)

    def start() -> threading.Thread:
        shutdown.clear()
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    try:
        for line in sys.stdin:
            cmd = line.strip()
            busy = run_thread is not None and run_thread.is_alive()
            if cmd == "run":
                if not busy:
                    run_thread = start()
            elif cmd.startswith("eval "):
                if busy:
                    print("EVAL_ERROR: a run is already going", flush=True)
                    continue
                _, timeout_s, video = cmd.split(maxsplit=2)
                state["trial"] = {"timeout": float(timeout_s), "video": video}
                run_thread = start()
            elif cmd == "stop":
                shutdown.set()
            elif cmd.startswith("actions "):
                print("ACTION_STEPS_SET 1 (the student acts every frame)", flush=True)
            elif cmd == "home":
                shutdown.set()
                if busy:
                    run_thread.join(timeout=20)
                try:
                    ow.do_home(robot)
                except Exception as e:  # noqa: BLE001
                    print(f"HOME_ERROR: {e}", flush=True)
            elif cmd == "quit":
                shutdown.set()
                break
    finally:
        shutdown.set()
        if run_thread is not None and run_thread.is_alive():
            run_thread.join(timeout=20)
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        print("UNLOADED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
