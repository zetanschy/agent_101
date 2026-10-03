#!/usr/bin/env python3
"""Score an openpi checkpoint on the live MuJoCo sim: N trials, graded by the sim itself.

    ./robot sim-eval --policy /checkpoints/openpi_pi05_lora_cap_to_mug_sim_50 --episodes 100
    ./robot sim-eval --report outputs/sim_eval/<run>              # rebuild a run's report
    ./robot sim-eval --compare outputs/sim_eval/<a> outputs/sim_eval/<b>

EVERY TRIAL IS THE SAME STAGE FOR EVERY POLICY. Trial i resets the sim with seed
(--seed + i), so the cap and mug start at the same spots, the cap up the same way and the
handle at the same angle, whichever checkpoint runs it. That makes the comparison
paired: two runs disagree on a trial only because the policies differ.

THE LOOP IS THE WEB UI'S. Each trial drives webui/openpi_worker.py's control loop: the
same ChunkSchedule (scripts/openpi/chunk_loop.py), real-time chunking, 15 of every 50
actions and degrees, at 30 Hz against the sim's realtime clock. That is the configuration
known to drive these checkpoints well on the arm (evals/README.md), so a score here means
the policy as deployed, not a friendlier loop.

GRADING. Success = every cap inside the mug cavity (the sim's metrics.cap_in_mug: radius
and height inside the mug), held for --settle seconds. A trial ends at that point or at
--timeout seconds of sim time. A failure gets one cause from the cap and mug trajectories
(engine status, ~10 Hz):
    mug knocked over          the mug's up-axis z fell below 0.8
    never picked up           the cap never rose 2 cm and moved < 3 cm
    pushed the cap away       moved >= 3 cm without being lifted
    still holding at timeout  lifted, and still in the air when time ran out
    dropped on the way        lifted, then landed > 6 cm from the mug
    missed the mug            lifted, released within 6 cm of the mug, not inside it
    cap fell off the table    the cap ended 5 cm below where it started

OUTPUT: outputs/sim_eval/<run>/
    trials.jsonl     one line per trial: stage, outcome, cause, timings, latencies
    report.md        success rate (95% Wilson interval), by cause, by where things started
    map.jpg          every trial's start on the overhead camera (cap dot, mug ring),
                     coloured by outcome
    videos/          every failure, plus the first few successes (front | grip, 10 fps)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "sim_eval"
FPS = 30
TRACE_EVERY = 3  # status + video frame every 3rd control tick: 10 Hz
LIFT, MOVED, NEAR_MUG, FELL = 0.02, 0.03, 0.06, 0.05  # metres
CAUSE_COLORS = {  # BGR
    "success": (80, 200, 80),
    "never picked up": (60, 60, 230),
    "pushed the cap away": (40, 140, 255),
    "dropped on the way": (220, 80, 220),
    "missed the mug": (0, 220, 255),
    "still holding at timeout": (255, 160, 40),
    "mug knocked over": (160, 160, 160),
    "cap fell off the table": (90, 90, 140),
    "other": (255, 255, 255),
}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # @dataclass looks its module up here
    spec.loader.exec_module(mod)
    return mod


# --- scoring -------------------------------------------------------------------------
def classify(trace: list[dict], success: bool) -> tuple[str, dict]:
    """One cause per failed trial, from the cap/mug trajectory (see the module doc)."""
    cap = np.array([t["cap"] for t in trace])
    mug = np.array([t["mug"] for t in trace])
    up = np.array([t["mug_up"] for t in trace])
    z0, xy0 = cap[0, 2], cap[0, :2]
    lift = cap[:, 2] - z0
    moved = np.linalg.norm(cap[:, :2] - xy0, axis=1)
    d_mug = np.linalg.norm(cap[:, :2] - mug[:, :2], axis=1)
    lifted = np.where(lift > LIFT)[0]
    f = {"max_lift_cm": round(float(lift.max()) * 100, 1), "max_moved_cm": round(float(moved.max()) * 100, 1),
         "final_cap_mug_cm": round(float(d_mug[-1]) * 100, 1),
         "mug_moved_cm": round(float(np.linalg.norm(mug[-1, :2] - mug[0, :2])) * 100, 1),
         "mug_up_min": round(float(up.min()), 3),
         "t_first_lift": round(float(trace[lifted[0]]["t"]), 2) if len(lifted) else None}
    if success:
        return "success", f
    if up.min() < 0.8:
        return "mug knocked over", f
    if lift[-1] < -FELL:
        return "cap fell off the table", f
    if not len(lifted):
        return ("pushed the cap away" if moved.max() >= MOVED else "never picked up"), f
    if lift[-1] > LIFT:
        return "still holding at timeout", f
    return ("missed the mug" if d_mug[-1] <= NEAR_MUG else "dropped on the way"), f


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


# --- running -------------------------------------------------------------------------
class Sim:
    """Control requests on a socket of their own (reset with a seed, status), next to the
    robot plugin's: the server takes several clients."""

    def __init__(self):
        from real2sim.live import protocol

        self.p = protocol
        self.sock = protocol.connect(os.environ["ROBOT_PORT"], 60.0)
        st = self.status()
        if "cap_xyz" not in st:
            raise SystemExit("this sim server predates sim_eval (no cap_xyz in its status): stop the "
                             "running sim (the web UI / teleop using it) so `./robot sim-eval` starts a fresh one")
        if st.get("clock") != "realtime":
            raise SystemExit(f"sim clock is {st.get('clock')}; the eval needs realtime (MuJoCo's default)")

    def reset(self, seed: int) -> dict:
        return self.p.call(self.sock, "reset", layout="random:1", seed=int(seed))["layout"]

    def status(self) -> dict:
        return self.p.call(self.sock, "status")


def run(a) -> pathlib.Path:
    ev = _load("openpi_evaluate", "scripts/openpi/evaluate.py")
    chunk_loop = _load("openpi_chunk_loop", "scripts/openpi/chunk_loop.py")
    import cv2
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot_robot_real2sim import Real2Sim, Real2SimConfig
    from openpi.policies.rtc import RealTimeChunker
    from openpi.training import config as pi0_config

    name = a.name or f"{pathlib.Path(a.policy.rstrip('/')).name}_seed{a.seed}_n{a.episodes}"
    out = OUT / name
    (out / "videos").mkdir(parents=True, exist_ok=True)
    done = {}
    if (out / "trials.jsonl").exists() and not a.overwrite:
        done = {r["trial"]: r for r in map(json.loads, (out / "trials.jsonl").read_text().splitlines()) if r}
        print(f"resuming {out}: {len(done)} trials already scored", flush=True)
    elif (out / "trials.jsonl").exists():
        (out / "trials.jsonl").unlink()
    sim = Sim()
    run_info = {k: v for k, v in vars(a).items() if k not in ("report", "compare")}
    run_info["sim_dr"] = sim.status().get("dr_level", "off")  # ./robot sim-eval --sim-dr LEVEL
    (out / "run.json").write_text(json.dumps(run_info, indent=2))
    cfg = pi0_config.get_config(a.config or ev.infer_config(a.policy))
    policy = ev.load_policy(cfg, a.policy)
    robot = Real2Sim(Real2SimConfig(port=os.environ["ROBOT_PORT"], id="sim_eval", cameras=None, use_degrees=True))
    os.environ["R2S_NEW_LAYOUT_EACH_EPISODE"] = "0"  # this script places every stage itself
    robot.connect()
    chunker = RealTimeChunker(policy, prefix_attention_schedule="exp", max_guidance_weight=5.0, jacobian="identity")
    raw = robot.get_observation()
    warm = ev.build_observation(robot, raw, a.task)
    t = time.perf_counter()
    policy.infer(warm)
    chunker.infer(warm, prefix_start=0, inference_delay=1)
    chunker.infer(warm, prefix_start=1, inference_delay=1)
    chunker.reset()
    print(f"loaded {a.policy} (config {cfg.name}); warm-up {time.perf_counter() - t:.0f} s", flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    motors = list(robot.action_features)
    kept_successes = sum(1 for r in done.values() if r["success"] and r.get("video"))
    n_ok, n_run = sum(r["success"] for r in done.values()), len(done)

    def infer_only(obs, prefix_start, delay):
        t0 = time.perf_counter()
        return chunker.infer(obs, prefix_start=prefix_start, inference_delay=delay)["actions"], time.perf_counter() - t0

    for i in range(a.episodes):
        if i in done:
            continue
        seed = a.seed + i
        layout = sim.reset(seed)
        chunker.reset()
        sched = chunk_loop.ChunkSchedule(a.actions, overlap=True, rtc=True)
        chunk, pending, promised, lat = None, None, 0, []
        trace, frames, in_since, success, t_success = [], [], None, False, None
        st0 = sim.status()
        t_start, tick_n = st0["t"], 0
        while True:
            tick = time.perf_counter()
            if tick_n % TRACE_EVERY == 0:
                st = sim.status()
                ts = st["t"] - t_start
                trace.append({"t": round(ts, 2), "cap": st["cap_xyz"][0], "mug": st["mug_xyz"], "mug_up": st["mug_up_z"],
                              "in": bool(all(st["caps_in_mug"]))})
                if trace[-1]["in"]:
                    in_since = ts if in_since is None else in_since
                    if ts - in_since >= a.settle:
                        success, t_success = True, in_since
                        break
                else:
                    in_since = None
                if ts >= a.timeout:
                    break
                obs = robot.get_observation()
                if tick_n == 0 and not (out / "stage.jpg").exists():
                    cv2.imwrite(str(out / "stage.jpg"), cv2.cvtColor(obs["front"], cv2.COLOR_RGB2BGR))
                frames.append(np.concatenate([cv2.resize(obs["front"], (320, 240)), cv2.resize(obs["grip"], (320, 240))], 1))
            sched.set_horizon(a.actions, in_flight=pending is not None)
            plan = sched.plan(in_flight=pending is not None)
            if plan.action == "infer":
                chunk, dt = infer_only(ev.build_observation(robot, robot.get_observation(), a.task),
                                       plan.request.prefix_start, plan.request.inference_delay)
                sched.adopt(len(chunk), delay=plan.request.inference_delay)
                lat.append(dt)
            elif plan.action == "collect":
                chunk, dt = pending.result()
                pending = None
                sched.adopt(len(chunk), delay=promised)
                lat.append(dt)
            else:
                if plan.request is not None:
                    promised = plan.request.inference_delay
                    pending = pool.submit(infer_only, ev.build_observation(robot, robot.get_observation(), a.task),
                                          plan.request.prefix_start, promised)
                act = chunk[sched.take()]
                robot.send_action({m: float(act[k]) for k, m in enumerate(motors) if k < len(act)})
            tick_n += 1
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - tick), 0.0))
        if pending is not None:
            pending.result()  # let it land: the next trial's chunker.reset() must not race it
        cause, feats = classify(trace, success)
        rec = {"trial": i, "seed": seed, "success": success, "cause": cause,
               "t_success": round(t_success, 2) if t_success is not None else None,
               "duration": trace[-1]["t"], "cap_xy": layout["caps"][0]["xy"], "cap_up": layout["caps"][0]["up"],
               "mug_xy": layout["mug"]["xy"], "handle_yaw_deg": layout["mug"]["handle_yaw_deg"],
               "cap_xyz0": trace[0]["cap"], "mug_xyz0": trace[0]["mug"],
               "inferences": len(lat), "latency_ms_mean": round(statistics.mean(lat) * 1000) if lat else None,
               "latency_ms_p95": round(float(np.percentile(lat, 95)) * 1000) if lat else None, **feats}
        if not success or kept_successes < a.keep_successes:
            rec["video"] = f"videos/trial_{i:03d}_{cause.replace(' ', '_')}.mp4"
            write_video(out / rec["video"], frames, f"trial {i}  {cause}")
            kept_successes += success
        with open(out / "trials.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        n_ok, n_run = n_ok + success, n_run + 1
        print(f"trial {i:3d} seed {seed}: {'OK  ' if success else 'FAIL'} {cause:24s} "
              f"{(rec['t_success'] or rec['duration']):5.1f} s  latency {rec['latency_ms_mean']} ms   "
              f"[{n_ok}/{n_run}]", flush=True)
    pool.shutdown(wait=False)
    robot.disconnect()
    report(out)
    return out


def write_video(path: pathlib.Path, frames: list, label: str) -> None:
    import cv2

    if not frames:
        return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS / TRACE_EVERY, (w, h))
    for k, f in enumerate(frames):
        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        cv2.putText(f, f"{label}  t={k * TRACE_EVERY / FPS:4.1f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(f)
    vw.release()


# --- reporting -----------------------------------------------------------------------
def load_trials(d: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in (d / "trials.jsonl").read_text().splitlines() if l.strip()]


def polar(xy) -> tuple[float, float]:
    """Base frame (base -y is forward): distance (m) and azimuth (deg from straight ahead,
    signed like base x)."""
    x, y = xy
    return math.hypot(x, y), math.degrees(math.atan2(x, -y))


def x_is_image_right() -> bool:
    """Which way base +x points in the overhead image, so sides read as the picture does."""
    cam = front_camera()
    if cam is None:
        return True
    return bool(cam.project(np.array([0.1, -0.25, 0.03]))[0] > cam.project(np.array([-0.1, -0.25, 0.03]))[0])


def bins_table(trials, key, edges, labels, title) -> list[str]:
    rows = [f"| {title} | trials | success |", "|---|---|---|"]
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = [t for t in trials if lo <= key(t) < hi]
        k = sum(t["success"] for t in sel)
        rows.append(f"| {lab} | {len(sel)} | {k}/{len(sel)} ({100 * k / len(sel):.0f}%) |" if sel else f"| {lab} | 0 | — |")
    return rows


def front_camera():
    try:
        from real2sim import scene

        return scene.load().camera("front")
    except Exception as e:  # noqa: BLE001 - the map falls back to a schematic
        print(f"(no camera model for the map: {e})")
        return None


def draw_map(trials, path: pathlib.Path, title: str) -> None:
    """Every trial's start on the overhead view: cap = filled dot, mug = ring, joined by a
    line, coloured by outcome. Drawn on a clean render of the table when the camera model
    loads, else on a top-down schematic in the base frame."""
    import cv2

    cam = front_camera()
    bg = None
    if cam is not None and (path.parent / "stage.jpg").exists():
        bg = cv2.imread(str(path.parent / "stage.jpg"))
    W, H = (cam.width, cam.height) if cam is not None else (640, 480)
    img = (bg // 3) if bg is not None else np.full((H, W, 3), 24, np.uint8)  # dimmed: markers read first

    def px(p3):
        if cam is not None:
            u, v = cam.project(np.asarray(p3, float))
            return int(round(u)), int(round(v))
        x, y = p3[0], p3[1]  # schematic: 1 px = 1.2 mm, base at the bottom centre
        return int(W / 2 + x / 0.0012), int(H - 20 + y / 0.0012)

    if cam is None:
        for r in (0.14, 0.34):
            cv2.ellipse(img, px((0, 0, 0)), (int(r / 0.0012),) * 2, 0, 180, 360, (70, 70, 70), 1)
        cv2.circle(img, px((0, 0, 0)), 6, (200, 200, 200), -1)
    for t in trials:
        col = CAUSE_COLORS.get(t["cause"], (255, 255, 255))
        c, m = px(t["cap_xyz0"]), px(t["mug_xyz0"])
        cv2.line(img, c, m, col, 1, cv2.LINE_AA)
        cv2.circle(img, m, 9, col, 2, cv2.LINE_AA)
        cv2.circle(img, c, 5, col, -1, cv2.LINE_AA)
        if not t["success"]:
            cv2.putText(img, str(t["trial"]), (c[0] + 6, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    counts = {}
    for t in trials:
        counts[t["cause"]] = counts.get(t["cause"], 0) + 1
    legend = sorted(counts.items(), key=lambda kv: (kv[0] != "success", -kv[1]))
    pad = 18 * (len(legend) + 3)
    canvas = np.concatenate([img, np.full((pad, img.shape[1], 3), 16, np.uint8)], 0)
    cv2.putText(canvas, title, (8, img.shape[0] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "dot = cap start, ring = mug start, line joins them; failures numbered by trial",
                (8, img.shape[0] + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    for k, (cause, n) in enumerate(legend):
        y = img.shape[0] + 52 + 18 * k
        cv2.circle(canvas, (14, y - 4), 5, CAUSE_COLORS.get(cause, (255, 255, 255)), -1)
        cv2.putText(canvas, f"{cause}: {n}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def summary_lines(trials, title: str) -> list[str]:
    n, k = len(trials), sum(t["success"] for t in trials)
    lo, hi = wilson(k, n)
    ok = [t for t in trials if t["success"]]
    lat = [t["latency_ms_mean"] for t in trials if t["latency_ms_mean"]]
    L = [f"# {title}", "",
         f"**Success: {k}/{n} = {100 * k / max(n, 1):.0f}%** (95% interval {100 * lo:.0f}–{100 * hi:.0f}%)", ""]
    if ok:
        ts = [t["t_success"] for t in ok]
        L.append(f"Time to success: median {statistics.median(ts):.1f} s, range {min(ts):.1f}–{max(ts):.1f} s. "
                 f"Inference latency (mean per trial): median {statistics.median(lat):.0f} ms.")
        L.append("")
    causes = {}
    for t in trials:
        causes.setdefault(t["cause"], []).append(t["trial"])
    L += ["## Outcomes", "", "| outcome | trials | which |", "|---|---|---|"]
    for cause, ids in sorted(causes.items(), key=lambda kv: (kv[0] != "success", -len(kv[1]))):
        which = "" if cause == "success" else ", ".join(map(str, ids[:20])) + (" …" if len(ids) > 20 else "")
        L.append(f"| {cause} | {len(ids)} | {which} |")
    L += ["", "## Where it fails", ""]
    L += bins_table(trials, lambda t: polar(t["cap_xy"])[0], [0.13, 0.20, 0.27, 0.35],
                    ["near (14–20 cm)", "mid (20–27 cm)", "far (27–34 cm)"], "cap distance from the base") + [""]
    neg, pos = ("image left", "image right") if x_is_image_right() else ("image right", "image left")
    L += bins_table(trials, lambda t: polar(t["cap_xy"])[1], [-71, -25, 25, 71],
                    [f"{neg} (25–70° off centre)", "centre (±25°)", f"{pos} (25–70° off centre)"],
                    "cap direction (overhead view)") + [""]
    L += bins_table(trials, lambda t: polar(t["mug_xy"])[0], [0.15, 0.21, 0.26, 0.31],
                    ["near (16–21 cm)", "mid (21–26 cm)", "far (26–30 cm)"], "mug distance from the base") + [""]
    L += bins_table(trials, lambda t: math.dist(t["cap_xy"], t["mug_xy"]), [0.0, 0.2, 0.3, 1.0],
                    ["< 20 cm", "20–30 cm", "> 30 cm"], "cap-to-mug distance") + [""]
    L += bins_table(trials, lambda t: 0 if t["cap_up"] == "open" else 1, [0, 1, 2],
                    ["open side up", "closed side up"], "cap orientation") + [""]
    return L


def report(d: pathlib.Path) -> None:
    trials = load_trials(d)
    run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
    real = run.get("where") == "real"
    title = f"{'Real-arm' if real else 'Sim'} eval: {pathlib.Path(run.get('policy', d.name)).name}"
    k = sum(t["success"] for t in trials)
    draw_map(trials, d / "map.jpg", f"{title}  ({k}/{len(trials)}): every trial")
    draw_map([t for t in trials if not t["success"]], d / "map_failures.jpg",
             f"{title}: the {len(trials) - k} failures, by cause")
    chown_like_repo(d)
    L = summary_lines(trials, title)
    L += ["## Maps", "", "Where each trial started, on the overhead camera:", "",
          "![every trial](map.jpg)", "", "![failures only](map_failures.jpg)", "",
          "## Failure videos", ""]
    L += [f"- trial {t['trial']}: {t['cause']} — [{t['video']}]({t['video']})" for t in trials
          if not t["success"] and t.get("video")]
    graded = (f"graded by the operator ({sum(t.get('auto') == ('in' if t['success'] else 'out') for t in trials)}"
              f"/{len(trials)} agree with the camera check)" if run.get("settle") == "operator"
              else f"success held {run.get('settle')} s")
    L += ["", f"Setup: {len(trials)} trials of {run.get('episodes')}, seeds {run.get('seed')}–"
          f"{run.get('seed', 0) + run.get('episodes', 0) - 1}, timeout {run.get('timeout')} s, {graded}, "
          f"task \"{run.get('task')}\", RTC with {run.get('actions')} of 50 actions, degrees, "
          f"{'the real arm' if real else run.get('where') or 'MuJoCo realtime'}"
          f"{'' if run.get('sim_dr', 'off') == 'off' else ', domain randomization ' + run['sim_dr']}."]
    (d / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(summary_lines(trials, title)[:4]))
    print(f"report: {d / 'report.md'}   map: {d / 'map.jpg'}")


def chown_like_repo(d: pathlib.Path) -> None:
    """The container runs as root; hand the run back to whoever owns the repo."""
    st = ROOT.stat()
    for p in [OUT, d, *d.rglob("*")]:
        try:
            os.chown(p, st.st_uid, st.st_gid)
        except OSError:
            pass


def compare(da: pathlib.Path, db: pathlib.Path) -> None:
    A, B = {t["seed"]: t for t in load_trials(da)}, {t["seed"]: t for t in load_trials(db)}
    seeds = sorted(set(A) & set(B))
    ra, rb = json.loads((da / "run.json").read_text()), json.loads((db / "run.json").read_text())
    na, nb = pathlib.Path(ra["policy"].rstrip("/")).name, pathlib.Path(rb["policy"].rstrip("/")).name
    if na == nb or ra.get("where") != rb.get("where"):  # the same model on the arm and in the sim
        na, nb = f"{na} ({ra.get('where', 'sim')})", f"{nb} ({rb.get('where', 'sim')})"
    both = [s for s in seeds if A[s]["success"] and B[s]["success"]]
    a_only = [s for s in seeds if A[s]["success"] and not B[s]["success"]]
    b_only = [s for s in seeds if B[s]["success"] and not A[s]["success"]]
    neither = [s for s in seeds if not A[s]["success"] and not B[s]["success"]]
    L = [f"# {na} vs {nb}", "", f"The same {len(seeds)} stages (seeds {seeds[0]}–{seeds[-1]}) for both.", "",
         "| | " + na + " | " + nb + " |", "|---|---|---|"]
    ka, kb = sum(A[s]["success"] for s in seeds), sum(B[s]["success"] for s in seeds)
    (la, ha), (lb, hb) = wilson(ka, len(seeds)), wilson(kb, len(seeds))
    L.append(f"| success | **{ka}/{len(seeds)}** ({100 * la:.0f}–{100 * ha:.0f}%) | "
             f"**{kb}/{len(seeds)}** ({100 * lb:.0f}–{100 * hb:.0f}%) |")
    for lab, f in (("median time to success (s)", lambda X: statistics.median([X[s]["t_success"] for s in seeds if X[s]["success"]] or [float("nan")])),
                   ("median inference latency (ms)", lambda X: statistics.median([X[s]["latency_ms_mean"] for s in seeds]))):
        L.append(f"| {lab} | {f(A):.1f} | {f(B):.1f} |")
    causes = sorted({X[s]["cause"] for X in (A, B) for s in seeds} - {"success"})
    for c in causes:
        L.append(f"| {c} | {sum(A[s]['cause'] == c for s in seeds)} | {sum(B[s]['cause'] == c for s in seeds)} |")
    L += ["", "| paired | stages |", "|---|---|", f"| both succeed | {len(both)} |", f"| only {na} | {len(a_only)} |",
          f"| only {nb} | {len(b_only)} |", f"| neither | {len(neither)} |", ""]
    if len(a_only) + len(b_only):  # exact two-sided sign test on the discordant stages
        n, k = len(a_only) + len(b_only), min(len(a_only), len(b_only))
        p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
        L.append(f"Sign test on the {n} stages where they disagree: p = {p:.3g}.")
    out = da.parent / f"compare_{da.name}__vs__{db.name}.md"
    out.write_text("\n".join(L) + "\n")
    chown_like_repo(out)
    print("\n".join(L))
    print(f"\n-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy")
    ap.add_argument("--task", default="Put the cap into the red cup")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000, help="trial i uses seed+i (the same stages for every policy)")
    ap.add_argument("--timeout", type=float, default=45.0, help="sim seconds per trial (demos took 14-30 s)")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds the cap must stay in the mug")
    ap.add_argument("--actions", type=int, default=15, help="actions executed per 50-action chunk (the web UI's)")
    ap.add_argument("--keep-successes", type=int, default=3, help="success videos to keep (all failures are kept)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default=None, help="run directory under outputs/sim_eval/")
    ap.add_argument("--overwrite", action="store_true", help="start over instead of resuming the run directory")
    ap.add_argument("--report", metavar="RUN_DIR")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    a = ap.parse_args()
    if a.report:
        report(pathlib.Path(a.report))
    elif a.compare:
        compare(*map(pathlib.Path, a.compare))
    elif a.policy:
        run(a)
    else:
        ap.error("need --policy, --report or --compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
