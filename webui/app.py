#!/usr/bin/env python3
"""Web UI backend for the SO-ARM101 policy.

Inference uses a PERSISTENT worker (webui/infer_worker.py): you 'Load' a model
once (slow pi05 load happens here) and then Start/Stop the rollout loop as many
times as you like without reloading. 'Home' runs webui/home.py (mutually
exclusive with a loaded model, since both drive the robot).
"""
import ast
import glob
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent          # /workspace
WEBUI = ROOT / "webui"
LOGDIR = ROOT / "outputs"
LOGDIR.mkdir(parents=True, exist_ok=True)
LOGFILE = LOGDIR / "webui_run.log"
DEFAULT_POLICY = "zetanschy/pi05_lora_cap_tu_cup"
RENAME = ('{"observation.images.front": "observation.images.base_0_rgb", '
          '"observation.images.grip": "observation.images.left_wrist_0_rgb"}')

_FPS = float(os.environ.get("CAM_FPS", "30") or 30)
# Three shapes, because the two workers report differently: lerobot-rollout logs
# "running slower (N Hz)" / "real_delay=N" (ticks), the openpi worker logs its
# measured inference time directly in ms.
_LAT_RE = re.compile(r"running slower \(([0-9.]+) Hz\)|real_delay=([0-9]+)|inference ([0-9]+) ms")

app = FastAPI()
_lock = threading.Lock()
_worker: subprocess.Popen | None = None   # persistent inference worker
_loaded: dict | None = None               # signature of what's loaded
_home: subprocess.Popen | None = None      # one-shot home process


# ---------- helpers ----------
def latency_stats(log: str):
    """Latency samples (ms) for the CURRENT run (since the last RUN_START):
    last / mean / p50 / p95 / min / max / n."""
    start = log.rfind("RUN_START")
    seg = log[start:] if start >= 0 else log
    s = []
    for m in _LAT_RE.finditer(seg):
        if m.group(1):
            hz = float(m.group(1))
            if hz > 0:
                s.append(1000.0 / hz)
        elif m.group(2):
            s.append(int(m.group(2)) * (1000.0 / _FPS))
        elif m.group(3):
            s.append(float(m.group(3)))          # openpi: already milliseconds
    if not s:
        return None
    ss = sorted(s)

    def pct(p):
        return ss[min(len(ss) - 1, int(p / 100.0 * len(ss)))]

    return {"last": round(s[-1]), "mean": round(sum(ss) / len(ss)),
            "p50": round(pct(50)), "p95": round(pct(95)),
            "min": round(ss[0]), "max": round(ss[-1]), "n": len(ss)}


def cameras_json(n: int = 2) -> str:
    w = os.environ.get("CAM_WIDTH", "640"); h = os.environ.get("CAM_HEIGHT", "480")
    fps = os.environ.get("CAM_FPS", "30"); fourcc = os.environ.get("CAM_FOURCC", "MJPG")

    def frag(name, idx):
        return (f'{name}: {{type: opencv, index_or_path: {idx}, '
                f'width: {w}, height: {h}, fps: {fps}, fourcc: "{fourcc}"}}')

    parts = [frag("front", os.environ.get("CAM_FRONT_INDEX", "4")),
             frag("grip", os.environ.get("CAM_GRIP_INDEX", "6"))]
    if n >= 3:
        parts.append(frag("side", os.environ.get("CAM_SIDE_INDEX", "1")))
    return "{ " + ",".join(parts) + "}"


def openpi_available() -> bool:
    """Whether this image can run openpi policies (the main image can; see Dockerfile)."""
    try:
        import openpi.training.config  # noqa: F401

        return True
    except ImportError:
        return False


def rl_available() -> bool:
    """Whether this image can run mjlab RL exports (needs onnxruntime; see Dockerfile)."""
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


OPENPI = openpi_available()
# `./robot webui|openpi-webui --sim mujoco|isaac` points the robot at the live sim's socket
SIM = os.environ.get("ROBOT_TYPE") == "real2sim"

# The SO-101's joints, in URDF names — the vocabulary mjlab's exporter writes into the
# ONNX metadata. Sourced from kinematics.py so there is one definition of the arm.
ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")

# (path, mtime) -> (joint names, goal term), either of which may be None.
_joint_cache: dict[tuple, tuple] = {}


def onnx_meta(path: str) -> tuple:
    """(joint names, goal term) an ONNX export was trained for; either may be None.

    Cached on (path, mtime): /api/models is polled by the page, and re-opening every
    export on each poll would be pointless work. None means 'unknown', not 'none' —
    callers must not treat it as a mismatch.

    The goal term is the last of mjlab's observation_names, and it is what the page
    needs to know whether to ask for a yaw (Push-T's footprint) or a height (Lift-T's
    point in the air). rl_worker.py reads the same field for the same reason.
    """
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return (None, None)
    if key in _joint_cache:
        return _joint_cache[key]
    joints = goal_term = None
    try:
        import onnxruntime as ort

        meta = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        ).get_modelmeta().custom_metadata_map
        if meta.get("joint_names"):
            joints = tuple(s.strip() for s in meta["joint_names"].split(","))
        names = [s.strip() for s in meta.get("observation_names", "").split(",") if s.strip()]
        if names:
            goal_term = names[-1]
    except Exception:  # noqa: BLE001 - an unreadable export is 'unknown', not fatal
        joints = goal_term = None
    _joint_cache[key] = (joints, goal_term)
    return (joints, goal_term)


def eval_openpi_checkpoint() -> str | None:
    """The openpi checkpoint the eval harness actually runs, or None if it can't be read.

    There are several orbax directories under /checkpoints and only one of them drives
    this bench well; evals/openpi_policy.py names it, and its comment on that constant
    says it is the one "the web UI's dropdown shows as the working one". So this reads
    that file rather than keeping a second copy of the path here to drift.

    Read as TEXT, not imported: that module pulls in inspect_robots and JAX, neither of
    which exists in this container. None means 'could not tell' and the caller falls
    back to listing every checkpoint, which is the safe direction to fail in -- an
    empty dropdown is worse than a long one.
    """
    try:
        tree = ast.parse((ROOT / "evals" / "openpi_policy.py").read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "DEFAULT_CHECKPOINT"
                   for t in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


# mjlab's experiment directories, in words. Anything not here falls back to the
# directory name with its underscores opened out, so a new task is listed rather than
# hidden -- it just reads less well until it is added.
_RL_TASKS = {"lift_t": "lift T", "lift_cube": "lift cube", "push_t": "push T"}


def rl_label(path: str) -> str:
    """`so101_lift_t_vision/2026-09-07_01-04-45/...onnx` -> `lift T (vision) · Sep 07 01:04`.

    The old label was the last two path components, which for these exports is the
    timestamp TWICE -- mjlab names the file after the run directory it sits in -- and
    dropped the experiment name, the only part that says what the policy does.

    The time comes from the run directory's own name (when training started) rather
    than the file's mtime, so re-exporting a checkpoint does not relabel the run.
    """
    parts = Path(path).parts
    exp, run = (parts[-3], parts[-2]) if len(parts) >= 3 else ("", "")
    name = exp[len("so101_"):] if exp.startswith("so101_") else exp

    task, variant = name.replace("_", " "), ""
    for stem, pretty in _RL_TASKS.items():
        if name == stem or name.startswith(stem + "_"):
            task, variant = pretty, name[len(stem):].strip("_").replace("_", " ")
            break

    try:
        when = datetime.strptime(run, "%Y-%m-%d_%H-%M-%S").strftime("%b %d %H:%M")
    except ValueError:
        when = run
    return f"{task}{f' ({variant})' if variant else ''} · {when}"


def is_openpi_checkpoint(path: str) -> bool:
    """An orbax checkpoint directory, i.e. an openpi model rather than a lerobot one.

    Detected from the model itself, not from the image, so one UI can hold both kinds
    in the same dropdown and dispatch per selection. openpi writes params/ (weights)
    and assets/<repo_id>/norm_stats.json; lerobot writes config.json plus either
    model.safetensors or adapter_model.safetensors.
    """
    p = Path(path)
    return p.is_dir() and (p / "params").is_dir() and (p / "assets").is_dir()


def is_rl_policy(path: str) -> bool:
    """A mjlab-trained RL policy, i.e. an ONNX export rather than a VLA checkpoint.

    Detected from the artifact, like is_openpi_checkpoint: one dropdown holds all
    three stacks and Load dispatches per selection. mjlab writes exactly one .onnx per
    run, named after the run directory.
    """
    return path.endswith(".onnx")


def rl_args(p: dict):
    """Args for webui/rl_worker.py — no task string, no chunking; a goal instead.

    The RL policy was told where the goal is rather than seeing it (the footprint is
    1 mm tall and renders in a geom group the wrist camera never saw; the lift target
    is a point in the air that nothing can see), so on a real table the operator has
    to measure it. Everything else about the contract — joints, home pose, action
    scale, and which of the two goal terms this export wants — is read from the ONNX
    metadata by the worker. Both a yaw and a z go over regardless; the worker uses
    whichever its metadata calls for and ignores the other.
    """
    policy = p.get("policy") or ""
    gx = str(p.get("goal_x", "") or "0.23")
    gy = str(p.get("goal_y", "") or "0.0")
    gyaw = str(p.get("goal_yaw", "") or "0.0")
    gz = str(p.get("goal_z", "") or "0.12")
    args = [f"--policy={policy}", f"--goal-x={gx}", f"--goal-y={gy}",
            f"--goal-yaw={gyaw}", f"--goal-z={gz}"]
    if str(p.get("hz", "") or "").strip():
        args.append(f"--hz={p['hz']}")
    if str(p.get("max_deg_per_s", "") or "").strip():
        args.append(f"--max-deg-per-s={p['max_deg_per_s']}")
    if p.get("dry_run"):
        args.append("--dry-run")
    sig = {"stack": "rl", "policy": policy, "goal_x": gx, "goal_y": gy, "goal_yaw": gyaw,
           "goal_z": gz,
           "hz": str(p.get("hz", "") or ""), "max_deg_per_s": str(p.get("max_deg_per_s", "") or ""),
           "dry_run": bool(p.get("dry_run"))}
    return args, sig


def openpi_args(p: dict):
    """Args for webui/openpi_worker.py — a different stack, so a different arg set."""
    policy = p.get("policy") or DEFAULT_POLICY
    args = [f"--policy={policy}", f"--task={p['task']}", f"--units={p.get('units', 'degrees')}",
            f"--mode={p.get('op_mode', 'rtc')}"]
    if str(p.get("actions", "") or "").strip():
        args.append(f"--actions={p['actions']}")
    if str(p.get("config", "") or "").strip():
        args.append(f"--config={p['config']}")
    sig = {"stack": "openpi", "policy": policy, "task": p["task"],
           "units": p.get("units", "degrees"), "op_mode": p.get("op_mode", "rtc"),
           "actions": str(p.get("actions", "")),
           "config": str(p.get("config", "")), "cams": int(p.get("cams", 2))}
    return args, sig


def rollout_args(p: dict):
    policy = p.get("policy") or DEFAULT_POLICY
    mode = p.get("mode", "rtc")
    if mode not in ("sync", "rtc"):   # async isn't supported by the worker (rename)
        mode = "rtc"
    args = [
        f"--policy.path={policy}",
        f"--robot.type={os.environ.get('ROBOT_TYPE', 'so101_follower')}",
        f"--robot.port={os.environ.get('ROBOT_PORT', '/dev/ttyACM1')}",
        f"--robot.id={os.environ.get('ROBOT_ID', 'zetans_follower')}",
        f"--robot.cameras={cameras_json(int(p.get('cams', 2)))}",
        f"--task={p['task']}",
        f"--inference.type={mode}",
        "--policy.device=cuda", "--policy.dtype=bfloat16",
        "--display_data=false", "--duration=0",
        f"--rename_map={RENAME}",
    ]
    def num(key):
        return str(p.get(key, "") or "").strip()

    # --- policy knobs (both modes) ---
    if num("steps"):
        args.append(f"--policy.num_inference_steps={p['steps']}")
    # Actions executed per inference. pi05 ships 50, i.e. 1.7s open-loop at 30fps;
    # lower values re-plan more often, which matters most in sync.
    if num("action_steps"):
        args.append(f"--policy.n_action_steps={p['action_steps']}")

    # --- mode-specific ---
    # sync has no engine params of its own (SyncInferenceConfig is empty), so
    # everything tunable for sync lives on the policy above.
    if mode == "rtc":
        if num("horizon"):
            args.append(f"--inference.rtc.execution_horizon={p['horizon']}")
        if num("queue_threshold"):
            args.append(f"--inference.queue_threshold={p['queue_threshold']}")

    # --- execution (both modes) ---
    if num("interp"):
        args.append(f"--interpolation_multiplier={p['interp']}")
    if p.get("compile"):
        args.append("--use_torch_compile=true")

    rec = str(p.get("record", "") or "").strip()
    if rec:
        args += [f"--dataset.repo_id=eval_{rec}", f"--dataset.single_task={p['task']}"]
    # Every knob is part of the signature: changing any of them must force a
    # reload rather than silently reporting "already loaded".
    sig = {"policy": policy, "mode": mode, "task": p["task"],
           "steps": num("steps"), "action_steps": num("action_steps"),
           "horizon": num("horizon") if mode == "rtc" else "",
           "queue_threshold": num("queue_threshold") if mode == "rtc" else "",
           "interp": num("interp"), "compile": bool(p.get("compile")),
           "cams": int(p.get("cams", 2)), "record": rec}
    return args, sig


def _alive(proc):
    return proc is not None and proc.poll() is None


def _send(cmd: str):
    if _alive(_worker):
        try:
            _worker.stdin.write((cmd + "\n").encode())
            _worker.stdin.flush()
        except Exception:
            pass


def _log_text():
    return LOGFILE.read_text(errors="replace") if LOGFILE.exists() else ""


def _is_running(log: str) -> bool:
    return log.rfind("RUN_START") > log.rfind("RUN_STOP")


def _wait_marker(marker: str, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if marker in _log_text():
            return True
        if not _alive(_worker):
            return marker in _log_text()
        time.sleep(0.5)
    return False


def _wait_new_marker(marker: str, since_len: int, timeout: float) -> bool:
    """Wait for `marker` to appear in the log AFTER position since_len (avoids
    matching a stale marker from an earlier command)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if marker in _log_text()[since_len:]:
            return True
        if not _alive(_worker):
            return marker in _log_text()[since_len:]
        time.sleep(0.3)
    return False


def _err(msg, code=409):
    return JSONResponse({"ok": False, "msg": msg}, status_code=code)


def list_models():
    """All loadable policies, each tagged with the stack that can run it.

    openpi checkpoints are scanned from both places they turn up: /checkpoints (the
    hf_models mount, where downloaded ones land) and ./checkpoints/<config>/<exp>/<step>
    (openpi's own training output layout) -- and then narrowed to the ONE the eval
    harness runs, because the others are training artefacts that have never driven this
    bench well and picking one by accident costs a session. See eval_openpi_checkpoint.

    The RL exports are narrowed the same way and for the same reason, to the newest run
    of each experiment. Seventeen entries that differ only in a timestamp is a list you
    read past rather than read. Everything filtered out is still on disk and still
    loadable by pasting its path into the custom-path box.
    """
    out: list[dict] = [{"path": DEFAULT_POLICY, "kind": "lerobot"}]

    for p in sorted(glob.glob(str(ROOT / "outputs/train/*/"))):
        pp = Path(p)
        if (pp / "adapter_model.safetensors").exists() or (pp / "config.json").exists():
            out.append({"path": str(pp), "kind": "lerobot"})
    for p in sorted(glob.glob(str(ROOT / "outputs/train/*/checkpoints/*/"))):
        out.append({"path": str(Path(p)), "kind": "lerobot"})

    if OPENPI:
        # MARKED, NOT FILTERED. This used to list only the checkpoint evals run, which
        # kept the dropdown short right up until the next checkpoint was trained -- a
        # DAgger round lands in the same directory and simply never appeared, which
        # reads as "the web UI is broken" rather than as a filter doing its job. The
        # eval one is labelled and sorted first instead, so it is still obvious which
        # is which, and a new one is always visible.
        working = eval_openpi_checkpoint()
        found = []
        for pattern in ("/checkpoints/*", str(ROOT / "checkpoints/*/*/*")):
            for p in sorted(glob.glob(pattern)):
                if is_openpi_checkpoint(p):
                    found.append(Path(p))
        is_eval = (lambda q: working is not None and str(q) == str(Path(working)))
        # Newest first, but the eval checkpoint pinned to the top whatever its age.
        found.sort(key=lambda q: (not is_eval(q), -os.path.getmtime(q)))
        for q in found:
            out.append({"path": str(q), "kind": "openpi",
                        "label": q.name + (" · eval" if is_eval(q) else "")})

    # mjlab RL exports. thirdparty/ is inside the ./:/workspace mount, so the training
    # runs are visible in here without a mount of their own. Sorted by mtime, not by
    # path: the paths start with the experiment name, so a string sort buries the run
    # you just finished under whichever task sorts first.
    rl = glob.glob(str(ROOT / "thirdparty/mjlab/logs/rsl_rl/*/*/*.onnx"))
    seen_exp: set = set()
    for p in sorted(rl, key=lambda f: os.path.getmtime(f), reverse=True):
        joints, goal_term = onnx_meta(p)
        # Only offer policies whose joints ARE this arm's. mjlab trains other robots in
        # the same tree -- the YAM push-T runs sit right beside the SO-101 ones -- and
        # a 7-joint policy pointed at a 6-joint bus is not a mistake to make on
        # hardware. Unknown joints (no onnxruntime yet) are listed rather than hidden,
        # since the worker rejects a mismatch anyway.
        #
        # Filtered BEFORE the per-experiment dedup, so a robot that is not this one
        # cannot take up its experiment's one slot and hide it.
        if joints is not None and set(joints) != set(ARM_JOINTS):
            continue
        exp = Path(p).parts[-3]
        if exp in seen_exp:      # newest run wins; this loop is already newest-first
            continue
        seen_exp.add(exp)
        out.append({"path": str(Path(p)), "kind": "rl", "goal": goal_term,
                    "label": rl_label(p)})

    seen, uniq = set(), []
    for m in out:
        if m["path"] not in seen:
            seen.add(m["path"])
            uniq.append(m)
    return uniq


# ---------- endpoints ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return (WEBUI / "index.html").read_text()


@app.get("/api/models")
def models():
    return {"models": list_models(), "default": DEFAULT_POLICY, "openpi": OPENPI}


@app.get("/api/capabilities")
def capabilities():
    """Which stacks this image supports; the page dispatches per selected model."""
    return {"openpi": OPENPI, "lerobot": True, "rl": rl_available()}


@app.get("/api/status")
def status():
    log = _log_text()
    return {
        "loaded": _alive(_worker),
        "loaded_sig": _loaded,
        "running": _alive(_worker) and _is_running(log),
        "homing": _alive(_home),
        "sim": SIM,
        "latency": latency_stats(log),
        "log": log[-8000:],
    }


@app.post("/api/load")
def load(body: dict = Body(...)):
    # An RL policy has no task string — it was trained against one reward, not
    # prompted — so the requirement is checked per stack rather than up front.
    if not is_rl_policy((body.get("policy") or "").strip()) and not (body.get("task") or "").strip():
        return _err("task is required", 400)
    if _alive(_home):
        return _err("busy: homing — wait for it to finish", 409)
    policy_in = (body.get("policy") or "").strip()
    # A local path that does not exist would otherwise be handed to the lerobot
    # worker, which fails much later with an opaque draccus RolloutConfig error.
    if policy_in.startswith("/") and not Path(policy_in).exists():
        return _err(f"no such path in this container: {policy_in} "
                    "(is it mounted? openpi checkpoints come from /checkpoints)", 400)
    if is_rl_policy(policy_in):
        if not rl_available():
            return _err("this image cannot run RL policies — onnxruntime is missing "
                        "(rebuild: ./robot build)", 400)
        args, sig = rl_args(body)
        worker_script = "webui/rl_worker.py"
    elif is_openpi_checkpoint(policy_in):
        if not OPENPI:
            return _err("this image cannot run openpi policies (rebuild: ./robot build)", 400)
        args, sig = openpi_args(body)
        worker_script = "webui/openpi_worker.py"
    else:
        args, sig = rollout_args(body)
        worker_script = "webui/infer_worker.py"
    global _worker, _loaded
    with _lock:
        if _alive(_worker) and _loaded == sig:
            return {"ok": True, "msg": "already loaded", "loaded": sig}
        if _alive(_worker):          # different config -> unload old first
            _send("quit")
            try:
                _worker.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(_worker.pid), signal.SIGKILL)
                except Exception:
                    pass
        LOGFILE.write_text("")
        fh = open(LOGFILE, "ab", buffering=0)
        _worker = subprocess.Popen(
            ["python", worker_script, *args], cwd=str(ROOT),
            stdin=subprocess.PIPE, stdout=fh, stderr=subprocess.STDOUT,
            env=os.environ.copy(), start_new_session=True,
        )
        _loaded = None
    if _wait_marker("MODEL_LOADED", 300):
        _loaded = sig
        return {"ok": True, "loaded": sig}
    return _err("load failed — see the log", 500)


@app.post("/api/infer")
def infer():
    if not _alive(_worker):
        return _err("no model loaded — press Load first", 409)
    _send("run")
    return {"ok": True}


@app.post("/api/stop")
def stop():
    _send("stop")           # stops the loop, keeps the model resident
    return {"ok": True}


@app.post("/api/unload")
def unload():
    global _worker, _loaded
    with _lock:
        if _alive(_worker):
            _send("quit")
            try:
                _worker.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(_worker.pid), signal.SIGKILL)
                except Exception:
                    pass
        _worker = None
        _loaded = None
    return {"ok": True}


@app.post("/api/steps")
def steps(body: dict = Body(...)):
    if not _alive(_worker):
        return _err("no model loaded", 409)
    n = str(body.get("steps", "") or "").strip()
    if not n:
        return _err("steps required", 400)
    since = len(_log_text())
    _send(f"steps {int(n)}")
    _wait_new_marker("STEPS_SET", since, 5)
    if _loaded is not None:          # so a later Load with the same value isn't a needless reload
        _loaded["steps"] = str(int(n))
    return {"ok": True, "steps": int(n)}


@app.post("/api/action-steps")
def action_steps(body: dict = Body(...)):
    """Live-change the sync open-loop window without reloading the model."""
    if not _alive(_worker):
        return _err("no model loaded", 409)
    n = str(body.get("action_steps", "") or "").strip()
    if not n:
        return _err("action_steps required", 400)
    since = len(_log_text())
    # Same knob, different worker vocabulary: lerobot retunes n_action_steps on the
    # policy, openpi retunes how much of the predicted chunk the loop executes.
    # Same knob, different worker vocabulary. The loaded signature records which
    # stack is running, so this follows the model that is actually loaded.
    is_op = bool(_loaded and _loaded.get("stack") == "openpi")
    _send(f"actions {int(n)}" if is_op else f"actionsteps {int(n)}")
    _wait_new_marker("ACTION_STEPS_SET", since, 5)
    if _loaded is not None:          # keep the signature honest so Load isn't skipped later
        _loaded["action_steps"] = str(int(n))
    return {"ok": True, "action_steps": int(n)}


@app.post("/api/goal")
def goal(body: dict = Body(...)):
    """Live-move the RL policy's goal without reloading.

    For Push-T the goal is where the printed footprint physically sits, so this is the
    knob you reach for most: slide the footprint on the table, retype the numbers, keep
    running. For Lift-T it is the point in the air to lift the block to, and moving it
    live is how you find a height this arm can actually hold.

    z is optional so an older page — or a caller that only ever drove Push-T — keeps
    working; the worker then leaves the height where it was loaded with.
    """
    if not _alive(_worker):
        return _err("no model loaded", 409)
    if not (_loaded and _loaded.get("stack") == "rl"):
        return _err("goal applies to RL policies only", 409)
    try:
        x = float(body.get("goal_x")); y = float(body.get("goal_y"))
        yaw = float(body.get("goal_yaw"))
    except (TypeError, ValueError):
        return _err("goal_x, goal_y and goal_yaw must be numbers", 400)
    z = body.get("goal_z")
    try:
        z = None if z in (None, "") else float(z)
    except (TypeError, ValueError):
        return _err("goal_z must be a number", 400)
    since = len(_log_text())
    _send(f"goal {x} {y} {yaw}" + (f" {z}" if z is not None else ""))
    _wait_new_marker("GOAL_SET", since, 5)
    if _loaded is not None:      # keep the signature honest so Load isn't skipped later
        _loaded.update(goal_x=str(x), goal_y=str(y), goal_yaw=str(yaw))
        if z is not None:
            _loaded.update(goal_z=str(z))
    return {"ok": True, "goal": [x, y, yaw, z]}


@app.post("/api/home")
def home():
    global _home
    # Model loaded -> home via the worker (it owns the robot); no unload needed.
    # The worker stops any running loop and winds it down before homing.
    if _alive(_worker):
        since = len(_log_text())
        _send("home")
        if _wait_new_marker("HOME_DONE", since, 60):
            return {"ok": True, "via": "worker"}
        return _err("home may not have completed — check the log", 500)
    # No model -> one-shot home subprocess.
    with _lock:
        if _alive(_home):
            return _err("already homing", 409)
        LOGFILE.write_text("")
        fh = open(LOGFILE, "ab", buffering=0)
        _home = subprocess.Popen(
            ["python", "webui/home.py"], cwd=str(ROOT),
            stdout=fh, stderr=subprocess.STDOUT,
            env=os.environ.copy(), start_new_session=True,
        )
    return {"ok": True, "via": "subprocess"}


@app.get("/api/sim/watch.mjpg")
def sim_watch_stream(flip: int = 0, hz: float = 12.0):
    """Both sim cameras as the policy receives them, as an MJPEG stream: the browser
    twin of `./robot real2sim live watch` (same frames and overlay, real2sim.live.watch).
    One more client on the sim's socket that only OBSERVES: it never commands the arm
    and does not advance a lockstep sim. `flip` turns both views 180 deg, display only."""
    if not SIM:
        return _err("not a sim session: start the web UI with --sim mujoco|isaac", 400)
    import cv2
    from real2sim.live import protocol, watch

    def frames():
        sock = protocol.connect(os.environ["ROBOT_PORT"], 30.0)
        try:
            warp = watch._warper(protocol.call(sock, "hello", warp=True))
            st, t_st = {}, 0.0
            while True:
                t0 = time.monotonic()
                if t0 - t_st > 0.5:
                    st, t_st = protocol.call(sock, "status"), t0
                ok, jpg = cv2.imencode(".jpg", watch._frame(sock, warp, st, bool(flip), hint=""),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                time.sleep(max(0.0, 1.0 / max(hz, 1.0) - (time.monotonic() - t0)))
        except (OSError, ConnectionError):
            return  # the sim stopped, or the tab closed
        finally:
            sock.close()

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/sim/watch", response_class=HTMLResponse)
def sim_watch_page():
    """The Live watch tab: the stream above, a flip toggle and the stage reset."""
    return """<!doctype html><html><head><meta charset="utf-8"><title>Live watch</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#0b0d12;color:#e6e8ee;font:14px system-ui,sans-serif}
.bar{display:flex;gap:8px;align-items:center;padding:8px 12px}
button{cursor:pointer;border:1px solid #2a3040;background:#232936;color:#e6e8ee;padding:7px 12px;border-radius:8px;font-weight:600}
button:disabled{opacity:.4;cursor:not-allowed}
img{display:block;width:100%;height:auto}#msg{opacity:.8}</style></head><body>
<div class="bar"><button id="flip">↻ Flip 180°</button><button id="reset">🎲 Reset stage</button><span id="msg"></span></div>
<img id="v" alt="sim cameras">
<script>
let flip = false; try { flip = localStorage.getItem('r2sFlip') === '1'; } catch (e) {}
const v = document.getElementById('v'), msg = document.getElementById('msg');
const show = () => { v.src = '/api/sim/watch.mjpg?flip=' + (flip ? 1 : 0) + '&t=' + Date.now(); };
document.getElementById('flip').onclick = () => { flip = !flip; try { localStorage.setItem('r2sFlip', flip ? '1' : '0'); } catch (e) {} show(); };
document.getElementById('reset').onclick = async () => {
  msg.textContent = 'resetting…';
  const r = await (await fetch('/api/sim/reset', {method: 'POST'})).json();
  msg.textContent = r.ok ? 'new stage ✓ (' + r.layout + ')' : r.msg;
};
v.onerror = () => { msg.textContent = 'no sim stream — retrying'; setTimeout(show, 2000); };
show();
</script></body></html>"""


@app.post("/api/sim/reset")
def sim_reset():
    """New stage in the live sim: the cap(s) and mug at fresh random spots, the arm back
    at rest. The same request as `./robot real2sim live reset`, sent over the sim's
    socket (ROBOT_PORT), which takes several clients next to the robot's own. Refused
    while a policy runs: the reset would teleport the arm out from under it."""
    if not SIM:
        return _err("not a sim session: start the web UI with --sim mujoco|isaac", 400)
    if _alive(_worker) and _is_running(_log_text()):
        return _err("stop the policy first (a reset mid-run moves the arm under it)", 409)
    if _alive(_home):
        return _err("busy: homing — wait for it to finish", 409)
    from real2sim.live import protocol

    try:
        sock = protocol.connect(os.environ["ROBOT_PORT"], 30.0)
        try:
            layout = protocol.call(sock, "reset")["layout"]
        finally:
            sock.close()
    except (OSError, KeyError) as e:
        return _err(f"no live sim at {os.environ.get('ROBOT_PORT')}: {e}", 503)
    return {"ok": True, "layout": layout.get("kind")}



# --- Eval page: seeded stages, timed trials, operator grades (webui/eval.html) ----------
import importlib.util as _ilu
import json as _json

from fastapi.responses import FileResponse, Response

import eval_stage

_ev: dict = {}                       # the open eval run
_ev_lock = threading.Lock()
PREVIEW = ROOT / "outputs" / "eval_preview.jpg"   # kept fresh by the openpi worker


def _sim_eval():
    """scripts/openpi/sim_eval.py: the report, the map and --compare read real runs too."""
    if "sim_eval" not in sys.modules:
        spec = _ilu.spec_from_file_location("sim_eval", ROOT / "scripts" / "openpi" / "sim_eval.py")
        mod = _ilu.module_from_spec(spec)
        sys.modules["sim_eval"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["sim_eval"]


def _jsonl(path: Path) -> list[dict]:
    return [_json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _ev_next() -> int | None:
    d, n = _ev["dir"], _ev["run"]["episodes"]
    taken = {t["trial"] for t in _jsonl(d / "trials.jsonl")} | {t["trial"] for t in _jsonl(d / "skipped.jsonl")}
    return next((i for i in range(n) if i not in taken), None)


def _ev_prepare(i: int | None, force: bool = False) -> None:
    """Trial i's stage: layout + overhead render, cached per seed. In sim mode preparing
    IS placing (the robot's own sim resets to the stage), so it always runs there."""
    import cv2

    _ev.update(i=i, phase="done" if i is None else "place", result=None, t_success=None, auto=None, layout=None,
               after=None, sim_grade=None)
    if i is None:
        return
    seed = _ev["run"]["seed"] + i
    js, jpg = _ev["dir"] / "stages" / f"seed_{seed}.json", _ev["dir"] / "stages" / f"seed_{seed}.jpg"
    if force or SIM or not js.exists():
        layout, img = eval_stage.prepare(seed)
        js.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(jpg), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        js.write_text(_json.dumps(layout))
    _ev["layout"] = _json.loads(js.read_text())


def _ev_poll() -> None:
    """running -> homing -> grade. When the worker reports the trial over (timeout or
    stop), the arm goes home, and only then is the overhead frame taken for the camera
    check: right after a release the gripper still hangs over the mug and hides the cap."""
    import shutil

    if _ev.get("phase") == "running":
        log = _log_text()[_ev["since"]:]
        k = log.rfind("EVAL_DONE ")
        if k < 0 and _alive(_worker):
            return
        res = _json.loads(log[k + 10:].splitlines()[0]) if k >= 0 else {"reason": "worker stopped", "duration": None}
        sim_grade = eval_stage.sim_caps_in_mug() if SIM else None  # the trial's end, before the arm moves
        _ev.update(result=res, sim_grade=sim_grade, since=len(_log_text()), t_home=time.time())
        if _alive(_worker):
            _send("home")
            _ev["phase"] = "homing"
            return
        _ev["phase"] = "grade"
    if _ev.get("phase") == "homing":
        done = "HOME_DONE" in _log_text()[_ev["since"]:] or "HOME_ERROR" in _log_text()[_ev["since"]:]
        if not done and time.time() - _ev["t_home"] < 60:
            return
        if time.time() - _ev["t_home"] < 1.0:
            return
        time.sleep(0.4)  # a preview frame from after the arm settled
        after = _ev["dir"] / "videos" / f"trial_{_ev['i']:03d}_after.jpg"
        after.parent.mkdir(parents=True, exist_ok=True)
        if PREVIEW.exists():
            shutil.copyfile(PREVIEW, after)
        cam = eval_stage.auto_check(after) if after.exists() else {"verdict": "unknown", "detail": "no frame"}
        sg = _ev.get("sim_grade")
        _ev.update(phase="grade", after=after.name if after.exists() else None,
                   auto=({"verdict": "in" if sg else "out", "detail": f"the sim's own check; camera after homing: "
                          f"{cam['verdict']} ({cam['detail']})"} if sg is not None else cam))


def _ev_public() -> dict:
    if not _ev:
        return {"open": False}
    se = _sim_eval()
    trials = _jsonl(_ev["dir"] / "trials.jsonl")
    k = sum(t["success"] for t in trials)
    lo, hi = se.wilson(k, len(trials))
    i = _ev.get("i")
    out = {"open": True, "name": _ev["dir"].name, "run": _ev["run"], "i": i, "phase": _ev["phase"],
           "seed": None if i is None else _ev["run"]["seed"] + i, "n_done": len(trials), "n_ok": k,
           "ci": [round(100 * lo), round(100 * hi)], "result": _ev.get("result"), "auto": _ev.get("auto"),
           "t_success": _ev.get("t_success"), "recent": trials[-10:][::-1], "after": _ev.get("after"),
           "n_skipped": len(_jsonl(_ev["dir"] / "skipped.jsonl")),
           "elapsed": round(time.time() - _ev["t_run"], 1) if _ev["phase"] == "running" else None,
           "report": (_ev["dir"] / "report.md").exists(), "causes": eval_stage.CAUSES}
    if _ev.get("layout"):
        out["where"] = eval_stage.describe(_ev["layout"])
    return out


def _ev_chown() -> None:
    """The container runs as root: hand the run back to whoever owns the repo."""
    st = ROOT.stat()
    for f in [eval_stage.RUNS, *eval_stage.RUNS.rglob("*")]:
        try:
            os.chown(f, st.st_uid, st.st_gid)
        except OSError:
            pass


@app.get("/eval", response_class=HTMLResponse)
def eval_page():
    return (WEBUI / "eval.html").read_text()


@app.get("/api/eval/state")
def eval_state():
    with _ev_lock:
        _ev_poll()
        log = _log_text()
        return {**_ev_public(), "loaded": _loaded, "worker": _alive(_worker), "running": _alive(_worker) and _is_running(log),
                "homing": _alive(_home) or log.rfind("HOME_START") > log.rfind("HOME_DONE"), "sim": SIM,
                "stage_sim": eval_stage.sim_socket() is not None,
                "preview": PREVIEW.exists() and time.time() - PREVIEW.stat().st_mtime < 3}


@app.post("/api/eval/open")
def eval_open(body: dict = Body(...)):
    if not (_alive(_worker) and _loaded and _loaded.get("stack") == "openpi"):
        return _err("load an openpi model on the main page first (the Eval page drives the loaded one)", 409)
    if eval_stage.sim_socket() is None:
        return _err("no sim to draw the stages: start the page with `./robot real-eval`", 400)
    policy = Path(_loaded["policy"].rstrip("/")).name
    seed, n, timeout = int(body.get("seed") or 1000), int(body.get("episodes") or 100), float(body.get("timeout") or 45)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", (body.get("name") or "").strip()) or \
        f"{policy}_{'sim' if SIM else 'real'}_seed{seed}_n{n}"
    d = eval_stage.RUNS / name
    with _ev_lock:
        d.mkdir(parents=True, exist_ok=True)
        if (d / "run.json").exists():  # resume: the run's own settings win, so its stages stay its stages
            run = _json.loads((d / "run.json").read_text())
            if Path(run["policy"].rstrip("/")).name != policy:
                return _err(f"run {name} was made with {Path(run['policy']).name}, not {policy}", 409)
        else:
            run = {"policy": _loaded["policy"], "task": _loaded.get("task"), "episodes": n, "seed": seed,
                   "timeout": timeout, "settle": "operator", "actions": _loaded.get("actions") or 15,
                   "where": "sim (web UI)" if SIM else "real", "created": datetime.now().isoformat(timespec="seconds")}
            (d / "run.json").write_text(_json.dumps(run, indent=2))
        _ev.clear()
        _ev.update(dir=d, run=run)
        try:
            _ev_prepare(_ev_next())
        except Exception as e:  # noqa: BLE001
            return _err(f"could not draw the stage: {e}", 503)
    return {"ok": True, **_ev_public()}


@app.get("/api/eval/stage.jpg")
def eval_stage_jpg():
    import cv2

    if not _ev.get("layout"):
        return Response(status_code=404)
    img = cv2.imread(str(_ev["dir"] / "stages" / f"seed_{_ev['run']['seed'] + _ev['i']}.jpg"))
    eval_stage.draw_targets(img, _ev["layout"])
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")


@app.get("/api/eval/live.mjpg")
def eval_live(hz: float = 8.0):
    """The overhead camera (the worker's preview) with the stage's outlines on top."""
    import cv2
    import numpy as np

    def frames():
        blank = np.full((480, 640, 3), 24, np.uint8)
        while True:
            t0 = time.monotonic()
            img = cv2.imread(str(PREVIEW)) if PREVIEW.exists() and time.time() - PREVIEW.stat().st_mtime < 3 else None
            if img is None:
                img = blank.copy()
                cv2.putText(img, "no camera: load a model on the main page", (110, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 200, 200), 1, cv2.LINE_AA)
            elif _ev.get("layout") and _ev.get("phase") in ("place", "running", "grade"):
                eval_stage.draw_targets(img, _ev["layout"], label=_ev["phase"] == "place")
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", img)[1].tobytes() + b"\r\n"
            time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - t0)))

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/eval/run")
def eval_run():
    import shutil

    with _ev_lock:
        if _ev.get("phase") != "place":
            return _err("no stage waiting to run", 409)
        if not _alive(_worker):
            return _err("no model loaded", 409)
        log = _log_text()
        if _is_running(log) or _alive(_home) or log.rfind("HOME_START") > log.rfind("HOME_DONE"):
            return _err("busy: the arm is running or homing", 409)
        if PREVIEW.exists() and not (_ev["dir"] / "stage.jpg").exists():
            shutil.copyfile(PREVIEW, _ev["dir"] / "stage.jpg")  # the map's backdrop: this camera, this mat
        video = _ev["dir"] / "videos" / f"trial_{_ev['i']:03d}.mp4"
        _ev.update(since=len(log), t_run=time.time(), phase="running", t_success=None)
        _send(f"eval {_ev['run']['timeout']} {video}")
    return {"ok": True}


@app.post("/api/eval/success")
def eval_success():
    """The cap is in: stop the policy now, and remember when."""
    with _ev_lock:
        if _ev.get("phase") != "running":
            return _err("no trial running", 409)
        _ev["t_success"] = round(time.time() - _ev["t_run"], 2)
        _send("stop")
    return {"ok": True}


@app.post("/api/eval/stop")
def eval_stop():
    with _ev_lock:
        if _ev.get("phase") != "running":
            return _err("no trial running", 409)
        _send("stop")
    return {"ok": True}


@app.post("/api/eval/grade")
def eval_grade(body: dict = Body(...)):
    success = bool(body.get("success"))
    cause = "success" if success else (body.get("cause") or "")
    if not success and cause not in eval_stage.CAUSES:
        return _err("pick what went wrong", 400)
    with _ev_lock:
        if _ev.get("phase") != "grade":
            return _err("nothing to grade", 409)
        res, i = _ev["result"] or {}, _ev["i"]
        dur = res.get("duration")
        rec = {"trial": i, "seed": _ev["run"]["seed"] + i, "success": success, "cause": cause,
               "t_success": (_ev.get("t_success") or dur) if success else None, "duration": dur,
               **eval_stage.record_fields(_ev["layout"]),
               "inferences": res.get("inferences"), "latency_ms_mean": res.get("latency_ms_mean"),
               "latency_ms_p95": res.get("latency_ms_p95"), "end": res.get("reason"),
               "video": f"videos/trial_{i:03d}.mp4" if res.get("video") else None,
               "auto": (_ev.get("auto") or {}).get("verdict"), "note": (body.get("note") or "").strip(),
               "graded_at": datetime.now().isoformat(timespec="seconds")}
        with open(_ev["dir"] / "trials.jsonl", "a") as fh:
            fh.write(_json.dumps(rec) + "\n")
        _ev_chown()
        try:
            _ev_prepare(_ev_next())
        except Exception as e:  # noqa: BLE001
            return _err(f"saved; the next stage failed to draw: {e}", 503)
    return {"ok": True}


@app.post("/api/eval/redo")
def eval_redo():
    """Throw this trial away (a misplaced object, a bump) and run the same stage again."""
    with _ev_lock:
        if _ev.get("phase") != "grade":
            return _err("nothing to redo", 409)
        for f in (_ev["dir"] / "videos").glob(f"trial_{_ev['i']:03d}*"):
            f.unlink()
        _ev_prepare(_ev["i"], force=SIM)
    return {"ok": True}


@app.post("/api/eval/skip")
def eval_skip(body: dict = Body(default={})):
    """This stage cannot be set up as drawn (it happens): leave it out of the run."""
    with _ev_lock:
        if _ev.get("phase") not in ("place", "grade"):
            return _err("nothing to skip", 409)
        with open(_ev["dir"] / "skipped.jsonl", "a") as fh:
            fh.write(_json.dumps({"trial": _ev["i"], "seed": _ev["run"]["seed"] + _ev["i"],
                                  "why": (body.get("why") or "").strip()}) + "\n")
        _ev_prepare(_ev_next())
    return {"ok": True}


@app.post("/api/eval/place")
def eval_place():
    """Sim mode: put the objects back on the stage (the sim's reset is the placement)."""
    if not SIM:
        return _err("on the real arm you place the objects yourself", 400)
    with _ev_lock:
        if _ev.get("phase") != "place":
            return _err("no stage waiting", 409)
        _ev_prepare(_ev["i"], force=True)
    return {"ok": True}


@app.post("/api/eval/report")
def eval_report():
    if not _ev:
        return _err("no eval run open", 409)
    if not _jsonl(_ev["dir"] / "trials.jsonl"):
        return _err("no graded trials yet", 409)
    _sim_eval().report(_ev["dir"])
    return {"ok": True, "url": f"/eval/files/{_ev['dir'].name}/report.md"}


@app.get("/eval/files/{run}/{path:path}")
def eval_files(run: str, path: str):
    f = (eval_stage.RUNS / run / path).resolve()
    if eval_stage.RUNS.resolve() not in f.parents or not f.is_file():
        return Response(status_code=404)
    if f.suffix == ".md":  # the report, readable in the browser with its images
        body = f.read_text()
        imgs = "".join(f'<p><img src="{m}" style="max-width:100%"></p>' for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body))
        return HTMLResponse(f"<!doctype html><meta charset=utf-8><title>{run}</title><body style='background:#0e1016;"
                            f"color:#e8ebf1;font:13px system-ui;max-width:900px;margin:auto;padding:16px'>"
                            f"<pre style='white-space:pre-wrap'>{body}</pre>{imgs}</body>")
    return FileResponse(f)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("WEBUI_PORT", "8000")))
