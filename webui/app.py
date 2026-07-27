#!/usr/bin/env python3
"""Web UI backend for the SO-ARM101 policy.

Inference uses a PERSISTENT worker (webui/infer_worker.py): you 'Load' a model
once (slow pi05 load happens here) and then Start/Stop the rollout loop as many
times as you like without reloading. 'Home' runs webui/home.py (mutually
exclusive with a loaded model, since both drive the robot).
"""
import glob
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent          # /workspace
WEBUI = ROOT / "webui"
LOGDIR = ROOT / "outputs"
LOGDIR.mkdir(parents=True, exist_ok=True)
LOGFILE = LOGDIR / "webui_run.log"
DEFAULT_POLICY = "zetanschy/pi05_lora_cap_tu_cup"
RENAME = ('{"observation.images.front": "observation.images.base_0_rgb", '
          '"observation.images.grip": "observation.images.left_wrist_0_rgb"}')

_FPS = float(os.environ.get("CAM_FPS", "30") or 30)
_LAT_RE = re.compile(r"running slower \(([0-9.]+) Hz\)|real_delay=([0-9]+)")

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
    models = [DEFAULT_POLICY]
    for p in glob.glob(str(ROOT / "outputs/train/*/")):
        pp = Path(p)
        if (pp / "adapter_model.safetensors").exists() or (pp / "config.json").exists():
            models.append(str(pp))
    for p in sorted(glob.glob(str(ROOT / "outputs/train/*/checkpoints/*/"))):
        models.append(str(Path(p)))
    out = []
    for m in models:
        if m not in out:
            out.append(m)
    return out


# ---------- endpoints ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return (WEBUI / "index.html").read_text()


@app.get("/api/models")
def models():
    return {"models": list_models(), "default": DEFAULT_POLICY}


@app.get("/api/status")
def status():
    log = _log_text()
    return {
        "loaded": _alive(_worker),
        "loaded_sig": _loaded,
        "running": _alive(_worker) and _is_running(log),
        "homing": _alive(_home),
        "latency": latency_stats(log),
        "log": log[-8000:],
    }


@app.post("/api/load")
def load(body: dict = Body(...)):
    if not (body.get("task") or "").strip():
        return _err("task is required", 400)
    if _alive(_home):
        return _err("busy: homing — wait for it to finish", 409)
    args, sig = rollout_args(body)
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
            ["python", "webui/infer_worker.py", *args], cwd=str(ROOT),
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
    _send(f"actionsteps {int(n)}")
    _wait_new_marker("ACTION_STEPS_SET", since, 5)
    if _loaded is not None:          # keep the signature honest so Load isn't skipped later
        _loaded["action_steps"] = str(int(n))
    return {"ok": True, "action_steps": int(n)}


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("WEBUI_PORT", "8000")))
