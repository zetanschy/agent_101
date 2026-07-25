#!/usr/bin/env python3
"""Minimal web UI to operate the SO-ARM101 policy: move home, run inference
(sync/rtc/async), optionally record the rollout, pick a model, tune params.

Runs INSIDE the robot container (has serial + camera + GPU access). One robot
operation at a time; work is launched as a subprocess so it cleanly releases the
robot between runs. Reuses scripts/infer.sh for inference and webui/home.py for home.
"""
import glob
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

_FPS = float(os.environ.get("CAM_FPS", "30") or 30)
# sync mode logs "running slower (X Hz)"; RTC logs "real_delay=N" (control steps).
_LAT_RE = re.compile(r"running slower \(([0-9.]+) Hz\)|real_delay=([0-9]+)")


def latency_ms(log: str):
    """Most recent inference latency estimate (ms) parsed from the rollout log."""
    best = None
    for m in _LAT_RE.finditer(log):
        if m.group(1):
            hz = float(m.group(1))
            best = 1000.0 / hz if hz > 0 else best
        elif m.group(2):
            best = int(m.group(2)) * (1000.0 / _FPS)
    return round(best) if best else None

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent          # /workspace
WEBUI = ROOT / "webui"
LOGDIR = ROOT / "outputs"
LOGDIR.mkdir(parents=True, exist_ok=True)
LOGFILE = LOGDIR / "webui_run.log"
DEFAULT_POLICY = "zetanschy/pi05_lora_cap_tu_cup"

app = FastAPI()
_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_op: str | None = None
_meta: dict = {}


def _running() -> bool:
    return _proc is not None and _proc.poll() is None


def _start(op: str, cmd: list[str], meta: dict):
    global _proc, _op, _meta
    with _lock:
        if _running():
            return False, f"busy: '{_op}' is running — stop it first"
        LOGFILE.write_text("")
        fh = open(LOGFILE, "ab", buffering=0)
        _proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True, env=os.environ.copy(),
        )
        _op, _meta = op, meta
        return True, "started"


def _stop() -> str | None:
    global _op
    with _lock:
        op = _op
        if _running():
            try:
                pgid = os.getpgid(_proc.pid)
                os.killpg(pgid, signal.SIGINT)
                for _ in range(20):
                    if _proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if _proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
        _op = None
        return op


def list_models() -> list[str]:
    models = [DEFAULT_POLICY]
    for p in glob.glob(str(ROOT / "outputs/train/*/")):
        pp = Path(p)
        if (pp / "adapter_model.safetensors").exists() or (pp / "config.json").exists():
            models.append(str(pp))
    for p in sorted(glob.glob(str(ROOT / "outputs/train/*/checkpoints/*/"))):
        models.append(str(Path(p)))
    out: list[str] = []
    for m in models:
        if m not in out:
            out.append(m)
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEBUI / "index.html").read_text()


@app.get("/api/models")
def models():
    return {"models": list_models(), "default": DEFAULT_POLICY}


@app.get("/api/status")
def status():
    full = LOGFILE.read_text(errors="replace") if LOGFILE.exists() else ""
    return {"running": _running(), "op": _op if _running() else None,
            "meta": _meta if _running() else {}, "log": full[-8000:],
            "latency_ms": latency_ms(full)}


@app.post("/api/home")
def home():
    ok, msg = _start("home", ["python", str(WEBUI / "home.py")], {})
    return JSONResponse({"ok": ok, "msg": msg}, status_code=200 if ok else 409)


@app.post("/api/infer")
def infer(body: dict = Body(...)):
    task = (body.get("task") or "").strip()
    if not task:
        return JSONResponse({"ok": False, "msg": "task is required"}, status_code=400)
    policy = body.get("policy") or DEFAULT_POLICY
    mode = body.get("mode", "rtc")
    if mode not in ("sync", "rtc", "async"):
        mode = "rtc"
    cmd = ["bash", "scripts/infer.sh", "--no-display",
           "--policy", policy, "--task", task,
           "--duration", str(body.get("duration", 60)),
           "--cams", str(body.get("cams", 2)), f"--{mode}"]
    steps = str(body.get("steps", "") or "").strip()
    if steps:
        cmd += ["--steps", steps]
    if body.get("record"):
        cmd += ["--record", str(body["record"]).strip()]
    ok, msg = _start("infer", cmd, {"mode": mode, "policy": policy})
    return JSONResponse({"ok": ok, "msg": msg, "cmd": " ".join(cmd)},
                        status_code=200 if ok else 409)


@app.post("/api/stop")
def stop():
    return {"ok": True, "stopped": _stop()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("WEBUI_PORT", "8000")))
