"""Drive a running live sim over its socket with a recorded episode; compare with the offline replay.

    ./robot real2sim live serve --engine isaac --layout real:0 --start recorded      # terminal 1 (lockstep: isaac's default)
    $R2S_HOST_PY sim/real2sim/isaac/tests/live_replay.py --episode 0 [--images]       # terminal 2
    $R2S_HOST_PY sim/real2sim/isaac/tests/live_replay.py --realtime 60 --resets real:0,real:1,random:2

A script, not a pytest test: it needs a server (Kit, the GPU lock) that pytest runs of
this directory must not start.

LOCKSTEP (the server's --clock lockstep): every frame, observe (the encoders; with --images
both cameras as well) then act with the recorded action in lerobot units -- the loop
lerobot-record runs against the real arm. The observed joints are compared with the
offline replay's pose log (--log, default isaac/ep<N>_action): max |difference| per joint,
and the caps the server reports in the mug at the end. Start the server on the episode's
layout with its recorded start (--layout real:N --start recorded) and do not reset it
first: the first layout is placed on a fresh stage, as the replay's is. MEASURED: 0.0 deg
max on all joints over ep0's 488 frames, both cameras rendered every frame.

REALTIME (the server's default clock): --realtime S seconds of observe(both images) + act,
paced at the dataset's fps as lerobot-record paces them; reports the real-time factor
(sim seconds per wall second), the observe and act round trips and the server's GPU memory
(the server's pid from the socket's peer credentials). --resets then times a reset request
per layout.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[3]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from real2sim import episodes, paths, poselog, scene as r2s  # noqa: E402
from real2sim.live import protocol  # noqa: E402
from real2sim.live.server import MOTORS, default_socket  # noqa: E402

CAMS = ["front", "grip"]


def connect(path: str, wait_s: float) -> socket.socket:
    """The server listens only once Kit is up and the first layout has settled (~30 s)."""
    t0 = time.monotonic()
    while True:
        try:
            s = protocol.connect(path, timeout_s=120.0)
            return s
        except OSError:
            if time.monotonic() - t0 > wait_s:
                raise
            time.sleep(1.0)


def peer_pid(s: socket.socket) -> int:
    pid, _, _ = struct.unpack("3i", s.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    return pid


def vram(pid: int) -> dict:
    """nvidia-smi: the server process's memory and the whole card's, MiB."""
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout
        mine = sum(int(m) for p, m in (ln.split(",") for ln in apps.strip().splitlines() if ln.strip()) if int(p) == pid)
        tot = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split(",")
        return {"server_mib": mine, "gpu_used_mib": int(tot[0]), "gpu_total_mib": int(tot[1])}
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return {"error": repr(e)}


def ms(xs) -> dict:
    a = np.asarray(xs) * 1e3
    return {"p50": round(float(np.median(a)), 1), "p95": round(float(np.percentile(a, 95)), 1),
            "max": round(float(a.max()), 1), "n": int(len(a))}


def action(ep, k: int) -> dict:
    return {m: float(v) for m, v in zip(MOTORS, ep.action[min(k, len(ep) - 1)])}


def lockstep(s, sc, ep, ref: dict, images: bool) -> dict:
    u, T = sc.units(), len(ep)
    q, t_obs, lat_obs, lat_act = np.zeros((T, 6)), np.zeros(T), [], []
    wall = time.monotonic()
    for k in range(T):
        t0 = time.perf_counter()
        ob = protocol.call(s, "observe", images=CAMS if images else [])
        lat_obs.append(time.perf_counter() - t0)
        q[k] = u.to_urdf(np.array([ob["state"][m] for m in MOTORS]))
        t_obs[k] = ob["t"]
        if images and any(ob["images"][c].shape != (480, 640, 3) for c in CAMS):
            raise RuntimeError(f"frame {k}: images {[ob['images'][c].shape for c in CAMS]}")
        if k == T - 1:
            break
        t0 = time.perf_counter()
        protocol.call(s, "act", action=action(ep, k))
        lat_act.append(time.perf_counter() - t0)
        if k % 100 == 0:
            print(f"  frame {k}/{T - 1}  {time.monotonic() - wall:.0f} s", flush=True)
    wall = time.monotonic() - wall
    st = protocol.call(s, "status")
    d = np.degrees(np.abs(q - np.asarray(ref["q"], float)[:T]))
    first = int(np.argmax(d.max(1) > 1e-9)) if d.max() > 1e-9 else None
    return {"frames": T, "max_diff_deg": d.max(0).round(6).tolist(), "max_diff_deg_all": round(float(d.max()), 6),
            "first_differing_frame": first, "mean_diff_deg_after_frame_30": d[30:].mean(0).round(4).tolist(),
            "clock_ok": bool(np.allclose(t_obs, np.arange(T) / ep.fps)), "caps": st["caps"],
            "caps_in_mug": st["caps_in_mug"], "finite": st["finite"], "wall_s": round(wall, 1),
            "sim_per_wall": round((T - 1) / ep.fps / wall, 3), "observe_ms": ms(lat_obs), "act_ms": ms(lat_act),
            "engine_ms_p50": {k: v for k, v in st.items() if k.endswith("_ms_p50")}}


def realtime(s, ep, seconds: float) -> dict:
    fps = float(ep.fps)
    t_sim0 = protocol.call(s, "status")["t"]
    lat_obs, lat_act, n = [], [], 0
    t0 = nxt = time.monotonic()
    while time.monotonic() - t0 < seconds:
        a = time.perf_counter()
        ob = protocol.call(s, "observe", images=CAMS)
        b = time.perf_counter()
        protocol.call(s, "act", action=action(ep, n))
        lat_obs.append(b - a)
        lat_act.append(time.perf_counter() - b)
        n += 1
        nxt += 1.0 / fps
        time.sleep(max(0.0, nxt - time.monotonic()))
    wall = time.monotonic() - t0
    st = protocol.call(s, "status")
    return {"wall_s": round(wall, 1), "sim_s": round(st["t"] - t_sim0, 2), "real_time_factor": round((st["t"] - t_sim0) / wall, 3),
            "observations": n, "observations_per_s": round(n / wall, 2), "observe_ms": ms(lat_obs), "act_ms": ms(lat_act),
            "server_behind_s": st.get("behind_s"), "engine_ms_p50": {k: v for k, v in st.items() if k.endswith("_ms_p50")},
            "last_image_shapes": {c: list(ob["images"][c].shape) for c in CAMS}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--log", default=None, help="offline replay run dir or pose log (default isaac/ep<N>_action)")
    ap.add_argument("--images", action="store_true", help="lockstep: request both cameras every frame")
    ap.add_argument("--realtime", type=float, default=0.0, metavar="S", help="realtime-clock measurement for S seconds")
    ap.add_argument("--resets", default="", help="comma-separated layouts to time a reset request for")
    ap.add_argument("--socket", default=None)
    ap.add_argument("--wait", type=float, default=600.0, help="seconds to wait for the server to listen")
    ap.add_argument("--out", default=None, help="write the result JSON here too")
    a = ap.parse_args(argv)
    s = connect(str(a.socket or default_socket()), a.wait)
    hello = protocol.call(s, "hello")
    pid = peer_pid(s)
    sc = r2s.load(paths.dataset(hello["dataset"]))
    ep = episodes.load(sc.ds)[a.episode]
    res = {"engine": hello["engine"], "clock": hello["clock"], "scene_hash": hello["scene_hash"],
           "layout": hello["layout"]["kind"], "server_pid": pid, "vram_start": vram(pid)}
    if a.realtime:
        res["realtime"] = realtime(s, ep, a.realtime)
    else:
        log = Path(a.log) if a.log else paths.out_dir(sc.ds, hello["engine"], f"ep{a.episode}_action")
        ref = poselog.read(log / "poselog.npz" if log.is_dir() else log)
        res["reference"] = str(log)
        res["lockstep"] = lockstep(s, sc, ep, ref, a.images)
    res["vram_end"] = vram(pid)
    for spec in filter(None, a.resets.split(",")):
        t0 = time.perf_counter()
        r = protocol.call(s, "reset", layout=spec)
        res.setdefault("resets", []).append({"layout": spec, "s": round(time.perf_counter() - t0, 3),
                                             "caps": [c["id"] for c in r["layout"]["caps"]]})
    s.close()
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
