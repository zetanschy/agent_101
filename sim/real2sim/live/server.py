"""The live sim, served on a Unix socket to the `real2sim` lerobot robot.

    ./robot real2sim live serve --engine mujoco|isaac [--layout real:N|random[:K]]
                                [--clock auto|realtime|lockstep] [--start rest|recorded]
                                [--viewer] [--autoreset S] [--seed S]
    ./robot real2sim live reset [--layout ...]   # new layout now (another terminal)
    ./robot real2sim live status

Normally `./robot teleop|record|infer --sim ENGINE` starts and stops this for you.

CLOCK. `realtime` (default) advances 1/fps of sim time per 1/fps of wall time whether or
not a command arrived, the way the real servo keeps holding its last goal while a policy
thinks. That is the setting that behaves like the bench. `lockstep` advances exactly one
step per command and is deterministic, which suits an engine or a renderer slower than
real time: a slow step then stretches wall time instead of dropping sim time. When
`realtime` falls behind (Isaac's replay runs at 0.17-0.19x real time) it steps as fast
as it can and reports the ratio, since a sim that silently ran slow would teleoperate
like a different robot.

REQUESTS (protocol.py): hello, observe {images: [names]}, act {action: {motor: value}},
reset {layout?}, status. Values are lerobot units (arm degrees, gripper percent),
converted with the scene's fitted offsets and gripper map. That is the same conversion
the replay applies to the recorded actions, so a leader command means here what it
meant on the bench.

SINGLE-THREADED on purpose: MuJoCo's EGL renderer and Kit both belong to the thread that
created them, so requests, stepping and rendering all run in one select() loop.
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import socket
import time
from pathlib import Path

import numpy as np

from . import layouts, protocol

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def default_socket() -> Path:
    from .. import paths

    return paths.OUTPUTS / "live" / "sim.sock"


def engine_class(name: str):
    if name == "mujoco":
        from .engine_mujoco import MujocoEngine

        return MujocoEngine
    if name == "isaac":
        from ..isaac.live import IsaacEngine

        return IsaacEngine
    raise SystemExit(f"--engine {name!r}: expected mujoco or isaac")


class Server:
    def __init__(self, engine, scene, a, rng):
        from .. import units as U

        self.eng, self.scene, self.a, self.rng = engine, scene, a, rng
        self.units = scene.units()
        self.calib = U.calibration()
        self._lo, self._hi, _ = U._cal(self.calib)
        self.rest = layouts.rest_state(scene)
        self.rest_q = self.units.to_urdf(self.rest)
        self.layout = None
        self.fps = float(a.fps)
        self.n_steps = 0
        self._done_since = None
        self._lag = 0.0
        # PRE-RENDER: the state only changes on a step, so the frames of the next observation
        # can be rendered right after it, off the client's round trip (18 ms of a 33 ms tick
        # for both MuJoCo cameras). Only while a client keeps asking for images.
        self._frames, self._frames_step, self._images_wanted_until, self._pinhole = None, -1, 0.0, False

    # --- the scene ---------------------------------------------------------------
    def new_layout(self, spec: str | None = None, start: str | None = None) -> dict:
        """start 'rest' (default): the arm folded at rest, holding it. 'recorded' (real:N
        layouts only): episode N's first recorded state, holding its first recorded action,
        so a recording fed over the socket reproduces the offline replay (GoalStream's start)."""
        self.layout = layouts.parse(spec or self.a.layout, self.scene, self.rng)
        start = start or self.a.start
        rest_q, hold_q = self.rest_q, None
        if start == "recorded":
            if not self.layout["kind"].startswith("real:"):
                raise ValueError("--start recorded needs a real:N layout")
            from .. import episodes

            ep = episodes.load(self.scene.ds, include_excluded=True)[int(self.layout["kind"].split(":")[1])]
            rest_q, hold_q = self.units.to_urdf(ep.state[0]), self.units.to_urdf(ep.action[0])
        self.eng.reset(self.layout, rest_q, hold_q)
        self._print_t0 = 0.0
        self._done_since = None
        self._frames, self._frames_step = None, -1
        print(f"layout {self.layout['kind']}: mug at {self.layout['mug']['xy']}, "
              + ", ".join(f"cap {c['id']} at {c['xy']} ({c['up']} up)" for c in self.layout["caps"]), flush=True)
        return self.layout

    # --- bus units: degrees (use_degrees=True, this bench) or lerobot's RANGE_M100_100 ------
    # The openpi checkpoints were trained on normalized arm joints (scripts/openpi/evaluate.py
    # --units normalized). The bus maps ticks to -100..100 over each joint's calibrated range
    # (lerobot motors_bus.py, drive_mode 0 here); the gripper is RANGE_0_100 in both modes.
    def _to_bus(self, v, units: str) -> np.ndarray:
        if units == "degrees":
            return v
        from .. import units as U

        P = U.lerobot_to_ticks(v, self.calib)
        out = v.copy()
        lo, hi = self._lo[:5], self._hi[:5]
        out[:5] = (np.clip(P[:5], lo, hi) - lo) / (hi - lo) * 200.0 - 100.0
        return out

    def _from_bus(self, v, units: str) -> np.ndarray:
        if units == "degrees":
            return v
        from .. import units as U

        lo, hi = self._lo[:5], self._hi[:5]
        P = U.lerobot_to_ticks(v, self.calib)  # the gripper's ticks; the arm's are replaced below
        P[:5] = (np.clip(v[:5], -100.0, 100.0) + 100.0) / 200.0 * (hi - lo) + lo
        return U.ticks_to_lerobot(P, self.calib)

    def _render(self, cams, pinhole: bool) -> dict:
        return (self.eng.render_pinhole(cams) if pinhole else None) or self.eng.render(cams)

    def observation(self, images, units: str = "degrees", pinhole: bool = False) -> dict:
        v = self._to_bus(self.units.to_lerobot(self.eng.state()), units)
        out = {"state": {m: float(x) for m, x in zip(MOTORS, v)}, "t": self.eng.t}
        if images:
            cams = [c for c in images if c in self.eng.cameras]
            self._images_wanted_until, self._pinhole = time.monotonic() + 1.0, pinhole
            f = self._frames
            if f is not None and self._frames_step == self.n_steps and f[0] == pinhole and all(c in f[1] for c in cams):
                out["images"] = {c: f[1][c] for c in cams}
            else:
                out["images"] = self._render(cams, pinhole)
            out["pinhole"] = pinhole and self.eng.render_pinhole([]) is not None
        return out

    def command(self, action: dict, units: str = "degrees") -> None:
        v = self._from_bus(np.array([float(action[m]) for m in MOTORS]), units)
        self.eng.command(self.units.to_urdf(v))

    def status(self) -> dict:
        s = self.eng.status()
        s.update(t=round(self.eng.t, 3), layout=self.layout, engine=self.eng.name, clock=self.a.clock,
                 steps=self.n_steps, behind_s=round(self._lag, 3))
        return s

    def step(self) -> None:
        self.eng.step()
        self.n_steps += 1
        if time.monotonic() < self._images_wanted_until:
            cams = list(self.eng.cameras)
            self._frames, self._frames_step = (self._pinhole, self._render(cams, self._pinhole)), self.n_steps
        if self.a.autoreset and self.n_steps % int(self.fps / 2) == 0:
            st = self.eng.status()
            if st.get("caps_in_mug") and all(st["caps_in_mug"]):
                self._done_since = self._done_since if self._done_since is not None else self.eng.t
                if self.eng.t - self._done_since >= self.a.autoreset:
                    print(f"t={self.eng.t:.1f}s: every cap is in the mug -> new layout", flush=True)
                    self.new_layout()
            else:
                self._done_since = None

    # --- requests ------------------------------------------------------------------
    def handle(self, msg: dict):
        op = msg.get("op")
        if op == "hello":
            out = {"engine": self.eng.name, "fps": self.fps, "motors": list(MOTORS), "clock": self.a.clock,
                   "cameras": {k: list(v) for k, v in self.eng.cameras.items()}, "scene_hash": self.scene.hash,
                   "dataset": self.scene.ds, "layout": self.layout, "rest": self.rest.tolist()}
            if msg.get("warp") and self.eng.render_pinhole([]) is not None:
                out["warp"] = {c: self.eng.warp_spec(c) for c in self.eng.cameras}
            return out
        if op == "observe":
            return self.observation(msg.get("images") or [], msg.get("units", "degrees"), bool(msg.get("pinhole")))
        if op == "act":
            self.command(msg["action"], msg.get("units", "degrees"))
            if self.a.clock == "lockstep":
                self.step()
            return {"t": self.eng.t}
        if op == "reset":
            return {"layout": self.new_layout(msg.get("layout"), msg.get("start"))}
        if op == "status":
            return self.status()
        return {"error": f"unknown op {op!r}"}

    def serve(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        lst = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        lst.bind(str(path))
        os.chmod(path, 0o777)  # the container side runs as root, the tools on the host as the user
        lst.listen(4)
        clients: list[socket.socket] = []
        period = 1.0 / self.fps
        next_t = time.monotonic()
        last_print, steps_at_print = time.monotonic(), 0
        self._print_t0 = self.eng.t
        running = [True]

        def stop(*_):
            running[0] = False

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        print(f"real2sim live: {self.eng.name}, {self.a.clock} clock, {self.fps:g} fps, serving {path}", flush=True)
        try:
            while running[0]:
                now = time.monotonic()
                wait = max(0.0, next_t - now) if self.a.clock == "realtime" else 0.05
                try:
                    readable, _, _ = select.select([lst] + clients, [], [], wait)
                except InterruptedError:
                    continue
                for s in readable:
                    if s is lst:
                        c, _ = lst.accept()
                        c.settimeout(30.0)
                        clients.append(c)
                        continue
                    try:
                        protocol.send(s, self._safe(protocol.recv(s)))
                    except (ConnectionError, OSError):
                        clients.remove(s)
                        s.close()
                if self.a.clock == "realtime" and time.monotonic() >= next_t:
                    self.step()
                    next_t += period
                    behind = time.monotonic() - next_t
                    self._lag = max(0.0, behind)
                    if behind > 0.25:  # cannot keep up: step back to back, report the ratio
                        next_t = time.monotonic()
                if self.eng.reset_requested():
                    self.new_layout()
                if self.a.viewer and not self.eng.viewer_sync():
                    print("viewer closed; still serving (Ctrl-C to stop)", flush=True)
                    self.a.viewer = False
                if time.monotonic() - last_print >= 5.0:
                    wall = time.monotonic() - last_print
                    rt = (self.eng.t - self._print_t0) / wall  # _print_t0 restarts with every reset
                    st = self.eng.status()
                    cim = st.get("caps_in_mug") or []
                    pace = f"{rt:4.2f}x real time" if self.a.clock == "realtime" else f"{self.n_steps - steps_at_print:5d} steps (lockstep)"
                    print(f"t={self.eng.t:7.1f}s  {pace}  caps in mug {sum(cim)}/{len(cim)}  clients {len(clients)}", flush=True)
                    last_print, steps_at_print, self._print_t0 = time.monotonic(), self.n_steps, self.eng.t
        finally:
            for c in clients:
                c.close()
            lst.close()
            if path.exists():
                path.unlink()
            self.eng.close()

    def _safe(self, msg):
        try:
            return self.handle(msg)
        except Exception as e:  # a bad request must not take the sim down mid-session
            return {"error": f"{type(e).__name__}: {e}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="real2sim live serve")
    ap.add_argument("--engine", required=True, choices=("mujoco", "isaac"))
    ap.add_argument("--ds")
    ap.add_argument("--layout", default="random", help="real:N | random | random:K (default random)")
    ap.add_argument("--clock", choices=("auto", "realtime", "lockstep"), default="auto",
                    help="auto: realtime for mujoco (9.8x real-time physics), lockstep for isaac (0.19x: "
                         "one physics step per command keeps every recorded frame exactly 1/fps of sim)")
    ap.add_argument("--start", choices=("rest", "recorded"), default="rest",
                    help="recorded (real:N only): the episode's first state and first action, as the replay starts")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--socket", default=None, help="default sim/outputs/real2sim/live/sim.sock")
    ap.add_argument("--viewer", action="store_true", help="open the engine's interactive viewer (R = new layout)")
    ap.add_argument("--autoreset", type=float, default=0.0, metavar="S",
                    help="new layout S seconds after every cap is in the mug (0: never)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    if a.clock == "auto":
        a.clock = "realtime" if a.engine == "mujoco" else "lockstep"
    cls = engine_class(a.engine)
    cls.boot(a)  # Kit has to exist before the Isaac engine imports isaaclab
    from .. import paths, scene as scene_mod

    sc = scene_mod.load(paths.dataset(a.ds))
    eng = cls(sc, fps=a.fps, viewer=a.viewer)
    srv = Server(eng, sc, a, np.random.default_rng(a.seed))
    srv.new_layout()
    srv.serve(Path(a.socket) if a.socket else default_socket())
    return 0


def client_main(argv=None) -> int:
    """`live reset` / `live status` / `live ping` from another terminal."""
    ap = argparse.ArgumentParser(prog="real2sim live")
    ap.add_argument("op", choices=("reset", "status", "ping"))
    ap.add_argument("--layout")
    ap.add_argument("--socket", default=None)
    ap.add_argument("--timeout", type=float, default=5.0)
    a = ap.parse_args(argv)
    path = str(a.socket or default_socket())
    try:
        s = protocol.connect(path, a.timeout)
    except OSError as e:
        print(f"no live sim at {path}: {e}")
        return 1
    import json

    if a.op == "ping":
        r = protocol.call(s, "hello")
        print(f"{r['engine']} {r['clock']} {r['fps']:g} fps, scene {r['scene_hash']}")
    elif a.op == "reset":
        print(json.dumps(protocol.call(s, "reset", **({"layout": a.layout} if a.layout else {}))["layout"]))
    else:
        print(json.dumps(protocol.call(s, "status"), indent=1))
    s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
