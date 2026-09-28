"""Isaac Sim as the live follower: the evaluated replay's scene, servo and RTX look, stepped on demand.

    ./robot real2sim live serve --engine isaac [--layout real:N|random[:K]] [--clock lockstep|realtime] [--viewer]

NOTHING HERE IS NEW PHYSICS. The scene is scene.py's (the workshop arm with its widened
limits and the wrist payload, the table, the convex-piece caps and mug, the max-combined
materials), stepped as params.py says (480 Hz TGS, 64 position iterations, 16 substeps
per 1/30 s frame, one env), and every substep runs servo.PhysxServo on the goal the
firmware holds at that instant, as replay.py's action mode does. Only the goals' source
differs: live.goals.OnlineGoals (a command pushed at sim time t takes effect at
t + dead_time, firmware-clamped, zero-order held and slewed) instead of the recorded
array. MEASURED (isaac README 'Live'): fed episode 0's recorded actions over the socket in
lockstep, started where the recording starts, the encoders follow the offline replay's
pose log bit for bit over all 488 frames with both cameras rendered every frame, and the
cap ends in the mug.

RESET. The stage is built once, on the first layout, as replay.py builds that episode,
plus spare cap bodies (POOL_CAPS) appended after the mug; they leave the first layout
bitwise equal to the replay. Every reset then re-places: the arm written to rest_q, the
layout's caps and the mug placed, spare caps parked out of reach and hidden, the replay's
0.3 s settle with the arm held, servo and goals reset, t = 0: 1.5-1.7 s, 1.25 s of it the
settle. After that objects move only through contact. PhysX keeps its contact caches
across a re-placed reset, so a later episode follows a fresh replay closely (ep0 again:
bitwise for 214 frames, then within 0.09 deg) but not bit for bit. The stage is not rebuilt
in process: replicator's annotator teardown raised, and the rebuild after it hung.

IMAGES are the Isaac render path's (lookrender.py): LOOK's assets on the scene (look.py),
RTX real-time at the calibrated light units, the camera prims camcheck.py verified, and
the HdrColor AOV (linear scene radiance) of both oversized pinholes. The camera itself is
blender.camera_model.Response -- lens remap, optics blur, colour model, sRGB -- evaluated
on the GPU (_GpuCamera, the same operations): on the host it costs 17 ms (front) + 58 ms
(wrist) per observation, more than a 30 Hz tick; here 2 ms for both. So render_pinhole
stays None: a client-side warp would be slower and would skip the colour model. One RTX
frame per observation, 38 ms for both cameras, no settle renders: the renderer's temporal
history is the previous observation's frame. Its cost against a converged render (5 more)
along ep0: front 49.8-58.4 dB PSNR (mean |error| 0.09-0.24 of 255), wrist 42.6-57.3 dB
(0.12-0.91). The camera state is LOOK's reference (wb 'reference', wrist exposure
'constant'): a live frame has no dataset row whose auto white balance it could replay.
Images show the sim at the observation instant: the real cameras' 2.3 / 1.4 frame
latency (CALIB) is not emulated, which would mean writing past poses into the physics.

SPEED. The physics costs what it cost in the replay: 117-204 ms per frame (p50 by phase;
PhysX itself 123 of 138 ms in the reach, the Python around it 15 ms), so the realtime
clock runs at 0.16-0.21x real time with both cameras every frame and lockstep at 0.14x.
Faster means other physics (fewer iterations, CPU PhysX), which the replay's numbers
would no longer vouch for; the MuJoCo engine is the real-time one.

KIT. boot() starts it (AppLauncher: headless unless --viewer, cameras on) before anything
imports isaaclab or omni, so this module's top level imports stdlib and numpy only. The
process ends with os._exit (close(); an atexit guard for errors before serving; and
_quit_guard when Kit is asked to quit, e.g. the viewer window closed): Kit 4.5 hangs in
its shutdown on this box. The GPU lock (paths.GPU_LOCK) is held for the process's life:
live/run.sh wraps the server in flock, and boot() takes the lock itself when started any
other way, so no Isaac or Blender batch job can share the 12 GB card (5.2 GB here).
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque

import numpy as np

from ..live import layouts
from ..live.engine import Engine
from ..live.goals import OnlineGoals

WARMUP_RENDERS = 3  # RTX frames after a reset: the temporal history must not show the old layout
STEADY_MAX = 40  # renders after the build until the front view is steady (lookrender.settle)
# relative mean change between two renders that counts as steady. MEASURED on the first
# build: 0.048, 0.0025, 0.0091 while the textures load, then 1e-4..6e-4 for good, the
# anti-aliasing jitter of a static view, so lookrender.settle's 1e-4 ran all 40 renders
STEADY_TOL = 3e-4
POOL_CAPS = ("A", "B", "C")  # cap bodies on the stage: every layout names its caps A, B, C (layouts.py)
PARK_X = 3.0  # m, where spare caps wait (_park)
# the viewer camera, inside LOOK's backdrop cylinder (centre (-0.1, -0.3), R 0.8 m, top at 0.62 m):
# from outside, the wall hides the scene. Base frame, -y is forward.
VIEWER_EYE, VIEWER_TARGET = (0.30, -0.80, 0.58), (0.0, -0.20, 0.08)


# --- the GPU lock ------------------------------------------------------------------------

def _flock_holders(path: str) -> set[int]:
    """PIDs holding a flock on `path` (/proc/locks: 'N: FLOCK ADVISORY WRITE pid maj:min:inode ...')."""
    ino = os.stat(path).st_ino
    out = set()
    try:
        with open("/proc/locks") as f:
            for line in f:
                p = line.split()
                if len(p) > 5 and p[1] == "FLOCK" and p[5].rsplit(":", 1)[-1] == str(ino):
                    out.add(int(p[4]))
    except OSError:
        pass
    return out


def _ancestors() -> set[int]:
    out, pid = set(), os.getppid()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def hold_gpu_lock() -> str:
    """Hold paths.GPU_LOCK until this process exits, as `flock $R2S_GPU_LOCK` does.

    live/run.sh already runs the server under flock(1); the lock then belongs to that
    ancestor and taking it again here would deadlock, so an ancestor's lock counts as
    held. Otherwise wait for it (another Isaac or Blender job is running)."""
    import fcntl

    from .. import paths

    path = str(paths.GPU_LOCK)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)  # not inherited across exec (PEP 446)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        IsaacEngine._lock_fd = fd
        return f"GPU lock {path}: taken"
    except BlockingIOError:
        holders = _flock_holders(path)
        mine = holders & _ancestors()
        if mine:
            os.close(fd)
            return f"GPU lock {path}: held by the launcher (pid {min(mine)})"
        print(f"waiting for the GPU lock {path} (held by pid {sorted(holders) or '?'})", flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
        IsaacEngine._lock_fd = fd
        return f"GPU lock {path}: taken after waiting"


# --- the camera on the GPU ---------------------------------------------------------------

class _GpuCamera:
    """blender.camera_model.Response's lens and colour, the same operations in torch.

    lens    bilinear remap of the pinhole into the real distorted pixels (cv2.remap,
            BORDER_REPLICATE -> grid_sample, border; align_corners=True puts -1..1 on the
            edge pixels' centres, the OpenCV convention of camera.remap_maps), then the
            optics Gaussian (cv2.GaussianBlur's kernel for float images, ksize = round(8 sigma
            + 1) | 1, BORDER_REFLECT) as two 1-D convolutions
    colour  front  pi R / rho * L_PLA            (wb 'reference': g = 1)
            grip   L_PLA * sat(pi R / rho, sigma) / s   (wb 1, s = GRIP_EXPOSURE)
    encode  clip, sRGB, round to uint8
    cv2 quantises remap weights to 1/32 px and Response saturates in float64. MEASURED on
    the same HdrColor frames against the host code: front at most 2 levels apart (0.75 %
    of the values differ at all), wrist at most 1 (0.18 %)."""

    def __init__(self, resp, spec, device):
        import cv2
        import torch

        from ..look import colour

        self.torch, self.front = torch, resp.cam == "front"
        mx, my = resp.maps
        grid = np.stack([2.0 * mx / (spec.width - 1) - 1.0, 2.0 * my / (spec.height - 1) - 1.0], -1)
        self.grid = torch.from_numpy(grid[None].astype(np.float32)).to(device)
        self.size = (spec.height, spec.width)
        s = float(resp.sigma_px)
        self.kx = self.ky = None
        if s > 0:
            ks = int(round(s * 8 + 1)) | 1
            k = torch.from_numpy(cv2.getGaussianKernel(ks, s, cv2.CV_32F).ravel().copy()).to(device)
            self.kx, self.ky = k.view(1, 1, 1, ks).repeat(3, 1, 1, 1), k.view(1, 1, ks, 1).repeat(3, 1, 1, 1)
            p = ks // 2
            H, W = mx.shape

            def reflect(n):
                return torch.cat([torch.arange(p - 1, -1, -1), torch.arange(n), torch.arange(n - 1, n - 1 - p, -1)]).to(device)

            self.ix, self.iy = reflect(W), reflect(H)
        self.gain = float(np.pi / resp.rho)
        self.L = torch.tensor(np.asarray(resp.L, np.float32), device=device).view(3, 1, 1)
        self.luma = torch.tensor(np.asarray(colour.LUMA, np.float32), device=device).view(3, 1, 1)
        self.sigma, self.s = float(resp.sigma), float(resp.grip_exposure)

    def __call__(self, hdr) -> np.ndarray:
        """(H', W', 3|4) pinhole radiance (GPU tensor) -> the camera's (480, 640, 3) uint8."""
        torch = self.torch
        F = torch.nn.functional
        x = hdr[..., :3].float().permute(2, 0, 1)[None]
        y = F.grid_sample(x, self.grid, mode="bilinear", padding_mode="border", align_corners=True)
        if self.kx is not None:
            y = F.conv2d(y.index_select(3, self.ix), self.kx, groups=3)
            y = F.conv2d(y.index_select(2, self.iy), self.ky, groups=3)
        a = y[0] * self.gain
        if self.front:
            lin = a * self.L
        else:
            lum = (a * self.luma).sum(0, keepdim=True)
            lin = self.L * (lum + self.sigma * (a - lum)) / self.s
        lin = lin.clamp(0.0, 1.0)
        srgb = torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.pow(1.0 / 2.4) - 0.055)
        return (srgb * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()


# --- the engine ----------------------------------------------------------------------------

class IsaacEngine(Engine):
    name = "isaac"
    _app = None  # SimulationApp, from boot()
    _lock_fd = None
    _exit_code = 0
    _socket = _quit_sub = None

    @classmethod
    def boot(cls, args=None) -> None:
        if cls._app is not None:
            return
        print(hold_gpu_lock(), flush=True)
        from isaaclab.app import AppLauncher

        cls._app = AppLauncher({"headless": not bool(getattr(args, "viewer", False)), "enable_cameras": True}).app
        # Kit hangs at interpreter exit on this box: whatever ends the process (a finished
        # session, an error before serve), end it with os._exit and the right status
        import atexit

        hook = sys.excepthook

        def excepthook(*a):
            cls._exit_code = 1
            hook(*a)

        sys.excepthook = excepthook
        atexit.register(cls._hard_exit)
        cls._socket = getattr(args, "socket", None)
        cls._quit_guard()

    @classmethod
    def _quit_guard(cls) -> None:
        """End the process when Kit is asked to quit: closing the viewer window posts that.

        Kit's own shutdown then hangs on this box (the next app update never returns,
        MEASURED with post_quit), and a hung server holds the GPU lock for good. A PUSH
        subscriber still runs inside that update (pop subscribers never did), but
        try_cancel_shutdown from it did not prevent the hang. So the session ends here:
        the server's socket file removed, os._exit(0). Kit cannot outlive its window, so
        with Isaac closing the viewer ends the session rather than continuing headless."""
        import omni.kit.app

        def on_quit(_e):
            from ..live.server import default_socket

            print("isaac: Kit was asked to quit (the viewer window closed?); ending the session", flush=True)
            path = str(cls._socket or default_socket())
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
            cls._hard_exit()

        cls._quit_sub = omni.kit.app.get_app().get_shutdown_event_stream().create_subscription_to_push_by_type(
            omni.kit.app.POST_QUIT_EVENT_TYPE, on_quit, name="real2sim live: end on quit")

    @classmethod
    def _hard_exit(cls) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(cls._exit_code)

    def __init__(self, scene, fps: float = 30.0, viewer: bool = False, cameras=("front", "grip"), physics=()):
        if IsaacEngine._app is None:
            raise RuntimeError("IsaacEngine.boot() must start Kit first")
        import torch

        from . import look as LK
        from . import params as P
        from . import scene as S
        from . import servo as SV

        self.torch, self.S = torch, S
        self.scene, self.fps, self.viewer = scene, float(fps), bool(viewer)
        self.prm = P.override(P.defaults(), physics)
        ph = self.prm["physics"]
        n = float(ph["hz"]) / self.fps
        if abs(n - round(n)) > 1e-9:
            raise ValueError(f"physics {ph['hz']} Hz is not a multiple of {self.fps:g} fps")
        self.sub, self.dt = int(round(n)), 1.0 / float(ph["hz"])
        self.n_settle = int(round(ph["settle_s"] * ph["hz"]))
        self.model = SV.load_model(scene)
        self.gains = SV.physx_gains(self.model, True)
        self.fw = SV.firmware_limits(scene.units())
        self.lim_j = S.joint_limits_rad(scene, ph["limit_margin_deg"])
        self.cams = tuple(cameras)
        self.cameras = {c: (scene.camera(c).height, scene.camera(c).width) for c in self.cams}
        self.light_units = LK.units_for(scene.ds, "rt")
        self.sim = None
        self.obj_names: list = []
        self._k = 0
        self._rendered = False
        self._reset_flag = False
        self._kb_sub = None
        self._last_ui = 0.0
        self.timings: dict = {}
        self._ms = {"step": deque(maxlen=150), "render": deque(maxlen=150)}  # status(): where the time goes

    # --- lifecycle -------------------------------------------------------------------------
    def reset(self, layout: dict, rest_q: np.ndarray, hold_q=None) -> None:
        t0 = time.monotonic()
        sc, ep = layouts.scene_with_layout(self.scene, layout)
        placed = {o["name"]: o for o in self.S.object_placements(sc, self._opts(ep, rest_q))}
        built = self.sim is None
        if built:
            self._build(sc, ep, rest_q, placed)
        missing = sorted(set(placed) - set(self.obj_names))
        if missing:
            raise ValueError(f"layout {layout.get('kind')}: no body for {missing} on this stage ({self.obj_names})")
        t1 = time.monotonic()
        self.layout_scene = sc
        self._place(placed, rest_q, hold_q)
        self._warmup(STEADY_MAX if built else 0)
        t2 = time.monotonic()
        self.timings.update(reset_s=round(t2 - t0, 3), built=built, build_s=round(t1 - t0, 3),
                            place_settle_render_s=round(t2 - t1, 3))
        print(f"isaac reset: {'built the stage and placed' if built else 're-placed'} {self.caps} + mug in "
              f"{t2 - t0:.2f} s (build {t1 - t0:.2f} s, place + settle + renders {t2 - t1:.2f} s)", flush=True)

    def _opts(self, ep: int, rest_q):
        return self.S.Options(episode=ep, num_envs=1, render=False, cameras=self.cams, placement="config",
                              params=self.prm, q0=tuple(float(v) for v in rest_q))

    def _build(self, sc, ep: int, rest_q, placed: dict) -> None:
        """The stage, once: replay.py's construction of this layout (its caps, then the mug,
        in the replay's order) plus the spare cap bodies of POOL_CAPS appended after them,
        and lookrender.py's look, camera prims and HdrColor render products."""
        import omni.replicator.core as rep
        from isaaclab.scene import InteractiveScene

        from . import look as LK
        from . import lookrender as LR
        from . import servo as SV
        from . import usd_assets
        from .. import camera as cam_mod, paths, units as U
        from ..blender import camera_model as CM

        torch, S = self.torch, self.S
        opts = self._opts(ep, rest_q)
        # the look is built from the dataset's own scene (look.json's), not the layout copy
        look = LK.LookAssets(self.scene, units=self.light_units)
        sim = S.make_sim(opts, look)
        cfg = S.scene_cfg(sc, opts, self.gains, look)
        mats = self.prm["materials"]
        cap_usd = usd_assets.build(sc, paths.out_dir(sc.ds, "isaac", "usd", create=True),
                                   {"cap": mats["cap"], "mug": mats["mug"]}, look.object_looks())["cap"]
        spare = [f"cap_{c}" for c in POOL_CAPS if f"cap_{c}" not in placed]
        for i, name in enumerate(spare):
            setattr(cfg, name, S.rigid_cfg(name, cap_usd, self._park(i), (1.0, 0.0, 0.0, 0.0), opts))
        iscene = InteractiveScene(cfg)
        self.spawn_info = S.post_spawn(sim.stage, iscene, sc, opts, look)
        cam_paths = LR.spawn_cameras(sc, self.cams)
        # every cap's visibility authored now, before PhysX parses: authoring it for the first
        # time later (parking a cap) makes omni.physx resync the body, which logs rejected
        # PxRigidDynamic::setGlobalPose calls under the direct GPU API (MEASURED); value
        # changes afterwards are silent
        from pxr import UsdGeom

        for n in list(placed) + spare:
            if n.startswith("cap_"):
                img = UsdGeom.Imageable(sim.stage.GetPrimAtPath(f"/World/envs/env_0/{n}"))
                img.MakeVisible() if n in placed else img.MakeInvisible()
        sim.reset()
        look.post_reset(sim)
        robot = iscene["robot"]
        self.dev = robot.device
        names = list(robot.joint_names)
        self.perm = [names.index(n) for n in U.URDF_JOINTS]
        self.servo = SV.PhysxServo(self.model, self.perm, self.dt, 1, self.dev, flex=True)
        g = self.servo.physx_tensors()
        robot.write_joint_stiffness_to_sim(g["stiffness"].unsqueeze(0))
        robot.write_joint_damping_to_sim(g["damping"].unsqueeze(0))
        robot.write_joint_armature_to_sim(g["armature"].unsqueeze(0))
        robot.write_joint_effort_limit_to_sim(g["max_force"].unsqueeze(0))
        self.readback = S.after_reset(iscene, sc, opts)
        self.obj_names = list(placed) + spare
        self.assets = [iscene[n] for n in self.obj_names]
        self.origin = iscene.env_origins[:1]
        self.u = torch.zeros(1, len(names), device=self.dev)
        self.zeros = torch.zeros_like(self.u)
        self._shown = {n: n in placed for n in self.obj_names}
        # the look's camera: one render product and one HdrColor annotator per camera,
        # data left on the GPU for _GpuCamera
        self.specs = {c: cam_mod.render_spec(sc.camera(c), square=True) for c in self.cams}
        self.rps, self.anns, self.cam = {}, {}, {}
        for c in self.cams:
            self.rps[c] = rep.create.render_product(cam_paths[c], (self.specs[c].width, self.specs[c].height))
            self.anns[c] = rep.AnnotatorRegistry.get_annotator("HdrColor", device="cuda")
            self.anns[c].attach([self.rps[c]])
            resp = CM.Response(sc, look.look, c, self.specs[c], None, wb="reference", exposure="constant")
            self.cam[c] = _GpuCamera(resp, self.specs[c], self.dev)
        self.render_settings = look.set_mode("rt")
        self.look, self.sim, self.iscene, self.robot = look, sim, iscene, robot
        if self.viewer:
            self._viewer_setup()

    def _park(self, i: int) -> tuple:
        """Where spare cap i waits, closed end up: on the floor, 3 m out, far outside the
        arm's reach and the backdrop, and invisible to the cameras (_place)."""
        from .scene import FLOOR_BELOW

        return (PARK_X + 0.1 * i, 0.0, self.scene.table_z() - FLOOR_BELOW)

    def _hold(self, qa) -> None:
        r = self.robot
        r.write_joint_state_to_sim(qa, self.zeros)
        r.set_joint_position_target(qa)
        r.set_joint_velocity_target(self.zeros)
        r.set_joint_effort_target(self.zeros)

    def _place(self, placed: dict, rest_q, hold_q=None) -> None:
        """replay.py's initial state: arm at rest_q, objects placed once (spare caps parked),
        the settle with the arm held exactly at its start, then the servo and the goals at
        rest; t = 0."""
        from pxr import UsdGeom

        torch = self.torch
        dev, dt, r = self.dev, self.dt, self.robot
        q = np.clip(np.asarray(rest_q, dtype=float), self.lim_j[:, 0] + 1e-4, self.lim_j[:, 1] - 1e-4)
        qa = torch.zeros(1, len(self.perm), device=dev)
        qa[:, self.perm] = torch.tensor(q, dtype=torch.float32, device=dev)
        self._hold(qa)
        spare = 0
        for name, a in zip(self.obj_names, self.assets):
            if name in placed:
                pos, quat = placed[name]["pos"], placed[name]["quat"]
            else:
                pos, quat, spare = self._park(spare), (1.0, 0.0, 0.0, 0.0), spare + 1
            st = torch.zeros(1, 13, device=dev)
            st[0, :3] = torch.tensor(np.array(pos, dtype=float), dtype=torch.float32, device=dev) + self.origin[0]
            st[0, 3:7] = torch.tensor(quat, dtype=torch.float32, device=dev)
            a.write_root_state_to_sim(st)
            if self._shown[name] != (name in placed):
                img = UsdGeom.Imageable(self.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{name}"))
                img.MakeVisible() if name in placed else img.MakeInvisible()
                self._shown[name] = name in placed
        self.caps = [n for n in self.obj_names if n in placed and n.startswith("cap_")]
        self.iscene.write_data_to_sim()
        self.sim.forward()
        r.update(0.0)
        for a in self.assets:
            a.update(0.0)
        for _ in range(self.n_settle):
            self._hold(qa)
            r.write_data_to_sim()
            self.sim.step(render=False)
            r.update(dt)
            for a in self.assets:
                a.update(dt)
        self._hold(qa)
        r.write_data_to_sim()
        self.sim.forward()
        r.update(0.0)
        self.servo.reset(qa)
        # until the first command takes effect the servo holds the rest pose (live.goals)
        rq = np.asarray(rest_q, dtype=float)
        self.goals = OnlineGoals(self.model.dead_time, self.fw, rq if hold_q is None else np.asarray(hold_q, dtype=float),
                                 self.model.max_velocity, target0=rq)
        self._k = 0
        self._rendered = False

    def _warmup(self, steady_max: int) -> None:
        """Renders before the first observation: WARMUP_RENDERS, so the renderer's temporal
        history is the new layout, and after the build until the front view is steady
        (textures load asynchronously; lookrender.settle's test), up to steady_max."""
        prev, n, diffs = None, 0, []
        while True:
            self.sim.render()
            n += 1
            img = self._hdr(self.cams[0]).float()
            for c in self.cams[1:]:
                self._hdr(c)
            if prev is not None:
                diffs.append(float((img - prev).abs().mean()) / max(float(img.abs().mean()), 1e-6))
            if n >= max(WARMUP_RENDERS, steady_max):
                break
            if n >= WARMUP_RENDERS and (not steady_max or (diffs and diffs[-1] <= STEADY_TOL)):
                break
            prev = img.clone()
        self.timings["warmup_renders"] = n
        self.timings["warmup_rel_diff"] = [round(d, 5) for d in diffs]
        self._rendered = True

    def close(self) -> None:
        """End the process (module doc: Kit hangs in app.close). Called by the server once
        it has closed its socket; an exception in flight still exits non-zero."""
        if sys.exc_info()[0] is not None:
            import traceback

            traceback.print_exc()
            IsaacEngine._exit_code = 1
        IsaacEngine._hard_exit()

    # --- stepping ----------------------------------------------------------------------------
    @property
    def t(self) -> float:
        return self._k / self.fps

    def command(self, goal) -> None:
        self.goals.push(self.t, goal)

    def step(self) -> None:
        """One frame: replay.py's action-mode inner loop, the goal from OnlineGoals."""
        torch, t0 = self.torch, time.perf_counter()
        r, sim, dt, n, u, perm = self.robot, self.sim, self.dt, self.sub, self.u, self.perm
        for s in range(n):  # the replay's substep clock: (k + s/n) / fps
            u[:, perm] = torch.as_tensor(self.goals.at((self._k + s / n) / self.fps), device=self.dev).float()
            tgt, vel, ff = self.servo.step(r.data.joint_pos, r.data.joint_vel, u)
            r.set_joint_position_target(tgt)
            r.set_joint_velocity_target(vel)
            r.set_joint_effort_target(ff)
            r.write_data_to_sim()
            sim.step(render=False)
            r.update(dt)
            for a in self.assets:
                a.update(dt)
        self._k += 1
        self._rendered = False
        self._ms["step"].append(time.perf_counter() - t0)

    def state(self) -> np.ndarray:
        """The encoders: the joint angles, the Jaw's HORN (the pose log's q), URDF rad."""
        return self.servo.encoder(self.robot.data.joint_pos)[0, self.perm].double().cpu().numpy()

    def poses(self) -> dict:
        """{object: (pos, quat wxyz)} in the base frame."""
        o = self.origin[0]
        return {n: ((a.data.root_pos_w[0] - o).double().cpu().numpy(), a.data.root_quat_w[0].double().cpu().numpy())
                for n, a in zip(self.obj_names, self.assets)}

    def status(self) -> dict:
        from ..metrics import cap_in_mug

        p, sc = self.poses(), self.layout_scene
        mp, mq = p["mug"]
        cim = [bool(cap_in_mug(*p[c], mp, mq, sc.cap_dims(), sc.mug_dims())) for c in self.caps]
        finite = bool(np.isfinite(self.state()).all() and all(np.isfinite(np.r_[a, b]).all() for a, b in p.values()))
        ms = {f"{k}_ms_p50": round(float(np.median(v)) * 1e3, 1) for k, v in self._ms.items() if v}
        return {"caps": list(self.caps), "caps_in_mug": cim, "finite": finite, **ms}

    # --- images --------------------------------------------------------------------------------
    def _hdr(self, c: str):
        d = self.anns[c].get_data()
        if isinstance(d, np.ndarray):
            t = self.torch.from_numpy(d).to(self.dev)
        else:
            import warp as wp

            t = wp.to_torch(d)
        if t.dim() != 3 or tuple(t.shape[:2]) != self.cam[c].size:
            raise RuntimeError(f"{c}: HdrColor {tuple(t.shape)}, expected {self.cam[c].size} (+ channels)")
        return t

    def render(self, cams) -> dict:
        """One RTX frame (both render products render on every frame: a product whose
        updates were disabled came back empty, isaac README), both annotators read so none
        backs up, the requested cameras through _GpuCamera."""
        t0 = time.perf_counter()
        self.sim.render()
        self._rendered = True
        hdr = {c: self._hdr(c) for c in self.cams}
        out = {c: self.cam[c](hdr[c]) for c in cams}
        self._ms["render"].append(time.perf_counter() - t0)
        return out

    # --- viewer ----------------------------------------------------------------------------------
    def _viewer_setup(self) -> None:
        """The Kit window's camera and key R.

        The viewport shows the look as RTX tonemaps it at the render settings the cameras
        are calibrated at: dark (LOOK's scene units, lit white PLA ~0.3), but the arm, caps
        and mug read. No render setting may brighten it: RTX applies exposure and colour
        grading BEFORE the HdrColor the cameras read (MEASURED: film ISO x 2^10 scaled
        HdrColor x 1019, auto exposure x 2.3, colour-grading and cm2 gains likewise), and a
        camera's own USD exposure is ignored. The window closing: _quit_guard."""
        import carb.input
        import omni.appwindow

        self.sim.set_camera_view(eye=VIEWER_EYE, target=VIEWER_TARGET)
        if self._kb_sub is not None:
            return

        def on_key(e, *_):
            if e.type == carb.input.KeyboardEventType.KEY_PRESS and e.input == carb.input.KeyboardInput.R:
                self._reset_flag = True
            return True

        inp = carb.input.acquire_input_interface()
        self._kb_sub = inp.subscribe_to_keyboard_events(omni.appwindow.get_default_app_window().get_keyboard(), on_key)

    def viewer_sync(self) -> bool:
        """Keep the Kit window alive: one app update (it renders the viewport) unless an
        observation rendered since the last step, at most 30 per second. Closing the
        window ends the process first (_quit_guard); False only if Kit stopped otherwise."""
        if not self.viewer:
            return True
        if not self._app.is_running():
            self.viewer = False
            return False
        now = time.monotonic()
        if not self._rendered and now - self._last_ui >= 1.0 / 30.0:
            self.sim.render()
            self._rendered, self._last_ui = True, now
        return True

    def reset_requested(self) -> bool:
        f, self._reset_flag = self._reset_flag, False
        return f
