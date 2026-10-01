"""MuJoCo as the live follower: the replay's model, servo and renderer, stepped on demand.

Nothing here is new physics. The model is mujoco.model.build (the CoACD fingers, the
contact parameters and the payload of the evaluated replay), the servo is the identified
one from config/<ds>.servo.json, and a step is the replay's inner loop: n substeps, each
with the firmware goal the servo sees at that instant. So a leader or a policy drives
the same follower that put 6 of 6 caps in the mug on replay. With the recorded actions
fed in lockstep, this engine reproduces the offline replay (live/tests/test_live_mujoco.py).

Images come from MuJoCo's EGL renderer through the fitted cameras and their distortion
(1.5 ms per render + the torch warp: 8.6 ms for both cameras), dressed with the parts of
LOOK a rasterizer can carry (look_mujoco.py: the recorded mat texture, the fitted key
light). They are still a rasterizer's: a policy trained on real frames sees no
reflections in the mug and no room behind the arm. For photoreal
frames of a recorded episode, render its pose log with `./robot real2sim blender render`.
"""

from __future__ import annotations

import os

import numpy as np

from .engine import Engine
from .goals import OnlineGoals
from . import layouts

KEY_R = 82  # GLFW key code the viewer passes for 'R'


def _optics() -> dict:
    """{camera: blur sigma px}: the webcams' softness, fitted once for the Blender renderer."""
    try:
        from ..blender.camera_model import OPTICS

        return {c: float(v["sigma_px"]) for c, v in OPTICS.items()}
    except Exception:
        return {}


class _Warp:
    """The distortion warp of mujoco.render.FastRemap (the same bilinear sample, zero
    outside the pinhole) through torch's grid_sample, which is multithreaded C++.

    FastRemap's numpy taps cost 14 ms per 640x480 camera (MEASURED), i.e. 28 of a 30 Hz
    tick's 33 ms for both cameras. A single-gather fixed-point numpy version was slower
    still (35 ms). torch is already in the mjlab env, and on the CPU it leaves the GPU to
    the policy. With align_corners=True, -1..1 are the centres of the edge pixels: the
    OpenCV pixel convention of camera.remap_maps."""

    def __init__(self, cam, spec, sigma_px: float = 0.0):
        import torch

        from .. import camera as C

        mx, my = C.remap_maps(cam, spec)
        gx = 2.0 * mx / (spec.width - 1) - 1.0
        gy = 2.0 * my / (spec.height - 1) - 1.0
        self.torch = torch
        self.grid = torch.from_numpy(np.stack([gx, gy], -1)[None].astype(np.float32))
        self.kernel = None
        if sigma_px > 0:  # the lens softness Blender fitted (blender/camera_model.py OPTICS), separable
            r = int(np.ceil(2 * sigma_px))
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2)
            k = torch.from_numpy((k / k.sum()).astype(np.float32))
            self.kx, self.ky, self.r = k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), r
            self.kernel = k

    def _blur(self, x):
        """The Gaussian on the sRGB values, truncated at 2 sigma. Blender blurs in linear
        light; at sigma 0.8 / 2.5 px the difference is a fraction of a grey level away from
        edges, and skipping the two gamma passes and the 3rd sigma saves ~5 of the 12 ms
        the blur costs (MEASURED), which is what keeps both cameras in a 30 Hz tick."""
        F = self.torch.nn.functional
        x = F.conv2d(F.pad(x, (self.r, self.r, 0, 0), mode="replicate"), self.kx, groups=3)
        return F.conv2d(F.pad(x, (0, 0, self.r, self.r), mode="replicate"), self.ky, groups=3)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        t = self.torch
        x = t.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
        y = t.nn.functional.grid_sample(x, self.grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        if self.kernel is not None:
            y = self._blur(y)
        return y[0].permute(1, 2, 0).add_(0.5).clamp_(0, 255).to(t.uint8).numpy()


class MujocoEngine(Engine):
    name = "mujoco"

    def __init__(self, scene, fps: float = 30.0, viewer: bool = False, cameras=("front", "grip"), look: bool = True):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from ..mujoco.servo import ServoModel, firmware_limits_rad

        self.scene, self.fps, self.cams = scene, float(fps), tuple(cameras)
        self.servo = ServoModel.from_scene(scene)
        self.limits = firmware_limits_rad(scene.units())
        self.cameras = {c: (scene.camera(c).height, scene.camera(c).width) for c in self.cams}
        self.want_viewer, self.viewer = viewer, None
        self.b = self.d = self.renderer = None
        self._k = 0
        self._t_mj = 0.0
        self._reset_flag = False
        self._look = None
        if look:  # the dataset-derived mat and key light (look_mujoco.py); plain colours without look build
            try:
                from . import look_mujoco

                self._look = look_mujoco.hook(scene.ds)
            except FileNotFoundError as e:
                print(f"no look assets ({e}); './robot real2sim look build' adds the real mat and light", flush=True)

    # --- lifecycle -------------------------------------------------------------------
    def reset(self, layout: dict, rest_q: np.ndarray, hold_q=None, settle_s: float = 0.3) -> None:
        """A layout with the same bodies as the current model (the default random:1 always
        is) is re-placed IN this model: no rebuild, and the viewer window stays open.
        Anything else builds a fresh model. The first reset always builds, which is the
        path the replay-equivalence test checks bitwise."""
        import mujoco

        from .. import objects as objlib
        from ..mujoco import model as mdl
        from ..mujoco.render import Renderer

        sc, ep = layouts.scene_with_layout(self.scene, layout)
        q0 = np.asarray(rest_q, dtype=float)
        obs = objlib.episode_objects(sc, ep)
        if self.b is not None and [o["name"] for o in self.b.objects] == [o["name"] for o in obs]:
            with self._viewer_lock():
                self._replace(sc, obs, q0, settle_s)
            self._after_reset(sc, q0, hold_q)
            return
        self.layout_scene = sc
        b = mdl.build(sc, ep, self.servo, mdl.Physics(), mdl.Perturbation(), spec_hook=self._look)
        d = mujoco.MjData(b.model)
        self._settle(b, d, q0, settle_s)
        if self.renderer is not None:
            self.renderer.close()
        self.b, self.d, self._k = b, d, 0
        self._after_reset(sc, q0, hold_q)
        self.renderer = Renderer(b, self.cams)
        try:
            sig = _optics() if self._look is not None else {}
            self._warp = {c: _Warp(b.cameras[c]["cam"], b.cameras[c]["spec"], sig.get(c, 0.0)) for c in self.cams}
        except ImportError:  # no torch: FastRemap inside Renderer.render
            self._warp = None
        self._caps = [o for o in b.objects if o["kind"] == "cap"]
        self._mug = next(o for o in b.objects if o["kind"] == "mug")
        if self.want_viewer:
            self._open_viewer()

    @staticmethod
    def _settle(b, d, q0, settle_s: float) -> None:
        """The replay's reset: the arm held exactly at q0 while the objects settle; t = 0."""
        import mujoco

        from ..mujoco import model as mdl

        mdl.set_arm_state(b, d, q0)
        d.ctrl[b.act] = q0
        mujoco.mj_forward(b.model, d)
        arm_q0 = d.qpos[b.qadr].copy()
        for _ in range(int(round(settle_s / b.model.opt.timestep))):
            mujoco.mj_step(b.model, d)
            d.qpos[b.qadr] = arm_q0
            d.qvel[b.dadr] = 0.0
        d.time = 0.0
        mdl.set_arm_state(b, d, q0)
        mujoco.mj_forward(b.model, d)

    def _replace(self, sc, obs, q0, settle_s: float) -> None:
        """New poses for the same bodies: each object's free joint set where build() would
        have placed it (objects.episode_objects, on the table: an open-up cap on its top),
        velocities and solver state cleared by mj_resetData, then the same settle."""
        import mujoco

        from ..mujoco import model as mdl

        b, d = self.b, self.d
        mujoco.mj_resetData(b.model, d)
        cap_h = sc.cap_dims()["height"]
        for o, ob in zip(b.objects, obs):
            pos, quat = np.array(ob["pos"], dtype=float), np.array(ob["quat"], dtype=float)
            pos[2] = float(mdl.table_top(sc, *pos[:2])) + (cap_h if ob["kind"] == "cap" and ob.get("up") == "open" else 0.0)
            d.qpos[o["qadr"]:o["qadr"] + 3], d.qpos[o["qadr"] + 3:o["qadr"] + 7] = pos, quat
            o["pos0"], o["quat0"] = pos, quat
        self._settle(b, d, q0, settle_s)

    def _after_reset(self, sc, q0, hold_q) -> None:
        self.layout_scene, self._k, self._t_mj = sc, 0, float(self.d.time)
        self.goals = OnlineGoals(self.servo.dead_time, self.limits, q0 if hold_q is None else hold_q,
                                 self.servo.max_velocity, target0=q0)

    def _viewer_lock(self):
        import contextlib

        return self.viewer.lock() if self.viewer is not None else contextlib.nullcontext()

    def _viewer_reset_seen(self) -> bool:
        """MuJoCo's viewer resets OUR data on its Reset button / Backspace (mj_resetData:
        time back to 0, the arm to qpos0, the objects to where the model placed them). Sim
        time never runs backwards otherwise, so that is how it is recognised; the server
        then draws a new layout, which is what R does."""
        if self.d is not None and self.d.time + 1e-9 < self._t_mj:
            self._reset_flag = True
            return True
        return False

    def _open_viewer(self) -> None:
        import mujoco.viewer

        if self.viewer is not None:
            self.viewer.close()

        def key(code):
            if code == KEY_R:
                self._reset_flag = True

        try:
            self.viewer = mujoco.viewer.launch_passive(self.b.model, self.d, key_callback=key)
        except Exception as e:  # no display, or GLFW unavailable next to EGL
            print(f"viewer unavailable ({type(e).__name__}: {e}); serving headless", flush=True)
            self.want_viewer, self.viewer = False, None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()

    # --- stepping ----------------------------------------------------------------------
    @property
    def t(self) -> float:
        return self._k / self.fps

    def command(self, goal) -> None:
        self.goals.push(self.t, goal)

    def step(self) -> None:
        import mujoco

        if self._viewer_reset_seen():  # the server re-places before the next step
            return
        b, d, n = self.b, self.d, self.b.physics.substeps
        with self._viewer_lock():
            for s in range(n):  # the replay's substep clock: (k + s/n) / fps
                d.ctrl[b.act] = self.goals.at((self._k + s / n) / self.fps)
                mujoco.mj_step(b.model, d)
        self._k += 1
        self._t_mj = float(d.time)

    def state(self) -> np.ndarray:
        from ..mujoco import model as mdl

        return mdl.arm_q(self.b, self.d)

    def render(self, cams) -> dict:
        with self._viewer_lock():  # a viewer click writes the same data
            if self._warp is None:
                return {c: self.renderer.render(self.d, c) for c in cams}
            pin = {c: self.renderer.render(self.d, c, distort=False) for c in cams}
        return {c: self._warp[c](img) for c, img in pin.items()}

    def render_pinhole(self, cams) -> dict:
        with self._viewer_lock():
            return {c: self.renderer.render(self.d, c, distort=False) for c in cams}

    def warp_spec(self, cam: str) -> dict:
        from .. import camera as C

        b = self.b.cameras[cam]
        mx, my = C.remap_maps(b["cam"], b["spec"])
        sigma = _optics().get(cam, 0.0) if self._look is not None else 0.0
        return {"map_x": mx.astype(np.float32), "map_y": my.astype(np.float32), "sigma_px": float(sigma)}

    def status(self) -> dict:
        from ..metrics import cap_in_mug

        d, sc = self.d, self.layout_scene
        mp, mq = d.xpos[self._mug["body"]], d.xquat[self._mug["body"]]
        cim = [bool(cap_in_mug(d.xpos[c["body"]], d.xquat[c["body"]], mp, mq, sc.cap_dims(), sc.mug_dims()))
               for c in self._caps]
        return {"caps": [c["name"] for c in self._caps], "caps_in_mug": cim,
                "finite": bool(np.isfinite(d.qpos).all())}

    # --- viewer ------------------------------------------------------------------------
    def viewer_sync(self) -> bool:
        if self.viewer is None:
            return not self.want_viewer
        if not self.viewer.is_running():
            self.viewer = None
            return False
        self._viewer_reset_seen()
        self.viewer.sync()
        return True

    def reset_requested(self) -> bool:
        f, self._reset_flag = self._reset_flag, False
        return f
