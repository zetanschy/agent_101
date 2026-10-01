"""The simulated follower behind lerobot's Robot interface (sim/real2sim/live/).

get_observation and send_action are one socket round trip each to live.server, which
steps MuJoCo or Isaac with the identified servo and renders the fitted cameras. What
comes back is what the real bus returns: '<motor>.pos' in degrees (use_degrees=True, as
this bench records) or lerobot's -100..100 range normalisation (use_degrees=False, what
the openpi checkpoints were trained on), the gripper in percent either way, and one
HxWx3 uint8 image per camera. So
lerobot-teleoperate, lerobot-record, lerobot-rollout and the async robot client cannot
tell it from the arm, and a dataset recorded from it has the real dataset's features.
"""

from __future__ import annotations

import logging
import os

from lerobot.robots.robot import Robot

from .config_real2sim import Real2SimConfig

logger = logging.getLogger(__name__)

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
DEFAULT_CAMERAS = {"front": (480, 640), "grip": (480, 640)}


class _NewLayoutEachEpisode(logging.Handler):
    """lerobot-record gives a robot no hook between episodes, but it announces its reset
    phase through logging (log_say("Reset the environment"), lerobot_record.py), in this
    same process. The next observation after that line draws a new layout. The next
    episode then starts on fresh cap and mug positions, while the operator brings the
    leader back to rest. Off with R2S_NEW_LAYOUT_EACH_EPISODE=0."""

    def __init__(self, robot):
        super().__init__(logging.INFO)
        self.robot = robot

    def emit(self, record):
        if record.getMessage() == "Reset the environment":
            self.robot._reset_pending = True


class Real2Sim(Robot):
    config_class = Real2SimConfig
    name = "real2sim"

    def __init__(self, config: Real2SimConfig):
        super().__init__(config)
        self.config = config
        # {name: config}: lerobot sizes its image writers by len(robot.cameras)
        self.cameras = dict(config.cameras) if config.cameras else {k: None for k in DEFAULT_CAMERAS}
        self._sock = None
        self.info: dict = {}
        self._reset_pending = False
        self._keys = None
        self._episode_hook = None
        # degrees (this bench) or RANGE_M100_100 on the arm, as SOFollower's use_degrees
        self._units = "degrees" if config.use_degrees else "normalized"

    # --- features: the real follower's ---------------------------------------------
    @property
    def _cam_sizes(self) -> dict:
        out = {}
        for k, c in self.cameras.items():
            h, w = DEFAULT_CAMERAS.get(k, (480, 640))
            out[k] = (getattr(c, "height", None) or h, getattr(c, "width", None) or w)
        return out

    @property
    def observation_features(self) -> dict:
        return {**{f"{m}.pos": float for m in MOTORS}, **{k: (h, w, 3) for k, (h, w) in self._cam_sizes.items()}}

    @property
    def action_features(self) -> dict:
        return {f"{m}.pos": float for m in MOTORS}

    # --- lifecycle -------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def connect(self, calibrate: bool = True) -> None:
        from real2sim.live import protocol

        try:
            self._sock = protocol.connect(self.config.port, self.config.timeout_s)
        except OSError as e:
            raise ConnectionError(f"no live sim at {self.config.port} ({e}): start it with "
                                  "'./robot <cmd> --sim mujoco|isaac' or './robot real2sim live serve'") from e
        self._warp = self._cv2 = None
        try:  # warp the sim's pinhole renders here (cv2, ~1 ms a camera) instead of on its one thread
            import cv2

            self._cv2 = cv2
        except ImportError:
            pass
        self.info = protocol.call(self._sock, "hello", warp=self._cv2 is not None)
        if self._cv2 is not None and self.info.get("warp"):
            cv2 = self._cv2
            self._warp = {}
            for k, w in self.info["warp"].items():
                m1, m2 = cv2.convertMaps(w["map_x"], w["map_y"], cv2.CV_16SC2)  # fixed-point maps: faster remap
                self._warp[k] = (m1, m2, float(w.get("sigma_px") or 0.0))
        have = {k: tuple(v) for k, v in self.info["cameras"].items()}
        for k, hw in self._cam_sizes.items():
            if k not in have:
                raise ValueError(f"camera {k!r}: the sim renders {sorted(have)}")
            if have[k] != hw:
                raise ValueError(f"camera {k!r}: configured {hw[1]}x{hw[0]}, the sim renders {have[k][1]}x{have[k][0]}")
        logger.info(f"{self} connected: {self.info['engine']} ({self.info['clock']} clock), scene {self.info['scene_hash']}")
        self._listen_for_reset()
        if os.environ.get("R2S_NEW_LAYOUT_EACH_EPISODE", "1") not in ("0", ""):
            self._episode_hook = _NewLayoutEachEpisode(self)
            logging.getLogger().addHandler(self._episode_hook)

    def _listen_for_reset(self) -> None:
        """'l' = new object layout now, in the terminal that runs lerobot (pynput, the
        listener lerobot-record uses for its own keys). Not 'r': lerobot-record binds r to
        re-record. A recording also draws one by itself at every reset phase
        (_NewLayoutEachEpisode), never on a timer, which could teleport objects mid-episode."""
        try:
            from pynput import keyboard

            def press(key):
                if getattr(key, "char", None) == "l":
                    self._reset_pending = True

            self._keys = keyboard.Listener(on_press=press)
            self._keys.start()
            logger.info(f"{self}: press 'l' for a new object layout (a recording draws one every reset phase)")
        except Exception as e:  # headless: `./robot real2sim live reset` does the same
            logger.info(f"{self}: no keyboard listener ({type(e).__name__}); reset with './robot real2sim live reset'")

    @property
    def is_calibrated(self) -> bool:
        return True  # the sim's joint map is the scene's fitted calibration

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self._episode_hook is not None:
            logging.getLogger().removeHandler(self._episode_hook)
            self._episode_hook = None
        if self._keys is not None:
            self._keys.stop()
            self._keys = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # --- I/O ---------------------------------------------------------------------------
    def get_observation(self) -> dict:
        from real2sim.live import protocol

        if self._reset_pending:
            self._reset_pending = False
            layout = self.reset_scene()
            logger.info(f"{self}: new layout {layout.get('kind')}")
        r = protocol.call(self._sock, "observe", images=list(self.cameras), units=self._units,
                          pinhole=self._warp is not None)
        obs = {f"{m}.pos": float(r["state"][m]) for m in MOTORS}
        for k in self.cameras:
            img = r["images"][k]  # writable: protocol.recv backs arrays with fresh bytearrays
            if r.get("pinhole") and self._warp is not None:
                img = self._warped(k, img)
            obs[k] = img
        return obs

    def _warped(self, cam: str, pinhole):
        """The fitted camera's distortion (and lens softness) on the sim's pinhole render:
        the same remap maps the server would apply (camera.remap_maps), bilinear, black
        outside, then the Gaussian the Blender camera model fitted."""
        cv2 = self._cv2
        m1, m2, sigma = self._warp[cam]
        img = cv2.remap(pinhole, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img

    def send_action(self, action: dict) -> dict:
        from real2sim.live import protocol

        goal = {m: float(action[f"{m}.pos"]) for m in MOTORS}
        if self.config.max_relative_target is not None:  # the same guard SOFollower applies
            from lerobot.robots.utils import ensure_safe_goal_position

            present = protocol.call(self._sock, "observe", images=[], units=self._units)["state"]
            safe = ensure_safe_goal_position({m: (goal[m], present[m]) for m in MOTORS},
                                             self.config.max_relative_target)
            goal = {m: float(safe[m]) for m in MOTORS}
        protocol.call(self._sock, "act", action=goal, units=self._units)
        return {f"{m}.pos": v for m, v in goal.items()}

    # --- sim-only extras (used by this repo's scripts, never by lerobot) --------------
    def reset_scene(self, layout: str | None = None) -> dict:
        """A new object layout now: real:N, random, random:K (default: the server's)."""
        from real2sim.live import protocol

        return protocol.call(self._sock, "reset", **({"layout": layout} if layout else {}))["layout"]

    def sim_status(self) -> dict:
        from real2sim.live import protocol

        return protocol.call(self._sock, "status")
