"""What live.server needs from a physics engine (engine_mujoco.py, isaac/live.py).

Everything crossing this interface is in URDF radians and the base frame; the server
converts to lerobot units with the scene's fitted offsets and gripper map, so both
engines agree on what a leader command means and what the follower reports.
"""

from __future__ import annotations

import numpy as np


class Engine:
    name = "?"
    fps = 30.0
    cameras: dict = {}  # {name: (height, width)}, the fitted cameras as recorded

    @classmethod
    def boot(cls, args) -> None:
        """Start whatever must exist before the engine module imports (Isaac: Kit)."""

    def reset(self, layout: dict, rest_q: np.ndarray, hold_q: np.ndarray | None = None) -> None:
        """Objects to `layout` (layouts.py), arm to rest_q, objects settled; t = 0. The servo
        holds hold_q (default rest_q) until the first command takes effect, with its slewed
        target starting at rest_q: goals.OnlineGoals(q0=hold_q, target0=rest_q). A recording
        replayed over the socket passes its first state and first action (server --start
        recorded), which is GoalStream's start."""
        raise NotImplementedError

    def command(self, goal: np.ndarray) -> None:
        """A firmware goal (URDF rad, 6) sent now. It takes effect after the servo's dead
        time (goals.OnlineGoals) and is held until the next one does."""
        raise NotImplementedError

    def step(self) -> None:
        """Advance 1/fps of sim time."""
        raise NotImplementedError

    @property
    def t(self) -> float:
        raise NotImplementedError

    def state(self) -> np.ndarray:
        """What the encoders read (URDF rad, 6; the jaw is the horn, not the finger)."""
        raise NotImplementedError

    def render(self, cams) -> dict:
        """{name: (H, W, 3) uint8} through the fitted camera, distortion included."""
        raise NotImplementedError

    def render_pinhole(self, cams) -> dict | None:
        """Undistorted renders for a client that warps them itself with warp_spec (the
        lerobot plugin does, with cv2), or None when the engine only renders final frames.
        Warping on the client takes the per-pixel work off the single-threaded server."""
        return None

    def warp_spec(self, cam: str) -> dict | None:
        """{'map_x', 'map_y': (H, W) float32 in cv2.remap's convention, 'sigma_px': blur}:
        what turns render_pinhole(cam) into render(cam)."""
        return None

    def status(self) -> dict:
        """{'caps_in_mug': [bool per cap], 'caps': [names], 'finite': bool, ...}."""
        raise NotImplementedError

    def viewer_sync(self) -> bool:
        """Update the interactive viewer if there is one; False once it has been closed."""
        return True

    def reset_requested(self) -> bool:
        """True once after the operator asked for a reset from the viewer (key R)."""
        return False

    def close(self) -> None:
        pass
