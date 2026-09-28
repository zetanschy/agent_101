"""The servo's goal stream, online: commands arrive while the sim runs.

mujoco.servo.GoalStream holds a recorded goal array and looks goal k up at
t >= k/fps + dead_time. Live there is no array: a command pushed at sim time t_push
takes effect at t_push + dead_time and is held until the next one does -- the same
zero-order hold, dead time, firmware clamp and optional slew, so a teleoperated or
policy-driven follower behaves like the replayed one.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class OnlineGoals:
    """q0 is the goal held until the first command takes effect; target0 is where the
    firmware's slewed target starts (default q0). A live session starts both at the rest
    pose. A replay of a recording holds its first recorded goal while the target starts
    at the recorded state, which is GoalStream's behaviour."""

    def __init__(self, dead_time: float, limits=None, q0=None, max_velocity: float | None = None, target0=None):
        self.dead_time = float(dead_time)
        self.lim = None if limits is None else np.asarray(limits, dtype=float)
        self.q0 = np.zeros(6) if q0 is None else np.asarray(q0, dtype=float)
        self.vmax = max_velocity
        self._q: deque = deque()  # (t_effective, goal), increasing t
        self._active = self._clip(self.q0)
        self._target = np.array(target0, dtype=float) if target0 is not None else self._active.copy()
        self._t = None

    def _clip(self, g) -> np.ndarray:
        g = np.asarray(g, dtype=float)
        return g if self.lim is None else np.clip(g, self.lim[:, 0], self.lim[:, 1])

    def reset(self, q0, target0=None) -> None:
        self.__init__(self.dead_time, self.lim, q0, self.vmax, target0)

    def push(self, t: float, goal) -> None:
        self._q.append((float(t) + self.dead_time, self._clip(goal)))

    def at(self, t: float) -> np.ndarray:
        """The firmware goal at sim time t (call with increasing t)."""
        while self._q and self._q[0][0] <= t + 1e-9:
            self._active = self._q.popleft()[1]
        if self.vmax is None:
            return self._active.copy()
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        step = self.vmax * dt
        self._target = np.clip(self._active, self._target - step, self._target + step)
        return self._target.copy()

    @property
    def last(self) -> np.ndarray:
        """The newest command pushed (effective or not)."""
        return self._q[-1][1].copy() if self._q else self._active.copy()
