"""Camera latency: the joint state at the moment each camera frame was exposed.

lerobot 0.6.1 reads the motors, then takes the LATEST frame each camera thread holds
(joint_units report §1), so an image shows the arm as it was `lag` frames before
its row's observation.state. The calibration fits one lag per camera (params blocks
front.lag / grip.lag, in frames at 30 fps) and every image residual evaluates the arm
at row - lag, linearly interpolated, never across an episode boundary (the arm was
reset off camera between episodes).

MEASURED here because it is not small: the silhouettes were taken on frames where the
arm moves < 8 deg/s (front) / 15 deg/s (wrist approach), and a 3-frame latency at
15 deg/s is 1.5 deg of joint error, ~8 mm at the wrist camera -- tens of pixels on a
cap 10 cm away.
"""

from __future__ import annotations

import functools

import numpy as np

from .. import episodes as E, paths


class Clock:
    def __init__(self, state_all, ep_from, ep_to):
        self.state = np.asarray(state_all, dtype=float)
        n = len(self.state)
        self.lo = np.zeros(n, int)
        self.hi = np.zeros(n, int)
        for a, b in zip(ep_from, ep_to):
            self.lo[a:b], self.hi[a:b] = a, b - 1

    def at(self, rows, lag: float) -> np.ndarray:
        """(len(rows), 6) lerobot state at time (row - lag), clamped to the row's episode."""
        rows = np.asarray(rows, dtype=int)
        t = np.clip(rows - float(lag), self.lo[rows], self.hi[rows])
        i0 = np.floor(t).astype(int)
        i1 = np.minimum(i0 + 1, self.hi[rows])
        w = (t - i0)[:, None]
        return (1 - w) * self.state[i0] + w * self.state[i1]


@functools.lru_cache(maxsize=None)
def clock(ds: str | None = None) -> Clock:
    z = E.npz(paths.dataset(ds))
    return Clock(z["state"], z["ep_from"], z["ep_to"])
