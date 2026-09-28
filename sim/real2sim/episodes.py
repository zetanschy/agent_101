"""The recorded episodes: joints in LeRobot units, frames as memmaps, use flags.

    eps = real2sim.episodes.load()          # {episode index: Episode}, used ones only
    ep = eps[1]
    ep.state, ep.action                     # (T, 6) float32, deg x5 + gripper %
    ep.frames("grip")                       # (T, 480, 640, 3) uint8 memmap slice
    ep.t                                    # (T,) s, frame_index / fps (synthetic)

Needs `./robot real2sim core extract` first (episodes.npz + frames/<cam>.npy). The use
flags and exclusion reasons come from the scene config, not from the data: episode 3
of testingt_real2sim is 47 idle frames with action == state and is excluded there.

Time: row k of an episode is frame_index k, t = k / fps. The recorder reads state[k]
BEFORE it reads the leader for action[k], so state[k] can reflect action[k-1] at the
earliest (joint_units report §1). The follower trails the action by 4.5-4.9 frames.
The front camera often repeats frames (effective 15 fps in episode 2): frames are
what the camera DELIVERED at that tick, not fresh exposures (episodes report).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass

import numpy as np

from . import paths


@functools.lru_cache(maxsize=None)
def npz(ds: str | None = None) -> dict:
    """episodes.npz as a dict of arrays (schema in extract.py); meta parsed to a dict."""
    p = paths.episodes_npz(ds)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run ./robot real2sim core extract")
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


@functools.lru_cache(maxsize=None)
def frames(ds: str | None = None, cam: str = "front") -> np.ndarray:
    """All decoded frames of one camera, (N, H, W, 3) uint8, memory-mapped read-only.
    Row i is parquet row i. Slice with an Episode's [start:stop]."""
    p = paths.frames_npy(ds, cam)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run ./robot real2sim core extract")
    return np.load(p, mmap_mode="r")


@dataclass(frozen=True, eq=False)
class Episode:
    ds: str
    index: int
    start: int  # global row, inclusive
    stop: int  # global row, exclusive
    fps: int
    action: np.ndarray  # (T, 6) lerobot units
    state: np.ndarray  # (T, 6) lerobot units
    use: bool
    reason: str

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self)) / self.fps

    def frames(self, cam: str = "front") -> np.ndarray:
        return frames(self.ds, cam)[self.start:self.stop]

    def config(self) -> dict:
        from .scene import load

        return load(self.ds).episode(self.index)


def load(ds: str | None = None, include_excluded: bool = False) -> dict[int, Episode]:
    """{episode index: Episode}. Excluded episodes (scene use=false) only on request."""
    from .scene import load as load_scene

    ds = paths.dataset(ds)
    z, sc = npz(ds), load_scene(ds)
    out = {}
    for e, a, b in zip(z["episodes"], z["ep_from"], z["ep_to"]):
        cfg = sc["episodes"].get(str(int(e)), {"use": True, "reason": "not in the scene config"})
        if not (cfg["use"] or include_excluded):
            continue
        out[int(e)] = Episode(ds, int(e), int(a), int(b), int(z["fps"]), z["action"][a:b], z["state"][a:b],
                              bool(cfg["use"]), cfg.get("reason", ""))
    return out


def fps(ds: str | None = None) -> int:
    return int(npz(ds)["fps"])
