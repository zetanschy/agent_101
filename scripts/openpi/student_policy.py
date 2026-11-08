"""The mjlab BC student on a lerobot robot: the real arm (webui/student_worker.py, the
Eval page) or the live twin (sim_eval.py). Torch on the CPU, cv2 and numpy only.

The student comes as the TorchScript export of thirdparty/mjlab's
scripts/so101_twin/student.py (a .ts file), and everything it was trained against
travels inside it as contract.json, so nothing here is calibrated twice:

  * FRAMES: the env rendered undistorted pinholes at the calibrated K. A robot frame is
    undistorted with the same K and distortion, then shrunk to the student's input size.
  * JOINTS: lerobot values (degrees x5, gripper %) -> URDF radians by the twin's units.
  * ACTION: the env's IntegratedJointPositionAction, step for step: target += clip(scale
    * action), clamped to the joint (and Wrist_Roll) limits; the target goes out as the
    lerobot goal. A zero action holds the target. The target starts where the arm is.

One call per 30 Hz frame, ~2 ms on the CPU: the student is a closed loop, no chunks.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch


class StudentPolicy:
    def __init__(self, path: str):
        extra = {"contract.json": ""}
        self.net = torch.jit.load(path, map_location="cpu", _extra_files=extra).eval()
        c = self.contract = json.loads(extra["contract.json"])
        self.joints = c["joints_lerobot"]
        self.scale = np.asarray(c["scale"])
        self.clip = np.asarray(c["clip"])
        self.limits = np.asarray(c["limits"])
        u = c["units"]
        self.offsets = np.asarray(u["offsets_deg"])
        self.grip_a, self.grip_b = u["grip_a_deg"], u["grip_b_deg_per_pct"]
        self.maps = {}
        for cam, k in c["cameras"].items():
            fx, fy, cx, cy = k["K"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            self.maps[cam] = (
                (k["width"], k["height"]),
                tuple(k["size"]),
                cv2.initUndistortRectifyMap(K, np.asarray(k["dist"]), None, K,
                                            (k["width"], k["height"]), cv2.CV_16SC2),
            )
        self.reset()

    def reset(self) -> None:
        self.target = None

    def to_urdf(self, v: np.ndarray) -> np.ndarray:
        return np.radians(np.r_[v[:5] + self.offsets, self.grip_a + self.grip_b * v[5]])

    def to_lerobot(self, q: np.ndarray) -> np.ndarray:
        d = np.degrees(q)
        return np.r_[d[:5] - self.offsets, (d[5] - self.grip_a) / self.grip_b]

    def frame(self, cam: str, img: np.ndarray) -> np.ndarray:
        """An RGB robot frame as the env rendered it: undistorted, at the input size."""
        full, size, (m1, m2) = self.maps[cam]
        if img.shape[1::-1] != full:
            img = cv2.resize(img, full, interpolation=cv2.INTER_AREA)
        # Replicated edges, not black: the front lens's strong k2 leaves curved borders
        # that undistortion cannot fill, where the env saw more mat.
        img = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def __call__(self, obs: dict) -> dict:
        """A lerobot observation -> the lerobot action to send this frame."""
        q = self.to_urdf(np.array([obs[f"{j}.pos"] for j in self.joints], dtype=float))
        if self.target is None:
            self.target = q.copy()
        x = [torch.from_numpy(self.frame(c, obs[c]))[None] for c in ("front", "grip")]
        with torch.no_grad():
            a = self.net(*x, torch.tensor(q, dtype=torch.float32)[None])[0].numpy()
        step = np.clip(a * self.scale, self.clip[:, 0], self.clip[:, 1])
        self.target = np.clip(self.target + step, self.limits[:, 0], self.limits[:, 1])
        return {f"{j}.pos": float(v) for j, v in zip(self.joints, self.to_lerobot(self.target))}


if __name__ == "__main__":  # self-check: python scripts/openpi/student_policy.py <.ts>
    import sys

    p = StudentPolicy(sys.argv[1])
    rng = np.random.default_rng(0)
    q = rng.uniform(p.limits[:, 0], p.limits[:, 1])
    assert np.allclose(p.to_urdf(p.to_lerobot(q)), q), "units do not round-trip"
    obs = {f"{j}.pos": float(v) for j, v in zip(p.joints, p.to_lerobot(q))}
    obs |= {c: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in ("front", "grip")}
    for _ in range(100):
        act = p(obs)
    t = p.to_urdf(np.array([act[f"{j}.pos"] for j in p.joints]))
    assert np.all(t >= p.limits[:, 0] - 1e-9) and np.all(t <= p.limits[:, 1] + 1e-9), t
    assert np.all(np.abs(t - q) <= 100 * np.abs(p.clip).max(1) + 1e-9)
    print("ok", {k: round(v, 2) for k, v in act.items()})
