"""The two webcams as a post-process: a linear pinhole render in, the camera's 8-bit frame out.

Renderers are pinholes that output scene radiance (look.json scene units: total
horizontal irradiance at the table = 1). Everything the CAMERA does happens here, on
the host, identically for any renderer (Blender, Isaac RTX, MuJoCo):

    1. lens     remap the oversized pinhole into the real distorted 640 x 480 pixels
                (camera.remap_maps of the scene camera; bilinear, cv2)
    2. optics   a Gaussian blur of sigma_px (lens softness + the AV1 codec's
                smoothing), fitted per camera (see OPTICS below)
    3. colour   LOOK's camera models (look.json cameras; look/colour.py):
                  front  lin = pi L_PLA / rho_PLA * R * g_frame
                         g_frame = the frame's white-balance gain against LOOK's
                         reference state (plates.npz frame_gain, per row; 'frame'), or
                         the episode's mean gain ('episode'), or 1 ('reference')
                  grip   lin = L_PLA * sat(pi R / rho_PLA, sigma) / (s * wb)
                         wb = that frame's white balance (the static finger core,
                         LOOK), s = the exposure, sigma = 1.14 (LOOK)
    4. encode   clip to [0, 1], sRGB transfer (ASSUMED, colour.py), round to uint8

WRIST EXPOSURE (MEASURED here, build-phase probe on LOOK's 24 samples). LOOK's
per-frame exposure s comes from the mat's luminance against the front texture. It
spans 0.28-2.20, but the real fixed finger's luminance stays at 0.30-0.58 across the
same frames, and the low values all fall on carry / over-mug / release frames, where
the wrist camera sees the glossy mat at grazing angles (Fresnel reflection of the
room, and the white desk): the mat reference is view-dependent, so s is biased
exactly where the view changes most. Modes:
    'look'      LOOK's per-frame s (as published)
    'constant'  one exposure for the whole dataset (GRIP_EXPOSURE below, with its fit)
'constant' is the default: at the three rest samples LOOK's s is 1.35-1.89, where s = 1
already leaves the wrist background within +0.002 linear, and at the carry / over-mug /
release samples it scatters over 0.39-2.33 (samples/index.json grip.colour_model).

The per-frame camera states (front g_frame, wrist wb / s) are the real cameras' own
auto-white-balance and auto-exposure at that row: global per-frame numbers measured
on the real frames by LOOK, never pixels. They replay the camera's state the same
way the arm replays the recorded joints, so they only apply to renders aligned with a
real episode (frame index = dataset row); the manifest records which were used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .. import paths
from ..camera import RenderSpec, remap_maps
from ..look import colour

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # cv2 reads EXR only when this is set before import

# WRIST EXPOSURE 'constant': s = 1, the front reference state. Checked, not tuned: on the
# 24 samples (EEVEE, fitted scene) the wrist fingers' linear bias is +0.008 on ~0.5,
# i.e. s = 1.017 would null it.
GRIP_EXPOSURE = {"value": 1.0, "source": "measured: wrist fingers bias +0.008 linear at s = 1 (24 samples)"}
# OPTICS: the webcams' softness (lens, focus, the AV1 codec) as one Gaussian in linear
# light. Fitted on episode 0's 8 samples, 'all' region PSNR over sigma (EEVEE renders,
# fitted scene 6f282e812cc585f5) and checked on episodes 1-2 (16 samples):
#   front  0.8 px   ep0 23.05 -> 23.19 dB at 0.9-1.2 (SSIM 0.884 -> 0.881);
#                   ep1-2 22.52 -> 22.67 dB (the arm: +0.3 dB)
#   grip   2.5 px   ep0 19.51 -> 19.87 dB, flat to 3.0; ep1-2 18.48 -> 18.88 dB
#                   (SSIM 0.713 -> 0.739). The real fingers, 2-8 cm from the lens,
#                   want more (still rising at 4.5 px): defocus is depth-dependent and a
#                   single Gaussian is its first-order stand-in.
OPTICS = {"front": {"sigma_px": 0.8, "source": "fitted: ep0 samples, all-region PSNR (camera_model OPTICS)"},
          "grip": {"sigma_px": 2.5, "source": "fitted: ep0 samples, all-region PSNR (camera_model OPTICS)"}}


def read_exr(path) -> np.ndarray:
    """(H, W, 3) float32 linear RGB from a Blender EXR."""
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im[..., :3].astype(np.float32), cv2.COLOR_BGR2RGB)


@dataclass
class CameraState:
    """Per-row camera states of one dataset (LOOK's measurements)."""

    frame_gain: np.ndarray  # (N, 3) front white-balance gain per dataset row
    gain_episode: dict  # {episode: (3,)}
    grip_frames: dict  # {(episode, frame): {"white_balance", "exposure"}}
    samples: dict  # {(episode, frame): grip colour_model of LOOK's samples}

    @classmethod
    def load(cls, ds=None) -> "CameraState":
        d = paths.out_dir(ds, "look")
        z = np.load(d / "plates" / "plates.npz")
        ge = {int(e): z["gain_episode"][i] for i, e in enumerate(z["episodes"])}
        wm = json.loads((d / "table" / "wrist_model.json").read_text())
        gf = {tuple(map(int, k.split(":"))): v for k, v in wm["frames"].items()}
        idx = json.loads((d / "samples" / "index.json").read_text())
        ss = {(int(s["episode"]), int(s["frame"])): s["grip"]["colour_model"] for s in idx["samples"]}
        return cls(z["frame_gain"].astype(np.float32), ge, gf, ss)

    def grip(self, episode: int, frame: int) -> dict:
        """The wrist camera's (white_balance, exposure) at a frame: LOOK's sample entry
        when it is a sample, else the nearest frame of LOOK's wrist model."""
        if (episode, frame) in self.samples and self.samples[(episode, frame)]["white_balance"] is not None:
            return self.samples[(episode, frame)]
        ks = [k for (e, k) in self.grip_frames if e == episode]
        k = min(ks, key=lambda x: abs(x - frame))
        return self.grip_frames[(episode, k)]


class Response:
    """One camera's lens + colour, for pinhole renders of `spec` (job.json cameras.<cam>.pinhole)."""

    def __init__(self, scene, look: dict, cam: str, spec: RenderSpec, state: CameraState | None = None,
                 wb: str = "frame", exposure: str = "constant", sigma_px: float | None = None,
                 grip_exposure: float | None = None):
        self.cam, self.scene = cam, scene
        self.camera = scene.camera(cam)
        self.maps = remap_maps(self.camera, spec)
        self.L = np.asarray(look["cameras"]["front"]["white_reference"]["L_PLA_lin"], np.float32)
        self.rho = float(look["cameras"]["front"]["white_reference"]["rho_PLA"])
        self.sigma = float(look["cameras"]["grip"]["sigma"])
        self.state = state
        self.wb_mode, self.exp_mode = wb, exposure
        self.sigma_px = OPTICS[cam]["sigma_px"] if sigma_px is None else float(sigma_px)
        self.grip_exposure = GRIP_EXPOSURE["value"] if grip_exposure is None else float(grip_exposure)

    def describe(self) -> dict:
        d = {"lens": "camera.remap_maps (scene camera: K, dist)", "blur_sigma_px": self.sigma_px,
             "transfer": "sRGB (assumed, look/colour.py)"}
        if self.cam == "front":
            d["colour"] = {"model": "pi L_PLA / rho * R * g", "white_balance": self.wb_mode}
        else:
            d["colour"] = {"model": "L_PLA * sat(pi R / rho, sigma) / (s wb)", "sigma": self.sigma,
                           "white_balance": "per frame (LOOK)" if self.wb_mode != "reference" else "none",
                           "exposure": self.exp_mode, "exposure_value": self.grip_exposure
                           if self.exp_mode == "constant" else "per frame (LOOK)"}
        return d

    def lens(self, pin_lin) -> np.ndarray:
        import cv2

        img = cv2.remap(np.ascontiguousarray(pin_lin, np.float32), self.maps[0], self.maps[1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        if self.sigma_px > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.sigma_px, borderType=cv2.BORDER_REFLECT)
        return img

    def colour(self, radiance, episode: int, frame: int, row: int) -> np.ndarray:
        """Scene radiance (H, W, 3) -> camera-linear (H, W, 3), unclipped."""
        a = np.pi * np.asarray(radiance, np.float32) / self.rho  # white-normalised: lit white PLA = 1
        if self.cam == "front":
            g = np.ones(3, np.float32)
            if self.wb_mode == "frame":
                g = self.state.frame_gain[row]
            elif self.wb_mode == "episode":
                g = np.asarray(self.state.gain_episode[episode], np.float32)
            return a * self.L * g
        st = self.state.grip(episode, frame) if self.state is not None else None
        wb = np.asarray(st["white_balance"], np.float32) if (st and self.wb_mode != "reference") else \
            np.ones(3, np.float32)
        s = float(st["exposure"]) if (self.exp_mode == "look" and st) else self.grip_exposure
        return (self.L * colour.saturate(a, self.sigma) / (s * wb)).astype(np.float32)

    def __call__(self, pin_lin, episode: int, frame: int, row: int) -> np.ndarray:
        """Pinhole radiance -> the camera's 640 x 480 uint8 sRGB frame."""
        return colour.linear_to_srgb(self.colour(self.lens(pin_lin), episode, frame, row))


def responses(scene, look: dict, job: dict, ds=None, **kw) -> dict:
    """{cam: Response} for a job's cameras, sharing one CameraState."""
    state = CameraState.load(ds or scene.ds)
    out = {}
    for cam, c in job["cameras"].items():
        p = c["pinhole"]
        spec = RenderSpec(int(p["width"]), int(p["height"]), tuple(p["K"]))
        out[cam] = Response(scene, look, cam, spec, state, **kw)
    return out
