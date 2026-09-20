"""Intrinsics for the two real cameras, and the Isaac Lab configs that match them.

The sim is only as good as its cameras. Two things fix them to the real hardware:

1. MEASURED aspect behaviour. Capturing the same static scene at 4:3 and 16:9 from
   both cameras and template-matching one inside the other shows, at 0.987 (C270)
   and 0.998 (KWC-500) correlation, that the 4:3 frame is the centred middle 75% of
   the 16:9 frame's width at an identical vertical FOV. Both sensors are natively
   16:9 and crop horizontally for 4:3 — so 640x480, the resolution this repo records
   and infers at (CAM_WIDTH/CAM_HEIGHT in .env), is NARROWER than the sensor, not a
   taller version of it. Getting this backwards puts ~11 degrees of extra horizontal
   view into the sim for the C270 and ~15 for the KWC-500.

   Reproduce with:  ./robot sim-camera-check

2. SPEC diagonal FOV, as the absolute anchor: 55 deg for the Logitech C270, 80 deg
   for the Klip Xtreme KWC-500 FHD (its manual says 78; vendors quote diagonal for
   this class). This is the weak half — a spec sheet, not this unit. Replace it with
   a real calibration when you have a checkerboard:

       ./robot sim-calibrate --camera top
       ./robot sim-calibrate --camera wrist

   which writes fx/fy/cx/cy and distortion into config/cameras.json with
   source="calibrated", and everything downstream picks them up with no code change.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent
CONFIG = HERE / "config" / "cameras.json"

# Native sensor aspect, confirmed by the crop measurement above.
NATIVE_W, NATIVE_H = 16.0, 9.0
# Fraction of the native width a 4:3 mode keeps. 0.75 exactly: (4/3) / (16/9).
CROP_4X3 = 0.75


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    """A real camera, before it is pinned to a capture resolution."""

    name: str
    model: str
    diagonal_fov_deg: float
    device_env: str          # which .env key names its index
    note: str = ""

    @property
    def focal_sensor_units(self) -> float:
        """Focal length with the native sensor width normalised to 16.0."""
        half_diag = math.hypot(NATIVE_W, NATIVE_H) / 2
        return half_diag / math.tan(math.radians(self.diagonal_fov_deg) / 2)

    def half_width(self, width: int, height: int) -> float:
        """Half sensor width used by a capture mode, in the same units."""
        is_4x3 = abs(width / height - 4 / 3) < 1e-3
        return (CROP_4X3 if is_4x3 else 1.0) * NATIVE_W / 2

    def intrinsics(self, width: int, height: int) -> dict:
        """OpenCV intrinsics for one capture mode. Square pixels, centred principal point."""
        f = self.focal_sensor_units
        fx = (width / 2) / (self.half_width(width, height) / f)
        return {
            "width": width,
            "height": height,
            "fx": round(fx, 3),
            "fy": round(fx, 3),
            "cx": round(width / 2, 3),
            "cy": round(height / 2, 3),
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "hfov_deg": round(2 * math.degrees(math.atan(self.half_width(width, height) / f)), 3),
            "vfov_deg": round(2 * math.degrees(math.atan(NATIVE_H / 2 / f)), 3),
            "source": "spec",
            "model": self.model,
            "note": self.note,
        }


# The two cameras this workspace actually has. Names match the observation keys the
# lerobot datasets use (front / grip), so sim and real line up without a rename map.
TOP = CameraSpec(
    name="front",
    model="Logitech C270 HD",
    diagonal_fov_deg=55.0,
    device_env="CAM_FRONT_INDEX",
    note="overhead, looking down at the mat",
)
WRIST = CameraSpec(
    name="grip",
    model="Klip Xtreme KWC-500 FHD (Laguham)",
    diagonal_fov_deg=80.0,
    device_env="CAM_GRIP_INDEX",
    note="in klip_support-1, bolted to the SO-ARM101 wrist holes",
)
CAMERAS = {c.name: c for c in (TOP, WRIST)}


def default_config(width: int = 640, height: int = 480) -> dict:
    return {
        "_comment": "Regenerate spec values with: python -m sim_agent101.cameras --write. "
                    "Calibrated values are written by scripts/sim/calibrate_cameras.py and "
                    "must not be overwritten by --write (it refuses).",
        "cameras": {name: cam.intrinsics(width, height) for name, cam in CAMERAS.items()},
    }


def load(width: int = 640, height: int = 480) -> dict:
    """Intrinsics for both cameras, calibrated if available, spec otherwise."""
    if CONFIG.exists():
        cfg = json.loads(CONFIG.read_text())["cameras"]
        wrong = [n for n, c in cfg.items() if (c["width"], c["height"]) != (width, height)]
        if wrong:
            raise ValueError(
                f"config/cameras.json holds {width}x{height}-mismatched intrinsics for {wrong}; "
                "re-run the calibration at the resolution you intend to use, or delete the file "
                "to fall back to spec values"
            )
        return cfg
    return default_config(width, height)["cameras"]


def load_scaled(width: int, height: int) -> dict:
    """Intrinsics rescaled to a different render size, at the SAME field of view.

    load() refuses any size but the calibrated one, and it is right to: an intrinsic
    matrix is only meaningful with the image it was measured on, and quietly reusing
    one at another size is how the sim ends up with a different lens than the robot.

    A pure downscale is the exception. Halve both dimensions and every pixel quantity
    -- fx, fy, cx, cy -- halves with them, and the field of view is untouched. That is
    what an RL env wants: the real optics at 64x48, because a policy does not need
    640x480 and the rollout buffer very much notices.

    The ASPECT has to match. Squaring off a 4:3 lens is not a rescale, it either
    crops the width or stretches the picture, and this module exists because that
    exact mistake put 11 degrees of extra view into the sim once already.
    """
    native = load()
    out = {}
    for name, k in native.items():
        sx, sy = width / k["width"], height / k["height"]
        if abs(sx - sy) > 1e-6:
            raise ValueError(
                f"{name}: {width}x{height} is not {k['width']}x{k['height']} rescaled "
                f"(x by {sx:.4f}, y by {sy:.4f}). Same aspect only -- changing it changes the lens."
            )
        out[name] = dict(k, width=width, height=height,
                         fx=k["fx"] * sx, fy=k["fy"] * sy, cx=k["cx"] * sx, cy=k["cy"] * sy)
    return out


def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Print or write the spec-derived camera intrinsics.")
    p.add_argument("--write", action="store_true", help="write config/cameras.json")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    cfg = default_config(args.width, args.height)
    if args.write:
        if CONFIG.exists():
            existing = json.loads(CONFIG.read_text())["cameras"]
            calibrated = [n for n, c in existing.items() if c.get("source") == "calibrated"]
            if calibrated:
                raise SystemExit(
                    f"refusing to overwrite calibrated intrinsics for {calibrated}; "
                    f"delete {CONFIG} by hand if that is really what you want"
                )
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"wrote {CONFIG}")
    for name, c in cfg["cameras"].items():
        print(f"{name:6} {c['model']:35} {c['width']}x{c['height']}  "
              f"fx=fy={c['fx']:.1f}px  hFOV={c['hfov_deg']:.1f}  vFOV={c['vfov_deg']:.1f}  [{c['source']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
