"""Colour: sRGB <-> linear, CIE Lab, and the colour classes measured on this dataset's frames.

TRANSFER. Both cameras are UVC webcams (Logitech C270, Klip Xtreme KWC-500). Their
frames went MJPG -> BGR (OpenCV) -> AV1 yuv420p (lerobot) -> rgb24 (pyav, the core
extraction), and that YUV round trip is symmetric. The camera's own tone curve was
never measured: that needs bracketed exposures of a known target, and the dataset has
none. It is ASSUMED to be the sRGB curve (IEC 61966-2-1), which webcams target, and
every _lin value in LOOK is that decode.

COLOUR CLASSES (HSV in OpenCV's 8-bit convention: H 0-179, S and V 0-255). All MEASURED
on the decoded frames (build-phase probe, 5th / 95th percentiles):
    teal cap   front  closed tops H 85-94, S 83-112, V 151-175; open side H 84-95,
                      S 100-183, V 95-174. The glare corner (x > 525, y < 100) is
                      H 98-103, S 61-115, V 61-152, so the hue ceiling of 96 rather
                      than the episodes report's 100 keeps it out.
               grip   held cap H 81-95, S 100-236, V 56-171 (5 frames, 23-25 k px each)
    red enamel front  H <= 9 or >= 179 at S 113-255, V 33-129 (the shaded crescent)
               grip   H 0-8 / 179, S 147-255, V 40-105
    white PLA  max(R,G,B) >= 105 (front) / 95 (grip) and saturation <= 0.35. These
               are CALIB's measured edges (calib/masks.py): the arm's median maxc is
               170 and the mat's 24.
"""

from __future__ import annotations

import numpy as np

# (lo, hi) HSV bounds per camera, and the minimum blob area (px) after a 3x3 opening.
TEAL_HSV = {"front": ((76, 70, 60), (96, 255, 255)), "grip": ((76, 90, 45), (100, 255, 255))}
TEAL_MIN_AREA = {"front": 30, "grip": 150}  # front caps are 450-600 px; grip caps 20-25 k
RED_HSV = (((0, 110, 30), (10, 255, 255)), ((170, 110, 30), (180, 255, 255)))
WHITE_PLA = {"front": (105, 0.35), "grip": (95, 0.35)}  # (max(R,G,B) >=, saturation <=)

# sRGB -> CIE XYZ (D65), IEC 61966-2-1, and the D65 white point.
_M_RGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = _M_RGB_XYZ.sum(1)
LUMA = _M_RGB_XYZ[1]  # linear-light luminance weights (Y row)


def srgb_to_linear(x) -> np.ndarray:
    """8-bit (integer dtype) or 0..1 float sRGB -> linear 0..1 float32."""
    x = np.asarray(x)
    x = x.astype(np.float32) / 255.0 if np.issubdtype(x.dtype, np.integer) else x.astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x, u8: bool = True) -> np.ndarray:
    """Linear -> sRGB, clipped to [0, 1]; u8=True returns rounded uint8."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.rint(y * 255).astype(np.uint8) if u8 else y.astype(np.float32)


def linear_to_lab(rgb_lin) -> np.ndarray:
    """Linear sRGB (..., 3) -> CIE L*a*b* (D65)."""
    xyz = np.asarray(rgb_lin, dtype=np.float64) @ _M_RGB_XYZ.T / _WHITE_D65
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e76(lab_a, lab_b) -> np.ndarray:
    """CIE76 colour difference (Euclidean in Lab); about 2.3 is a just-noticeable difference."""
    return np.linalg.norm(np.asarray(lab_a) - np.asarray(lab_b), axis=-1)


def luminance(rgb_lin) -> np.ndarray:
    return np.asarray(rgb_lin, dtype=np.float32) @ LUMA.astype(np.float32)


def hsv(img) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV)


def _open_filter(m, min_area: int) -> np.ndarray:
    import cv2

    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def teal(img, cam: str, h=None) -> np.ndarray:
    """Teal-cap pixels (bool), blobs smaller than TEAL_MIN_AREA dropped."""
    import cv2

    lo, hi = TEAL_HSV[cam]
    m = cv2.inRange(hsv(img) if h is None else h, lo, hi) > 0
    return _open_filter(m, TEAL_MIN_AREA[cam])


def red(img, h=None) -> np.ndarray:
    import cv2

    h = hsv(img) if h is None else h
    m = (cv2.inRange(h, *RED_HSV[0]) > 0) | (cv2.inRange(h, *RED_HSV[1]) > 0)
    return _open_filter(m, 20)


def white_pla(img, cam: str) -> np.ndarray:
    im = np.asarray(img, dtype=np.int16)
    mx, mn = im.max(2), im.min(2)
    vmin, smax = WHITE_PLA[cam]
    return (mx >= vmin) & ((mx - mn) <= smax * np.maximum(mx, 1))


# --- the wrist camera relative to the front camera ------------------------------------
#
# The two webcams differ by more than a white balance. Measured on surfaces both see
# (materials.grip_colour_model), after each camera's white is made the same: the wrist
# camera renders the teal cap, the red enamel and the yellow-green marks all MORE
# saturated than the C270 does. The model is a per-frame white balance, taken from
# the fixed finger that is always in view (white PLA). Then one saturation factor
# sigma about the grey axis, in white-normalised linear units (white = 1, 1, 1).
# Then a per-frame exposure s, from the mat's luminance against the front texture.
# Saturation preserves luminance, so s is independent of sigma.
#
#     front_lin = L_PLA * sat(s * wb * grip_lin / L_PLA, 1 / sigma)
#     grip_lin  = L_PLA * sat(front_lin / L_PLA, sigma) / (s * wb)
#     sat(a, k) = y(a) + k (a - y(a)),   y = LUMA . a

def saturate(a, k: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    y = (a @ LUMA)[..., None]
    return y + k * (a - y)


def grip_white_balance(img, core, L_pla, lin=None) -> np.ndarray:
    """Per-channel gains taking this wrist frame's white (the static finger core, white
    PLA pixels, median) to the front reference white L_PLA's chromaticity (G fixed at 1).
    `lin`: the frame's srgb_to_linear, if already computed."""
    lin = srgb_to_linear(img) if lin is None else lin
    m = core & white_pla(img, "grip")
    if m.sum() < 500:
        return None
    jaw = np.median(lin[m], 0)
    L = np.asarray(L_pla, float)
    return ((L / L[1]) / (jaw / jaw[1])).astype(np.float32)


def grip_to_front(grip_lin, wb, s: float, sigma: float, L_pla) -> np.ndarray:
    L = np.asarray(L_pla, float)
    return (L * saturate(s * np.asarray(wb) * np.asarray(grip_lin, float) / L, 1.0 / sigma)).astype(np.float32)


def front_to_grip(front_lin, wb, s: float, sigma: float, L_pla) -> np.ndarray:
    L = np.asarray(L_pla, float)
    return (L * saturate(np.asarray(front_lin, float) / L, sigma) / (s * np.asarray(wb))).astype(np.float32)
