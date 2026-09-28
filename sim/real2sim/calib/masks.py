"""Real-image segmentation for the calibration: white PLA, teal caps, red mug, steel.

Every threshold here was MEASURED on this dataset's frames (host cv2 4.11, the
decoded rgb24 caches) and is only as general as that lighting: one overhead key
light from the top right, a black glossy mat, AV1 at crf 30.

WHITE PLA (the printed arm and fingers). maxc = max(R, G, B), sat = (maxc - minc) / maxc.
    front  arm pixels (the fit6r model silhouette eroded 4 px, 3 episodes x 13
           frames): maxc 5/25/50 % = 97 / 170 / 199, sat median 0.09, p90 0.16.
           Arm-free background: maxc median 24, p99 127 (glare corner, mug steel,
           cables, fluorescent marks), 2.5 % of it above 100.
           -> maxc >= 105 and sat <= 0.35: the 50 % edge between ~200 and ~25.
    grip   fingers maxc 150-185 (the stepped inner faces ~120-150 in shade),
           mat 7-30, blur edge ~4 px wide -> maxc >= 95 and sat <= 0.35.
TEAL (caps). HSV(78-100, S >= 75, V >= 80) in the front view and V >= 60 in the
    wrist view: the episodes report's masks, which found every cap in every
    episode; cap top HSV(89, 88, 160). The overhead glare corner (x > 525,
    y < 100) throws false teal in episodes 1-3, which observations.py drops by
    shape, not by position.
RED (mug enamel). H <= 10 or >= 170, S >= 120, V >= 35 (episodes report; the red
    in shade is HSV(5, 240, 66)).

Everything returns numpy arrays in the camera's pixel grid, (0, 0) = centre of the
top-left pixel (camera.py convention). `bilinear` is the exact lookup the least
squares uses: cv2.remap rounds positions to 1/32 px, which would flatten the
finite-difference Jacobian below that step.
"""

from __future__ import annotations

import numpy as np

WHITE = {"front": (105, 0.35), "grip": (95, 0.35)}  # (maxc >=, sat <=), measured above
TEAL_HSV = {"front": ((78, 75, 80), (100, 255, 255)), "grip": ((78, 75, 60), (100, 255, 255))}
RED_HSV = (((0, 120, 35), (10, 255, 255)), ((170, 120, 35), (180, 255, 255)))


def _cv2():
    import cv2

    return cv2


def white(img, cam: str) -> np.ndarray:
    """White-PLA mask, opened 3x3 so single noisy pixels and 1-px cable glints go."""
    cv2 = _cv2()
    im = np.asarray(img, dtype=np.int16)
    mx, mn = im.max(2), im.min(2)
    vmin, smax = WHITE[cam]
    m = (mx >= vmin) & ((mx - mn) <= smax * np.maximum(mx, 1))
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def teal(img, cam: str) -> np.ndarray:
    cv2 = _cv2()
    lo, hi = TEAL_HSV[cam]
    m = cv2.inRange(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV), lo, hi)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def red(img) -> np.ndarray:
    cv2 = _cv2()
    hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, *RED_HSV[0]) | cv2.inRange(hsv, *RED_HSV[1])
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def sdf(mask) -> np.ndarray:
    """Signed distance to the mask boundary in px, negative inside, float32.

    Distances are measured between pixel centres, so the boundary sits at +-0.5:
    a pixel just outside reads +1, just inside -1 -> shift by 0.5 so the zero level
    is the edge between them."""
    cv2 = _cv2()
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.full(m.shape, 1e3, np.float32)
    if m.all():
        return np.full(m.shape, -1e3, np.float32)
    out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    inside = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(m, -(inside - 0.5), out - 0.5).astype(np.float32)


def outline(mask, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Boundary pixels of a mask (inside pixels with a 4-neighbour outside) as (N, 2)
    (u, v) floats, and outward unit normals from the smoothed mask gradient."""
    cv2 = _cv2()
    m = np.asarray(mask, dtype=np.uint8)
    edge = m & ~cv2.erode(m, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
    v, u = np.nonzero(edge)
    if step > 1:
        u, v = u[::step], v[::step]
    sm = cv2.GaussianBlur(m.astype(np.float32), (7, 7), 1.5)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)[v, u]
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)[v, u]
    n = -np.stack([gx, gy], 1)  # the mask falls off outward
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return np.stack([u, v], 1).astype(float), n


def bilinear(img, uv, fill: float = np.nan) -> np.ndarray:
    """img (H, W) sampled at float pixel positions uv (..., 2); `fill` outside."""
    img = np.asarray(img)
    H, W = img.shape[:2]
    u, v = np.asarray(uv, dtype=float)[..., 0], np.asarray(uv, dtype=float)[..., 1]
    ok = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    uu, vv = np.clip(np.nan_to_num(u), 0, W - 1.000001), np.clip(np.nan_to_num(v), 0, H - 1.000001)
    x0, y0 = np.floor(uu).astype(int), np.floor(vv).astype(int)
    wx, wy = uu - x0, vv - y0
    val = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x0 + 1] * wx * (1 - wy)
           + img[y0 + 1, x0] * (1 - wx) * wy + img[y0 + 1, x0 + 1] * wx * wy)
    return np.where(ok, val, fill)


def bilinear_stack(stack, f, uv, fill: float = np.nan) -> np.ndarray:
    """stack (F, H, W) sampled at (f[i], uv[i]): one gather for many frames."""
    F, H, W = stack.shape
    u, v = np.asarray(uv, dtype=float)[..., 0], np.asarray(uv, dtype=float)[..., 1]
    ok = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    uu, vv = np.clip(np.nan_to_num(u), 0, W - 1.000001), np.clip(np.nan_to_num(v), 0, H - 1.000001)
    x0, y0 = np.floor(uu).astype(int), np.floor(vv).astype(int)
    wx, wy = uu - x0, vv - y0
    val = (stack[f, y0, x0] * (1 - wx) * (1 - wy) + stack[f, y0, x0 + 1] * wx * (1 - wy)
           + stack[f, y0 + 1, x0] * (1 - wx) * wy + stack[f, y0 + 1, x0 + 1] * wx * wy)
    return np.where(ok, val, fill)


def components(mask, min_area: int = 1):
    """[(area, centroid (u, v), bbox (x, y, w, h), label mask)] of the 8-connected
    components, largest first."""
    cv2 = _cv2()
    n, lab, st, c = cv2.connectedComponentsWithStats(np.asarray(mask, np.uint8), connectivity=8)
    out = [(int(st[i, 4]), c[i].astype(float), tuple(int(x) for x in st[i, :4]), lab == i)
           for i in range(1, n) if st[i, 4] >= min_area]
    return sorted(out, key=lambda t: -t[0])
