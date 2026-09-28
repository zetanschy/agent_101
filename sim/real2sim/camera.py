"""The OpenCV camera model in numpy, and the same camera as MuJoCo, Blender and USD see it.

CONVENTIONS (every real2sim camera, every engine adapter):
    axes     OpenCV: x right, y down, z forward (the optical axis).
    pixels   (u, v) with (0, 0) the CENTRE of the top-left pixel, u right, v down.
             The image spans u in [-0.5, W - 0.5], and its centre is ((W-1)/2, (H-1)/2).
    K        (fx, fy, cx, cy) in pixels.
    dist     (k1, k2, p1, p2, k3), OpenCV's 5-coefficient model:
               r2 = x^2 + y^2,  rad = 1 + k1 r2 + k2 r2^2 + k3 r2^3
               xd = x rad + 2 p1 x y + p2 (r2 + 2 x^2)
               yd = y rad + p1 (r2 + 2 y^2) + 2 p2 x y
               u = fx xd + cx,  v = fy yd + cy
    pose     T_parent_cam, a 4x4 taking camera-frame points into the PARENT frame:
             'world' (= the URDF base frame, the real2sim world) or a link name from
             kinematics.LINKS (the wrist camera rides on 'gripper'). A camera on a link
             needs that link's pose, T_world_parent, at the frame being projected.

RENDERERS ARE PINHOLES. None of MuJoCo, Blender or Isaac 4.5 renders OpenCV distortion,
and Omniverse renders square pixels only. So a distorted camera is rendered as an
OVERSIZED PINHOLE that covers every ray the real image contains (`render_spec`) and
then warped into the real camera's pixels (`remap_maps` + `remap`):

    spec = render_spec(cam)                       # W' x H' pinhole, same pose
    img  = <engine renders spec.camera(cam)>       # to_mujoco / to_blender / to_usd
    real_like = remap(img, remap_maps(cam, spec))  # W x H, distorted like the camera

fx != fy (the wrist camera: 680.7 vs 655.9) is handled the same way: render_spec(cam,
square=True) renders at fx' = fy' = max(fx, fy), so no axis is undersampled, and the
remap applies the real anisotropic K. The wrist camera's barrel distortion needs a
~762 x 576 pinhole to cover its 640 x 480 image (cameras report §4).

MUJOCO's principal point convention is VERIFIED by a render test
(tests/test_camera.py::test_mujoco_projection, spheres must land within 0.5 px of
Camera.project): principalpixel = ((W-1)/2 - cx, (H-1)/2 - cy), i.e. BOTH components
are the image centre minus the principal point. mjlab's klip_camera.py uses
(cx - W/2, -(cy - H/2)), which mirrors x (25 px on the wrist camera).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .transforms import inv_T, make_T, mat_to_quat

# OpenCV camera axes <-> the OpenGL-style axes of MuJoCo, Blender and USD cameras
# (x right, y UP, looking down -z). Its own inverse: R_gl = R_cv @ CV_TO_GL.
CV_TO_GL = np.diag([1.0, -1.0, -1.0])
ZERO_DIST = (0.0, 0.0, 0.0, 0.0, 0.0)


def opencv_to_opengl(T) -> np.ndarray:
    """T_parent_cam with OpenCV camera axes -> the same camera with OpenGL axes."""
    T = np.array(T, dtype=float)
    T[:3, :3] = T[:3, :3] @ CV_TO_GL
    return T


opengl_to_opencv = opencv_to_opengl  # the flip is an involution


# --- distortion ------------------------------------------------------------------

def distort(xn, dist) -> np.ndarray:
    """Normalized pinhole coords (..., 2) -> distorted normalized coords."""
    k1, k2, p1, p2, k3 = dist
    xn = np.asarray(xn, dtype=float)
    x, y = xn[..., 0], xn[..., 1]
    r2 = x * x + y * y
    rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return np.stack([x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                     y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y], -1)


def undistort(xd, dist, iters: int = 30, tol: float = 1e-12) -> np.ndarray:
    """Distorted normalized coords (..., 2) -> pinhole normalized coords.

    Newton's method on the full 5-coefficient model with its analytic Jacobian, from
    x0 = xd. Converges to ~1e-15 in < 10 steps over both cameras' images. A point
    where the model folds over (Jacobian determinant <= 0: the polynomial is not
    invertible there, which happens outside the calibrated field) or that does not
    converge comes back NaN rather than as a plausible wrong answer.
    """
    k1, k2, p1, p2, k3 = dist
    xd = np.asarray(xd, dtype=float)
    if not any(dist):
        return xd.copy()
    x = xd.copy()
    for _ in range(iters):
        u, v = x[..., 0], x[..., 1]
        r2 = u * u + v * v
        rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dr = 2 * k1 + r2 * (4 * k2 + 6 * k3 * r2)  # d(rad)/dx = dr * x
        f = distort(x, dist) - xd
        a = rad + dr * u * u + 2 * p1 * v + 6 * p2 * u
        b = dr * u * v + 2 * p1 * u + 2 * p2 * v
        d = rad + dr * v * v + 6 * p1 * v + 2 * p2 * u
        det = a * d - b * b
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.stack([(d * f[..., 0] - b * f[..., 1]) / det, (a * f[..., 1] - b * f[..., 0]) / det], -1)
        x = x - np.clip(step, -0.5, 0.5)
        if np.nanmax(np.abs(step), initial=0.0) < tol:
            break
    res = np.abs(distort(x, dist) - xd).max(-1)
    u, v = x[..., 0], x[..., 1]
    r2 = u * u + v * v
    rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    dr = 2 * k1 + r2 * (4 * k2 + 6 * k3 * r2)
    det = (rad + dr * u * u + 2 * p1 * v + 6 * p2 * u) * (rad + dr * v * v + 6 * p1 * v + 2 * p2 * u) \
        - (dr * u * v + 2 * p1 * u + 2 * p2 * v) ** 2
    bad = ~(res < 1e-9) | ~(det > 0)
    x[bad] = np.nan
    return x


# --- the camera --------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Camera:
    """One calibrated camera. Immutable: use .replace(...) to derive variants."""

    name: str
    width: int
    height: int
    K: tuple  # (fx, fy, cx, cy)
    dist: tuple = ZERO_DIST
    parent: str = "world"
    T_parent_cam: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        object.__setattr__(self, "K", tuple(float(k) for k in self.K))
        object.__setattr__(self, "dist", tuple(float(k) for k in self.dist))
        object.__setattr__(self, "T_parent_cam", np.array(self.T_parent_cam, dtype=float).reshape(4, 4))
        if len(self.K) != 4 or len(self.dist) != 5:
            raise ValueError(f"{self.name}: K needs 4 values, dist 5")

    fx = property(lambda s: s.K[0])
    fy = property(lambda s: s.K[1])
    cx = property(lambda s: s.K[2])
    cy = property(lambda s: s.K[3])

    @property
    def Kmat(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def distorted(self) -> bool:
        return any(self.dist)

    def replace(self, **kw) -> "Camera":
        return replace(self, **kw)

    # frames
    def T_world_cam(self, T_world_parent=None) -> np.ndarray:
        """Camera pose in the world (URDF base) frame, OpenCV axes."""
        if self.parent == "world":
            if T_world_parent is not None and not np.allclose(T_world_parent, np.eye(4)):
                raise ValueError(f"{self.name} is fixed in the world; T_world_parent must be None")
            return self.T_parent_cam.copy()
        if T_world_parent is None:
            raise ValueError(f"{self.name} rides on '{self.parent}': pass T_world_parent (that link's pose)")
        return np.asarray(T_world_parent, dtype=float) @ self.T_parent_cam

    def world_to_cam(self, P, T_world_parent=None) -> np.ndarray:
        T = inv_T(self.T_world_cam(T_world_parent))
        return np.asarray(P, dtype=float) @ T[:3, :3].T + T[:3, 3]

    # projection
    def project_cam(self, Pc, distort_: bool = True) -> np.ndarray:
        """Camera-frame points (..., 3) -> pixels (..., 2); NaN at or behind the camera."""
        Pc = np.asarray(Pc, dtype=float)
        z = Pc[..., 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = np.where(z > 1e-9, Pc[..., :2] / z, np.nan)
        return self.denormalize(xn, distort_)

    def project(self, P_world, T_world_parent=None, distort_: bool = True) -> np.ndarray:
        """World points (..., 3) -> pixels (..., 2), distorted unless distort_=False."""
        return self.project_cam(self.world_to_cam(P_world, T_world_parent), distort_)

    def denormalize(self, xn, distort_: bool = True) -> np.ndarray:
        xn = np.asarray(xn, dtype=float)
        xd = distort(xn, self.dist) if (distort_ and self.distorted) else xn
        return np.stack([self.fx * xd[..., 0] + self.cx, self.fy * xd[..., 1] + self.cy], -1)

    def normalize(self, uv, undistort_: bool = True) -> np.ndarray:
        """Pixels (..., 2) -> pinhole normalized coords (x/z, y/z)."""
        uv = np.asarray(uv, dtype=float)
        xd = np.stack([(uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy], -1)
        return undistort(xd, self.dist) if (undistort_ and self.distorted) else xd

    def undistort_pixels(self, uv) -> np.ndarray:
        """Distorted pixels -> where the same rays land in the undistorted (pinhole) image."""
        return self.denormalize(self.normalize(uv), distort_=False)

    def distort_pixels(self, uv_pinhole) -> np.ndarray:
        return self.denormalize(self.normalize(uv_pinhole, undistort_=False), distort_=True)

    def pixel_to_ray(self, uv, T_world_parent=None) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (..., 2) -> (camera centre (3,), unit ray directions (..., 3)) in the world."""
        xn = self.normalize(uv)
        d = np.concatenate([xn, np.ones(xn.shape[:-1] + (1,))], -1)
        T = self.T_world_cam(T_world_parent)
        d = d @ T[:3, :3].T
        return T[:3, 3].copy(), d / np.linalg.norm(d, axis=-1, keepdims=True)

    def unproject_to_plane(self, uv, z: float = 0.0, T_world_parent=None, normal=(0.0, 0.0, 1.0),
                           point=None) -> np.ndarray:
        """Pixels -> world points on a plane (default: horizontal at height z). NaN if parallel/behind."""
        o, d = self.pixel_to_ray(uv, T_world_parent)
        n = np.asarray(normal, dtype=float)
        p0 = np.asarray(point, dtype=float) if point is not None else np.array([0.0, 0.0, z])
        with np.errstate(divide="ignore", invalid="ignore"):
            s = ((p0 - o) @ n) / (d @ n)
        s = np.where(s > 0, s, np.nan)
        return o + s[..., None] * d

    # config round trip
    def to_config(self, sources: dict | None = None) -> dict:
        """The scene-config camera group (see scene.py); sources: {'intrinsics': ..., ...}."""
        s = sources or {}
        return {
            "source": s.get("camera", "real2sim.camera.Camera.to_config"), "width": self.width, "height": self.height,
            "intrinsics": {"source": s.get("intrinsics", "unspecified"), "K": list(self.K)},
            "distortion": {"source": s.get("distortion", "unspecified"), "dist": list(self.dist)},
            "pose": {"source": s.get("pose", "unspecified"), "parent": self.parent,
                     "T_parent_cam": self.T_parent_cam.round(9).tolist()},
        }

    @classmethod
    def from_config(cls, name: str, g: dict) -> "Camera":
        return cls(name, int(g["width"]), int(g["height"]), g["intrinsics"]["K"], g["distortion"]["dist"],
                   g["pose"]["parent"], g["pose"]["T_parent_cam"])


# --- rendering through a pinhole ----------------------------------------------------

@dataclass(frozen=True)
class RenderSpec:
    """An undistorted pinhole camera to render, big enough to feed a remap to `cam`."""

    width: int
    height: int
    K: tuple  # (fx', fy', cx', cy')

    def camera(self, cam: Camera) -> Camera:
        """`cam`'s pose with this pinhole's intrinsics and no distortion: what engines render."""
        return cam.replace(width=self.width, height=self.height, K=self.K, dist=ZERO_DIST)


def render_spec(cam: Camera, margin: float = 2.0, square: bool = False, oversample: float = 1.0) -> RenderSpec:
    """The smallest pinhole (same pose, focal scaled by `oversample`) covering every ray
    of `cam`'s distorted image plus `margin` pixels for bilinear sampling.

    square=True forces fx' = fy' = max(fx, fy): Omniverse cannot render non-square pixels.
    """
    W, H = cam.width, cam.height
    e = np.linspace(-0.5, W - 0.5, 4 * W + 1), np.linspace(-0.5, H - 0.5, 4 * H + 1)
    border = np.concatenate([np.stack([e[0], np.full_like(e[0], -0.5)], -1),
                             np.stack([e[0], np.full_like(e[0], H - 0.5)], -1),
                             np.stack([np.full_like(e[1], -0.5), e[1]], -1),
                             np.stack([np.full_like(e[1], W - 0.5), e[1]], -1)])
    gu, gv = np.meshgrid(np.linspace(-0.5, W - 0.5, 33), np.linspace(-0.5, H - 0.5, 25))
    pts = np.concatenate([border, np.stack([gu.ravel(), gv.ravel()], -1)])
    xn = cam.normalize(pts)
    if np.isnan(xn).any():
        raise ValueError(f"{cam.name}: distortion is not invertible inside the image; check dist {cam.dist}")
    fx, fy = (max(cam.fx, cam.fy),) * 2 if square else (cam.fx, cam.fy)
    fx, fy = fx * oversample, fy * oversample
    (x0, y0), (x1, y1) = xn.min(0), xn.max(0)
    return RenderSpec(int(math.ceil(fx * (x1 - x0) + 2 * margin + 1)), int(math.ceil(fy * (y1 - y0) + 2 * margin + 1)),
                      (fx, fy, margin - fx * x0, margin - fy * y0))


def _grid(W, H):
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    return np.stack([u, v], -1)


def remap_maps(cam: Camera, spec: RenderSpec) -> tuple[np.ndarray, np.ndarray]:
    """(map_x, map_y), each (H, W) float32: where each pixel of `cam`'s distorted image
    samples the pinhole render, i.e. distorted = bilinear_sample(render, map_x, map_y)."""
    xn = cam.normalize(_grid(cam.width, cam.height))
    fx, fy, cx, cy = spec.K
    return (fx * xn[..., 0] + cx).astype(np.float32), (fy * xn[..., 1] + cy).astype(np.float32)


def undistort_maps(cam: Camera, spec: RenderSpec) -> tuple[np.ndarray, np.ndarray]:
    """The inverse warp: (map_x, map_y), each (H', W'), sampling a REAL image so it
    becomes the pinhole view `spec` (for comparing real frames against a raw render)."""
    fx, fy, cx, cy = spec.K
    g = _grid(spec.width, spec.height)
    uv = cam.denormalize(np.stack([(g[..., 0] - cx) / fx, (g[..., 1] - cy) / fy], -1))
    return uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)


def bilinear_sample(img, map_x, map_y, fill=0) -> np.ndarray:
    """Pure-numpy cv2.remap(INTER_LINEAR, BORDER_CONSTANT): pixel centres at integers."""
    img = np.asarray(img)
    H, W = img.shape[:2]
    x, y = np.asarray(map_x, dtype=np.float64), np.asarray(map_y, dtype=np.float64)
    ok = (x > -1) & (x < W) & (y > -1) & (y < H)
    x0, y0 = np.floor(np.nan_to_num(x, nan=-2)), np.floor(np.nan_to_num(y, nan=-2))
    wx, wy = x - x0, y - y0
    out = np.zeros(x.shape + img.shape[2:], dtype=np.float64)
    for dy, dx, w in ((0, 0, (1 - wx) * (1 - wy)), (0, 1, wx * (1 - wy)), (1, 0, (1 - wx) * wy), (1, 1, wx * wy)):
        xi, yi = (x0 + dx).astype(int), (y0 + dy).astype(int)
        inside = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        v = np.where(inside[(...,) + (None,) * (img.ndim - 2)],
                     img[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)], fill)
        out += np.nan_to_num(w)[(...,) + (None,) * (img.ndim - 2)] * v
    out[~ok] = fill
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.rint(out), np.iinfo(img.dtype).min, np.iinfo(img.dtype).max).astype(img.dtype)
    return out.astype(img.dtype)


def remap(img, maps, fill=0, use_cv2: bool = True) -> np.ndarray:
    """Apply (map_x, map_y). cv2.remap when importable (fast), else bilinear_sample.
    cv2 rounds sample positions to 1/32 px, so the two agree to <= 1 grey level on
    smooth images and to about (local gradient)/32 on noise."""
    if use_cv2:
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None and np.asarray(img).dtype in (np.uint8, np.uint16, np.float32):
            return cv2.remap(np.ascontiguousarray(img), maps[0], maps[1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return bilinear_sample(img, maps[0], maps[1], fill)


# --- engine cameras -------------------------------------------------------------------

def to_mujoco(cam: Camera, pixel_pitch: float = 3e-6) -> dict:
    """MuJoCo <camera> attributes in the parent body's frame. Pinhole only: pass
    render_spec(...).camera(cam) for a distorted camera. Render at (width, height)
    = `resolution`; the offscreen buffer must be at least that big
    (<visual><global offwidth offheight>). sensorsize is arbitrary (3 um pixels): only
    its ratio to the focal length matters."""
    if cam.distorted:
        raise ValueError(f"{cam.name} is distorted; render render_spec(cam).camera(cam) and remap")
    T = opencv_to_opengl(cam.T_parent_cam)
    W, H = cam.width, cam.height
    return {
        "parent": cam.parent, "pos": T[:3, 3].tolist(), "quat": mat_to_quat(T[:3, :3]).tolist(),
        "resolution": [W, H], "sensorsize": [W * pixel_pitch, H * pixel_pitch],
        "focalpixel": [cam.fx, cam.fy],
        "principalpixel": [(W - 1) / 2 - cam.cx, (H - 1) / 2 - cam.cy],  # VERIFIED, see module doc
    }


def mujoco_camera_xml(cam: Camera, name: str | None = None) -> str:
    a = to_mujoco(cam)
    f = lambda v: " ".join(f"{x:.10g}" for x in v)  # noqa: E731
    return (f'<camera name="{name or cam.name}" pos="{f(a["pos"])}" quat="{f(a["quat"])}" '
            f'resolution="{f(a["resolution"])}" sensorsize="{f(a["sensorsize"])}" '
            f'focalpixel="{f(a["focalpixel"])}" principalpixel="{f(a["principalpixel"])}"/>')


def to_blender(cam: Camera, sensor_width_mm: float = 36.0) -> dict:
    """bpy camera data + render settings (BlenderProc's CameraUtility formulas, READ;
    sensor_fit forced HORIZONTAL so the AUTO rule cannot switch axes). fx != fy
    becomes a pixel aspect ratio. Set the camera OBJECT's matrix (relative to its
    parent) to T_parent_cam_gl: Blender cameras look down -z with +y up."""
    if cam.distorted:
        raise ValueError(f"{cam.name} is distorted; render render_spec(cam).camera(cam) and remap")
    W, H = cam.width, cam.height
    ax, ay = (1.0, cam.fx / cam.fy) if cam.fx >= cam.fy else (cam.fy / cam.fx, 1.0)
    return {
        "parent": cam.parent, "T_parent_cam_gl": opencv_to_opengl(cam.T_parent_cam).tolist(),
        "lens": cam.fx * sensor_width_mm / W, "sensor_width": sensor_width_mm, "sensor_fit": "HORIZONTAL",
        "shift_x": -(cam.cx - (W - 1) / 2) / W, "shift_y": (cam.cy - (H - 1) / 2) / W * (ay / ax),
        "resolution_x": W, "resolution_y": H, "pixel_aspect_x": ax, "pixel_aspect_y": ay,
    }


def to_usd(cam: Camera, horizontal_aperture: float = 20.955) -> dict:
    """UsdGeom.Camera attributes (focal length and apertures share one unit; only the
    ratios matter). Offsets use the sign measured in sim/sim_agent101/assets/objects.py:
    a positive horizontal offset moves content left (cx falls), positive vertical moves
    it down (cy rises). That code centres on W/2; this uses the pixel-centre (W-1)/2,
    a 0.5 px difference that ISAAC should confirm with a point-projection render.
    Omniverse renders square pixels and derives the vertical aperture from the
    horizontal one, so fx must equal fy: pass render_spec(cam, square=True).camera(cam)."""
    if cam.distorted:
        raise ValueError(f"{cam.name} is distorted; render render_spec(cam, square=True).camera(cam) and remap")
    if not math.isclose(cam.fx, cam.fy, rel_tol=1e-9):
        raise ValueError(f"{cam.name}: fx {cam.fx} != fy {cam.fy}; Omniverse renders square pixels only")
    W, H = cam.width, cam.height
    ha = horizontal_aperture
    va = ha * H / W
    return {
        "parent": cam.parent, "T_parent_cam_gl": opencv_to_opengl(cam.T_parent_cam).tolist(),
        "focalLength": cam.fx * ha / W, "horizontalAperture": ha, "verticalAperture": va,
        "horizontalApertureOffset": ha * ((W - 1) / 2 - cam.cx) / W,
        "verticalApertureOffset": va * (cam.cy - (H - 1) / 2) / H,
        "resolution": [W, H],
    }


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """T_world_cam (OpenCV axes) of a camera at `eye` looking at `target`; for tests and tools."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    return make_T(np.stack([x, np.cross(z, x), z], 1), eye)
