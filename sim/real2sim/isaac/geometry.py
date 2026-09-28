"""Collision geometry of the cap and the mug for PhysX, and the offline contact checks
that read it back. Pure numpy (+ scipy, trimesh lazily): runs in the Isaac process and
on the host, so the penetration numbers in eval.json are computed on EXACTLY the
convex pieces PhysX collided with.

WHY EXPLICIT CONVEX PIECES. Both objects are hollow, and PhysX gives a dynamic body
only three concave options (Isaac report §6, tooling report "Isaac details"):
    convexHull           fills the cavity: the hull mug put a lid on itself (the cap
                         came to rest ON it at z = 0.096, MEASURED by the tooling run)
    convexDecomposition  V-HACD on a voxel grid; the understand-phase cup worked with
                         the defaults, but a 1.2 mm cap skirt is below any affordable
                         voxel size, and the result is not reproducible across cookers
    sdf                  against the SDF jaws, an SDF cap knocked the cup over at 240 Hz
                         (MEASURED, tooling report); a 2.5 g SDF-vs-SDF pair is the
                         least stable contact PhysX has
Both shapes are a revolved profile, so they decompose EXACTLY into convex prisms: a
disk plus annular sectors. Each sector's outer face follows the circle with `facets`
flat facets and its inner face is the chord between the sector's end points, which
keeps it convex. The chord sits inside the real wall: the collision wall is never
thinner than the real one and at most `sagitta` thicker, 0.26 mm for the cap
(16 sectors, R_i 13.3 mm) and 0.19 mm for the mug (32 sectors, R_i 38.5 mm), both
far below the 1 mm penetration budget. The handle is the core's swept tube
(objects.mug_parts), cut into pieces by the angle along its arc.

Frames and conventions are objects.py's: the cap origin is the centre of the rim
plane with +z towards the closed top; the mug origin is the centre of the bottom
face with +z up and the handle along +x.
"""

from __future__ import annotations

import functools

import numpy as np

from ..transforms import quat_to_mat


def _ring(radius: float, angles, z: float) -> np.ndarray:
    return np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, z)], -1)


def _prism(bottom: np.ndarray, top_z: float) -> np.ndarray:
    top = bottom.copy()
    top[:, 2] = top_z
    return np.concatenate([bottom, top])


def _sectors(r_in: float, r_out: float, z0: float, z1: float, sectors: int, facets: int, phase: float = 0.0):
    """Annular sectors between r_in and r_out, z0..z1: outer arc in `facets` facets,
    inner face the chord (so each piece is convex)."""
    out = []
    for i in range(sectors):
        a0 = phase + 2 * np.pi * i / sectors
        a1 = phase + 2 * np.pi * (i + 1) / sectors
        outer = _ring(r_out, np.linspace(a0, a1, facets + 1), z0)
        inner = _ring(r_in, np.array([a0, a1]), z0)
        out.append(_prism(np.concatenate([outer, inner]), z1))
    return out


def sagitta(r_in: float, sectors: int) -> float:
    """How much thicker than the real wall the chord makes a sector at its middle (m)."""
    return float(r_in * (1 - np.cos(np.pi / sectors)))


def cap_pieces(dims: dict, sectors: int = 16, facets: int = 4) -> list[np.ndarray]:
    """Convex pieces of the hollow cap (objects.cap_mesh's shape): the top disk
    (h - top .. h) and `sectors` skirt sectors (0 .. h - top). Vertex arrays, cap frame."""
    R, h, w, t = dims["diameter"] / 2, dims["height"], dims["wall"], dims["top"]
    disk = _prism(_ring(R, np.linspace(0, 2 * np.pi, sectors * facets, endpoint=False), h - t), h)
    return [disk] + _sectors(R - w, R, 0.0, h - t, sectors, facets)


def mug_pieces(dims: dict, sectors: int = 32, facets: int = 3, handle_pieces: int = 8,
               handle_overlap_deg: float = 3.0) -> list[np.ndarray]:
    """Convex pieces of the mug: the floor disk (0 .. floor), `sectors` wall sectors
    (floor .. height) and `handle_pieces` pieces of the core's handle tube, cut by the
    angle around (outer radius, handle mid-height) in the x-z plane, each widened by
    `handle_overlap_deg` so that neighbouring pieces overlap instead of leaving slits."""
    from ..objects import mug_parts

    Ro, H, w, F = dims["outer_diameter"] / 2, dims["height"], dims["wall"], dims["floor"]
    floor = _prism(_ring(Ro, np.linspace(0, 2 * np.pi, sectors * facets, endpoint=False), 0.0), F)
    pieces = [floor] + _sectors(Ro - w, Ro, F, H, sectors, facets)
    hd = dims["handle"]
    V = np.asarray(mug_parts(dims)["handle"].vertices)
    zc = (hd["z_top"] + hd["z_bottom"]) / 2
    ang = np.degrees(np.arctan2(V[:, 2] - zc, V[:, 0] - Ro))
    edges = np.linspace(ang.min(), ang.max(), handle_pieces + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        sel = V[(ang >= a - handle_overlap_deg) & (ang <= b + handle_overlap_deg)]
        if len(sel) >= 4:
            pieces.append(sel)
    return pieces


# --- volumes and containment depth ---------------------------------------------------

@functools.lru_cache(maxsize=64)
def _hull_cached(key: bytes, shape: tuple):
    from scipy.spatial import ConvexHull

    return ConvexHull(np.frombuffer(key, dtype=np.float64).reshape(shape))


def hull(vertices: np.ndarray):
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    return _hull_cached(v.tobytes(), v.shape)


def pieces_volume(pieces) -> float:
    return float(sum(hull(p).volume for p in pieces))


def cap_profile(dims: dict) -> np.ndarray:
    """(r, z) outline of the cap's revolved solid, counter-clockwise, the same polygon
    objects.cap_mesh revolves. The segment on the axis (r = 0) is not a surface."""
    R, h, w, t = dims["diameter"] / 2, dims["height"], dims["wall"], dims["top"]
    return np.array([(R - w, 0), (R, 0), (R, h), (0, h), (0, h - t), (R - w, h - t)], dtype=float)


def mug_body_profile(dims: dict) -> np.ndarray:
    """(r, z) outline of the mug body (objects.mug_parts' body), counter-clockwise."""
    Ro, H, w, F = dims["outer_diameter"] / 2, dims["height"], dims["wall"], dims["floor"]
    return np.array([(0, 0), (Ro, 0), (Ro, H), (Ro - w, H), (Ro - w, F), (0, F)], dtype=float)


def revolved_depth(points_local: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """(N,) depth (m) of points inside the solid of revolution of `profile` about z
    (object frame), 0 outside: the exact distance to the nearest SURFACE segment.

    Exact, seam-free, and cheap (a 2-D polygon distance per point). The convex pieces
    PhysX collides with differ from this solid by the chord sagitta only (< 0.26 mm,
    module doc), so penetration measured here is the physical overlap to that
    accuracy. Segments lying on the axis are skipped: they are not a surface."""
    P = np.asarray(points_local, dtype=float).reshape(-1, 3)
    q = np.stack([np.hypot(P[:, 0], P[:, 1]), P[:, 2]], -1)
    poly = np.asarray(profile, dtype=float)
    a, b = poly, np.roll(poly, -1, axis=0)
    inside = np.zeros(len(q), dtype=bool)
    dist = np.full(len(q), np.inf)
    for (ax, az), (bx, bz) in zip(a, b):
        # even-odd rule on the closed polygon (axis segments included for the test)
        cond = (az > q[:, 1]) != (bz > q[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = ax + (q[:, 1] - az) * (bx - ax) / (bz - az)
        inside ^= cond & (q[:, 0] < xint)
        if ax == 0.0 and bx == 0.0:
            continue  # on the axis: not a surface
        d = np.array([bx - ax, bz - az])
        s = np.clip(((q - [ax, az]) @ d) / (d @ d), 0.0, 1.0)
        dist = np.minimum(dist, np.linalg.norm(q - ([ax, az] + s[:, None] * d), axis=1))
    return np.where(inside, dist, 0.0)


def pieces_contain(points: np.ndarray, pieces, tol: float = 1e-9) -> np.ndarray:
    """(N,) True where a point is inside (or on) at least one convex piece."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    out = np.zeros(len(P), dtype=bool)
    for piece in pieces:
        eq = hull(piece).equations
        out |= (P @ eq[:, :3].T + eq[:, 3]).max(1) <= tol
    return out


def transform(points: np.ndarray, pos, quat) -> np.ndarray:
    """Points given in a body frame -> world, for a body at (pos, quat wxyz)."""
    return np.asarray(points, dtype=float) @ quat_to_mat(quat).T + np.asarray(pos, dtype=float)


def to_local(points: np.ndarray, pos, quat) -> np.ndarray:
    """World points -> the body frame of a body at (pos, quat wxyz)."""
    return (np.asarray(points, dtype=float) - np.asarray(pos, dtype=float)) @ quat_to_mat(quat)


def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """n area-weighted random points on a triangle mesh."""
    rng = np.random.default_rng(seed)
    tri = np.asarray(verts, dtype=float)[np.asarray(faces)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(tri), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return ((1 - s)[:, None] * tri[idx, 0] + (s * (1 - r2))[:, None] * tri[idx, 1] + (s * r2)[:, None] * tri[idx, 2])


# --- mass properties of rigid parts ------------------------------------------------------

def combine_mass(parts) -> tuple[float, np.ndarray, np.ndarray]:
    """Combine rigid parts [(mass, com (3,), inertia 3x3 about that com, in the common
    frame)] into (mass, com, inertia about the combined com), parallel-axis theorem."""
    m = sum(p[0] for p in parts)
    c = sum(p[0] * np.asarray(p[1], dtype=float) for p in parts) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in parts:
        d = np.asarray(ci, dtype=float) - c
        I += np.asarray(Ii, dtype=float) + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return float(m), c, I


def principal(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inertia tensor -> (diagonal (3,), rotation R whose columns are the principal axes,
    det +1): I = R diag R^T, the form USD's MassAPI (diagonalInertia, principalAxes) wants."""
    w, R = np.linalg.eigh(np.asarray(I, dtype=float))
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    return w, R


def box_inertia(mass: float, size) -> np.ndarray:
    x, y, z = size
    return mass / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


# --- the table top ---------------------------------------------------------------------

def table_tilt(scene) -> tuple[float, float]:
    """(dz/dx, dz/dy) of the table top when the scene carries one (CALIB fits
    table.tilt), else (0, 0). real2sim.objects places objects at table_z() only, so the
    Isaac scene adds the local height for a tilted table (scene.object_placements)."""
    t = scene["table"].get("tilt")
    return (float(t[0]), float(t[1])) if t else (0.0, 0.0)


def table_top_at(scene, x, y):
    """Table top height (m, base frame) at x, y (scalars or arrays)."""
    sx, sy = table_tilt(scene)
    return scene.table_z() + sx * np.asarray(x) + sy * np.asarray(y)
