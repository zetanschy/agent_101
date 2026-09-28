"""Convex collision pieces vs the core's meshes; exact containment depth; mass sums."""
import numpy as np

from real2sim import objects
from real2sim.isaac import geometry as G


def test_cap_pieces_match_the_core_mesh(scene):
    cd = scene.cap_dims()
    pieces = G.cap_pieces(cd)
    mesh = objects.cap_mesh(cd)
    assert len(pieces) == 17
    # chords thicken the 1.2 mm skirt by <= the sagitta: a few % of volume, never less
    v = G.pieces_volume(pieces)
    assert mesh.volume <= v <= mesh.volume * 1.10
    assert G.sagitta(cd["diameter"] / 2 - cd["wall"], 16) < 3e-4
    # the pieces never reach outside the real solid's bounding cylinder
    V = np.concatenate(pieces)
    assert np.hypot(V[:, 0], V[:, 1]).max() <= cd["diameter"] / 2 + 1e-9
    assert V[:, 2].min() >= -1e-12 and V[:, 2].max() <= cd["height"] + 1e-12


def test_mug_pieces_match_the_core_mesh(scene):
    md = scene.mug_dims()
    pieces = G.mug_pieces(md)
    parts = objects.mug_parts(md)
    body = G.pieces_volume(pieces[:33])
    assert parts["body"].volume <= body <= parts["body"].volume * 1.05
    assert len(pieces) > 33  # the handle pieces


def test_revolved_depth_is_exact(scene):
    cd = scene.cap_dims()
    R, h, t = cd["diameter"] / 2, cd["height"], cd["top"]
    pts = np.array([[R - 0.0006, 0, 0.005],  # inside the skirt wall, 0.6 mm from the outer face
                    [R + 0.001, 0, 0.005],  # outside
                    [0, 0, h - 0.0005],  # in the top disk, 0.5 mm below the top
                    [0, 0, 0.005],  # in the open cavity: not solid
                    [0, R - 0.0006, 0.0003]])  # 0.3 mm above the rim bottom, any azimuth
    d = G.revolved_depth(pts, G.cap_profile(cd)) * 1000
    assert np.allclose(d, [0.6, 0.0, 0.5, 0.0, 0.3], atol=1e-6)


def test_pieces_and_profile_agree(scene):
    cd = scene.cap_dims()
    rng = np.random.default_rng(0)
    R, h = cd["diameter"] / 2, cd["height"]
    X = rng.uniform([-R, -R, 0], [R, R, h], (50000, 3))
    a = G.pieces_contain(X, G.cap_pieces(cd))
    b = G.revolved_depth(X, G.cap_profile(cd)) > 0
    assert (b & ~a).mean() < 0.002  # the solid is covered (facet sagitta only)
    assert (a & ~b).mean() < 0.03  # the chords add a little wall


def test_combine_mass_parallel_axis():
    m, c, I = G.combine_mass([(1.0, [0, 0, 0], np.zeros((3, 3))), (1.0, [1.0, 0, 0], np.zeros((3, 3)))])
    assert m == 2.0 and np.allclose(c, [0.5, 0, 0])
    assert np.allclose(I, np.diag([0.0, 0.5, 0.5]))
    w, R = G.principal(np.diag([3.0, 1.0, 2.0]))
    assert np.allclose(np.sort(w), [1, 2, 3]) and np.isclose(np.linalg.det(R), 1.0)
