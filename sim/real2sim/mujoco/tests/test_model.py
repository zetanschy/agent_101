"""The MuJoCo scene: geometry, limits, masses, and the camera convention."""
import math

import numpy as np
import pytest

from real2sim import episodes, kinematics
from real2sim.mujoco import model as mdl
from real2sim.units import JAW_TOUCH_DEG


def test_builds_with_the_episode(built, scene):
    m = built.model
    assert len(built.finger_geoms["fixed"]) >= 30 and len(built.finger_geoms["moving"]) >= 15  # CoACD pieces
    names = [o["name"] for o in built.objects]
    assert names == [f"cap_{c['id']}" for c in scene.episode(0)["caps"]] + ["mug"]
    assert m.opt.timestep == pytest.approx(1 / (30 * built.physics.substeps))


def test_joint_ranges_hold_every_recorded_state(built, scene):
    u = scene.units()
    rng = built.model.jnt_range[[built.model.joint(n).id for n in kinematics.URDF_JOINTS]]
    for ep in episodes.load(include_excluded=True).values():
        q = u.to_urdf(np.concatenate([ep.state, ep.action]))
        assert (q[:, :5] >= rng[:5, 0]).all() and (q[:, :5] <= rng[:5, 1]).all()
    assert math.degrees(rng[5, 0]) == pytest.approx(JAW_TOUCH_DEG)


def test_mount_and_webcam_mass(built):
    m = built.model
    assert m.body("klip_support").mass[0] == pytest.approx(mdl.MOUNT_MASS)
    assert m.body("kwc500").mass[0] == pytest.approx(mdl.WEBCAM_MASS)


def test_cap_inertia_is_the_hollow_cap(built, scene):
    m = built.model
    b = m.body("cap_A")
    d = scene.cap_dims()
    assert b.mass[0] == pytest.approx(d["mass"])
    assert 0.5 * d["height"] < b.ipos[2] < d["height"]  # COM toward the closed top (origin at the rim)


def test_mug_cavity(built, scene):
    """Every wall box's inner face sits at the inner radius (the cap falls in, not on a lid)."""
    m = built.model
    d = scene.mug_dims()
    Ri = d["outer_diameter"] / 2 - d["wall"]
    walls = [g for g in range(m.ngeom) if m.geom(g).name.startswith("mug_wall")]
    assert len(walls) == built.physics.mug_walls
    for g in walls:
        r = np.hypot(*m.geom_pos[g][:2])
        assert r - m.geom_size[g][0] == pytest.approx(Ri, abs=1e-6)


def _seg_centroids(b, data, cam, n):
    import mujoco

    spec = b.cameras[cam]["spec"]
    r = mujoco.Renderer(b.model, spec.height, spec.width)
    r.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[mdl.MARKER_GROUP] = 1  # markers only: nothing can occlude them
    r.update_scene(data, camera=cam, scene_option=opt)
    seg = r.render()[..., 0].copy()
    r.close()
    out = []
    for i in range(n):
        ys, xs = np.nonzero(seg == b.model.geom(f"marker{i}").id)
        out.append((xs.mean(), ys.mean()) if len(xs) > 15 else (np.nan, np.nan))
    return np.array(out)


@pytest.mark.parametrize("cam", ["front", "grip"])
def test_cameras_render_where_camera_projects(scene, servo, cam):
    """real2sim.camera.to_mujoco inside this model: marker spheres (world for the front
    camera, on `gripper` for the wrist camera at a recorded pose) land where
    Camera.project puts them on the pinhole render, within 0.5 px."""
    import mujoco

    c = scene.camera(cam)
    pin = mdl.render_spec(c).camera(c)
    q = scene.units().to_urdf(episodes.load()[0].state[230])
    kin = kinematics.default()
    Tp = None if c.parent == "world" else kin.link_T(q, c.parent)
    rng = np.random.default_rng(1)
    uv = rng.uniform([40, 40], [pin.width - 40, pin.height - 40], (10, 2))
    o, dirs = pin.pixel_to_ray(uv, Tp)
    P = o + dirs * rng.uniform(0.15 if cam == "grip" else 0.8, 0.3 if cam == "grip" else 1.0, (10, 1))
    parent = "world" if c.parent == "world" else c.parent
    local = P if Tp is None else (P - Tp[:3, 3]) @ Tp[:3, :3]
    b = mdl.build(scene, None, servo, objects=False, markers=[(parent, p, 0.003 if cam == "grip" else 0.006) for p in local])
    d = mujoco.MjData(b.model)
    mdl.set_arm_state(b, d, q)
    mujoco.mj_forward(b.model, d)
    got = _seg_centroids(b, d, cam, len(P))
    err = np.linalg.norm(got - pin.project(P, Tp), axis=1)
    # a sphere's image centroid sits off its projected centre by ~(r/z)^2 x (distance
    # from the principal point); 3-6 mm spheres at 0.15-1 m: < 0.2 px typical
    assert np.isfinite(err).sum() >= 8 and np.nanmedian(err) < 0.2 and np.nanmax(err) < 1.0, err


def test_table_tilt_and_overrides(scene, servo):
    """--set table.tilt tilts the table box and lifts objects onto the local height."""
    from real2sim.mujoco.evaluate import apply_overrides

    sc = apply_overrides(scene, ["table.tilt=[0.02, -0.01]"])
    assert sc.overrides["table.tilt"]["now"] == [0.02, -0.01] and sc.hash != scene.hash
    b = mdl.build(sc, 0, servo, cameras=())
    cap = b.obj("cap_A")
    x, y = cap["pos0"][:2]
    assert cap["pos0"][2] == pytest.approx(sc.table_z() + 0.02 * x - 0.01 * y)  # closed-up: origin on the table
    q = b.model.geom_quat[b.table_geom]
    from real2sim.transforms import quat_to_mat

    normal = quat_to_mat(q)[:, 2]
    assert normal @ np.array([-0.02, 0.01, 1.0]) / np.linalg.norm([-0.02, 0.01, 1.0]) == pytest.approx(1.0, abs=1e-9)


def test_grasp_placement_puts_the_cap_between_the_fingers(scene, servo):
    """placement='grasp': at the real blocked frame both fingers are within 1 mm of the cap."""
    import mujoco

    b = mdl.build(scene, 0, servo, cameras=(), placement="grasp")
    m, d = b.model, mujoco.MjData(b.model)
    pk = scene.episode(0)["picks"][0]
    q = scene.units().to_urdf(episodes.load()[0].state[pk["close"][1]])
    mdl.set_arm_state(b, d, q)
    mujoco.mj_forward(m, d)
    cg = b.obj("cap_A")["geoms"][0]
    fr = np.zeros(6)
    for side in ("fixed", "moving"):
        dist = min(mujoco.mj_geomDistance(m, d, cg, g, 0.05, fr) for g in b.finger_geoms[side])
        assert dist < 1e-3, (side, dist)
