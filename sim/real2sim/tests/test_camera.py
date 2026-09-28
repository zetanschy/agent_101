"""Camera model round trips, the pinhole+remap render path, and the engine conventions."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from real2sim import camera as C
from real2sim import kinematics, paths


def _grid(cam, n=41):
    u, v = np.meshgrid(np.linspace(0, cam.width - 1, n), np.linspace(0, cam.height - 1, n))
    return np.stack([u.ravel(), v.ravel()], -1)


@pytest.mark.parametrize("name", ["front", "grip"])
def test_distort_round_trip(scene, name):
    cam = scene.camera(name)
    uv = _grid(cam)
    back = cam.distort_pixels(cam.undistort_pixels(uv))
    assert np.nanmax(np.abs(back - uv)) < 0.05 and not np.isnan(back).any()


@pytest.mark.parametrize("name", ["front", "grip"])
def test_project_unproject_round_trip(scene, name):
    cam = scene.camera(name)
    Tp = None if cam.parent == "world" else kinematics.default().link_T(np.array(kinematics.HOME), cam.parent)
    uv = _grid(cam, 21)
    z = scene.table_z() if cam.parent == "world" else -0.5  # a plane the wrist camera can see
    P = cam.unproject_to_plane(uv, z=z, T_world_parent=Tp)
    ok = ~np.isnan(P).any(1)
    assert ok.mean() > 0.5
    assert np.abs(cam.project(P[ok], T_world_parent=Tp) - uv[ok]).max() < 0.05


@pytest.mark.parametrize("name,square", [("front", False), ("grip", False), ("grip", True)])
def test_render_spec_and_remap(scene, name, square):
    cam = scene.camera(name)
    spec = C.render_spec(cam, square=square)
    pin = spec.camera(cam)
    mx, my = C.remap_maps(cam, spec)
    assert np.all((mx >= 0) & (mx <= spec.width - 1) & (my >= 0) & (my <= spec.height - 1))
    # a world point: its pinhole pixel, looked up through the map at its distorted pixel
    rng = np.random.default_rng(0)
    uv = rng.uniform([5, 5], [cam.width - 6, cam.height - 6], (200, 2))
    Tp = None if cam.parent == "world" else kinematics.default().link_T(np.array(kinematics.HOME), cam.parent)
    P = cam.unproject_to_plane(uv, z=scene.table_z() if Tp is None else -0.5, T_world_parent=Tp)
    P = P[~np.isnan(P).any(1)]
    uv_d, uv_p = cam.project(P, Tp), pin.project(P, Tp)
    look = np.stack([C.bilinear_sample(mx, uv_d[:, 0], uv_d[:, 1]), C.bilinear_sample(my, uv_d[:, 0], uv_d[:, 1])], 1)
    assert np.abs(look - uv_p).max() < 0.05
    # numpy and cv2 remaps agree on a smooth image (cv2 quantizes positions to 1/32 px)
    yy, xx = np.mgrid[: spec.height, : spec.width]
    img = np.stack([xx * 255.0 / spec.width, yy * 255.0 / spec.height,
                    127 + 100 * np.sin(xx / 17.0) * np.cos(yy / 23.0)], -1).astype(np.uint8)
    a, b = C.remap(img, (mx, my), use_cv2=False), C.remap(img, (mx, my), use_cv2=True)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1
    if name == "grip":  # the wrist lens needs a much bigger pinhole (~762x576)
        assert spec.width > 700 and spec.height > 540


def test_opencv_opengl_flip():
    T = C.look_at([0.1, 0.2, 1.0], [0, 0, 0])
    assert np.allclose(C.opengl_to_opencv(C.opencv_to_opengl(T)), T)
    gl = C.opencv_to_opengl(T)
    assert np.allclose(gl[:3, 2], -T[:3, 2]) and np.allclose(gl[:3, 1], -T[:3, 1])


def _render_spheres(cam, pts, r=0.004):
    import mujoco

    spheres = "".join(f'<geom name="s{i}" type="sphere" size="{r}" pos="{p[0]} {p[1]} {p[2]}"/>' for i, p in enumerate(pts))
    xml = (f'<mujoco><visual><global offwidth="{cam.width}" offheight="{cam.height}"/></visual>'
           f"<worldbody>{C.mujoco_camera_xml(cam, 'c')}{spheres}</worldbody></mujoco>")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rnd = mujoco.Renderer(m, cam.height, cam.width)
    rnd.enable_segmentation_rendering()
    rnd.update_scene(d, camera="c")
    seg = rnd.render()[..., 0].copy()
    rnd.close()
    out = []
    for i in range(len(pts)):
        ys, xs = np.nonzero(seg == i)
        out.append((xs.mean(), ys.mean()) if len(xs) > 20 else (np.nan, np.nan))
    return np.array(out)


@pytest.mark.parametrize("name,square", [("front", False), ("grip", False), ("grip", True)])
def test_mujoco_projection(scene, name, square):
    """THE principal-point convention: MuJoCo renders spheres where Camera.project says."""
    pytest.importorskip("mujoco")
    cam = scene.camera(name)
    pin = C.render_spec(cam, square=square).camera(cam)
    if pin.parent != "world":  # place the camera in the world at the wrist's HOME pose
        pin = pin.replace(parent="world", T_parent_cam=pin.T_world_cam(
            kinematics.default().link_T(np.array(kinematics.HOME), pin.parent)))
    rng = np.random.default_rng(3)
    uv = rng.uniform([30, 30], [pin.width - 30, pin.height - 30], (16, 2))
    o, d = pin.pixel_to_ray(uv)
    P = o + d * rng.uniform(0.25, 0.6 if name == "grip" else 1.1, (16, 1))
    got = _render_spheres(pin, P, r=0.004 if name == "grip" else 0.008)
    err = np.linalg.norm(got - pin.project(P), axis=1)
    assert np.isfinite(err).sum() >= 12 and np.nanmax(err) < 0.5, err


def test_blender_and_usd_formulas():
    cam = C.Camera("t", 640, 480, (600.0, 600.0, 330.0, 250.0))
    b = C.to_blender(cam)
    assert np.isclose(b["lens"], 600 * 36 / 640) and np.isclose(b["shift_x"], -(330 - 319.5) / 640)
    assert np.isclose(b["shift_y"], (250 - 239.5) / 640)
    u = C.to_usd(cam)
    assert np.isclose(u["focalLength"] / u["horizontalAperture"], 600 / 640)
    with pytest.raises(ValueError):
        C.to_usd(cam.replace(K=(600.0, 590.0, 330.0, 250.0)))


def test_klip_camera_pose_matches_mjlab():
    """The PROVISIONAL layer's grip pose (config/<ds>.json, OpenCV) == mjlab klip_camera.py's
    compiled camera (MuJoCo axes). Only that layer: CALIB's <ds>.calib.json legitimately
    replaces the klip pose with the dataset fit, so the merged scene no longer matches it."""
    if shutil.which("uv") is None:
        pytest.skip("uv (mjlab env) not available")
    code = ("import json, mujoco; from mjlab.asset_zoo.robots.so_arm101.klip_camera import get_so101_klip_spec; "
            "m = get_so101_klip_spec().compile(); c = m.camera('camera_klip'); "
            "print(json.dumps({'body': m.body(m.cam_bodyid[c.id]).name, 'pos': m.cam_pos[c.id].tolist(), "
            "'quat': m.cam_quat[c.id].tolist()}))")
    r = subprocess.run([*paths.MJLAB_PY, "-c", code], capture_output=True, text=True, timeout=300)
    if r.returncode:
        pytest.skip(f"mjlab env failed: {r.stderr[-300:]}")
    k = json.loads(r.stdout.strip().splitlines()[-1])
    from real2sim.transforms import quat_to_mat

    grip = C.Camera.from_config("grip", json.loads(paths.config_path().read_text())["cameras"]["grip"])
    assert k["body"] == grip.parent == "gripper"
    assert np.allclose(grip.T_parent_cam[:3, 3], k["pos"], atol=1e-6)
    assert np.allclose(grip.T_parent_cam[:3, :3] @ C.CV_TO_GL, quat_to_mat(k["quat"]), atol=1e-6)
