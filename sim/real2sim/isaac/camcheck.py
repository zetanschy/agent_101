"""Where Isaac's RTX cameras put a known 3-D point, against real2sim.camera.

    ./robot real2sim isaac camcheck [--episode 0 --frame 231]

Emissive marker spheres are placed at known world points (a grid on the table for the
front camera, a grid 12-20 cm in front of the wrist camera at the pose of one recorded
frame), the scene is rendered with the production cameras (scene.camera_cfgs) and
with a variant that hands sim_agent101's camera_cfg the raw OpenCV cx, cy (no +0.5),
and every marker's intensity centroid is compared with Camera.project of its centre
in the render_spec pinhole. The distorted 640x480 image (render.py's remap) is checked
the same way against the full camera model. Writes outputs/<ds>/isaac/camcheck.json.

Lights are off, the table, the floor's grid, the objects and the robot's visuals are
hidden and anti-aliasing is off:
the markers are emissive and float in black, so a centroid is a marker and nothing else
(with the glossy table visible its reflection pulled the front centroids 1-4 px). Marker
sizes keep the perspective bias of a sphere's centroid below 0.05 px.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=231)
    p.add_argument("--ds", default=None)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    return p.parse_args(argv)


def centroids(img, expected, radius: float = 12.0, thresh: float = 60.0):
    """Intensity-weighted centroid of bright pixels within `radius` px of each expected point."""
    import numpy as np

    g = img[..., :3].astype(float).mean(-1)
    H, W = g.shape
    vv, uu = np.mgrid[0:H, 0:W]
    out = []
    for u0, v0 in expected:
        if not (0 <= u0 < W and 0 <= v0 < H):
            out.append((np.nan, np.nan))
            continue
        m = (np.hypot(uu - u0, vv - v0) < radius) & (g > thresh)
        w = np.where(m, g - thresh, 0.0)
        s = w.sum()
        out.append((float((w * uu).sum() / s), float((w * vv).sum() / s)) if s > 0 else (np.nan, np.nan))
    return np.array(out)


def main(args):
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene

    from sim_agent101.assets.objects import camera_cfg

    from .. import camera as cam_mod
    from .. import episodes, kinematics, paths, scene as r2s_scene, units as U
    from ..transforms import mat_to_quat
    from . import scene as S
    from . import servo as SV

    sc = r2s_scene.load(args.ds)
    ep = episodes.load(sc.ds, include_excluded=True)[args.episode]
    q = sc.units().to_urdf(ep.state[args.frame])
    kin = kinematics.default()
    T_g = kin.link_T(q, "gripper")
    tz = sc.table_z()
    # marker points: front = table grid; grip = a grid in front of the wrist camera
    front, grip = sc.camera("front"), sc.camera("grip")
    pts_f = np.array([[x, y, tz + 0.05] for x in np.linspace(-0.25, 0.25, 5) for y in np.linspace(-0.45, -0.10, 4)])
    Tc = grip.T_world_cam(T_g)
    spec_g = cam_mod.render_spec(grip, square=True)
    pin_g = spec_g.camera(grip)
    pts_g = []
    for z in (0.12, 0.16, 0.20):
        for un, vn in ((-0.35, -0.3), (0.35, -0.3), (0.0, -0.1), (-0.35, 0.1), (0.35, 0.1)):
            pts_g.append(Tc[:3, :3] @ np.array([un * z, vn * z, z]) + Tc[:3, 3])
    pts_g = np.array(pts_g)

    look = S.Look()
    look.lights = lambda: {"dome_light": AssetBaseCfg(prim_path="/World/DomeLight",
                                                       spawn=sim_utils.DomeLightCfg(intensity=1.0))}
    look.render_cfg = lambda: sim_utils.RenderCfg(rendering_mode="quality", enable_translucency=False,
                                                  antialiasing_mode="Off")
    opts = S.Options(episode=args.episode, render=True, cameras=("front", "grip"))
    model = SV.load_model(sc)
    cfg = S.scene_cfg(sc, opts, SV.physx_gains(model), look)
    # the no-+0.5 variants, same poses
    for name in ("front", "grip"):
        cam = sc.camera(name)
        pin = cam_mod.render_spec(cam, square=True).camera(cam)
        T_gl = cam_mod.opencv_to_opengl(pin.T_parent_cam)
        parent = "{ENV_REGEX_NS}" if cam.parent == "world" else "{ENV_REGEX_NS}/Robot/" + cam.parent
        intr = {name: {"width": pin.width, "height": pin.height, "fx": pin.fx, "fy": pin.fy, "cx": pin.cx, "cy": pin.cy}}
        setattr(cfg, f"raw_{name}", camera_cfg(name, f"{parent}/raw_{name}", pos=tuple(T_gl[:3, 3]),
                                               rot_quat=tuple(mat_to_quat(T_gl[:3, :3])), width=pin.width,
                                               height=pin.height, intrinsics=intr,
                                               near_m=S.GRIP_NEAR if cam.parent != "world" else 0.05))
    emissive = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), emissive_color=(1.0, 1.0, 1.0))
    for i, p in enumerate(pts_f):
        setattr(cfg, f"mf{i:02d}", AssetBaseCfg(prim_path=f"/World/markers/f{i:02d}",
                                                spawn=sim_utils.SphereCfg(radius=0.004, visual_material=emissive),
                                                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(p))))
    for i, p in enumerate(pts_g):
        setattr(cfg, f"mg{i:02d}", AssetBaseCfg(prim_path=f"/World/markers/g{i:02d}",
                                                spawn=sim_utils.SphereCfg(radius=0.002, visual_material=emissive),
                                                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(p))))
    sim = S.make_sim(opts, look)
    iscene = InteractiveScene(cfg)
    S.post_spawn(sim.stage, iscene, sc, opts, look)
    from pxr import Usd, UsdGeom

    hide = [sim.stage.GetPrimAtPath(p) for p in ("/World/envs/env_0/Table", "/World/ground")]
    hide += [sim.stage.GetPrimAtPath(f"/World/envs/env_0/{o['name']}") for o in S.object_placements(sc, opts)]
    for pr in Usd.PrimRange(sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if pr.GetName() in ("visuals", "klip_support", "kwc500_head"):
            hide.append(pr)
    for pr in hide:
        UsdGeom.Imageable(pr).MakeInvisible()
    sim.reset()
    if float(iscene.env_origins[0].abs().max()) > 1e-9:
        raise RuntimeError("camcheck places markers in the world frame: env 0 must sit at the origin")
    robot = iscene["robot"]
    names = list(robot.joint_names)
    perm = [names.index(n) for n in U.URDF_JOINTS]
    qa = torch.zeros(1, len(names), device=robot.device)
    qa[0, perm] = torch.tensor(q, dtype=torch.float32, device=robot.device)
    robot.write_joint_state_to_sim(qa, torch.zeros_like(qa))
    for _ in range(40):
        sim.render()
    res = {"episode": args.episode, "frame": args.frame, "scene_hash": sc.hash}
    from .render import CameraRecorder  # noqa: F401  (same remap path as replays)

    for name, cam, pts in (("front", front, pts_f), ("grip", grip, pts_g)):
        spec = cam_mod.render_spec(cam, square=True)
        pin = spec.camera(cam)
        Tp = None if cam.parent == "world" else T_g
        exp_pin = pin.project(pts, Tp)
        exp_dist = cam.project(pts, Tp)
        row = {}
        for variant in ("cam", "raw"):
            c = iscene[f"{variant}_{name}"]
            c.update(0.0, force_recompute=True)
            img = c.data.output["rgb"][0].cpu().numpy()
            import imageio.v3 as iio

            iio.imwrite(paths.out_dir(sc.ds, "isaac", create=True) / f"camcheck_{variant}_{name}_pinhole.png", img[..., :3])
            got = centroids(img, exp_pin)
            d = got - exp_pin
            ok = np.all(np.isfinite(d), axis=1)
            row[variant] = {"n": int(ok.sum()), "of": int(len(d)), "mean_du_dv_px": d[ok].mean(0).round(3).tolist(),
                            "rms_px": float(np.sqrt((d[ok] ** 2).sum(1).mean())), "max_px": float(np.abs(d[ok]).max())}
            if variant == "cam":
                dist = cam_mod.remap(np.ascontiguousarray(img[..., :3]), cam_mod.remap_maps(cam, spec))
                got2 = centroids(dist, exp_dist, radius=10.0)
                d2 = got2 - exp_dist
                ok2 = np.all(np.isfinite(d2), axis=1)
                row["distorted"] = {"n": int(ok2.sum()), "mean_du_dv_px": d2[ok2].mean(0).round(3).tolist(),
                                    "rms_px": float(np.sqrt((d2[ok2] ** 2).sum(1).mean())),
                                    "max_px": float(np.abs(d2[ok2]).max())}
                import imageio.v3 as iio

                iio.imwrite(paths.out_dir(sc.ds, "isaac", create=True) / f"camcheck_{name}.png", dist)
        res[name] = row
    out = paths.out_dir(sc.ds, "isaac", create=True) / "camcheck.json"
    out.write_text(json.dumps(res, indent=1))
    print("CAMCHECK", json.dumps(res), flush=True)


if __name__ == "__main__":
    _args = parse_args()
    from isaaclab.app import AppLauncher

    _args.headless = True
    _args.enable_cameras = True
    _app = AppLauncher(_args).app
    _code = 1
    try:
        main(_args)
        _code = 0
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_code)
