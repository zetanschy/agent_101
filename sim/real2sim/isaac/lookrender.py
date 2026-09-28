"""Isaac RTX renders through LOOK's assets and the real cameras (ISAAC part 2).

    ./robot real2sim isaac calibrate                 # RTX light units + dome orientation (3 Kit runs, cached)
    ./robot real2sim isaac render --episode 0 --log <run dir | poselog.npz> [--modes rt,pt]
    ./robot real2sim isaac render --episode 1 --kinematic --samples-only --modes rt,pt
    ./robot real2sim isaac samples [--modes rt,pt]   # LOOK's 24 samples (episodes 0-2) + score

One Kit process renders one MODE (run.sh loops): RTX takes no light edit after its first
frame (look.py), so every process authors its lights once, at the mode's calibrated
intensities.

PLAYBACK, NOT PHYSICS. A render replays a finished pose log (or, with --kinematic, the
recorded state with the objects static at their reset poses, as the Blender track's
kinematic source): the arm joints and the object poses are WRITTEN each frame and no
physics step is taken. So a render can never change a physics result.

CAMERA TIME. CALIB's latency ("compare a sim frame of state time t with the real image
of row t + frames"): the image of row k shows the world at t = k - latency, so each
camera is rendered at its own t, joints linearly interpolated between log rows and object
poses by lerp + normalised quaternion lerp (the Blender job's rule, blender/job.py).

PIPELINE per camera image: the arm (the Jaw at the FINGER angle x_finger of a physics log,
the encoder's for --kinematic) and the objects at t; RTX renders the camera's oversized
square-pixel pinhole (camera.render_spec(cam, square=True), the prims camcheck.py verified)
as the HdrColor AOV, linear, in scene radiance (look.py calibration, re-checked on a grey
quad at the start of every render); then real2sim.blender.camera_model.Response (lens
distortion, LOOK's exposure and white balance, sRGB) makes the 640x480 uint8 frame, the
same code the Blender renderer goes through.

MODES
  rt   RTX real-time ('RayTracedLighting', DLAA), --rt-frames renders per pose (temporal
       denoisers need a history; a pose jump leaves ghosts otherwise)
  pt   path tracing, --pt-spp samples per frame up to --pt-total-spp per pose (the
       accumulation restarts at every pose change), OptiX denoiser unless --pt-no-denoise

OUTPUT sim/outputs/real2sim/<ds>/isaac/render/<tag>/:
    samples_<mode>/<id>_<cam>.png      LOOK's sample frames (scored by ./robot real2sim look score)
    <mode>/front.mp4, grip.mp4, side_by_side.mp4   episode renders (not with --samples-only)
    manifest_ep<N>_<mode>.json         source, scene hash, look build, calibration and its
                                       check, settings, timings (s per image), VRAM
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

LUMA = (0.2126, 0.7152, 0.0722)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--log", help="a replay's run directory or pose log .npz")
    p.add_argument("--kinematic", action="store_true", help="recorded state, objects static at reset (not physical)")
    p.add_argument("--calibrate", choices=("key", "dome"), help="measure one light's RTX units (modes: --modes)")
    p.add_argument("--probe-dome", action="store_true", help="measure the dome's azimuth convention")
    p.add_argument("--mode", default="rt", choices=("rt", "pt"), help="render mode of this process")
    p.add_argument("--modes", default="rt,pt", help="modes a calibration process measures")
    p.add_argument("--samples-only", action="store_true", help="only LOOK's sample frames of this episode")
    p.add_argument("--no-samples", action="store_true", help="skip the sample frames")
    p.add_argument("--frames", default=None, help="first:last episode frames of the video (default: all)")
    p.add_argument("--cameras", default="front,grip")
    p.add_argument("--rt-frames", type=int, default=6, help="RTX real-time renders per pose (samples)")
    p.add_argument("--rt-video-frames", type=int, default=3, help="RTX real-time renders per video frame")
    p.add_argument("--pt-spp", type=int, default=32)
    p.add_argument("--pt-total-spp", type=int, default=256)
    p.add_argument("--pt-no-denoise", action="store_true")
    p.add_argument("--no-steel-fill", action="store_true", help="the steel reflects no unobserved fill (look.py)")
    p.add_argument("--wb", choices=("frame", "episode", "reference"), default="frame")
    p.add_argument("--exposure", choices=("constant", "look"), default="constant")
    p.add_argument("--tag", default=None, help="output directory name (default from the source)")
    p.add_argument("--ds", default=None)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(argv)
    if not (a.calibrate or a.probe_dome) and bool(a.log) == bool(a.kinematic):
        p.error("a render needs exactly one of --log / --kinematic")
    return a


def vram_mib() -> dict:
    """This process's and the whole GPU's memory in use (nvidia-smi), MiB."""
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout
        mine = sum(int(m) for pid, m in (l.split(",") for l in apps.strip().splitlines() if l.strip())
                   if int(pid) == os.getpid())
        tot = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return {"process_mib": mine, "gpu_used_mib": int(tot.strip().splitlines()[0])}
    except Exception as e:  # noqa: BLE001 -- a missing nvidia-smi must not lose the render
        return {"error": repr(e)}


class Source:
    """Joint angles and object poses at a fractional episode frame t."""

    def __init__(self, sc, episode: int, log_path: str | None, lim):
        import numpy as np

        from .. import episodes, poselog, units as U

        self.np = np
        self.ep = episodes.load(sc.ds, include_excluded=True)[episode]
        if log_path:
            p = Path(log_path)
            p = p / "poselog.npz" if p.is_dir() else p
            lg = poselog.read(p)
            if int(lg["meta"]["episode"]) != episode:
                raise SystemExit(f"{p} is episode {lg['meta']['episode']}, not {episode}")
            q = np.array(lg["q"], dtype=float)
            if "x_finger" in lg:  # what is SEEN is the finger; the encoder reads the horn
                q[:, U.GRIPPER] = lg["x_finger"]
            self.frame0 = int(lg["frame"][0])
            self.objects = [str(n) for n in lg["objects"]]
            self.pos, self.quat = np.array(lg["obj_pos"], float), np.array(lg["obj_quat"], float)
            self.label = f"{lg['meta']['engine']} {lg['meta']['mode']} replay ({p.parent.name})"
            self.meta = {k: lg["meta"].get(k) for k in ("engine", "mode", "scene_hash", "seed", "placement")}
            self.path = str(p)
        else:
            q = sc.units().to_urdf(self.ep.state)
            self.frame0, self.objects, self.pos, self.quat = 0, [], None, None
            self.label = "kinematic: arm = recorded state, objects static at reset"
            self.meta = {"engine": "real", "mode": "kinematic", "scene_hash": sc.hash}
            self.path = None
        self.q = np.clip(q, lim[:, 0] + 1e-4, lim[:, 1] - 1e-4)
        self.T = len(self.q)

    def _w(self, t: float):
        k = self.np.clip(t - self.frame0, 0.0, self.T - 1.0)
        i = int(self.np.floor(k))
        j = min(i + 1, self.T - 1)
        return i, j, float(k - i)

    def joints(self, t: float):
        i, j, a = self._w(t)
        return (1 - a) * self.q[i] + a * self.q[j]

    def object_poses(self, t: float) -> dict:
        np = self.np
        if self.pos is None:
            return {}
        i, j, a = self._w(t)
        out = {}
        for o, name in enumerate(self.objects):
            p = (1 - a) * self.pos[i, o] + a * self.pos[j, o]
            qa, qb = self.quat[i, o], self.quat[j, o]
            qb = qb if qa @ qb >= 0 else -qb
            qq = (1 - a) * qa + a * qb
            out[name] = (p, qq / np.linalg.norm(qq))
        return out


def spawn_cameras(sc, cams) -> dict:
    """The camera prims of scene.camera_cfgs (the TiledCamera configs camcheck verified),
    without the Isaac Lab sensor: this renders them through its own HdrColor render
    products. Returns {cam: prim path}."""
    from . import scene as S

    opts = S.Options(episode=0, render=True, cameras=tuple(cams))
    out = {}
    for key, cc in S.camera_cfgs(sc, opts).items():
        path = cc.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        cc.spawn.func(path, cc.spawn, translation=tuple(cc.offset.pos), orientation=tuple(cc.offset.rot))
        out[key[4:]] = path
    return out


# --- the dome probe: a synthetic environment and tilted quads, all authored at spawn -------

PROBE_DIRS = {"+x": (1, 0), "+y": (0, 1), "-x": (-1, 0), "-y": (0, -1)}


def probe_env(path: Path) -> Path:
    """1.0 in a +-45 deg wedge about LOOK's azimuth 0 (+x), 0.5 about +90 deg (+y), both at
    elevation -10..70 deg, black elsewhere; LOOK's layout (column = atan2(y, x) from -pi,
    row 0 = +z)."""
    import numpy as np

    from .look import _write_float_image

    H, W = 256, 512
    az = -np.pi + (np.arange(W) + 0.5) / W * 2 * np.pi
    el = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    A, EL = np.meshgrid(az, el)
    env = np.zeros((H, W, 3), np.float32)
    band = (EL > np.radians(-10)) & (EL < np.radians(70))
    env[band & (np.abs(np.angle(np.exp(1j * A))) < np.radians(45))] = 1.0
    env[band & (np.abs(np.angle(np.exp(1j * (A - np.pi / 2)))) < np.radians(45))] = 0.5
    _write_float_image(path, env)
    return path


def probe_quads(look, stage) -> dict:
    """Four grey quads tilted 45 deg towards +-x, +-y (lit by the wedges they face) and one
    mirror tilted 35 deg towards +x (its reflection of a view from above is the +x wedge at
    ~20 deg elevation): a lit mirror means the diffuse dome's inputs:specular 0 is ignored."""
    import numpy as np
    from pxr import Gf, UsdGeom, Vt

    z = look.sc.table_z() + 0.03
    mirror = look._preview(stage, "/World/Looks/probe_mirror", (1.0, 1.0, 1.0), 0.05, 1.0)
    out = {}
    for name, (dx, dy), tilt, mat in [(n, d, 45.0, look.paths["grey"]) for n, d in PROBE_DIRS.items()] + \
                                     [("mirror+x", (1, 0), 35.0, mirror)]:
        c = np.array([0.10 * dx - 0.02 + (0.12 if name == "mirror+x" else 0.0), -0.30 + 0.10 * dy, z])
        t = np.radians(tilt)
        nrm = np.array([dx * np.sin(t), dy * np.sin(t), np.cos(t)])
        u = np.array([-dy, dx, 0.0])
        v = np.cross(nrm, u)
        s = 0.025
        pts = [c - s * u - s * v, c + s * u - s * v, c + s * u + s * v, c - s * u + s * v]
        path = "/World/envs/env_0/probe_" + name.replace("+", "p").replace("-", "m")
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in pts]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, nrm))] * 4))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        m.CreateDoubleSidedAttr(True)
        look._bind(m.GetPrim(), mat)
        out[name] = c
    return out


def main(args) -> None:
    import cv2
    import numpy as np
    import omni.replicator.core as rep
    import torch
    from isaaclab.scene import InteractiveScene
    from pxr import UsdGeom

    from .. import camera as cam_mod, episodes, paths, scene as r2s_scene, units as U
    from . import look as LK
    from . import scene as S
    from . import servo as SV
    from .evaluate import scene_of_run

    t_start = time.time()
    run = None
    if args.log:
        run = Path(args.log) if Path(args.log).is_dir() else Path(args.log).parent
        info = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
        sc = scene_of_run(info, args.ds) if info else r2s_scene.load(args.ds)
        placement = info.get("placement", "config")
    else:
        sc, placement = r2s_scene.load(args.ds), "config"
    cams = tuple(c for c in args.cameras.split(",") if c)
    setup = args.calibrate or ("probe" if args.probe_dome else "render")
    penv = None
    if setup == "probe":
        penv = str(probe_env(paths.out_dir(sc.ds, "isaac", "look_tex", create=True) / "probe_env.hdr"))
    units = LK.units_for(sc.ds, args.mode) if setup == "render" else None
    look = LK.LookAssets(sc, units=units, setup=setup, probe_env=penv, steel_fill=not args.no_steel_fill)

    opts = S.Options(episode=args.episode, num_envs=1, render=False, cameras=cams, placement=placement,
                     kinematic_objects=True)
    sim = S.make_sim(opts, look)
    iscene = InteractiveScene(S.scene_cfg(sc, opts, SV.physx_gains(SV.load_model(sc), True), look))
    spawn_info = S.post_spawn(sim.stage, iscene, sc, opts, look)
    quads = probe_quads(look, sim.stage) if setup == "probe" else {}
    cam_paths = spawn_cameras(sc, cams)
    sim.reset()
    robot = iscene["robot"]
    names = list(robot.joint_names)
    perm = [names.index(n) for n in U.URDF_JOINTS]
    lim = S.joint_limits_rad(sc, opts.params["physics"]["limit_margin_deg"])
    src = Source(sc, args.episode, args.log, lim)
    obj_assets = {o["name"]: iscene[o["name"]] for o in S.object_placements(sc, opts)}
    dev = robot.device
    specs = {c: cam_mod.render_spec(sc.camera(c), square=True) for c in cams}
    rps, anns = {}, {}
    for c in cams:
        rps[c] = rep.create.render_product(cam_paths[c], (specs[c].width, specs[c].height))
        anns[c] = rep.AnnotatorRegistry.get_annotator("HdrColor")
        anns[c].attach([rps[c]])
    lat = {c: float((sc["cameras"][c].get("latency") or {}).get("frames", 0.0)) for c in cams}

    def pose(t: float) -> None:
        q = torch.zeros(1, len(names), device=dev)
        q[0, perm] = torch.tensor(src.joints(t), dtype=torch.float32, device=dev)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        for name, (p, qq) in src.object_poses(t).items():
            if name in obj_assets:
                st = torch.tensor([[*p, *qq]], dtype=torch.float32, device=dev)
                st[:, :3] += iscene.env_origins[:1]
                obj_assets[name].write_root_pose_to_sim(st)
        sim.forward()

    def grab(cam, n: int):
        # both render products render every frame: disabling one left its annotator empty
        # after re-enabling (MEASURED); every annotator is read each frame so that none
        # builds up a backlog
        for _ in range(n):
            sim.render()
            got = {c: anns[c].get_data() for c in cams}
        a = np.asarray(got[cam], dtype=np.float32)
        if a.ndim != 3 or a.shape[:2] != (specs[cam].height, specs[cam].width):
            raise RuntimeError(f"{cam}: HdrColor {a.shape}, expected {(specs[cam].height, specs[cam].width)}")
        return a[..., :3]

    def luma_median(img) -> float:
        h, w = img.shape[:2]
        return float(np.median(img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] @ np.array(LUMA)))

    def settle(n_max: int = 40) -> int:
        """Textures and shaders load asynchronously: render until the front view is steady."""
        prev = None
        for i in range(1, n_max + 1):
            img = grab(cams[0], 5)
            if prev is not None and np.abs(img - prev).mean() <= 1e-4 * max(float(np.abs(img).mean()), 1e-6):
                return 5 * i
            prev = img
        return 5 * n_max

    pose(0.0)
    n_settle = settle()
    renders_per_pose = {"rt": args.rt_frames, "pt": int(np.ceil(args.pt_total_spp / args.pt_spp)) + 2}

    # --- calibration: one light, every requested mode --------------------------------------
    if setup in ("key", "dome"):
        exp = look.expected()
        patch = {"modes": {}, "expected": exp}
        for mode in [m for m in args.modes.split(",") if m]:
            look.set_mode(mode, args.pt_spp, args.pt_total_spp, not args.pt_no_denoise)
            settle()
            meas = luma_median(grab(cams[0], renders_per_pose[mode] + 4))
            rec = {setup: {"measured_render_units": meas, "expected_scene_radiance": exp[setup],
                           "render_units_per_scene_unit": meas / exp[setup],
                           "source": f"measured: lookrender --calibrate {setup}, grey quad albedo {LK.TEST_ALBEDO}"}}
            if setup == "dome":  # emission needs no light: the unit emitter renders its own units
                look.bind_table("emit")
                m_e = luma_median(grab(cams[0], renders_per_pose[mode] + 4))
                rec["emit"] = {"measured_render_units": m_e, "expected_scene_radiance": 1.0,
                               "render_units_per_scene_unit": m_e,
                               "source": "measured: lookrender --calibrate dome, unit emitter"}
                look.bind_table("grey")
            patch["modes"][mode] = rec
            print(f"CALIBRATION {setup} {mode} {json.dumps(rec)}", flush=True)
        LK.update_calibration(sc.ds, patch, look.tex)
        return

    # --- the dome probe ---------------------------------------------------------------------
    if setup == "probe":
        look.set_mode("rt")
        settle()
        img = grab(cams[0], 40)
        pin = specs[cams[0]].camera(sc.camera(cams[0]))
        vals = {}
        for name, c in quads.items():
            u, v = pin.project(np.asarray(c)[None])[0]
            iu, iv = int(round(u)), int(round(v))
            vals[name] = float(np.median(img[iv - 3:iv + 4, iu - 3:iu + 4] @ np.array(LUMA)))
        grey = {k: vals[k] for k in PROBE_DIRS}
        order = sorted(grey, key=grey.get, reverse=True)
        ang = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": -90.0}
        rot = -ang[order[0]] % 360.0  # the rotation that brings LOOK's +x back to +x
        turn = (ang[order[1]] - ang[order[0]]) % 360.0  # +90: same handedness, 270: mirrored
        res = {"quad_luma": vals, "look_+x_rendered_at": order[0], "look_+y_rendered_at": order[1],
               "mirrored": bool(abs(turn - 270.0) < 1e-6),
               "diffuse_dome_specular_0_honoured": bool(vals["mirror+x"] < 0.1 * max(grey.values()))}
        LK.update_calibration(sc.ds, {"dome_probe": res, "dome_rotation_deg": {
            "value": rot, "source": f"measured: lookrender --probe-dome, LOOK +x rendered at {order[0]}"}}, look.tex)
        print("PROBE", json.dumps(res), flush=True)
        return

    # --- a render: one mode, the scene in scene radiance -------------------------------------
    mode = args.mode
    settings = look.set_mode(mode, args.pt_spp, args.pt_total_spp, not args.pt_no_denoise)
    n_pose = renders_per_pose[mode]
    n_video = args.rt_video_frames if mode == "rt" else n_pose
    settle()

    from ..blender import camera_model as CM
    from ..blender.post import side_by_side
    from .render import VideoOut

    tag = args.tag or (f"ep{args.episode}_kinematic" if args.kinematic else f"ep{args.episode}_{run.name}")
    out = paths.out_dir(sc.ds, "isaac", "render", tag, create=True)
    idx = json.loads((paths.out_dir(sc.ds, "look") / "samples" / "index.json").read_text())
    samples = [s for s in idx["samples"] if int(s["episode"]) == args.episode] if not args.no_samples else []
    state = CM.CameraState.load(sc.ds)
    resp = {c: CM.Response(sc, look.look, c, specs[c], state, wb=args.wb, exposure=args.exposure) for c in cams}
    ep = episodes.load(sc.ds, include_excluded=True)[args.episode]
    row0 = int(ep.start)  # the dataset row of frame 0 (the front white balance is per row)
    rec = {"settings": settings, "renders_per_pose": n_pose, "renders_per_video_frame": n_video,
           "units": look.units, "samples": {}, "video": {}}
    sdir = out / f"samples_{mode}"
    sdir.mkdir(exist_ok=True)
    t0, n_img = time.time(), 0
    for s in samples:
        fr, row = int(s["frame"]), int(s["row"])
        for c in cams:
            pose(fr - lat[c])
            img = resp[c](grab(c, n_pose), args.episode, fr, row)
            cv2.imwrite(str(sdir / f"{s['id']}_{c}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            n_img += 1
    if n_img:
        rec["samples"] = {"images": n_img, "s_per_image": round((time.time() - t0) / n_img, 3), "dir": str(sdir),
                          "vram": vram_mib()}
        print(f"SAMPLES {mode} {n_img} images, {rec['samples']['s_per_image']} s/image", flush=True)
    if not args.samples_only:
        lo, hi = 0, len(ep) - 1
        if args.frames:
            a_, _, b_ = args.frames.partition(":")
            lo, hi = int(a_ or 0), int(b_ or hi)
        vdir = out / mode
        vdir.mkdir(exist_ok=True)
        vids = {c: VideoOut(vdir / f"{c}.mp4", 640, 480, int(ep.fps)) for c in cams}
        t1 = time.time()
        for k in range(lo, hi + 1):
            for c in cams:
                pose(k - lat[c])
                vids[c].write(resp[c](grab(c, n_video), args.episode, k, row0 + k))
            if (k - lo) % 100 == 0:
                print(f"  {mode} frame {k}/{hi}  {time.time() - t1:.1f} s", flush=True)
        n = (hi - lo + 1) * len(cams)
        rec["video"] = {"frames": [lo, hi], "s_per_image": round((time.time() - t1) / n, 3),
                        "files": {c: {"path": str(v.path), "rc": v.close()} for c, v in vids.items()},
                        "vram": vram_mib()}
        job = {"cameras": {c: {} for c in cams}, "dataset": sc.ds, "frames": list(range(lo, hi + 1)),
               "episode": args.episode, "engine": f"isaac {mode}"}
        rec["video"]["side_by_side"] = str(side_by_side(vdir, job, row0, int(ep.fps), src.label))
        print(f"VIDEO {mode} {rec['video']['s_per_image']} s/image -> {vdir}", flush=True)
    # the check, LAST (hiding the robot hides the wrist camera, whose render product then
    # lagged ~30 frames, MEASURED): the calibrated lights on the grey quad
    look.bind_table("grey")
    for h in ("Robot", "mug"):
        UsdGeom.Imageable(sim.stage.GetPrimAtPath(f"/World/envs/env_0/{h}")).MakeInvisible()
    m = luma_median(grab(cams[0], n_pose + 4))
    exp_all = look.expected()["all"]
    rec["check"] = {"grey_quad_measured": m, "expected": exp_all, "ratio": m / exp_all,
                    "note": "table quad albedo 0.5, robot and mug hidden, caps and backdrop shown, crop median"}
    print(f"CHECK {mode} {json.dumps(rec['check'])}", flush=True)
    manifest = {"source": src.label, "source_meta": src.meta, "log": src.path, "episode": args.episode, "mode": mode,
                "scene_hash": sc.hash, "scene_overrides": getattr(sc, "overrides", {}),
                "look": {k: look.look.get(k) for k in ("scene_hash", "inputs_hash", "built")}, "look_info": look.info,
                "payload_kg": spawn_info.get("payload", {}).get("gripper_mass_with_payload"), "latency_frames": lat,
                "cameras": {c: {"pinhole": [specs[c].width, specs[c].height], "response": resp[c].describe()}
                            for c in cams},
                "grip_exposure": CM.GRIP_EXPOSURE, "optics": CM.OPTICS, "settle_renders": n_settle,
                "boot_s": round(t0 - t_start, 1), "render": rec, "wall_s": round(time.time() - t_start, 1)}
    (out / f"manifest_ep{args.episode}_{mode}.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"RENDER {out}", flush=True)


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
    except BaseException:  # noqa: BLE001 -- os._exit below would swallow it
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_code)  # Kit 4.5 hangs in app.close() on this box
