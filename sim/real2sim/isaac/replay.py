"""Replay one real episode in Isaac Sim 4.5 / Isaac Lab 2.1, physically.

    ./robot real2sim isaac replay --episode 0 --mode action [--num-envs K] [--render] [--gui]

MODES: the MuJoCo track's (real2sim.mujoco.replay), on the same servo model (servo.py):
  action     THE PHYSICALLY HONEST DEFAULT. All six servos get the recorded ACTION,
             clamped to the firmware limits, zero-order held per frame, delayed by the
             identified dead time and slewed (the MuJoCo track's GoalStream, vectorised).
             Nothing else steers the arm.
  track      HYBRID. The five arm servos get the goal that makes the servo model follow
             the recorded STATE (track.py: inverse dynamics along a spline through
             observation.state, u = q_ref + (tau + kd qd_ref) / kp), keeping the torque
             clamp and the compliance; the Jaw stays on the recorded action through the
             servo and its torque limit (the jaw state is blocked by the cap).
  kinematic  NOT PHYSICAL (meta physical=false): the arm is written to the recorded state
             every frame, the objects are kinematic. Render and camera checks only.

TIMING. Frame k is t_k = k / fps; the logged state of frame k is the simulation at t_k,
before the 16 physics steps to t_(k+1); ctrl is the goal in force at t_k. Before t_0 the
objects settle for settle_s under gravity with the arm held at the recorded first frame.
Objects are placed once (placement.py) and afterwards move only through contact.

ENSEMBLES. --num-envs K runs K copies in parallel; env 0 is always the nominal replay
and envs 1..K-1 draw perturbations from --perturb (seeded by --seed).

--perturb mujoco[:SIGMA] draws the MuJoCo track's ensemble scenes: env e >= 1 gets
exactly real2sim.mujoco.evaluate.draw_perturbation of seed (--seed, default 1000) + e - 1,
imported, not copied. The MuJoCo eval's seeds are 1000 + i, so --num-envs K+1 replays
the K scenes of its `--ensemble K`, seed by seed: per-cap xy, table height (objects move
with it), cap diameter (and height, if drawn) and mass, and the four friction groups
(finger = the robot's material, cap, mug, mat = the table's). SIGMA is the MuJoCo
--sigma string (cap_xy=M,table_z=M,cap_d=LO:HI), e.g. mujoco:cap_xy=0.003,table_z=0.001.

The older per-key spec:
    cap_xy=<m>      per-cap xy offset, normal with that sigma
    table_z=<m>     table height offset, normal (objects move with the table)
    cap_d=<m>       cap diameter offset, normal (cap scaled in x-y only)
    friction=<f>    every object's and the robot's friction scaled by U(1-f, 1+f)
    mass=<f>        every cap's mass scaled by U(1-f, 1+f)
Geometry perturbations (table_z, cap_d) need every env parsed on its own
(replicate_physics off), which costs start-up time; README has the numbers.

OUTPUT sim/outputs/real2sim/<ds>/isaac/ep<N>_<mode>[_<tag>]/:
    poselog.npz          env 0, real2sim.poselog (URDF base frame). q holds what the
                         encoders read (the Jaw HORN), as the MuJoCo track's does; extras:
                         x_finger, x_flex (finger - horn), x_tau (servo torques), and
                         contact.py's: x_touch, x_fc_force, x_pen_fc, x_pen_obj, x_arm_obj
                         (the MuJoCo track's definitions) plus the per-link PhysX detail
                         x_robot_* (T, L, F) and x_objects_* (T, C, 2)
    ensemble/env_KKK.npz the other envs, same format
    run.json             arguments, parameters with provenance, PhysX read-backs,
                         placements, perturbations, timings
    front.mp4, grip.mp4  with --render: RTX frames warped into the real cameras
Evaluate with ./robot real2sim isaac eval <run dir>.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

MODES = ("action", "track", "kinematic")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--mode", choices=MODES, default="action")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--render", action="store_true", help="RTX frames of both cameras -> mp4")
    p.add_argument("--gui", action="store_true", help="open the Kit window (implies rendering)")
    p.add_argument("--cameras", default="front,grip")
    p.add_argument("--placement", choices=("config", "grasp"), default="config")
    p.add_argument("--frames", default=None, help="first:last episode frames to replay (default: all)")
    p.add_argument("--perturb", default="", help="mujoco[:SIGMA] (the MuJoCo track's draws) or "
                   "cap_xy=0.004,table_z=0.002,cap_d=0.001,friction=0.2,mass=0.3")
    p.add_argument("--seed", type=int, default=None, help="perturbation seed (default 0; 1000 with --perturb mujoco)")
    p.add_argument("--physics", action="append", default=[], metavar="KEY=VALUE",
                   help="override an isaac/params.py value, e.g. hz=240 or materials.cap.static_friction=0.6")
    p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                   help="override a merged-scene value for a sensitivity run, e.g. table.z=0.0147")
    p.add_argument("--no-flex", action="store_true", help="rigid gripper (ignore the servo model's Jaw flex)")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None, help="run directory (default: outputs/<ds>/isaac/ep<N>_<mode>[_<tag>])")
    p.add_argument("--ds", default=None)
    p.add_argument("--warmup", type=int, default=30, help="RTX frames rendered before frame 0 (denoiser)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    return p.parse_args(argv)


def mujoco_draws(scene, episode: int, num_envs: int, seed0: int, sigma_spec: str = "") -> list[dict]:
    """[{...}] per env; env 0 nominal, env e the MuJoCo track's draw of seed seed0 + e - 1
    (module doc), kept verbatim under "mujoco" and mapped onto this replay's knobs."""
    import numpy as np

    from ..mujoco.evaluate import draw_perturbation

    sigma = None
    if sigma_spec:  # the MuJoCo cli's --sigma syntax (mujoco/cli.py _sigma)
        sigma = {}
        for item in sigma_spec.split(","):
            k, _, v = item.partition("=")
            sigma[k] = tuple(float(x) for x in v.split(":")) if ":" in v else float(v)
    cd = scene.cap_dims()
    out = [{"env": 0, "nominal": True}]
    for e in range(1, num_envs):
        s = seed0 + e - 1
        p = draw_perturbation(scene, episode, np.random.default_rng(s), sigma).as_dict()
        if any(p.get("mug_dxy") or ()):
            raise NotImplementedError(f"MuJoCo draw {s} moves the mug; this replay cannot yet")
        d = {"env": e, "mujoco_seed": s, "sigma": sigma, "mujoco": p,
             "cap_dxy": {c: [float(v) for v in xy] for c, xy in p["cap_dxy"].items()},
             "table_dz": float(p["table_dz"]),
             "friction_scales": {k: float(v) for k, v in (p.get("friction_scale") or {}).items()}}
        if p.get("cap_diameter") is not None:
            d["cap_scale"] = float(p["cap_diameter"]) / cd["diameter"]
        if p.get("cap_height") is not None:
            d["cap_zscale"] = float(p["cap_height"]) / cd["height"]
        if p.get("cap_mass") is not None:
            d["cap_mass"] = float(p["cap_mass"])
        out.append(d)
    return out


def draw_perturbations(spec: str, num_envs: int, seed: int, cap_ids: list[str], cap_d: float) -> list[dict]:
    """[{...}] per env; env 0 nominal. See module doc for the keys."""
    import numpy as np

    sig = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        k, _, v = item.partition("=")
        if k not in ("cap_xy", "table_z", "cap_d", "friction", "mass"):
            raise ValueError(f"unknown perturbation {k!r}")
        sig[k] = float(v)
    rng = np.random.default_rng(seed)
    out = [{"env": 0, "nominal": True}]
    for e in range(1, num_envs):
        d = {"env": e}
        if "cap_xy" in sig:
            d["cap_dxy"] = {c: rng.normal(0.0, sig["cap_xy"], 2).round(6).tolist() for c in cap_ids}
        if "table_z" in sig:
            d["table_dz"] = float(round(rng.normal(0.0, sig["table_z"]), 6))
        if "cap_d" in sig:
            d["cap_scale"] = float(round(1.0 + rng.normal(0.0, sig["cap_d"]) / cap_d, 6))
        if "friction" in sig:
            d["friction_scale"] = float(round(rng.uniform(1 - sig["friction"], 1 + sig["friction"]), 6))
        if "mass" in sig:
            d["mass_scale"] = float(round(rng.uniform(1 - sig["mass"], 1 + sig["mass"]), 6))
        out.append(d)
    return out


def main(args) -> None:
    import numpy as np
    import torch
    from isaaclab.scene import InteractiveScene

    from .. import episodes, kinematics, paths, poselog, scene as r2s_scene, units as U
    from ..transforms import quat_to_mat
    from . import params as P
    from . import scene as S
    from . import servo as SV
    from .contact import Contacts

    t_start = time.time()
    sc, overrides = P.scene_with_overrides(r2s_scene.load(args.ds), args.set)
    ep = episodes.load(sc.ds, include_excluded=True)[args.episode]
    if not ep.use:
        print(f"WARNING: episode {args.episode} is excluded in the scene ({ep.reason})", flush=True)
    units = sc.units()
    f0, f1 = 0, len(ep) - 1
    if args.frames:
        a, _, b = args.frames.partition(":")
        f0, f1 = int(a or 0), int(b or len(ep) - 1)
    T = f1 - f0 + 1
    prm = P.override(P.defaults(), args.physics)
    ph = prm["physics"]
    hz, fps = int(ph["hz"]), ep.fps
    if hz % fps:
        raise SystemExit(f"physics rate {hz} Hz is not a multiple of the {fps} fps frame rate")
    sub, dt = hz // fps, 1.0 / hz
    render = args.render or args.gui
    cams = tuple(c for c in args.cameras.split(",") if c) if render else ()
    cap_ids = [c["id"] for c in sc.episode(args.episode)["caps"]]
    kind, _, sigma_spec = args.perturb.partition(":")
    if kind == "mujoco":
        args.seed = 1000 if args.seed is None else args.seed
        pert = mujoco_draws(sc, args.episode, args.num_envs, args.seed, sigma_spec)
    else:
        args.seed = 0 if args.seed is None else args.seed
        pert = draw_perturbations(args.perturb, args.num_envs, args.seed, cap_ids, sc.cap_dims()["diameter"])
    opts = S.Options(episode=args.episode, num_envs=args.num_envs, render=render, cameras=cams,
                     placement=args.placement, kinematic_objects=args.mode == "kinematic", params=prm, perturb=pert)
    name = f"ep{args.episode}_{args.mode}" + (f"_{args.tag}" if args.tag else "")
    run = Path(args.out) if args.out else paths.out_dir(sc.ds, "isaac", name)
    run.mkdir(parents=True, exist_ok=True)
    model = SV.load_model(sc)
    flex = not args.no_flex
    gains = SV.physx_gains(model, flex)

    look = S.Look()
    sim = S.make_sim(opts, look)
    iscene = InteractiveScene(S.scene_cfg(sc, opts, gains, look))
    spawn_info = S.post_spawn(sim.stage, iscene, sc, opts, look)
    t_built = time.time()
    sim.reset()
    t_reset = time.time()
    look.post_reset(sim)

    robot = iscene["robot"]
    dev = robot.device
    E = args.num_envs
    names = list(robot.joint_names)
    perm = [names.index(n) for n in U.URDF_JOINTS]
    link_idx = [list(robot.body_names).index(n) for n in kinematics.LINKS]
    objs = S.object_placements(sc, opts)
    obj_names = [o["name"] for o in objs]
    assets = [iscene[n] for n in obj_names]
    origins = iscene.env_origins
    servo = SV.PhysxServo(model, perm, dt, E, dev, flex=flex)
    g = servo.physx_tensors()
    rep = lambda v: v.unsqueeze(0).repeat(E, 1)  # noqa: E731
    robot.write_joint_stiffness_to_sim(rep(g["stiffness"]))
    robot.write_joint_damping_to_sim(rep(g["damping"]))
    robot.write_joint_armature_to_sim(rep(g["armature"]))
    robot.write_joint_effort_limit_to_sim(rep(g["max_force"]))
    readback = S.after_reset(iscene, sc, opts)

    # --- initial state: the recorded first frame, objects placed once ---------------------
    A = units.to_urdf(ep.action)
    Sx = units.to_urdf(ep.state)
    lim_j = S.joint_limits_rad(sc, ph["limit_margin_deg"])
    Sx_in = np.clip(Sx, lim_j[:, 0] + 1e-4, lim_j[:, 1] - 1e-4)  # written states stay inside the limits
    qa = torch.zeros(E, len(names), device=dev)
    qa[:, perm] = torch.tensor(Sx_in[f0], dtype=torch.float32, device=dev)
    zeros = torch.zeros_like(qa)

    def hold_arm():
        robot.write_joint_state_to_sim(qa, zeros)
        robot.set_joint_position_target(qa)
        robot.set_joint_velocity_target(zeros)
        robot.set_joint_effort_target(zeros)

    hold_arm()
    cap_h = sc.cap_dims()["height"]
    for o, asset in zip(objs, assets):
        st = torch.zeros(E, 13, device=dev)
        # an open-up cap's origin (its rim plane) is on top: a taller cap raises it
        down = o["kind"] == "cap" and quat_to_mat(o["quat"])[2, 2] < 0
        for e in range(E):
            pos = np.array(o["pos"], dtype=float)
            pe = pert[e]
            if o["kind"] == "cap" and "cap_dxy" in pe:
                pos[:2] += pe["cap_dxy"][o["name"][4:]]
            if down:
                pos[2] += (pe.get("cap_zscale", 1.0) - 1.0) * cap_h
            pos[2] += pe.get("table_dz", 0.0)
            st[e, :3] = torch.tensor(pos, dtype=torch.float32, device=dev) + origins[e]
            st[e, 3:7] = torch.tensor(o["quat"], dtype=torch.float32, device=dev)
        asset.write_root_state_to_sim(st)
        if o["kind"] == "cap" and any("mass_scale" in pe or "cap_mass" in pe for pe in pert):
            # uniform density: the inertia scales with the mass (Isaac Lab's
            # randomize_rigid_body_mass(recompute_inertia=True) does the same)
            v = asset.root_physx_view
            m, inertia = v.get_masses().clone(), v.get_inertias().clone()
            for e in range(E):
                m1 = pert[e].get("cap_mass", float(m[e, 0])) * pert[e].get("mass_scale", 1.0)
                inertia[e] *= m1 / float(m[e, 0])
                m[e] = m1
            v.set_masses(m, torch.arange(E))
            v.set_inertias(inertia, torch.arange(E))
    # frictions: the old spec's one factor on everything but the table, and the MuJoCo
    # draws' factor per material group (the table's, "mat", is authored in post_spawn)
    group = {"robot": "finger", "cap": "cap", "mug": "mug"}
    if any("friction_scale" in pe or "friction_scales" in pe for pe in pert):
        for name, a in [("robot", robot)] + [(o["kind"], x) for o, x in zip(objs, assets)]:
            mat = a.root_physx_view.get_material_properties().clone()
            for e in range(E):
                mat[e, :, :2] *= pert[e].get("friction_scale", 1.0) * pert[e].get("friction_scales", {}).get(group[name], 1.0)
            a.root_physx_view.set_material_properties(mat, torch.arange(E))
    iscene.write_data_to_sim()
    sim.forward()
    robot.update(0.0)
    for a in assets:
        a.update(0.0)
    caps = [n for n in obj_names if n.startswith("cap_")]
    contacts = Contacts(sim, E, caps, "mug" in obj_names, dev)

    # settle: the objects come to rest under gravity while the arm is held exactly at the
    # recorded first frame (the MuJoCo track's settle_s), then the clock starts at t_0
    n_settle = int(round(ph["settle_s"] * hz)) if args.mode != "kinematic" else 0
    for _ in range(n_settle):
        hold_arm()
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        for a in assets:
            a.update(dt)
    hold_arm()
    robot.write_data_to_sim()
    sim.forward()
    robot.update(0.0)
    servo.reset(qa)

    rec = None
    if render:
        from .render import CameraRecorder

        rec = CameraRecorder(sc, cams, run, fps)
        for _ in range(args.warmup):
            sim.render()

    # --- the goal streams --------------------------------------------------------------------
    stream = SV.GoalStreamT(A, fps, model.dead_time, model.max_velocity, SV.firmware_limits(units), Sx[f0], dev)
    arm_goal = None
    if args.mode == "track":
        from .track import track_goals

        arm_goal = torch.tensor(track_goals(model, Sx[f0:f1 + 1], fps, sub, spawn_info["payload"]["inertial"]),
                                dtype=torch.float32, device=dev)

    O, L = len(objs), len(kinematics.LINKS)
    log = {k: np.zeros(s) for k, s in {
        "q": (T, E, 6), "finger": (T, E), "ctrl": (T, E, 6), "tau": (T, E, 6), "act": (T, E, 6), "lpos": (T, E, L, 3),
        "lquat": (T, E, L, 4), "opos": (T, E, O, 3), "oquat": (T, E, O, 4)}.items()}
    cont = {}

    def record(i: int, u_art, tau=None, act=None):
        o = origins.cpu().numpy()
        q = robot.data.joint_pos
        log["q"][i] = servo.encoder(q)[:, perm].cpu().numpy()  # the Jaw's is the HORN (the encoder)
        log["finger"][i] = q[:, perm[U.GRIPPER]].cpu().numpy()
        log["ctrl"][i] = u_art[:, perm].cpu().numpy()
        if tau is not None:
            log["tau"][i] = tau[:, perm].cpu().numpy()
            log["act"][i] = act[:, perm].cpu().numpy()
        log["lpos"][i] = robot.data.body_pos_w[:, link_idx].cpu().numpy() - o[:, None]
        log["lquat"][i] = robot.data.body_quat_w[:, link_idx].cpu().numpy()
        for j, a in enumerate(assets):
            log["opos"][i, :, j] = a.data.root_pos_w.cpu().numpy() - o
            log["oquat"][i, :, j] = a.data.root_quat_w.cpu().numpy()
        fr = contacts.frame()
        for k, v in list(contacts.frame_extras(fr).items()) + [(f"x_{k}", v) for k, v in fr.items()]:
            cont.setdefault(k, np.zeros((T,) + v.shape, dtype=v.dtype))[i] = v

    def grab():
        sim.render()
        for n in cams:
            c = iscene[f"cam_{n}"]
            c.update(0.0, force_recompute=True)
            rec.add(n, c.data.output["rgb"][0].cpu().numpy())

    t_loop = time.time()
    t_render = 0.0
    u = torch.zeros(E, len(names), device=dev)
    if args.mode == "kinematic":
        for i in range(T):
            k = f0 + i
            qa[:, perm] = torch.tensor(Sx_in[k], dtype=torch.float32, device=dev)
            hold_arm()
            robot.write_data_to_sim()
            sim.forward()
            robot.update(dt * sub)
            for a in assets:
                a.update(dt * sub)
            contacts.new_frame()
            servo.reset(qa)
            record(i, qa)
            if rec:
                t = time.time()
                grab()
                t_render += time.time() - t
    else:
        u[:] = qa
        contacts.new_frame()
        record(0, u)
        if rec:
            grab()
        for i in range(T - 1):
            k = f0 + i
            contacts.new_frame()
            u_frame = None
            for s_ in range(sub):
                u[:, perm] = stream.at((k + s_ / sub) / fps).float()
                if arm_goal is not None:
                    u[:, perm[:U.N_ARM]] = arm_goal[i * sub + s_]
                if u_frame is None:
                    u_frame = u.clone()
                q, qd = robot.data.joint_pos, robot.data.joint_vel
                tgt, vel, ff = servo.step(q, qd, u)
                robot.set_joint_position_target(tgt)
                robot.set_joint_velocity_target(vel)
                robot.set_joint_effort_target(ff)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(dt)
                for a in assets:
                    a.update(dt)
                contacts.accumulate(dt)
            tau = servo.torque(robot.data.joint_pos, robot.data.joint_vel, tgt, vel, ff)
            act = servo.actuator_force(robot.data.joint_pos, robot.data.joint_vel, u)
            record(i + 1, u_frame, tau, act)
            if rec:
                t = time.time()
                grab()
                t_render += time.time() - t
            if i % 100 == 0:
                print(f"  frame {k + 1}/{f1}  {time.time() - t_loop:.1f} s", flush=True)
    t_end = time.time()
    videos = rec.close() if rec else {}

    # --- write -------------------------------------------------------------------------------
    finite = bool(np.isfinite(log["q"]).all() and np.isfinite(log["opos"]).all())
    wall = round(t_end - t_loop, 2)
    meta_common = {
        "physical": args.mode != "kinematic", "finite": finite, "wall_s": wall,
        "servo": model.to_config(), "dead_time_s": model.dead_time, "flex": servo.flex, "hz": hz,
        "caps": caps, "t0": f0, "placement": args.placement, "settle_s": ph["settle_s"],
        "placements": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in o.items()} for o in objs],
        "params": prm, "num_envs": E, "readback": readback, "spawn": spawn_info, "cameras": list(cams),
        "scene_overrides": overrides, "scene_layers": [p.name for p in sc.layers],
        "contact_links": contacts.links, "contact_filters": contacts.obj_filters + ["table"],
        "q_jaw_is": "the HORN (what the encoder reads); x_finger is the PhysX Jaw joint, x_flex = finger - horn",
    }
    (run / "ensemble").mkdir(exist_ok=True)
    for e in range(E):
        lg = poselog.make(
            fps, np.arange(f0, f1 + 1), log["q"][:, e], log["ctrl"][:, e], kinematics.LINKS,
            log["lpos"][:, e], log["lquat"][:, e], obj_names, log["opos"][:, e], log["oquat"][:, e],
            meta=poselog.meta("isaac", args.mode, args.episode, sc.hash, dt=dt, substeps=sub, seed=args.seed,
                              perturbation=pert[e], **meta_common),
            x_finger=log["finger"][:, e], x_flex=log["finger"][:, e] - log["q"][:, e, U.GRIPPER],
            x_tau=log["tau"][:, e], x_actuator_force=log["act"][:, e], **{k: v[:, e] for k, v in cont.items()})
        poselog.write(run / ("poselog.npz" if e == 0 else f"ensemble/env_{e:03d}.npz"), lg)
    timings = {"boot_to_scene_s": round(t_built - t_start, 2), "reset_s": round(t_reset - t_built, 2),
               "loop_s": wall, "render_s": round(t_render, 2),
               "physics_steps": (T - 1) * sub if args.mode != "kinematic" else 0, "frames": T,
               "ms_per_frame": round(1000 * (t_end - t_loop) / max(T, 1), 2), "num_envs": E}
    info = {"args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, type(None)))},
            "scene_hash": sc.hash, "scene_layers": [str(p) for p in sc.layers], "dataset": sc.ds,
            "episode": args.episode, "mode": args.mode, "frames": [f0, f1], "timings": timings, "videos": videos,
            "perturbations": pert, "provenance": P.provenance(), **meta_common}
    (run / "run.json").write_text(json.dumps(info, indent=1, default=float))
    print(f"REPLAY {run} {json.dumps(timings)}", flush=True)


if __name__ == "__main__":
    _args = parse_args()
    from isaaclab.app import AppLauncher

    _args.headless = not _args.gui
    _args.enable_cameras = bool(_args.render or _args.gui)
    _app = AppLauncher(_args).app
    _code = 1
    try:
        main(_args)
        _code = 0
    except BaseException:  # noqa: BLE001 -- os._exit below would swallow it
        traceback.print_exc()
    finally:
        # Kit 4.5 hangs in app.close() on this box (Isaac report §7): flush by hand, exit hard.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_code)
