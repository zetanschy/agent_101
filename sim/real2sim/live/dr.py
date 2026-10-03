"""Domain randomization for the live MuJoCo sim: a fresh draw with every new layout.

    ./robot record --sim mujoco --sim-dr all --name cap_to_mug_dr --episodes 50
    ./robot teleop --sim mujoco --sim-dr visual
    ./robot sim-eval --policy /checkpoints/X --sim-dr all      # score under the same randomization

The sim draws a new layout before every recorded episode (lerobot_plugin), at every
Reset, and on `live reset`. With --dr it also draws new appearance and physics at the same
moment. So each demonstration is taught under different conditions, and a policy can't
learn to lean on any one of them. The values are drawn from the RANGES below. They are
centred on the scene's fitted values (calib, servo, look), so the identified sim is the
middle of the distribution, and wide enough to take in what the real arm differs by
(the real cap, for one, is a more saturated cyan than the fitted teal).

This is the collect-under-randomization strategy (NVIDIA's sim-to-real SO-101 workshop,
strategy 1): you teleoperate while the conditions change, so the demonstrations show how
to cope with each one. Randomizing a recorded dataset afterwards could not do that for
the physics: the recorded actions were the answer to the old dynamics.

Levels: off (default: the identified sim, bitwise the replay), visual, physics, all.
Every draw is printed by the server and appended to sim/outputs/real2sim/live/dr_log.jsonl
(wall time, layout, values), in recording order. A seeded reset (sim-eval) seeds the draw
too, so stage i is the same randomized stage for every policy.
"""

from __future__ import annotations

import colorsys
import json
import time
from pathlib import Path

import numpy as np

LEVELS = ("off", "visual", "physics", "all")

# One place to widen or narrow it. (lo, hi) = uniform; a single number = +- that.
RANGES = {
    # --- visual -----------------------------------------------------------------------
    "light_gain": (0.6, 1.4),        # key light brightness
    "light_tilt_deg": 15.0,          # key light direction, within this cone of the fitted one
    "light_tint": 0.10,              # warm/cool colour temperature, +- per channel
    "ambient_gain": (0.6, 1.4),      # the fill (headlight ambient)
    "mat_gain": (0.75, 1.25),        # the mat texture's brightness
    "cap_hue_deg": (172.0, 192.0),   # fitted teal: 179 deg, sat 0.34, val 0.63;
    "cap_sat": (0.25, 0.85),         # the real cap reads ~185 deg, sat ~0.8 on the arm
    "cap_val": (0.55, 0.85),
    "mug_gain": (0.75, 1.2),         # the red enamel
    "arm_gain": (0.85, 1.05),        # the printed (white) parts
    "front_cam_mm": 8.0,             # overhead camera mount: shift per axis
    "front_cam_deg": 1.5,            #   and tilt
    "grip_cam_mm": 2.0,              # wrist camera on its clip
    "grip_cam_deg": 2.0,
    # --- physics ----------------------------------------------------------------------
    "kp_gain": (0.8, 1.2),           # servo stiffness
    "effort_gain": (0.85, 1.15),     # the torque clamp
    "damping_gain": (0.7, 1.4),      # joint viscous damping
    "friction_gain": (0.7, 2.0),     # joint dry friction (stiction): small corrections stall
    "dead_time_frames": (1.0, 3.0),  # bus + firmware latency (fitted: 1 frame)
    "deadband_deg": (0.0, 0.35),     # servo dead zone: 0-4 encoder ticks, no torque inside it
    "sensor_bias_deg": 1.0,          # a calibration offset per joint, held for the episode
    "sensor_noise_deg": 0.10,        # read noise, every observation
    "contact_friction_gain": (0.7, 1.3),  # cap, fingers, mat, mug: slide coefficient
    "cap_mass_gain": (0.6, 1.6),
    "mug_mass_gain": (0.8, 1.25),
}


def _u(rng, key, n=None):
    r = RANGES[key]
    lo, hi = (r if isinstance(r, tuple) else (-r, r))
    return float(rng.uniform(lo, hi)) if n is None else rng.uniform(lo, hi, n).round(5).tolist()


def draw(level: str, rng: np.random.Generator) -> dict:
    """One episode's values (JSON-serializable). Empty for 'off'."""
    if level not in LEVELS:
        raise ValueError(f"--dr {level!r}: expected one of {LEVELS}")
    p: dict = {"level": level}
    if level in ("visual", "all"):
        h, s, v = _u(rng, "cap_hue_deg") / 360, _u(rng, "cap_sat"), _u(rng, "cap_val")
        p["visual"] = {
            "light_gain": _u(rng, "light_gain"), "light_tilt": _u(rng, "light_tilt_deg", 2),
            "light_tint": _u(rng, "light_tint", 3), "ambient_gain": _u(rng, "ambient_gain"),
            "mat_gain": _u(rng, "mat_gain"), "cap_rgb": [round(c, 4) for c in colorsys.hsv_to_rgb(h, s, v)],
            "mug_gain": _u(rng, "mug_gain"), "arm_gain": _u(rng, "arm_gain"),
            "front_cam_mm": _u(rng, "front_cam_mm", 3), "front_cam_deg": _u(rng, "front_cam_deg", 3),
            "grip_cam_mm": _u(rng, "grip_cam_mm", 3), "grip_cam_deg": _u(rng, "grip_cam_deg", 3),
        }
    if level in ("physics", "all"):
        p["physics"] = {
            "kp_gain": _u(rng, "kp_gain", 6), "effort_gain": _u(rng, "effort_gain", 6),
            "damping_gain": _u(rng, "damping_gain", 6), "friction_gain": _u(rng, "friction_gain", 6),
            "dead_time_frames": _u(rng, "dead_time_frames"), "deadband_deg": _u(rng, "deadband_deg", 6),
            "sensor_bias_deg": _u(rng, "sensor_bias_deg", 6), "sensor_noise_deg": RANGES["sensor_noise_deg"],
            "contact_friction_gain": {k: _u(rng, "contact_friction_gain") for k in ("cap", "finger", "mat", "mug")},
            "cap_mass_gain": _u(rng, "cap_mass_gain"), "mug_mass_gain": _u(rng, "mug_mass_gain"),
        }
    return p


def summary(p: dict) -> str:
    out = [f"dr {p.get('level')}"]
    if "visual" in p:
        v = p["visual"]
        out.append(f"light x{v['light_gain']:.2f} ambient x{v['ambient_gain']:.2f} mat x{v['mat_gain']:.2f} "
                   f"cap rgb {tuple(round(c * 255) for c in v['cap_rgb'])}")
    if "physics" in p:
        f = p["physics"]
        out.append(f"kp x{min(f['kp_gain']):.2f}-{max(f['kp_gain']):.2f} stiction x{min(f['friction_gain']):.2f}-"
                   f"{max(f['friction_gain']):.2f} dead time {f['dead_time_frames']:.1f} fr "
                   f"deadband <= {max(f['deadband_deg']):.2f} deg cap mass x{f['cap_mass_gain']:.2f}")
    return ", ".join(out)


def log(path: Path, p: dict, layout: dict, seed) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps({"wall": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "layout": layout, "dr": p}) + "\n")


# --- applying a draw to a built MuJoCo model ----------------------------------------------
def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _small_rot(deg3) -> np.ndarray:
    v = np.radians(np.asarray(deg3, float))
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return np.r_[np.cos(a / 2), np.sin(a / 2) * v / a]


class Applier:
    """Sets a draw on a built model, always from the model's own NOMINAL values (captured
    the first time it sees that model), so draws never compound across episodes."""

    def __init__(self, b):
        import mujoco

        from ..mujoco import model as mdl
        from ..units import URDF_JOINTS

        m, self.b, self.mj = b.model, b, mujoco
        name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or ""  # noqa: E731
        G = mujoco.mjtObj.mjOBJ_GEOM
        geoms = {name(G, i): i for i in range(m.ngeom)}
        self.key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, "key")
        self.mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "look_table")
        self.mat_geom = geoms.get("mat_visual")
        self.caps = [geoms[f"{o['name']}_visual"] for o in b.objects if o["kind"] == "cap" and f"{o['name']}_visual" in geoms]
        self.mug_geoms = [geoms[g] for g in ("mug_body_visual", "mug_handle_visual") if g in geoms]
        printed = np.array(mdl.COLORS["printed"][:3])
        self.arm_geoms = [i for i in range(m.ngeom) if m.geom_group[i] == mdl.VISUAL_GROUP
                          and np.allclose(m.geom_rgba[i, :3], printed, atol=1e-3)]
        self.cams = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in ("front", "grip")}
        self.dofs = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in URDF_JOINTS]
        self.act = list(b.act)
        fingers = {int(g) for ids in b.finger_geoms.values() for g in ids}  # geom ids
        by_mat = {"cap": [], "finger": [], "mat": [], "mug": []}
        for nm, i in geoms.items():
            if m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0:
                continue  # visual only
            k = ("finger" if i in fingers or "_finger" in nm else "cap" if nm.startswith("cap")
                 else "mug" if nm.startswith("mug") else "mat" if nm == "table" else None)
            if k:
                by_mat[k].append(i)
        self.contacts = by_mat
        self.bodies = {"cap": [o["body"] for o in b.objects if o["kind"] == "cap" and "body" in o],
                       "mug": [o["body"] for o in b.objects if o["kind"] == "mug" and "body" in o]}
        self.nom = {
            "light_diffuse": m.light_diffuse.copy(), "light_dir": m.light_dir.copy(),
            "ambient": m.vis.headlight.ambient.copy(), "mat_rgba": m.mat_rgba.copy(), "geom_rgba": m.geom_rgba.copy(),
            "cam_pos": m.cam_pos.copy(), "cam_quat": m.cam_quat.copy(),
            "gainprm": m.actuator_gainprm.copy(), "biasprm": m.actuator_biasprm.copy(),
            "forcerange": m.actuator_forcerange.copy(), "damping": m.dof_damping.copy(),
            "frictionloss": m.dof_frictionloss.copy(), "friction": m.geom_friction.copy(),
            "mass": m.body_mass.copy(), "inertia": m.body_inertia.copy(),
        }

    def apply(self, p: dict) -> dict:
        """Returns the engine-side extras: {dead_time, deadband (rad, 6), bias (rad, 6), noise (rad)}."""
        m, n = self.b.model, self.nom
        for k, attr in (("light_diffuse", m.light_diffuse), ("light_dir", m.light_dir), ("mat_rgba", m.mat_rgba),
                        ("geom_rgba", m.geom_rgba), ("cam_pos", m.cam_pos), ("cam_quat", m.cam_quat),
                        ("gainprm", m.actuator_gainprm), ("biasprm", m.actuator_biasprm),
                        ("forcerange", m.actuator_forcerange), ("damping", m.dof_damping),
                        ("frictionloss", m.dof_frictionloss), ("friction", m.geom_friction), ("mass", m.body_mass),
                        ("inertia", m.body_inertia)):
            attr[:] = n[k]
        m.vis.headlight.ambient[:] = n["ambient"]
        extra = {"dead_time": None, "deadband": None, "bias": None, "noise": 0.0}
        v = p.get("visual")
        if v:
            if self.key >= 0:
                m.light_diffuse[self.key] = np.clip(n["light_diffuse"][self.key] * v["light_gain"]
                                                    * (1 + np.asarray(v["light_tint"])), 0, 2)
                d = n["light_dir"][self.key]
                q = _small_rot([v["light_tilt"][0], v["light_tilt"][1], 0.0])
                d2 = _quat_mul(_quat_mul(q, np.r_[0, d]), q * [1, -1, -1, -1])[1:]
                m.light_dir[self.key] = d2 / np.linalg.norm(d2)
            m.vis.headlight.ambient[:] = n["ambient"] * v["ambient_gain"]
            if self.mat >= 0:
                m.mat_rgba[self.mat, :3] = np.clip(n["mat_rgba"][self.mat, :3] * v["mat_gain"], 0, 1)
            if self.mat_geom is not None:
                m.geom_rgba[self.mat_geom, :3] = np.clip(n["geom_rgba"][self.mat_geom, :3] * v["mat_gain"], 0, 1)
            for i in self.caps:
                m.geom_rgba[i, :3] = v["cap_rgb"]
            for i in self.mug_geoms:
                m.geom_rgba[i, :3] = np.clip(n["geom_rgba"][i, :3] * v["mug_gain"], 0, 1)
            for i in self.arm_geoms:
                m.geom_rgba[i, :3] = np.clip(n["geom_rgba"][i, :3] * v["arm_gain"], 0, 1)
            for c, (mm, dg) in (("front", ("front_cam_mm", "front_cam_deg")), ("grip", ("grip_cam_mm", "grip_cam_deg"))):
                cid = self.cams[c]
                if cid >= 0:
                    m.cam_pos[cid] = n["cam_pos"][cid] + np.asarray(v[mm]) / 1000
                    m.cam_quat[cid] = _quat_mul(n["cam_quat"][cid], _small_rot(v[dg]))
        f = p.get("physics")
        if f:
            for j, a in enumerate(self.act):
                kp = n["gainprm"][a, 0] * f["kp_gain"][j]
                m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1] = kp, -kp
                m.actuator_forcerange[a] = n["forcerange"][a] * f["effort_gain"][j]
            for j, dof in enumerate(self.dofs):
                m.dof_damping[dof] = n["damping"][dof] * f["damping_gain"][j]
                m.dof_frictionloss[dof] = n["frictionloss"][dof] * f["friction_gain"][j]
            for k, ids in self.contacts.items():
                for i in ids:
                    m.geom_friction[i, 0] = n["friction"][i, 0] * f["contact_friction_gain"][k]
            for k, gain in (("cap", f["cap_mass_gain"]), ("mug", f["mug_mass_gain"])):
                for bid in self.bodies[k]:
                    m.body_mass[bid], m.body_inertia[bid] = n["mass"][bid] * gain, n["inertia"][bid] * gain
            extra.update(dead_time=f["dead_time_frames"], deadband=np.radians(f["deadband_deg"]),
                         bias=np.radians(f["sensor_bias_deg"]), noise=float(np.radians(f["sensor_noise_deg"])))
        return extra
