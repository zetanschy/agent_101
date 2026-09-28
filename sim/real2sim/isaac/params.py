"""Every Isaac-side number that is not in the scene config, with where it comes from.

The scene config (real2sim.scene) owns what the DATA determine: cameras, table
height, joint offsets, gripper map, object sizes and poses, and the servo model. This
module owns what only the Isaac stack needs: PhysX stepping and solver settings, the
contact materials, and the wrist payload PhysX must carry. Pure Python, so eval
scripts on the host read the same values the replay used (eval.json records them).

Provenance labels as in the scene config: measured / fitted / estimated / assumed /
spec / read. "measured" here means an Isaac run in this track (the experiments are
listed in isaac/README.md with their numbers); override any value from the command
line with --physics key=value and it lands in the pose log's meta.
"""

from __future__ import annotations

import copy

PHYSICS = {
    "hz": {
        "value": 480,
        "source": "measured (ep0 grasp, README 'physics settings'): 480 Hz = 16 steps per 1/30 s frame; "
                  "240 Hz dropped the cap 74 frames early (PhysX finger-cap separation -3.75 mm); 960 Hz "
                  "matches 480 Hz + 64 iterations (0.08 vs 0.10 mm) at 1.6x its cost",
    },
    "solver": {"value": "TGS", "source": "read: Isaac Lab default (PhysX 5 TGS solves friction every iteration)"},
    "robot_position_iterations": {
        "value": 64,
        "source": "measured: 32 (sim_agent101 SO101_CFG) left the squeezed 2.5 g cap 0.77 mm deep in the fingers "
                  "(PhysX -1.93 mm) and let it turn 16.5 deg in the grasp; 64 converges: 0.10 mm, 0.2 deg, "
                  "+26 % time",
    },
    "object_position_iterations": {
        "value": 64,
        "source": "assumed: equal to the arm's, so a finger-cap contact island is never solved with fewer",
    },
    "velocity_iterations": {"value": 1, "source": "read: PhysX guidance, one velocity iteration with TGS"},
    "contact_offset": {
        "value": None,
        "source": "measured: None = PhysX's own per-shape value (1.4-1.5 mm on the arm); 2 mm everywhere was "
                  "worse (0.78 mm, PhysX -2.40 mm) and 1.9x slower (more pairs)",
    },
    "rest_offset": {"value": 0.0, "source": "read: PhysX guidance, rest offset = graphics geometry (no gap, no overlap)"},
    "max_depenetration_velocity": {
        "value": 5.0,
        "source": "measured: 0.5 m/s kept the solver from pushing the squeezing fingers out of the cap "
                  "(0.77 mm, 16.5 deg slip); 5 m/s (sim_agent101 SO101_CFG's arm value): 0.18 mm, 0.3 deg. "
                  "It only bounds the speed an overlap is resolved at, and the overlaps here are < 0.2 mm",
    },
    "ccd": {"value": False, "source": "measured: sweep CCD on the objects changed nothing on ep0 "
                                      "(identical metrics) and cost +75 % time; drops land in the mug <= 0.28 mm deep"},
    "bounce_threshold": {"value": 0.2, "source": "assumed: m/s; irrelevant with restitution 0 (MATERIALS)"},
    "enhanced_determinism": {"value": False, "source": "provisional: Isaac Lab default (see README 'determinism')"},
    "settle_s": {"value": 0.3, "source": "read: the MuJoCo track's settle (replay.Config.settle_s): objects "
                                         "settle under gravity before frame 0, the arm held at state[0]"},
    "limit_margin_deg": {"value": 5.0, "source": "read: the MuJoCo track's LIMIT_MARGIN_DEG: joint range = "
                                                 "firmware range +- 5 deg (the firmware clamps the GOAL; the joint "
                                                 "has no hard stop there); the Jaw's lower limit is the finger touch"},
}

# Contact materials, the MuJoCo track's (model.MATERIALS, ESTIMATED from handbook dry-
# friction ranges) with PhysX told to combine them the way MuJoCo does: the element-wise
# MAX (frictionCombineMode "max"), so every PAIR has the same coefficient in both
# engines: finger-cap 0.40, cap-mat 0.60, cap-mug 0.30, mug-mat 0.60, finger-mat 0.60.
# Static = dynamic (MuJoCo has one Coulomb coefficient). Restitution 0: MuJoCo's contacts
# are critically damped (no bounce); a real cap on steel bounces a little, measured
# nowhere, so both engines leave it out. Ensembles perturb friction.
MATERIALS = {
    "robot": {"static_friction": 0.40, "dynamic_friction": 0.40, "restitution": 0.0, "friction_combine": "max",
              "restitution_combine": "max", "source": "estimated: PLA fingers on PP/HDPE 0.3-0.5 (MuJoCo finger)"},
    "cap": {"static_friction": 0.30, "dynamic_friction": 0.30, "restitution": 0.0, "friction_combine": "max",
            "restitution_combine": "max", "source": "estimated: PP/HDPE cap, its lowest pairing (steel) (MuJoCo cap)"},
    "mat": {"static_friction": 0.60, "dynamic_friction": 0.60, "restitution": 0.0, "friction_combine": "max",
            "restitution_combine": "max", "source": "estimated: black glossy vinyl/silicone mat 0.4-0.9 (MuJoCo mat)"},
    "mug": {"static_friction": 0.30, "dynamic_friction": 0.30, "restitution": 0.0, "friction_combine": "max",
            "restitution_combine": "max", "source": "estimated: enamel outside, polished steel inside 0.2-0.4 (MuJoCo mug)"},
}

# What the wrist carries beyond the URDF's gripper link (0.087 kg). The printed mount
# and the webcam were visual-only in every earlier sim (Isaac report §8, sim README:212).
PAYLOAD = {
    "mount_mass": {"value": 0.018, "source": "read: mjlab klip_camera.py _MOUNT_MASS (printed PLA bracket)"},
    "webcam_mass": {
        "value": 0.020,
        "range": [0.010, 0.074],
        "source": "estimated (the MuJoCo track's WEBCAM_MASS): the 30 x 26 x 16 mm head at ~1 g/cm^3 12.5 g, "
                  "lens barrel ~2 g, carried USB cable ~5 g; the KWC-500 spec's 73.7 g (2.6 oz, klipxtreme.com) "
                  "includes the monitor clip and all 1.5 m of cable, an upper bound",
    },
}

LOOK_DEFAULTS = {
    "cap": {"color": (0.41, 0.63, 0.62), "roughness": 0.45, "source": "measured: teal mask mean RGB (105,160,159)/255"},
    "mug": {"color": (0.62, 0.04, 0.04), "roughness": 0.25, "metallic": 0.0, "source": "assumed: red enamel"},
}


def defaults() -> dict:
    """A fresh, mutable copy: {'physics': {k: value}, 'materials': ..., 'payload': {k: value}}."""
    return {"physics": {k: v["value"] for k, v in PHYSICS.items()},
            "materials": {k: {kk: vv for kk, vv in v.items() if kk != "source"} for k, v in MATERIALS.items()},
            "payload": {k: v["value"] for k, v in PAYLOAD.items()}}


def override(params: dict, items) -> dict:
    """Apply 'section.key=value' or 'key=value' (physics) strings; values parsed as Python literals."""
    import ast

    out = copy.deepcopy(params)
    for it in items or ():
        key, _, val = it.partition("=")
        try:
            v = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            v = val
        parts = key.split(".")
        if len(parts) == 1:
            parts = ["physics"] + parts
        d = out
        for p in parts[:-1]:
            d = d[p]
        if parts[-1] not in d:
            raise KeyError(f"unknown parameter {key!r}")
        d[parts[-1]] = v
    return out


def provenance() -> dict:
    return {"physics": copy.deepcopy(PHYSICS), "materials": copy.deepcopy(MATERIALS), "payload": copy.deepcopy(PAYLOAD)}


def scene_with_overrides(sc, items):
    """--set path=value on the merged scene (e.g. table.z=0.0147, objects.cap.diameter=0.0242)
    for sensitivity runs: returns (new Scene with its own hash, {path: {was, now}})."""
    import ast

    from ..scene import Scene

    if not items:
        return sc, {}
    data = copy.deepcopy(dict(sc))
    done = {}
    for it in items:
        key, _, val = it.partition("=")
        v = ast.literal_eval(val)
        d = data
        parts = key.split(".")
        for p in parts[:-1]:
            d = d[p]
        if parts[-1] not in d:
            raise KeyError(f"--set {key}: no such scene key")
        done[key] = {"was": d[parts[-1]], "now": v}
        d[parts[-1]] = v
    return Scene(data, sc.ds, sc.layers, sc.origin), done
