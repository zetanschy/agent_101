"""The sample-frame set every renderer is scored on: 24 timesteps x both cameras.

SELECTION. Episodes 0-2, eight phases each: rest, approach, descend, grasp, lift,
carry, over_mug and release. Both cameras use the same timestep, so front and wrist
renders of one sample share the arm pose and the object poses. The phase windows
come from the scene's picks (inclusive episode-local frames, scene.py):

    rest      frames 3..30 (every episode opens at rest: episodes report)
    approach  close0 - 110 .. close0 - 60        descend   close0 - 30 .. close0 - 3
    grasp     close1 + 2 .. lift0 - 1 (settled)  lift      lift0 .. lift1
    carry     lift1 + 1 .. release0 - 1          over_mug  release0 - 8 .. release0
    release   drop0 .. drop1 + 6 (cap leaving the jaws / in the mug)

The multi-cap episodes spread the phases over their caps (PICK_PLAN), so that all
six picks, both cap orientations and all three mug positions are covered. Inside a
window the frame with the slowest arm wins. The follower's joint speed from
observation.state is penalised when that frame repeats the previous front image.
The C270 delivers duplicates, 87 % of fast-motion odd frames in episode 2, and the
front image already lags the state by 2-3 frames, so slow frames keep that lag's
pixel error small. Each sample records its speed and duplicate flags.

PER SAMPLE (samples/index.json): the real front and wrist images as delivered (PNG,
8-bit sRGB) and their label maps (PNG, codes in look/__init__). Also the state
(lerobot units), the front frame's white-balance gain against the LOOK reference
state (plates.py), the wrist frame's colour model (its own white balance from the
static finger, the exposure of the nearest mosaic frame, table.py, and sigma), the label
pixel counts, and QA numbers: the model-vs-refined robot IoU
in the front view, the model-vs-data finger IoU in the wrist view (a direct measure
of the wrist calibration), and the finger prior's gripper-% spread.
"""

from __future__ import annotations

import json

import numpy as np

from .. import episodes as E
from .. import objects
from . import LABELS, colour
from .segment import GripFingerPrior, LabelRenderer, front_labels, front_mug_mask, grip_labels

PHASES = ("rest", "approach", "descend", "grasp", "lift", "carry", "over_mug", "release")
# cap id per phase after rest, per episode (the order of PHASES[1:])
PICK_PLAN = {0: "AAAAAAA", 1: "ABCABCB", 2: "ABABABA"}


def _window(pick, phase, n):
    c0, c1 = pick["close"]
    l0, l1 = pick["lift"]
    r0 = pick["release"][0]
    d0, d1 = pick["drop"]
    w = {"approach": (c0 - 110, c0 - 60), "descend": (c0 - 30, c0 - 3), "grasp": (c1 + 2, l0 - 1),
         "lift": (l0, l1), "carry": (l1 + 1, r0 - 1), "over_mug": (r0 - 8, r0), "release": (d0, d1 + 6)}[phase]
    return max(w[0], 0), min(w[1], n - 1)


def select(scene) -> list[dict]:
    """[{id, episode, frame, row, phase, cap, speed_deg_s, front_dup, grip_dup}] (see module doc)."""
    out = []
    for e, ep in E.load(scene.ds).items():
        picks = {p["cap"]: p for p in scene.episode(e)["picks"]}
        st = ep.state.astype(float)
        speed = np.abs(np.gradient(st[:, :5], axis=0)).max(1) * ep.fps
        f, g = ep.frames("front"), ep.frames("grip")

        def dup(fr, k):
            return bool(k > 0 and np.array_equal(np.asarray(fr[k]), np.asarray(fr[k - 1])))

        plan = [("rest", None)] + list(zip(PHASES[1:], PICK_PLAN[e]))
        for phase, cap in plan:
            lo, hi = (3, 30) if phase == "rest" else _window(picks[cap], phase, len(ep))
            ks = np.arange(lo, hi + 1)
            score = speed[ks] + np.array([1000.0 * (dup(f, k) and speed[k] > 5) for k in ks])
            k = int(ks[np.argmin(score)])
            out.append({"id": f"e{e}_{k:04d}_{phase}", "episode": e, "frame": k, "row": ep.start + k,
                        "phase": phase, "cap": cap, "speed_deg_s": round(float(speed[k]), 2),
                        "front_dup": dup(f, k), "grip_dup": dup(g, k)})
    return out


def _nearest(model: dict, e: int, k: int):
    ks = [int(key.split(":")[1]) for key in model if int(key.split(":")[0]) == e]
    if not ks:
        return None, None
    kk = min(ks, key=lambda x: abs(x - k))
    return model[f"{e}:{kk}"], kk


def build(scene, renderer: LabelRenderer, plates, prior: GripFingerPrior, wrist_model: dict, sigma: float, L_pla,
          out_dir, log=print) -> dict:
    """Write samples/<id>_{front,grip}[_labels].png and samples/index.json."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    units = scene.units()
    eps = E.load(scene.ds)
    clean_u8 = colour.linear_to_srgb(np.nan_to_num(plates["clean_lin"]))
    mug_masks = {e: front_mug_mask(plates["mug_hull"][e], colour.linear_to_srgb(plates["plate_lin"][e]), clean_u8)
                 for e in eps}
    samples = select(scene)
    for s in samples:
        e, k = s["episode"], s["frame"]
        ep = eps[e]
        q = units.to_urdf(ep.state[k])
        renderer.set_mug(objects.mug_pose(scene, e))
        f_img, g_img = np.asarray(ep.frames("front")[k]), np.asarray(ep.frames("grip")[k])
        plate_u8 = colour.linear_to_srgb(plates["plate_lin"][e])
        f_lab, f_qa = front_labels(f_img, renderer.parts(q, "front"), plate_u8, mug_masks[e])
        w, spread = prior.prior(float(ep.state[k][5]))
        g_lab, g_qa = grip_labels(g_img, renderer.parts(q, "grip"), w, spread)
        near, near_k = _nearest(wrist_model, e, k)
        wb = colour.grip_white_balance(g_img, prior.static_core(), L_pla)
        grip_model = {"white_balance": None if wb is None else np.round(wb, 4).tolist(), "sigma": sigma,
                      "exposure": None if near is None else near["exposure"], "exposure_from_frame": near_k,
                      "convention": "front_lin = colour.grip_to_front(grip_lin, white_balance, exposure, sigma, L_PLA)"}
        entry = {"state": np.round(ep.state[k].astype(float), 4).tolist()}
        for cam, img, lab, qa, extra in (
                ("front", f_img, f_lab, f_qa, {"white_balance_gain": np.round(plates["frame_gain"][ep.start + k], 4).tolist()}),
                ("grip", g_img, g_lab, g_qa, {"colour_model": grip_model})):
            cv2.imwrite(str(out_dir / f"{s['id']}_{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{s['id']}_{cam}_labels.png"), lab)
            entry[cam] = {"image": f"{s['id']}_{cam}.png", "labels": f"{s['id']}_{cam}_labels.png",
                          "label_px": {n: int((lab == v).sum()) for n, v in LABELS.items()}, "qa": qa, **extra}
        s.update(entry)
    renderer.set_mug(None)
    iou_f = [s["front"]["qa"]["robot_model_vs_refined_iou"] for s in samples]
    iou_g = [s["grip"]["qa"]["fingers_model_vs_data_iou"] for s in samples]
    summary = {"n": len(samples), "phases": list(PHASES), "episodes": sorted(eps),
               "front_robot_model_vs_refined_iou_median": round(float(np.median(iou_f)), 4),
               "grip_fingers_model_vs_data_iou_median": round(float(np.median(iou_g)), 4),
               "grip_fingers_model_vs_data_iou_min_max": [round(float(min(iou_g)), 4), round(float(max(iou_g)), 4)],
               "front_duplicates": int(sum(s["front_dup"] for s in samples)),
               "speed_deg_s_median_max": [round(float(np.median([s["speed_deg_s"] for s in samples])), 2),
                                          round(float(max(s["speed_deg_s"] for s in samples)), 2)]}
    index = {"schema": "real2sim.look.samples/1", "dataset": scene.ds, "scene_hash": scene.hash,
             "labels": LABELS, "cameras": {c: scene.camera(c).to_config() for c in ("front", "grip")},
             "summary": summary, "samples": samples}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    log(f"  samples: {len(samples)} x 2 cameras; front robot IoU (model vs refined) median {summary['front_robot_model_vs_refined_iou_median']}, "
        f"grip fingers IoU (model vs data) median {summary['grip_fingers_model_vs_data_iou_median']}")
    return index
