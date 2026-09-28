"""`./robot real2sim look build`: every LOOK asset, from the dataset, in one pass (~3 min).

    plates -> lighting -> white reference (front) -> table (front + wrist mosaic)
           -> wrist colours -> albedo -> backdrop + environment -> samples -> baselines

Everything lands in sim/outputs/real2sim/<ds>/look/, indexed by look.json. The
cameras, table height, joint offsets and object poses come from the merged scene
(scene.load), so re-run this after CALIB updates <ds>.calib.json. look.json records
the scene hash it was built from, and `info` warns when that hash is stale.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time

import numpy as np

from .. import episodes as E
from .. import paths
from ..poselog import git_describe
from ..scene import content_hash
from ..scene import load as load_scene
from . import LABEL_RGB, LOOK_VERSION, backdrop, colour, imagemetrics, lighting, materials, plates, samples, table
from .segment import GripFingerPrior, LabelRenderer


# The scene groups LOOK reads. Their hash (look.json inputs_hash) decides staleness,
# so a new servo layer from MUJOCO does not force a rebuild, while any new camera,
# table height, joint offset or object pose from CALIB does.
LOOK_INPUTS = ("cameras", "table", "robot", "objects", "episodes")


def inputs_hash(scene) -> str:
    return content_hash({k: scene[k] for k in LOOK_INPUTS if k in scene})


def _u8(lin):
    return colour.linear_to_srgb(np.nan_to_num(lin))


def _write_rgb(path, rgb_u8) -> str:
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2BGR))
    return path.name


def _write_png16_srgb(path, lin) -> str:
    """Linear 0..1 -> sRGB-encoded 16-bit PNG: the dark mat keeps its precision
    (8-bit sRGB has ~6 codes between albedo 0.005 and 0.02)."""
    import cv2

    y = colour.linear_to_srgb(lin, u8=False)
    cv2.imwrite(str(path), cv2.cvtColor(np.rint(y * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
    return path.name


def _write_plates(pl, out) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for e in pl["episodes"]:
        g = pl["gain_episode"][e]
        full = np.where(pl["valid"][e][..., None], pl["plate_lin"][e], np.nan_to_num(pl["clean_lin"]) * g)
        files[f"front_ep{e}"] = _write_rgb(out / f"front_ep{e}.png", _u8(full))
    files["front_clean"] = _write_rgb(out / "front_clean.png", _u8(pl["clean_lin"]))
    import cv2

    cv2.imwrite(str(out / "front_clean_valid.png"), pl["clean_valid"].astype(np.uint8) * 255)
    np.savez_compressed(out / "plates.npz", clean_lin=np.nan_to_num(pl["clean_lin"]).astype(np.float16),
                        clean_valid=pl["clean_valid"], frame_gain=pl["frame_gain"],
                        episodes=np.array(pl["episodes"]),
                        gain_episode=np.array([pl["gain_episode"][e] for e in pl["episodes"]]),
                        valid=np.array([pl["valid"][e] for e in pl["episodes"]]),
                        mug_hull=np.array([pl["mug_hull"][e] for e in pl["episodes"]]),
                        shadow=np.array([pl["shadow"][e] for e in pl["episodes"]]),
                        shadow_ratio=np.array([np.nan_to_num(pl["shadow_ratio"][e], nan=-1) for e in pl["episodes"]]).astype(np.float16))
    files.update({"clean_valid": "front_clean_valid.png", "arrays": "plates.npz"})
    return files


def _preview(path, tiles, width=1600):
    """Contact sheet of RGB uint8 tiles (list of rows)."""
    import cv2

    rows = []
    for row in tiles:
        h = min(t.shape[0] for t in row)
        row = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in row]
        rows.append(np.concatenate(row, 1))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.concatenate(rows, 0)
    s = width / sheet.shape[1]
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(sheet, None, fx=s, fy=s), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def _overlay(img, lab, alpha=0.45):
    col = np.zeros_like(img)
    for v, c in LABEL_RGB.items():
        col[lab == v] = c
    m = (lab > 0)[..., None]
    return np.where(m, (img * (1 - alpha) + col * alpha).astype(np.uint8), img)


def baseline_plate_scores(index, sdir, pl) -> dict:
    """What a PERFECT static background scores: each front sample against its own
    episode plate (arm-free, current white balance). An upper bound for the
    background region; the arm regions show what 'missing arm' costs."""
    def ren(s, cam):
        if cam != "front":
            return None
        e = s["episode"]
        g = np.asarray(s["front"]["white_balance_gain"], np.float32) / pl["gain_episode"][e]
        return _u8(np.where(pl["valid"][e][..., None], pl["plate_lin"][e],
                            np.nan_to_num(pl["clean_lin"]) * pl["gain_episode"][e]) * g)

    res = imagemetrics.score(None, index["dataset"], renders=ren)
    return {"front": res["summary"]["front"]}


def run(ds=None, log=print) -> dict:
    t_all = time.time()
    scene = load_scene(ds)
    out = paths.out_dir(scene.ds, "look", create=True)
    T = {}
    log(f"look build: {scene.ds}, scene hash {scene.hash} -> {out}")
    rnd = LabelRenderer(scene)
    try:
        return _run(scene, out, rnd, T, t_all, log)
    finally:
        rnd.close()


def _run(scene, out, rnd, T, t_all, log) -> dict:

    t = time.time()
    pl = plates.build(scene, rnd, log)
    plate_files = _write_plates(pl, out / "plates")
    T["plates"] = time.time() - t

    t = time.time()
    light = lighting.fit_key_light(pl, scene, log)
    gloss = lighting.fit_gloss(pl, scene, light, log=log)
    T["lighting"] = time.time() - t

    t = time.time()
    z = E.npz(scene.ds)
    prior = GripFingerPrior(E.frames(scene.ds, "grip"), z["state"][:, 5])
    log(f"  finger prior: {prior.n_frames} dark-background wrist frames of {prior.n_scanned} scanned")
    mf = materials.front(scene, rnd, pl, light, log)
    L_pla = mf["L_PLA_lin"]
    cm = materials.grip_colour_model(scene, prior, mf, log)
    sigma = cm["sigma"]
    T["white_reference"] = time.time() - t

    t = time.time()
    tab = table.build(scene, pl, light, gloss, rnd, prior, L_pla, sigma, log)
    mg = materials.grip(scene, prior, tab["wrist"]["model"], sigma, L_pla, log)
    alb, alb_stats = table.albedo(tab, L_pla, materials.RHO_PLA)
    tdir = out / "table"
    tdir.mkdir(exist_ok=True)
    import cv2

    tfiles = {"radiance": _write_rgb(tdir / "radiance.png", _u8(tab["radiance"])),
              "albedo": _write_png16_srgb(tdir / "albedo.png", alb),
              "albedo_preview": _write_rgb(tdir / "albedo_preview.png", _u8(np.clip(alb * 8, 0, 1)))}
    np.save(tdir / "radiance_lin.npy", tab["radiance"].astype(np.float16))
    np.save(tdir / "albedo_lin.npy", alb.astype(np.float16))
    np.save(tdir / "specular_front_lin.npy", tab["specular"].astype(np.float16))
    cv2.imwrite(str(tdir / "source.png"), tab["source"])
    (tdir / "wrist_model.json").write_text(json.dumps({
        "convention": "front_lin = real2sim.look.colour.grip_to_front(grip_lin, white_balance, exposure, sigma, L_PLA); "
                      "key 'episode:frame' (episode-local)", "sigma": sigma, "L_PLA_lin": L_pla,
        "frames": tab["wrist"]["model"], "mosaic_frames": tab["wrist"]["frames"]}))
    tfiles.update({"radiance_lin": "radiance_lin.npy", "albedo_lin": "albedo_lin.npy",
                   "specular_front_lin": "specular_front_lin.npy", "source": "source.png",
                   "wrist_model": "wrist_model.json"})
    T["table"] = time.time() - t

    t = time.time()
    to_scene = (materials.RHO_PLA / (np.pi * np.asarray(L_pla))).tolist()
    cyl = backdrop.cylinder(scene, rnd, prior, tab["wrist"]["model"], L_pla, sigma, log)
    env, seen, env_info = backdrop.env_map(scene, tab["radiance"], tab["fine"], cyl, to_scene,
                                           light["ambient_fraction"]["median"])
    bdir = out / "backdrop"
    bdir.mkdir(exist_ok=True)
    cyl_scene = np.nan_to_num(cyl["tex"]) * np.asarray(to_scene, np.float32)
    np.save(bdir / "cylinder_lin.npy", cyl_scene.astype(np.float16))
    bfiles = {"cylinder_preview": _write_rgb(bdir / "cylinder_preview.png", _u8(np.nan_to_num(cyl["tex"]))),
              "cylinder_lin": "cylinder_lin.npy", "env_hdr": "env.hdr",
              "env_preview": _write_rgb(bdir / "env_preview.png", _u8(env / (2 * env_info["fill_radiance"] + 1e-6)))}
    cv2.imwrite(str(bdir / "cylinder_seen.png"), cyl["ok"].astype(np.uint8) * 255)
    cv2.imwrite(str(bdir / "env.hdr"), cv2.cvtColor(env.astype(np.float32), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(bdir / "env_seen.png"), seen.astype(np.uint8) * 255)
    bfiles.update({"cylinder_seen": "cylinder_seen.png", "env_seen": "env_seen.png"})
    T["backdrop"] = time.time() - t

    t = time.time()
    index = samples.build(scene, rnd, pl, prior, tab["wrist"]["model"], sigma, L_pla, out / "samples", log)
    T["samples"] = time.time() - t
    t = time.time()
    base = baseline_plate_scores(index, out / "samples", pl)
    T["baselines"] = time.time() - t

    mats = materials.assemble(mf, mg, alb_stats, tab["summary"], light, gloss)
    a = light["ambient_fraction"]["median"]
    look = {
        "schema": "real2sim.look/1", "look_version": LOOK_VERSION, "dataset": scene.ds, "scene_hash": scene.hash,
        "inputs_hash": inputs_hash(scene), "inputs": list(LOOK_INPUTS),
        "scene_layers": [p.name for p in scene.layers],
        "built": datetime.datetime.now().isoformat(timespec="seconds"), "git": git_describe(),
        "scene_units": "total horizontal irradiance at the table = 1; a horizontal Lambertian face of albedo rho has "
                       "radiance rho / pi. Camera-linear = exposure * radiance, per channel.",
        "cameras": {
            "front": {"response": "sRGB transfer (assumed; colour.py)",
                      "exposure_scene_to_camera_lin": np.round(np.pi * np.asarray(L_pla) / materials.RHO_PLA, 5).tolist(),
                      "white_reference": {"L_PLA_lin": L_pla, "rho_PLA": materials.RHO_PLA,
                                          "per_frame_luminance_p5_p50_p95": mf["L_PLA_luminance_per_frame_p5_p50_p95"],
                                          "source": "measured: FK printed parts, white, world normal within 26 deg of +z, rest "
                                                    "frames of ep0-2, 60-90th luminance band; rho assumed"},
                      "white_balance": {"gain_episode": pl["summary"]["gain_episode"],
                                        "frame_gain_std": pl["summary"]["frame_gain_std"],
                                        "frame_gain_p5_p95": pl["summary"]["frame_gain_p5_p95"],
                                        "per_frame": "plates/plates.npz frame_gain (N, 3): camera = reference x gain",
                                        "source": "measured: background pixels vs the episode / clean plates (plates.py)"}},
            "grip": {"response": "sRGB transfer (assumed)",
                     "model": "front_lin = colour.grip_to_front(grip_lin, white_balance, exposure, sigma, L_PLA); "
                              "a render in front-reference units becomes the wrist camera's via colour.front_to_grip",
                     "sigma": sigma, "colour_model_fit": cm,
                     "white_balance": {"p5_p50_p95": tab["wrist"]["summary"]["white_balance_p5_p50_p95"],
                                       "source": "measured: per frame, the static finger core (white PLA) to the front "
                                                 "white's chromaticity"},
                     "exposure": {"p5_p50_p95": tab["wrist"]["summary"]["exposure_p5_p50_p95"],
                                  "source": "measured: per mosaic frame, plain-mat luminance vs the front texture"},
                     "per_frame": "table/wrist_model.json; per sample: samples/index.json grip.colour_model"}},
        "lights": {
            "key": {**light, "normal_irradiance_scene_units": round((1 - a) / np.sin(np.radians(light["elevation_deg"])), 5),
                    "source": "fitted: the four mug shadows (lighting.fit_key_light)"},
            "ambient": {"horizontal_irradiance_scene_units": round(a, 5), "uniform_sky_radiance": round(a / np.pi, 5),
                        "environment_map": "backdrop/env.hdr carries it (fill radiance where unobserved)",
                        "source": "fitted: umbra depth of the mug shadows (median of 4 episodes)"},
            "mat_gloss": {**{str(k): {kk: vv for kk, vv in v.items() if kk != "rms_by_alpha"} for k, v in gloss.items()},
                          "source": "fitted: GGX lobe of the front glare (lighting.fit_gloss)"},
        },
        "materials": mats,
        "materials_measured": {"front": mf, "grip": mg, "table_albedo": alb_stats},
        "textures": {
            "table": {**tab["summary"]["grid_fine"], "files": tfiles, "units": {
                "radiance": "front camera-linear, reference white-balance state (sRGB-encoded PNG / linear npy)",
                "albedo": "linear albedo (16-bit sRGB-encoded PNG / linear npy)",
                "source.png": "0 filled, 1 front camera, 2 wrist mosaic"}, "summary": tab["summary"]},
            "backdrop_cylinder": {"centre_xy": list(backdrop.CENTRE_XY), "radius": backdrop.RADIUS,
                                  "z0": scene.table_z(), "height": backdrop.HEIGHT, "res": backdrop.RES_CYL,
                                  "shape": list(cyl["tex"].shape[:2]), "units": "scene radiance (cylinder_lin.npy)",
                                  "layout": "col = azimuth phi = atan2(y - cy, x - cx) from -pi to +pi; row 0 = top",
                                  "files": bfiles, "summary": cyl["summary"]},
            "environment": {**env_info, "file": "backdrop/env.hdr", "units": "scene radiance, key light excluded"},
        },
        "plates": {"files": plate_files, "summary": pl["summary"]},
        "samples": {"index": "samples/index.json", "summary": index["summary"]},
        "baselines": {"front_episode_plate_as_render": base,
                      "note": "a perfect static background (the arm-free plate at the frame's white balance): the "
                              "background row is the ceiling a renderer's background can reach; the arm/fingers rows "
                              "are what a missing arm costs"},
        "seconds": {k: round(v, 1) for k, v in T.items()},
    }
    look["seconds"]["total"] = round(time.time() - t_all, 1)
    (out / "look.json").write_text(json.dumps(look, indent=1, default=_json_default))
    _previews(out, pl, tab, alb, cyl, env / (2 * env_info["fill_radiance"] + 1e-6), index)
    log(f"look build done in {look['seconds']['total']} s -> {out / 'look.json'}")
    return look


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _previews(out, pl, tab, alb, cyl, env, index):
    import cv2

    pdir = out / "previews"
    pdir.mkdir(exist_ok=True)
    tiles = [[_u8(np.nan_to_num(pl["plate_lin"][e])) for e in pl["episodes"][:2]],
             [_u8(np.nan_to_num(pl["plate_lin"][e])) for e in pl["episodes"][2:4]] + [_u8(pl["clean_lin"])]]
    _preview(pdir / "plates.jpg", tiles)
    src = np.stack([tab["source"] == 1, tab["source"] == 2, tab["source"] == 0], -1).astype(np.uint8) * 160
    _preview(pdir / "table.jpg", [[_u8(tab["radiance"] * 3), _u8(np.clip(alb * 8, 0, 1)), src]])
    _preview(pdir / "backdrop.jpg", [[_u8(np.nan_to_num(cyl["tex"]))], [_u8(env)]])
    sdir = out / "samples"
    rows = []
    for i in range(0, len(index["samples"]), 4):
        row = []
        for s in index["samples"][i:i + 4]:
            for cam in ("front", "grip"):
                img = cv2.cvtColor(cv2.imread(str(sdir / s[cam]["image"])), cv2.COLOR_BGR2RGB)
                lab = cv2.imread(str(sdir / s[cam]["labels"]), cv2.IMREAD_UNCHANGED)
                row.append(_overlay(img, lab))
        rows.append(row)
    _preview(pdir / "samples.jpg", rows, width=2400)


def info(ds=None) -> str:
    """Summary of look.json, and whether it is stale against the current scene."""
    p = paths.out_dir(ds, "look", "look.json")
    if not p.exists():
        return f"{p} missing: run ./robot real2sim look build"
    lk = json.loads(p.read_text())
    sc = load_scene(ds)
    k = lk["lights"]["key"]
    lines = [f"look {lk['dataset']} built {lk['built']} ({lk['git']}), {lk['seconds'].get('total')} s",
             f"inputs hash {lk['inputs_hash']} (layers {', '.join(lk['scene_layers'])})" +
             ("  up to date" if lk["inputs_hash"] == inputs_hash(sc) else
              f"  STALE: the scene's {'/'.join(LOOK_INPUTS)} changed (now {inputs_hash(sc)}); re-run look build"),
             f"key light az {k['azimuth_deg']} el {k['elevation_deg']} radius {k['angular_radius_deg']} deg; "
             f"ambient fraction {lk['lights']['ambient']['horizontal_irradiance_scene_units']}",
             f"front white reference L_PLA {lk['cameras']['front']['white_reference']['L_PLA_lin']}"]
    for name, m in lk["materials"].items():
        lines.append(f"  {name:20s} base colour {m.get('base_color_linear')} roughness {m.get('roughness')}")
    lines.append(f"samples: {lk['samples']['summary']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="./robot real2sim look", description="LOOK part 1 assets")
    ap.add_argument("cmd", choices=["build", "info"])
    ap.add_argument("--ds")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        run(a.ds)
    else:
        print(info(a.ds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
