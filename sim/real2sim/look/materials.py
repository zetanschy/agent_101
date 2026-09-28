"""Materials: the measured colour of each surface in each camera, and its linear albedo.

CAMERA MODEL. camera_lin = G_cam * E * albedo, per channel. camera_lin is the sRGB
decode (colour.py; the tone curve is ASSUMED to be sRGB). E is the irradiance and
G_cam folds in exposure and white balance. G_cam * E is unknown, so it is taken
from a WHITE REFERENCE seen under the same light: white PLA, rho_PLA = 0.80
(assumed, typical of white PLA; the dataset has no known target). Then

    albedo = rho_PLA * L / L_PLA / E_rel(n)

E_rel is the irradiance relative to a horizontal face, from the light fit
(lighting.py: distant key at (az, el) plus a uniform ambient fraction a):
    E_rel(n) = (1 - a) max(0, n.l) / sin(el) + a (1 + n_z) / 2      (= 1 for n = +z)
Every surface used here, apart from the mug wall, is horizontal: cap tops, the mat,
the desk.

FRONT CAMERA (reference state). Values are divided by each frame's white-balance
gain (plates.py), so they are all quoted in one average camera state. L_PLA comes
from the FK arm's printed parts: model eroded 3 px, white in the image, and a world
normal within 26 deg of vertical (depth render, segment.normals_world). The rest
frames of episodes 0-2 are used, where the folded arm lies flat and FK fits best.
L_PLA is the mean colour of the 60-90th luminance percentile band: the lit faces.
Their spread across frames is reported as the reference's uncertainty.

WRIST CAMERA. The two webcams differ by more than a white balance. Both are made to
agree on white (the wrist camera's static finger core against the front's L_PLA).
Even then, the wrist camera renders the teal cap, the red enamel and the fluorescent
marks more saturated (grip_colour_model fits one saturation factor sigma). Its values
go into the front reference state through colour.grip_to_front, with each frame's
own white balance and the exposure from the mosaic (table.py), and the same L_PLA
applies. The exposure is fitted on the mat's luminance alone, so the cap's
LUMINANCE is an independent check of the cross-camera calibration. The cap is the
one surface both cameras measure under the same light: a closed top on the table,
seen from above by the wrist camera during the descend and by the front camera at
rest.

Metals and gloss. The steel interior is a mirror of its surroundings, so its
"colour" is reported as observed and not inverted. Its render parameters are spec
(stainless F0) plus a visual estimate of roughness. The mat's roughness comes from
the glare (lighting.py). The mat's albedo gets a second, PLA-free estimate from the
glare fit: k_d / k_s with F0 = 0.04 and the source's solid angle (see
`mat_albedo_from_gloss`).
"""

from __future__ import annotations

import numpy as np

from .. import episodes as E
from .. import objects
from . import colour
from .segment import PART, erode

RHO_PLA = 0.80  # assumed: white PLA diffuse reflectance (no reference target in the data)
F0_DIELECTRIC = 0.04  # spec: IOR 1.5 plastics / rubber / enamel
F0_STAINLESS = (0.55, 0.56, 0.55)  # spec: polished stainless steel, linear (typical tabulated)
HORIZONTAL_NZ = 0.9  # |normal| within 26 deg of +z
REST_FRAMES = range(0, 30, 3)  # every episode opens at rest (episodes report: 0-33 / 0-36 / 0-86)


def e_rel(n, light) -> np.ndarray:
    """Irradiance (N,) on unit normals n (N, 3) relative to a horizontal face."""
    a = light["ambient_fraction"]["median"]
    d = np.asarray(light["direction_to_light"])
    return (1 - a) * np.clip(n @ d, 0, None) / d[2] + a * (1 + n[:, 2]) / 2


def _band_mean(lin, lo=60, hi=90) -> np.ndarray:
    y = colour.luminance(lin)
    a, b = np.percentile(y, [lo, hi])
    sel = (y >= a) & (y <= b)
    return lin[sel].mean(0) if sel.any() else lin.mean(0)


def front(scene, renderer, plates, light, log=print) -> dict:
    """The white reference, black servo and mount, and the objects, in the front camera."""
    units = scene.units()
    eps = E.load(scene.ds)
    g = plates["frame_gain"]
    pla, servo, mount, per_frame = [], [], [], []
    renderer.set_mug(None)
    for e, ep in eps.items():
        fr = ep.frames("front")
        for k in REST_FRAMES:
            q = units.to_urdf(ep.state[k])
            parts = renderer.parts(q, "front")
            nz = renderer.normals_world(q, "front")[..., 2]
            img = np.asarray(fr[k])
            lin = colour.srgb_to_linear(img) / g[ep.start + k]
            m = erode(np.isin(parts, (PART["printed"], PART["fingers"])), 3) & colour.white_pla(img, "front") \
                & (nz > HORIZONTAL_NZ)
            if m.sum() > 200:
                pla.append(lin[m])
                per_frame.append(colour.luminance(_band_mean(lin[m])))
            servo.append(lin[erode(parts == PART["servo"], 2) & (nz > 0.7)])
            mount.append(lin[erode(parts == PART["mount"], 2)])
    L_pla = _band_mean(np.concatenate(pla))
    res = {"L_PLA_lin": np.round(L_pla, 5).tolist(),
           "L_PLA_luminance_per_frame_p5_p50_p95": np.round(np.percentile(per_frame, [5, 50, 95]), 4).tolist(),
           "L_PLA_frames": len(per_frame)}
    ref = RHO_PLA / L_pla
    for name, arr in (("servo_black", servo), ("mount_black", mount)):
        a = np.concatenate(arr)
        res[name] = {"n": int(len(a)), "lin_median": np.round(np.median(a, 0), 5).tolist(),
                     "albedo": np.round(np.median(a, 0) * ref, 4).tolist()}
    res.update(_front_objects(scene, renderer, plates, light, ref))
    log(f"  front white reference L_PLA {res['L_PLA_lin']} (per-frame luminance p5/50/95 "
        f"{res['L_PLA_luminance_per_frame_p5_p50_p95']})")
    return res


def _front_objects(scene, renderer, plates, light, ref) -> dict:
    """Caps at rest (frame 0 of each episode: closed tops vs open sides), mug enamel
    and steel (episode plates: static), mat and fluorescent marks (clean plate)."""
    from .segment import front_mug_mask

    out = {"cap_closed_top": [], "cap_open_side": [], "mug_red": [], "mug_red_albedo": [], "steel": []}
    cam = scene.camera("front")
    yy, xx = np.mgrid[0:cam.height, 0:cam.width]
    clean_u8 = colour.linear_to_srgb(np.nan_to_num(plates["clean_lin"]))
    for e, ep in E.load(scene.ds).items():
        img = np.asarray(ep.frames("front")[0])
        lin = colour.srgb_to_linear(img) / plates["frame_gain"][ep.start]
        teal = colour.teal(img, "front")
        for c in scene.episode(e)["caps"]:
            cx, cy = c["px"]
            disk = ((xx - cx) ** 2 + (yy - cy) ** 2 < (c["r_px"] - 3) ** 2) & teal
            out["cap_closed_top" if c["up"] == "closed" else "cap_open_side"].append(lin[disk])
        plate_u8 = colour.linear_to_srgb(plates["plate_lin"][e])
        mug = front_mug_mask(plates["mug_hull"][e], plate_u8, clean_u8)
        renderer.set_mug(objects.mug_pose(scene, e))
        q = scene.units().to_urdf(ep.state[0])
        n = renderer.normals_world(q, "front")
        plin = plates["norm_lin"][e]
        red = colour.red(plate_u8) & mug & (np.linalg.norm(n, axis=-1) > 0.5)  # model normal present
        E_r = e_rel(n[red].reshape(-1, 3), light)
        lit = E_r > 0.3
        out["mug_red"].append(plin[red])
        out["mug_red_albedo"].append((plin[red] * ref / E_r[:, None])[lit])
        out["steel"].append(plin[erode(mug & ~colour.red(plate_u8) & ~colour.teal(plate_u8, "front"), 2)
                                 & (np.abs(n[..., 2]) > 0.2)])
    renderer.set_mug(None)
    res = {}
    for k in ("cap_closed_top", "cap_open_side", "mug_red", "steel"):
        a = np.concatenate(out[k])
        res[k] = {"n": int(len(a)), "lin_median": np.round(np.median(a, 0), 5).tolist(),
                  "albedo_if_horizontal": np.round(np.median(a, 0) * ref, 4).tolist()}
    a = np.concatenate(out["mug_red_albedo"])
    res["mug_red"]["albedo_irradiance_corrected"] = np.round(np.median(a, 0), 4).tolist() if len(a) else None
    res["mug_red"]["n_lit"] = int(len(a))
    clean = plates["clean_lin"]
    u8 = clean_u8
    hsv = colour.hsv(u8)
    fluor = plates["clean_valid"] & (hsv[..., 0] > 20) & (hsv[..., 0] < 45) & (hsv[..., 1] > 80)
    plain = plates["clean_valid"] & ~fluor & (u8.max(-1) < 40)
    for k, m in (("mat_plain", plain), ("fluorescent_marks", fluor)):
        res[k] = {"n": int(m.sum()), "lin_median": np.round(np.median(clean[m], 0), 6).tolist(),
                  "albedo": np.round(np.median(clean[m], 0) * ref, 5).tolist()}
    return res


def _grip_windows(scene, e):
    """Episode-local frame windows of the wrist camera: the descend onto each closed cap
    (its top face, horizontal, seen from above), and each carry (the mug in view)."""
    ep = scene.episode(e)
    up = {c["id"]: c["up"] for c in ep["caps"]}
    descend = [range(max(p["close"][0] - 45, 0), p["close"][0] - 5, 5) for p in ep["picks"] if up[p["cap"]] == "closed"]
    carry = [range(p["lift"][1], p["release"][0], 5) for p in ep["picks"]]
    return descend, carry


def grip_colour_model(scene, prior, front_res, log=print) -> dict:
    """Fit the wrist camera's saturation sigma relative to the front camera
    (colour.py), on the three chromatic surfaces both cameras see. The teal cap top
    comes from the descend frames, the red enamel from the carry frames, and the
    yellow-green marks from every 5th frame where they show. The per-frame white
    balance comes from the static finger core. Chromaticities (a / sum a) are
    compared in white-normalised units, and sigma comes from a 1-D search over 0.5-3."""
    from .segment import dilate

    L = np.asarray(front_res["L_PLA_lin"])
    core, zone = prior.static_core(), dilate(prior.frequency() > 0.02, 6)
    got = {"cap": [], "red": [], "fluor": []}
    s_cap = []
    a_cap_front = np.asarray(front_res["cap_closed_top"]["lin_median"]) / L
    for e, ep in E.load(scene.ds).items():
        fr = ep.frames("grip")
        descend, carry = _grip_windows(scene, e)
        todo = [(k, "cap") for r in descend for k in r] + [(k, "red") for r in carry for k in r] + \
               [(k, "fluor") for k in range(0, len(ep), 5)]
        for k, what in todo:
            img = np.asarray(fr[k])
            wb = colour.grip_white_balance(img, core, L)
            if wb is None:
                continue
            h = colour.hsv(img)
            if what == "cap":
                m = erode(colour.teal(img, "grip", h) & ~zone, 6)
            elif what == "red":
                m = erode(colour.red(img, h) & ~zone, 2)
            else:
                m = (h[..., 0] > 20) & (h[..., 0] < 45) & (h[..., 1] > 80) & (h[..., 2] > 40) & ~zone
            if m.sum() < (2000 if what == "cap" else 300):
                continue
            a = np.median(colour.srgb_to_linear(img)[m], 0) * wb / L
            got[what].append(a)
            if what == "cap":
                s_cap.append(float((a_cap_front @ colour.LUMA) / (a @ colour.LUMA)))
    front_a = {"cap": a_cap_front, "red": np.asarray(front_res["mug_red"]["lin_median"]) / L,
               "fluor": np.asarray(front_res["fluorescent_marks"]["lin_median"]) / L}
    grip_a = {k: np.median(v, 0) for k, v in got.items() if v}
    chrom = lambda a: np.asarray(a) / np.sum(a)  # noqa: E731
    sig = np.round(np.arange(0.5, 3.0001, 0.01), 3)

    def err(k, sg):
        return float(np.sum((chrom(np.clip(colour.saturate(front_a[k], sg), 1e-6, None)) - chrom(grip_a[k])) ** 2))

    keys = sorted(grip_a)
    tot = [sum(err(k, sg) for k in keys) for sg in sig]
    sigma = float(sig[int(np.argmin(tot))])
    res = {"sigma": sigma, "surfaces": {}, "cap_exposure_front_over_grip": None}
    for k in keys:
        per = [err(k, sg) for sg in sig]
        res["surfaces"][k] = {"frames": len(got[k]), "grip_white_normalised": np.round(grip_a[k], 5).tolist(),
                              "front_white_normalised": np.round(front_a[k], 5).tolist(),
                              "best_sigma_alone": float(sig[int(np.argmin(per))]),
                              "chromaticity_rms_at_sigma": round(float(np.sqrt(err(k, sigma) / 3)), 4),
                              "chromaticity_rms_at_1": round(float(np.sqrt(err(k, 1.0) / 3)), 4)}
    if s_cap:
        res["cap_exposure_front_over_grip"] = {"median": round(float(np.median(s_cap)), 4),
                                               "p10_p90": np.round(np.percentile(s_cap, [10, 90]), 4).tolist(),
                                               "frames": len(s_cap)}
    log(f"  wrist colour model: saturation sigma {sigma} (alone: "
        f"{ {k: v['best_sigma_alone'] for k, v in res['surfaces'].items()} })")
    return res


def grip(scene, prior, wrist_model: dict, sigma: float, L_pla, log=print) -> dict:
    """The wrist camera's view of the closed cap tops, through the full wrist->front
    model (per-frame white balance and mat-based exposure, sigma). Because the
    exposure comes from the mat, the cap's LUMINANCE here is an independent check of
    the cross-camera calibration."""
    L = np.asarray(L_pla)
    frames = {tuple(map(int, k.split(":"))): v for k, v in wrist_model.items()}
    core = prior.static_core()
    from .segment import dilate

    zone = dilate(prior.frequency() > 0.02, 6)
    caps, n = [], 0
    for e, ep in E.load(scene.ds).items():
        fr = ep.frames("grip")
        descend, _ = _grip_windows(scene, e)
        ks = [kk for (ee, kk) in frames if ee == e]
        for k in (k for r in descend for k in r):
            if not ks:
                break
            m = frames[(e, min(ks, key=lambda kk: abs(kk - k)))]
            img = np.asarray(fr[k])
            wb = colour.grip_white_balance(img, core, L)
            mask = erode(colour.teal(img, "grip") & ~zone, 6)
            if wb is None or mask.sum() < 2000:
                continue
            caps.append(colour.grip_to_front(colour.srgb_to_linear(img)[mask], wb, m["exposure"], sigma, L))
            n += 1
    res = {"frames": n}
    if caps:
        a = np.median(np.concatenate(caps), 0)
        res["cap_closed_top"] = {"lin_median_front_ref": np.round(a, 5).tolist(),
                                 "albedo": np.round(a * RHO_PLA / L, 4).tolist()}
    log(f"  wrist cap-top albedo through the full model: {res.get('cap_closed_top', {}).get('albedo')} ({n} frames)")
    return res


def mat_albedo_from_gloss(k_d, k_s, light) -> list:
    """The mat's diffuse albedo WITHOUT the PLA reference. For a disk source of angular
    radius beta and radiance Ls: a mirror-like dielectric reflects k_s = Ls F0 Omega
    (the fit's lobe is a mean over the cone), and the diffuse is
    k_d = rho Ls pi sin^2(beta) sin(el) / (pi (1 - a)).
    So rho = k_d (1 - a) F0 Omega / (k_s sin^2(beta) sin(el))."""
    beta, el = np.radians(light["angular_radius_deg"]), np.radians(light["elevation_deg"])
    a = light["ambient_fraction"]["median"]
    omega = 2 * np.pi * (1 - np.cos(beta))
    return (np.asarray(k_d) * (1 - a) * F0_DIELECTRIC * omega / (np.asarray(k_s) * np.sin(beta) ** 2 * np.sin(el))).round(5).tolist()


def assemble(front_res, grip_res, table_albedo_stats, table_summary, light, gloss) -> dict:
    """The render-facing material table (look.json 'materials'). Albedo is linear,
    base colour for a principled / UsdPreviewSurface shader."""
    rough_mat = float(np.median([g["roughness"] for g in gloss.values()]))

    def mat(albedo, source, **kw):
        return {"base_color_linear": albedo, "source": source, **kw}

    cap_f = front_res["cap_closed_top"]["albedo_if_horizontal"]
    cap_g = grip_res.get("cap_closed_top", {}).get("albedo")
    agree = None
    if cap_g:
        agree = round(float(np.abs(np.log(np.asarray(cap_g) / np.asarray(cap_f))).max()), 4)
    return {
        "white_pla": mat([RHO_PLA] * 3, "assumed: white PLA rho 0.80, the white reference of both cameras",
                         roughness=0.55, metallic=0.0, specular_f0=F0_DIELECTRIC),
        "servo_black": mat(front_res["servo_black"]["albedo"], "measured: front, FK servo faces (n_z > 0.7), PLA-referenced",
                           roughness=0.5, metallic=0.0),
        "mount_black": mat(front_res["mount_black"]["albedo"], "measured: front, klip mount + webcam body pixels, PLA-referenced",
                           roughness=0.5, metallic=0.0),
        "cap_teal": mat(cap_f, "measured: front, closed cap tops at rest (horizontal), PLA-referenced",
                        wrist_camera_albedo=cap_g, cross_camera_max_log_ratio=agree, roughness=0.35, metallic=0.0,
                        specular_f0=F0_DIELECTRIC,
                        open_side_observed_albedo=front_res["cap_open_side"]["albedo_if_horizontal"]),
        "mug_enamel_red": mat(front_res["mug_red"]["albedo_irradiance_corrected"],
                              "measured: front, red enamel wall of the static mug, irradiance-corrected with the model "
                              "normals and the light fit (E_rel > 0.3 only)",
                              observed_albedo_if_horizontal=front_res["mug_red"]["albedo_if_horizontal"],
                              roughness=0.25, metallic=0.0, specular_f0=F0_DIELECTRIC),
        "mug_steel_interior": {"base_color_linear": list(F0_STAINLESS), "metallic": 1.0, "roughness": 0.15,
                               "source": "spec: stainless F0; roughness estimated (cap reflections on the interior are sharp but "
                                         "not mirror-sharp, ep1/ep2 last frames)",
                               "observed_lin_median_front": front_res["steel"]["lin_median"]},
        "mat_black_glossy": mat(table_albedo_stats["mat_median"],
                                "measured: front clean plate, plain mat, specular removed, PLA-referenced; texture "
                                "table/albedo.png",
                                albedo_from_gloss_no_pla=mat_albedo_from_gloss(table_summary["k_diffuse_lin"],
                                                                                 table_summary["k_specular"], light),
                                roughness=rough_mat, metallic=0.0, specular_f0=F0_DIELECTRIC,
                                roughness_source="fitted: GGX lobe of the front glare given the shadow-fitted source "
                                                 "(lighting.fit_gloss)"),
        "fluorescent_marks": mat(table_albedo_stats["fluorescent_median"],
                                 "measured: front clean plate, yellow-green marks (H 20-45, S > 80), PLA-referenced",
                                 roughness=0.6, metallic=0.0),
        "desk_white": mat(table_albedo_stats["wrist_region_p50_p90"][1] if table_albedo_stats["wrist_region_p50_p90"] else None,
                          "measured: wrist mosaic beyond the front view, 90th percentile (the white desk; the 50th is "
                          "mostly mat)", roughness=0.7, metallic=0.0),
    }
