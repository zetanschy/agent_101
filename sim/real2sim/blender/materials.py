"""The Blender material table: look.json's measured materials as Principled BSDF inputs.

Every colour starts from LOOK's (look.json "materials"): linear albedo referenced to
white PLA (rho 0.80, assumed), measured on this dataset's frames. This module maps them
onto Blender's Principled BSDF, adds what LOOK does not measure, and holds the three
renderer-side decisions below (ROOM REFLECTION, RENDER FIT, PER-CAMERA COLOUR), each
with the measurement behind it:

    ior          F0 = ((n - 1) / (n + 1))^2, so LOOK's dielectric F0 0.04 is n = 1.5
                 (spec: plastics, enamel, the mat's lacquer). Blender's "Specular IOR
                 Level" stays at its physical 0.5, i.e. the reflectance is F0 itself.
    metals       base colour = F0 (Blender's metallic workflow), LOOK's stainless F0.
    roughness    LOOK's values. Blender's Roughness is the PERCEPTUAL one: GGX alpha =
                 roughness^2, which is how lighting.fit_gloss reports the mat (alpha
                 0.021-0.027 -> roughness 0.15-0.16).

ROOM REFLECTION. The fill (LOOK's uniform ambient) lights diffusely only: every
dielectric colour above is an effective albedo that already holds its reflection of
the room (job.textures REFLECTIONS). The steel is the exception: its colour is a spec
F0, nothing of the room is in it, and the real interior is the brightest part of the
mug (front camera-linear p50 0.08-0.12, p90 0.22-0.25 over the rest/approach samples;
LOOK's median 0.128 / 0.144 / 0.112). With the room black, it renders black in both
engines (front mug dE 30.9 in Cycles, whose glossy bounces do trace key -> lit wall ->
floor), so the steel adds scale x F0 x the unobserved environment in its mirror
direction, where that direction points up and out of the mug
(bl_render._add_room_reflection; up_ramp_z = mirror z over which it fades in).
scale 0.5: at 1 (the fill itself) the steel's median came out 1.75-2.07x the real one
(front, e0/e1/e2 rest-approach samples, EEVEE and Cycles), and independently the mat's
glare fit (F0 0.04, LOOK lighting.fit_gloss k_s / E_key) caps the zenith it mirrors
at 0.46 x fill: the room above the table is darker than its average.

RENDER FIT (the enamel). Through this renderer the red enamel came out too dark in
both cameras: R real / render on the real red pixels 2.1 on e0's wrist carry / over-mug
/ release samples (the lit side; 1.8-2.2 on e1/e2), and 4-5 in the front view, which
sees the key-shadowed side. LOOK's albedo rests on 416 lit front pixels (materials.
mug_enamel_red). RENDER_FIT scales its R by the wrist factor; the remaining ~2.2x on the
shadowed side says LOOK's uniform ambient under-lights vertical faces (README gaps).

PER-CAMERA COLOUR. The two webcams see the teal cap differently: one saturation
factor (sigma 1.15, LOOK colour.py) fits the enamel and the fluorescent marks, but the
cap alone wants 1.59, and through the shared model the wrist render's cap came out
3.9x too red and 16-30 % too bright in G/B (median over e0's approach / descend /
grasp samples, cap region; the front cap at rest is within 3-8 % per channel). LOOK
measures the cap top in the wrist frames (materials.cap_teal.wrist_camera_albedo, the
albedo that reproduces them through the full wrist model), so the wrist pass renders
the cap with that albedo (PER_CAMERA). It is the cap's spectrum through the wrist
sensor, a camera property, so the cap keeps one geometry and one roughness.

Which mesh gets which material (PART_MATERIAL) follows LOOK's part classes
(look/segment.py): printed parts and fingers are white PLA, the STS3215 servos are
black, the klip bracket and the KWC-500 body are the mount's black. The mug body
carries two slots (enamel outside, steel inside and on the rim, job.mug_face_slots).
"""

from __future__ import annotations

import os

PART_MATERIAL = {"printed": "white_pla", "fingers": "white_pla", "servo": "servo_black", "mount": "mount_black"}

# Materials the renderer needs, in the order they are created; each maps to a look.json
# entry of the same name (the table's colour is a texture, see job.textures).
NAMES = ("white_pla", "servo_black", "mount_black", "cap_teal", "mug_enamel_red", "mug_steel_interior",
         "mat_black_glossy")
# material -> {camera: the look.json key of that camera's base colour} (PER-CAMERA COLOUR)
PER_CAMERA = {"cap_teal": {"grip": "wrist_camera_albedo"}}
# material -> per-channel factor on LOOK's base colour, fitted through this renderer (RENDER FIT)
RENDER_FIT = {"mug_enamel_red": {"scale": (2.1, 1.0, 1.0),
                                 "source": "fitted: e0 wrist samples, red pixels, R real/render (materials.py RENDER FIT)"}}
ROOM_SCALE = 0.5  # ROOM REFLECTION

# SHARED LOOK ONLY. RENDER_FIT and PER_CAMERA are this renderer's own fits; Isaac RTX
# (isaac/look.py) applies look.json alone. R2S_BLENDER_SHARED_LOOK=1 drops both, so a
# Blender-vs-Isaac comparison can be scored on identical material inputs (compare/).
SHARED_LOOK_ONLY = os.environ.get("R2S_BLENDER_SHARED_LOOK", "") not in ("", "0")


def _ior(f0) -> float:
    f0 = float(f0)
    s = f0 ** 0.5
    return round((1 + s) / (1 - s), 4)


def materials(look: dict) -> dict:
    """{name: {base_color, roughness, metallic, ior, texture?, source}} for bl_render.py."""
    lm = look["materials"]
    out = {}
    for name in NAMES:
        m = lm[name]
        metallic = float(m.get("metallic", 0.0))
        entry = {"base_color": [float(v) for v in m["base_color_linear"]], "roughness": float(m.get("roughness", 0.5)),
                 "metallic": metallic, "ior": _ior(m.get("specular_f0", 0.04)) if metallic < 0.5 else 1.5,
                 "source": f"look.json materials.{name}: {m.get('source', '')}"}
        if name in RENDER_FIT and not SHARED_LOOK_ONLY:
            f = RENDER_FIT[name]
            entry["base_color"] = [round(c * k, 5) for c, k in zip(entry["base_color"], f["scale"])]
            entry["source"] += f"; x {list(f['scale'])} ({f['source']})"
        for cam, key in ({} if SHARED_LOOK_ONLY else PER_CAMERA.get(name, {})).items():
            if m.get(key):
                entry.setdefault("per_camera", {})[cam] = {
                    "base_color": [float(v) for v in m[key]],
                    "source": f"look.json materials.{name}.{key} (PER-CAMERA COLOUR)"}
        if name == "mug_steel_interior":
            entry["room_reflection"] = {"up_ramp_z": 0.1, "scale": ROOM_SCALE, "source": "materials.py ROOM REFLECTION"}
        if name == "mat_black_glossy":
            entry["texture"] = "table_albedo"
            entry["source"] += "; base colour = table/albedo_lin.npy (dataset-derived texture, declared)"
        out[name] = entry
    return out
