"""Blender side of a render job: build the scene once, then render every unique pose.

Run by cli.py inside the GPU lock, never by hand:

    blender -b --factory-startup --python-expr "<sim/ on sys.path>" --python bl_render.py -- \
        --job <job dir> --out <pinhole dir> [--cams front,grip] [--mode beauty|silhouette]

It reads only job.json / job.npz / tex_*.npy (job.py wrote them) and writes, per camera,
<out>/<cam>/<render index:05d>.exr: the oversized PINHOLE of that camera (job.json
cameras.<cam>.pinhole), linear scene radiance, half float. Distortion, exposure,
white balance and the sRGB curve are the camera's, applied on the host
(camera_model.py), so the same post-process serves any renderer.

After each frame it prints one line  R2S_FRAME <cam> <index> <seconds> <path>  (the
host streams on these), and at the end  R2S_DONE <json>  with the timings.

PERFORMANCE. The scene is built once: meshes are created from numpy in one foreach_set
per array, textures are float images made from numpy, and per frame only the 7 link
empties and the object empties get a new matrix_world (every mesh is parented to its
link, so the geometry is never touched again). Cameras render in passes (all frames of
one camera, then the next), so the resolution never changes inside a pass and Cycles'
persistent data keeps its BVH and textures from frame to frame.

LIGHT. The key is a sun (LOOK's fitted disk source), the world is env.hdr for diffuse
and camera rays and its observed part only for glossy rays (job.textures REFLECTIONS);
the steel adds the rest explicitly (materials.py ROOM REFLECTION). A camera pass sets
that camera's base colours first (materials.py PER-CAMERA COLOUR).

SILHOUETTE mode swaps every material for an emission of its part code and hides the
table, backdrop and lights: check.py (`./robot real2sim blender check`) compares it
with LOOK's MuJoCo label renderer to prove that meshes, link poses, cameras and the
remap agree end to end.
"""

import json
import math
import sys
import time
from pathlib import Path

import bpy
import mathutils
import numpy as np

# part codes of look/segment.py PART (printed 1, fingers 2, mug 4, servo 5, mount 6), cap 3
SIL_CODE = {"printed": 1, "fingers": 2, "cap": 3, "mug": 4, "servo": 5, "mount": 6}


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {"cams": None, "mode": "beauty"}
    it = iter(a)
    for k in it:
        if k == "--job":
            out["job"] = Path(next(it))
        elif k == "--out":
            out["out"] = Path(next(it))
        elif k == "--cams":
            out["cams"] = next(it).split(",")
        elif k == "--mode":
            out["mode"] = next(it)
    return out


# --- scene building ----------------------------------------------------------------------

def mesh_object(name, V, F, mats, slots=None, smooth_deg=30.0, parent=None, uv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(V, np.float64), [], np.asarray(F, np.int64))
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        loops = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loops)
        layer.data.foreach_set("uv", np.asarray(uv, np.float32)[loops].ravel())
    for m in mats:
        me.materials.append(m)
    if slots is not None:
        me.polygons.foreach_set("material_index", np.asarray(slots, np.int32))
    if smooth_deg:
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(smooth_deg))
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
        ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    return ob


def empty(name):
    ob = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def float_image(name, arr):
    """(H, W, 4) float32 in Blender pixel order (row 0 = bottom) -> a linear float image.
    Alpha is a separate data channel (CHANNEL_PACKED): straight alpha would zero the
    colour of the backdrop's alpha-0 (unseen) texels, which camera rays must still see."""
    H, W = arr.shape[:2]
    img = bpy.data.images.new(name, W, H, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Linear Rec.709"
    img.alpha_mode = "CHANNEL_PACKED"
    img.pixels.foreach_set(np.ascontiguousarray(arr, np.float32).ravel())
    img.pack()
    return img


def principled(name, m, images, quality):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*m["base_color"], 1.0)
    b.inputs["Roughness"].default_value = m["roughness"]
    b.inputs["Metallic"].default_value = m["metallic"]
    b.inputs["IOR"].default_value = m["ior"]
    if m.get("texture"):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = images[m["texture"]]
        t.interpolation = "Linear"
        t.extension = "EXTEND"
        nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if m.get("room_reflection") and quality.get("room_reflection", True):
        _add_room_reflection(nt, b, m, images["env_unseen"])
    return mat


def _add_room_reflection(nt, bsdf, m, env_unseen):
    """BSDF + scale x F0 x env_unseen(mirror direction) x up(mirror z): the metal's reflection of
    the room nobody observed (materials.py ROOM REFLECTION). Only mirror rays that leave
    upwards get it: inside the mug those escape through the opening (the floor, the rim),
    while the walls' mirror rays point down into the mug and are traced as usual."""
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = env_unseen
    env.interpolation = "Linear"
    nt.links.new(tc.outputs["Reflection"], env.inputs["Vector"])
    xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Reflection"], xyz.inputs[0])
    up = nt.nodes.new("ShaderNodeMapRange")
    up.inputs["From Min"].default_value, up.inputs["From Max"].default_value = 0.0, m["room_reflection"]["up_ramp_z"]
    up.inputs["To Max"].default_value = m["room_reflection"]["scale"]
    up.clamp = True
    nt.links.new(xyz.outputs["Z"], up.inputs["Value"])
    tint = nt.nodes.new("ShaderNodeMixRGB")
    tint.blend_type = "MULTIPLY"
    tint.inputs["Fac"].default_value = 1.0
    nt.links.new(env.outputs["Color"], tint.inputs[1])
    tint.inputs[2].default_value = (*m["base_color"], 1.0)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tint.outputs["Color"], em.inputs["Color"])
    nt.links.new(up.outputs["Result"], em.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(bsdf.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])


def emission(name, rgb):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def backdrop_material(image):
    """The room picture as emission. Its alpha is the seen mask (job.textures): for a
    glossy ray a texel no wrist frame saw is transparent, so reflections fall through to
    the world's env_glossy (black there) instead of mirroring the uniform fill twice.
    Camera rays still see the fill: that is the picture of the unseen room."""
    mat = bpy.data.materials.new("backdrop")
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = image
    t.interpolation = "Linear"
    t.extension = "REPEAT"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(t.outputs["Color"], em.inputs["Color"])
    unseen = nt.nodes.new("ShaderNodeMath")
    unseen.operation = "SUBTRACT"
    unseen.inputs[0].default_value = 1.0
    nt.links.new(t.outputs["Alpha"], unseen.inputs[1])
    lp = nt.nodes.new("ShaderNodeLightPath")
    fac = nt.nodes.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Glossy Ray"], fac.inputs[0])
    nt.links.new(unseen.outputs[0], fac.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fac.outputs[0], mix.inputs["Fac"])
    nt.links.new(em.outputs[0], mix.inputs[1])
    nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def table_quad(t, mat):
    V = np.array([[t["x0"], t["y0"], t["z"]], [t["x1"], t["y0"], t["z"]], [t["x1"], t["y1"], t["z"]],
                  [t["x0"], t["y1"], t["z"]]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    return mesh_object("table", V, np.array([[0, 1, 2, 3]]), [mat], smooth_deg=0, uv=uv)


def backdrop_cylinder(b, mat, n=256):
    """Open vertical cylinder (inward-facing) with u = (phi + pi) / 2 pi, v = height fraction,
    phi = atan2(y - cy, x - cx): LOOK's cylinder texture layout (look/backdrop.py)."""
    cx, cy = b["centre_xy"]
    R, H, z0 = b["radius"], b["height"], b["z0"]
    phi = -math.pi + np.arange(n + 1) * 2 * math.pi / n
    V, uv = [], []
    for z, v in ((z0, 0.0), (z0 + H, 1.0)):
        V += [[cx + R * math.cos(p), cy + R * math.sin(p), z] for p in phi]
        uv += [[(p + math.pi) / (2 * math.pi), v] for p in phi]
    m = n + 1
    F = [[i, m + i, m + i + 1, i + 1] for i in range(n)]  # faces point inwards
    return mesh_object("backdrop", np.array(V), np.array(F), [mat], smooth_deg=0, uv=np.array(uv))


def build(job, jd: Path, mode: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    z = np.load(jd / "job.npz")
    images = {}
    if mode == "beauty":
        for key in ("table_albedo", "backdrop", "env", "env_glossy", "env_unseen"):
            images[key] = float_image(key, np.load(jd / job["textures"][key]))
    mats = {}
    if mode == "beauty":
        for name, m in job["materials"].items():
            mats[name] = principled(name, m, images, job.get("quality", {}))
    else:
        for part, code in SIL_CODE.items():
            mats[part] = emission(f"sil_{part}", (code / 10.0, 0.0, 0.0))
    link_obj = {n: empty(f"link_{n}") for n in job["links"]}
    for a in job["arm"]:
        key = a["key"]
        mat = mats[a["material"]] if mode == "beauty" else mats[a["part"]]
        mesh_object(a["name"], z[f"{key}_v"], z[f"{key}_f"], [mat], parent=link_obj[a["link"]])
    obj_obj = [empty(f"obj_{n}") for n in job["objects"]]
    for om in job["object_meshes"]:
        key = om["key"]
        if mode == "beauty":
            ms = [mats[s] for s in om["slots"]]
        else:
            ms = [mats["cap" if om["name"].startswith("cap") else "mug"]] * len(om["slots"])
        mesh_object(om["name"], z[f"{key}_v"], z[f"{key}_f"], ms,
                    slots=z[f"{key}_slot"] if om["has_slots"] else None, parent=obj_obj[om["object"]])
    world = bpy.data.worlds.new("world")
    sc.world = world
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    if mode == "beauty":
        table_quad(job["table"], mats["mat_black_glossy"])
        cyl = backdrop_cylinder(job["backdrop"], backdrop_material(images["backdrop"]))
        # the far field is a picture of the room for the wrist camera and for reflections;
        # its light is already the environment's (look/backdrop.py), so it must not light
        # or shadow anything a second time
        cyl.visible_diffuse = False
        cyl.visible_shadow = False
        # diffuse (and camera) rays see env.hdr with its uniform fill = the ambient; glossy
        # rays see only what was observed (job.textures REFLECTIONS). Measured on this
        # Blender, both engines: a world switched on Light Path 'Is Glossy Ray' lights a
        # white diffuse plane at 1.0 (Cycles) / 0.99 (EEVEE) and leaves a metal plane at 0.
        wn = world.node_tree
        mix = wn.nodes.new("ShaderNodeMixRGB")
        for slot, key in ((1, "env"), (2, "env_glossy")):
            env = wn.nodes.new("ShaderNodeTexEnvironment")
            env.image = images[key]
            env.interpolation = "Linear"
            wn.links.new(env.outputs["Color"], mix.inputs[slot])
        wn.links.new(wn.nodes.new("ShaderNodeLightPath").outputs["Is Glossy Ray"], mix.inputs["Fac"])
        wn.links.new(mix.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = job["lights"]["world"]["strength"]
        s = job["lights"]["sun"]
        ld = bpy.data.lights.new("key", "SUN")
        ld.energy = s["strength"]
        ld.angle = math.radians(s["angle_deg"])
        ld.use_shadow = True
        if hasattr(ld, "use_shadow_jitter"):
            ld.use_shadow_jitter = True
        lo = bpy.data.objects.new("key", ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = mathutils.Vector(s["direction_to_light"]).to_track_quat("Z", "Y").to_euler()
    else:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
    cams = {}
    for name, c in job["cameras"].items():
        b = c["blender"]
        cd = bpy.data.cameras.new(name)
        cd.lens, cd.sensor_width, cd.sensor_fit = b["lens"], b["sensor_width"], b["sensor_fit"]
        cd.shift_x, cd.shift_y = b["shift_x"], b["shift_y"]
        cd.clip_start, cd.clip_end = c["clip_start"], c["clip_end"]
        co = bpy.data.objects.new(f"cam_{name}", cd)
        sc.collection.objects.link(co)
        if b["parent"] != "world":
            co.parent = link_obj[b["parent"]]
            co.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            co.matrix_basis = mathutils.Matrix(b["T_parent_cam_gl"])  # rows: a nested list alone is read column-major
        else:
            co.matrix_world = mathutils.Matrix(b["T_parent_cam_gl"])
        cams[name] = co
    return sc, z, link_obj, obj_obj, cams, mats


def camera_colours(mats, job, cam):
    """Base colours of this camera's pass: materials.py PER-CAMERA COLOUR, else the default."""
    for name, m in job["materials"].items():
        if "per_camera" not in m or name not in mats:
            continue
        rgb = m["per_camera"].get(cam, {}).get("base_color", m["base_color"])
        b = next(n for n in mats[name].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        b.inputs["Base Color"].default_value = (*rgb, 1.0)


def configure(sc, job, mode):
    q = job.get("quality", {})
    r = sc.render
    r.resolution_percentage = 100
    r.use_persistent_data = True
    r.image_settings.file_format = "OPEN_EXR"
    r.image_settings.color_depth = "16"
    r.image_settings.color_mode = "RGB"
    r.image_settings.exr_codec = "ZIP"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    engine = job["engine"]
    if engine == "cycles":
        r.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        dev = None
        for typ in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = typ
                prefs.get_devices()
            except TypeError:
                continue
            if any(d.type == typ for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == typ
                dev = typ
                break
        cy = sc.cycles
        cy.device = "GPU" if dev else "CPU"
        cy.samples = int(q.get("samples", 64)) if mode == "beauty" else 1
        cy.use_adaptive_sampling = mode == "beauty"
        cy.adaptive_threshold = float(q.get("adaptive_threshold", 0.01))
        cy.use_denoising = mode == "beauty" and q.get("denoise", True)
        cy.denoiser = "OPTIX" if dev == "OPTIX" else "OPENIMAGEDENOISE"
        cy.max_bounces = int(q.get("max_bounces", 6))
        cy.diffuse_bounces, cy.glossy_bounces = int(q.get("diffuse_bounces", 3)), int(q.get("glossy_bounces", 4))
        cy.transmission_bounces, cy.transparent_max_bounces = 2, 4
        cy.caustics_reflective = cy.caustics_refractive = False
        cy.sample_clamp_indirect = float(q.get("clamp_indirect", 10.0))
        if mode != "beauty":
            cy.pixel_filter_type = "BOX"
            cy.filter_width = 0.01
        return {"engine": "CYCLES", "device": cy.device, "backend": dev, "samples": cy.samples,
                "adaptive_threshold": cy.adaptive_threshold, "denoiser": cy.denoiser if cy.use_denoising else None}
    r.engine = "BLENDER_EEVEE"
    ee = sc.eevee
    ee.taa_render_samples = int(q.get("samples", 64)) if mode == "beauty" else 1
    ee.use_shadows = True
    ee.shadow_ray_count = int(q.get("shadow_rays", 2))
    ee.shadow_step_count = int(q.get("shadow_steps", 8))
    ee.use_raytracing = bool(q.get("raytracing", True))
    ee.use_fast_gi = bool(q.get("fast_gi", True))
    if mode != "beauty":
        r.filter_size = 0.0
    return {"engine": "BLENDER_EEVEE", "samples": ee.taa_render_samples, "raytracing": ee.use_raytracing,
            "fast_gi": ee.use_fast_gi, "shadow_rays": ee.shadow_ray_count}


def set_pose(z, cam, i, link_obj, obj_obj, job):
    L = z[f"{cam}_links"][i]
    for k, n in enumerate(job["links"]):
        link_obj[n].matrix_world = mathutils.Matrix(L[k].tolist())
    O = z[f"{cam}_objects"][i]
    for k, ob in enumerate(obj_obj):
        ob.matrix_world = mathutils.Matrix(O[k].tolist())


def main():
    a = args()
    jd, out = a["job"], a["out"]
    job = json.loads((jd / "job.json").read_text())
    t0 = time.time()
    sc, z, link_obj, obj_obj, cams, mats = build(job, jd, a["mode"])
    info = configure(sc, job, a["mode"])
    t_build = time.time() - t0
    print(f"R2S_BUILD {t_build:.2f} {json.dumps(info)}", flush=True)
    timing = {"build_s": round(t_build, 2), "engine": info, "cameras": {}}
    for cam in a["cams"] or list(job["cameras"]):
        c = job["cameras"][cam]
        b = c["blender"]
        sc.camera = cams[cam]
        if a["mode"] == "beauty":
            camera_colours(mats, job, cam)
        r = sc.render
        r.resolution_x, r.resolution_y = b["resolution_x"], b["resolution_y"]
        r.pixel_aspect_x, r.pixel_aspect_y = b["pixel_aspect_x"], b["pixel_aspect_y"]
        (out / cam).mkdir(parents=True, exist_ok=True)
        times = []
        for i in range(c["renders"]):
            set_pose(z, cam, i, link_obj, obj_obj, job)
            r.filepath = str(out / cam / f"{i:05d}.exr")
            t = time.time()
            bpy.ops.render.render(write_still=True)
            dt = time.time() - t
            times.append(dt)
            print(f"R2S_FRAME {cam} {i} {dt:.3f} {r.filepath}", flush=True)
        timing["cameras"][cam] = {"renders": len(times), "first_s": round(times[0], 3) if times else None,
                                  "mean_s_after_first": round(float(np.mean(times[1:])), 4) if len(times) > 1 else None,
                                  "total_s": round(float(np.sum(times)), 2)}
    timing["total_s"] = round(time.time() - t0, 2)
    print("R2S_DONE " + json.dumps(timing), flush=True)


main()
