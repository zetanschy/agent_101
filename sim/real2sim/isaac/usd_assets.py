"""The cap and the mug as USD rigid bodies: visual mesh + convex collision pieces +
a physics material, written once per geometry/material hash.

Needs pxr, i.e. the Isaac process (import after AppLauncher). The files are small
text .usda under sim/outputs/real2sim/<ds>/isaac/usd/ and are spawned with Isaac Lab's
UsdFileCfg, whose rigid_props / collision_props / mass_props only MODIFY schemas that
already exist (isaaclab/sim/schemas/schemas.py modify_*), so every API the spawner
tunes is authored here:

    /Object                    Xform, RigidBodyAPI + PhysxRigidBodyAPI + MassAPI(mass)
      visual                   the core's watertight mesh (objects.py), rendered
      collision/piece_NN       one convex piece each (geometry.py), purpose=guide,
                               CollisionAPI + MeshCollisionAPI(convexHull, <= 255 verts)
                               + PhysxCollisionAPI, bound to the physics material
      PhysicsMaterial          UsdPhysics.MaterialAPI + PhysxMaterialAPI(combine modes)
      Looks/visual_material    a plain UsdPreviewSurface; LOOK rebinds it

Isaac Lab 2.1 cannot attach a physics material through UsdFileCfg (Isaac report §2),
which is why the material is bound inside the file. Mass is explicit and inertia is
NOT authored: PhysX derives the centre of mass and inertia from the pieces at uniform
density, which puts the hollow cap's centre of mass towards its top disk and the mug's
towards its floor, as for the real objects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BUILDER_VERSION = 1  # bump when the file layout below changes


def _mesh(stage, path: str, verts: np.ndarray, faces: np.ndarray):
    from pxr import Gf, UsdGeom, Vt

    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in verts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in np.asarray(faces).reshape(-1)]))
    m.CreateSubdivisionSchemeAttr("none")
    lo, hi = np.min(verts, 0), np.max(verts, 0)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    return m


def _hull_mesh(piece: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated convex hull with outward winding (USD's default right-handed)."""
    from .geometry import hull

    h = hull(piece)
    V = h.points[h.vertices]
    remap = {int(v): i for i, v in enumerate(h.vertices)}
    F = np.array([[remap[int(i)] for i in s] for s in h.simplices])
    c = V.mean(0)
    for k, (a, b, d) in enumerate(F):  # orient every triangle away from the centroid
        n = np.cross(V[b] - V[a], V[d] - V[a])
        if n @ (V[a] - c) < 0:
            F[k] = (a, d, b)
    return V, F


def content_key(kind: str, dims: dict, material: dict, pieces: list, color) -> str:
    blob = json.dumps({"v": BUILDER_VERSION, "kind": kind, "dims": dims, "material": material, "color": color,
                       "pieces": [np.round(p, 7).tolist() for p in pieces]}, sort_keys=True, default=float)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def write_rigid_object(path: Path, visual_mesh, pieces: list, mass: float, material: dict, color, roughness: float,
                       metallic: float = 0.0) -> Path:
    """Write one rigid object USD (see module doc). `material`: {static_friction,
    dynamic_friction, restitution, friction_combine, restitution_combine}."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))

    vis = _mesh(stage, "/Object/visual", np.asarray(visual_mesh.vertices), np.asarray(visual_mesh.faces))
    look = UsdShade.Material.Define(stage, "/Object/Looks/visual_material")
    sh = UsdShade.Shader.Define(stage, "/Object/Looks/visual_material/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(float(c) for c in color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    look.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(look)

    pm = UsdShade.Material.Define(stage, "/Object/PhysicsMaterial")
    mapi = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    mapi.CreateStaticFrictionAttr(float(material["static_friction"]))
    mapi.CreateDynamicFrictionAttr(float(material["dynamic_friction"]))
    mapi.CreateRestitutionAttr(float(material["restitution"]))
    xapi = PhysxSchema.PhysxMaterialAPI.Apply(pm.GetPrim())
    xapi.CreateFrictionCombineModeAttr(material.get("friction_combine", "average"))
    xapi.CreateRestitutionCombineModeAttr(material.get("restitution_combine", "average"))

    UsdGeom.Xform.Define(stage, "/Object/collision")
    for i, piece in enumerate(pieces):
        V, F = _hull_mesh(piece)
        m = _mesh(stage, f"/Object/collision/piece_{i:02d}", V, F)
        m.CreatePurposeAttr(UsdGeom.Tokens.guide)
        prim = m.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim).CreateHullVertexLimitAttr(255)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return path


def build(scene, out_dir: Path, material_of: dict, look: dict, force: bool = False) -> dict:
    """{'cap': path, 'mug': path} for this scene's dimensions. material_of / look:
    {'cap': {...}, 'mug': {...}} physics materials and (color, roughness, metallic)."""
    from .. import objects
    from . import geometry as G

    cd, md = scene.cap_dims(), scene.mug_dims()
    todo = {
        "cap": (objects.cap_mesh(cd), G.cap_pieces(cd), cd["mass"]),
        "mug": (objects.mug_mesh(md), G.mug_pieces(md), md["mass"]),
    }
    out = {}
    for kind, (vis, pieces, mass) in todo.items():
        dims = cd if kind == "cap" else md
        lk = look[kind]
        key = content_key(kind, dims, material_of[kind], pieces, lk["color"])
        p = Path(out_dir) / f"{kind}_{key}.usda"
        if force or not p.exists():
            write_rigid_object(p, vis, pieces, mass, material_of[kind], lk["color"], lk["roughness"], lk.get("metallic", 0.0))
        out[kind] = p
    return out
