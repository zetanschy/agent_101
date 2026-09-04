#!/usr/bin/env python3
"""Convert the printed CAD (STL) into USD that Isaac Sim can simulate.

    ./robot sim-assets            # convert everything that is missing
    ./robot sim-assets --force    # reconvert

Two parts, and they want opposite treatment:

  t_20_factor_0.5  the pushed T. 100 x 100 mm footprint, 25 mm bar and stem,
      16 mm thick, 44.8 cm^3 by mesh volume. It is a DYNAMIC rigid body, and its
      collision must be a convex DECOMPOSITION: a T is non-convex, and the convex
      hull fills in both notches, turning a shape that catches the gripper into a
      pentagon that slides off it. That single flag is the difference between a
      push policy that transfers and one that does not.

  klip_support-1  the printed camera mount that bolts into the SO-ARM101 wrist
      holes. It rides on the wrist, so it is kinematically part of the arm --
      converted as a visual with no rigid body of its own. Its only job in sim is
      to sit where it sits in real life, so the wrist camera looks out from the
      right place and the mount occludes the same sliver of the frame.

Mass defaults to a 20%%-infill PLA estimate (~48 g for the 87.5 cm3 T). Weigh the real part
and pass --t-mass if you want it exact -- pushing dynamics are sensitive to it.
"""

import argparse
import os
import shutil
import sys
import tempfile
import traceback
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--force", action="store_true", help="reconvert even if the USD exists")
parser.add_argument("--t-mass", type=float, default=0.048,
                    help="mass of the printed T in kg; 87.5 cm3 PLA at ~20%% infill (default 0.048)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[2] / "sim" / "sim_agent101" / "assets"
CAD, USD = HERE / "cad", HERE / "usd"


def convert(stl: str, name: str, *, dynamic: bool, mass: float | None,
            translation=(0.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0)) -> pathlib.Path:
    out = USD / f"{name}.usd"
    if out.exists() and not args.force:
        print(f"  {name}: already converted ({out.relative_to(HERE.parents[2])})")
        return out
    # Isaac Lab 2.1's MeshConverter does basename.split(".") and unpacks two values,
    # so any dot in the stem -- "t_20_factor_0.5.stl" -- crashes it. Stage a
    # copy under a dot-free name rather than renaming the user's CAD.
    staged = pathlib.Path(tempfile.mkdtemp(prefix="sim_assets_")) / f"{name}.stl"
    shutil.copyfile(CAD / stl, staged)
    cfg = MeshConverterCfg(
        asset_path=str(staged),
        usd_dir=str(USD),
        usd_file_name=f"{name}.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        translation=translation,
        rotation=rotation,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        # convexDecomposition, not convexHull: see the module docstring. This is
        # Isaac Lab 2.1's spelling; 2.3 moved it to mesh_collision_props.
        collision_approximation="convexDecomposition",
    )
    if dynamic:
        cfg.mass_props = sim_utils.MassPropertiesCfg(mass=mass)
        cfg.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # A thin plate skidding on a mat: let the solver work, and do not let a
            # deep penetration fling it across the table.
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        )
    else:
        cfg.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        cfg.collision_approximation = "none"
    MeshConverter(cfg)
    print(f"  {name}: converted -> {out.name}")
    return out


def main() -> int:
    USD.mkdir(parents=True, exist_ok=True)
    missing = [f for f in ("t_20_factor_0.5.stl", "klip_support-1.stl") if not (CAD / f).exists()]
    if missing:
        raise SystemExit(f"missing CAD in {CAD}: {missing}")
    print("converting CAD -> USD")
    convert("t_20_factor_0.5.stl", "t_block", dynamic=True, mass=args.t_mass)
    # The mount's pose is BAKED IN here rather than set through AssetBaseCfg.init_state.
    # Isaac Lab does not apply a child AssetBaseCfg's init_state as the prim's local
    # transform, so the value in the config and the one Isaac's property panel shows
    # were different quantities -- 142 mm apart, which is what made the mount look
    # placed in the viewport while every measurement said it was floating. Baking makes
    # the geometry itself carry the pose, so there is exactly one number to trust.
    #
    # Consequence: re-run this after changing KLIP_MOUNT_* in assets/objects.py. The
    # cameras follow from code (mount_local_to_gripper) and do not need reconversion.
    from sim_agent101.assets.objects import KLIP_MOUNT_POS, KLIP_MOUNT_ROT

    print(f"  klip_support: baking pose {tuple(round(v, 5) for v in KLIP_MOUNT_POS)} "
          f"rot {tuple(round(v, 5) for v in KLIP_MOUNT_ROT)}")
    convert("klip_support-1.stl", "klip_support", dynamic=False, mass=None,
            translation=tuple(KLIP_MOUNT_POS), rotation=tuple(KLIP_MOUNT_ROT))
    print(f"\nUSD in {USD}")
    for f in sorted(USD.glob("*.usd")):
        print(f"  {f.name}  {f.stat().st_size / 1024:.0f} KB")
    return 0


code = 1
try:
    code = main()
except BaseException:  # noqa: BLE001 - os._exit below would swallow the traceback
    traceback.print_exc()
finally:
    # Deliberately NOT app.close(): Kit 4.5 hangs inside it on this box. A headless
    # run prints its result and never returns from close(), so calling it first does
    # not help. Everything we care about is on disk by here; let the OS reclaim.
    #
    # os._exit skips BOTH the stdio flush and the traceback, so both are done by hand
    # above and below -- without them a failure here exits 1 in total silence.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
