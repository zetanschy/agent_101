"""The printed parts and the real cameras, as Isaac Lab configs.

USD is generated from the STL by ./robot sim-assets, not committed -- the STL is
the source of truth and the conversion is deterministic.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass  # noqa: F401  (re-exported for task modules)

from ..cameras import CAMERAS

HERE = pathlib.Path(__file__).parent
USD = HERE / "usd"


@dataclasses.dataclass(frozen=True)
class TBlockGeometry:
    """Measured off t_20_factor_0.5_scaled.stl, in metres.

    The mesh origin sits at the middle of the crossbar, NOT at the centroid -- the
    stem hangs off toward -y. Anything that reasons about "where the T is" wants
    the centroid, so it is derived here once rather than re-guessed per call site.
    """

    bar_width: float = 0.080      # x extent of the crossbar
    bar_depth: float = 0.020      # y extent of the crossbar
    stem_width: float = 0.020     # x extent of the stem
    stem_length: float = 0.060    # y extent of the stem, hanging toward -y
    thickness: float = 0.016      # z extent
    volume_m3: float = 44.80e-6   # by mesh integration

    @property
    def centroid_offset(self) -> tuple[float, float, float]:
        """Centroid in mesh coordinates: the point a push should be measured against."""
        bar_a = self.bar_width * self.bar_depth
        stem_a = self.stem_width * self.stem_length
        cy = (bar_a * 0.0 + stem_a * -(self.bar_depth / 2 + self.stem_length / 2)) / (bar_a + stem_a)
        return (0.0, cy, self.thickness / 2)


T_BLOCK_GEOMETRY = TBlockGeometry()

# Printed PLA sliding on a rubber mat: grippy. Set too low, a nudge turns into a
# hockey shot no policy can undo. Isaac Lab 2.1 has no physics_material on
# UsdFileCfg (only shapes and the ground plane carry one), so the task applies this
# as the simulation's default material -- see PushTEnvCfg.__post_init__.
PLA_ON_MAT = dict(static_friction=0.9, dynamic_friction=0.75, restitution=0.0)

# The pushed T. Spawned from the converted USD so the collision stays the convex
# DECOMPOSITION the converter built -- a convex hull would fill both notches.
T_BLOCK_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/TBlock",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD}/t_block.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            # A flat plate on a mat comes to rest quickly; without sleep it jitters
            # forever and the "has it settled" check never fires.
            enable_gyroscopic_forces=True,
        ),
        # Printed in grey.
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.55, 0.57), roughness=0.75, metallic=0.0
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.0, T_BLOCK_GEOMETRY.thickness / 2)),
)

# The printed camera mount. It bolts into the SO-ARM101 wrist holes and rides with
# the wrist, so it is a visual child of the gripper body rather than a body of its
# own -- it must occlude the same sliver of the wrist frame as the real one, and it
# must never be something the solver can push against.
KLIP_SUPPORT_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/klip_support",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD}/klip_support.usd",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.9, 0.9, 0.9), roughness=0.8, metallic=0.0
        ),
    ),
)


def camera_cfg(
    name: str,
    prim_path: str,
    pos: tuple[float, float, float],
    rot_quat: tuple[float, float, float, float],
    *,
    width: int = 640,
    height: int = 480,
    data_types: tuple[str, ...] = ("rgb",),
    intrinsics: dict | None = None,
) -> TiledCameraCfg:
    """A TiledCamera whose optics match the real camera called `name`.

    Built through from_intrinsic_matrix rather than by setting focal_length by hand,
    so a calibration run (./robot sim-calibrate) flows straight through: it writes
    fx/fy/cx/cy into config/cameras.json and this picks them up untouched.
    """
    if intrinsics is None:
        from ..cameras import load

        intrinsics = load(width, height)
    k = intrinsics[name]
    if (k["width"], k["height"]) != (width, height):
        raise ValueError(f"{name}: intrinsics are {k['width']}x{k['height']}, camera wants {width}x{height}")
    spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[k["fx"], 0.0, k["cx"], 0.0, k["fy"], k["cy"], 0.0, 0.0, 1.0],
        width=width,
        height=height,
        # 1 cm near plane: the wrist camera sits centimetres from the jaws and a
        # default near plane clips them out of the frame entirely.
        clipping_range=(0.01, 20.0),
        focus_distance=0.4,
        f_stop=0.0,  # 0 disables depth of field; these are fixed-focus webcams
    )
    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=width,
        height=height,
        data_types=list(data_types),
        spawn=spawn,
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot_quat, convention="opengl"),
    )


def _describe() -> str:
    g = T_BLOCK_GEOMETRY
    lines = [
        f"T block: {g.bar_width*1000:.0f}x{(g.bar_depth+g.stem_length)*1000:.0f}x{g.thickness*1000:.0f} mm, "
        f"{g.volume_m3*1e6:.1f} cm^3, centroid offset {tuple(round(v, 4) for v in g.centroid_offset)}",
    ]
    for n, c in CAMERAS.items():
        lines.append(f"{n}: {c.model}, spec dFOV {c.diagonal_fov_deg} deg")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_describe())
