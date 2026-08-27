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

# --- the printed camera mount, and the webcam that sits in it --------------------
#
# klip_support-1 measured off the STL (all mm, in the STL's own frame):
#   bbox            55 x 35 x 50, origin at a corner (x 0..55, y -35..0, z 0..50)
#   camera bore     dia 19.0 through the base plate, centred (17.0, -17.5), axis Z
#   arm             rises diagonally from the plate to z=50, where it meets the wrist
#
# The bore is what matters: the KWC-500's lens barrel drops into it, so the bore axis
# IS the camera's optical axis and the bore centre is where the lens sits.
BORE_CENTRE = (0.017, -0.0175, 0.0)     # on the underside of the plate
BORE_DIAMETER = 0.019
PLATE_THICKNESS = 0.0035
# Which face of the plate the webcam sits against. -1 = the underside (z = 0),
# +1 = the top face (z = PLATE_THICKNESS).
CAMERA_SIDE = -1
# Face of the plate the camera stack grows out from. Everything below is measured
# from here, so the BASE of each cylinder lands on the plate and the body extends
# away from it -- not the far end touching while the barrel floats through.
CAMERA_SEAT_Z = 0.0 if CAMERA_SIDE < 0 else PLATE_THICKNESS
BARREL_LENGTH = 0.012
BODY_LENGTH = 0.016

# Where the mount meets the arm, relative to the `gripper` body.
#
# Three numbers to turn, deliberately kept as euler degrees rather than a quaternion
# so they can be nudged by hand. Rotation order is X then Y then Z, applied to the
# support's own frame (origin at a corner of the base plate, bore axis +Z).
#
#   ROLL  about X: swings the bore plate between pointing down and pointing forward
#   PITCH about Y: tilts the camera's look direction
#   YAW   about Z: which side of the wrist the mount sits on
#
# Not derived from the bolt holes -- the support's arm section has only C-shaped
# notches ~8 mm apart, which do not match the 31 mm bolt spacing of the SO-ARM101's
# own camera mount, so the mesh alone never said how the two mate. Posed by eye
# instead. To adjust:  ./robot sim-play --gui --pose
# Posed by hand in the viewport against the real print, then read off the property
# panel: Translate (0.07456, 0.0919, -0.02029), Orient (90.003, 0.002, -179.983).
# Rounded only where the panel's own noise made it obvious (90.003 -> 90).
KLIP_MOUNT_POS = (0.07456, 0.09190, -0.02029)
KLIP_MOUNT_ROLL_DEG = 90.0
KLIP_MOUNT_PITCH_DEG = 0.0
KLIP_MOUNT_YAW_DEG = -180.0


def _euler_quat(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """(w, x, y, z) for an intrinsic X-Y-Z rotation, in degrees."""
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


KLIP_MOUNT_ROT = _euler_quat(KLIP_MOUNT_ROLL_DEG, KLIP_MOUNT_PITCH_DEG, KLIP_MOUNT_YAW_DEG)


KLIP_SUPPORT_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/klip_support",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD}/klip_support.usd",
        # No visual_material here on purpose. Isaac Lab binds it at the klip_support
        # prim, and USD material bindings inherit down -- which repainted the black
        # webcam below it white. The converted STL already renders as light printed
        # plastic, which is what the real part is.
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=KLIP_MOUNT_POS, rot=KLIP_MOUNT_ROT),
)

# The webcam itself. The KWC-500 has no published CAD, so it is two cylinders sized to
# the bore it drops into. Both are seated BASE-first on the plate face and grow away
# from it, rather than passing through it with only their far end touching. Nothing
# reads their geometry -- they are here to occlude what the real camera occludes.
KWC500_BARREL_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/klip_support/kwc500_barrel",
    spawn=sim_utils.CylinderCfg(
        radius=BORE_DIAMETER / 2 - 0.0004,   # a hair under the bore, so it seats
        height=BARREL_LENGTH,
        axis="Z",
        # The KWC-500 is a black webcam.
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.02, 0.02), roughness=0.35, metallic=0.05
        ),
        collision_props=None,
        rigid_props=None,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(BORE_CENTRE[0], BORE_CENTRE[1], CAMERA_SEAT_Z + CAMERA_SIDE * BARREL_LENGTH / 2)
    ),
)
KWC500_BODY_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/klip_support/kwc500_body",
    spawn=sim_utils.CylinderCfg(
        radius=0.014,
        height=BODY_LENGTH,
        axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.03, 0.03, 0.03), roughness=0.6, metallic=0.0
        ),
        collision_props=None,
        rigid_props=None,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(BORE_CENTRE[0], BORE_CENTRE[1],
             CAMERA_SEAT_Z + CAMERA_SIDE * (BARREL_LENGTH + BODY_LENGTH / 2))
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
