"""The printed parts and the real cameras, as Isaac Lab configs.

USD is generated from the STL by ./robot sim-assets, not committed -- the STL is
the source of truth and the conversion is deterministic.
"""

from __future__ import annotations

import math
import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass  # noqa: F401  (re-exported for task modules)

from ..cameras import CAMERAS

HERE = pathlib.Path(__file__).parent
USD = HERE / "usd"


from ..tblock import T_BLOCK_GEOMETRY, TBlockGeometry  # noqa: F401

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
BORE_CENTRE = (0.017, -0.0175, 0.0)     # bore axis through the plate, from the STL
BORE_DIAMETER = 0.019
# Base plate of the support. Confirmed against the built scene, not just the STL:
# sampling the support mesh within 16 mm of the bore axis gives exactly two face
# planes, z = 0.000 and z = 3.500 mm, 294 vertices each. So a 3.5 mm sliver between
# the webcam and the plate in the viewport is the plate's own thickness -- the body
# is attached to the far face, not floating. That is a side to choose, not a gap.
PLATE_SEAT_Z = 0.0
PLATE_TOP_Z = 0.0035
# Which face of the plate the webcam is glued to. +1 = the face the bore looks out
# of, which is the one you see; -1 = the hidden face. Both seat the body flush by
# construction, so a gap in the viewport means this is on the wrong side, not that
# the seating maths is off -- the base-face-to-plate distance measures 0.00 mm over
# 60 samples either way.
CAMERA_SIDE = -1

# The KWC-500's body is a rectangular block, not a cylinder -- only its lens barrel is
# round, and that is the part that drops into the bore. The block's base is GLUED to
# the plate, so it starts exactly at PLATE_SEAT_Z with no gap.
BODY_W, BODY_D, BODY_L = 0.030, 0.026, 0.016
# The lens fills the bore and sits nearly flush with the far face -- it is a
# webcam lens in a 3.5 mm plate, not a lens sticking 8 mm into the workspace.
BARREL_LENGTH = PLATE_TOP_Z + 0.0005

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
# Placed by hand in the viewport, against the two wrist holes, and LEFT THERE.
#
# It was briefly translated 30 mm off those holes so that the bore would land on the
# lens position the hand-eye solve reported. That was the wrong way round. The
# support bolts to two holes -- a hard mechanical constraint anyone can check by
# eye -- while the wrist calibration is the least trustworthy number in this repo:
# two solves of the same lens disagree by 70 px on cx, because every board view sat
# at roughly one distance and cx then trades off freely against board position.
# Moving a bolted part to satisfy that is fitting the geometry to the weaker
# measurement. The holes win; the camera follows the bore.
#
# Not solved, and that is deliberate. I tried repeatedly to snap this onto a detected
# hole pair and every attempt was worse than the hand placement: matching by diameter
# found nothing (the mount's dia-4.0 holes are clearance for dia-2.0 screws), matching
# by proximity was circular whenever the starting pose was off, and matching by
# spacing found a pair whose four orientations penetrate the wrist by 1.4 to 12 mm.
# The hand placement beats all of them, so it stands.
#
# If you want to adjust it: ./robot sim-play --gui --pose writes the total pose to
# sim/outputs/mount_pose.txt on exit -- paste it here and re-run ./robot sim-assets,
# since the pose is baked into the USD.
KLIP_MOUNT_POS = (-0.01455, 0.08710, -0.01039)
KLIP_MOUNT_ROLL_DEG = -179.755
KLIP_MOUNT_PITCH_DEG = -34.920
KLIP_MOUNT_YAW_DEG = -90.061


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


def _quat_rotate(q, v):
    w, x, y, z = q
    t = (2*(y*v[2] - z*v[1]), 2*(z*v[0] - x*v[2]), 2*(x*v[1] - y*v[0]))
    return (v[0] + w*t[0] + (y*t[2] - z*t[1]),
            v[1] + w*t[1] + (z*t[0] - x*t[2]),
            v[2] + w*t[2] + (x*t[1] - y*t[0]))


def mount_local_to_gripper(p):
    """A point in the support's own frame, expressed relative to the gripper body.

    The support's placement is BAKED into its USD (see scripts/sim_convert_assets.py),
    so its prim sits at identity and anything that must ride with it -- the webcam, the
    wrist camera -- cannot simply be parented under it and inherit the pose. They are
    siblings, positioned through here instead, off the same KLIP_MOUNT_* constants.
    """
    off = _quat_rotate(KLIP_MOUNT_ROT, p)
    return tuple(KLIP_MOUNT_POS[i] + off[i] for i in range(3))


KLIP_SUPPORT_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/klip_support",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD}/klip_support.usd",
        # Printed in grey. Safe to bind here now that the webcam is a SIBLING rather
        # than a child: a material bound at this prim inherits down the hierarchy, and
        # when the camera lived underneath it this repainted it grey too.
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.45, 0.45, 0.47), roughness=0.85, metallic=0.0
        ),
    ),
    # No init_state: the mount pose is baked into the USD at conversion time, because
    # Isaac Lab does not apply a child AssetBaseCfg's init_state as the prim's local
    # transform -- config and property panel disagreed by 142 mm.
)

# The webcam. Where the lens actually is comes from calibration when available --
# the hand-posed mount put the bore 22 mm from the measured lens, and with the model
# built around the guess the camera rendered the inside of its own body. Building it
# around the measured pose instead means the lens is at the camera and the body sits
# BEHIND it, which is both physically right and impossible to look into.
# Which way up the webcam sits IN the bore. The mount pose cannot carry this --
# the bore is a circle, so the body goes in at any roll and the STL says nothing
# about it. A quarter turn, not a tuning knob: of 0/90/180/270 only +90 puts the
# jaws where the real camera puts them.
CAMERA_BORE_ROLL_DEG = 90.0

# Where the camera actually is, and this WINS over the bore derivation below.
#
# Hand-tuned in the viewport to (0.00295, 0.08000, -0.00470) / (-45.864, -1.498,
# -5.688), then dollied 24 mm further along the optical axis, toward the gripper,
# with the focal widened to match, then a final Nelder-Mead over all nine camera
# parameters (pose, focal, principal point) against the real frame, each evaluation
# an ACTUAL Isaac render: 700 of them in one Kit session. That last pass moved the
# camera 2.9 mm and 2.2 deg and took the silhouette's mean boundary error from 17 px
# to 3.1 px (IoU 0.932). cx/cy shifted by 2 px, which is the confirmation that the
# earlier grid had them right.
#
# That pairing is a DOLLY ZOOM and the two halves cannot be separated: closer plus
# wider leaves the subject the same size and only changes how much the near end
# diverges from the far end. Silhouette IoU is nearly blind to it -- once the
# gripper overflows the frame the overflow stops contributing -- so it was fitted
# on the width PROFILE instead: a pure zoom scales every row by one constant,
# while a camera at the wrong distance tilts that ratio top-to-bottom. That tilt
# went from 1.45 to 1.04, mean silhouette edge error to 17 px, IoU to 0.905.
#
# It is worth being clear that this is now an EMPIRICAL pose, not a derived one:
# 20 deg of total rotation away from the bore is more than a barrel can rock in a
# 19 mm hole, so something upstream -- the mount pose, or the gripper link frame
# the workshop USD defines -- carries error this is absorbing. It matches the real
# camera, which is what sim2real needs, but do not read it as a measurement of
# where the lens sits relative to the mount.
#
# Fitted jointly with the focal, with cx/cy pinned to the image centre. That
# pinning is what makes the pose meaningful at all: a lens shift and a camera tilt
# are indistinguishable from one view, so letting both float just trades one for
# the other -- see the note in config/cameras.json.
#
# Euler in DEGREES, in the order Isaac's property panel shows them. Set either to
# None to fall back to the pure bore derivation.
CAMERA_POS_OVERRIDE = (0.00313, 0.06116, -0.01911)
CAMERA_EULER_OVERRIDE_DEG = (-44.989, -1.976, -3.675)


def _quat_mul(a, b):
    """Hamilton product, (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def _quat_from_euler_deg(ex, ey, ez):
    """(w, x, y, z) from an Isaac property-panel Orient triple, in degrees.

    Composed Rx @ Ry @ Rz. With this camera's angles the candidate conventions
    span only 0.4 degrees, so the choice is not load-bearing here -- but it would
    be for a pose with two or three large angles, so it is written down rather
    than left implicit.
    """
    hx, hy, hz = (math.radians(a) / 2 for a in (ex, ey, ez))
    qx = (math.cos(hx), math.sin(hx), 0.0, 0.0)
    qy = (math.cos(hy), 0.0, math.sin(hy), 0.0)
    qz = (math.cos(hz), 0.0, 0.0, math.sin(hz))
    return _quat_mul(_quat_mul(qx, qy), qz)


def _webcam_placement():
    """(camera pos, barrel pos, body pos, camera rot) in the gripper frame.

    Everything here comes from the HAND-PLACED mount. The barrel and body lines
    are untouched and keep KLIP_MOUNT_ROT, so the support assembly stays exactly
    where it was put; only the camera is derived.

    The camera is NOT at the barrel's midpoint. Put it there and it sits inside
    its own black housing and renders a solid black frame -- it goes at the bore
    MOUTH. CAMERA_SIDE = -1 glues the body to the hidden face, so the bore looks
    out along the mount's local +Z, and a USD camera looks along its own -Z:
    hence the half turn about X, then the quarter turn for the bore roll.
    """
    barrel = mount_local_to_gripper((BORE_CENTRE[0], BORE_CENTRE[1],
                                     PLATE_SEAT_Z + (PLATE_TOP_Z - PLATE_SEAT_Z) / 2))
    body = mount_local_to_gripper((BORE_CENTRE[0], BORE_CENTRE[1],
                                   PLATE_SEAT_Z + CAMERA_SIDE * BODY_L / 2))
    cam = mount_local_to_gripper((BORE_CENTRE[0], BORE_CENTRE[1], PLATE_TOP_Z))
    half = math.radians(CAMERA_BORE_ROLL_DEG) / 2
    quat = _quat_mul(KLIP_MOUNT_ROT, (0.0, 1.0, 0.0, 0.0))
    quat = _quat_mul(quat, (math.cos(half), 0.0, 0.0, math.sin(half)))
    if CAMERA_POS_OVERRIDE is not None:
        cam = tuple(CAMERA_POS_OVERRIDE)
    if CAMERA_EULER_OVERRIDE_DEG is not None:
        quat = _quat_from_euler_deg(*CAMERA_EULER_OVERRIDE_DEG)
    return cam, barrel, body, quat


_CAM_POS, _LENS_POS, _BODY_POS, _CAM_ROT = _webcam_placement()


def wrist_camera_in_gripper():
    """Where the wrist camera goes: the bore of the hand-placed mount."""
    return _CAM_POS, _CAM_ROT

KWC500_BODY_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/kwc500_body",
    spawn=sim_utils.CuboidCfg(
        size=(BODY_W, BODY_D, BODY_L),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.02, 0.02), roughness=0.5, metallic=0.0
        ),
        collision_props=None,
        rigid_props=None,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=_BODY_POS, rot=KLIP_MOUNT_ROT),
)
KWC500_BARREL_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/kwc500_barrel",
    spawn=sim_utils.CylinderCfg(
        radius=BORE_DIAMETER / 2 - 0.0004,
        height=BARREL_LENGTH,
        axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.02, 0.02), roughness=0.3, metallic=0.05
        ),
        collision_props=None,
        rigid_props=None,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=_LENS_POS, rot=KLIP_MOUNT_ROT),
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
    near_m: float = 0.01,
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
        # Near plane. The wrist camera sits centimetres from the jaws, so the
        # default would clip them out of the frame entirely -- but it also has to
        # be FAR enough to clip the camera's own modelled housing.
        #
        # The fitted optical centre lands about 12 mm behind the bore mouth, i.e.
        # inside the KWC-500's body, which is where a webcam's optical centre
        # actually is: behind the front element, in a 16 mm-deep case. A real
        # camera cannot see its own shell; a simulated one at the same point looks
        # straight at the inside of the cuboid standing in for it and renders
        # 10-13 mm of black. Clipping past the housing is the fix -- moving the
        # camera out in front of it instead is what put the focal length 14% long
        # and made the datasheet look wrong.
        clipping_range=(near_m, 20.0),
        focus_distance=0.4,
        f_stop=0.0,  # 0 disables depth of field; these are fixed-focus webcams
    )

    # Put the principal point back. from_intrinsic_matrix() discards it --
    # isaaclab/utils/sensors.py returns horizontal/vertical_aperture_offset = 0.0
    # unconditionally, warning that Omniverse cannot do aperture offsets. That
    # warning is stale: the RTX renderer honours both attributes exactly. Measured
    # -- a 0.25 offset on a 1.0 aperture moves the image by 0.25*width, to the
    # pixel, and the vertical does the same. Left at zero, every camera renders as
    # though its optical axis were dead centre, which throws the wrist camera's
    # cy out by 30 px and the overhead's by 60.
    #
    # The signs are measured, not derived: a POSITIVE horizontal offset slides the
    # image content left, so cx falls; a positive vertical offset slides it down,
    # so cy rises.
    #   cx = W/2 - (off_h / aperture_h) * W
    #   cy = H/2 + (off_v / aperture_v) * H
    ah = spawn.horizontal_aperture
    # Omniverse derives the vertical aperture from the horizontal one and the render
    # aspect ratio, ignoring the vertical_aperture attribute (also measured: scaling
    # it by 1.2 changes nothing). Use the derived value, which is what it acts on.
    av = ah * height / width
    spawn.horizontal_aperture_offset = ah * (width / 2 - k["cx"]) / width
    spawn.vertical_aperture_offset = av * (k["cy"] - height / 2) / height

    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=width,
        height=height,
        data_types=list(data_types),
        spawn=spawn,
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot_quat, convention="opengl"),
    )
