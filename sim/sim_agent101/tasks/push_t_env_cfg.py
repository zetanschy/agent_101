"""Push the printed T to a goal pose, with this workspace's real cameras.

Scene geometry (arm pose, mat placement, lightbox) follows the workshop so the two
tasks share a world; only the manipulated object and the cameras are ours.

CAMERA EXTRINSICS ARE THE WEAK POINT and are marked below. Intrinsics are pinned to
the real lenses (measured crop behaviour + spec dFOV, replaceable by calibration),
but where the cameras SIT is currently derived from the real frames by eye. Run
`./robot sim-compare-cameras` to put the sim render next to a live capture and
nudge OVERHEAD_* / WRIST_* until they agree. A policy trained on a wrist view that
is 2 cm off will not transfer, however good the intrinsics are.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from .. import mdp
from ..assets.objects import (
    KLIP_MOUNT_POS,
    KLIP_MOUNT_ROT,
    KLIP_SUPPORT_CFG,
    KWC500_BARREL_CFG,
    KWC500_BODY_CFG,
    PLA_ON_MAT,
    T_BLOCK_CFG,
    T_BLOCK_GEOMETRY,
    camera_cfg,
    wrist_camera_in_gripper,
)
from ..assets.so101 import JOINT_NAMES, SO101_CFG, WORKSHOP_USD

# Mat top surface, from the workshop's scene. Everything on the table is placed
# relative to this so a change to the mat does not silently bury the T.
MAT_Z = 0.032
# Top face of the mat, where things actually rest. Not MAT_Z: mat.usda has its own
# thickness, and this is where a dropped T was measured to settle. Spawning at MAT_Z
# + half-thickness instead makes the T fall 5 mm at every reset and burn the first
# frames of the episode on a settling transient.
MAT_SURFACE = 0.035
# The real mat is 93 x 55.5 cm, laid with its long side across the bench: in WORLD
# axes 0.555 along x (forward from the base) by 0.93 along y. mat.usda is the
# workshop's 18 x 12 inch desk mat, a unit cube pre-scaled to 0.4572 x 0.3048 x 0.006,
# spawned here with a 90 degree yaw so its local x lies along world y. MAT_SCALE
# stretches it to the real size on top of that bake; thickness is left alone.
MAT_SIZE_XY = (0.555, 0.93)
_WORKSHOP_MAT_LOCAL_XY = (0.4572, 0.3048)          # local x, local y, as baked in mat.usda
MAT_SCALE = (MAT_SIZE_XY[1] / _WORKSHOP_MAT_LOCAL_XY[0],   # local x -> world y
             MAT_SIZE_XY[0] / _WORKSHOP_MAT_LOCAL_XY[1],   # local y -> world x
             1.0)
# Near edge flush with the BACK of the base plate, so the whole base sits on the
# mat, as it does on the real bench. Not the robot's placement origin: the base
# reaches 46.7 mm behind that (measured off the built scene -- /Robot/base spans
# x -96.7..-1.1 mm with the robot placed at -50), so anchoring on the origin puts
# the mat's edge through the middle of the base plate.
BASE_BEHIND_ORIGIN = 0.0467
MAT_CENTRE = (SO101_CFG.init_state.pos[0] - BASE_BEHIND_ORIGIN + MAT_SIZE_XY[0] / 2, 0.0)
MAT_YAW_DEG = 90.0          # lays mat.usda's long side across the bench (world y)
# mat.usda carries no diffuse colour of its own; it renders near-black. The table
# matches it so the two merge, as they do on the bench.
TABLE_COLOR = (0.03, 0.03, 0.03)

# Where the mat ACTUALLY is, and this WINS when set: (x, y, yaw_deg) in the env frame.
# The derivation above is the tidy version -- flush with the base, square to the arm.
# A hand-laid mat is neither: in the real overhead frame its far edge runs
# diagonally across the top, which no axis-aligned rectangle can reproduce. So the
# mat is placed by fitting its rendered footprint to the dark region of the real
# frame, with the camera held at the pose the arm silhouette fixed. Size stays 93 x
# 55.5 cm as measured. None = the derivation.
#
# What the fit found, and what it could not: the only edge in view is the TABLE's
# end (both are black), running at 59 deg to the env x axis. A 555 mm mat starting
# at the base rear would overhang that edge by 83-280 mm, and there is no black past
# it in the frame -- so the mat is laid square to the table and flush with its end,
# which puts its near edge about 162 mm BEHIND the base rear rather than at it. The
# overhead cannot see that edge (black on black); if the mat really does start at
# the base, the mat is not 555 deep and this number is the one to change.
MAT_POSE_OVERRIDE = (0.0144, -0.0385, 59.13)
if MAT_POSE_OVERRIDE is not None:
    MAT_CENTRE = (MAT_POSE_OVERRIDE[0], MAT_POSE_OVERRIDE[1])
    MAT_YAW_DEG = MAT_POSE_OVERRIDE[2]

# Where the T and the goal may sit. NOT the mat's centre any more: now that the mat
# starts at the back of the base, its centre is only 56 mm from the robot origin and
# the base plate itself reaches x = -1 mm, so a T sampled about the mat centre spawns
# *inside the arm* and gets shoved out -- which reads as "T drifted 14 mm with no
# contact", a friction bug that is nothing of the sort.
#
# So the play area is anchored to the ARM, not the mat: a band in front of the base
# at the distance the arm comfortably works, clipped to stay on the mat. Anchoring
# on the mat's midpoint instead would march the T away from the robot every time
# the mat got bigger -- the real one is 93 cm long -- until it was out of reach.
BASE_AHEAD_ORIGIN = 0.0489      # /Robot/base reaches x = -1.1 mm with the robot at -50
PLAY_AHEAD_OF_BASE = 0.22       # centre of the band, forward of the base's leading edge
PLAY_HALF = 0.06                # +/- about that centre; 220 +/- 60 mm is well inside reach
_base_front = SO101_CFG.init_state.pos[0] + BASE_AHEAD_ORIGIN
_mat_far = MAT_CENTRE[0] + MAT_SIZE_XY[0] / 2
PLAY_CENTRE = (min(_base_front + PLAY_AHEAD_OF_BASE, _mat_far - PLAY_HALF - 0.03), 0.0)

# --- camera extrinsics ------------------------------------------------------
# Measured, when config/extrinsics.json exists (./robot calib-solve). Both cameras
# then sit where calibration says, in the robot's own frame, rather than where I
# guessed. The fallbacks below are those guesses and are wrong in known ways: the
# overhead estimate came from back-solving one frame's field of view and put the
# camera at 0.59 m when it is 1.11 m, and the wrist angle was set by eye.
from .. import extrinsics as _ex  # noqa: E402

_MEASURED = _ex.available()
if _MEASURED:
    OVERHEAD_POS, OVERHEAD_ROT = _ex.overhead_in_env()
else:
    OVERHEAD_POS = (MAT_CENTRE[0], MAT_CENTRE[1], MAT_SURFACE + 0.59)
    OVERHEAD_ROT = (math.cos(math.radians(-45.0)), 0.0, 0.0, math.sin(math.radians(-45.0)))

# Fitted overhead pose, and this WINS over the calibration when set. Same story as
# the wrist camera's CAMERA_POS_OVERRIDE in assets/objects.py: the hand-eye chain
# is the least trustworthy number in this repo, so the camera is placed by matching
# the rendered arm silhouette to the real overhead frame over actual Isaac renders.
# Env frame, opengl convention, quaternion (w, x, y, z) -- kept as a quaternion
# because nothing here needs to read it off a GUI, and an euler round-trip is one
# more place for a convention to go wrong. None = use the calibration above.
#
# Result of a 600-evaluation Nelder-Mead over the arm silhouette, each evaluation two
# real renders (robot on / off, for a shadow-immune depth-difference mask): the
# calibration's pose moved 14.6 mm and 1.4 deg, and its focal length and principal
# point held (993.7 -> 991.4, cy 300.4 -> 303.3) -- so the overhead calibration was
# GOOD, and the offset I once flagged as suspect in cy was real. Mean boundary error
# 14.4 -> 5.9 px on a ~100 px wide arm seen from 1.1 m.
OVERHEAD_POS_OVERRIDE = (-0.10061, -0.00532, 1.12434)
OVERHEAD_ROT_OVERRIDE = (0.697998, 0.079330, -0.047049, -0.710135)
if OVERHEAD_POS_OVERRIDE is not None:
    OVERHEAD_POS = tuple(OVERHEAD_POS_OVERRIDE)
if OVERHEAD_ROT_OVERRIDE is not None:
    OVERHEAD_ROT = tuple(OVERHEAD_ROT_OVERRIDE)

# The wrist camera comes from the hand-placed mount's bore, NOT from the hand-eye
# calibration -- and from objects.py rather than being worked out again here, so
# the camera and the modelled webcam around it can never disagree.
WRIST_POS, WRIST_ROT = wrist_camera_in_gripper()
# ------------------------------------------------------------------------------


def _quat_x(deg: float):
    half = math.radians(deg) / 2
    return (math.cos(half), math.sin(half), 0.0, 0.0)


def _quat_z(deg: float):
    half = math.radians(deg) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


@configclass
class PushTSceneCfg(InteractiveSceneCfg):
    """Arm, mat, lightbox, the T, its goal, and the two real cameras."""

    env_spacing = 4.0
    num_envs = 1

    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper")],
    )

    # NOT the workshop's lightbox-simple.usd. That is a studio enclosure with a roof,
    # and this workspace's real rig is an open table under room light -- with the box
    # in place the overhead camera, correctly positioned 0.59 m up, renders the inside
    # of its lid. A dome plus a soft key light is both closer to the real scene and
    # the thing domain randomisation should be perturbing.
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(0.85, 0.87, 0.92)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DiskLightCfg(intensity=6000.0, radius=0.15, color=(1.0, 0.96, 0.9)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.1, -0.35, 0.75)),
    )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{WORKSHOP_USD}/mat.usda", scale=MAT_SCALE),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(MAT_CENTRE[0], MAT_CENTRE[1], MAT_Z), rot=_quat_z(MAT_YAW_DEG)),
    )

    # The TABLE the mat lies on. Both are black, so in the real overhead frame the
    # mat's own edges are invisible and the only boundary you can see is the table's
    # end, running diagonally across the upper part of the frame. Fitting the "mat"
    # to that dark region put its near edge 162 mm behind the base -- impossible for
    # a mat that starts at the base -- which is how it became clear the line is the
    # table's. So: the mat stays where it was measured to be, and this table's far
    # edge sits on the fitted line. Its other edges are out of every camera's view;
    # the size is just "big enough". Same black as the mat, so the two merge as they
    # do in reality. Top face at MAT_Z - 3 mm, so the mat rests on it.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.4, 1.5, 0.028),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR, roughness=0.9),
            collision_props=None,
            rigid_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.3911, 0.2039, MAT_Z - 0.003 - 0.028/2), rot=_quat_z(59.13)
        ),
    )

    klip_support = KLIP_SUPPORT_CFG
    kwc500_barrel = KWC500_BARREL_CFG
    kwc500_body = KWC500_BODY_CFG

    t_block: RigidObjectCfg = T_BLOCK_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PLAY_CENTRE[0], PLAY_CENTRE[1], MAT_SURFACE)
        )
    )

    # Where the T has to end up: the same mesh, kinematic and see-through, so the
    # operator can aim during teleop. It must not collide -- it is a target, not a
    # second object on the mat.
    goal_marker: RigidObjectCfg = T_BLOCK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.UsdFileCfg(
            usd_path=T_BLOCK_CFG.spawn.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            # OPAQUE, deliberately. It used to be opacity=0.35, which needs
            # sim.render.enable_translucency -- and translucency is the single
            # biggest cost in the frame budget (39.2 -> 31.8 ms with it off, which
            # is the difference between making 30 fps and not). With it off a
            # translucent prim renders solid BLACK: a black T on a black mat, i.e.
            # the operator's target disappears exactly when recording starts. An
            # opaque marker costs nothing and reads like a printed target, which is
            # what it is on the real bench.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.72, 0.30), roughness=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(PLAY_CENTRE[0], PLAY_CENTRE[1], MAT_SURFACE + 0.001)),
    )

    camera_front = camera_cfg("front", "{ENV_REGEX_NS}/OverheadCam", pos=OVERHEAD_POS, rot_quat=OVERHEAD_ROT)
    # near_m past the webcam's own modelled housing -- see the note in camera_cfg.
    camera_grip = camera_cfg(
        "grip", "{ENV_REGEX_NS}/Robot/gripper/wrist_cam", pos=WRIST_POS, rot_quat=WRIST_ROT,
        near_m=0.02,
    )


@configclass
class ActionsCfg:
    joint_positions = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_NAMES, scale=1.0, use_default_offset=False
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_pos_rel = ObsTerm(func=base_mdp.joint_pos_rel)
        t_pose = ObsTerm(func=mdp.t_block_pose_obs)
        goal_pose = ObsTerm(func=mdp.goal_pose_obs)
        t_to_goal = ObsTerm(func=mdp.t_block_to_goal_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            # Kept as a dict: the lerobot recorder wants named streams, not one flat
            # vector it would have to slice back apart.
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    # Start region: a 16 x 16 cm patch in the middle of the mat, any yaw. Wide enough
    # that the policy cannot memorise one start, small enough that the T is always
    # inside the overhead frame.
    # The real arm is white; the USD ships yellow. Startup, not reset: it never changes.
    paint_robot_white = EventTerm(
        func=mdp.set_robot_color, mode="startup", params={"color": (0.93, 0.93, 0.95)}
    )

    reset_t_block = EventTerm(
        func=mdp.reset_t_block_pose,
        mode="reset",
        params={"pose_range": {"x": (-PLAY_HALF, PLAY_HALF), "y": (-PLAY_HALF, PLAY_HALF),
                               "yaw": (-math.pi, math.pi)}},
    )
    reset_goal = EventTerm(
        func=mdp.reset_goal_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (PLAY_CENTRE[0] - PLAY_HALF, PLAY_CENTRE[0] + PLAY_HALF),
                "y": (PLAY_CENTRE[1] - PLAY_HALF, PLAY_CENTRE[1] + PLAY_HALF),
                "yaw": (-math.pi, math.pi),
            }
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class PushTEnvCfg(ManagerBasedRLEnvCfg):
    """Teleop / data-collection variant: no success termination, no rewards."""

    scene: PushTSceneCfg = PushTSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards = None

    def __post_init__(self) -> None:
        # decimation 4 with sim.dt 1/120 gives step_dt = 1/30 exactly, which is the
        # rate our real datasets are recorded at -- and lerobot's aggregate_datasets
        # refuses to merge datasets whose fps differ, so this is not a preference.
        # sim.render_interval MUST track it: leaving it at 2 renders twice per step
        # and costs 20 ms a frame, which alone is the difference between hitting
        # 30 Hz and not.
        self.decimation = 4
        self.episode_length_s = 60.0
        self.scene.num_envs = 1
        # Looking over the operator's shoulder at the mat.
        self.viewer.eye = (-0.25, -0.45, 0.35)
        self.viewer.lookat = (0.22, 0.0, 0.05)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "quality"
        # Translucency OFF. It was on for the goal marker, which used to be 35%
        # transparent; the marker is opaque now precisely so this can be off, because
        # translucency is the biggest single item in the frame budget (39.2 -> 31.8 ms)
        # and 30 fps recording has only 33.3 ms to spend.
        self.sim.render.enable_translucency = False
        # Give the cameras a genuinely fresh frame after a reset, rather than the last
        # frame of the previous episode -- the first frame of every recorded episode
        # depends on it.
        self.rerender_on_reset = True
        # The T's contact with the mat is the whole task. Isaac Lab 2.1 cannot attach
        # a physics material to a USD-spawned rigid body, so set it as the sim-wide
        # default: the T is the only thing here sliding on anything.
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(**PLA_ON_MAT)


@configclass
class DREventCfg(EventCfg):
    """EventCfg plus domain randomization.

    Declared as class-level FIELDS, not assigned in __post_init__: @configclass
    turns class attributes into dataclass fields, and EventManager walks the
    fields. Attributes bolted on afterwards are set on the instance, show up in
    vars(), and are silently ignored by the manager -- the terms appear to exist
    and never run.

    Ranges, and why:
      physics   friction is the whole task -- it sets how far the T slides per
                push -- so it gets the widest treatment, then mass, then gains.
                Isaac Lab's randomizers rebase to the default on each reset, so
                scaling every episode does not compound.
      lighting  the widest range here. The real rig is an open bench under room
                light and this genuinely varies from session to session.
      cameras   the NARROWEST, deliberately: millimetres where the workshop uses
                centimetres. Both cameras were fitted against real frames (wrist
                3.1 px, overhead 5.9 px of boundary error). Randomising wider than
                the calibration residual throws the calibration away.
      colour    a few percent about white, not a four-colour palette. The real arm
                is white and we know it.
    """

    t_block_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("t_block"),
            "static_friction_range": (0.6, 1.2),      # PLA_ON_MAT default 0.9
            "dynamic_friction_range": (0.5, 1.0),     # default 0.75
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    t_block_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("t_block", body_names=".*"),
            "mass_distribution_params": (0.8, 1.3),   # x the measured 48 g
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    arm_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    lighting = EventTerm(
        func=mdp.randomize_lighting,
        mode="reset",
        params={"dome_intensity": (600.0, 1400.0),    # default 900
                "key_intensity": (3500.0, 8500.0),    # default 6000
                "color_temperature": (4500.0, 7500.0)},
    )
    camera_jitter = EventTerm(
        func=mdp.randomize_camera_pose,
        mode="reset",
        params={"cameras": {
            "/World/envs/env_0/OverheadCam": {
                "pos_m": (0.004, 0.004, 0.004), "rot_deg": (0.3, 0.3, 0.3)},
            "/World/envs/env_0/Robot/gripper/wrist_cam": {
                "pos_m": (0.002, 0.002, 0.002), "rot_deg": (0.5, 0.5, 0.5)},
        }},
    )
    robot_shade = EventTerm(
        func=mdp.randomize_robot_color, mode="reset",
        params={"grey": (0.86, 0.97), "tint": 0.03},
    )


@configclass
class PushTDREnvCfg(PushTEnvCfg):
    """push-T with domain randomization, for data meant to transfer.

    A separate task id rather than a flag on the plain env, so that "which scene
    was this recorded in" is answerable from the dataset's task name alone.
    """

    events: DREventCfg = DREventCfg()


@configclass
class PushTEvalEnvCfg(PushTEnvCfg):
    """Adds the success test, so a rollout can be scored instead of just watched."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 30.0
        self.terminations.success = DoneTerm(
            func=mdp.t_block_at_goal,
            params={"pos_tol": 0.02, "yaw_tol_deg": 15.0, "require_settled": True},
        )
