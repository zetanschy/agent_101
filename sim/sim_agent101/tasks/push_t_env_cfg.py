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
MAT_CENTRE = (0.22, 0.0)

# --- camera extrinsics: TUNE THESE AGAINST REAL FRAMES ------------------------
# Overhead C270. The height is not a guess: the real overhead frame spans roughly
# 0.40 m of mat across its 37.6-degree horizontal FOV, which puts the lens
# 0.20 / tan(18.8 deg) = 0.59 m above the surface, looking straight down.
OVERHEAD_POS = (MAT_CENTRE[0], MAT_CENTRE[1], MAT_SURFACE + 0.59)
# Identity looks straight down (opengl convention). The extra yaw matches the real
# frame's orientation: on the real overhead camera the arm enters from the bottom of
# the image and reaches up, which identity renders as left-to-right.
OVERHEAD_YAW_DEG = -90.0

# Wrist KWC-500. Explicit, and NOT derived from the mount -- which is unsatisfying and
# worth fixing. Composing the camera out of the posed mount plus the bore offset was
# tried and abandoned: every sign convention I tried put the lens either against the
# side of the wrist or inside its own camera body (a pure black frame, which the
# sim-play check catches). These values reproduce the real wrist view -- jaws low in
# shot, workspace beyond -- and were arrived at by matching against a real capture.
#
# Consequence to know about: re-posing klip_support does NOT move the camera. Move
# both, or resolve the bore-axis convention and derive one from the other.
WRIST_POS = (-0.005, 0.055, -0.055)
WRIST_PITCH_DEG = -40.0
WRIST_ROT = (0.939693, -0.34202, 0.0, 0.0)   # _quat_x(WRIST_PITCH_DEG), inlined: it is defined below this block
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
        spawn=sim_utils.UsdFileCfg(usd_path=f"{WORKSHOP_USD}/mat.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(MAT_CENTRE[0], MAT_CENTRE[1], MAT_Z), rot=_quat_z(90)),
    )

    klip_support = KLIP_SUPPORT_CFG
    kwc500_barrel = KWC500_BARREL_CFG
    kwc500_body = KWC500_BODY_CFG

    t_block: RigidObjectCfg = T_BLOCK_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(MAT_CENTRE[0], MAT_CENTRE[1], MAT_SURFACE)
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.75, 0.3), opacity=0.35, roughness=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(MAT_CENTRE[0], MAT_CENTRE[1], MAT_SURFACE + 0.001)),
    )

    camera_front = camera_cfg("front", "{ENV_REGEX_NS}/OverheadCam", pos=OVERHEAD_POS, rot_quat=_quat_z(OVERHEAD_YAW_DEG))
    camera_grip = camera_cfg(
        "grip", "{ENV_REGEX_NS}/Robot/gripper/wrist_cam", pos=WRIST_POS, rot_quat=WRIST_ROT
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
        params={"pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "yaw": (-math.pi, math.pi)}},
    )
    reset_goal = EventTerm(
        func=mdp.reset_goal_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (MAT_CENTRE[0] - 0.06, MAT_CENTRE[0] + 0.06),
                "y": (MAT_CENTRE[1] - 0.06, MAT_CENTRE[1] + 0.06),
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
        self.decimation = 2
        self.episode_length_s = 60.0
        self.scene.num_envs = 1
        # Looking over the operator's shoulder at the mat.
        self.viewer.eye = (-0.25, -0.45, 0.35)
        self.viewer.lookat = (0.22, 0.0, 0.05)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "quality"
        self.sim.render.enable_translucency = True   # the goal marker is see-through
        # The T's contact with the mat is the whole task. Isaac Lab 2.1 cannot attach
        # a physics material to a USD-spawned rigid body, so set it as the sim-wide
        # default: the T is the only thing here sliding on anything.
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(**PLA_ON_MAT)


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
