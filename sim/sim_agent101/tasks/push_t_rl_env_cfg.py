"""Push the T with RL: mjlab's objective, this workspace's bench and camera.

The teleop push-T scene (push_t_env_cfg.py) exists to be driven by hand and recorded;
it has no rewards. This one is the RL twin, and the reward is not invented here -- it
is ported from mjlab's Mjlab-Push-T-Yam-D435-Push, the variant whose gripper is
penalized for being open so the task stays a PUSHING problem rather than becoming a
scoop. The shapes and the reasoning behind each term are in mdp/push_t_rl.py.

Four things are ours rather than mjlab's:

  BLACK TABLE, with collision. mjlab pushes on its own terrain; the teleop scene here
  uses the workshop's mat USD over a visual-only table. This env has one black table
  that is the actual collider, so the scene is a cuboid per env instead of a USD --
  which matters when it is cloned four thousand times.

  THE WRIST CAMERA AND ITS PRINTED SUPPORT, from assets/objects.py: the KWC-500 in
  the klip_support bracket, at the pose fitted against real frames (3.1 px of
  silhouette error) with the mount modelled around it. The camera is IN THE SCENE but
  NOT in the policy's observation -- see ObservationsCfg.

  THE 10 CM T. Ours, printed, measured in tblock.py: a 100 mm crossbar, a 75 mm stem,
  20 mm thick. t_20_factor_0.5.stl is the 20 cm design at half scale, so the file name
  says 20 and the part is 10.

  A TIGHTER GOAL BOX. mjlab randomizes the goal over +/-2 cm in x and +/-10 cm in y.
  Through this wrist camera that is too wide: at 10 cm of lateral offset the goal
  leaves the frame, and mjlab hit exactly this on its own D435 variant -- the goal
  rendered at 0 px and coverage above 0.90 collapsed from 62% to 19%, because a policy
  can place the block coarsely from goal_pose as state but cannot close the last
  millimetres against a target it never sees. GOAL_* below is sized to what this
  camera covers.

Because the scene carries a camera, this task needs --enable_cameras:

    ./robot sim-train --task Agent101-So101-Push-T-RL --enable_cameras
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .. import mdp
from ..assets.objects import (
    KLIP_SUPPORT_CFG,
    KWC500_BARREL_CFG,
    KWC500_BODY_CFG,
    PLA_ON_MAT,
    T_BLOCK_CFG,
    T_BLOCK_GEOMETRY,
    camera_cfg,
    wrist_camera_in_gripper,
)
from ..assets.so101 import JOINT_NAMES, SO101_CFG

# --- the bench --------------------------------------------------------------
# Top face where the teleop scene's mat surface is, so the T, the arm and the camera
# all sit at the heights the real numbers were measured at. 2.5 cm thick and 1.2 x 1.0
# m: big enough that nothing runs off it, small enough to be one cheap cuboid.
TABLE_TOP = 0.035
TABLE_THICKNESS = 0.025
# Only as big as the task. It was 1.2 x 1.0 m, which at env_spacing 1.5 puts each
# env's table AABB almost against its neighbours' and hands the broadphase 4096 large
# overlapping boxes to sort out. 0.8 x 0.7 still covers the whole workspace box
# (0.44 x 0.60) and the arm's base behind it, with room to spare.
TABLE_SIZE = (0.8, 0.7)
TABLE_COLOR = (0.03, 0.03, 0.03)

# --- where the task happens -------------------------------------------------
# Anchored on the ARM, not on the table: a band in front of the base at the distance
# it comfortably works. Same reasoning as the teleop scene -- a play area centred on
# the furniture marches away from the robot every time the furniture changes.
BASE_AHEAD_ORIGIN = 0.0489
PLAY_CENTRE = (SO101_CFG.init_state.pos[0] + BASE_AHEAD_ORIGIN + 0.20, 0.0)

# THE SPAWN BOX AND THE GOAL BOX DO NOT OVERLAP, and that is not a detail. mjlab
# separates them by at least 5 cm on purpose -- "ranges keep the goal clear of the
# block's spawn box". Centre both on the same point and a fraction of episodes start
# already solved: the first version of this env did, and an UNTRAINED policy scored
# Episode_Reward/success = 0.021, which is the reward paying out for the reset draw
# rather than for anything the policy did. It also poisons the shaping, since those
# episodes hand out coverage and precision for free.
#
# So the block spawns 6 cm nearer the base than the goal sits, and the two x ranges
# are disjoint by 5.5 cm however the draws fall. The y ranges do overlap, as mjlab's
# do -- a pure sideways push is a legitimate task.
SPAWN_CENTRE = (PLAY_CENTRE[0] - 0.06, 0.0)
GOAL_CENTRE = (PLAY_CENTRE[0] + 0.06, 0.0)

# Offsets from SPAWN_CENTRE. mjlab uses +/-5 cm by +/-12 cm; y is pulled in because
# this arm reaches 0.32 m against the YAM's, not because of the camera.
SPAWN_X = (-0.035, 0.035)
SPAWN_Y = (-0.07, 0.07)

# Absolute, around GOAL_CENTRE. Deliberately smaller than mjlab's +/-0.02 x +/-0.10 --
# see the module docstring: past about 6 cm of lateral offset the footprint leaves
# this camera's frame and the last millimetres stop being learnable.
GOAL_X = (-0.03, 0.03)
GOAL_Y = (-0.05, 0.05)

# Episode ends if the block is pushed outside this, in env-local coordinates.
WORKSPACE_X = (PLAY_CENTRE[0] - 0.22, PLAY_CENTRE[0] + 0.22)
WORKSPACE_Y = (-0.30, 0.30)

# Area overlap that counts as solved. mjlab's number, and its reasoning: gym-pusht
# uses 0.95 and ManiSkill 0.90, but at 0.90 the bonus only fired when the block
# happened to spawn near the goal yaw, so it paid for luck instead of shaping.
SUCCESS_COVERAGE = 0.70

# --- where the arm starts, which is also where its actions are measured from ------
# NOT the workshop's REST_POSE. That is "arm up and out of the way" and it puts the
# gripper 302 mm above the env origin; the table is at 35. With mjlab's 0.8 rad action
# scale measured from there, the LOWEST the gripper reached over 600 steps x 64 envs of
# random actions was 115 mm -- it cannot touch the block, so no policy can ever learn
# to push it. mjlab hit the same wall on its own arm and fixed it the same way: it
# calls 0.8 "the action scale that puts the block's side within reach", which is a
# statement about a scale AND a home pose together.
#
# Solved with kinematics.fk_gripper and scipy least_squares inside the URDF's real
# joint limits: gripper 40 mm above the table, 16 cm to the SIDE of the block's spawn
# centre. Position error 0.00 mm and 1.01 rad of margin to the nearest joint limit, so
# 0.8 rad of action in any direction stays legal rather than being clipped.
#
# 16 cm, not 10: the block spawns anywhere in +/-7 cm of y, so a home 10 cm aside
# leaves only 3 cm to a gripper whose jaws are about that wide, and the episodes where
# the two overlap start with the arm already shoving the block. Measured at 10 cm, the
# mean block height over 8 envs rose 2.8 mm in the first 60 steps with ZERO actions
# commanded, which is contact, not settling. 16 cm leaves 9 cm of clearance.
#
# Solved in the BASE frame, whose origin is the table top now that the robot stands on
# it -- so the target z below is the height above the surface, with no bench offset to
# carry around.
#
# Three things this pose is answering, all of them measured:
#   BESIDE the block, and clear of its whole spawn box, not over it. The first solve
#   put the gripper 40 mm above the spawn point and the jaws rested on the block:
#   sim-play reported the T drifting 28.8 mm with nothing pushing it, which is the
#   scene nudging its own task object before the policy has done anything.
#   LOW. The whole point of replacing REST_POSE is to start where the work is.
#   AWAY FROM THE LIMITS. Selecting on limit margin AFTER solving position, rather
#   than folding margin into the residual: this arm has 5 joints for a 3D target, so
#   the null space is 2-dimensional and the solver will happily park in a corner of
#   it. Every pose that pinned a joint (elbow at 1.571 pulling back toward the base,
#   wrist_pitch at -1.658 reaching over the block) came out of that.
#
# Orientation is deliberately unconstrained. At this reach the SO-101 cannot point its
# gripper straight down -- Wrist_Pitch pins and the best alignment is -0.37 against a
# perfect -1.0 -- and a pushing task does not need it to.
PUSH_HOME = {
    "Rotation": -0.6992,
    "Pitch": 0.7312,
    "Elbow": 0.4016,
    "Wrist_Pitch": -0.0414,
    "Wrist_Roll": -0.0540,
    # Shut, just off the hard limit. The jaw-open penalty measures from -0.175 rad, so
    # this starts the episode with that term already at zero.
    "Jaw": -0.1500,
}

WRIST_POS, WRIST_ROT = wrist_camera_in_gripper()


def _quat_z(deg: float) -> tuple[float, float, float, float]:
    import math

    half = math.radians(deg) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


@configclass
class PushTRLSceneCfg(InteractiveSceneCfg):
    """One arm, one black table, the printed T, and the wrist camera on its mount."""

    # THE COLLIDER, unlike the teleop scene's table, which is decoration over the
    # workshop's mat USD. Static: collision props and no rigid body.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_SIZE[0], TABLE_SIZE[1], TABLE_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR, roughness=0.9),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(**PLA_ON_MAT),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(PLAY_CENTRE[0], 0.0, TABLE_TOP - TABLE_THICKNESS / 2)
        ),
    )

    # STANDING ON THE TABLE, at z = TABLE_TOP. The teleop scene puts the arm at z = 0
    # with the mat surface at 0.035 -- that is its measured bench geometry and it is
    # left alone -- but this env's table is a 25 mm slab spanning z 0.010 to 0.035, so
    # a robot at z = 0 has its base plate UNDER the slab and the links rise straight
    # through it. It renders exactly as it sounds.
    #
    # Raising the robot moves its base frame with it, which is why PUSH_HOME is solved
    # against a base-frame z that now means "height above the table" directly.
    robot = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # The home pose goes in init_state because use_default_offset=True measures
        # every action from it: changing it moves both where the episode starts and
        # what "zero action" means.
        init_state=SO101_CFG.init_state.replace(
            pos=(SO101_CFG.init_state.pos[0], SO101_CFG.init_state.pos[1], TABLE_TOP),
            joint_pos=dict(PUSH_HOME),
        ),
    )

    # The printed mount and the modelled webcam, exactly as the teleop scene carries
    # them: visual only, riding the wrist, so what the camera sees includes its own
    # bracket the way the real frame does.
    klip_support = KLIP_SUPPORT_CFG
    kwc500_body = KWC500_BODY_CFG
    kwc500_barrel = KWC500_BARREL_CFG

    # TABLE_TOP exactly, NOT the top plus half the thickness. The T's USD origin is on
    # its bottom face -- measured: spawned at TABLE_TOP + thickness/2 it reads z =
    # 0.045 at reset and settles at 0.035, a 10 mm drop burning the first frames of
    # every episode on a transient. The teleop scene carries the same warning about
    # its mat.
    t_block: RigidObjectCfg = T_BLOCK_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(SPAWN_CENTRE[0], SPAWN_CENTRE[1], TABLE_TOP)
        )
    )

    # Where it has to end up, drawn as an opaque green T. Kinematic and collisionless:
    # it is a picture, and reset_goal_pose moves it.
    goal_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.UsdFileCfg(
            usd_path=T_BLOCK_CFG.spawn.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            # Opaque, so this env never needs sim.render.enable_translucency -- which
            # costs about 7 ms a frame, and without which a translucent prim renders
            # solid black on a black table.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.72, 0.30), roughness=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(GOAL_CENTRE[0], GOAL_CENTRE[1], TABLE_TOP + 0.001)),
    )

    # Small. The real camera is 640x480 and this is the same optics through
    # from_intrinsic_matrix; 64x48 keeps the 4:3 aspect, because a square render
    # against 4:3 intrinsics throws away a quarter of the width.
    camera_grip = camera_cfg(
        "grip", "{ENV_REGEX_NS}/Robot/gripper/wrist_cam",
        pos=WRIST_POS, rot_quat=WRIST_ROT, width=64, height=48, near_m=0.02,
    )

    dome = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=900.0),
    )
    key = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(color=(1.0, 0.98, 0.95), intensity=6000.0, radius=0.1),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, -0.4, 0.9)),
    )


@configclass
class ActionsCfg:
    """mjlab's action term: a scaled delta on the rest pose, every joint.

    scale 0.8, which is mjlab's own correction and not a default -- under the stock
    scale the side of the block is simply not reachable and every run fails
    identically. The Jaw is included so the jaw-open penalty has something to act on.
    """

    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_NAMES, scale=0.8, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """State, and the camera deliberately left out of it.

    mjlab's D435 variant feeds a 42x24 RGB frame to the actor and keeps goal_pose as
    state because the footprint is 1 mm tall. That is the right end state here too,
    and it is not free: both obs groups carrying an image put their PPO buffer at
    2.4 GB, and Isaac's tiled rendering is a heavier bill than MuJoCo Warp's. So the
    camera is in the SCENE -- rendered, recordable, ready -- and the policy is state
    for now. Adding it is one ObsTerm and a much longer run.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        t_pose = ObsTerm(func=mdp.t_block_pose_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        goal_pose = ObsTerm(func=mdp.goal_pose_obs)
        t_to_goal = ObsTerm(func=mdp.t_block_to_goal_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=base_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )
    reset_t_block = EventTerm(
        func=mdp.reset_t_block_pose, mode="reset",
        params={"pose_range": {"x": SPAWN_X, "y": SPAWN_Y, "yaw": (-3.14, 3.14)}},
    )
    reset_goal = EventTerm(
        func=mdp.reset_goal_pose, mode="reset",
        # Absolute, unlike the block's, which is an offset from where it spawned.
        params={"pose_range": {"x": (GOAL_CENTRE[0] + GOAL_X[0], GOAL_CENTRE[0] + GOAL_X[1]),
                               "y": (GOAL_CENTRE[1] + GOAL_Y[0], GOAL_CENTRE[1] + GOAL_Y[1]),
                               "yaw": (-3.14, 3.14)}},
    )
    # mjlab randomizes the sliding friction of the block and of the fingertips at
    # startup, and on a pushing task that is the physical parameter that matters most:
    # it sets how far the block travels per push.
    t_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("t_block"),
                "static_friction_range": (0.5, 1.1), "dynamic_friction_range": (0.4, 0.9),
                "restitution_range": (0.0, 0.05), "num_buckets": 64, "make_consistent": True},
    )
    robot_paint = EventTerm(
        func=mdp.set_arm_appearance, mode="startup",
        params={"asset_name": "robot", "color": (0.93, 0.93, 0.95)},
    )


@configclass
class RewardsCfg:
    """mjlab's terms and weights. The argument for each is in mdp/push_t_rl.py."""

    position = RewTerm(func=mdp.push_position_reward, weight=1.0, params={"scale": 5.0})
    orientation = RewTerm(func=mdp.push_orientation_reward, weight=2.5, params={"gate_distance": 0.08})
    coverage = RewTerm(func=mdp.push_coverage, weight=0.6)
    # Every SceneEntityCfg is passed EXPLICITLY, never left to the default in the
    # function signature. The manager only resolves the ones it finds in params, so a
    # default one keeps body_ids as slice(None) and the first reward step dies with
    # "'slice' object is not subscriptable".
    ee_guidance = RewTerm(
        func=mdp.push_ee_guidance, weight=1.0,
        params={"scale": 5.0, "gate_distance": 0.05,
                "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"])},
    )
    success = RewTerm(func=mdp.push_coverage_success, weight=2.0, params={"threshold": SUCCESS_COVERAGE})
    precision = RewTerm(func=mdp.push_precision_bonus, weight=2.0,
                        params={"pos_scale": 20.0, "yaw_scale": 3.0})
    off_table = RewTerm(func=mdp.push_displacement_penalty, weight=-2.0)
    # The D435-Push variant's addition, and the reason to prefer it: without this the
    # policy holds the gripper open and forks the block, which scores better and
    # transfers worse.
    jaw_open = RewTerm(func=mdp.push_jaw_open_penalty, weight=-2.0,
                       params={"robot_cfg": SceneEntityCfg("robot", joint_names=["Jaw"])})
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-10.0,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    joint_vel_hinge = RewTerm(
        func=mdp.push_joint_velocity_hinge, weight=-0.01,
        params={"max_vel": 0.5, "robot_cfg": SceneEntityCfg("robot", joint_names=[".*"])})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    t_out_of_bounds = DoneTerm(
        func=mdp.push_t_out_of_bounds,
        params={"x_range": WORKSPACE_X, "y_range": WORKSPACE_Y},
    )
    # mjlab's ee_ground_collision, as a height test on the WRIST body only, 20 mm below
    # the surface. Both numbers are measured, not guessed:
    #
    #   The jaw is excluded because at the home pose its origin is only 17 mm above the
    #   table, so any test tight enough to mean something fires on a graze.
    #   20 mm below means the wrist is unambiguously inside a 25 mm slab. At 5 mm this
    #   term ended 130 episodes an iteration and mean episode length was 15 steps --
    #   half a second -- so the policy never saw the task at all.
    hand_through_table = DoneTerm(
        func=mdp.push_ee_below_surface,
        params={"height": TABLE_TOP - 0.020,
                "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"])},
    )


@configclass
class CurriculumCfg:
    """mjlab ramps the velocity penalty in three stages, and so does this.

    Applied at full strength from the start, the cheapest way to avoid it is to stop
    moving, and the block never gets pushed at all.
    """

    joint_vel_hinge = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={"term_name": "joint_vel_hinge", "weight": -0.1, "num_steps": 500 * 24},
    )
    joint_vel_hinge_late = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={"term_name": "joint_vel_hinge", "weight": -1.0, "num_steps": 1000 * 24},
    )


@configclass
class PushTRLEnvCfg(ManagerBasedRLEnvCfg):
    # 4096, which is what mjlab's hyperparameters assume: 4096 x 24 steps is the 98k
    # transitions an iteration those numbers were tuned against.
    scene: PushTRLSceneCfg = PushTRLSceneCfg(num_envs=2048, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 30 Hz control, as everything else in this package runs at, on the 1/120 s
        # step the T's contact needs. mjlab uses 5 ms and decimation 4 for the same
        # 50 Hz-ish reason; the finer step here is what the teleop scene was tuned on.
        self.decimation = 4
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0            # mjlab's
        self.viewer.eye = (0.9, 0.9, 0.7)
        self.viewer.lookat = (PLAY_CENTRE[0], 0.0, TABLE_TOP)
        # The T sliding on the table is the whole task, and Isaac Lab 2.1 cannot
        # attach a physics material to a USD-spawned rigid body, so it goes on as the
        # sim-wide default.
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(**PLA_ON_MAT)
        # ANTIALIASING OFF, and this is not a quality preference. Isaac Lab defaults
        # to DLSS, which upscales from a lower-resolution input and has a minimum it
        # will accept -- at a 64x48 tile the renderer reports "render resolution of
        # (222, 162) is below minimal input resolution of 300" and the process dies
        # right there, silently, with exit code 0 and no traceback. A small render is
        # the whole point of a camera in an RL env, so the antialiaser goes instead.
        self.sim.render.antialiasing_mode = "Off"
        # PhysX's GPU collision stack, raised from Isaac Lab's 64 MB default. At 4096
        # envs -- an arm standing on a table, a decomposed T block on it, per env --
        # PhysX overflows it and says so, once per substep: "collisionStackSize buffer
        # overflow detected, please increase its size to at least 376226584 in the
        # scene desc! CONTACTS HAVE BEEN DROPPED". Dropped contacts are the part that
        # matters: the sim keeps running and silently stops resolving some of the
        # pushing this task is about.
        #
        # 256 MB, not more. A bigger buffer is not the fix and 512 MB made it worse
        # -- at 4096 envs the process aborted in scene cloning with malloc(): invalid
        # size. The fix is the hand_through_table termination above, which stops the
        # deep-penetration contact storms that were driving the demand up to 896 MB
        # inside a single run.
        self.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024


@configclass
class PushTRLPlayEnvCfg(PushTRLEnvCfg):
    """A handful of envs and no observation noise, for watching a checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
