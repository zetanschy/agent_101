"""Reach a commanded gripper pose: the RL half of this package.

Push-T is the goal, but it is an IMITATION task -- the scene exists to be teleoped
into a LeRobot dataset, and push_t_env_cfg.PushTEnvCfg has `rewards = None`. Nothing
in this repo had ever exercised the other half of Isaac Lab: a reward, thousands of
parallel envs, PPO, a checkpoint. Reach is the standard first task for that, it needs
no object and no cameras, and it trains in minutes -- so it is the cheapest possible
proof that the arm model, the actuator gains and the training stack all work before
a harder task depends on them.

Follows Isaac Lab's own manager_based/manipulation/reach and the SO-ARM101 recipe
from the Seeed wiki (MuammerBay/isaac_so_arm101, BSD-3), with four changes this
workspace forces:

  1. our arm. SO101_CFG is the workshop's no-camera USD with the workshop's tuned
     gains, and its joints are Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll/Jaw, not
     the lerobot names the upstream config uses.
  2. our bench. The target box is derived from the same numbers push-T places the
     T with, so the two tasks train in one volume -- see WORKSPACE below.
  3. 30 Hz control on a 1/120 s physics step, matching push_t_env_cfg and DATASET_FPS
     rather than the reference's 1/60. Same integrator step as push-T means the
     workshop's gains behave identically in both, which is the whole point of
     reusing them.
  4. no orientation reward. See RewardsCfg.

Train:  ./robot sim-train
Watch:  ./robot sim-policy --num-envs 16
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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
from ..assets.so101 import JOINT_NAMES, SO101_CFG
from .push_t_env_cfg import MAT_SURFACE

# The body the reward chases. Same name push_t_env_cfg's FrameTransformer already
# tracks, so "where the end effector is" means one thing across both tasks.
EE_BODY = "gripper"

# The five arm joints. The Jaw is deliberately NOT an action here: reach has nothing
# to grip, and leaving it out keeps the policy from spending exploration on a degree
# of freedom that cannot affect the reward. It stays at REST_POSE.
#
# NOTE this is a different action space from push-T, which drives all six. A reach
# checkpoint therefore does not drop into the push-T action pipeline unchanged; it
# is a training-stack proof, not a policy meant to be co-trained.
ARM_JOINTS = [j for j in JOINT_NAMES if j != "Jaw"]

# Actions are a DELTA on the rest pose, scaled: target = REST_POSE + 0.5 * action.
# push-T instead sends absolute targets (scale=1.0, use_default_offset=False),
# because a teleop recording IS absolute joint angles. For RL the offset form is
# what makes exploration tractable -- an untrained policy with init_noise_std=1.0
# emitting absolute radians flails through the whole joint range on step one --
# and it is what the reference configs use. Anything deploying a reach checkpoint
# on the real arm has to apply this same affine map.
ACTION_SCALE = 0.5

# --- where the arm is asked to reach ----------------------------------------
# In metres from the ROBOT ROOT (the arm's placement origin; its base plate spans
# -46.7 to +48.9 mm around it), and above the mat surface.
#
# The far edge is 0.25 m, which is SHORTER than push-T's play area: the T is sampled
# 0.19 to 0.31 m from the root. That is deliberate. The SO-101 reaches about 0.32 m,
# so past ~0.27 m the set of orientations it can achieve at a given point collapses
# and the tracking error cannot go to zero however good the policy is. The arm does
# get out to the T at 0.31 m, but it does it lying nearly flat on the mat, which is
# one specific configuration rather than a region worth sampling free-space targets
# in. Widen REACH_AHEAD if you want the boundary; expect the mean tracking error to
# stop falling when you do.
REACH_AHEAD = (0.10, 0.25)          # forward of the root
REACH_LATERAL = 0.10                # +/- across the bench
REACH_ABOVE_MAT = (0.05, 0.20)      # height over MAT_SURFACE

# UniformPoseCommand's ranges are in the ROBOT BASE frame, and this arm is spawned
# with a 90 degree yaw (assets/so101.py), so base and world do not share axes:
# world (x, y) = (-base_y, base_x). Forward on the bench is therefore base -y, which
# is why the y range below is negative -- the same shape the upstream SO-ARM config
# arrived at empirically. Get this wrong and the targets appear beside the arm
# instead of in front of it, which trains perfectly well and looks like nothing is
# broken until you compare it against the real bench.
CMD_POS_X = (-REACH_LATERAL, REACH_LATERAL)
CMD_POS_Y = (-REACH_AHEAD[1], -REACH_AHEAD[0])
CMD_POS_Z = (MAT_SURFACE + REACH_ABOVE_MAT[0], MAT_SURFACE + REACH_ABOVE_MAT[1])


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """The arm, a floor and a light. Nothing else, on purpose.

    push-T's mat, table and light box are visual only (no collision props), so they
    change nothing physical here -- and this scene gets cloned a few thousand times,
    where "visual only" still means a USD prim and a draw call per environment. The
    bench survives where it matters: the target box above is derived from its
    geometry, so the policy is trained over the volume the real mat occupies.

    What this costs: the arm can sweep below the bench plane, which the real one
    cannot. Targets are all above the mat, so it has no reason to -- but a reach
    checkpoint should be watched in ./robot sim-policy before it is trusted near
    real hardware.
    """

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        # Well below the bench: it is a horizon for the viewport, not a surface.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    )

    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=900.0),
    )


@configclass
class CommandsCfg:
    ee_pose = base_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_BODY,
        # A new target every 4 s, so a 12 s episode is three of them and the policy
        # has to learn to travel between targets rather than to sit at one.
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=base_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=CMD_POS_X, pos_y=CMD_POS_Y, pos_z=CMD_POS_Z,
            # Fixed, because the orientation reward is off -- see RewardsCfg. A
            # commanded orientation that nothing is scored against would only show
            # up as a misleading marker in the viewport.
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS,
        scale=ACTION_SCALE, use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            # CONCATENATED, unlike push-T, which keeps its terms as a dict because
            # the lerobot recorder wants named streams. rsl_rl wants one flat vector.
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        # The reference range, kept: scaling the rest pose by 0.5 to 1.5 starts each
        # episode somewhere different, which is most of the exploration this task gets.
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Distance, mostly.

    Two position terms doing different jobs: the L2 distance is the coarse pull from
    anywhere in the workspace, the tanh kernel is what makes the last few centimetres
    worth anything. Both are needed -- distance alone plateaus early, tanh alone has
    no gradient until the gripper happens to be close.

    Orientation is weighted ZERO. The arm has five joints, so position and orientation
    cannot both be commanded freely; scoring an orientation it cannot reach adds a
    reward floor the policy can never climb off, and the upstream SO-ARM config zeroes
    it for the same reason. The term is left in place, not deleted, because a task
    that DOES pin orientation (approach the T from above, say) only needs a weight.
    """

    ee_position = RewTerm(
        func=mdp.position_command_error, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "command_name": "ee_pose"},
    )
    ee_position_fine = RewTerm(
        func=mdp.position_command_error_tanh, weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]),
                "std": 0.1, "command_name": "ee_pose"},
    )
    ee_orientation = RewTerm(
        func=mdp.orientation_command_error, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "command_name": "ee_pose"},
    )

    # Smoothness. Tiny at first so they do not drown the task out before it is
    # learned, then raised by the curriculum -- and they matter more here than the
    # numbers suggest, because a jerky policy is exactly what a real STS3215 will
    # not reproduce.
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-0.0001,
                        params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """Time-out only: there is nothing to fail at and nothing to knock over."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Turn the smoothness penalties up once the task itself is learned.

    4500 steps is the reference's number and lands well after the tracking error has
    come down. Applied from the start instead, the cheapest way to avoid an action-rate
    penalty is to not move.
    """

    action_rate = CurrTerm(func=base_mdp.modify_reward_weight,
                           params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500})
    joint_vel = CurrTerm(func=base_mdp.modify_reward_weight,
                         params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500})


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """The training environment."""

    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 1/120 s physics, 4 substeps per control step = 30 Hz, the rate .env records
        # and infers at and the rate push-T runs at. The reference uses 1/60 with
        # decimation 2, which is the same 30 Hz for half the physics cost; this keeps
        # the finer step so the workshop's actuator gains behave identically in both
        # of this package's tasks.
        self.decimation = 4
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        # Over the operator's shoulder, like push-T's, but closer -- the arm is 30 cm.
        self.viewer.eye = (0.8, 0.8, 0.6)
        self.viewer.lookat = (0.15, 0.0, 0.1)


@configclass
class ReachPlayEnvCfg(ReachEnvCfg):
    """A handful of envs, no observation noise: for watching a checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.0
        self.observations.policy.enable_corruption = False


@configclass
class ReachDREventCfg(EventCfg):
    """Reset events plus actuator-gain randomization.

    The one domain randomization that can matter to a STATE-observation policy: with
    no pixels in the observation, lighting and camera jitter are irrelevant here and
    the only sim2real gap left is how the arm responds to a position command. The
    workshop's gains were tuned on one real SO-101, and ours is a different unit.

    Same 0.8-1.2 / 0.7-1.3 band push-T's DR uses, so the two tasks disagree about
    nothing.
    """

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


@configclass
class ReachDREnvCfg(ReachEnvCfg):
    """Reach with randomized actuator gains, for a policy meant to leave the sim."""

    events: ReachDREventCfg = ReachDREventCfg()
