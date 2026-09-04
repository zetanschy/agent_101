"""PPO, as rsl_rl wants it, for the reach task.

These are the reference numbers -- Isaac Lab's own reach config and the SO-ARM101
recipe use the same ones, and they are not tuned for this arm. That is deliberate
for a first task: if reach does not learn with the settings that work for everyone
else's reach, the problem is the arm model or the scene, not the hyperparameters,
and looking for it in the learning rate wastes a day.

Sizing: 4096 envs x 24 steps = 98k transitions per iteration, 1000 iterations is
about 100M steps. Reach is solved long before that -- watch the mean reward
plateau in tensorboard and stop, or pass --max-iterations. There are no cameras in
this scene, so the whole thing fits comfortably in a few GB.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    # Checkpoints land in sim/outputs/rsl_rl/<experiment_name>/<timestamp>/.
    experiment_name = "so101_reach"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        # 64x64 is small, and reach is a small problem: 20 observations in, 5 joints
        # out. A bigger net learns the same thing slower.
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        # "adaptive" retunes the learning rate to hold desired_kl, so this is a
        # starting point rather than a setting.
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class StdFloorActorCriticCfg(RslRlPpoActorCriticCfg):
    """RslRlPpoActorCriticCfg plus a floor under the exploration noise.

    class_name is what rsl_rl eval()s to find the policy class; agents/std_floor.py
    registers ActorCriticStdFloor into the namespace it looks in, and std_min/std_max
    are passed straight through as constructor kwargs.
    """

    class_name: str = "ActorCriticStdFloor"
    std_min: float = 0.2
    std_max: float = 1.0


@configclass
class PushTPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """mjlab's Mjlab-Push-T-Yam-D435-Push hyperparameters, as rsl_rl 2.3.3 takes them.

    Not retuned. These are the numbers behind the runs that worked over there, walked
    down the config chain (d435_push -> d435 -> precise_random_goal -> precise ->
    reachable -> lift_cube_vision), and the point of copying them exactly is that if
    this task fails to learn with the settings that work on the same objective, the
    problem is the arm or the scene, not the optimizer.

    Two of them are load-bearing and would be easy to lose:

      STD FLOOR at 0.2. mjlab measured sigma collapsing to 0.026 with only 3.1% of
      episodes ever reaching contact geometry -- there was no exploration left to find
      the descent with. rsl_rl has no such bound, so agents/std_floor.py adds it. Its
      note is explicit that entropy_coef alone does NOT do this job.

      OBS NORMALIZATION on. mjlab's runner sets obs_normalization=True on both actor
      and critic; the rsl_rl equivalent is empirical_normalization, which Isaac Lab's
      own configs leave off.

    What could NOT be carried across: mjlab's actor and critic are CNN policies over a
    42x24 wrist frame, with obs_groups mapping actor/critic to (state, camera). Our
    observation is state only for now, so the CNN half has nothing to attach to and
    the MLP dims are what remain. That is a real difference in the policy, not just in
    plumbing -- see the README.
    """

    # 4096 envs x 24 steps = 98k transitions an iteration, 5000 iterations = 491M steps.
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "so101_push_t"
    run_name = ""
    resume = False
    # mjlab: obs_normalization=True on both actor and critic.
    empirical_normalization = True
    policy = StdFloorActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        std_min=0.2,
        std_max=1.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
