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
class PushTPPORunnerCfg(ReachPPORunnerCfg):
    """Push-T is a much harder problem than reach, and sized accordingly.

    Reach converges in 250 iterations; mjlab's push-T runs are quoted in hundreds of
    millions of steps. Bigger rollouts (48 steps against 24) because the episode is
    20 s rather than 12 and the reward only pays once contact happens, and a wider
    net because the observation now carries the block AND the goal.
    """

    num_steps_per_env = 48
    max_iterations = 5000
    experiment_name = "so101_push_t"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
