"""RL agent configurations, kept apart from the environments they train on.

An env cfg says what the task IS; these say how it is learned. Splitting them is
Isaac Lab's own convention and the reason gym.register can hand a training script
the right hyperparameters for a task it has never seen.
"""

from .rsl_rl_ppo_cfg import ReachPPORunnerCfg  # noqa: F401
