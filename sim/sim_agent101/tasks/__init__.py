"""Gym registration for this workspace's tasks.

Two families. Push-T is teleoped and recorded, so its cfgs carry no rewards and
nothing here points at an agent. Reach is trained, so its ids also carry an
rsl_rl_cfg_entry_point -- that is how scripts/sim/train.py gets the hyperparameters
for a task it is only given the name of.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Agent101-So101-Push-T",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.push_t_env_cfg:PushTEnvCfg"},
)

gym.register(
    id="Agent101-So101-Push-T-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.push_t_env_cfg:PushTEvalEnvCfg"},
)

gym.register(
    id="Agent101-So101-Push-T-DR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.push_t_env_cfg:PushTDREnvCfg"},
)

gym.register(
    id="Agent101-So101-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
    },
)

gym.register(
    id="Agent101-So101-Reach-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:ReachPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
    },
)

gym.register(
    id="Agent101-So101-Reach-DR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:ReachDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
    },
)
