"""Gym registration for this workspace's tasks."""

import gymnasium as gym

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
