"""Reward terms for reach: how far the gripper is from the commanded pose.

Isaac Lab keeps these same three functions in
isaaclab_tasks.manager_based.manipulation.reach.mdp, and they are copied here for
the reason assets/so101.py does not import the workshop's Python: that is a TASK
path, and task paths move between releases -- the same module was
omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp two releases ago.
isaaclab.utils.math does not move. Three short functions is a cheap price for one
less directory name to be pinned to.

The commanded pose is in the ROBOT BASE frame, which is why every one of these
starts by pushing it out to world: the arm is spawned with a 90 degree yaw, so
base x is world y and comparing the two directly is wrong in a way that still
produces a plausible-looking number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _desired_and_current(env, command_name: str, asset_cfg: SceneEntityCfg):
    """(desired position, current body position), both in WORLD."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3])
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return des_pos_w, curr_pos_w


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Distance from the body to the commanded position, in metres.

    Used with a NEGATIVE weight: it is the coarse pull that gets the gripper into
    the neighbourhood from anywhere in the workspace. On its own it stops improving
    once the error is small compared to the noise, which is what the tanh term below
    is for.
    """
    des_pos_w, curr_pos_w = _desired_and_current(env, command_name, asset_cfg)
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """1 - tanh(distance / std): a POSITIVE reward that only pays out up close.

    std is the length scale at which it turns on -- 0.1 m here, so the last
    centimetres are where nearly all of this term's gradient lives.
    """
    des_pos_w, curr_pos_w = _desired_and_current(env, command_name, asset_cfg)
    return 1 - torch.tanh(torch.norm(curr_pos_w - des_pos_w, dim=1) / std)


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Shortest-path angle between the body's orientation and the commanded one.

    Kept, and weighted ZERO by default -- see the note in tasks/reach_env_cfg.py.
    A 5 degree-of-freedom arm cannot hit an arbitrary orientation at an arbitrary
    position, so asking it to reproduces as a reward that can never be satisfied.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_w = quat_mul(asset.data.root_quat_w, command[:, 3:7])
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return quat_error_magnitude(curr_quat_w, des_quat_w)
