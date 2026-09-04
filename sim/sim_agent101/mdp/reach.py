"""Terms for the reach task: distance to a commanded pose, and the ghost's paint.

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


def set_arm_appearance(
    env, env_ids, asset_name: str = "robot",
    color: tuple[float, float, float] = (0.93, 0.93, 0.95),
    opacity: float | None = None,
) -> None:
    """Paint one arm's printed parts, and optionally make them see-through.

    The same hook set_robot_color uses in push_t.py -- every printed part of the
    workshop USD binds Looks/material_a_3d_printed -- with opacity added, because the
    real-arm view needs two arms in one place: the solid one is what the policy
    commanded, the translucent one is where the real robot actually is.

    Opacity on OmniPBR needs BOTH inputs, and the enable flag is the one that is easy
    to miss: set opacity_constant alone and the arm stays solid with no error. The
    render setting matters too -- with sim.render.enable_translucency off, a
    translucent prim renders solid BLACK, which is how push-T's goal marker once
    disappeared into the mat.
    """
    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf, UsdShade

    asset = env.scene[asset_name]
    shader = f"{asset.cfg.prim_path}/Looks/material_a_3d_printed/Shader"
    # UsdShade.CreateInput rather than prim.CreateAttribute, and NOT inside an
    # Sdf.ChangeBlock. A change block defers notification, so authoring a brand new
    # attribute in one produces a spec with no typeName and the first Set() throws
    # "Empty typeName for ...inputs:enable_opacity". set_robot_color gets away with a
    # change block because diffuse_color_constant already exists on the shader; the
    # opacity inputs do not.
    for prim in sim_utils.find_matching_prims(shader):
        sh = UsdShade.Shader(prim)
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        if opacity is not None:
            sh.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))

