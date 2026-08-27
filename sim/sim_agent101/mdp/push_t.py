"""Push-T terms: where the T is, where it should go, and when that counts as done.

Planar task, so everything is (x, y, yaw). Two things this module is careful about:

  * The T's mesh origin is the middle of the crossbar, not the centroid. Success
    measured at the origin rewards parking the crossbar on the goal while the stem
    points anywhere. Everything here works on the centroid.

  * A T has 180-degree rotational symmetry about neither axis -- it is symmetric
    only under identity -- so yaw error is wrapped to (-pi, pi] and NOT folded.
    Fold it and the policy learns it may finish upside down, which on the real mat
    is a different outcome.
"""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

# Where the goal lives on the env. A manager-based env has no first-class place for
# per-episode task parameters, so it is stashed on the env object and created lazily
# by the reset term; observation and termination terms read it back.
GOAL_ATTR = "_push_t_goal"


def _goal(env) -> torch.Tensor:
    """(num_envs, 3) of x, y, yaw. Zeros until the reset term has run."""
    goal = getattr(env, GOAL_ATTR, None)
    if goal is None:
        goal = torch.zeros((env.num_envs, 3), device=env.device)
        setattr(env, GOAL_ATTR, goal)
    return goal


def t_pose_world(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")) -> torch.Tensor:
    """(num_envs, 3) centroid x, y and yaw of the T, relative to its env origin."""
    asset: RigidObject = env.scene[asset_cfg.name]
    from ..assets.objects import T_BLOCK_GEOMETRY

    quat = asset.data.root_quat_w
    offset = torch.tensor(T_BLOCK_GEOMETRY.centroid_offset, device=env.device).repeat(env.num_envs, 1)
    centroid = asset.data.root_pos_w + math_utils.quat_apply(quat, offset)
    yaw = math_utils.euler_xyz_from_quat(quat)[2]
    local = centroid - env.scene.env_origins
    return torch.stack([local[:, 0], local[:, 1], math_utils.wrap_to_pi(yaw)], dim=-1)


def t_block_pose_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")) -> torch.Tensor:
    """Observation: the T's planar pose, with yaw split into sin/cos so it is continuous."""
    pose = t_pose_world(env, asset_cfg)
    return torch.stack([pose[:, 0], pose[:, 1], torch.sin(pose[:, 2]), torch.cos(pose[:, 2])], dim=-1)


def goal_pose_obs(env) -> torch.Tensor:
    """Observation: the goal pose, same encoding as t_block_pose_obs."""
    g = _goal(env)
    return torch.stack([g[:, 0], g[:, 1], torch.sin(g[:, 2]), torch.cos(g[:, 2])], dim=-1)


def t_block_to_goal_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")) -> torch.Tensor:
    """Observation: (dx, dy, dyaw) from the T to the goal. What the policy has to null."""
    pose = t_pose_world(env, asset_cfg)
    g = _goal(env)
    return torch.stack(
        [g[:, 0] - pose[:, 0], g[:, 1] - pose[:, 1], math_utils.wrap_to_pi(g[:, 2] - pose[:, 2])], dim=-1
    )


def reset_t_block_pose(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
) -> None:
    """Drop the T somewhere in the start region, flat on the mat.

    Only x, y and yaw are sampled: the T is a flat plate and the task is planar, so
    spawning it tilted just gives the solver a settling transient to burn through
    while the episode clock is already running.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    default = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)

    def sample(key: str) -> torch.Tensor:
        lo, hi = pose_range.get(key, (0.0, 0.0))
        return torch.empty(n, device=env.device).uniform_(lo, hi)

    pos = default[:, 0:3] + env.scene.env_origins[env_ids]
    pos[:, 0] += sample("x")
    pos[:, 1] += sample("y")
    yaw = sample("yaw")
    zeros = torch.zeros(n, device=env.device)
    quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)


def reset_goal_pose(env, env_ids: torch.Tensor, pose_range: dict[str, tuple[float, float]]) -> None:
    """Pick where the T has to end up. Stored on the env, not in the scene."""
    goal = _goal(env)
    n = len(env_ids)
    for i, key in enumerate(("x", "y", "yaw")):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        goal[env_ids, i] = torch.empty(n, device=env.device).uniform_(lo, hi)
    if "goal_marker" in env.scene.keys():
        marker = env.scene["goal_marker"]
        pos = torch.zeros((n, 3), device=env.device)
        pos[:, 0:2] = goal[env_ids, 0:2]
        pos += env.scene.env_origins[env_ids]
        zeros = torch.zeros(n, device=env.device)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, goal[env_ids, 2])
        marker.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)


def t_block_settled(
    env, lin_vel_thresh: float = 0.01, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")
) -> torch.Tensor:
    """True where the T has stopped moving. Success should not fire mid-slide."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, :2].norm(dim=-1) < lin_vel_thresh


def t_block_at_goal(
    env,
    pos_tol: float = 0.02,
    yaw_tol_deg: float = 15.0,
    require_settled: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
) -> torch.Tensor:
    """Termination: the T is within tolerance of the goal and has come to rest.

    Defaults are the real task's tolerances -- 2 cm and 15 degrees on an 80 mm part
    is roughly "a person would call that done".
    """
    err = t_block_to_goal_obs(env, asset_cfg)
    close = err[:, :2].norm(dim=-1) < pos_tol
    aligned = err[:, 2].abs() < torch.deg2rad(torch.tensor(yaw_tol_deg, device=env.device))
    done = close & aligned
    if require_settled:
        done = done & t_block_settled(env, asset_cfg=asset_cfg)
    return done


def set_robot_color(env, env_ids, color: tuple[float, float, float] = (0.93, 0.93, 0.95)) -> None:
    """Paint the arm's printed parts. The real SO-ARM101 here is white; the USD ships
    yellow, and a policy trained on yellow links has to unlearn them on the real robot.

    The arm's printed geometry all binds one material, Looks/material_a_3d_printed --
    same hook the workshop's randomize_robot_color uses, but set rather than sampled.
    """
    import isaaclab.sim as sim_utils
    from pxr import Sdf

    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        shader = f"{robot.cfg.prim_path}/Looks/material_a_3d_printed/Shader"
        for prim in sim_utils.find_matching_prims(shader):
            prim.GetAttribute("inputs:diffuse_color_constant").Set(tuple(color))
