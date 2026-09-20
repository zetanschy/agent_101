"""Push-T rewards, ported from mjlab's Mjlab-Push-T-Yam-D435-Push.

mjlab is not an Isaac Lab backend -- it reimplements Isaac Lab's manager API on
MuJoCo Warp -- so nothing here could be imported. What ports is the SHAPE of the
objective, which is the part that took that project twelve runs to get right, and
its own comments are the argument for each piece. Kept, with the reasoning, because
a reward you cannot explain is a reward you cannot debug:

  position + orientation are ADDED, not multiplied. Multiplied, a wrong yaw zeroes
  the translation gradient and the policy stalls with the block still across the
  table.

  orientation is GATED on position. Yaw can be changed without moving the block --
  spun in place, an ungated rotation term pays in full for a skill that needs no
  pushing, and mjlab measured a policy that did exactly that: 0.001 m of travel in
  a whole episode. The gate means the only route to the rotation bonus runs through
  translation.

  yaw enters through cos, not a Gaussian. At the +/-pi/2 mean error of a uniformly
  randomized goal a Gaussian kernel is ~6e-5, which is nothing to descend.

  ee_guidance is 3D and capped. In the plane it is maximised by hovering above the
  block at any height, and a policy trained on the planar version parked 23 cm up
  and never touched it. It is also gated OFF near the goal: its target is the
  standoff point on the far side of the block, which is where a push starts and the
  wrong place to finish, because force through the centroid produces no torque.

  coverage is the quantity gym-pusht actually scores, and it refines what the
  others get close to. Success fires at 0.70 rather than ManiSkill's 0.90: at 0.90
  the bonus only ever fired when the block happened to spawn near the goal yaw, so
  it rewarded luck.

  precision is a SECOND, finer scale added alongside -- ~0 far away, 1.0 at the
  goal. Sharpening the coarse kernels instead would fix the near field and flatten
  the far one, which is what made rotation unlearnable there.

What is ours rather than mjlab's: the T is this workspace's printed 10 cm one, so
the coverage point set is built from tblock.py instead of their scene constants,
and the gripper-open penalty becomes a JAW penalty over the SO-101's single jaw
joint rather than a two-finger travel.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .push_t import _goal, t_pose_world

_T_POINTS: dict[str, torch.Tensor] = {}


def _errors(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """(planar distance to the goal, absolute wrapped yaw error).

    Planar on purpose: a 3D error would pay the policy for lifting the block, which
    is the opposite of a pushing task.
    """
    t = t_pose_world(env, asset_cfg)
    goal = _goal(env)
    dist = torch.linalg.norm(goal[:, :2] - t[:, :2], dim=-1)
    yaw_err = torch.atan2(torch.sin(goal[:, 2] - t[:, 2]), torch.cos(goal[:, 2] - t[:, 2])).abs()
    return dist, yaw_err


def _t_points(device: torch.device, spacing: float = 0.0025) -> torch.Tensor:
    """Points tiling the T's area, in the CENTROID frame, cached per device.

    Sampling rather than exact polygon clipping: the shape never changes and this
    runs over thousands of envs every step, so coverage becomes one rigid transform
    plus an inside test. 2.5 mm spacing puts ~700 points on a 10 cm T, which
    resolves the 0.70 success threshold far finer than the reward needs.

    The centroid frame, not the mesh frame: t_pose_world already reports the
    centroid, because the mesh origin sits at the middle of the crossbar and
    scoring there would reward parking the bar on the goal with the stem anywhere.
    """
    key = f"{device}:{spacing}"
    if key in _T_POINTS:
        return _T_POINTS[key]

    from ..assets.objects import T_BLOCK_GEOMETRY as g

    cy = g.centroid_offset[1]
    boxes = _t_boxes()
    xs = torch.arange(-g.bar_width / 2, g.bar_width / 2, spacing, device=device) + spacing / 2
    lo = -(g.bar_depth / 2 + g.stem_length) - cy
    hi = g.bar_depth / 2 - cy
    ys = torch.arange(lo, hi, spacing, device=device) + spacing / 2
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    _T_POINTS[key] = pts[_inside_t(pts, boxes)]
    return _T_POINTS[key]


def _t_boxes() -> tuple[tuple[float, float, float, float], ...]:
    """The T as two axis-aligned boxes (x_half, y_centre, y_half), centroid frame."""
    from ..assets.objects import T_BLOCK_GEOMETRY as g

    cy = g.centroid_offset[1]
    return (
        (g.bar_width / 2, 0.0 - cy, g.bar_depth / 2),
        (g.stem_width / 2, -(g.bar_depth / 2 + g.stem_length / 2) - cy, g.stem_length / 2),
    )


def _inside_t(pts: torch.Tensor, boxes=None) -> torch.Tensor:
    boxes = boxes if boxes is not None else _t_boxes()
    out = torch.zeros(pts.shape[:-1], dtype=torch.bool, device=pts.device)
    for x_half, y_c, y_half in boxes:
        out |= (pts[..., 0].abs() <= x_half) & ((pts[..., 1] - y_c).abs() <= y_half)
    return out


def coverage(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")) -> torch.Tensor:
    """Fraction of the block's area lying inside the goal footprint, in [0, 1].

    Block and goal are the same shape, so this is the intersection over goal area
    that gym-pusht reports.
    """
    t = t_pose_world(env, asset_cfg)
    goal = _goal(env)
    pts = _t_points(t.device)                                    # (P, 2)

    rel = t[:, 2] - goal[:, 2]
    cos_r, sin_r = torch.cos(rel), torch.sin(rel)
    px = pts[None, :, 0] * cos_r[:, None] - pts[None, :, 1] * sin_r[:, None]
    py = pts[None, :, 0] * sin_r[:, None] + pts[None, :, 1] * cos_r[:, None]

    d = t[:, :2] - goal[:, :2]
    cos_g, sin_g = torch.cos(goal[:, 2]), torch.sin(goal[:, 2])
    ox = d[:, 0] * cos_g + d[:, 1] * sin_g
    oy = -d[:, 0] * sin_g + d[:, 1] * cos_g

    local = torch.stack([px + ox[:, None], py + oy[:, None]], dim=-1)   # (B, P, 2)
    return _inside_t(local).float().mean(dim=-1)


def position_reward(
    env, scale: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")
) -> torch.Tensor:
    """ManiSkill's position term, (1 - tanh(scale*d))**2 / 2, max 0.5.

    tanh rather than a Gaussian so the gradient survives at the spawn distance
    instead of underflowing.
    """
    dist, _ = _errors(env, asset_cfg)
    return torch.square(1.0 - torch.tanh(scale * dist)) / 2.0


def orientation_reward(
    env, gate_distance: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")
) -> torch.Tensor:
    """((cos(dyaw)+1)/2)**2 / 2, gated on how far the block still has to travel."""
    dist, yaw_err = _errors(env, asset_cfg)
    aligned = torch.square((torch.cos(yaw_err) + 1.0) / 2.0) / 2.0
    return aligned * (1.0 - dist / gate_distance).clamp(0.0, 1.0)


def precision_bonus(
    env, pos_scale: float = 20.0, yaw_scale: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
) -> torch.Tensor:
    """The fine scale: ~0 beyond a few centimetres, 1.0 on the goal."""
    dist, yaw_err = _errors(env, asset_cfg)
    fine_pos = torch.square(1.0 - torch.tanh(pos_scale * dist)) / 2.0
    fine_yaw = torch.square(1.0 - torch.tanh(yaw_scale * yaw_err)) / 2.0
    return 4.0 * fine_pos * fine_yaw


def coverage_success(
    env, threshold: float = 0.70, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")
) -> torch.Tensor:
    """Sparse, paid every step it holds, so arriving early beats arriving late."""
    return (coverage(env, asset_cfg) >= threshold).float()


def ee_guidance(
    env,
    scale: float = 5.0,
    standoff: float = 0.06,
    gate_distance: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
) -> torch.Tensor:
    """Gripper proximity to a push pose: 3D, capped at 0.05, gated off near the goal."""
    from ..assets.objects import T_BLOCK_GEOMETRY

    robot: Articulation = env.scene[robot_cfg.name]
    ee = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    block: RigidObject = env.scene[asset_cfg.name]
    block_pos = block.data.root_pos_w

    goal = _goal(env)
    goal_w = goal[:, :2] + env.scene.env_origins[:, :2]
    to_goal = goal_w - block_pos[:, :2]
    goal_dist = to_goal.norm(dim=-1)
    direction = to_goal / goal_dist.clamp_min(1e-6).unsqueeze(-1)

    push_pose = block_pos.clone()
    push_pose[:, :2] = block_pos[:, :2] - direction * standoff
    push_pose[:, 2] = block_pos[:, 2] + T_BLOCK_GEOMETRY.thickness / 2.0

    dist = (ee - push_pose).norm(dim=-1)
    proximity = (1.0 - torch.tanh(scale * dist)).clamp_min(0.0).sqrt() / 20.0
    return proximity * (goal_dist / gate_distance).clamp(0.0, 1.0)


def displacement_penalty(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block")
) -> torch.Tensor:
    """Squared height error off its resting plane, plus how far it has tipped.

    Keeps it a pushing task: no lifting the block, no standing it on its side.
    """
    import isaaclab.utils.math as math_utils

    block: RigidObject = env.scene[asset_cfg.name]
    rest_z = block.data.default_root_state[:, 2]
    height_err = torch.square(block.data.root_pos_w[:, 2] - rest_z)
    up = math_utils.quat_apply(
        block.data.root_quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3),
    )
    return height_err + (1.0 - up[:, 2])


def jaw_open_penalty(
    env,
    closed_tol: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Jaw"]),
) -> torch.Tensor:
    """Penalize holding the jaw open, normalised over its travel.

    mjlab's version of this, on a two-finger gripper, exists because the
    unconstrained policy found a better solution than pushing: hold the fingers
    apart and use them as a fork. It scored better and transferred worse. The SO-101
    has one jaw joint rather than two fingers, so the same idea is one term over
    JAW_RANGE_DEG, in radians, measured from shut.
    """
    from ..kinematics import JAW_RANGE_DEG

    robot: Articulation = env.scene[robot_cfg.name]
    lo, hi = (torch.pi / 180.0 * v for v in JAW_RANGE_DEG)
    opening = robot.data.joint_pos[:, robot_cfg.joint_ids] - lo
    return (opening - closed_tol).clamp_min(0.0).sum(dim=-1) / (hi - lo)


def joint_velocity_hinge(
    env, max_vel: float = 0.5, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Quadratic penalty on only the amount by which |joint velocity| exceeds a cap."""
    robot: Articulation = env.scene[robot_cfg.name]
    excess = (robot.data.joint_vel[:, robot_cfg.joint_ids].abs() - max_vel).clamp_min(0.0)
    return (excess**2).sum(dim=-1)


def t_out_of_bounds(
    env,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("t_block"),
) -> torch.Tensor:
    """End the episode once the block has been shoved off the working area."""
    t = t_pose_world(env, asset_cfg)
    return ((t[:, 0] < x_range[0]) | (t[:, 0] > x_range[1])
            | (t[:, 1] < y_range[0]) | (t[:, 1] > y_range[1]))


def ee_below_surface(
    env,
    height: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper", "jaw"]),
) -> torch.Tensor:
    """End the episode once the hand has gone INTO the table.

    mjlab ends its push-T episodes on an end-effector/ground contact above 10 N, and
    the reason to have some version of that is not tidiness. Without it the arm spends
    training driving through the surface, and at 4096 envs that is not merely ugly:
    PhysX's GPU collision stack overflows on the deep-penetration contact patches --
    "collisionStackSize buffer overflow detected... CONTACTS HAVE BEEN DROPPED" -- and
    the demand grows as the policy flails, from 577 MB to 896 MB within one run at
    HALF that env count. Dropped contacts mean the simulator quietly stops resolving
    some of the pushing the task is made of.

    A height test rather than mjlab's contact sensor, deliberately. The contact version
    needs activate_contact_sensors on every arm plus a filtered sensor per env, which
    is its own per-env memory at this scale; this needs nothing, and for a flat table
    at a known height it answers the same question. It costs the distinction between
    resting a finger on the table and slamming it, which the reward's action-rate and
    velocity terms already discourage.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    z = robot.data.body_pos_w[:, robot_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    return (z < height).any(dim=-1)
