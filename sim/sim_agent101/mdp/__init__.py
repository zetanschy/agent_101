"""MDP terms for this workspace's tasks, on top of the workshop's and Isaac Lab's own."""

from .push_t_rl import (  # noqa: F401
    coverage as push_coverage,
    ee_below_surface as push_ee_below_surface,
    coverage_success as push_coverage_success,
    displacement_penalty as push_displacement_penalty,
    ee_guidance as push_ee_guidance,
    jaw_open_penalty as push_jaw_open_penalty,
    joint_velocity_hinge as push_joint_velocity_hinge,
    orientation_reward as push_orientation_reward,
    position_reward as push_position_reward,
    precision_bonus as push_precision_bonus,
    t_out_of_bounds as push_t_out_of_bounds,
)

from .reach import (  # noqa: F401
    orientation_command_error,
    set_arm_appearance,
    position_command_error,
    position_command_error_tanh,
)

from .randomize import (  # noqa: F401
    randomize_camera_pose,
    randomize_lighting,
    randomize_robot_color,
)

from .push_t import (  # noqa: F401
    goal_pose_obs,
    reset_goal_pose,
    reset_t_block_pose,
    set_mount_transform,
    set_robot_color,
    t_block_at_goal,
    t_block_pose_obs,
    t_block_settled,
    t_block_to_goal_obs,
    t_pose_world,
)
