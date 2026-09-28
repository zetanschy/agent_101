"""The simulated follower, live: teleop it, record with it, run policies on it.

    real leader / policy  ->  lerobot (Docker)  ->  lerobot_robot_real2sim  --unix socket-->  live.server  ->  engine
                                                    (the `real2sim` robot)                      (native)      mujoco | isaac

The sim is a lerobot ROBOT, not a new workflow. `./robot teleop|record|infer ... --sim
mujoco|isaac` starts `live.server` natively in the engine's interpreter, then runs this
repo's usual scripts with ROBOT_TYPE=real2sim and ROBOT_PORT=<the socket>. The lerobot
plugin (sim/real2sim/lerobot_plugin) answers get_observation / send_action over the
socket, so lerobot-teleoperate, lerobot-record, lerobot-rollout and the async robot client
drive it like the real arm. Datasets come out of lerobot 0.6.1 itself, with the same
features as the real ones.

What the follower does with a command is the replay's servo, live: the identified model
(config/<ds>.servo.json), the firmware goal clamp, and one frame of dead time
(goals.OnlineGoals, the online twin of mujoco.servo.GoalStream). Observations are what
the real bus returns: joint readings in lerobot units through the scene's fitted offsets
and gripper map, plus both fitted cameras rendered with their distortion.

Objects start from a layout (layouts.py): a real episode's calibrated layout, or a random
one for data collection. They move only through contact, as in the replay.
"""
