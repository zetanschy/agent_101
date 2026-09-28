"""`--robot.type=real2sim`: the simulated SO-101 follower, as a lerobot robot config.

The fields mirror SOFollowerConfig so this repo's scripts (scripts/robot/common.sh:
--robot.type/--robot.port/--robot.id, --robot.cameras) run unchanged with
ROBOT_TYPE=real2sim. `port` is the live sim's socket, not a serial device. The camera
configs only NAME the cameras to return and their size (the real ones', from .env); the
images come from the sim's fitted cameras, and a size mismatch is refused at connect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

DEFAULT_SOCKET = "/workspace/sim/outputs/real2sim/live/sim.sock"  # the repo's path inside the lerobot image


@RobotConfig.register_subclass("real2sim")
@dataclass
class Real2SimConfig(RobotConfig):
    port: str = DEFAULT_SOCKET
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # accepted for SOFollowerConfig compatibility; same meaning where one exists
    max_relative_target: float | dict[str, float] | None = None
    disable_torque_on_disconnect: bool = True
    use_degrees: bool = True
    timeout_s: float = 30.0
