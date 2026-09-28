"""lerobot plugin: the `real2sim` robot, the live sim behind lerobot's Robot interface.

lerobot's CLIs call register_third_party_plugins(), which imports every installed
distribution named lerobot_robot_*. importlib.metadata finds a distribution from any
*.dist-info directory on sys.path, so the dist-info shipped beside this package is the
whole installation: put sim/real2sim/lerobot_plugin (and sim/, for real2sim.live) on
PYTHONPATH and `--robot.type=real2sim` exists. `./robot ... --sim ENGINE` does that.
"""

from .config_real2sim import Real2SimConfig  # noqa: F401  (registers --robot.type=real2sim)
from .real2sim import Real2Sim  # noqa: F401
