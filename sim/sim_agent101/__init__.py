"""Sim-to-real for this workspace's SO-ARM101: the push-T task, in Isaac Lab.

Layered on NVIDIA's Sim-to-Real-SO-101-Workshop (thirdparty/sim2real_so101), which
supplies the arm USD, the teleop plumbing, the lerobot recorder and the domain
randomisation terms. This package adds only what is specific to THIS robot:

  cameras.py        the real Logitech C270 and Klip Xtreme KWC-500, measured
  assets/objects.py the printed T and the printed wrist camera mount
  tasks/            the push-T environment

Importing this module registers the Gym environments, same as the workshop's.
"""

from . import assets, mdp, tasks  # noqa: F401
