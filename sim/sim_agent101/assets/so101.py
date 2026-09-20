"""The SO-ARM101 articulation, pointed at the workshop's USD.

Deliberately NOT `from sim_to_real_so101.assets.so101 import SO101_CFG`. The
workshop package is pinned to Isaac Lab 2.3 APIs, and this repo has to run on
whatever Isaac Lab actually boots on the box in front of it. Depending on the
submodule for the ASSET (a USD file, version-agnostic) and not for the CODE keeps
that choice open.

Actuator gains, joint names and the rest home pose are the workshop's, and they are
the valuable part: they were tuned against a real SO-101. Kept verbatim, with the
gear ratios its comments recorded.

Source: thirdparty/sim2real_so101/source/sim_to_real_so101/assets/so101.py
        (NVIDIA, Apache-2.0)
"""

from __future__ import annotations

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Root of the vendored workshop, which owns the arm USD.
WORKSHOP = pathlib.Path(__file__).resolve().parents[3] / "thirdparty" / "sim2real_so101"
WORKSHOP_USD = WORKSHOP / "source" / "sim_to_real_so101" / "assets" / "usd"

# The no-camera build: this workspace mounts its own camera in klip_support-1, so
# the workshop's modelled camera would be a second, wrong lens hanging off the wrist.
SO101_USD = WORKSHOP_USD / "SO-ARM101-USD-NO-CAMERA.usd"

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Rest pose, radians. The workshop's, which is a sane "arm up and out of the way".
REST_POSE = {
    "Rotation": -0.2736,
    "Pitch": -0.6109,
    "Elbow": -0.0745,
    "Wrist_Pitch": 1.5148,
    "Wrist_Roll": -1.6034,
    "Jaw": -0.1465,
}


def _quat_z(deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a yaw-only rotation, so this module needs no isaacsim import."""
    import math

    half = math.radians(deg) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_USD),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=dict(REST_POSE),
        pos=(-0.05, 0.0, 0.0),
        rot=_quat_z(90),
    ),
    actuators={
        # Gear 1/191, stall 34.4 N-m
        "rotation": ImplicitActuatorCfg(joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=55, damping=0.7),
        # Gear 1/345, stall 62.1 N-m -- carries the whole arm
        "pitch": ImplicitActuatorCfg(joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=30, damping=0.8),
        # Gear 1/191, stall 34.4 N-m
        "elbow": ImplicitActuatorCfg(joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=25, damping=0.7),
        # Gear 1/147, stall 26.5 N-m
        "wrist_pitch": ImplicitActuatorCfg(joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=12, damping=0.5),
        "wrist_roll": ImplicitActuatorCfg(joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=7, damping=0.5),
        "gripper": ImplicitActuatorCfg(joint_names_expr=["Jaw"], effort_limit_sim=30, stiffness=4, damping=0.3),
    },
)
