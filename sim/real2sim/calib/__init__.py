"""CALIB: a dataset-only, joint, multi-view calibration of the replay's geometry.

RULE 1 of real2sim: camera parameters come from THIS dataset's recorded episodes
(front + wrist frames together with observation.state) and nowhere else. The stored
calibrations (cameras.json, extrinsics.json, the ChArUco images, the Isaac overrides,
the understand phase's fit6r) enter only as starting points and, for the trusted
front K, as the weak prior the task allows.

UNKNOWNS (params.py), all in the URDF base frame: both cameras' pose, K, distortion
and latency (the wrist camera's pose on the `gripper` link); the 5 arm joint zero
offsets and the gripper map; the table height; one cap type (diameter, height,
open-end lip) and its resting centre in every episode; the mug (rim radius, rim
height) and its centre + handle yaw in every episode.

OBSERVATIONS, fitted jointly (robust least squares, fit.py):
    front  arm silhouettes vs the FK arm rendered in MuJoCo (silhouette.py), resting cap
           blobs, the mug's rim ring, red wall and handle (objfit.py)
    grip   fixed-finger + moving-jaw silhouettes (mount, lens, gripper map), each resting
           cap's silhouette during its approach, the mug rim at each release (objfit.py)
    both   the same objects at the same world points in both views at the same
           timesteps, through FK of the recorded joints at each image's exposure
           (timing.py: images trail observation.state)
    physics soft constraints the real episodes prove (physfit.py)
Held-out scoring and cross-view metrics: evaluate.py. The per-frame real observations
other tracks reuse: observations.py. Orchestration: pipeline.py; outputs: report.py.

    ./robot real2sim calib observe | fit [--quick] | report | label | test
"""
