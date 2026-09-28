"""real2sim: replay the real SO-ARM101 teleop episodes in simulation, physically.

A LeRobot dataset of the real arm (joints plus two cameras) goes in. Every real
episode comes back out as a simulated replay in two independent stacks that are
compared against the recording and against each other:

    mujoco  MuJoCo physics, rendered in Blender (Cycles/EEVEE)
    isaac   Isaac Sim 4.5 / Isaac Lab 2.1 PhysX, rendered with RTX

A replay only counts if it is PHYSICALLY FEASIBLE. Objects are placed once at reset
and move only through simulated contact. Nothing is welded to the gripper, no object
is scripted or teleported, and the servos stay within the torque the real STS3215 can
produce. It must also be CLOSE TO REALITY, and that is measured against the dataset
(`metrics`), never asserted.

This package is the engine-agnostic CORE. It imports only the standard library and
numpy at module level; mujoco, cv2 and trimesh are imported inside the functions
that need them. So `import real2sim.<module>` works in every interpreter this repo
uses: host python3 (3.10), the Isaac conda env 45pysaac (3.10), the mjlab uv env
(3.13) and Blender's bundled Python (3.13). Put `sim/` on PYTHONPATH (`paths` holds
the exact command line for each interpreter).

FRAME. Everything is expressed in the URDF BASE FRAME: the `base` body of mjlab's
so101_calib.xml with the base at identity. That is not the Isaac push-T env frame
(robot at (-0.05, 0, 0), yawed +90 deg) and not the mjlab scene frame (base yawed
+90 deg). In this frame, base -y is forward (image up in the overhead camera) and
+x is image left. z is up. The robot's own frame decides where "the table" is; the
table height is a scene parameter (`scene.table_z()`), not the origin.

UNITS. SI throughout: metres, radians, kilograms, seconds. The only exceptions are
named in the key: `_deg`, `_mm`, `_pct`, `_px`. Joint vectors are ordered as
units.URDF_JOINTS (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw), which is
LeRobot's order (shoulder_pan ... gripper). Quaternions are (w, x, y, z).
Recorded values stay in LeRobot units: arm joints in DEGREES, gripper in RANGE_0_100
percent. `units` converts, with the joint offsets and gripper map from the scene.

CAMERAS use the OpenCV model (`camera.Camera`): x right, y down, z forward. Pixel
(0, 0) is the CENTRE of the top-left pixel, and distortion is (k1, k2, p1, p2, k3).
The same object converts to MuJoCo, Blender and USD cameras.

FILES (per dataset `<ds>`, under sim/outputs/real2sim/<ds>/, gitignored):
    episodes.npz        action/state in LeRobot units plus episode bounds (`extract`)
    video/<cam>/...mp4  readable copies of the root-owned dataset videos
    frames/<cam>.npy    decoded uint8 N x 480 x 640 x 3, memmap it (`episodes.frames`)
    meshes/             cap and mug OBJ + STL, dims hash in the name (`objects`)
    <track>/            each track's own outputs
and the scene description in sim/real2sim/config/:
    <ds>.json           provisional values, every group with "source"   (CORE)
    <ds>.calib.json     cameras, table, joint offsets, object poses     (CALIB)
    <ds>.servo.json     identified actuator model                       (MUJOCO)
merged in that order by `scene.load`. Engine outputs share one pose-log format
(`poselog`).

OWNERSHIP. core = this directory's modules, run.sh, config/<ds>.json, tests/. calib/,
mujoco/, isaac/, look/ + blender/, and compare/ belong to their tracks, and each
track owns its run.sh. `./robot real2sim <track> ...` dispatches to them.

Modules: paths, extract, episodes, units, scene, kinematics, camera, objects,
poselog, metrics.
"""

SCHEMA = "real2sim.scene/1"
