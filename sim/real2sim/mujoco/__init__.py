"""MUJOCO track: the real episodes replayed physically in MuJoCo 3.11 (CPU, mjlab uv env).

    ./robot real2sim mujoco servo-fit            identify the STS3215 servo model -> config/<ds>.servo.json
    ./robot real2sim mujoco replay --episode N --mode action|track|kinematic [--seed S] [--ensemble K]
                                   [--render] [--gui]
    ./robot real2sim mujoco eval                 every used episode, ensembles -> mujoco/eval.json

WHAT IS SIMULATED. The SO-ARM101 of paths.MJCF with its base at the world origin (the
URDF base frame every real2sim module uses), driven through the identified servo
model (servo.py) by the RECORDED ACTION delayed by the identified dead time, grasping
the episode's caps off a table whose top is at scene.table_z() and dropping them into
the mug. All objects are placed once at reset (objects.episode_objects) and move only
through contact: no weld, no equality, no scripted motion, no force added to any
object. The gripper can squeeze no harder than the servo's identified torque limit,
and the arm no harder than the STS3215 stall torque.

MODULES
    servo      the servo model (engine-agnostic SI parameters) and the command stream
    servo_fit  its identification from action -> observation.state (ep0+ep1, ep2 held out)
    grip_hold  the gripper's blocked-jaw data: compliance and torque limit
    fingers    finger collision geometry: CoACD of the real finger meshes (cached)
    model      the MjSpec scene: arm, mount, cameras, table, caps, mug, contacts
    replay     the replay driver (modes action / track / kinematic) and its pose log
    evaluate   per-pick outcomes, physics-honesty and fidelity metrics, ensembles
    render     MuJoCo EGL renders of both cameras through the real distortion -> mp4
    viewer     the interactive mujoco.viewer (needs a display)
    cli        the command line behind run.sh

Everything here runs in the mjlab uv env ($R2S_MJLAB_PY: mujoco 3.11, numpy 2, scipy,
imageio; no cv2), except fingers.build, which needs CoACD ($R2S_COACD_PY).
"""

import os as _os
import sys as _sys

if __name__ == "mujoco":
    # This directory is sim/real2sim/mujoco, the package `real2sim.mujoco`. Python started
    # with its CURRENT DIRECTORY at sim/real2sim (e.g. `python -m pytest` there, which is
    # how `./robot real2sim core test` runs) puts that directory on sys.path, and a bare
    # `import mujoco` then finds this package instead of the MuJoCo library (MEASURED: the
    # core tests failed on `mujoco.MjModel`). Hand the importer the real library: CPython's
    # import returns whatever sys.modules holds for the name once this file has run.
    _here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _saved = list(_sys.path)
    _sys.path[:] = [p for p in _sys.path if _os.path.abspath(p or _os.curdir) != _here]
    del _sys.modules["mujoco"]
    try:
        import importlib as _importlib

        _sys.modules["mujoco"] = _importlib.import_module("mujoco")
    finally:
        _sys.path[:] = _saved
