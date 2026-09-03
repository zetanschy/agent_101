"""Domain randomization for push-T.

Isaac Lab 2.1 ships randomizers for physics -- friction, mass, actuator gains --
and those are wired straight into EventCfg. It ships nothing usable here for
LIGHTS or CAMERAS, so those two are written out below as USD attribute writes,
following the same pattern as set_robot_color in push_t.py.

On magnitudes. The workshop jitters its cameras by +/-2 cm and +/-2.9 deg. That is
the right call when you never measured where your cameras are; it is the wrong one
here, where both were fitted to the real frames (the wrist to 3.1 px of boundary
error, the overhead to 5.9 px). Randomising wider than the calibration residual
throws away the calibration. So the defaults below are roughly that residual --
millimetres, not centimetres -- and the lighting, which really is uncontrolled on
an open bench under room light, gets the wide range instead.
"""

from __future__ import annotations

import torch

DOME_LIGHT = "/World/DomeLight"
KEY_LIGHT = "/World/KeyLight"

# Defaults are read from the stage the first time a term runs and offsets are applied
# to THOSE, never to the current value. Offsetting from the current value makes every
# reset a step in a random walk, and after a hundred episodes the cameras have wandered
# somewhere unrelated to where they were calibrated.
_DEFAULTS: dict[str, object] = {}


def _uniform(lo: float, hi: float) -> float:
    return float(torch.empty(1).uniform_(lo, hi).item())


def randomize_lighting(
    env,
    env_ids,
    dome_intensity: tuple[float, float] | None = (600.0, 1400.0),
    key_intensity: tuple[float, float] | None = (3500.0, 8500.0),
    color_temperature: tuple[float, float] | None = (4500.0, 7500.0),
) -> None:
    """Jitter the two lights' intensity and colour temperature.

    The highest-value randomization in this scene by some distance: the real rig is
    an open table under whatever the room is doing, so brightness and colour cast
    vary far more between real sessions than any geometric quantity does.
    """
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Sdf, Usd

    stage = env.sim.stage
    with Sdf.ChangeBlock():
        for path, rng in ((DOME_LIGHT, dome_intensity), (KEY_LIGHT, key_intensity)):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            if rng is not None:
                attr = prim.GetAttribute("inputs:intensity")
                if attr:
                    attr.Set(_uniform(*rng))
            if color_temperature is not None:
                enable = prim.GetAttribute("inputs:enableColorTemperature")
                temp = prim.GetAttribute("inputs:colorTemperature")
                if enable and temp:
                    enable.Set(True)
                    temp.Set(_uniform(*color_temperature))
    _ = (env_ids, prim_utils, Usd)


def _cache_xform(stage, path: str):
    """(translate, orient) as first seen, so offsets never compound."""
    from pxr import UsdGeom

    if path in _DEFAULTS:
        return _DEFAULTS[path]
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        _DEFAULTS[path] = None
        return None
    ops = {op.GetOpType(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    t_op = ops.get(UsdGeom.XformOp.TypeTranslate)
    o_op = ops.get(UsdGeom.XformOp.TypeOrient)
    _DEFAULTS[path] = (t_op, o_op, None if t_op is None else t_op.Get(),
                       None if o_op is None else o_op.Get())
    return _DEFAULTS[path]


def randomize_camera_pose(env, env_ids, cameras: dict[str, dict] | None = None) -> None:
    """Jitter each camera about its CALIBRATED pose.

    cameras maps a prim path to {"pos_m": (dx, dy, dz), "rot_deg": (rx, ry, rz)},
    each entry the half-width of a uniform range. Rotation is applied as a small
    quaternion on the right, i.e. in the camera's own frame, so "rot_deg" means
    pan/tilt/roll of the camera rather than rotation about the env axes.
    """
    import math

    from pxr import Gf, Sdf

    if cameras is None:
        return
    stage = env.sim.stage
    with Sdf.ChangeBlock():
        for path, spec in cameras.items():
            cached = _cache_xform(stage, path)
            if not cached:
                continue
            t_op, o_op, t0, q0 = cached
            if t_op is not None and t0 is not None:
                d = spec.get("pos_m", (0.0, 0.0, 0.0))
                t_op.Set(Gf.Vec3d(float(t0[0] + _uniform(-d[0], d[0])),
                                  float(t0[1] + _uniform(-d[1], d[1])),
                                  float(t0[2] + _uniform(-d[2], d[2]))))
            if o_op is not None and q0 is not None:
                r = spec.get("rot_deg", (0.0, 0.0, 0.0))
                half = [math.radians(_uniform(-v, v)) / 2 for v in r]
                dq = Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
                for axis, h in enumerate(half):
                    v = [0.0, 0.0, 0.0]
                    v[axis] = math.sin(h)
                    dq = dq * Gf.Quatd(math.cos(h), Gf.Vec3d(*v))
                o_op.Set(q0 * dq)
    _ = env_ids


def randomize_robot_color(env, env_ids, grey: tuple[float, float] = (0.86, 0.97),
                          tint: float = 0.03) -> None:
    """Small jitter about white, not a colour palette.

    The workshop cycles orange / teal / white / black because its users' arms vary.
    This arm is white and we know it; the useful randomization is the shade and cast
    a camera actually reports under changing light, which is a few percent.
    """
    from .push_t import set_robot_color

    g = _uniform(*grey)
    set_robot_color(env, env_ids,
                    color=(min(1.0, g + _uniform(-tint, tint)),
                           min(1.0, g + _uniform(-tint, tint)),
                           min(1.0, g + _uniform(-tint, tint))))
