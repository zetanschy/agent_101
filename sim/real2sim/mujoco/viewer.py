"""Watch a replay in the interactive mujoco.viewer (needs a display: DISPLAY=:0 on this box).

    ./robot real2sim mujoco replay --episode 0 --mode action --gui
    ./robot real2sim mujoco view --log <pose log .npz>

The replay is simulated first (faster than real time, replay.run) and the viewer then
PLAYS ITS POSE LOG at the recorded 30 fps, looping, with the model's collision pieces
toggleable (groups 2 = visuals, 3 = collision; keys 2/3 in the viewer). Playing a log
rather than stepping physics in the viewer's thread keeps what you see identical to
what was scored, and lets the same viewer show any engine's log.
"""

from __future__ import annotations

import os
import time

from . import model as mdl
from .render import set_from_log


def display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def play_log(built: mdl.Built, log: dict, speed: float = 1.0, loop: bool = True) -> None:
    """Open the passive viewer on `built` and play `log` until the window closes."""
    if not display_available():
        raise RuntimeError("no display: the viewer needs X (DISPLAY=:0 on this machine); "
                           "use --render for an mp4 instead")
    os.environ.pop("MUJOCO_GL", None)  # the viewer needs GLFW, not EGL
    import mujoco
    import mujoco.viewer

    d = mujoco.MjData(built.model)
    set_from_log(built, d, log, 0)
    period = 1.0 / (log["fps"] * speed)
    with mujoco.viewer.launch_passive(built.model, d) as v:
        v.opt.geomgroup[:] = 0
        v.opt.geomgroup[mdl.VISUAL_GROUP] = 1
        while v.is_running():
            for k in range(len(log["q"])):
                if not v.is_running():
                    break
                t0 = time.time()
                set_from_log(built, d, log, k)
                v.sync()
                time.sleep(max(0.0, period - (time.time() - t0)))
            if not loop:
                break
