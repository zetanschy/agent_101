"""MuJoCo track tests: mjlab uv env (mujoco 3.11), headless EGL, the real dataset extracted."""
import os
import sys
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[3]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
os.environ.setdefault("MUJOCO_GL", "egl")

from real2sim import paths  # noqa: E402


@pytest.fixture(scope="session")
def scene():
    if not paths.episodes_npz().exists():
        pytest.skip("run ./robot real2sim core extract first")
    from real2sim.scene import load

    return load()


@pytest.fixture(scope="session")
def servo(scene):
    from real2sim.mujoco.servo import ServoModel

    return ServoModel.from_scene(scene)


@pytest.fixture(scope="session")
def built(scene, servo):
    from real2sim.mujoco import model

    return model.build(scene, 0, servo)
