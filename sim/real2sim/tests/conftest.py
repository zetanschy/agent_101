"""Shared setup: import real2sim from this checkout, headless MuJoCo, skip helpers."""
import os
import sys
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[2]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
os.environ.setdefault("MUJOCO_GL", "egl")

from real2sim import paths  # noqa: E402


@pytest.fixture(scope="session")
def scene():
    from real2sim.scene import load

    return load()


@pytest.fixture(scope="session")
def extracted():
    if not paths.episodes_npz().exists():
        pytest.skip("run ./robot real2sim core extract first")
    return True
