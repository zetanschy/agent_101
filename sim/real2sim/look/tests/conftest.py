"""LOOK tests: import real2sim from this checkout, headless MuJoCo, skip helpers."""
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
    from real2sim.scene import load

    return load()


@pytest.fixture(scope="session")
def built():
    """The look.json of the current build; skips when `look build` has not run."""
    import json

    p = paths.out_dir(None, "look", "look.json")
    if not p.exists():
        pytest.skip("run ./robot real2sim look build first")
    return json.loads(p.read_text())
