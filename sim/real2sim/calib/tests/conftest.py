"""Import real2sim from this checkout; headless MuJoCo."""
import os
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[3]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
os.environ.setdefault("MUJOCO_GL", "egl")
