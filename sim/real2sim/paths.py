"""Where real2sim reads and writes, and which interpreter runs what.

One source of truth for both Python and shell. The run.sh files do not repeat any
of this: sim/real2sim/env.sh evaluates `python3 -m real2sim.paths --shell`, so a
path or an interpreter changes here and nowhere else.

INTERPRETERS (all measured on this box, understand-phase tooling report):
    HOST_PY    python3 3.10: mujoco 3.6, cv2 4.11, trimesh 4.11, pandas, pyarrow,
               pytest 6.2. The core's own commands and tests run here.
    MJLAB_PY   the mjlab uv env, Python 3.13: mujoco 3.11, mujoco-warp, torch; NO cv2.
               `env -u PYTHONPATH` is mandatory: the login shell's ROS Humble
               PYTHONPATH holds python3.10 site-packages, and a 3.13 interpreter that
               sees them fails inside uv. Only sim/ is put back.
    ISAAC_PY   scripts/sim/sim.sh, i.e. conda env 45pysaac (Isaac Sim 4.5.0, Isaac
               Lab 2.1, mujoco 3.6, cv2). sim.sh itself prepends sim/ to PYTHONPATH.
    BLENDER    Blender 5.2 snap, headless. Blender IGNORES PYTHONPATH unless told
               --python-use-system-env, and that would drag the ROS paths in too, so
               sim/ is inserted with --python-expr, which Blender runs before the
               --python script that follows. Append: script, "--", script args.
    COACD_PY   the only env with CoACD (conda astribot_simu, Python 3.10).
    DOCKER_IMAGE  agent101/lerobot (lerobot 0.6.1, pyav): the only thing that can
               read the dataset's root-owned mode-600 videos.

GPU. Every Isaac run and every Blender GPU render must hold GPU_LOCK (flock), so
heavy GPU jobs from different processes serialize on one 12 GB card. Point
R2S_GPU_LOCK at a shared lock file to join another scheduler's queue.

DATASET. Picked by --ds, else $R2S_DATASET, else DEFAULT_DATASET.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent  # sim/real2sim
SIM = PKG.parent  # sim/  (goes on PYTHONPATH)
REPO = SIM.parent  # repo root
CONFIG_DIR = PKG / "config"
OUTPUTS = SIM / "outputs" / "real2sim"  # gitignored (sim/outputs/), on the big disk

HF_USER = "soarm101"
DEFAULT_DATASET = "testingt_real2sim_20260927_152936"

# The arm model every engine and every FK in real2sim uses. The Isaac USD agrees with
# it (Isaac report: FK gripper position identical), and so does the repo's URDF FK
# (tests/test_kinematics.py, < 0.01 mm).
MJLAB_SO101 = REPO / "thirdparty/mjlab/src/mjlab/asset_zoo/robots/so_arm101"
MJCF = MJLAB_SO101 / "xmls" / "so101_calib.xml"
KLIP_CAMERA_PY = MJLAB_SO101 / "klip_camera.py"  # mjlab's wrist-camera mount (READ only)
SO101_CONSTANTS_PY = MJLAB_SO101 / "so101_constants.py"  # GRASP_SITE_POS (READ only)
# lerobot 0.6.1 loads calibration/robots/<robot.name>/ and SOFollower.name is
# "so_follower" -- not so101_follower (joint_units report, section 3).
FOLLOWER_CALIBRATION = REPO / "calibration" / "robots" / "so_follower" / "zetans_follower.json"
# The repo's URDF FK. Loaded BY FILE PATH in tests: importing the sim_agent101
# package boots Isaac.
SIM_AGENT_KINEMATICS = SIM / "sim_agent101" / "kinematics.py"

CONFIG_LAYERS = ("", "calib", "servo")  # merge order: <ds>.json <- .calib.json <- .servo.json

HOST_PY = ("env", "-u", "PYTHONPATH", f"PYTHONPATH={SIM}", "python3")
MJLAB_PY = (
    "env", "-u", "PYTHONPATH", f"PYTHONPATH={SIM}",
    "uv", "run", "--project", str(REPO / "thirdparty" / "mjlab"), "--frozen", "--no-sync", "python",
)
ISAAC_PY = ("bash", str(REPO / "scripts" / "sim" / "sim.sh"))
BLENDER = (
    "/snap/bin/blender", "-b", "--factory-startup",
    "--python-expr", f"import sys; sys.path.insert(0, {str(SIM)!r})",
    "--python",
)
COACD_PY = ("env", "-u", "PYTHONPATH", f"PYTHONPATH={SIM}", "/mnt/storage/anaconda3/envs/astribot_simu/bin/python")
DOCKER_IMAGE = "agent101/lerobot:latest"

GPU_LOCK = Path(os.environ.get("R2S_GPU_LOCK", str(OUTPUTS / ".gpu.lock")))


def gpu_locked(cmd) -> tuple[str, ...]:
    """Prefix a command so it waits for, and holds, the shared GPU lock."""
    return ("flock", str(GPU_LOCK), *map(str, cmd))


def dataset(ds: str | None = None) -> str:
    """The dataset NAME (no user prefix): argument, else $R2S_DATASET, else the default."""
    name = ds or os.environ.get("R2S_DATASET") or DEFAULT_DATASET
    return name.split("/")[-1]


def repo_id(ds: str | None = None) -> str:
    return f"{HF_USER}/{dataset(ds)}"


def hf_root(ds: str | None = None) -> Path:
    """Local LeRobot dataset root, honouring HF_LEROBOT_HOME as lerobot does."""
    home = os.environ.get("HF_LEROBOT_HOME") or str(Path.home() / ".cache" / "huggingface" / "lerobot")
    return Path(home) / repo_id(ds)


def out_dir(ds: str | None = None, *parts: str, create: bool = False) -> Path:
    """sim/outputs/real2sim/<ds>/<parts...>; create=True makes the directory."""
    p = OUTPUTS.joinpath(dataset(ds), *parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def config_path(ds: str | None = None, layer: str = "") -> Path:
    """config/<ds>.json, or config/<ds>.<layer>.json for layer 'calib' / 'servo'."""
    if layer not in CONFIG_LAYERS:
        raise ValueError(f"layer must be one of {CONFIG_LAYERS}, got {layer!r}")
    return CONFIG_DIR / (f"{dataset(ds)}.json" if not layer else f"{dataset(ds)}.{layer}.json")


def episodes_npz(ds: str | None = None) -> Path:
    return out_dir(ds, "episodes.npz")


def frames_npy(ds: str | None = None, cam: str = "front") -> Path:
    return out_dir(ds, "frames", f"{cam}.npy")


def shell_exports(ds: str | None = None) -> str:
    """`export R2S_...=...` lines for env.sh. Command lines are space-joined, so a
    run.sh uses them unquoted ($R2S_MJLAB_PY script.py); no path here has a space."""
    vals = {
        "R2S_REPO": str(REPO), "R2S_SIM": str(SIM), "R2S_PKG": str(PKG),
        "R2S_DATASET": dataset(ds), "R2S_OUT": str(out_dir(ds)), "R2S_OUTPUTS": str(OUTPUTS),
        "R2S_HOST_PY": " ".join(HOST_PY), "R2S_MJLAB_PY": " ".join(MJLAB_PY),
        "R2S_ISAAC_PY": " ".join(ISAAC_PY), "R2S_COACD_PY": " ".join(COACD_PY),
        # BLENDER's --python-expr argument contains spaces, so it is exported as a
        # shell-quoted string: use it as  eval "$R2S_BLENDER script.py -- args".
        "R2S_BLENDER": " ".join(shlex.quote(a) for a in BLENDER),
        "R2S_DOCKER_IMAGE": DOCKER_IMAGE, "R2S_GPU_LOCK": str(GPU_LOCK),
    }
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in vals.items())


if __name__ == "__main__":
    if "--shell" in sys.argv:
        print(shell_exports())
    else:
        for k in ("REPO", "SIM", "PKG", "CONFIG_DIR", "OUTPUTS", "MJCF", "FOLLOWER_CALIBRATION", "GPU_LOCK"):
            print(f"{k:22s} {globals()[k]}")
        print(f"{'dataset':22s} {dataset()}  ({hf_root()})")
        for k in ("HOST_PY", "MJLAB_PY", "ISAAC_PY", "BLENDER", "COACD_PY"):
            print(f"{k:22s} {shlex.join(globals()[k])}")
