# shellcheck shell=bash
# Source this from a real2sim run.sh: it exports R2S_* (paths, dataset, interpreters,
# GPU lock) from real2sim/paths.py, the one place they are defined, so the shell and
# Python cannot disagree. Idempotent; honours R2S_DATASET and R2S_GPU_LOCK if set.
#
#   $R2S_HOST_PY  $R2S_MJLAB_PY  $R2S_ISAAC_PY  $R2S_COACD_PY   use unquoted: $R2S_MJLAB_PY x.py
#   $R2S_BLENDER  shell-quoted:  eval "$R2S_BLENDER script.py -- args"
#   flock "$R2S_GPU_LOCK" <cmd>   around every Isaac run and every Blender GPU render
if [ -z "${R2S_ENV_LOADED:-}" ]; then
  _r2s_sim="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  # host python3 with ONLY sim/ on the path: the login shell's ROS PYTHONPATH is not ours.
  eval "$(env -u PYTHONPATH PYTHONPATH="$_r2s_sim" PYTHONDONTWRITEBYTECODE=1 python3 -m real2sim.paths --shell)"
  export R2S_ENV_LOADED=1
  unset _r2s_sim
fi
