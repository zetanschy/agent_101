#!/usr/bin/env bash
# Run a python entry point inside the Isaac Sim environment.
#
# Isaac Sim is a native conda install here, not a container: it needs the GPU, a
# 2.3 GB shader cache in $HOME, and 37 GB of Omniverse data that would be absurd to
# bake into an image. So unlike the rest of this repo, the sim commands do NOT go
# through Docker.
#
# SIM_CONDA_ENV picks the environment (.env). Two are installed on this box and only
# one works: 45pysaac (Isaac Sim 4.5.0 + Isaac Lab 2.1.0) boots; env_isaaclab
# (Isaac Sim 5.1.0 + Isaac Lab 2.3.0) hangs in Kit startup at ~700 ms with the main
# thread spinning, with or without cameras. If you ever fix 5.1, the only thing that
# should need changing is this variable -- sim/sim_agent101 deliberately depends on
# the workshop's USD assets rather than its Isaac Lab 2.3 Python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# .env then .env.local, the order the compose files use: shared config first, then
# the gitignored overrides and secrets (HF_TOKEN, WANDB_API_KEY). The sim commands are
# the ones that do NOT go through Docker, so without this nothing ever loads
# .env.local for them -- and the failure is silent, because a missing key just means
# sim-train quietly logs to tensorboard instead of wandb.
set -a
[ -f "$ROOT/.env" ] && . "$ROOT/.env"
[ -f "$ROOT/.env.local" ] && . "$ROOT/.env.local"
set +a

ENV_NAME="${SIM_CONDA_ENV:-45pysaac}"
CONDA_ROOT="${SIM_CONDA_ROOT:-$HOME/anaconda3}"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"

if [ ! -x "$PY" ]; then
  echo "no python at $PY" >&2
  echo "set SIM_CONDA_ENV / SIM_CONDA_ROOT in .env, or: conda env list" >&2
  exit 1
fi

# Kit refuses to start without this and there is no interactive prompt to answer
# from a script.
export OMNI_KIT_ACCEPT_EULA=YES
# sim/ holds our package; the workshop's own package is NOT on the path on purpose.
export PYTHONPATH="$ROOT/sim${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" "$@"
