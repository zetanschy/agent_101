#!/usr/bin/env bash
# One-shot setup for openpi (JAX) training on a box WITHOUT Docker — e.g. a Vast.ai
# instance, which is itself a container and cannot nest one.
#
#   git clone --recursive https://github.com/zetanschy/agent_101 && cd agent_101
#   bash scripts/setup-openpi-cloud.sh
#   bash scripts/openpi_train.sh --exp-name=smoke --batch-size 16 --overwrite
#
# This mirrors Dockerfile.openpi, which is the version that has been verified to
# import cleanly — deliberately NOT `uv sync` on openpi itself, because openpi's
# dependency set pins lerobot to an UPSTREAM commit (replacing the le101 fork),
# torch==2.7.1, and private i2rt, none of which the training path needs.
#
# The curated list below was derived by walking every module openpi.training.config
# and policy_config actually import. Notable inclusions/exclusions:
#   - pytest IS required: openpi imports it at module level in models_pytorch.
#   - tensorflow / tensorflow_datasets / dlimp are NOT: their only importer
#     (training/droid_rlds_dataset.py) imports them inside a function.
#   - lerobot needs the `dataset` extra, because openpi's policy_config reaches
#     training.checkpoints -> training.data_loader -> lerobot.datasets.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== python =="
py=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  found python $py (openpi requires >=3.10,<3.13)"
case "$py" in
  3.10|3.11|3.12) ;;
  *) echo "  unsupported; creating a 3.12 venv with uv" >&2
     command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; . "$HOME/.local/bin/env"; }
     uv venv --python 3.12 .venv-openpi
     # shellcheck disable=SC1091
     . .venv-openpi/bin/activate
     echo "  now on $(python -c 'import sys;print(sys.version.split()[0])') — re-activate with: . .venv-openpi/bin/activate" ;;
esac

echo "== system deps (ffmpeg/libgl for video decode; skipped if not root) =="
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 libgeos-dev portaudio19-dev build-essential git >/dev/null
  echo "  installed"
else
  echo "  skipped (not root) — ensure ffmpeg + libgl are present"
fi

python -m pip install -q --upgrade pip

# Leave an existing torch alone: rented images usually ship a CUDA build, and
# replacing it risks breaking the very nvidia-* wheels JAX then wants. torch is only
# used here by lerobot's dataset code, on CPU, so any recent version is fine.
echo "== torch =="
if python -c "import torch" 2>/dev/null; then
  echo "  keeping existing torch $(python -c 'import torch;print(torch.__version__)')"
else
  echo "  installing CPU torch (JAX owns the GPU here)"
  python -m pip install -q --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.7,<2.12" "torchvision>=0.22,<0.27"
fi

echo "== lerobot fork (le101) =="
git submodule update --init thirdparty/le101 2>/dev/null || true
python -m pip install -q -e "thirdparty/le101[feetech,dataset]"

echo "== jax + the openpi inference/training imports =="
python -m pip install -q "jax[cuda12]==0.5.3" \
    "flax==0.10.2" "orbax-checkpoint==0.11.13" "jaxtyping==0.2.36" \
    "ml_collections==1.0.0" "beartype==0.19.0" "transformers==4.53.2" \
    "augmax>=0.3.4" "equinox>=0.11.8" "etils[epath]" "treescope>=0.1.7" \
    "numpydantic>=1.6.6" "dm-tree>=0.1.8" "flatbuffers>=24.3.25" \
    "fsspec[gcs]>=2024.6.0" "tqdm-loggable>=0.2" "tyro>=0.9.5" \
    "sentencepiece>=0.2.0" "einops>=0.8.0" "filelock>=3.16.1" \
    optax chex pandas safetensors tqdm pytest "wandb>=0.19.1"

echo "== openpi (no deps, see header) =="
git submodule update --init thirdparty/openpi 2>/dev/null || true
python -m pip install -q --no-deps -e thirdparty/openpi
python -m pip install -q --no-deps -e thirdparty/openpi/packages/openpi-client

# openpi ships patched transformers modules and checks for them at runtime on the
# pytorch path. We train in JAX so it is not strictly needed, but the version is
# pinned to match, so apply it as openpi's README instructs.
cp -r thirdparty/openpi/src/openpi/models_pytorch/transformers_replace/* \
      "$(python -c 'import transformers,pathlib;print(pathlib.Path(transformers.__file__).parent)')/"

echo "== verify =="
python - <<'PY'
import jax
from openpi.training import config as c
cfg = c.get_config("pi05_soarm101_lora_cap_to_cup")
print(f"  jax {jax.__version__} | devices: {jax.devices()}")
print(f"  config OK: dataset={cfg.data.repo_id} batch={cfg.batch_size} steps={cfg.num_train_steps}")
print(f"  action_horizon={cfg.model.action_horizon}")
PY

cat <<'EOF'

== next ==
  bash scripts/openpi_train.sh --exp-name=smoke --batch-size 16 --overwrite
      -> computes norm stats (~12 min), then trains. Kill after ~20 steps to check
         that the batch fits and to read the step time.
  bash scripts/openpi_train.sh --exp-name=cap_to_cup_200 --overwrite
      -> the real run. Use tmux. Add --resume (NOT --overwrite) after a crash.
EOF
