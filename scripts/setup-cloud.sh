#!/usr/bin/env bash
# One-shot setup for a box WITHOUT Docker — a rented GPU (Vast.ai / RunPod) where
# you install straight onto the host. Assumes a CUDA base image + Python 3.12.
#   git clone --recursive https://github.com/zetanschy/agent_101 && cd agent_101
#   bash scripts/setup-cloud.sh
#
# On your own machine you do NOT need this: the image already has everything, so
#   ./robot build && ./robot preflight && ./robot train --dataset ... --name ...
# The training compose has no serial/camera devices, so it runs on a GPU box with
# no arms attached. Docker also keeps this identical across machines, which the
# host install cannot (python version, driver, system ffmpeg all vary).
set -euo pipefail

echo "== fetch the lerobot fork (le101 submodule) =="
# If cloned with --recursive it's already here; otherwise init the submodule,
# falling back to a direct clone if this isn't a git checkout.
git submodule update --init --recursive thirdparty/le101 2>/dev/null \
  || { [ -e thirdparty/le101/pyproject.toml ] || git clone --depth 1 https://github.com/zetanschy/le101 thirdparty/le101; }

echo "== install: cu128 torch, then editable fork (training + pi + LoRA) =="
python -m pip install -q --upgrade pip
python -m pip install -q --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.12" "torchvision>=0.22,<0.27"
python -m pip install -q -e "thirdparty/le101[training,pi,peft]"

echo "== GPU check =="
python -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok, '| GPU:', torch.cuda.get_device_name(0) if ok else 'NONE — rent a GPU box!')"

cat <<'EOF'

== next steps ==
1) bash scripts/login.sh         # HF (WRITE) + wandb keys in one go -> .env.local
                                 # or copy .env.local over from your workstation:
                                 #   scp .env.local user@box:agent_101/
                                 # train.sh turns wandb on by itself once a key exists.
2) bash scripts/preflight.sh             # gpu kernels, vram, ram, disk, cpu, python
   bash scripts/preflight.sh --smoke     # 20 real steps: proves the batch fits
3) bash scripts/train.sh --dataset <you>/<dataset> --name pi05_run --push
   (LoRA + bf16 + grad-checkpointing -> fits ~24GB. Drop --push to keep it local.)
3) checkpoints land in outputs/train/pi05_run/ — pushed to the Hub if you passed --push.
   Loss curves show up in wandb automatically (project agent_101).
EOF
