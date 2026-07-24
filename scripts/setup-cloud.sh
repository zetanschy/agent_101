#!/usr/bin/env bash
# One-shot setup for a fresh cloud GPU box (Vast.ai / RunPod) to fine-tune with
# the le101 fork (lerobot 0.6.1 — LoRA/RL). Assumes a CUDA base image + Python 3.12.
#   git clone https://github.com/zetanschy/agent_101 && cd agent_101
#   bash scripts/setup-cloud.sh
set -euo pipefail

echo "== fetch the lerobot fork (le101) =="
[ -d thirdparty/le101 ] || git clone --depth 1 https://github.com/zetanschy/le101 thirdparty/le101

echo "== install: cu128 torch, then editable fork (training + pi + LoRA) =="
python -m pip install -q --upgrade pip
python -m pip install -q --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.12" "torchvision>=0.22,<0.27"
python -m pip install -q -e "thirdparty/le101[training,pi,peft]"

echo "== GPU check =="
python -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok, '| GPU:', torch.cuda.get_device_name(0) if ok else 'NONE — rent a GPU box!')"

cat <<'EOF'

== next steps ==
1) hf auth login                 # token with WRITE access (pull dataset, push checkpoints)
2) bash scripts/train.sh --dataset <you>/<dataset> --name pi05_run --push
   (LoRA + bf16 + grad-checkpointing -> fits ~24GB. Drop --push to keep it local.)
3) checkpoints land in outputs/train/pi05_run/ — pushed to the Hub if you passed --push.
EOF
