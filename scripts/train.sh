#!/usr/bin/env bash
# LoRA fine-tune pi05 (or any policy base) on a LeRobot dataset.
# Runs anywhere the le101 fork (lerobot 0.6.1) + peft is installed — a cloud GPU
# (Vast.ai/RunPod, via setup-cloud.sh) or this repo's image. Just GPU + dataset.
#
#   bash scripts/train.sh --dataset zetanschy/cap_to_cup --name pi05_cap
#   bash scripts/train.sh --dataset zetanschy/cap_to_cup --name pi05_cap --push
#   bash scripts/train.sh --dataset zetanschy/cap_to_cup --name pi05_cap \
#        --steps 20000 --batch 32 --rank 16
#
# LoRA + bf16 + gradient checkpointing keep VRAM low (fits ~24GB). pi05 applies
# sensible default LoRA targets (gemma_expert q/v + action projections).
# Resume a crashed run: same --name, add `--resume true`.
# --push  auto-uploads the trained model to <your-hf-user>/<name> (needs hf login).
#
# pi05_base was pretrained on DROID's cameras (base_0_rgb/left_wrist_0_rgb/
# right_wrist_0_rgb). Our datasets use front/grip, so by default we rename them to
# a subset of those (front->base_0_rgb, grip->left_wrist_0_rgb) — the validator
# accepts a dataset that's a subset of the policy's cameras. Override with
# --rename-map '<json>' (e.g. add side->right_wrist_0_rgb for 3-cam), or --no-rename.
set -euo pipefail

dataset=""; name="run"; base="lerobot/pi05_base"; steps=20000; batch=16; rank=16; push=0; extra=()
rename='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.grip": "observation.images.left_wrist_0_rgb"}'
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)     dataset="$2"; shift 2 ;;
    --name)        name="$2"; shift 2 ;;
    --base)        base="$2"; shift 2 ;;
    --steps)       steps="$2"; shift 2 ;;
    --batch)       batch="$2"; shift 2 ;;
    --rank)        rank="$2"; shift 2 ;;
    --push)        push=1; shift ;;
    --rename-map)  rename="$2"; shift 2 ;;   # override the camera rename JSON
    --no-rename)   rename=""; shift ;;       # disable (dataset cameras already match)
    *) extra+=("$1"); shift ;;
  esac
done
[ -n "$dataset" ] || { echo "need --dataset <hf_repo_id>   (e.g. zetanschy/cap_to_cup)" >&2; exit 1; }
[ -n "$rename" ] && renameargs=(--rename_map="$rename") || renameargs=()

# Push target: default OFF (no HF write needed). --push -> <your-login>/<name>.
if [ "$push" = 1 ]; then
  who=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null) \
    || { echo "not logged in — run 'hf auth login', or drop --push" >&2; exit 1; }
  pushargs=(--policy.repo_id="${who}/${name}")
  echo "will push trained model -> hf.co/${who}/${name}"
else
  pushargs=(--policy.push_to_hub=false)
fi

out="outputs/train/${name}"
echo "LoRA fine-tune: base=$base  dataset=$dataset  ->  $out  (batch=$batch steps=$steps rank=$rank)"

set -x
lerobot-train \
  --policy.path="$base" \
  --dataset.repo_id="$dataset" \
  --peft.method_type=LORA \
  --peft.r="$rank" \
  --output_dir="$out" \
  --batch_size="$batch" \
  --steps="$steps" \
  --save_freq=5000 \
  --num_workers=8 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  "${renameargs[@]}" \
  "${pushargs[@]}" \
  "${extra[@]}"
