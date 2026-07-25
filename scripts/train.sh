#!/usr/bin/env bash
# LoRA fine-tune pi05 (or any policy base) on a LeRobot dataset.
# Runs anywhere the le101 fork (lerobot 0.6.1) + peft is installed — a cloud GPU
# (Vast.ai/RunPod, via setup-cloud.sh) or this repo's image. Just GPU + dataset.
#
# On this machine use the wrapper (GPU compose + tokens wired in):
#   ./robot train --dataset zetanschy/cap_to_cup --name pi05_cap
#   ./robot train --dataset zetanschy/cap_to_cup --name pi05_cap --push
#   ./robot train --dataset zetanschy/cap_to_cup --name pi05_cap \
#        --steps 20000 --batch 32 --rank 16
# On a cloud GPU box (setup-cloud.sh, no docker) call it directly:
#   bash scripts/train.sh --dataset zetanschy/cap_to_cup --name pi05_cap
#
# LoRA + bf16 + gradient checkpointing keep VRAM low (fits ~24GB). pi05 applies
# sensible default LoRA targets (gemma_expert q/v + action projections).
# Resume a crashed run: same --name, add `--resume true`.
# --push  auto-uploads the trained model to <your-hf-user>/<name> (needs hf login).
#
# Weights & Biases: ON automatically once credentials exist (WANDB_API_KEY, or a
# previous `wandb login`), OFF otherwise — nothing to remember per run. The run is
# named after --name, in project $WANDB_PROJECT (.env). Force with --wandb /
# --no-wandb; --wandb-offline logs to disk for a later `wandb sync`. Resuming with
# the same --name re-attaches to the same wandb run (id read from the output dir).
#   ./robot login                                # stores the key in .env.local
#   echo 'WANDB_API_KEY=<key>' >> .env.local     # or by hand; gitignored
#
# pi05_base was pretrained on DROID's cameras (base_0_rgb/left_wrist_0_rgb/
# right_wrist_0_rgb). Our datasets use front/grip, so by default we rename them to
# a subset of those (front->base_0_rgb, grip->left_wrist_0_rgb) — the validator
# accepts a dataset that's a subset of the policy's cameras. Override with
# --rename-map '<json>' (e.g. add side->right_wrist_0_rgb for 3-cam), or --no-rename.
set -euo pipefail

# Credentials/config from .env (project/entity) and .env.local (the secrets), so
# this works both in the container and on a bare cloud box. HF_TOKEN is included
# because outside docker nothing else loads it — compose's env_file does that job
# in the container, and without it --push fails whoami even after ./robot login.
# Already-exported vars win; commented-out lines are skipped by the prefix filter.
root="$(cd "$(dirname "$0")/.." && pwd)"
for f in "$root/.env" "$root/.env.local"; do
  [ -f "$f" ] || continue
  while IFS='=' read -r k v; do
    case "$k" in WANDB_*|HF_TOKEN) ;; *) continue ;; esac
    if [ -z "${!k:-}" ] && [ -n "$v" ]; then export "$k=$v"; fi
  done < "$f"
done

dataset=""; name="run"; base="lerobot/pi05_base"; steps=20000; batch=16; rank=16; push=0; extra=()
wandb=auto; wbproject="${WANDB_PROJECT:-agent_101}"; wbentity="${WANDB_ENTITY:-}"; wbmode=""; wbartifacts=1
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
    --wandb)       wandb=1; shift ;;         # force on (fails fast if no credentials)
    --no-wandb)    wandb=0; shift ;;         # force off
    --wandb-project) wbproject="$2"; shift 2 ;;
    --wandb-entity)  wbentity="$2"; shift 2 ;;   # team/org, if not your personal one
    --wandb-offline) wbmode="offline"; shift ;;  # log to disk; `wandb sync` later
    --no-wandb-artifacts) wbartifacts=0; shift ;;  # don't upload LoRA checkpoints
    --rename-map)  rename="$2"; shift 2 ;;   # override the camera rename JSON
    --no-rename)   rename=""; shift ;;       # disable (dataset cameras already match)
    *) extra+=("$1"); shift ;;
  esac
done
[ -n "$dataset" ] || { echo "need --dataset <hf_repo_id>   (e.g. zetanschy/cap_to_cup)" >&2; exit 1; }
[ -n "$rename" ] && renameargs=(--rename_map="$rename") || renameargs=()

# wandb: a key in the env or a `wandb login` netrc entry is enough. Offline needs neither.
have_wandb_creds() {
  [ -n "${WANDB_API_KEY:-}" ] && return 0
  grep -qs 'api\.wandb\.ai' "${HOME}/.netrc" && return 0
  return 1
}
if [ "$wandb" = auto ]; then
  if [ -n "$wbmode" ] || have_wandb_creds; then wandb=1; else wandb=0; fi
fi
if [ "$wandb" = 1 ]; then
  if [ -z "$wbmode" ] && ! have_wandb_creds; then
    echo "wandb: no credentials found. Either" >&2
    echo "  echo 'WANDB_API_KEY=<key>' >> .env.local   (from wandb.ai/authorize)" >&2
    echo "  wandb login                                (caches ~/.netrc)" >&2
    echo "or use --wandb-offline / --no-wandb." >&2
    exit 1
  fi
  # job_name becomes the wandb run name (output_dir is set below, so it's unaffected).
  wandbargs=(--wandb.enable=true --wandb.project="$wbproject" --job_name="$name")
  if [ -n "$wbentity" ]; then wandbargs+=(--wandb.entity="$wbentity"); fi
  if [ -n "$wbmode" ]; then wandbargs+=(--wandb.mode="$wbmode"); fi
  if [ "$wbartifacts" = 0 ]; then wandbargs+=(--wandb.disable_artifact=true); fi
  echo "wandb: project=$wbproject${wbentity:+ entity=$wbentity} run=$name${wbmode:+ (${wbmode})}"
else
  wandbargs=(--wandb.enable=false)
  echo "wandb: off (no credentials — see --wandb / --wandb-offline)"
fi

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
  "${wandbargs[@]}" \
  "${extra[@]}"
