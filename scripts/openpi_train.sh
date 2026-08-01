#!/usr/bin/env bash
# openpi (JAX) LoRA fine-tune: computes norm stats if they are missing, then trains.
#
#   ./robot openpi-train --exp-name=cap_to_cup_deg --overwrite
#   ./robot openpi-train <config> --exp-name=... [--batch-size 8] [--overwrite]
#   ./robot openpi-train --force-norm-stats ...      # recompute even if present
#
# Norm stats are a separate step in openpi — scripts/train.py does NOT compute them,
# and a missing or stale assets/ directory means training runs against the wrong
# statistics without complaining. So a fresh box (cloud or otherwise) needs the stats
# built before the first step, and doing it here means one command instead of two,
# with no way to forget the first.
#
# The stats live at assets/<config>/<dataset repo_id>/norm_stats.json, so they are
# keyed by dataset: change the repo_id and they are correctly treated as missing.
set -euo pipefail
cd "$(dirname "$0")/.."

# JAX preallocates only 75% of VRAM by default, i.e. ~18GB of a 24GB card — under the
# 22.5GB openpi documents for LoRA fine-tuning, so it OOMs before the model even fits.
# The compose services set this; a bare box (setup-openpi-cloud.sh) has nothing that
# would, so set it here and let an explicit value from the caller win.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"

# Credentials from .env / .env.local, same loader as train.sh. Needed because
# `bash scripts/login.sh` on a Docker-less box writes .env.local, and openpi's
# train.py wants WANDB_API_KEY (wandb_enabled defaults True) while the dataset pull
# wants HF_TOKEN if it is ever private. Already-exported values win.
for f in .env .env.local; do
  [ -f "$f" ] || continue
  while IFS='=' read -r k v; do
    case "$k" in WANDB_*|HF_TOKEN) ;; *) continue ;; esac
    if [ -z "${!k:-}" ] && [ -n "$v" ]; then export "$k=$v"; fi
  done < "$f"
done

force=0; stats_only=0; push=1; exp=""; cfg="pi05_soarm101_lora_cap_to_cup"; args=()
# The config name is only ever the FIRST argument, and only if it is not a flag.
# Everything after it is forwarded verbatim, so flags that take a separate value
# (`--num-workers 2`) keep their value instead of it being mistaken for the config.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then cfg="$1"; shift; fi
while [ $# -gt 0 ]; do
  case "$1" in
    --force-norm-stats) force=1; shift ;;
    --norm-stats-only)  stats_only=1; force=1; shift ;;   # ./robot openpi-norm-stats
    --no-push)          push=0; shift ;;                  # keep the result local
    --exp-name)         exp="$2"; args+=("$1" "$2"); shift 2 ;;
    --exp-name=*)       exp="${1#*=}"; args+=("$1"); shift ;;
    *) args+=("$1"); shift ;;
  esac
done

# openpi's scripts/ sits next to the package, but where that is depends on how it was
# installed: /opt/openpi in the images, thirdparty/openpi via setup-openpi-cloud.sh.
# Derive it from the imported module instead of hardcoding either.
openpi_root=$(python -c "import openpi, pathlib; print(pathlib.Path(openpi.__file__).parents[2])")
[ -f "$openpi_root/scripts/train.py" ] || {
  echo "openpi scripts not found under $openpi_root — is openpi installed?" >&2; exit 1; }

# Ask openpi where the stats belong rather than reconstructing the path here.
stats=$(python - "$cfg" <<'PY'
import sys
from openpi.training import config as _c
cfg = _c.get_config(sys.argv[1])
print(cfg.assets_dirs / cfg.data.repo_id / "norm_stats.json")
PY
)
echo "config:      $cfg"
echo "openpi:      $openpi_root"
echo "norm stats:  $stats"

if [ "$force" = 1 ] || [ ! -f "$stats" ]; then
  if [ "$force" = 1 ]; then echo "==> recomputing norm stats (--force-norm-stats)"
  else echo "==> norm stats missing, computing them first (walks the whole dataset)"; fi
  python "$openpi_root/scripts/compute_norm_stats.py" --config-name "$cfg"
  [ -f "$stats" ] || { echo "norm stats still missing at $stats" >&2; exit 1; }
else
  echo "==> norm stats present, skipping (use --force-norm-stats to redo)"
fi

if [ "$stats_only" = 1 ]; then
  echo "norm stats only — not training"
  exit 0
fi

# openpi's TrainConfig defaults project_name="openpi", so its runs landed in a
# different wandb project than the lerobot ones. Both stacks should report into the
# same workspace, so follow WANDB_PROJECT from .env — the same value train.sh uses —
# unless the caller named a project explicitly.
case " ${args[*]} " in
  *" --project-name"*) ;;
  *) [ -n "${WANDB_PROJECT:-}" ] && args+=("--project-name=$WANDB_PROJECT") ;;
esac
echo "wandb project: ${WANDB_PROJECT:-openpi (openpi default)}"

echo "==> training"
set -x
python "$openpi_root/scripts/train.py" "$cfg" "${args[@]}"
set +x

# openpi's train.py has no Hub upload of its own, so a finished run leaves its only
# copy on a machine you are probably about to destroy. Push the final checkpoint
# unless told not to.
[ "$push" = 1 ] || { echo "done (--no-push: checkpoint left local)"; exit 0; }
[ -n "$exp" ] || { echo "done (no --exp-name, so nothing to name a repo after)"; exit 0; }

ckpt_dir="checkpoints/$cfg/$exp"
# Sort by the numeric basename (the step), not by path text: "9000" must lose to
# "17000", and a path-field sort would key on the experiment name instead.
# Basename must be ALL digits: orbax leaves "<step>.orbax-checkpoint-tmp-NN" dirs
# behind when a write is interrupted (a full disk, last time), and those sort as the
# highest step while containing a partial checkpoint.
latest=$(ls -d "$ckpt_dir"/[0-9]* 2>/dev/null \
         | awk -F/ '$NF ~ /^[0-9]+$/ {print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$latest" ] || { echo "no checkpoint found under $ckpt_dir — nothing to push" >&2; exit 1; }

who=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null) || {
  echo "not logged in to Hugging Face; checkpoint kept at $latest" >&2; exit 1; }
repo="$who/$exp"
echo "==> pushing $latest -> hf.co/$repo  ($(du -sh "$latest" | cut -f1))"
python - "$latest" "$repo" <<'PY'
import sys
from huggingface_hub import HfApi
local, repo = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=local, repo_id=repo, repo_type="model",
                  commit_message=f"openpi checkpoint {local}")
print(f"pushed: https://huggingface.co/{repo}")
PY
