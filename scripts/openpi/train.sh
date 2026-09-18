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
cd "$(dirname "$0")/../.."

# JAX preallocates only 75% of VRAM by default, i.e. ~18GB of a 24GB card — under the
# 22.5GB openpi documents for LoRA fine-tuning, so it OOMs before the model even fits.
# The compose services set this; a bare box (setup-openpi-cloud.sh) has nothing that
# would, so set it here and let an explicit value from the caller win.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"

# Credentials from .env / .env.local, same loader as train.sh. Needed because
# `bash scripts/setup/login.sh` on a Docker-less box writes .env.local, and openpi's
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
# DAgger round settings. --init-from is what makes a run a round: it warm-starts from
# an existing checkpoint, inherits that checkpoint's norm stats instead of recomputing,
# and defaults the LR schedule to decay over the round's own length. --dagger adds the
# intervention-weighted sampling on top; the two are separable on purpose, because
# warm-starting on a flat dataset is a legitimate thing to want.
init_from=""; dagger=0; dagger_rebuild=0; inherit_stats=1; steps=""; dry=0
lr_warmup=""; lr_decay=""; repo_override=""
DEFAULT_LR_WARMUP_STEPS=1000
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
    --init-from)        init_from="$2"; shift 2 ;;
    --init-from=*)      init_from="${1#*=}"; shift ;;
    --no-inherit-norm-stats) inherit_stats=0; shift ;;
    --dagger)           dagger=1; shift ;;
    --dry-run)          dry=1; shift ;;   # compose everything, print it, train nothing
    --dagger-rebuild)   dagger=1; dagger_rebuild=1; shift ;;
    --human-weight)     export DAGGER_HUMAN_WEIGHT="$2"; shift 2 ;;
    --human-weight=*)   export DAGGER_HUMAN_WEIGHT="${1#*=}"; shift ;;
    --auto-weight)      export DAGGER_AUTO_WEIGHT="$2"; shift 2 ;;
    --auto-weight=*)    export DAGGER_AUTO_WEIGHT="${1#*=}"; shift ;;
    --pre-window-s)     export DAGGER_PRE_WINDOW_S="$2"; shift 2 ;;
    --pre-window-s=*)   export DAGGER_PRE_WINDOW_S="${1#*=}"; shift ;;
    --pre-min-weight)   export DAGGER_PRE_MIN_WEIGHT="$2"; shift 2 ;;
    --pre-min-weight=*) export DAGGER_PRE_MIN_WEIGHT="${1#*=}"; shift ;;
    --epoch-scale)      export DAGGER_EPOCH_SCALE="$2"; shift 2 ;;
    --epoch-scale=*)    export DAGGER_EPOCH_SCALE="${1#*=}"; shift ;;
    --steps)            steps="$2"; shift 2 ;;
    --steps=*)          steps="${1#*=}"; shift ;;
    --lr-warmup)        lr_warmup="$2"; shift 2 ;;
    --lr-warmup=*)      lr_warmup="${1#*=}"; shift ;;
    --lr-decay-steps)   lr_decay="$2"; shift 2 ;;
    --lr-decay-steps=*) lr_decay="${1#*=}"; shift ;;
    # Remembered as well as forwarded: the DAgger index and the inherited norm
    # stats are both keyed on the dataset actually being trained on, which is the
    # override when there is one and the config's own default when there is not.
    --data.repo-id)     repo_override="$2"; args+=("$1" "$2"); shift 2 ;;
    --data.repo-id=*)   repo_override="${1#*=}"; args+=("$1"); shift ;;
    *) args+=("$1"); shift ;;
  esac
done

# --steps is openpi's --num-train-steps under a shorter name, because a DAgger round
# sets it every time and the LR schedule has to be derived from it.
[ -n "$steps" ] && args+=("--num-train-steps=$steps")

if [ -n "$init_from" ]; then
  # A round of a few thousand steps against a schedule written for 30k would spend
  # the whole round at peak LR and never anneal. openpi's CosineDecaySchedule treats
  # decay_steps as the TOTAL length and optax decays over decay_steps - warmup_steps,
  # so a short round must shrink the warmup too or optax is handed a negative span
  # and dies AFTER the weights are loaded and a W&B run has opened.
  if [ -z "$lr_decay" ] && [ -n "$steps" ]; then
    lr_decay="$steps"
    if [ -z "$lr_warmup" ] && [ "$steps" -le "$DEFAULT_LR_WARMUP_STEPS" ]; then
      lr_warmup=$(( steps / 10 )); [ "$lr_warmup" -ge 1 ] || lr_warmup=1
    fi
  fi
fi

effective_warmup="${lr_warmup:-$DEFAULT_LR_WARMUP_STEPS}"
if [ -n "$lr_decay" ] && [ "$effective_warmup" -ge "$lr_decay" ]; then
  echo "--lr-warmup $effective_warmup must be below --lr-decay-steps $lr_decay:" >&2
  echo "openpi's schedule decays over the difference, and optax refuses a" >&2
  echo "non-positive span. Pass --lr-warmup explicitly, or raise --steps." >&2
  exit 1
fi
[ -n "$lr_warmup" ] && args+=("--lr-schedule.warmup-steps=$lr_warmup")
[ -n "$lr_decay" ]  && args+=("--lr-schedule.decay-steps=$lr_decay")

# openpi's scripts/ sits next to the package, but where that is depends on how it was
# installed: /opt/openpi in the images, thirdparty/openpi via setup-openpi-cloud.sh.
# Derive it from the imported module instead of hardcoding either.
openpi_root=$(python -c "import openpi, pathlib; print(pathlib.Path(openpi.__file__).parents[2])")
[ -f "$openpi_root/scripts/train.py" ] || {
  echo "openpi scripts not found under $openpi_root — is openpi installed?" >&2; exit 1; }

# Ask openpi where the stats belong rather than reconstructing the path here, and
# ask it about the dataset ACTUALLY being trained on: the stats are keyed by repo_id,
# so a --data.repo-id override that this did not see would check for, compute and
# then train against three different datasets' statistics.
read -r repo_id stats <<EOF
$(python - "$cfg" "$repo_override" <<'PY'
import sys
from openpi.training import config as _c
cfg = _c.get_config(sys.argv[1])
repo_id = sys.argv[2] or cfg.data.repo_id
print(repo_id, cfg.assets_dirs / repo_id / "norm_stats.json")
PY
)
EOF
echo "config:      $cfg"
echo "openpi:      $openpi_root"
echo "dataset:     $repo_id"
echo "norm stats:  $stats"

# WARM START. Every registered config already defaults weight_loader to a
# CheckpointWeightLoader, so tyro exposes --weight-loader.params-path and nothing
# here needs a config of its own. Accept either the checkpoint directory or its
# params/ subdirectory, because both are the obvious thing to paste.
if [ -n "$init_from" ]; then
  init_params="${init_from%/}"
  [ "$(basename "$init_params")" = "params" ] || init_params="$init_params/params"
  [ -d "$init_params" ] || {
    echo "--init-from $init_from: no params directory at $init_params" >&2; exit 1; }
  args+=("--weight-loader.params-path=$init_params")
  echo "init from:   $init_params"
fi

# NORM STATS ARE INHERITED, NOT RECOMPUTED, for a warm start. Recomputing them over
# the corrective dataset would silently change what "normalised" means underneath
# weights that were trained against the old scaling -- the inputs would shift while
# the network stayed still, which looks like a bad dataset rather than a bug.
if [ -n "$init_from" ] && [ "$inherit_stats" = 1 ] && [ "$force" != 1 ]; then
  echo "==> inheriting norm stats from the parent checkpoint"
  python - "$init_params" "$stats" <<'PY'
import shutil, sys
from pathlib import Path

params, dest = Path(sys.argv[1]), Path(sys.argv[2])
# openpi replicates assets into every checkpoint (checkpoints.py::save_assets), so
# the parent's stats sit at <step>/assets/<asset id>/norm_stats.json where the asset
# id is the PARENT's repo_id -- a different dataset from this run's. Discover it
# rather than assume it.
assets = params.parent / "assets"
found = sorted(assets.rglob("norm_stats.json"))
if not found:
    raise SystemExit(
        f"{assets} holds no norm_stats.json, so the parent checkpoint carries no "
        "statistics to inherit. Pass --no-inherit-norm-stats to compute this "
        "dataset's own instead, and know that it moves the input scaling.")
if len(found) > 1:
    raise SystemExit(
        f"{assets} holds several norm_stats.json {[str(f) for f in found]}; cannot "
        "tell which the parent trained with.")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(found[0], dest)
print(f"  {found[0]} -> {dest}")
PY
elif [ "$force" = 1 ] || [ ! -f "$stats" ]; then
  if [ "$force" = 1 ]; then echo "==> recomputing norm stats (--force-norm-stats)"
  else echo "==> norm stats missing, computing them first (walks the whole dataset)"; fi
  python "$openpi_root/scripts/compute_norm_stats.py" --config-name "$cfg" \
    ${repo_override:+--data.repo-id="$repo_override"}
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

# DAGGER SAMPLING. The weights reach the trainer through the environment rather than
# the command line because openpi owns that line: its tyro parser rejects any flag its
# own dataclasses do not define, so --human-weight there is a parse error, not an
# option. scripts/openpi/train_dagger.py reads these and patches the loader.
trainer="$openpi_root/scripts/train.py"
if [ "$dagger" = 1 ]; then
  export DAGGER_ENABLED=1 DAGGER_REPO_ID="$repo_id"
  [ "$dagger_rebuild" = 1 ] && export DAGGER_REBUILD=1
  trainer="scripts/openpi/train_dagger.py"
  echo "==> dagger sampling on ($repo_id)"
  python scripts/openpi/dagger_weights.py --dataset "$repo_id" \
    ${DAGGER_HUMAN_WEIGHT:+--human-weight="$DAGGER_HUMAN_WEIGHT"} \
    ${DAGGER_AUTO_WEIGHT:+--auto-weight="$DAGGER_AUTO_WEIGHT"} \
    ${DAGGER_PRE_WINDOW_S:+--pre-window-s="$DAGGER_PRE_WINDOW_S"} \
    ${DAGGER_PRE_MIN_WEIGHT:+--pre-min-weight="$DAGGER_PRE_MIN_WEIGHT"} \
    ${DAGGER_EPOCH_SCALE:+--epoch-scale="$DAGGER_EPOCH_SCALE"} \
    ${DAGGER_REBUILD:+--rebuild}
fi

if [ "$dry" = 1 ]; then
  echo "==> dry run, not training. The command would be:"
  printf '  %q' python "$trainer" "$cfg" "${args[@]}"; echo
  [ "$dagger" = 1 ] && echo "  with DAGGER_ENABLED=1 DAGGER_REPO_ID=$repo_id"
  exit 0
fi

echo "==> training"
set -x
python "$trainer" "$cfg" "${args[@]}"
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
