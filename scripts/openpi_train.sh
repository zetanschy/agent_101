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

force=0; cfg="pi05_soarm101_lora_cap_to_cup"; args=()
# The config name is only ever the FIRST argument, and only if it is not a flag.
# Everything after it is forwarded verbatim, so flags that take a separate value
# (`--num-workers 2`) keep their value instead of it being mistaken for the config.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then cfg="$1"; shift; fi
while [ $# -gt 0 ]; do
  case "$1" in
    --force-norm-stats) force=1; shift ;;
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

echo "==> training"
set -x
python "$openpi_root/scripts/train.py" "$cfg" "${args[@]}"
