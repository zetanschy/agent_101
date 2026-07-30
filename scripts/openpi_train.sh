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

force=0; cfg="pi05_soarm101_lora_cap_to_cup"; args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --force-norm-stats) force=1; shift ;;
    -*) args+=("$1"); shift ;;
    *)  cfg="$1"; shift ;;          # first bare word is the config name
  esac
done

# Ask openpi where the stats belong rather than reconstructing the path here.
stats=$(python - "$cfg" <<'PY'
import sys
from openpi.training import config as _c
cfg = _c.get_config(sys.argv[1])
print(cfg.assets_dirs / cfg.data.repo_id / "norm_stats.json")
PY
)
echo "config:      $cfg"
echo "norm stats:  $stats"

if [ "$force" = 1 ] || [ ! -f "$stats" ]; then
  if [ "$force" = 1 ]; then echo "==> recomputing norm stats (--force-norm-stats)"
  else echo "==> norm stats missing, computing them first (walks the whole dataset)"; fi
  python /opt/openpi/scripts/compute_norm_stats.py --config-name "$cfg"
  [ -f "$stats" ] || { echo "norm stats still missing at $stats" >&2; exit 1; }
else
  echo "==> norm stats present, skipping (use --force-norm-stats to redo)"
fi

echo "==> training"
set -x
python /opt/openpi/scripts/train.py "$cfg" "${args[@]}"
