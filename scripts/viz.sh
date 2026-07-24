#!/usr/bin/env bash
# Visualize a recorded dataset episode. Opens a scrubbable rerun timeline
# (pause, drag the playhead, seek to any second, loop) that stays open until
# you close it. Regenerates the recording only if not already cached.
#
#   ./robot viz --name cap_to_cup                 # episode 0
#   ./robot viz --name cap_to_cup --episode 3
#   ./robot viz --repo-id soarm/glue_pick_and_place --episode 0
#   ./robot viz                                   # list local datasets
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

name=""; episode=0; repo=""; extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name)    name="$2"; shift 2 ;;
    --episode) episode="$2"; shift 2 ;;
    --repo-id) repo="$2"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done

cache="/root/.cache/huggingface/lerobot"; rrd_dir="/workspace/rrd"
[ -z "$repo" ] && [ -n "$name" ] && repo="${HF_USER}/${name}"

# No dataset given -> list what's available locally.
if [ -z "$repo" ]; then
  echo "usage: ./robot viz --name <dataset> [--episode N]"
  echo "local datasets:"
  find "$cache" -maxdepth 3 -type d -name meta 2>/dev/null \
    | sed "s#^$cache/##; s#/meta\$##; s/^/  /" | grep . || echo "  (none recorded yet)"
  exit 1
fi

# Save the episode to a .rrd (once, cached) then open it — a static recording is
# fully scrubbable, unlike the live-streaming viewer.
mkdir -p "$rrd_dir"
rrd="$rrd_dir/${repo//\//_}_episode_${episode}.rrd"
if [ ! -f "$rrd" ]; then
  run lerobot-dataset-viz --repo-id "$repo" --episode-index "$episode" \
      --save 1 --output-dir "$rrd_dir" "${extra[@]}"
  [ -f "$rrd" ] || rrd=$(ls -t "$rrd_dir"/*_episode_"${episode}".rrd 2>/dev/null | head -1)
fi

run rerun "$rrd"
