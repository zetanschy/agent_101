#!/usr/bin/env bash
# Teleoperate the follower with the leader.
#   ./robot teleop            # 2 cameras (front + grip)
#   ./robot teleop --cams 3   # add the side camera
#   ./robot teleop --no-display [extra lerobot flags...]
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

cams=2
display="${DISPLAY_DATA}"
extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --cams) cams="$2"; shift 2 ;;
    --no-display) display=false; shift ;;
    *) extra+=("$1"); shift ;;
  esac
done

run lerobot-teleoperate \
  "${robot_args[@]}" \
  "${teleop_args[@]}" \
  --robot.cameras="$(cameras_json "$cams")" \
  --display_data="$display" \
  "${extra[@]}"
