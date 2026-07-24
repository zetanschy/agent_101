#!/usr/bin/env bash
# Record a teleoperated dataset.
#   ./robot record --name cap_pick_and_place \
#                  --task "Pick the cap and place it into the red mug" \
#                  --episodes 50
#   ./robot record --name foo --task "..." --cams 3 --push
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

cams=2
display="${DISPLAY_DATA}"
name=""
task=""
episodes=50
push="${PUSH_TO_HUB}"
extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --cams)     cams="$2"; shift 2 ;;
    --name)     name="$2"; shift 2 ;;
    --task)     task="$2"; shift 2 ;;
    --episodes) episodes="$2"; shift 2 ;;
    --push)     push=true; shift ;;
    --no-display) display=false; shift ;;
    *) extra+=("$1"); shift ;;
  esac
done

if [ -z "$name" ] || [ -z "$task" ]; then
  echo "error: --name and --task are required" >&2
  echo "e.g. ./robot record --name cap_pick_and_place --task \"Pick the cap...\" --episodes 50" >&2
  exit 1
fi

run lerobot-record \
  "${robot_args[@]}" \
  "${teleop_args[@]}" \
  --robot.cameras="$(cameras_json "$cams")" \
  --display_data="$display" \
  --dataset.repo_id="${HF_USER}/${name}" \
  --dataset.num_episodes="$episodes" \
  --dataset.single_task="$task" \
  --dataset.fps="${DATASET_FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub="$push" \
  --play_sounds=false \
  "${extra[@]}"
