#!/usr/bin/env bash
# Shared building blocks for the lerobot wrapper scripts.
# Runs INSIDE the container; env vars come from .env (docker-compose env_file).
set -euo pipefail

# --- reusable flag groups -------------------------------------------------
robot_args=(
  --robot.type="${ROBOT_TYPE}"
  --robot.port="${ROBOT_PORT}"
  --robot.id="${ROBOT_ID}"
)

teleop_args=(
  --teleop.type="${TELEOP_TYPE}"
  --teleop.port="${TELEOP_PORT}"
  --teleop.id="${TELEOP_ID}"
)

# One camera entry in lerobot's inline-dict syntax.
cam_fragment() { # name index
  echo "$1: {type: opencv, index_or_path: $2, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: \"${CAM_FOURCC}\"}"
}

# Build the --robot.cameras value. Arg = number of cameras (2 or 3).
cameras_json() {
  local n="${1:-2}"
  local frags=()
  frags+=("$(cam_fragment front "${CAM_FRONT_INDEX}")")
  frags+=("$(cam_fragment grip  "${CAM_GRIP_INDEX}")")
  if [ "$n" -ge 3 ]; then
    frags+=("$(cam_fragment side "${CAM_SIDE_INDEX}")")
  fi
  local IFS=,
  echo "{ ${frags[*]}}"
}

# Print a command before running it (so you can copy/inspect it).
run() {
  printf '\n\033[1;36m$ %s\033[0m\n\n' "$*"
  "$@"
}
