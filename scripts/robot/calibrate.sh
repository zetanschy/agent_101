#!/usr/bin/env bash
# Calibrate one arm. Run once per arm before first use.
#   ./robot calibrate follower
#   ./robot calibrate leader
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

which="${1:-}"
case "$which" in
  follower) run lerobot-calibrate "${robot_args[@]}" ;;
  leader)   run lerobot-calibrate "${teleop_args[@]}" ;;
  *) echo "usage: ./robot calibrate {follower|leader}" >&2; exit 1 ;;
esac
