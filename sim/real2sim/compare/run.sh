#!/usr/bin/env bash
# real2sim INTEGRATION: ./robot real2sim compare <cmd> [args]
#
#   summary [--ds N]              every track's measured numbers, scene hash checked
#                                 -> sim/outputs/real2sim/<ds>/compare/summary.{json,md}
#   video --episode N [--ds N]    real | MuJoCo+Blender | Isaac RTX, front over wrist
#                                 -> compare/ep<N>_triptych.mp4 (needs both renders of ep N)
#
# Reads outputs only; nothing is simulated or rendered here (host python3, no GPU lock).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../env.sh"
export PYTHONDONTWRITEBYTECODE=1
quiet=(-W "ignore:A NumPy version:UserWarning")

sub="${1:-help}"; shift || true
case "$sub" in
  summary) exec $R2S_HOST_PY "${quiet[@]}" -m real2sim.compare.summary "$@" ;;
  video)   exec $R2S_HOST_PY "${quiet[@]}" -m real2sim.compare.video "$@" ;;
  help|-h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown compare command '$sub' (summary|video)" >&2; exit 2 ;;
esac
