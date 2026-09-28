#!/usr/bin/env bash
# real2sim LOOK part 2 (Blender renderer): ./robot real2sim blender <cmd> [args]
#
#   render --episode N (--kinematic | --poselog PATH|mujoco) [--engine eevee|cycles] [--samples-only]
#          [--frames A:B[:S]] [--cams front,grip] [--png] [--keep-pinhole] [--no-video]
#          [--exposure constant|look] [--wb frame|episode|reference] [--quality KEY=VALUE]
#                         both cameras, real distortion, mp4s + side_by_side.mp4 + sample PNGs
#                         + manifest.json (+ score.json) in sim/outputs/real2sim/<ds>/blender/<tag>/
#   samples [--engine eevee|cycles] [--poselog mujoco|PATTERN]
#                         the 24 LOOK sample frames (episodes 0-2) + metrics; kinematic, or
#                         physics ('mujoco' = the nominal action replays, or a path with {ep})
#   score DIR [--wb]      LOOK's image metrics of DIR/samples/<id>_<cam>.png
#   check [--episode N] [--engine ..]  silhouettes vs LOOK's MuJoCo label renderer (geometry only)
#   test [pytest args]    the blender tests (host python3, no GPU, ~3 s)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../env.sh"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
quiet=(-W "ignore:A NumPy version:UserWarning")

sub="${1:-help}"; shift || true
case "$sub" in
  render|samples|score) exec $R2S_HOST_PY "${quiet[@]}" -m real2sim.blender.cli "$sub" "$@" ;;
  check) exec $R2S_HOST_PY "${quiet[@]}" -m real2sim.blender.check "$@" ;;
  # run from sim/: inside sim/real2sim, the track package real2sim/mujoco would shadow the mujoco library
  test)  cd "$R2S_SIM" && exec $R2S_HOST_PY "${quiet[@]}" -m pytest -q -p no:cacheprovider "$here/tests" "$@" ;;
  help|-h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown blender command '$sub' (render|samples|score|check|test)" >&2; exit 2 ;;
esac
