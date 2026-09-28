#!/usr/bin/env bash
# real2sim LOOK, part 1 (engine-agnostic look assets): ./robot real2sim look <cmd> [args]
#
#   build [--ds N]          every look asset, from the dataset, into sim/outputs/real2sim/<ds>/look/
#                           (~2.5 min, host python3 + MuJoCo EGL, no GPU lock needed). Re-run it
#                           whenever CALIB changes the cameras; `info` says when it is stale.
#   info [--ds N]           summary of look.json, and whether the scene changed since
#   score <render_dir> [--wb] [--pattern '{id}_{cam}.png'] [--erode 2] [--out f.json]
#                           image metrics of renders (one PNG per sample id and camera) against
#                           the real sample set, per camera and region
#   test [pytest args]      the LOOK tests (host python3, ~6 s)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../env.sh"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
# host scipy 1.8 warns about numpy 1.26 on import; it works (tooling report)
quiet=(-W "ignore:A NumPy version:UserWarning")

sub="${1:-help}"; shift || true
case "$sub" in
  build) exec $R2S_HOST_PY "${quiet[@]}" -m real2sim.look.build build "$@" ;;
  info)  exec $R2S_HOST_PY "${quiet[@]}" -m real2sim.look.build info "$@" ;;
  score) exec $R2S_HOST_PY "${quiet[@]}" -m real2sim.look.imagemetrics "$@" ;;
  test)  cd "$here" && exec $R2S_HOST_PY "${quiet[@]}" -m pytest -q -p no:cacheprovider tests "$@" ;;
  help|-h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown look command '$sub' (build|info|score|test)" >&2; exit 2 ;;
esac
