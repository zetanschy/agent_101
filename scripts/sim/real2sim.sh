#!/usr/bin/env bash
# ./robot real2sim <track> [args]  ->  sim/real2sim/<track>/run.sh [args]
#
# One dispatcher, and each track owns its run.sh: calib, mujoco, isaac, look,
# blender, compare. No track, `core`, or a core subcommand goes to the core's
# sim/real2sim/run.sh (extract | info | meshes | test). The R2S_* environment
# (interpreters, dataset, GPU lock) is exported before the track runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="$ROOT/sim/real2sim"
. "$PKG/env.sh"

track="${1:-core}"; [ $# -gt 0 ] && shift
case "$track" in
  core) exec bash "$PKG/run.sh" "$@" ;;
  extract|info|meshes|test|help|-h|--help) exec bash "$PKG/run.sh" "$track" "$@" ;;
esac
if [ -f "$PKG/$track/run.sh" ]; then
  exec bash "$PKG/$track/run.sh" "$@"
fi
have=""
for d in "$PKG"/*/; do [ -f "$d/run.sh" ] && have="$have $(basename "$d")"; done
echo "real2sim: no track '$track' (sim/real2sim/$track/run.sh does not exist)." >&2
echo "tracks with a run.sh:${have:- none yet}; core commands: extract info meshes test" >&2
exit 2
