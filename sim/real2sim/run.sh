#!/usr/bin/env bash
# real2sim CORE commands:  ./robot real2sim core <extract|info|meshes|test> [args]
#
#   extract [--ds N] [--force]  dataset -> episodes.npz, readable video copies and
#                               decoded frame caches (needs Docker: the videos are
#                               root-owned). ~11 s; --verify only checks the outputs.
#   info [N]                    the merged scene with provenance, joint ranges, hash
#   meshes [--force]            (re)build the cap/mug OBJ + STL from the scene dims
#   test [pytest args]          the core test suite (host python3; ~1 min)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
export PYTHONDONTWRITEBYTECODE=1

sub="${1:-help}"; shift || true
case "$sub" in
  extract) exec $R2S_HOST_PY -m real2sim.extract "$@" ;;
  info)    exec $R2S_HOST_PY -m real2sim.scene "$@" ;;
  meshes)  exec $R2S_HOST_PY -m real2sim.objects "$@" ;;
  test)    cd "$here" && exec $R2S_HOST_PY -m pytest -q -p no:cacheprovider tests "$@" ;;
  help|-h|--help)
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown core command '$sub' (extract|info|meshes|test)" >&2; exit 2 ;;
esac
