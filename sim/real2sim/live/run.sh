#!/usr/bin/env bash
# real2sim LIVE: the simulated follower as a lerobot robot.  ./robot real2sim live <cmd> [args]
#
#   serve --engine mujoco|isaac [--layout real:N|random[:K]] [--clock realtime|lockstep]
#         [--viewer] [--autoreset S] [--seed S]
#                    run the sim on sim/outputs/real2sim/live/sim.sock until Ctrl-C
#   reset [--layout L] | status | ping      talk to a running sim from another terminal
#   test                                    this package's tests (mjlab env, MuJoCo)
#
# Usually you do not call serve yourself: `./robot teleop|record|infer ... --sim ENGINE`
# starts it, points the lerobot tools at it (ROBOT_TYPE=real2sim) and stops it after.
# MuJoCo runs in the mjlab uv env and needs no GPU lock. Isaac runs in the Isaac env and
# holds the GPU lock for the whole session, so no batch render can run out the 12 GB card
# under it.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../env.sh"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL="${MUJOCO_GL:-egl}"

sub="${1:-help}"; shift || true
case "$sub" in
  serve)
    engine=""; prev=""
    for a in "$@"; do [ "$prev" = "--engine" ] && engine="$a"; case "$a" in --engine=*) engine="${a#*=}";; esac; prev="$a"; done
    case "$engine" in
      mujoco) exec $R2S_MJLAB_PY -m real2sim.live.server "$@" ;;
      isaac)  exec flock "$R2S_GPU_LOCK" $R2S_ISAAC_PY -m real2sim.live.server "$@" ;;
      *) echo "live serve: --engine mujoco|isaac" >&2; exit 2 ;;
    esac ;;
  reset|status|ping) exec $R2S_HOST_PY -c "from real2sim.live.server import client_main; raise SystemExit(client_main())" "$sub" "$@" ;;
  test) cd "$here/../.." && exec $R2S_MJLAB_PY -m pytest -q -p no:cacheprovider real2sim/live/tests "$@" ;;
  help|-h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown live command '$sub' (serve|reset|status|ping|test)" >&2; exit 2 ;;
esac
