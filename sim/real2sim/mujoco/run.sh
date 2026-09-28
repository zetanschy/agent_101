#!/usr/bin/env bash
# real2sim MUJOCO track:  ./robot real2sim mujoco <command> [args]
#
#   fingers [--force]        CoACD of the two finger parts (astribot_simu env, ~60 s, cached)
#   servo-fit [--iters N]    identify the servo model -> config/<ds>.servo.json (~10 min, 8 procs)
#   replay --episode N [--mode action|track|kinematic] [--seed S] [--ensemble K] [--render] [--gui]
#   eval [--episodes 0,1,2] [--modes action,track] [--ensemble 20] [--render]
#                            -> mujoco/eval[_T].json; what-if options (--set, --placement, --cap-shape ...) need --tag T
#   render --log L.npz       mp4 (real | sim, front and wrist) from a pose log
#   view --log L.npz         play a pose log in mujoco.viewer (needs DISPLAY)
#   info                     servo model, physics, scene feasibility
#   test [pytest args]       this track's tests (mjlab env)
#
# Everything runs in the mjlab uv env (mujoco 3.11) except `fingers` (CoACD). MuJoCo EGL
# rendering does not take the GPU lock (it is light); nothing here uses Isaac or Blender.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../env.sh"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL="${MUJOCO_GL:-egl}"

sub="${1:-help}"; shift || true
case "$sub" in
  fingers) exec $R2S_COACD_PY -m real2sim.mujoco.fingers "$@" ;;
  servo-fit|replay|eval|render|view|info) exec $R2S_MJLAB_PY -m real2sim.mujoco.cli "$sub" "$@" ;;
  test)    cd "$here" && exec $R2S_MJLAB_PY -m pytest -q -p no:cacheprovider tests "$@" ;;
  help|-h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown mujoco command '$sub' (fingers|servo-fit|replay|eval|render|view|info|test)" >&2; exit 2 ;;
esac
