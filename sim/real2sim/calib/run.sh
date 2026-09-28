#!/usr/bin/env bash
# real2sim CALIB:  ./robot real2sim calib <observe|fit|report|label|test> [args]
#
#   observe          per-frame REAL observations of every row (teal blobs in both cameras,
#                    front rim circle, white-PLA masks), then their cap labels  (~1 min)
#   fit [--quick]    hold-out fit (episodes 0+1 -> scored on 2), leave-one-episode-out
#                    folds, the final fit on 0+1+2, report + overlays, and -- if the
#                    held-out gates pass -- config/<ds>.calib.json          (~20 min)
#                    --quick: the hold-out fold alone on fewer frames, < 5 min, report
#                    only (never writes the config)
#   report [--quick] rebuild report.json, overlays and the config from saved folds
#   label            re-label the observations with the current scene (after a fit)
#   test             the calib unit tests (host python3)
# Host python3 throughout (cv2 + scipy + mujoco 3.6); MuJoCo EGL renders need no GPU lock.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../env.sh"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl

sub="${1:-help}"; shift || true
case "$sub" in
  observe) $R2S_HOST_PY -m real2sim.calib.observations all "$@" ;;
  fit)     exec $R2S_HOST_PY -m real2sim.calib.pipeline "$@" ;;
  report)  exec $R2S_HOST_PY -m real2sim.calib.report "$@" ;;
  label)   exec $R2S_HOST_PY -m real2sim.calib.observations label "$@" ;;
  test)    cd "$here" && exec $R2S_HOST_PY -m pytest -q -p no:cacheprovider tests "$@" ;;
  help|-h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown calib command '$sub' (observe|fit|report|label|test)" >&2; exit 2 ;;
esac
