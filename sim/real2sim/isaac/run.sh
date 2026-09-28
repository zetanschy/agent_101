#!/usr/bin/env bash
# real2sim ISAAC track:  ./robot real2sim isaac <command> [args]
#
#   replay --episode N [--mode action|track|kinematic] [--num-envs K] [--render] [--gui]
#          [--placement config|grasp] [--perturb mujoco[:SIGMA] | cap_xy=..,table_z=..,cap_d=..,friction=..,mass=..]
#          [--seed S] [--physics key=value]... [--set scene.path=value]... [--tag T] [--frames a:b]
#            Isaac Sim 4.5 / Isaac Lab 2.1 (scripts/sim/sim.sh, conda 45pysaac), holding the
#            GPU lock; writes sim/outputs/real2sim/<ds>/isaac/ep<N>_<mode>[_<tag>]/
#   eval <run dir>... [--no-penetration] [--ensemble-stride N]
#            eval.json next to each run's pose log (host python3, no GPU)
#   all [replay args]
#            replay every used episode in action and track mode, then eval them all
#            (--tag T names the runs ep<N>_<mode>_T, as for replay)
#   calibrate
#            RTX light units (key alone, dome alone, emission) in both render modes and the
#            dome's azimuth convention: three short Kit runs, cached per look build
#            (outputs/<ds>/isaac/look_tex/<build>/calibration.json); render needs it
#   render --episode N (--log <run dir> | --kinematic) [--modes rt,pt] [--samples-only] [--frames a:b]
#            LOOK's assets through RTX (look.py, lookrender.py), one Kit run per mode:
#            playback of a pose log, no physics; front/grip mp4 + side_by_side.mp4 against
#            the real video, and LOOK's sample frames of that episode (render/<tag>/)
#   samples [--modes rt,pt] [lookrender args]
#            LOOK's 24 sample frames (episodes 0-2, kinematic) in each mode, then scored by
#            ./robot real2sim look score (outputs/<ds>/isaac/render/samples_kinematic/)
#   camcheck
#            render marker spheres at known points and measure the pixel error of both
#            cameras against real2sim.camera (the principal-point convention check)
#   test     the track's tests that need no Kit (Isaac env python: torch, mujoco)
#
# R2S_ISAAC_TIMEOUT (default 7200 s) bounds one Kit process: Kit can hang at exit on this
# box, and a hung process would hold the GPU lock for every other track.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../env.sh"
export PYTHONDONTWRITEBYTECODE=1
timeout_s="${R2S_ISAAC_TIMEOUT:-7200}"
# eval is numpy on one core per run: BLAS threads only spin (a 16-env eval took 221 s with
# the default threads, 75 s with one, MEASURED)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

isaac() {  # isaac <module> [args]: one Kit process under the GPU lock
  local mod="$1"; shift
  flock "$R2S_GPU_LOCK" timeout --kill-after=30 "$timeout_s" $R2S_ISAAC_PY -m "$mod" "$@"
}

sub="${1:-help}"; shift || true
case "$sub" in
  replay) isaac real2sim.isaac.replay "$@" ;;
  eval) exec $R2S_HOST_PY -m real2sim.isaac.evaluate "$@" ;;
  all)  # every used episode in action and track mode (the same replay args for each), then eval
    tag=""; prev=""
    for a in "$@"; do [ "$prev" = "--tag" ] && tag="$a"; prev="$a"; done
    eps=$($R2S_HOST_PY -c "from real2sim import scene; print(' '.join(map(str, scene.load().used_episodes())))")
    dirs=()
    for e in $eps; do
      for m in action track; do
        isaac real2sim.isaac.replay --episode "$e" --mode "$m" "$@"
        dirs+=("$R2S_OUT/isaac/ep${e}_${m}${tag:+_$tag}")
      done
    done
    exec $R2S_HOST_PY -m real2sim.isaac.evaluate "${dirs[@]}" ;;
  camcheck) isaac real2sim.isaac.camcheck "$@" ;;
  calibrate)
    for l in key dome; do isaac real2sim.isaac.lookrender --calibrate "$l" "$@"; done
    isaac real2sim.isaac.lookrender --probe-dome "$@" ;;
  render|samples)
    modes="rt"; [ "$sub" = samples ] && modes="rt,pt"
    rest=(); prev=""
    for a in "$@"; do  # --modes is this script's: one Kit process per mode (lookrender.py)
      if [ "$prev" = "--modes" ]; then modes="$a"; elif [ "$a" != "--modes" ]; then rest+=("$a"); fi; prev="$a"
    done
    if [ "$sub" = render ]; then
      for m in ${modes//,/ }; do isaac real2sim.isaac.lookrender --mode "$m" "${rest[@]}"; done
    else
      eps=$($R2S_HOST_PY -c "from real2sim import scene; print(' '.join(map(str, scene.load().used_episodes())))")
      for m in ${modes//,/ }; do
        for e in $eps; do
          isaac real2sim.isaac.lookrender --episode "$e" --kinematic --samples-only --mode "$m" \
            --tag samples_kinematic "${rest[@]}"
        done
        bash "$here/../look/run.sh" score "$R2S_OUT/isaac/render/samples_kinematic/samples_$m"
      done
    fi ;;
  # the Isaac env's python WITHOUT Kit: it has torch (the servo tests) and the rest
  test) cd "$here" && exec $R2S_ISAAC_PY -m pytest -q -p no:cacheprovider tests "$@" ;;
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" ;;
  *) echo "unknown isaac command '$sub' (replay|eval|all|calibrate|render|samples|camcheck|test)" >&2; exit 2 ;;
esac
