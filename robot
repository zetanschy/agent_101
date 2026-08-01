#!/usr/bin/env bash
# Single entrypoint. Runs lerobot commands inside the docker container with
# all serial/camera/GPU/X11 passthrough already wired up in docker-compose.yml.
set -euo pipefail
cd "$(dirname "$0")"

DC="docker compose"
RUN="$DC run --rm lerobot"

# Where are we? Vast.ai instances are containers and cannot nest Docker, so the same
# command has to run natively there and through compose everywhere else. Detected
# rather than remembered; override with ROBOT_MODE=native|docker when detection is
# wrong (e.g. a box with a broken daemon you want to bypass).
if [ -n "${ROBOT_MODE:-}" ]; then
  MODE="$ROBOT_MODE"
elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  MODE=docker
else
  MODE=native
fi

# Commands that need the robot itself (serial/cameras/X11) only work through compose;
# training and checks run either way.
needs_docker() {
  [ "$MODE" = docker ] && return 0
  echo "'$1' needs Docker (serial/camera/X11 passthrough), but this box has none." >&2
  echo "On a Vast-style container use the training commands, which run natively." >&2
  exit 1
}

# Run natively when there is no Docker, else inside the given compose runner.
native_or() {           # native_or <runner-fn> <cmd...>
  local runner="$1"; shift
  if [ "$MODE" = native ]; then "$@"; else "$runner" "$@"; fi
}

# Let the container's X clients talk to the host display (best effort).
grant_display() { command -v xhost >/dev/null 2>&1 && xhost +local:root >/dev/null 2>&1 || true; }

# Run a command in the GPU-only training compose: no serial devices, cameras or
# X11, so this works on a machine with a GPU and no arms attached. Secrets come
# from .env.local via env_file; an exported value wins because `run -e` overrides it.
train_run() {
  local envargs=()
  for v in HF_TOKEN WANDB_API_KEY; do
    if [ -n "${!v:-}" ]; then envargs+=(-e "$v=${!v}"); fi
  done
  $DC -f docker-compose.train.yml run --rm "${envargs[@]}" train "$@"
}

# Same, for the openpi (JAX) image: GPU only, no serial/camera devices.
openpi_run() {
  local envargs=()
  for v in HF_TOKEN WANDB_API_KEY; do
    if [ -n "${!v:-}" ]; then envargs+=(-e "$v=${!v}"); fi
  done
  $DC -f docker-compose.openpi.yml run --rm "${envargs[@]}" openpi-train "$@"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  build|setup)  # one command for every box: build the image, or install natively
             if [ "$MODE" = native ]; then
               case "${1:-}" in
                 --openpi|openpi) bash ./scripts/setup-openpi-cloud.sh ;;
                 *)               bash ./scripts/setup-cloud.sh ;;
               esac
             else $DC build "$@"; fi ;;
  pull)      $DC pull "$@" ;;
  stop|kill) # force-stop any running lerobot containers (escape hatch for a wedged run)
             ids=$(docker ps -q --filter ancestor=agent101/lerobot)
             [ -n "$ids" ] && docker kill $ids && echo "stopped." || echo "nothing running." ;;
  doctor)    bash ./scripts/doctor.sh "$@" ;;   # host-side pre-flight: cameras + USB health
  shell|bash) needs_docker shell; grant_display; $RUN bash "$@" ;;
  teleop) needs_docker teleop;    grant_display; $RUN ./scripts/teleop.sh "$@" ;;
  record) needs_docker record;    grant_display; $RUN ./scripts/record.sh "$@" ;;
  infer) needs_docker infer;     grant_display; $RUN ./scripts/infer.sh "$@" ;;   # run a trained policy (sync/rtc/async)
  home) needs_docker home;      $RUN python webui/home.py "$@" ;;               # move follower to calibrated-zero
  webui) needs_docker webui;     port="${WEBUI_PORT:-8000}"; echo "web UI -> http://localhost:${port}"
             $DC run --rm -p "${port}:8000" lerobot python webui/app.py ;;
  data) needs_docker data;      grant_display; $RUN ./scripts/data.sh "$@" ;;   # dataset tools: viz / upload / delete / list
  calibrate) needs_docker calibrate; $RUN ./scripts/calibrate.sh "$@" ;;
  login)     bash ./scripts/login.sh "$@" ;;  # HF + wandb tokens -> .env.local (host-side)
  train)     native_or train_run bash scripts/train.sh "$@" ;;    # LoRA fine-tune on the GPU
  preflight) native_or train_run bash scripts/preflight.sh "$@" ;; # check GPU/VRAM/RAM/disk first
  openpi-eval|openpi)   # reference stack: openpi (JAX) policy on the real arm
             needs_docker openpi-eval
             $DC -f docker-compose.openpi.yml run --rm openpi \
               python scripts/evaluate_openpi.py "$@" ;;
  openpi-webui)         # same browser panel, openpi backend (separate port)
             needs_docker openpi-webui
             port="${OPENPI_WEBUI_PORT:-8001}"; echo "openpi web UI -> http://localhost:${port}"
             $DC -f docker-compose.openpi.yml run --rm -p "${port}:8000" openpi \
               python webui/app.py ;;
  openpi-build) if [ "$MODE" = native ]; then bash ./scripts/setup-openpi-cloud.sh
                else $DC -f docker-compose.openpi.yml build "$@"; fi ;;
  # openpi training (GPU only, no arm). Norm stats MUST run first: openpi does not
  # compute them during training, and without them the run trains on wrong statistics.
  # openpi's scripts live in the submodule (/opt/openpi), but we stay in /workspace so
  # its ./checkpoints and ./assets land in this repo (gitignored) instead of inside the
  # submodule checkout.
  openpi-norm-stats)
             native_or openpi_run bash scripts/openpi_train.sh --norm-stats-only "$@" ;;
  openpi-train)
             native_or openpi_run bash scripts/openpi_train.sh "$@" ;;   # computes norm stats if absent
  run)       grant_display; $RUN "$@" ;;      # raw: ./robot run lerobot-train ...
  help|-h|--help|"")
    cat <<'EOF'
robot — one entrypoint for the SO-ARM101, on any box

Docker is detected, not assumed. With Docker (this workstation, an SSH machine that
has it) everything runs in the images; without it (a Vast.ai container, which cannot
nest Docker) the training commands run natively instead — same commands either way.
Force with ROBOT_MODE=native|docker. Commands that drive the arm need Docker and say
so rather than failing with "docker: command not found".

  ./robot setup                 prepare this box: build the image, or install
                                natively when there is no Docker (add --openpi)
  ./robot build                 same thing (alias)
  ./robot calibrate follower    calibrate an arm (follower|leader)
  ./robot teleop [--cams 3]     teleoperate (2 cams default)
  ./robot record --name N --task "..." [--episodes 50] [--cams 3] [--push]
  ./robot preflight [--smoke]   check GPU/VRAM/RAM/disk before a long run
  ./robot train --dataset U/D --name RUN [--steps 20000] [--batch 16] [--push]
                                LoRA fine-tune on the GPU (wandb on if logged in)
  ./robot infer --policy R --task "..." [--rtc|--async] [--duration 60]   run a trained policy
  ./robot openpi-build                 build the openpi (JAX) reference image
  ./robot openpi-eval --policy P --task "..." [--actions 15] [--dry-run]
                                openpi checkpoint on the arm + latency report
  ./robot webui                        browser control panel: home/infer/record/params
  ./robot home                         move follower to calibrated-zero pose
  ./robot data list                    list recorded datasets
  ./robot data viz --name N [--episode 0]      visualize an episode (scrubbable)
  ./robot data upload --name N                 push a dataset to Hugging Face
  ./robot data delete --name N --episodes 3,5  remove episodes (add --new-name to keep source)
  ./robot data repair --name N [--apply]       fix "Episode length mismatch" before delete
  ./robot data merge  --name OUT --from A,B,C  concatenate datasets into one
  ./robot doctor                pre-flight: check cameras stream + USB health
  ./robot login                 store HF + wandb tokens once (./robot login --status)
  ./robot stop                  force-stop a wedged run (if ctrl-c won't quit)
  ./robot shell                 drop into a container shell
  ./robot run <lerobot-cmd ...> run any lerobot command raw

Shared config (ports, ids, cameras) lives in .env.
EOF
    ;;
  *) echo "unknown command: $cmd (try ./robot help)" >&2; exit 1 ;;
esac
