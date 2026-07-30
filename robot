#!/usr/bin/env bash
# Single entrypoint. Runs lerobot commands inside the docker container with
# all serial/camera/GPU/X11 passthrough already wired up in docker-compose.yml.
set -euo pipefail
cd "$(dirname "$0")"

DC="docker compose"
RUN="$DC run --rm lerobot"

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
  build)     $DC build "$@" ;;
  pull)      $DC pull "$@" ;;
  stop|kill) # force-stop any running lerobot containers (escape hatch for a wedged run)
             ids=$(docker ps -q --filter ancestor=agent101/lerobot)
             [ -n "$ids" ] && docker kill $ids && echo "stopped." || echo "nothing running." ;;
  doctor)    bash ./scripts/doctor.sh "$@" ;;   # host-side pre-flight: cameras + USB health
  shell|bash) grant_display; $RUN bash "$@" ;;
  teleop)    grant_display; $RUN ./scripts/teleop.sh "$@" ;;
  record)    grant_display; $RUN ./scripts/record.sh "$@" ;;
  infer)     grant_display; $RUN ./scripts/infer.sh "$@" ;;   # run a trained policy (sync/rtc/async)
  home)      $RUN python webui/home.py "$@" ;;               # move follower to calibrated-zero
  webui)     port="${WEBUI_PORT:-8000}"; echo "web UI -> http://localhost:${port}"
             $DC run --rm -p "${port}:8000" lerobot python webui/app.py ;;
  data)      grant_display; $RUN ./scripts/data.sh "$@" ;;   # dataset tools: viz / upload / delete / list
  calibrate) $RUN ./scripts/calibrate.sh "$@" ;;
  login)     bash ./scripts/login.sh "$@" ;;  # HF + wandb tokens -> .env.local (host-side)
  train)     train_run bash scripts/train.sh "$@" ;;    # LoRA fine-tune on the GPU
  preflight) train_run bash scripts/preflight.sh "$@" ;; # check GPU/VRAM/RAM/disk first
  openpi-eval|openpi)   # reference stack: openpi (JAX) policy on the real arm
             $DC -f docker-compose.openpi.yml run --rm openpi \
               python scripts/evaluate_openpi.py "$@" ;;
  openpi-build) $DC -f docker-compose.openpi.yml build "$@" ;;
  # openpi training (GPU only, no arm). Norm stats MUST run first: openpi does not
  # compute them during training, and without them the run trains on wrong statistics.
  # openpi's scripts live in the submodule (/opt/openpi), but we stay in /workspace so
  # its ./checkpoints and ./assets land in this repo (gitignored) instead of inside the
  # submodule checkout.
  openpi-norm-stats)
             openpi_run python /opt/openpi/scripts/compute_norm_stats.py \
               --config-name "${1:-pi05_soarm101_lora_cap_to_cup}" "${@:2}" ;;
  openpi-train)
             openpi_run bash scripts/openpi_train.sh "$@" ;;   # computes norm stats if absent
  run)       grant_display; $RUN "$@" ;;      # raw: ./robot run lerobot-train ...
  help|-h|--help|"")
    cat <<'EOF'
robot — lerobot in docker for the SO-ARM101

  ./robot build                 build the image
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
