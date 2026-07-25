#!/usr/bin/env bash
# Single entrypoint. Runs lerobot commands inside the docker container with
# all serial/camera/GPU/X11 passthrough already wired up in docker-compose.yml.
set -euo pipefail
cd "$(dirname "$0")"

DC="docker compose"
RUN="$DC run --rm lerobot"

# Let the container's X clients talk to the host display (best effort).
grant_display() { command -v xhost >/dev/null 2>&1 && xhost +local:root >/dev/null 2>&1 || true; }

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
  train)     # LoRA fine-tune on the GPU compose (no robot devices / cameras / X11).
             # Secrets come from .env.local via env_file; an exported value wins
             # because `run -e` overrides env_file.
             envargs=()
             for v in HF_TOKEN WANDB_API_KEY; do
               if [ -n "${!v:-}" ]; then envargs+=(-e "$v=${!v}"); fi
             done
             $DC -f docker-compose.train.yml run --rm "${envargs[@]}" \
               train bash scripts/train.sh "$@" ;;
  run)       grant_display; $RUN "$@" ;;      # raw: ./robot run lerobot-train ...
  help|-h|--help|"")
    cat <<'EOF'
robot — lerobot in docker for the SO-ARM101

  ./robot build                 build the image
  ./robot calibrate follower    calibrate an arm (follower|leader)
  ./robot teleop [--cams 3]     teleoperate (2 cams default)
  ./robot record --name N --task "..." [--episodes 50] [--cams 3] [--push]
  ./robot train --dataset U/D --name RUN [--steps 20000] [--batch 16] [--push]
                                LoRA fine-tune on the GPU (wandb on if logged in)
  ./robot infer --policy R --task "..." [--rtc|--async] [--duration 60]   run a trained policy
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
