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
  data)      grant_display; $RUN ./scripts/data.sh "$@" ;;   # dataset tools: viz / upload / delete / list
  calibrate) $RUN ./scripts/calibrate.sh "$@" ;;
  login)     $RUN hf auth login "$@" ;;      # cache HF token in the volume
  run)       grant_display; $RUN "$@" ;;      # raw: ./robot run lerobot-train ...
  help|-h|--help|"")
    cat <<'EOF'
robot — lerobot in docker for the SO-ARM101

  ./robot build                 build the image
  ./robot calibrate follower    calibrate an arm (follower|leader)
  ./robot teleop [--cams 3]     teleoperate (2 cams default)
  ./robot record --name N --task "..." [--episodes 50] [--cams 3] [--push]
  ./robot data list                    list recorded datasets
  ./robot data viz --name N [--episode 0]      visualize an episode (scrubbable)
  ./robot data upload --name N                 push a dataset to Hugging Face
  ./robot data delete --name N --episodes 3,5  remove episodes (add --new-name to keep source)
  ./robot doctor                pre-flight: check cameras stream + USB health
  ./robot login                 log in to Hugging Face (once)
  ./robot stop                  force-stop a wedged run (if ctrl-c won't quit)
  ./robot shell                 drop into a container shell
  ./robot run <lerobot-cmd ...> run any lerobot command raw

Shared config (ports, ids, cameras) lives in .env.
EOF
    ;;
  *) echo "unknown command: $cmd (try ./robot help)" >&2; exit 1 ;;
esac
