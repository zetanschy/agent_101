#!/usr/bin/env bash
# Run a trained policy on the SO-ARM101 (inference / rollout).
#
#   ./robot infer --policy zetanschy/pi05_lora_cap_tu_cup --task "Put the cap into the red cup"
#   ./robot infer --policy <repo> --task "..." --rtc          # Real-Time Chunking (best for slow VLAs)
#   ./robot infer --policy <repo> --task "..." --async         # decoupled policy-server + robot-client
#   ./robot infer ... --duration 60 --cams 2 --no-display
#
# Modes:
#   (default) sync  — lerobot-rollout, one policy call per control tick. Simple;
#                     the arm pauses each step while pi05 thinks (slow but exact).
#   --rtc           — lerobot-rollout Real-Time Chunking; predicts action chunks and
#                     keeps the arm moving between inferences. RECOMMENDED for pi05.
#   --async         — policy_server (GPU) + robot_client (control) over gRPC, decoupled.
#                     NOTE: the async client has no --rename_map, so it needs a policy
#                     whose camera names match the robot (front/grip). Our pi05 was
#                     trained with renamed cams, so async will feature-mismatch — use
#                     sync/rtc for this policy. Kept here for policies that match.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

policy="zetanschy/pi05_lora_cap_tu_cup"; task=""; duration=60; cams=2; mode=sync
display="${DISPLAY_DATA}"; recname=""; isteps=""; extra=()
rename='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.grip": "observation.images.left_wrist_0_rgb"}'
while [ $# -gt 0 ]; do
  case "$1" in
    --policy)     policy="$2"; shift 2 ;;
    --task)       task="$2"; shift 2 ;;
    --duration)   duration="$2"; shift 2 ;;
    --cams)       cams="$2"; shift 2 ;;
    --sync)       mode=sync; shift ;;
    --rtc)        mode=rtc; shift ;;
    --async)      mode=async; shift ;;
    --record)     recname="$2"; shift 2 ;;   # also record the rollout to eval_<name>
    --steps)      isteps="$2"; shift 2 ;;     # policy denoising steps (lower = faster)
    --no-display) display=false; shift ;;
    --rename-map) rename="$2"; shift 2 ;;
    --no-rename)  rename=""; shift ;;
    *) extra+=("$1"); shift ;;
  esac
done
[ -n "$task" ] || { echo "need --task \"<instruction>\"  (VLAs like pi05 require it)" >&2; exit 1; }
camjson="$(cameras_json "$cams")"
[ -n "$rename" ] && renameargs=(--rename_map="$rename") || renameargs=()
[ -n "$isteps" ] && stepargs=(--policy.num_inference_steps="$isteps") || stepargs=()
[ -n "$recname" ] && recargs=(--dataset.repo_id="eval_${recname}" --dataset.single_task="$task" --dataset.num_episodes=1) || recargs=()

case "$mode" in
  sync|rtc)
    inf=(--inference.type="$mode")
    [ "$mode" = rtc ] && inf+=(--inference.rtc.execution_horizon=10)
    run lerobot-rollout \
      --policy.path="$policy" \
      "${robot_args[@]}" \
      --robot.cameras="$camjson" \
      --task="$task" \
      --duration="$duration" \
      --policy.device=cuda \
      --policy.dtype=bfloat16 \
      --display_data="$display" \
      "${inf[@]}" \
      "${renameargs[@]}" \
      "${stepargs[@]}" \
      "${recargs[@]}" \
      "${extra[@]}"
    ;;
  async)
    echo "launching policy_server (background) + robot_client ..."
    python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps="${CAM_FPS}" &
    server_pid=$!
    trap 'kill $server_pid 2>/dev/null || true' EXIT
    sleep 5
    run python -m lerobot.async_inference.robot_client \
      "${robot_args[@]}" \
      --robot.cameras="$camjson" \
      --task="$task" \
      --server_address=127.0.0.1:8080 \
      --policy_type=pi05 \
      --pretrained_name_or_path="$policy" \
      --policy_device=cuda \
      --client_device=cpu \
      --actions_per_chunk=50 \
      "${extra[@]}"
    ;;
esac
