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
                 --openpi|openpi) bash ./scripts/openpi/setup_cloud.sh ;;
                 *)               bash ./scripts/setup/setup_cloud.sh ;;
               esac
             else $DC build "$@"; fi ;;
  pull)      $DC pull "$@" ;;
  stop|kill) # force-stop any running lerobot containers (escape hatch for a wedged run)
             ids=$(docker ps -q --filter ancestor=agent101/lerobot)
             [ -n "$ids" ] && docker kill $ids && echo "stopped." || echo "nothing running." ;;
  doctor)    bash ./scripts/setup/doctor.sh "$@" ;;   # host-side pre-flight: cameras + USB health
  shell|bash) needs_docker shell; grant_display; $RUN bash "$@" ;;
  teleop) needs_docker teleop;    grant_display; $RUN ./scripts/robot/teleop.sh "$@" ;;
  record) needs_docker record;    grant_display; $RUN ./scripts/robot/record.sh "$@" ;;
  infer) needs_docker infer;     grant_display; $RUN ./scripts/robot/infer.sh "$@" ;;   # run a trained policy (sync/rtc/async)
  home) needs_docker home;      $RUN python webui/home.py "$@" ;;               # move follower to calibrated-zero
  webui) needs_docker webui;     port="${WEBUI_PORT:-8000}"; echo "web UI -> http://localhost:${port}"
             $DC run --rm -p "${port}:8000" lerobot python webui/app.py ;;
  data) needs_docker data;      grant_display; $RUN ./scripts/robot/data.sh "$@" ;;   # dataset tools: viz / upload / delete / list
  calibrate) needs_docker calibrate; $RUN ./scripts/robot/calibrate.sh "$@" ;;
  login)     bash ./scripts/setup/login.sh "$@" ;;  # HF + wandb tokens -> .env.local (host-side)
  train)     native_or train_run bash scripts/robot/train.sh "$@" ;;    # LoRA fine-tune on the GPU
  preflight) native_or train_run bash scripts/setup/preflight.sh "$@" ;; # check GPU/VRAM/RAM/disk first
  openpi-eval|openpi)   # reference stack: openpi (JAX) policy on the real arm
             needs_docker openpi-eval
             $DC -f docker-compose.openpi.yml run --rm openpi \
               python scripts/openpi/evaluate.py "$@" ;;
  openpi-webui)         # same browser panel, openpi backend (separate port)
             needs_docker openpi-webui
             port="${OPENPI_WEBUI_PORT:-8001}"; echo "openpi web UI -> http://localhost:${port}"
             $DC -f docker-compose.openpi.yml run --rm -p "${port}:8000" openpi \
               python webui/app.py ;;
  openpi-build) if [ "$MODE" = native ]; then bash ./scripts/openpi/setup_cloud.sh
                else $DC -f docker-compose.openpi.yml build "$@"; fi ;;
  rtc-parity)           # check openpi's RTC port against lerobot's, in both images
             needs_docker rtc-parity
             ref="${1:-/workspace/outputs/rtc_ref.json}"
             $DC run --rm --entrypoint python lerobot scripts/openpi/rtc_parity.py --dump "$ref" \
               && $DC -f docker-compose.openpi.yml run --rm openpi-train \
                    python scripts/openpi/rtc_parity.py --check "$ref" ;;
  # --- sim2real (Isaac Sim) -------------------------------------------------
  # These do NOT go through Docker: Isaac Sim is a native conda install that needs
  # the GPU plus a large shader/asset cache in $HOME. scripts/sim/sim.sh picks the env.
  sim-assets)           # convert the printed CAD (STL) into simulatable USD
             bash ./scripts/sim/sim.sh scripts/sim/convert_assets.py "$@" ;;
  sim-play)             # build the push-T scene, step it, render both cameras
             bash ./scripts/sim/sim.sh scripts/sim/play.py "$@" ;;
  sim-train)            # train a policy with rsl_rl PPO (reach, by default)
             # HEADLESS unless you ask for --gui: rendering 4096 arms is most of
             # the cost of training them. Checkpoints and tensorboard logs land in
             # sim/outputs/rsl_rl/<experiment>/<timestamp>/, and wandb is on when
             # there are credentials for it -- the same rule ./robot train follows.
             bash ./scripts/sim/sim.sh scripts/sim/train.py "$@" ;;
  sim-policy)           # run a trained checkpoint and report its tracking error
             # Windowed by default, the opposite of sim-train: this one is for
             # watching. Defaults to the newest so101_reach checkpoint.
             #   --goal task    (default) the env samples a new target every 4 s
             #   --goal manual  one arm, and you drag /World/GoalHandle or use
             #                  W/S A/D Q/E to move the target yourself
             #   --goal auto    one arm, target walking a circle; works headless
             #   --real         put the REAL arm in the loop: a ghost arm in the
             #                  viewport shows where it actually is. Starts the
             #                  bus-side bridge in the container and stops it after.
             #   --engage       with --real, actually command the arm. Without it
             #                  the bridge is read-only and NOTHING MOVES.
             _real=0; _engage=0
             for _a in "$@"; do
               [ "$_a" = "--real" ] && _real=1
               [ "$_a" = "--engage" ] && _engage=1
             done
             if [ "$_real" = 1 ]; then
               # Same shape as sim-teleop: Isaac must run natively for the GPU, the
               # serial bus must run in Docker, so start the bridge here and clean it
               # up on the way out however this exits.
               needs_docker "sim-policy --real"
               _fresh() { python3 -c "
import json,pathlib,sys,time
f=pathlib.Path('sim/outputs/policy/state.json')
sys.exit(0 if f.exists() and time.time()-json.loads(f.read_text())['t'] < 3 else 1)
" 2>/dev/null; }
               echo "starting the policy bridge in the background ..."
               [ "$_engage" = 1 ] \
                 && echo "  --engage: THE ARM WILL MOVE. Keep a hand near the power." \
                 || echo "  read-only: the bridge will not command anything."
               _bridge=$($DC run -d lerobot python scripts/robot/policy_bridge.py \
                          $([ "$_engage" = 1 ] && echo --engage)) || exit 1
               trap '[ -n "$_bridge" ] && { docker stop -t 2 "$_bridge" >/dev/null 2>&1; docker rm -f "$_bridge" >/dev/null 2>&1; }' EXIT INT TERM
               for _ in $(seq 1 60); do _fresh && break; sleep 1; done
               if ! _fresh; then
                 echo "the bridge never published arm state. Its output:" >&2
                 docker logs "$_bridge" 2>&1 | tail -20 >&2
                 exit 1
               fi
             fi
             bash ./scripts/sim/sim.sh scripts/sim/policy.py "$@" ;;
  policy-bridge)        # the bus side of sim-policy --real, on its own
             # sim-policy --real starts this for you. Run it yourself when you want
             # it in its own terminal, or to watch the arm's state without Isaac.
             needs_docker policy-bridge
             $RUN python scripts/robot/policy_bridge.py "$@" ;;
  leader-publish)       # stream the leader's joint angles to sim/outputs/calib/joints.json
             # The serial buses live in Docker, so anything reading an arm runs here.
             # Two consumers: calib-capture (which cannot open the follower bus
             # itself while this holds it) and sim-teleop.
             #   --no-follower  publish the LEADER only, driving nothing. Use this
             #                  for sim-teleop, where the simulator IS the follower
             #                  and a real arm lurching to meet the leader is just
             #                  a hazard.
             needs_docker leader-publish
             $RUN python scripts/sim/calibrate_extrinsics.py teleop "$@" ;;
  sim-record)           # teleoperate the sim and record a LeRobot dataset
             # Same one-command shape as sim-teleop: the joint publisher is started
             # in the container, the sim runs natively, and both are cleaned up.
             # Randomizes the T, the goal and the whole scene between episodes.
             _fresh() { python3 -c "
import json,pathlib,sys,time
f=pathlib.Path('sim/outputs/calib/joints.json')
sys.exit(0 if f.exists() and time.time()-json.loads(f.read_text())['t'] < 3 else 1)
" 2>/dev/null; }
             _pub=""
             if _fresh; then
               echo "using the joint stream that is already running"
             else
               needs_docker sim-record
               echo "starting the leader publisher in the background ..."
               _pub=$($DC run -d lerobot \
                        python scripts/sim/calibrate_extrinsics.py teleop --no-follower) || exit 1
               trap '[ -n "$_pub" ] && { docker stop -t 2 "$_pub" >/dev/null 2>&1; docker rm -f "$_pub" >/dev/null 2>&1; }' EXIT INT TERM
               for _ in $(seq 1 60); do _fresh && break; sleep 1; done
               if ! _fresh; then
                 echo "the publisher never produced fresh angles. Its output:" >&2
                 docker logs "$_pub" 2>&1 | tail -20 >&2
                 exit 1
               fi
             fi
             bash ./scripts/sim/sim.sh scripts/sim/record.py "$@" ;;
  sim-teleop)           # drive the Isaac scene from the real leader arm
             # ONE command. Isaac must run natively (GPU, shader cache) and the
             # serial bus must run in Docker, so this starts the publisher in the
             # background, runs the sim against it, and stops it again. Nothing to
             # coordinate across two terminals.
             #
             # If something is already publishing fresh angles -- you started
             # leader-publish yourself, or calib-teleop is up for a calibration --
             # that is left alone and simply consumed.
             _fresh() { python3 -c "
import json,pathlib,sys,time
f=pathlib.Path('sim/outputs/calib/joints.json')
sys.exit(0 if f.exists() and time.time()-json.loads(f.read_text())['t'] < 3 else 1)
" 2>/dev/null; }
             _pub=""
             if _fresh; then
               echo "using the joint stream that is already running"
             else
               needs_docker sim-teleop
               echo "starting the leader publisher in the background ..."
               # -d without --rm on purpose: --rm deletes the container the moment
               # it exits, so when the publisher dies at startup `docker logs` finds
               # nothing and the failure is unreportable. Remove it in the trap.
               _pub=$($DC run -d lerobot \
                        python scripts/sim/calibrate_extrinsics.py teleop --no-follower) || exit 1
               trap '[ -n "$_pub" ] && { docker stop -t 2 "$_pub" >/dev/null 2>&1; docker rm -f "$_pub" >/dev/null 2>&1; }' EXIT INT TERM
               for _ in $(seq 1 60); do _fresh && break; sleep 1; done
               if ! _fresh; then
                 echo "the publisher never produced fresh angles. Its output:" >&2
                 docker logs "$_pub" 2>&1 | tail -20 >&2
                 exit 1
               fi
             fi
             bash ./scripts/sim/sim.sh scripts/sim/teleop.py "$@" ;;
  sim-shell)            # a python REPL inside the Isaac Sim environment
             bash ./scripts/sim/sim.sh "$@" ;;
  sim-camera-check)     # check the sim's camera assumptions against the real ones
             python3 ./scripts/sim/camera_check.py "$@" ;;
  sim-calibrate)        # measure real intrinsics from a checkerboard
             python3 ./scripts/sim/calibrate_cameras.py "$@" ;;
  sim-compare-cameras)  # sim render next to a live capture, for tuning extrinsics
             python3 ./scripts/sim/compare_cameras.py "$@" ;;
  print-targets)        # every printable: checkerboard, charuco, push-T goal
             python3 ./scripts/sim/print_targets.py "$@" ;;
  calib-board)          # just the ChArUco board (print-targets makes all three)
             python3 ./scripts/sim/calibrate_extrinsics.py board "$@" ;;
  calib-capture)        # record arm poses looking at the board
             # On the HOST: the live preview needs a GUI OpenCV and the container
             # ships the headless build. It shells into the container for each joint
             # read, which is the only part that needs lerobot and the serial bus.
             needs_docker calib-capture
             python3 ./scripts/sim/calibrate_extrinsics.py capture "$@" ;;
  calib-teleop)         # drive the follower from the leader, publishing its joints
             # The calibration flow's name for leader-publish: same publisher, but
             # this one DOES drive the follower, because calibration needs the real
             # arm to move to each pose. Leave it running in one terminal while
             # calib-capture runs in another -- the follower bus can only be opened
             # once, so capture reads the joints this publishes rather than the arm.
             needs_docker calib-teleop
             $RUN python scripts/sim/calibrate_extrinsics.py teleop "$@" ;;
  calib-solve)          # solve both cameras' pose in the robot base frame
             python3 ./scripts/sim/calibrate_extrinsics.py solve "$@" ;;
  # openpi training (GPU only, no arm). Norm stats MUST run first: openpi does not
  # compute them during training, and without them the run trains on wrong statistics.
  # openpi's scripts live in the submodule (/opt/openpi), but we stay in /workspace so
  # its ./checkpoints and ./assets land in this repo (gitignored) instead of inside the
  # submodule checkout.
  openpi-norm-stats)
             native_or openpi_run bash scripts/openpi/train.sh --norm-stats-only "$@" ;;
  openpi-train)
             native_or openpi_run bash scripts/openpi/train.sh "$@" ;;   # computes norm stats if absent
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
                                LoRA fine-tune, LEROBOT stack (pytorch)
  ./robot infer --policy R --task "..." [--rtc|--async] [--duration 60]   run a trained policy
  ./robot openpi-train --exp-name=RUN [--overwrite|--resume]
                                LoRA fine-tune, OPENPI stack (jax). An ALTERNATIVE to
                                `train`, not a follow-up: pick one. Dataset comes from
                                the config, and norm stats are computed automatically.
  ./robot openpi-build                 build the openpi (JAX) reference image
  ./robot openpi-eval --policy P --task "..." [--actions 15] [--dry-run] [--rtc]
                                openpi checkpoint on the arm + latency report
  ./robot rtc-parity            check openpi's real-time-chunking port against
                                lerobot's reference implementation (no arm needed)
  ./robot sim-assets            convert the printed T + camera mount CAD to USD
  ./robot sim-play [--gui]      build the push-T scene in Isaac Sim and check it
  ./robot sim-train             train reach with rsl_rl PPO (headless, 4096 envs)
                                --task Agent101-So101-Reach-DR for randomized gains
                                wandb on when credentials exist, as ./robot train;
                                --no-wandb / --wandb-offline / --name to override
                                tensorboard --logdir sim/outputs/rsl_rl
  ./robot sim-policy            watch the newest checkpoint and score its tracking
  ./robot sim-policy --goal manual
                                one arm, and you drag the goal around while the
                                policy chases it (--goal auto to run it hands-free)
  ./robot sim-policy --real --goal manual
                                the same, with the REAL arm in the loop: a ghost arm
                                shows where it is. Read-only until you add --engage
  ./robot policy-bridge         just the bus side, if you want it in its own terminal
  ./robot sim-teleop            drive the Isaac scene from the real leader arm
                                (starts and stops the joint publisher for you)
  ./robot leader-publish --no-follower
                                just the publisher, if you want it in its own
                                terminal
  ./robot sim-play --gui --pose physics off, so prims can be dragged; prints the
                                camera-mount transform on exit
  ./robot sim-camera-check      verify the sim camera model against the real cameras
  ./robot sim-calibrate --camera front|grip   measure real intrinsics (checkerboard)
  ./robot sim-compare-cameras   sim render vs live capture, to tune camera placement
  ./robot print-targets         PDFs to print: checkerboard, charuco, push-T goal
  ./robot calib-board           just the charuco
  ./robot calib-teleop          drive the arm from the leader while calibrating
  ./robot calib-capture         record poses (run calib-teleop in another terminal)
  ./robot calib-solve           solve where both cameras are, in the base frame
                                to measure where both cameras really are (extrinsics)
  ./robot webui                        browser control panel: home/infer/record/params
                                       loads lerobot, openpi and mjlab RL (.onnx) policies
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
