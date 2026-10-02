#!/usr/bin/env bash
# Runs ON a rented Vast.ai box, in tmux (the vast-train skill copies it to /root):
# setup, an openpi pi0.5 LoRA fine-tune, the push to the Hub, a check that the push
# landed, then the box tears itself down. So a finished run stops billing whether or
# not anyone is watching it.
#
#   bash /root/remote_run.sh --exp-name E --dataset zetanschy/ds [--steps N]
#        [--teardown destroy|stop|keep]
#   bash /root/remote_run.sh --finish-only --exp-name E [--teardown ...]
#        # a run already going: wait for its EXIT line in the log, then verify + tear down
#
# Teardown, after the run:
#   pushed and verified on the Hub -> --teardown (destroy by default: billing ends)
#   anything else (failed, not pushed) -> stop, never destroy: GPU billing ends, the
#     disk and its checkpoint stay, so it can be inspected or resumed (`--resume`)
# The box manages itself with the per-instance key Vast puts in PID 1's environment
# (CONTAINER_ID / CONTAINER_API_KEY; an ssh session does not inherit them). That key
# cannot read the account's credit, so a credit guard has to live on the caller's side.
set -o pipefail
cd /root/agent_101  # the clone (this file itself is copied to /root)

exp=""; dataset=""; steps=""; teardown=destroy; finish_only=0
log=/root/run.log
while [ $# -gt 0 ]; do
  case "$1" in
    --exp-name)    exp="$2"; shift 2 ;;
    --dataset)     dataset="$2"; shift 2 ;;
    --steps)       steps="$2"; shift 2 ;;
    --teardown)    teardown="$2"; shift 2 ;;
    --finish-only) finish_only=1; shift ;;
    --log)         log="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$exp" ] || { echo "need --exp-name" >&2; exit 2; }
case "$teardown" in destroy|stop|keep) ;; *) echo "--teardown destroy|stop|keep" >&2; exit 2 ;; esac

say() { echo "== $(date -u +%FT%TZ) $*"; }
pid1() { tr '\0' '\n' < /proc/1/environ | grep "^$1=" | cut -d= -f2-; }

vast() { # METHOD [json body]: this instance, through Vast's REST API
  local id key; id=$(pid1 CONTAINER_ID); key=$(pid1 CONTAINER_API_KEY)
  [ -n "$id" ] && [ -n "$key" ] || { say "no CONTAINER_ID/CONTAINER_API_KEY: cannot tear down"; return 1; }
  curl -sS -X "$1" -H "Authorization: Bearer $key" -H "Content-Type: application/json" \
       ${2:+-d "$2"} "https://console.vast.ai/api/v0/instances/$id/"
  echo
}

pushed() { # the final checkpoint's params/ are on the Hub under <hf user>/<exp>
  python3 - "$exp" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
repo = f"{api.whoami()['name']}/{sys.argv[1]}"
files = api.list_repo_files(repo)
n = sum(f.startswith("params/") for f in files)
print(f"  hf.co/{repo}: {len(files)} files, {n} under params/")
sys.exit(0 if n else 1)
PY
}

finish() { # rc of the run
  local rc=$1
  [ -f .venv-openpi/bin/activate ] && . .venv-openpi/bin/activate
  # train.sh exports HF_TOKEN only to itself; the check needs it too
  [ -n "${HF_TOKEN:-}" ] || export "$(grep -m1 '^HF_TOKEN=' .env.local 2>/dev/null || echo HF_TOKEN=)"
  if [ "$rc" = 0 ] && pushed; then
    say "pushed and verified; teardown: $teardown"
    case "$teardown" in
      destroy) vast DELETE ;;
      stop)    vast PUT '{"state": "stopped"}' ;;
    esac
  else
    say "run exit $rc or the push is not on the Hub: stopping (disk and checkpoint kept)"
    [ "$teardown" = keep ] || vast PUT '{"state": "stopped"}'
  fi
}

push_checkpoints() { # while training: each saved checkpoint -> the Hub, so a dead host loses <= save_interval
  local done_step=-1 dir step
  while sleep 120; do
    # orbax writes <step>.orbax-checkpoint-tmp-* and renames it to <step> when complete
    dir=$(ls -d checkpoints/*/"$exp"/[0-9]* 2>/dev/null | awk -F/ '$NF ~ /^[0-9]+$/ {print $NF, $0}' \
          | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$dir" ] || continue
    step=$(basename "$dir")
    [ "$step" -gt "$done_step" ] || continue
    say "pushing checkpoint $step to the Hub (in-progress copy)"
    if python3 - "$dir" "$exp" "$step" <<'PY'
import sys
from huggingface_hub import HfApi
local, exp, step = sys.argv[1:4]
api = HfApi()
repo = f"{api.whoami()['name']}/{exp}"
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=local, repo_id=repo, repo_type="model",
                  commit_message=f"checkpoint step {step} (training in progress)")
api.upload_file(path_or_fileobj=step.encode(), path_in_repo="agent101_step.txt", repo_id=repo,
                repo_type="model", commit_message=f"step {step}")
print(f"  pushed step {step} -> hf.co/{repo}")
PY
    then done_step=$step; fi
  done
}

if [ "$finish_only" = 1 ]; then
  say "waiting for the run's EXIT line in $log"
  until rc=$(grep -aoE "EXIT [0-9]+" "$log" 2>/dev/null | tail -1 | cut -d' ' -f2) && [ -n "$rc" ]; do sleep 60; done
  finish "$rc" 2>&1 | tee -a "$log"
  exit 0
fi

[ -n "$dataset" ] || { echo "need --dataset" >&2; exit 2; }
{
  say "setup"
  bash scripts/openpi/setup_cloud.sh
  rc=$?
  if [ "$rc" = 0 ]; then
    [ -f .venv-openpi/bin/activate ] && . .venv-openpi/bin/activate
    # A shorter run must anneal over its own length, not the config's 30k.
    extra=()
    [ -n "$steps" ] && extra=(--steps "$steps" --lr-decay-steps "$steps")
    say "train $exp on $dataset ${steps:+($steps steps) }with $(python --version)"
    [ -n "${HF_TOKEN:-}" ] || export "$(grep -m1 '^HF_TOKEN=' .env.local 2>/dev/null || echo HF_TOKEN=)"
    push_checkpoints & pusher=$!
    bash scripts/openpi/train.sh --exp-name="$exp" --data.repo-id="$dataset" --overwrite "${extra[@]}"
    rc=$?
    kill "$pusher" 2>/dev/null
  fi
  say "EXIT $rc"
  finish "$rc"
} 2>&1 | tee -a "$log"
