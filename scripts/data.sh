#!/usr/bin/env bash
# Dataset tools:  ./robot data <viz|upload|delete|list> ...
#
#   ./robot data list                                  # list local datasets
#   ./robot data viz    --name cap_to_cup [--episode 0]      # scrubbable timeline
#   ./robot data upload --name cap_to_cup                    # push (to soarm101/cap_to_cup)
#   ./robot data upload --name cap_to_cup --to my_clean_set  # push under a specific name
#   ./robot data upload --name cap_to_cup --to otheruser/ds --private
#   ./robot data delete --name cap_to_cup --episodes 3,5,7   # remove episodes (in place)
#   ./robot data delete --name cap_to_cup --episodes 3 --new-name cap_to_cup_clean
#   ./robot data delete --name cap_to_cup --episodes 3 --push   # edit + push result
#   ./robot data repair --name cap_to_cup [--apply]          # fix video/data length mismatch
#   ./robot data merge  --name cap_to_cup_all --from cap_to_cup,cap_to_cup2,cap_to_cup3
#
# `delete` failing with "Episode length mismatch: N vs M"? Run `repair` first —
# see scripts/repair_timestamps.py for what causes it.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

sub="${1:-list}"; shift || true

name=""; repo=""; episode=0; episodes=""; new=""; push=0; to=""; private=0; from=""; extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name)         name="$2"; shift 2 ;;
    --repo-id)      repo="$2"; shift 2 ;;
    --from)         from="$2"; shift 2 ;;      # merge: comma-separated sources
    --episode)      episode="$2"; shift 2 ;;
    --episodes)     episodes="$2"; shift 2 ;;
    --new-name)     new="${HF_USER}/$2"; shift 2 ;;
    --new-repo-id)  new="$2"; shift 2 ;;
    --to)           to="$2"; shift 2 ;;      # upload: target HF repo id (name -> HF_USER/name)
    --private)      private=1; shift ;;      # upload: make the HF repo private
    --push)         push=1; shift ;;
    *) extra+=("$1"); shift ;;
  esac
done
[ -z "$repo" ] && [ -n "$name" ] && repo="${HF_USER}/${name}"

cache="/root/.cache/huggingface/lerobot"
list_datasets() {
  find "$cache" -maxdepth 3 -type d -name meta 2>/dev/null \
    | sed "s#^$cache/##; s#/meta\$##; s/^/  /" | grep . || echo "  (none recorded yet)"
}
need_repo() {
  [ -n "$repo" ] && return
  echo "need --name <dataset> (or --repo-id)" >&2
  echo "local datasets:"; list_datasets; exit 1
}

case "$sub" in
  list|"")
    echo "local datasets:"; list_datasets ;;

  viz)
    need_repo
    exec ./viz.sh --repo-id "$repo" --episode "$episode" "${extra[@]}" ;;

  upload)
    need_repo
    # Push under YOUR Hugging Face login namespace by default (not the local
    # dataset's namespace, which you may not have write access to).
    who=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null) \
      || { echo "not logged in to Hugging Face — run: ./robot login" >&2; exit 1; }
    base="${repo##*/}"                                   # dataset name without namespace
    if [ -n "$to" ]; then
      [[ "$to" == */* ]] && target="$to" || target="${who}/${to}"   # bare name -> <you>/name
    else
      target="${who}/${base}"
    fi
    pyargs=""; [ "$private" = 1 ] && pyargs="private=True"
    echo "uploading $repo -> hf.co/datasets/$target ..."
    run env SRC="$repo" TGT="$target" python -c "import os; from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds=LeRobotDataset(os.environ['SRC']); ds.repo_id=os.environ['TGT']; ds.push_to_hub($pyargs)" ;;

  delete)
    need_repo
    [ -n "$episodes" ] || { echo "need --episodes, e.g. --episodes 3,5,7" >&2; exit 1; }
    args=(--repo_id "$repo" --operation.type delete_episodes --operation.episode_indices "[$episodes]")
    if [ -n "$new" ]; then
      args+=(--new_repo_id "$new")
      echo "deleting episodes [$episodes] from $repo -> new dataset $new (source kept)"
    else
      echo "deleting episodes [$episodes] from $repo IN PLACE (original moved aside as *_old)"
    fi
    [ "$push" = 1 ] && args+=(--push_to_hub true)
    run lerobot-edit-dataset "${args[@]}" ;;

  repair)
    need_repo
    run python ./repair_timestamps.py --repo-id "$repo" "${extra[@]}" ;;

  merge)
    [ -n "$name" ] || { echo "need --name <output dataset>" >&2; exit 1; }
    [ -n "$from" ] || { echo "need --from a,b,c   (datasets to merge, in order)" >&2; exit 1; }
    run python ./merge_datasets.py --name "$name" --from "$from" --user "${HF_USER}" "${extra[@]}" ;;

  *)
    echo "usage: ./robot data {list|viz|upload|delete|repair} --name <dataset> ..." >&2
    exit 1 ;;
esac
