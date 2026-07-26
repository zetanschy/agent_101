#!/usr/bin/env bash
# One login for every account a training run touches: Hugging Face (pull datasets,
# push checkpoints) and Weights & Biases (loss curves). Both tokens land in ONE
# gitignored file, .env.local, which every compose service and scripts/train.sh
# read — so they survive `--rm` container runs (a container's ~/.netrc does not)
# and move to a cloud GPU box by copying that single file.
#
#   ./robot login              # fill in whatever's missing, then verify
#   ./robot login --status     # what's configured (never prints the secrets)
#   ./robot login --hf         # just one of them (--wandb likewise)
#   ./robot login --force      # replace tokens that are already stored
#   ./robot login --no-verify  # skip the network round-trip
#
# Get tokens: hf.co/settings/tokens (needs WRITE to push) | wandb.ai/authorize
# HF alternative: `./robot run hf auth login` caches a token in the mounted HF
# cache instead. .env.local wins over it, since HF_TOKEN takes precedence.
set -euo pipefail
cd "$(dirname "$0")/.."

envfile=".env.local"
IMAGE="${IMAGE:-agent101/lerobot}"
do_hf=0; do_wb=0; force=0; status=0; verify=1
while [ $# -gt 0 ]; do
  case "$1" in
    --hf|--huggingface) do_hf=1; shift ;;
    --wandb|--wb)       do_wb=1; shift ;;
    --force|--relogin)  force=1; shift ;;
    --status|--show)    status=1; shift ;;
    --no-verify)        verify=0; shift ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown flag: $1  (try ./robot login --help)" >&2; exit 1 ;;
  esac
done
if [ "$do_hf" = 0 ] && [ "$do_wb" = 0 ]; then do_hf=1; do_wb=1; fi

# --- .env.local read/write (0600, one KEY=value per line) -----------------
get() { [ -f "$envfile" ] || return 0; sed -n "s/^$1=//p" "$envfile" | tail -n1; }

put() { # KEY VALUE — replace the key in place, or append it
  local k="$1" v="$2" tmp
  tmp=$(mktemp "${envfile}.XXXXXX")
  chmod 600 "$tmp"
  if [ -f "$envfile" ]; then
    grep -v "^${k}=" "$envfile" >> "$tmp" || true
  else
    echo "# Local secrets — gitignored, never commit. Written by ./robot login." >> "$tmp"
  fi
  printf '%s=%s\n' "$k" "$v" >> "$tmp"
  mv "$tmp" "$envfile"
  chmod 600 "$envfile"
}

mask() { # show only the tail, enough to tell two tokens apart
  local v="$1"; [ ${#v} -gt 4 ] && echo "…${v: -4}" || echo "…"
}

ask() { # KEY  LABEL  URL   -> ensure the key has a value
  local k="$1" label="$2" url="$3" cur val
  cur=$(get "$k")
  if [ -n "$cur" ] && [ "$force" = 0 ]; then
    echo "  $label: already set $(mask "$cur")   (replace with --force)"
    return 0
  fi
  if [ ! -t 0 ]; then   # non-interactive (CI, piped) — can't prompt safely
    echo "  $label: missing, and no terminal to prompt on." >&2
    echo "     add it with:  echo '$k=<token>' >> $envfile" >&2
    return 1
  fi
  echo "  $label — paste a token from $url"
  read -rsp "     $k: " val < /dev/tty; echo
  if [ -z "$val" ]; then echo "     empty input — skipped." >&2; return 1; fi
  put "$k" "$val"
  echo "     stored in $envfile $(mask "$val")"
}

# --- verification ---------------------------------------------------------
# Verify where the credentials will actually be USED, which is the image whenever
# there is one — a random host python (conda, system) can be years out of date and
# fail on keys that work fine in the container. Only fall back to host python when
# there is no image, i.e. a bare cloud box installed by setup-cloud.sh.
verify_env="(none)"
if command -v docker >/dev/null 2>&1 && docker image inspect "$IMAGE" >/dev/null 2>&1; then
  verify_env="the $IMAGE container"
elif python -c "import huggingface_hub, wandb" >/dev/null 2>&1; then
  verify_env="this host python ($(python -c 'import sys;print(sys.executable)'))"
fi

in_env() { # CMD...  (passes the tokens through, nothing is printed by us)
  case "$verify_env" in
    "the $IMAGE container")
      docker run --rm --entrypoint bash \
        -e HF_TOKEN="${HF_TOKEN:-}" -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        "$IMAGE" -lc "$1" ;;
    "(none)")
      echo "(no $IMAGE image and no python with huggingface_hub/wandb — skipping verify)" >&2
      return 0 ;;
    *)
      env HF_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" bash -lc "$1" ;;
  esac
}

echo "credentials in $envfile:"
rc=0
if [ "$status" = 1 ]; then
  for k in HF_TOKEN WANDB_API_KEY; do
    v=$(get "$k"); [ -n "$v" ] && echo "  $k: set $(mask "$v")" || echo "  $k: — not set"
  done
else
  if [ "$do_hf" = 1 ]; then ask HF_TOKEN      "Hugging Face"     "hf.co/settings/tokens" || rc=1; fi
  if [ "$do_wb" = 1 ]; then ask WANDB_API_KEY "Weights & Biases" "wandb.ai/authorize"    || rc=1; fi
fi

if [ "$verify" = 1 ]; then
  # Exported values win over the file, matching what ./robot train forwards.
  export HF_TOKEN="${HF_TOKEN:-$(get HF_TOKEN)}"
  export WANDB_API_KEY="${WANDB_API_KEY:-$(get WANDB_API_KEY)}"
  echo "verifying in $verify_env (needs network):"
  # Both CLIs read the token from the env and exit non-zero on a bad one.
  if [ "$do_hf" = 1 ] || [ "$status" = 1 ]; then
    if [ -z "$HF_TOKEN" ]; then
      echo "  Hugging Face: no token — checking the mounted HF cache instead"
    fi
    in_env 'set -o pipefail; hf auth whoami 2>&1 | sed "s/^/  Hugging Face: /"' || rc=1
  fi
  if [ "$do_wb" = 1 ] || [ "$status" = 1 ]; then
    if [ -n "$WANDB_API_KEY" ]; then
      # Captured rather than streamed so an outdated wandb can be named as the cause.
      if out=$(in_env 'wandb login --verify "$WANDB_API_KEY" 2>&1'); then
        echo "$out" | grep -vE "netrc|debug-cli" | sed 's/^/  W\&B: /'
      else
        echo "$out" | grep -vE "netrc|debug-cli" | tail -3 | sed 's/^/  W\&B: /'
        if echo "$out" | grep -q "must be 40 characters"; then
          echo "  W&B: ^ that is the OLD 40-char key rule. Your key is the newer"
          echo "       wandb_v1_ format, and that wandb is too old to accept it."
          echo "       fix: pip install -U wandb   (or ./robot build, so this verifies"
          echo "       in the container, which is where training reads the key anyway)"
        fi
        rc=1
      fi
    else
      echo "  W&B: no key — skipped"
    fi
  fi
fi

if [ "$rc" = 0 ]; then
  echo
  # HF can also be authenticated via the mounted cache, but wandb has no
  # persistent store in a --rm container, so no key really does mean no logging.
  if [ -z "${WANDB_API_KEY:-}$(get WANDB_API_KEY)" ]; then
    echo "no W&B key stored — training will run with wandb off."
    echo "add one any time:  ./robot login --wandb"
  else
    echo "done — ./robot train picks both up automatically, wandb included."
  fi
fi
exit "$rc"
