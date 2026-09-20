"""What one eval trial cost in LLM tokens, and in dollars.

    ./robot eval-cost                 # newest log
    ./robot eval-cost LOG.json
    ./robot eval-cost --in 10 --out 50 --cached 1     # override the rates

The agent policy records `llm_usage` per trial into the log -- llm_calls plus whatever
token counters the provider returned -- so this reads the log rather than re-querying
anyone. Only the LLM side has a token bill at all: the openpi pi0.5 policy runs on the
GPU in this box, so its trials show no usage and are priced at zero here. Comparing the
two on cost therefore means deciding what an hour of local inference is worth, which is
a judgement this script does not make for you.

RATES ARE A DEFAULT, NOT A FACT. They are GPT-6 Astra's published standard-tier prices
as of September 2026 (10 / 50 / 1 dollars per million input / output / cached-input
tokens) and they go stale: OpenAI charges double on input past a 272k-token request,
half for batch and flex, double for fast mode. Pass --in/--out/--cached for anything
other than a plain standard-tier run, and check the model you actually invoked -- the
log records it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

# Dollars per MILLION tokens. See the module docstring before trusting these.
DEFAULT_IN = 10.0
DEFAULT_OUT = 50.0
DEFAULT_CACHED = 1.0

# The counter names providers use for the same three quantities.
_IN_KEYS = ("input_tokens", "prompt_tokens")
_OUT_KEYS = ("output_tokens", "completion_tokens")
_CACHED_KEYS = ("cache_read_input_tokens", "cached_input_tokens")


def _pick(usage: dict, keys: tuple[str, ...]) -> int:
  for key in keys:
    if key in usage:
      return int(usage[key])
  return 0


def _wire_usage(log_dir: pathlib.Path, wire_rel: str) -> dict[str, int]:
  """Sum token counts out of a trial's raw wire capture.

  The Responses wire records only `llm_calls` into `llm_usage` -- the token counters
  live in each captured response instead -- so a run on that wire prices at zero
  without this. The capture is the provider's own accounting, so it is the better
  source anyway; `llm_usage` is preferred only because it is cheaper to read.
  """
  path = log_dir / wire_rel
  if not path.is_file():
    return {}
  totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
  for line in path.read_text().splitlines():
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError:
      continue
    usage = (record.get("response") or {}).get("usage") or {}
    if not isinstance(usage, dict):
      continue
    totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
    totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
    details = usage.get("input_tokens_details") or {}
    if isinstance(details, dict):
      totals["cached_tokens"] += int(details.get("cached_tokens", 0) or 0)
  return totals


def newest_log(log_dir: str) -> str:
  logs = glob.glob(os.path.join(log_dir, "*.json"))
  if not logs:
    raise SystemExit(f"no eval logs in {log_dir}")
  return max(logs, key=os.path.getmtime)


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("log", nargs="?", help="eval log (default: newest in --log-dir)")
  p.add_argument("--log-dir", default="outputs/evals")
  p.add_argument("--in", dest="rate_in", type=float, default=DEFAULT_IN,
                 help=f"$ per million input tokens (default {DEFAULT_IN})")
  p.add_argument("--out", dest="rate_out", type=float, default=DEFAULT_OUT,
                 help=f"$ per million output tokens (default {DEFAULT_OUT})")
  p.add_argument("--cached", dest="rate_cached", type=float, default=DEFAULT_CACHED,
                 help=f"$ per million cached input tokens (default {DEFAULT_CACHED})")
  a = p.parse_args(argv)

  path = a.log or newest_log(a.log_dir)
  log = json.loads(pathlib.Path(path).read_text())
  print(f"log: {path}")
  print(f"status: {log.get('status')}   rates: ${a.rate_in}/${a.rate_out}/"
        f"${a.rate_cached} per Mtok (in/out/cached)\n")

  header = f"{'trial':22s} {'calls':>6s} {'in':>9s} {'cached':>9s} {'out':>8s} {'$':>8s}"
  print(header)
  print("-" * len(header))
  totals = {"calls": 0, "in": 0, "cached": 0, "out": 0, "usd": 0.0}
  priced = 0
  for sample in log.get("samples", []):
    metas = sample.get("trial_metadata") or [{}]
    for meta in metas:
      usage = (meta or {}).get("llm_usage") or {}
      calls = int(usage.get("llm_calls", 0))
      tin = _pick(usage, _IN_KEYS)
      tout = _pick(usage, _OUT_KEYS)
      tcached = _pick(usage, _CACHED_KEYS)
      if calls and not (tin or tout) and (meta or {}).get("wire_capture"):
        wire = _wire_usage(pathlib.Path(path).parent, meta["wire_capture"])
        tin = wire.get("input_tokens", 0)
        tout = wire.get("output_tokens", 0)
        tcached = wire.get("cached_tokens", 0)
      # Providers report cached reads inside the input total; charging both would
      # double-count the cheap half of the bill.
      billed_in = max(tin - tcached, 0)
      usd = (billed_in * a.rate_in + tcached * a.rate_cached + tout * a.rate_out) / 1e6
      if calls:
        priced += 1
      totals["calls"] += calls
      totals["in"] += billed_in
      totals["cached"] += tcached
      totals["out"] += tout
      totals["usd"] += usd
      print(f"{sample.get('scene_id', '?'):22s} {calls:6d} {billed_in:9d} "
            f"{tcached:9d} {tout:8d} {usd:8.4f}")
  print("-" * len(header))
  print(f"{'TOTAL':22s} {totals['calls']:6d} {totals['in']:9d} "
        f"{totals['cached']:9d} {totals['out']:8d} {totals['usd']:8.4f}")
  if priced:
    print(f"\nmean per trial: ${totals['usd'] / priced:.4f} over {priced} trial(s) "
          f"with usage")
  else:
    print("\nno llm_usage in this log: either a local policy (openpi/lerobot pay no "
          "token bill) or a run that made no LLM calls.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
