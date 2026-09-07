"""Correct an operator verdict, into a NEW log that says it was corrected.

    ./robot eval-amend --scene layout-right --judgement n --note "..."
    ./robot eval-amend LOG.json --scene layout-left --judgement partial

The operator IS the scorer for this benchmark, so a mistyped or missed verdict is a
recording error rather than a result -- and it happens: one trial was scored on the
typo "yu", and another was left "skip" because the episode was cancelled before the
prompt. Both silently scored 0.

WHAT THIS DELIBERATELY DOES NOT DO IS EDIT THE ORIGINAL. Inspect Robots logs are
immutable, auditable records, and a comparison between two policies is only worth
anything if its inputs cannot be quietly rewritten. So this writes a new file, leaves
the source untouched, and stamps the amended trial's own `trial_metadata` with the
source log, the scene, the verdict before and after, and the reason. Anyone reading the
result can see it was amended, from what, and why.

The stamp goes on the TRIAL rather than the eval spec because the log schema is strict:
EvalLog.from_dict builds EvalSpec(**data["eval"]) and SceneResult(**sample), so an
invented key in either makes the log unreadable by the framework that wrote it. Only
trial_metadata is free-form, and it happens to be the right place anyway.

The score is recomputed with the framework's own rule (is_affirmative_verdict), not a
local guess at it. Note what that rule implies: "skip" and "n" are both non-affirmative
and both score 0.0, so correcting a skip to a fail changes the RECORD and not the
number. That is the point -- the number was never wrong, the reason was missing.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import glob
import json
import os
import pathlib
import sys

from inspect_robots.scorer import is_affirmative_verdict


def newest_log(log_dir: str) -> str:
  logs = glob.glob(os.path.join(log_dir, "*.json"))
  if not logs:
    raise SystemExit(f"no eval logs in {log_dir}")
  return max(logs, key=os.path.getmtime)


def amend(
  log: dict,
  scene_id: str,
  judgement: str,
  note: str | None,
  source: str,
  reason: str,
) -> dict:
  """Return a copy of `log` with one scene's verdict replaced and scores redone."""
  out = copy.deepcopy(log)
  targets = [s for s in out.get("samples", []) if s.get("scene_id") == scene_id]
  if not targets:
    have = sorted(s.get("scene_id") for s in out.get("samples", []))
    raise SystemExit(f"no scene {scene_id!r} in this log; it has {have}")
  sample = targets[0]

  before = list(sample.get("operator_judgements") or [])
  # Only the first epoch is amended: this exists for a mis-recorded verdict, and
  # rewriting every epoch of a repeated scene would hide which one was corrected.
  sample["operator_judgements"] = [judgement] + before[1:]
  if note is not None:
    notes = list(sample.get("operator_notes") or [])
    sample["operator_notes"] = [note] + notes[1:]

  value = 1.0 if is_affirmative_verdict(judgement) else 0.0
  epochs = sample.get("epochs") or [{}]
  epochs[0] = {**(epochs[0] or {}), "operator": value}
  sample["epochs"] = epochs
  sample["reduced"] = {**(sample.get("reduced") or {}), "operator": value}

  scored = [
    (s.get("reduced") or {}).get("operator")
    for s in out.get("samples", [])
    if (s.get("reduced") or {}).get("operator") is not None
  ]
  if scored:
    results = out.setdefault("results", {})
    metrics = results.setdefault("metrics", {})
    metrics["operator"] = sum(scored) / len(scored)

  # Provenance goes in the amended trial's own metadata, NOT on the eval spec.
  # EvalLog.from_dict builds EvalSpec(**data["eval"]) and SceneResult(**sample), both
  # strict, so an invented key there makes the log unreadable by the framework that
  # wrote it -- which defeats the point of amending it in place of the original.
  # trial_metadata is a tuple of free-form dicts and survives the round trip, and it
  # puts the record on the trial it actually describes.
  trials = list(sample.get("trial_metadata") or [{}])
  first = dict(trials[0] or {})
  amendments = list(first.get("amended") or [])
  amendments.append(
    {
      "source_log": os.path.basename(source),
      "scene_id": scene_id,
      "verdict_before": before[0] if before else None,
      "verdict_after": judgement,
      "note_after": note,
      "reason": reason,
      "amended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
  )
  first["amended"] = amendments
  trials[0] = first
  sample["trial_metadata"] = trials
  return out


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("log", nargs="?", help="log to amend (default: newest in --log-dir)")
  p.add_argument("--log-dir", default="outputs/evals")
  p.add_argument("--scene", required=True, help="scene_id whose verdict is wrong")
  p.add_argument("--judgement", required=True, help="the correct verdict (y/n/partial/skip)")
  p.add_argument("--note", default=None, help="the grader note to record with it")
  p.add_argument("--reason", default="operator verdict mis-recorded during the run",
                 help="why the amendment was needed; stored in the log")
  p.add_argument("-o", "--out", default=None, help="output path (default: <log>_amended.json)")
  a = p.parse_args(argv)

  source = a.log or newest_log(a.log_dir)
  log = json.loads(pathlib.Path(source).read_text())
  out = amend(log, a.scene, a.judgement, a.note, source, a.reason)

  dest = a.out or source.replace(".json", "_amended.json")
  pathlib.Path(dest).write_text(json.dumps(out, indent=2))
  print(f"source: {source}  (unchanged)")
  print(f"wrote : {dest}")
  print(f"metric: operator {(log.get('results') or {}).get('metrics', {}).get('operator')} "
        f"-> {out['results']['metrics']['operator']}")
  for s in out["samples"]:
    print(f"  {s['scene_id']:20s} {str(s.get('operator_judgements')):12s} "
          f"{(s.get('reduced') or {}).get('operator')}  {s.get('operator_notes')}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
