"""Assemble a result log from trials that happened in different runs.

    python -m evals.compose --base LATEST.json --drop layout-right \\
        --graft-log CANCELLED.json --graft-scene layout-near --as layout-near-2 \\
        --judgement n --note "Failed to grasp three times as commanded"

WHY THIS IS EVER LEGITIMATE. A session is a sequence of physical trials, and the run
boundaries are an artefact of the operator's terminal, not of the experiment: a trial
interrupted by a Ctrl-C still happened, still moved the arm, and still has an outcome a
person watched. Losing it because the process exited before the verdict prompt would
throw away the most informative trial of the session -- which is exactly what happened
here, where the cancelled trial ran 38 decisions over 308 s while the trial it replaces
ran 2.

WHY IT IS DANGEROUS, and what this does about it. A hand-assembled log that looked like
a single clean run would be the wrong artefact to build a policy comparison on. So:

  * The sources are never modified. This only writes a new file.
  * Every grafted trial carries provenance in its own trial_metadata: the log it came
    from, its original scene id, its status there, the verdict applied afterwards and
    the reason. The stamp lives on the trial rather than the eval spec because the log
    schema is strict -- EvalLog.from_dict builds EvalSpec(**data["eval"]) and
    SceneResult(**sample), so an invented key in either makes the log unreadable.
  * A grafted trial keeps its true `status` unless `--status` says otherwise. Process
    and outcome are two different facts: a trial can be `cancelled` (the operator hit
    Ctrl-C) and a failure (the arm never grasped). Relabelling is allowed because the
    listing reads better, but the ORIGINAL status is always kept in the provenance
    stamp, so the substitution is visible rather than lost.
  * Scene ids must stay unique (`--as`), because the framework keys a trial's frames,
    actions and transcript off `<scene_id>-e<epoch>`. Two trials sharing an id would
    silently collide.

FRAMES ARE HARD-LINKED, not copied or left behind. A log has ONE stats.frames_dir, and
a grafted trial's frames live under its own run's directory, so the report and
`inspect-robots video` would find nothing for it. Hard links put them where the composed
log looks, under the new scene id, at no disk cost and with no dangling-symlink risk --
both directories are on the same filesystem by construction, being siblings under the
log directory.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import pathlib
import sys

from inspect_robots.scorer import is_affirmative_verdict


def _link_frames(
  src_log: str, dst_log: str, src_frames: str, dst_frames: str,
  src_prefix: str, dst_prefix: str
) -> int:
  """Hard-link one trial's frames into the composed log's frames directory.

  `stats.frames_dir` is stored relative to the WORKING DIRECTORY of the run, not to
  the log file, so it is resolved with the framework's own resolve_frames_dir rather
  than joined onto the log's parent -- joining gave outputs/evals/outputs/evals/... and
  silently linked nothing.
  """
  from inspect_robots._video import resolve_frames_dir

  src = resolve_frames_dir(src_frames, pathlib.Path(src_log))
  dst = resolve_frames_dir(dst_frames, pathlib.Path(dst_log))
  if src is None or dst is None:
    return 0
  dst.mkdir(parents=True, exist_ok=True)
  linked = 0
  for path in sorted(src.glob(f"{src_prefix}_*.npy")):
    target = dst / (dst_prefix + path.name[len(src_prefix) :])
    if target.exists():
      continue
    try:
      os.link(path, target)
    except OSError:
      # Different filesystem, or a permission wrinkle: copy rather than fail the graft.
      target.write_bytes(path.read_bytes())
    linked += 1
  return linked


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--base", required=True, help="log whose trials are kept")
  p.add_argument("--drop", default=None, help="scene_id to remove from the base")
  p.add_argument("--graft-log", required=True, help="log to take a trial from")
  p.add_argument("--graft-scene", required=True, help="scene_id to take")
  p.add_argument("--as", dest="new_id", required=True, help="scene_id to give it here")
  p.add_argument("--judgement", required=True, help="operator verdict for it (y/n/partial)")
  p.add_argument("--note", default=None, help="grader note for it")
  p.add_argument("--reason", default="trial interrupted before the verdict prompt",
                 help="why it is being grafted; stored in the log")
  p.add_argument("--status", default=None,
                 help="override the grafted trial's status (e.g. success); the original "
                      "is preserved in the provenance stamp")
  p.add_argument("-o", "--out", default=None)
  a = p.parse_args(argv)

  base_path = pathlib.Path(a.base)
  log_dir = base_path.parent
  out = json.loads(base_path.read_text())
  src = json.loads(pathlib.Path(a.graft_log).read_text())

  if a.drop:
    kept = [s for s in out["samples"] if s.get("scene_id") != a.drop]
    if len(kept) == len(out["samples"]):
      raise SystemExit(f"--drop {a.drop!r} matched nothing")
    out["samples"] = kept

  matches = [s for s in src.get("samples", []) if s.get("scene_id") == a.graft_scene]
  if not matches:
    raise SystemExit(f"--graft-scene {a.graft_scene!r} not in {a.graft_log}")
  sample = copy.deepcopy(matches[0])
  if any(s.get("scene_id") == a.new_id for s in out["samples"]):
    raise SystemExit(f"--as {a.new_id!r} already exists in the base log")

  original_status = sample.get("status")
  original_scene = sample.get("scene_id")
  if a.status:
    sample["status"] = a.status
  sample["scene_id"] = a.new_id
  sample["operator_judgements"] = [a.judgement]
  sample["operator_notes"] = [a.note if a.note is not None else ""]
  value = 1.0 if is_affirmative_verdict(a.judgement) else 0.0
  sample["reduced"] = {**(sample.get("reduced") or {}), "operator": value}
  sample["epochs"] = [{**((sample.get("epochs") or [{}])[0] or {}), "operator": value}]

  linked = _link_frames(
    a.graft_log,
    a.base,
    (src.get("stats") or {}).get("frames_dir", ""),
    (out.get("stats") or {}).get("frames_dir", ""),
    f"{original_scene}-e0",
    f"{a.new_id}-e0",
  )

  trials = list(sample.get("trial_metadata") or [{}])
  first = dict(trials[0] or {})
  first["grafted"] = {
    "source_log": os.path.basename(a.graft_log),
    "source_scene_id": original_scene,
    "source_status": original_status,
    "status_shown": sample.get("status"),
    "verdict_applied": a.judgement,
    "note_applied": a.note,
    "reason": a.reason,
    "frames_linked": linked,
    "composed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }
  trials[0] = first
  sample["trial_metadata"] = trials
  out["samples"].append(sample)

  scored = [
    (s.get("reduced") or {}).get("operator")
    for s in out["samples"]
    if (s.get("reduced") or {}).get("operator") is not None
  ]
  out.setdefault("results", {}).setdefault("metrics", {})["operator"] = (
    sum(scored) / len(scored) if scored else 0.0
  )
  out["results"]["total_scenes"] = len(out["samples"])
  out["results"]["total_trials"] = len(out["samples"])

  dest = a.out or str(base_path).replace(".json", "_composed.json")
  pathlib.Path(dest).write_text(json.dumps(out, indent=2))
  print(f"base : {a.base}  (unchanged)")
  print(f"graft: {a.graft_log}  (unchanged)  frames hard-linked: {linked}")
  print(f"wrote: {dest}")
  print(f"metric: operator {out['results']['metrics']['operator']:.4f}\n")
  for s in out["samples"]:
    tm = (s.get("trial_metadata") or [{}])[0] or {}
    tag = "  <- grafted" if "grafted" in tm else ""
    print(f"  {s['scene_id']:20s} status={s.get('status'):10s} "
          f"{str(s.get('operator_judgements')):8s} "
          f"{(s.get('reduced') or {}).get('operator')}  "
          f"{s.get('operator_notes')}{tag}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
