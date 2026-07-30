#!/usr/bin/env python
"""Merge several LeRobot datasets into one (lerobot's `aggregate_datasets`).

Episodes are concatenated in the order given and re-indexed; videos and parquet
files are re-packed into the destination's shard layout. Sources are only ever
read.

    ./robot data merge --name cap_to_cup_all \
        --from cap_to_cup,cap_to_cup2_20260725_182259,cap_to_cup3_20260725_193652

`aggregate_datasets` requires identical fps, features AND robot_type, compared
key-exactly. Two cosmetic renames across lerobot versions block otherwise valid
merges of datasets recorded before/after an upgrade:

  robot_type          so101_follower      -> so_follower      (same physical arm)
  video info depth    video.is_depth_map  -> is_depth_map     (same value)

Both are unified by default, to whichever variant most sources agree on. Anything
beyond those two — resolution, codec, fps, action/state shape — still refuses, so
genuinely incompatible datasets can't be merged by accident. Nothing is rewritten
on disk: the odd ones out get a temporary shadow root of symlinks with a patched
meta/info.json and aggregate reads through that. --strict skips all unification.

Sources are checked for the video/data length mismatch first (see
repair_timestamps.py) — merging carries the inconsistency into the result.
"""

import argparse
import copy
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.feature_utils import features_equal_for_merge
from lerobot.utils.constants import HF_LEROBOT_HOME

DEPTH_ALIASES = ("is_depth_map", "video.is_depth_map")


def resolve(name: str, default_user: str) -> str:
    return name if "/" in name else f"{default_user}/{name}"


def depth_key_of(features: dict) -> str | None:
    """Which spelling of the depth flag this dataset's video features use."""
    for f in features.values():
        if f.get("dtype") != "video":
            continue
        for alias in DEPTH_ALIASES:
            if alias in (f.get("info") or {}):
                return alias
    return None


def rename_depth_key(features: dict, target: str) -> dict:
    out = copy.deepcopy(features)
    for f in out.values():
        if f.get("dtype") != "video":
            continue
        info = f.get("info") or {}
        for alias in DEPTH_ALIASES:
            if alias in info and alias != target:
                info[target] = info.pop(alias)
    return out


def check_consistent(root: Path) -> list[str]:
    """Episodes whose video span disagrees with their frame count."""
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files])
    bad = set()
    for col in [c for c in df.columns if c.endswith("/from_timestamp")]:
        key = col[len("videos/") : -len("/from_timestamp")]
        span = (
            (df[f"videos/{key}/to_timestamp"] - df[f"videos/{key}/from_timestamp"]) * fps
        ).round().astype(int)
        bad |= set(df.loc[span != df["length"], "episode_index"].tolist())
    return sorted(str(b) for b in bad)


def joint_units(root: Path) -> str:
    """Whether actions are in degrees or lerobot's normalized RANGE_M100_100.

    Nothing in info.json records this, and the two are indistinguishable by dtype or
    shape — but normalized values clamp at exactly ±100, while degrees overshoot it
    (a real SO-101 wrist_roll reaches -166). lerobot 0.6.1 added use_degrees and
    defaults it to True, so datasets recorded either side of that upgrade disagree,
    and merging them teaches contradictory actions for identical poses.
    """
    stats = root / "meta" / "stats.json"
    if not stats.exists():
        return "unknown"
    action = json.loads(stats.read_text()).get("action")
    if not action:
        return "unknown"
    lo, hi = min(action["min"]), max(action["max"])
    return "degrees" if lo < -100.5 or hi > 100.5 else "normalized"


def shadow(src_root: Path, tmp: Path, robot_type: str, features: dict) -> Path:
    """A read-only stand-in for src_root with robot_type/features overridden.

    Everything is symlinked except meta/info.json, so this costs no disk and
    cannot modify the source.
    """
    dst = tmp / src_root.name
    (dst / "meta").mkdir(parents=True)
    for item in src_root.iterdir():
        if item.name != "meta":
            (dst / item.name).symlink_to(item)
    for item in (src_root / "meta").iterdir():
        if item.name != "info.json":
            (dst / "meta" / item.name).symlink_to(item)
    info = json.loads((src_root / "meta" / "info.json").read_text())
    info["robot_type"] = robot_type
    info["features"] = features
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="output dataset (bare name -> <user>/<name>)")
    p.add_argument("--from", dest="sources", required=True, help="comma-separated datasets to merge, in order")
    p.add_argument("--user", default="soarm101", help="namespace for bare names")
    p.add_argument("--strict", action="store_true", help="refuse to unify version renames")
    p.add_argument("--force", action="store_true", help="overwrite an existing output dataset")
    args = p.parse_args()

    sources = [resolve(s.strip(), args.user) for s in args.sources.split(",") if s.strip()]
    if len(sources) < 2:
        print("need at least two datasets in --from", file=sys.stderr)
        return 1
    out_repo = resolve(args.name, args.user)
    out_root = HF_LEROBOT_HOME / out_repo

    roots = [HF_LEROBOT_HOME / s for s in sources]
    for s, r in zip(sources, roots, strict=True):
        if not (r / "meta" / "info.json").exists():
            print(f"not found locally: {s}  ({r})", file=sys.stderr)
            return 1

    if out_root.exists():
        if not args.force:
            print(f"output already exists: {out_root}\nuse --force to overwrite", file=sys.stderr)
            return 1
        shutil.rmtree(out_root)

    # Pre-flight: a source with the length/span mismatch would poison the merge.
    broken = {s: bad for s, r in zip(sources, roots, strict=True) if (bad := check_consistent(r))}
    if broken:
        print("these sources have video/data length mismatches — merge them and it spreads:", file=sys.stderr)
        for s, bad in broken.items():
            print(f"  {s}: episodes {', '.join(bad)}", file=sys.stderr)
            print(f"    fix: ./robot data repair --name {s.split('/')[-1]} --apply", file=sys.stderr)
        return 1

    # Units are the one incompatibility the fps/features/robot_type checks miss, and it
    # is silent: the merge succeeds and only the fine-tune comes out worse.
    units = {s: joint_units(r) for s, r in zip(sources, roots, strict=True)}
    if len({u for u in units.values() if u != "unknown"}) > 1:
        print("\nthese sources use DIFFERENT joint units — merging them teaches the policy", file=sys.stderr)
        print("two conventions for the same physical pose:", file=sys.stderr)
        for s, u in units.items():
            print(f"  {u:11s} {s}", file=sys.stderr)
        print("\nlerobot 0.6.1 added use_degrees (default True); anything recorded before it", file=sys.stderr)
        print("is normalized ±100. Merge one group at a time, or re-record the odd ones out.", file=sys.stderr)
        return 1

    infos = [json.loads((r / "meta" / "info.json").read_text()) for r in roots]
    types = [i.get("robot_type") for i in infos]
    feats = [i["features"] for i in infos]

    total = 0
    print(f"merging {len(sources)} datasets -> {out_repo}")
    for s, i, t in zip(sources, infos, types, strict=True):
        total += i["total_episodes"]
        print(f"  {s:44s} {i['total_episodes']:4d} eps  {i['total_frames']:7d} frames  robot_type={t}")
    print(f"  {'=> expected total':44s} {total:4d} eps")

    # Unify the two cosmetic renames, then let lerobot's own comparison decide.
    want_type = Counter(types).most_common(1)[0][0]
    depth_keys = [k for k in (depth_key_of(f) for f in feats) if k]
    want_depth = Counter(depth_keys).most_common(1)[0][0] if depth_keys else DEPTH_ALIASES[0]
    if args.strict:
        want_type, want_depth = types[0], depth_key_of(feats[0]) or DEPTH_ALIASES[0]
    norm = [rename_depth_key(f, want_depth) for f in feats]

    for s, f in zip(sources[1:], norm[1:], strict=True):
        if not features_equal_for_merge(norm[0], f):
            print(f"\n{s} has incompatible features — not just a version rename:", file=sys.stderr)
            for key in sorted(set(norm[0]) | set(f)):
                if norm[0].get(key) != f.get(key):
                    print(f"  {key}:\n    {sources[0]}: {norm[0].get(key)}\n    {s}: {f.get(key)}", file=sys.stderr)
            return 1

    if len(set(types)) > 1:
        if args.strict:
            print(f"\nrobot_type differs: {sorted(set(types))} — drop --strict to unify", file=sys.stderr)
            return 1
        print(f"\nrobot_type differs {sorted(set(types))} -> unifying to '{want_type}'")
    if len(set(depth_keys)) > 1:
        print(f"depth flag differs {sorted(set(depth_keys))} -> unifying to '{want_depth}'")
    if len(set(types)) > 1 or len(set(depth_keys)) > 1:
        print("  (renames only — fps, resolution, codec and action/state all match already;")
        print("   applied to a temp symlink shadow, your datasets are not modified)")

    with tempfile.TemporaryDirectory(prefix="lerobot-merge-") as tmpdir:
        tmp = Path(tmpdir)
        roots = [
            r if (t == want_type and f == n) else shadow(r, tmp, want_type, n)
            for r, t, f, n in zip(roots, types, feats, norm, strict=True)
        ]
        print()
        aggregate_datasets(repo_ids=sources, aggr_repo_id=out_repo, roots=roots, aggr_root=out_root)

    info = json.loads((out_root / "meta" / "info.json").read_text())
    print(f"\nmerged -> {out_root}")
    print(f"  {info['total_episodes']} episodes, {info['total_frames']} frames")
    if info["total_episodes"] != total:
        print(f"  WARNING: expected {total} episodes", file=sys.stderr)
        return 1
    print(f"\npush it with:  ./robot data upload --name {out_repo.split('/')[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
