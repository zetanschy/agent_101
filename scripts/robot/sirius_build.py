#!/usr/bin/env python3
"""Build a Sirius training set: demonstrations and deployment rollouts in one dataset.

    ./robot sirius-build --name cap_to_cup_sirius_r1 \\
        --demos zetanschy/cap_to_cup_200 --rollouts zetanschy/rollout_cap_to_cup_sirius_r1
    # round 2 aggregates everything so far (Sirius: D_i = D_{i-1} + D'):
    ./robot sirius-build --name cap_to_cup_sirius_r2 \\
        --demos zetanschy/cap_to_cup_200 --rollouts zetanschy/rollout_..._r1,zetanschy/rollout_..._r2

Sirius (Liu et al., RSS 2023, "Robot Learning on the Job") trains each round on ALL the
data gathered so far, re-weighted by what kind of sample each frame is: a human
demonstration, a robot action, a human intervention, or a robot action in the moments
before an intervention. openpi trains on one dataset, so this puts the demonstrations
and the `./robot dagger` rollouts into one. It adds two per-frame flags, which are all
the weighting needs (scripts/openpi/dagger_weights.py, scheme "sirius"):

    demonstration  True on every frame of a --demos source
    intervention   the rollouts' own column (True while the operator drove); False on demos

A frame's class follows from them: demo, intv, or robot. preintv (the robot frames just
before each intervention) is assigned at training time, because its window length is a
weighting hyperparameter (--preintv-s), not a property of the recording.

Sources are only read. Each gets a temporary shadow: videos and task tables are
symlinked, and the data rows, per-episode stats and stats.json are rewritten with the
two columns added. lerobot's aggregate_datasets then merges the shadows, demos first and
then the rollouts in the order given. The episode order matters: the FIFO and FILO memory
strategies read it as arrival order.

Everything else is merge_datasets.py's: the same robot_type and depth-key unification,
the same refusals (units, resolution, codec, fps). Missing sources are pulled from the
Hub first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_datasets as md  # noqa: E402  (same directory: resolve, depth keys, units, consistency)

from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402

FLAG = {"dtype": "bool", "shape": [1], "names": None}
FLAGS = ("intervention", "demonstration")


def _stats(values: np.ndarray) -> dict:
    """lerobot's per-feature stats for a (N,) flag, shaped as its writer stores them."""
    s = get_feature_stats(values.astype(np.float32).reshape(-1, 1), axis=0, keepdims=False)
    s["count"] = np.array([len(values)])
    return s


def _pull(repo: str, root: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"  pulling {repo} from the Hub ...", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(root))


def shadow(src: Path, tmp: Path, is_demo: bool, robot_type: str, features: dict) -> tuple[Path, dict]:
    """A copy of `src` with both flags in its data, per-episode stats and stats.json."""
    dst = tmp / src.name
    (dst / "meta").mkdir(parents=True)
    for item in src.iterdir():
        if item.name not in ("meta", "data"):
            (dst / item.name).symlink_to(item)
    for item in (src / "meta").iterdir():
        if item.name not in ("info.json", "stats.json", "episodes"):
            (dst / "meta" / item.name).symlink_to(item)

    counts = {"frames": 0, "demo": 0, "intv": 0, "robot": 0}
    flags_by_ep: dict[int, dict[str, np.ndarray]] = {}
    for f in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        n = t.num_rows
        if "intervention" in t.column_names:
            if is_demo:
                raise SystemExit(f"{src.name} has an `intervention` column: it is a rollout, pass it in --rollouts")
            intv = np.asarray(t.column("intervention").to_numpy(zero_copy_only=False), dtype=bool)
        else:
            if not is_demo:
                raise SystemExit(f"{src.name} has no `intervention` column, so it is not a `./robot dagger` "
                                 "rollout: pass it in --demos")
            intv = np.zeros(n, dtype=bool)
            t = t.append_column("intervention", pa.array(intv, pa.bool_()))
        demo = np.full(n, is_demo)
        t = t.append_column("demonstration", pa.array(demo, pa.bool_()))
        meta = json.loads(t.schema.metadata[b"huggingface"]) if t.schema.metadata and b"huggingface" in t.schema.metadata else None
        if meta is not None:
            for k in FLAGS:
                meta["info"]["features"][k] = {"dtype": "bool", "_type": "Value"}
            t = t.replace_schema_metadata({**t.schema.metadata, b"huggingface": json.dumps(meta).encode()})
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
        eps = t.column("episode_index").to_numpy()
        for e in np.unique(eps):
            m = eps == e
            d = flags_by_ep.setdefault(int(e), {"intervention": [], "demonstration": []})
            d["intervention"].append(intv[m])
            d["demonstration"].append(demo[m])
        counts["frames"] += n
        counts["demo"] += int(demo.sum())
        counts["intv"] += int(intv.sum())
        counts["robot"] += int((~demo & ~intv).sum())
    flags_by_ep = {e: {k: np.concatenate(v) for k, v in d.items()} for e, d in flags_by_ep.items()}

    ep_stats = []
    (dst / "meta" / "episodes").mkdir(parents=True)
    for f in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f)
        for k in FLAGS:
            for stat in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                col = f"stats/{k}/{stat}"
                if col in t.column_names:
                    t = t.drop_columns([col])
        per_ep = [{k: _stats(flags_by_ep[int(e)][k]) for k in FLAGS} for e in t.column("episode_index").to_numpy()]
        ep_stats += per_ep
        for k in FLAGS:
            for stat in per_ep[0][k]:
                typ = pa.list_(pa.int64()) if stat == "count" else pa.list_(pa.float64())
                t = t.append_column(f"stats/{k}/{stat}", pa.array([s[k][stat].tolist() for s in per_ep], typ))
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)

    stats = json.loads((src / "meta" / "stats.json").read_text())
    for k in FLAGS:
        agg = aggregate_stats([{k: s[k]} for s in ep_stats])[k]
        stats[k] = {s: np.asarray(v).tolist() for s, v in agg.items()}
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    info = json.loads((src / "meta" / "info.json").read_text())
    info["robot_type"] = robot_type
    info["features"] = {**features, **{k: dict(FLAG) for k in FLAGS}}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return dst, counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="output dataset (bare name -> <user>/<name>)")
    p.add_argument("--demos", required=True, help="comma-separated demonstration datasets")
    p.add_argument("--rollouts", required=True, help="comma-separated `./robot dagger` datasets, oldest first")
    p.add_argument("--user", default=os.environ.get("HF_USER") or "soarm101", help="namespace for bare names")
    p.add_argument("--force", action="store_true", help="overwrite an existing output dataset")
    a = p.parse_args()

    split = lambda s: [md.resolve(x.strip(), a.user) for x in s.split(",") if x.strip()]  # noqa: E731
    demos, rollouts = split(a.demos), split(a.rollouts)
    sources = demos + rollouts
    if not demos or not rollouts:
        print("need at least one --demos and one --rollouts dataset", file=sys.stderr)
        return 1
    out_repo = md.resolve(a.name, a.user)
    out_root = HF_LEROBOT_HOME / out_repo
    if out_root.exists():
        if not a.force:
            print(f"output already exists: {out_root}\nuse --force to overwrite", file=sys.stderr)
            return 1
        shutil.rmtree(out_root)

    roots = [HF_LEROBOT_HOME / s for s in sources]
    for s, r in zip(sources, roots, strict=True):
        if not (r / "meta" / "info.json").exists():
            _pull(s, r)

    broken = {s: bad for s, r in zip(sources, roots, strict=True) if (bad := md.check_consistent(r))}
    if broken:
        print("video/data length mismatches (fix with ./robot data repair --name X --apply):", file=sys.stderr)
        for s, bad in broken.items():
            print(f"  {s}: episodes {', '.join(bad)}", file=sys.stderr)
        return 1
    units = {s: md.joint_units(r) for s, r in zip(sources, roots, strict=True)}
    if len({u for u in units.values() if u != "unknown"}) > 1:
        print("these sources use DIFFERENT joint units:", file=sys.stderr)
        for s, u in units.items():
            print(f"  {u:11s} {s}", file=sys.stderr)
        return 1

    infos = [json.loads((r / "meta" / "info.json").read_text()) for r in roots]
    want_type = Counter(i.get("robot_type") for i in infos).most_common(1)[0][0]
    depth = [k for k in (md.depth_key_of(i["features"]) for i in infos) if k]
    want_depth = Counter(depth).most_common(1)[0][0] if depth else md.DEPTH_ALIASES[0]
    base = [{k: v for k, v in md.rename_depth_key(i["features"], want_depth).items() if k not in FLAGS} for i in infos]
    for s, f in zip(sources[1:], base[1:], strict=True):
        if not md.features_equal_for_merge(base[0], f):
            print(f"{s} has features incompatible with {sources[0]} beyond the two flags:", file=sys.stderr)
            for key in sorted(set(base[0]) | set(f)):
                if base[0].get(key) != f.get(key):
                    print(f"  {key}: {base[0].get(key)}  vs  {f.get(key)}", file=sys.stderr)
            return 1

    print(f"Sirius set -> {out_repo}")
    total = {"frames": 0, "demo": 0, "intv": 0, "robot": 0}
    with tempfile.TemporaryDirectory(prefix="sirius-build-") as tmpdir:
        shadows = []
        for s, r, i, f in zip(sources, roots, infos, base, strict=True):
            sh, c = shadow(r, Path(tmpdir), s in demos, want_type, f)
            shadows.append(sh)
            for k in total:
                total[k] += c[k]
            kind = "demos   " if s in demos else "rollouts"
            print(f"  {kind} {s:52s} {i['total_episodes']:4d} eps {c['frames']:7d} frames"
                  + ("" if s in demos else f"  ({c['intv']} intervention, {c['robot']} robot)"))
        print()
        aggregate_datasets(repo_ids=sources, aggr_repo_id=out_repo, roots=shadows, aggr_root=out_root)

    info = json.loads((out_root / "meta" / "info.json").read_text())
    if info["total_frames"] != total["frames"] or not all(k in info["features"] for k in FLAGS):
        print(f"merge came out wrong: {info['total_frames']} frames vs {total['frames']} expected", file=sys.stderr)
        return 1
    print(f"\nbuilt {out_root}: {info['total_episodes']} episodes, {info['total_frames']} frames")
    print(f"  demo {total['demo']} ({100 * total['demo'] / total['frames']:.0f}%), robot {total['robot']} "
          f"({100 * total['robot'] / total['frames']:.0f}%), intervention {total['intv']} "
          f"({100 * total['intv'] / total['frames']:.0f}%)")
    # The paper ends a round when the interventions reach a third of the initial demonstrations.
    print(f"  interventions are {total['intv'] / max(total['demo'], 1):.2f} of the demonstration frames "
          "(Sirius ran a round until ~0.33)")
    print(f"\ntrain:  ./robot openpi-sirius-train --data.repo-id={out_repo} --exp-name={out_repo.split('/')[-1]}")
    print(f"push :  ./robot data upload --name {out_repo.split('/')[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
