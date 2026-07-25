#!/usr/bin/env python
"""Repair episodes whose video span is longer than their recorded frame count.

At record time LeRobot stores each episode's video span as the *encoded file's
duration* (datasets/dataset_writer.py: `ep_duration_in_s = get_video_duration_in_s`),
so when the encoder emits more frames than the control loop logged rows, that
episode's `to_timestamp` overshoots its `length`.

This is invisible during training: reads only ever use `from_timestamp + i/fps`
(datasets/dataset_reader.py), so frames past `length` are already unreachable.
But `lerobot-edit-dataset` extracts video by timestamp span while rebuilding the
output metadata from `length/fps` (datasets/dataset_tools.py), so it asserts the
two agree — and delete_episodes crashes on an otherwise healthy dataset:

    AssertionError: Episode length mismatch: 397 vs 439

Rewriting `to_timestamp = from_timestamp + length/fps` fixes that. Nothing
readable is lost — the unreachable tail is exactly what gets dropped.

    ./robot data repair --name cap_to_cup3           # report only
    ./robot data repair --name cap_to_cup3 --apply   # rewrite to_timestamp
"""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_HOME


def video_keys(schema_names: list[str]) -> list[str]:
    return sorted(
        n[len("videos/") : -len("/from_timestamp")]
        for n in schema_names
        if n.startswith("videos/") and n.endswith("/from_timestamp")
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="e.g. soarm101/cap_to_cup3")
    p.add_argument("--root", default=None, help="dataset dir (default: the local lerobot cache)")
    p.add_argument("--apply", action="store_true", help="write the fix (default: report only)")
    args = p.parse_args()

    root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    if not (root / "meta" / "info.json").exists():
        print(f"no dataset at {root}", file=sys.stderr)
        return 1
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]

    total_bad = 0
    for path in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)):
        table = pq.read_table(path)
        length = table.column("length").to_pylist()
        episode = table.column("episode_index").to_pylist()
        patched = {}

        for key in video_keys(table.schema.names):
            frm = table.column(f"videos/{key}/from_timestamp").to_pylist()
            to = table.column(f"videos/{key}/to_timestamp").to_pylist()
            files = list(
                zip(
                    table.column(f"videos/{key}/chunk_index").to_pylist(),
                    table.column(f"videos/{key}/file_index").to_pylist(),
                )
            )
            # Trimming the LAST episode of a video file would make a later
            # `record --resume` compute its from_timestamp from the trimmed value
            # and overlap real frames, so flag it instead of silently fixing.
            last_in_file = {f: max(i for i, g in enumerate(files) if g == f) for f in set(files)}

            fixed, rows = list(to), []
            for i, (f, t, ln) in enumerate(zip(frm, to, length)):
                span = round(t * fps) - round(f * fps)
                if span == ln:
                    continue
                rows.append((episode[i], ln, span, span - ln, i == last_in_file[files[i]]))
                fixed[i] = f + ln / fps

            if rows:
                total_bad += len(rows)
                print(f"{key}: {len(rows)} of {table.num_rows} episodes overshoot")
                print("   ep  length   span   extra")
                for ep, ln, span, extra, is_last in rows:
                    tail = "  <- LAST in its video file: fixing breaks `record --resume`" if is_last else ""
                    print(f"  {ep:3d}  {ln:6d} {span:6d}  {extra:+6d}{tail}")
                patched[f"videos/{key}/to_timestamp"] = fixed

        if not patched:
            continue
        if not args.apply:
            print(f"\n{path.replace(str(root), '.')}: would rewrite {len(patched)} column(s) — rerun with --apply")
            continue

        # Backup goes OUTSIDE the dataset dir: push_to_hub only ignores images/
        # and videos/, so anything left in meta/ would be uploaded to the Hub.
        backup = root.parent / f"{root.name}_repair-backup" / Path(path).relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():  # keep the pristine original, not the last run's output
            shutil.copy2(path, backup)
        # Replace just the one column per video key so the other 106 columns —
        # including the nested stats/tasks structs — round-trip untouched.
        for name, values in patched.items():
            idx = table.schema.get_field_index(name)
            table = table.set_column(idx, table.schema.field(idx), [values])
        pq.write_table(table, path)
        print(f"\nwrote {path.replace(str(root), '.')}")
        print(f"original saved to {backup}")

    if total_bad == 0:
        print("all episodes consistent — nothing to repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
