"""Shrink stored eval frames to the size the report actually renders.

    ./robot eval-shrink                    # newest log: encode MP4s, then stride the frames
    ./robot eval-shrink --dry-run          # what it would save, touching nothing
    ./robot eval-shrink --stride 4         # 16x instead of 4x, at a coarser flipbook
    ./robot eval-shrink --watch            # alongside a live session, as trials finish

WHY THERE IS ANYTHING TO SAVE. `store_frames` writes one uncompressed .npy per camera
per control step: 640x480x3 uint8 is 0.88 MiB, twice a step, thirty steps a second --
55 MB/s, or 3.1 GiB per minute of arm time. A ten-trial session is 20-40 GB of raw
pixels, and this bench has filled its disk on one.

WHY STRIDING IS FREE. The report never shows those pixels. `_html.py` sets
_FRAME_MAX_SIDE = 448 and decimates anything larger with `array[::stride, ::stride]` on
EVERY render -- so a 640x480 frame is displayed at 320x240 (stride 2) whether it was
stored that way or not. Doing the same decimation once, on disk, is therefore
byte-identical in the report and four times smaller. Both consumers go through that one
loader: the transcript's inline frames and the flipbook panels alike.

That is the whole idea, and its limit: `--stride 4` saves sixteen times but really does
show a coarser flipbook than today, because 160x120 is below what the report would have
rendered. Stride 2 is the free one.

MP4 FIRST, ALWAYS. `inspect-robots video` encodes from these same files, so a run
shrunk before it is encoded yields a smaller MP4 forever. This runs the encode first
unless told not to; at ~300:1 the video costs almost nothing to keep.

WHAT THIS DOES NOT FIX. Peak usage during a run: the eval writes full-size frames while
it is running, and only `--no-frames` avoids that. `--watch` is the middle ground -- a
trial's frames stop changing the moment the next trial starts, so sweeping them while
the session continues bounds the peak at roughly one trial instead of the whole run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

# inspect_robots/_html.py::_FRAME_MAX_SIDE. A frame whose longest side exceeds this is
# decimated at render time, so storing it larger buys the report nothing.
MAX_SIDE = 448

# A file this new may still be mid-write, or belong to the trial now running. Only
# --watch defaults to caring; a finished run's frames are all settled.
WATCH_MIN_AGE_S = 120.0


def newest_log(log_dir: pathlib.Path) -> pathlib.Path:
  """The most recently written eval log, which is what `./robot eval-video` also picks."""
  logs = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
  if not logs:
    raise SystemExit(f"no eval logs in {log_dir}")
  return logs[0]


def frames_dir_of(log: pathlib.Path) -> pathlib.Path:
  """The frames directory a log points at.

  `stats.frames_dir` is stored relative to the WORKING DIRECTORY of the run, which is
  /workspace for every containerised run here -- so it resolves as-is from the repo
  root, and falls back to a path relative to the log itself for anything moved by hand.
  """
  stats = json.loads(log.read_text()).get("stats", {})
  raw = stats.get("frames_dir")
  if not raw:
    raise SystemExit(f"{log.name} stored no frames (run with frames, or nothing to do)")
  candidates = [pathlib.Path(raw), log.parent / pathlib.Path(raw).name, log.parent.parent / raw]
  for path in candidates:
    if path.is_dir():
      return path
  raise SystemExit(f"{log.name} points at {raw}, which is not a directory here")


def stride_for(shape: tuple[int, ...], stride: int | None, max_side: int) -> int:
  """How much to decimate this frame: the report's own rule unless told otherwise."""
  if stride is not None:
    return max(stride, 1)
  longest = max(shape[0], shape[1])
  return math.ceil(longest / max_side) if longest > max_side else 1


def shrink_file(path: pathlib.Path, stride: int | None, max_side: int, dry_run: bool):
  """Decimate one frame in place. Returns (before, after) bytes; equal means untouched.

  Written to a sibling temp file and renamed, so an interrupted sweep leaves either the
  old frame or the new one and never a half-written array the report would skip.
  """
  before = path.stat().st_size
  try:
    array = np.load(path, allow_pickle=False)
  except Exception:  # noqa: BLE001 - a corrupt frame is the report's problem, not ours
    return before, before
  if array.dtype != np.uint8 or array.ndim < 2:
    return before, before
  step = stride_for(array.shape, stride, max_side)
  if step <= 1:
    return before, before  # already at or below what the report renders: idempotent
  smaller = np.ascontiguousarray(array[::step, ::step])
  after = smaller.nbytes + 128  # the .npy header
  if dry_run:
    return before, after
  tmp = path.with_suffix(".npy.tmp")
  try:
    with tmp.open("wb") as fh:
      np.save(fh, smaller, allow_pickle=False)
      fh.flush()
      os.fsync(fh.fileno())
    os.replace(tmp, path)
  finally:
    tmp.unlink(missing_ok=True)
  return before, path.stat().st_size


def sweep(frames: pathlib.Path, *, stride, max_side, dry_run, min_age_s, quiet=False):
  """One pass over a frames directory. Returns (files, before, after)."""
  now = time.time()
  touched = before_total = after_total = 0
  skipped_young = 0
  for path in sorted(frames.glob("*.npy")):
    try:
      if min_age_s and now - path.stat().st_mtime < min_age_s:
        skipped_young += 1
        continue
    except FileNotFoundError:
      continue
    before, after = shrink_file(path, stride, max_side, dry_run)
    before_total += before
    after_total += after
    if after != before:
      touched += 1
  if not quiet:
    verb = "would shrink" if dry_run else "shrank"
    print(
      f"{frames.name}: {verb} {touched} frame(s), "
      f"{before_total / 2**30:.2f} -> {after_total / 2**30:.2f} GiB"
      + (f"  ({skipped_young} too new)" if skipped_young else "")
    )
  return touched, before_total, after_total


def encode_video(log: pathlib.Path) -> None:
  """Full-size MP4s first: the encoder reads the same frames this is about to shrink."""
  import subprocess

  print(f"encoding MP4s from {log.name} before shrinking ...", flush=True)
  result = subprocess.run(
    ["inspect-robots", "video", str(log)], check=False, capture_output=True, text=True
  )
  sys.stdout.write(result.stdout)
  if result.returncode != 0:
    # Not fatal: the frames are still there and still shrinkable. But say so loudly,
    # because the point of encoding first is that it cannot be done properly after.
    sys.stderr.write(result.stderr)
    raise SystemExit(
      "video encode failed; NOT shrinking (re-run with --no-video to shrink anyway)"
    )


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("target", nargs="?", default=None,
                 help="an eval log (.json) or a frames directory; default: the newest log")
  p.add_argument("--log-dir", default="outputs/evals")
  p.add_argument("--stride", type=int, default=None,
                 help="decimate by this factor; default is the report's own rule "
                      f"(longest side down to {MAX_SIDE}, i.e. 2 for 640x480)")
  p.add_argument("--max-side", type=int, default=MAX_SIDE,
                 help=f"the size the report renders at (default {MAX_SIDE})")
  p.add_argument("--video", action=argparse.BooleanOptionalAction, default=None,
                 help="encode MP4s first (default: yes for a log, no for a bare "
                      "directory or --watch)")
  p.add_argument("--dry-run", action="store_true", help="report the saving, touch nothing")
  p.add_argument("--watch", action="store_true",
                 help="keep sweeping while a session runs; Ctrl-C to stop")
  p.add_argument("--interval", type=float, default=60.0, help="--watch: seconds between sweeps")
  p.add_argument("--min-age", type=float, default=None,
                 help="skip frames written less than this many seconds ago "
                      f"(default {WATCH_MIN_AGE_S:g} under --watch, 0 otherwise)")
  args = p.parse_args(argv)

  target = pathlib.Path(args.target) if args.target else newest_log(pathlib.Path(args.log_dir))
  log = target if target.suffix == ".json" else None
  frames = frames_dir_of(log) if log else target
  if not frames.is_dir():
    raise SystemExit(f"{frames} is not a directory")

  min_age = args.min_age if args.min_age is not None else (WATCH_MIN_AGE_S if args.watch else 0.0)
  want_video = args.video if args.video is not None else (log is not None and not args.watch)
  if want_video:
    if log is None:
      raise SystemExit("--video needs a log to encode from, not a bare frames directory")
    if not args.dry_run:
      encode_video(log)

  print(f"frames   : {frames}")
  print(f"target   : {'stride ' + str(args.stride) if args.stride else f'longest side <= {args.max_side}'}")
  if args.dry_run:
    print("mode     : DRY RUN, nothing is written")

  if not args.watch:
    _, before, after = sweep(frames, stride=args.stride, max_side=args.max_side,
                             dry_run=args.dry_run, min_age_s=min_age)
    if before:
      print(f"saved    : {(before - after) / 2**30:.2f} GiB "
            f"({before / max(after, 1):.1f}x smaller)")
    return 0

  print(f"watching every {args.interval:g}s, skipping frames newer than {min_age:g}s "
        "(Ctrl-C to stop)")
  total = 0
  try:
    while True:
      touched, before, after = sweep(frames, stride=args.stride, max_side=args.max_side,
                                     dry_run=args.dry_run, min_age_s=min_age, quiet=True)
      total += before - after
      if touched:
        print(f"[{time.strftime('%H:%M:%S')}] {touched} frame(s), "
              f"{(before - after) / 2**30:.2f} GiB this pass, {total / 2**30:.2f} GiB total",
              flush=True)
      time.sleep(args.interval)
  except KeyboardInterrupt:
    print(f"\nstopped; {total / 2**30:.2f} GiB reclaimed")
  return 0


if __name__ == "__main__":
  sys.exit(main())
