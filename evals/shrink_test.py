"""Tests for evals/shrink.py -- because this one deletes pixels irreversibly.

    python -m evals.shrink_test        # needs numpy only; no arm, no logs, no docker

The property that matters is the one the whole idea rests on: shrinking a frame on disk
must produce exactly what the REPORT would have produced from the original, or "free"
is a lie. `_html.py` decimates with `array[::stride, ::stride]` at render time, so the
test decimates the original the same way and compares pixels.

The rest is about not losing data by accident: idempotence, atomicity, and refusing to
touch frames that are still being written.
"""

from __future__ import annotations

import math
import pathlib
import tempfile
import time

import numpy as np

from evals import shrink

H, W = 480, 640


def frame(path: pathlib.Path, h: int = H, w: int = W, seed: int = 0) -> np.ndarray:
  """A frame with structure, so a wrong stride or an axis swap shows up as a mismatch."""
  rng = np.random.default_rng(seed)
  array = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
  np.save(path, array, allow_pickle=False)
  return array


def test_the_shrunk_frame_is_what_the_report_would_have_rendered():
  """The claim: identical pixels, four times smaller, for a 640x480 frame."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "trial_front_000000.npy"
    original = frame(path)
    # What inspect_robots/_html.py::_load_frame does to it at render time.
    stride = math.ceil(max(original.shape[:2]) / shrink.MAX_SIDE)
    rendered = original[::stride, ::stride]

    before, after = shrink.shrink_file(path, None, shrink.MAX_SIDE, dry_run=False)

    stored = np.load(path)
    assert stored.shape == rendered.shape == (240, 320, 3), stored.shape
    np.testing.assert_array_equal(stored, rendered)
    assert before / after > 3.9, f"expected ~4x, got {before / after:.2f}x"


def test_a_second_pass_changes_nothing():
  """Idempotent: the report's own rule is already satisfied, so leave it alone."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "trial_front_000000.npy"
    frame(path)
    shrink.shrink_file(path, None, shrink.MAX_SIDE, dry_run=False)
    once = path.read_bytes()
    before, after = shrink.shrink_file(path, None, shrink.MAX_SIDE, dry_run=False)
    assert before == after, "a second pass shrank an already-shrunk frame"
    assert path.read_bytes() == once


def test_an_explicit_stride_is_honoured_and_still_matches_decimation():
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "trial_grip_000003.npy"
    original = frame(path, seed=3)
    shrink.shrink_file(path, 4, shrink.MAX_SIDE, dry_run=False)
    np.testing.assert_array_equal(np.load(path), original[::4, ::4])


def test_dry_run_writes_nothing_but_reports_the_saving():
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "trial_front_000000.npy"
    original = frame(path)
    before, after = shrink.shrink_file(path, None, shrink.MAX_SIDE, dry_run=True)
    assert after < before
    np.testing.assert_array_equal(np.load(path), original), "dry run modified the frame"


def test_a_small_frame_is_left_exactly_alone():
  """Already at or below what the report renders -- and a stride of 1 would rewrite it
  for nothing, which on a 25,000-file directory is 25,000 pointless writes."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "trial_front_000000.npy"
    original = frame(path, h=240, w=320)
    stat_before = path.stat()
    before, after = shrink.shrink_file(path, None, shrink.MAX_SIDE, dry_run=False)
    assert before == after
    assert path.stat().st_mtime_ns == stat_before.st_mtime_ns, "rewrote an untouched frame"
    np.testing.assert_array_equal(np.load(path), original)


def test_a_frame_that_is_not_a_frame_is_skipped_rather_than_destroyed():
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    junk = d / "trial_front_000001.npy"
    junk.write_bytes(b"not an npy at all")
    floats = d / "trial_front_000002.npy"
    np.save(floats, np.zeros((H, W, 3), dtype=np.float32), allow_pickle=False)

    for path in (junk, floats):
      raw = path.read_bytes()
      before, after = shrink.shrink_file(path, None, shrink.MAX_SIDE, dry_run=False)
      assert before == after
      assert path.read_bytes() == raw, f"{path.name} was modified"


def test_sweep_skips_frames_that_are_too_new():
  """A trial still being written must not be shrunk under the eval's feet."""
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    old, new = d / "trial_front_000000.npy", d / "trial_front_000001.npy"
    frame(old)
    frame(new, seed=1)
    stale = time.time() - 600
    import os

    os.utime(old, (stale, stale))

    touched, _, _ = shrink.sweep(d, stride=None, max_side=shrink.MAX_SIDE,
                                 dry_run=False, min_age_s=120, quiet=True)
    assert touched == 1
    assert np.load(old).shape == (240, 320, 3)
    assert np.load(new).shape == (H, W, 3), "a frame written seconds ago was shrunk"


def test_sweep_leaves_mp4s_and_everything_else_alone():
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    frame(d / "trial_front_000000.npy")
    video = d / "trial_front.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42 not really")
    raw = video.read_bytes()
    shrink.sweep(d, stride=None, max_side=shrink.MAX_SIDE, dry_run=False, min_age_s=0, quiet=True)
    assert video.read_bytes() == raw


def test_no_temp_files_are_left_behind():
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    for i in range(3):
      frame(d / f"trial_front_{i:06d}.npy", seed=i)
    shrink.sweep(d, stride=None, max_side=shrink.MAX_SIDE, dry_run=False, min_age_s=0, quiet=True)
    assert not list(d.glob("*.tmp")), [p.name for p in d.glob("*.tmp")]


def main() -> int:
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
