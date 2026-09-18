#!/usr/bin/env python3
"""Tests for scripts/openpi/dagger_weights.py — the DAgger sampling index.

    ./robot run python scripts/openpi/dagger_weights_test.py    # no GPU, no arm

Every test here guards a way the index can be wrong WITHOUT ANYTHING LOOKING WRONG.
That is the whole hazard of this file: a bad weight rule or a bad rounding scheme
still produces an index, training still runs to completion, and the only symptom is
a policy that did not improve as much as it should have. So the properties checked
are the ones with no downstream alarm -- the ramp not crossing an episode boundary,
the rounding not preferring low frame indices, the cache noticing that the labels
underneath it changed.

Synthetic datasets rather than fixtures: they are five lines to build, and a real
DAgger dataset is 12k frames of video nobody needs to read to test arithmetic.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dagger_weights as dw  # noqa: E402


def make_dataset(root: pathlib.Path, episodes: list[list[bool]], fps: float = 30.0):
  """A minimal LeRobot v3.0 dataset: meta/info.json plus one parquet shard.

  `episodes` is one list of per-frame operator flags per episode.
  """
  import pyarrow as pa
  import pyarrow.parquet as pq

  flags = [f for ep in episodes for f in ep]
  ep_index = [i for i, ep in enumerate(episodes) for _ in ep]
  total = len(flags)

  (root / "meta").mkdir(parents=True, exist_ok=True)
  (root / "meta" / "info.json").write_text(json.dumps({
    "codebase_version": "v3.0",
    "fps": fps,
    "total_frames": total,
    "total_episodes": len(episodes),
    "features": {"intervention": {"dtype": "bool", "shape": [1]}},
  }))
  (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
  pq.write_table(pa.table({
    "index": pa.array(range(total), pa.int64()),
    "episode_index": pa.array(ep_index, pa.int64()),
    "intervention": pa.array(flags, pa.bool_()),
  }), root / "data" / "chunk-000" / "file-000.parquet")
  return root


def tmpdir() -> pathlib.Path:
  return pathlib.Path(tempfile.mkdtemp(prefix="dagger_weights_test_"))


# --------------------------------------------------------------------------- #
# The weight rule
# --------------------------------------------------------------------------- #


def test_operator_frames_take_human_weight_and_the_rest_take_auto():
  root = make_dataset(tmpdir(), [[False] * 10 + [True] * 5])
  cfg = dw.DaggerConfig(human_weight=1.0, auto_weight=0.5, pre_window_s=0.0)
  w, is_human, _ = dw.frame_weights(root, cfg)
  assert np.allclose(w[is_human], 1.0)
  assert np.allclose(w[~is_human], 0.5)
  shutil.rmtree(root)


def test_the_pre_intervention_ramp_falls_into_the_takeover():
  """The seconds before a takeover are the failure run-up. They must fade OUT,
  not stay at auto_weight -- reinforcing them teaches the policy the approach
  that made a human reach for the leader."""
  root = make_dataset(tmpdir(), [[False] * 60 + [True] * 10], fps=30.0)
  cfg = dw.DaggerConfig(auto_weight=0.5, pre_window_s=1.0, pre_min_weight=0.0)
  w, _, _ = dw.frame_weights(root, cfg)
  # 30 frames = 1 s of window before frame 60.
  assert w[0] == 0.5, "outside the window, untouched"
  assert w[29] == 0.5, "the window edge is still auto_weight"
  assert w[59] < w[45] < w[31], f"the ramp must DECREASE toward the takeover: {w[55:60]}"
  assert w[59] < 0.05, f"the frame at the takeover is nearly zero, got {w[59]}"
  shutil.rmtree(root)


def test_frames_after_the_last_takeover_keep_auto_weight():
  """There is no next takeover to ramp toward, and treating +inf as 'close'
  would silently zero the tail of every episode."""
  root = make_dataset(tmpdir(), [[True] * 5 + [False] * 60])
  cfg = dw.DaggerConfig(auto_weight=0.5, pre_window_s=1.0)
  w, _, _ = dw.frame_weights(root, cfg)
  assert np.allclose(w[5:], 0.5), w[5:10]
  shutil.rmtree(root)


def test_the_ramp_does_not_reach_across_an_episode_boundary():
  """The last seconds of episode 0 are not the run-up to a takeover at the start
  of episode 1: different scene, different attempt. Weighted globally they would
  be, and nothing downstream would say so."""
  root = make_dataset(tmpdir(), [[False] * 60, [True] * 10])
  cfg = dw.DaggerConfig(auto_weight=0.5, pre_window_s=1.0)
  w, _, _ = dw.frame_weights(root, cfg)
  assert np.allclose(w[:60], 0.5), "episode 0 is untouched by episode 1's takeover"
  shutil.rmtree(root)


# --------------------------------------------------------------------------- #
# Quantisation — the part that fails silently
# --------------------------------------------------------------------------- #


def test_expansion_draws_in_proportion_to_weight():
  w = np.array([1.0] * 100 + [0.5] * 100, np.float32)
  cfg = dw.DaggerConfig(epoch_scale=8)
  index = dw.expand_index(w, cfg)
  drawn = np.bincount(index, minlength=200)
  ratio = drawn[:100].sum() / drawn[100:].sum()
  assert 1.9 < ratio < 2.1, f"a 2:1 weight must draw about 2:1, got {ratio:.3f}"


def test_expansion_is_deterministic_for_a_seed():
  w = np.array([1.0, 0.5, 0.33, 0.25] * 50, np.float32)
  a = dw.expand_index(w, dw.DaggerConfig(seed=7))
  b = dw.expand_index(w, dw.DaggerConfig(seed=7))
  c = dw.expand_index(w, dw.DaggerConfig(seed=8))
  assert np.array_equal(a, b), "same seed, same index"
  assert not np.array_equal(a, c), "a different seed must actually move the draw"


def test_rounding_does_not_favour_low_frame_indices():
  """THE BUG THIS EXISTS FOR. Most autonomous frames carry the identical weight,
  so a largest-remainder rounding breaks every tie by index and promotes a
  contiguous prefix -- dropping whole stretches of late episodes rather than
  sampling them thinly. A Bernoulli draw on the fraction spreads the loss."""
  w = np.full(4000, 0.37, np.float32)
  index = dw.expand_index(w, dw.DaggerConfig(epoch_scale=1, seed=3))
  drawn = np.bincount(index, minlength=4000)
  halves = drawn[:2000].sum(), drawn[2000:].sum()
  skew = abs(halves[0] - halves[1]) / max(sum(halves), 1)
  assert skew < 0.05, f"draws are skewed across the dataset: {halves}, skew {skew:.3f}"


def test_a_higher_epoch_scale_stops_quantisation_dropping_frames():
  """The index is built ONCE and reused for the whole run, so a frame rounded to
  zero is excluded permanently rather than missed for one epoch."""
  w = np.array([1.0] * 50 + [0.2] * 50, np.float32)
  never = lambda k: int((np.bincount(  # noqa: E731
    dw.expand_index(w, dw.DaggerConfig(epoch_scale=k)), minlength=100) == 0).sum())
  assert never(1) > never(8), f"scale 8 must drop fewer frames: {never(1)} vs {never(8)}"
  assert never(8) == 0, f"scale 8 drops {never(8)} frames of a 5:1 spread"


def test_all_zero_weights_are_refused():
  try:
    dw.expand_index(np.zeros(10, np.float32), dw.DaggerConfig())
  except ValueError as e:
    assert "zero" in str(e)
  else:
    raise AssertionError("an all-zero weighting would sample nothing and must raise")


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


def test_build_caches_the_index_and_reuses_it():
  root = make_dataset(tmpdir(), [[False] * 20 + [True] * 10])
  cfg = dw.DaggerConfig()
  first, _ = dw.build(root, cfg)
  assert (dw.INDEX_CACHE / root.name / dw.INDEX_FILE).is_file()
  second, _ = dw.build(root, cfg)
  assert np.array_equal(first, second)
  shutil.rmtree(root)
  shutil.rmtree(dw.INDEX_CACHE / root.name, ignore_errors=True)


def test_the_index_is_never_written_inside_the_dataset():
  """A LeRobotDataset directory is pushed to the Hub wholesale, and push_to_hub
  uploads whatever it finds. Two derived files were published as part of a
  dataset once; they do not go back in."""
  root = make_dataset(tmpdir(), [[False] * 20 + [True] * 10])
  dw.build(root, dw.DaggerConfig())
  strays = sorted(q.name for q in root.rglob("dagger_index.*"))
  assert not strays, f"the index cache leaked into the dataset: {strays}"
  shutil.rmtree(root)
  shutil.rmtree(dw.INDEX_CACHE / root.name, ignore_errors=True)


def test_relabelling_the_dataset_invalidates_the_cache():
  """Re-record an episode and the operator frames move. A cache keyed on a file's
  name or size would not notice, and the round would train on last week's index."""
  root = tmpdir()
  make_dataset(root, [[False] * 20 + [True] * 10])
  cfg = dw.DaggerConfig()
  first, _ = dw.build(root, cfg)
  shutil.rmtree(root / "data")
  make_dataset(root, [[True] * 20 + [False] * 10])
  second, _ = dw.build(root, cfg)
  assert not np.array_equal(first, second), "the index followed stale labels"
  shutil.rmtree(root)


def test_a_changed_weight_rule_invalidates_the_cache():
  root = make_dataset(tmpdir(), [[False] * 20 + [True] * 10])
  first, _ = dw.build(root, dw.DaggerConfig(human_weight=1.0))
  second, _ = dw.build(root, dw.DaggerConfig(human_weight=4.0))
  assert not np.array_equal(first, second)
  shutil.rmtree(root)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_v2_dataset_is_refused_rather_than_guessed_at():
  root = make_dataset(tmpdir(), [[True] * 5])
  info = json.loads((root / "meta" / "info.json").read_text())
  info["codebase_version"] = "v2.1"
  (root / "meta" / "info.json").write_text(json.dumps(info))
  try:
    dw.read_labels(root)
  except ValueError as e:
    assert "v2.1" in str(e)
  else:
    raise AssertionError("a v2 layout must be refused, not assumed equivalent")
  shutil.rmtree(root)


def test_a_dataset_without_the_intervention_column_says_so():
  root = make_dataset(tmpdir(), [[True] * 5])
  info = json.loads((root / "meta" / "info.json").read_text())
  info["features"] = {}
  (root / "meta" / "info.json").write_text(json.dumps(info))
  try:
    dw.read_labels(root)
  except ValueError as e:
    assert "intervention" in str(e) and "dagger" in str(e).lower()
  else:
    raise AssertionError("a non-DAgger dataset must be named as such")
  shutil.rmtree(root)


def test_invalid_configs_are_refused():
  for kwargs, expect in [
    ({"human_weight": -1.0}, "non-negative"),
    ({"human_weight": 0.0, "auto_weight": 0.0}, "both be zero"),
    ({"pre_window_s": -1.0}, "non-negative"),
    ({"pre_min_weight": 9.0}, "exceed"),
    ({"epoch_scale": 0}, "at least 1"),
  ]:
    try:
      dw.DaggerConfig(**kwargs)
    except ValueError as e:
      assert expect in str(e), f"{kwargs}: {e}"
    else:
      raise AssertionError(f"{kwargs} must be refused")


# --------------------------------------------------------------------------- #
# The environment seam, and the loader wrapper
# --------------------------------------------------------------------------- #


def test_dagger_is_off_unless_the_environment_enables_it():
  assert dw.config_from_env({}) is None
  assert dw.config_from_env({"DAGGER_HUMAN_WEIGHT": "3"}) is None, \
    "a weight alone must not turn weighting on"
  assert dw.config_from_env({"DAGGER_ENABLED": "1"}) is not None


def test_the_environment_maps_onto_every_knob():
  cfg = dw.config_from_env({
    "DAGGER_ENABLED": "1", "DAGGER_HUMAN_WEIGHT": "2.5", "DAGGER_AUTO_WEIGHT": "0.25",
    "DAGGER_PRE_WINDOW_S": "3", "DAGGER_PRE_MIN_WEIGHT": "0.1",
    "DAGGER_EPOCH_SCALE": "4", "DAGGER_SEED": "11",
  })
  assert (cfg.human_weight, cfg.auto_weight) == (2.5, 0.25)
  assert (cfg.pre_window_s, cfg.pre_min_weight) == (3.0, 0.1)
  assert (cfg.epoch_scale, cfg.seed) == (4, 11)


def test_enabling_dagger_without_a_dataset_is_an_error():
  try:
    dw.install_if_configured({"DAGGER_ENABLED": "1"})
  except ValueError as e:
    assert "DAGGER_REPO_ID" in str(e)
  else:
    raise AssertionError("weighting with no dataset named must not silently no-op")


def test_the_wrapper_reads_the_base_through_the_index():
  base = ["a", "b", "c"]
  view = dw.IndexRemapDataset(base, np.array([2, 2, 0], np.int64))
  assert len(view) == 3
  assert [view[i] for i in range(3)] == ["c", "c", "a"]


def main() -> int:
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
