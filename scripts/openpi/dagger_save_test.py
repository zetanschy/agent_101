#!/usr/bin/env python3
"""The dataset half of DAgger: does a recorded session survive being written?

    ./robot dagger-save-test        # in the openpi image; no arm, no policy, no GPU

Everything else about dagger.py can be checked by reading it. This cannot: a session
writes frames for twenty minutes and you find out whether they are readable afterwards.
The first version of dagger.py did NOT wrap recording in VideoEncodingManager, so it
would have written frames and left a dataset that could not be opened at all -- finalize
never ran, so no parquet, no metadata, and LeRobotDataset falls through to the Hub
looking for what should have been on disk. Twenty minutes of corrections, gone at the
moment you close the session.

So this drives the same lifecycle dagger.py drives, with the same feature dict and the
same create() arguments, and then reopens the result and looks at it:

    create -> add_frame* -> save_episode -> (a second episode)
           -> add_frame* -> clear_episode_buffer   (left arrow: a failed attempt)
           -> finalize via VideoEncodingManager -> reopen -> read the frames back

What it asserts is what a training run needs: the episodes that were kept are there, the
one that was discarded is not, and `intervention` comes back per frame exactly as it
went in -- that column is the difference between a DAgger dataset and a recording.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
H, W = 64, 96  # small on purpose: this is about the plumbing, not the pixels
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _dagger():
  """dagger.py itself, for the feature dict it will really write."""
  spec = importlib.util.spec_from_file_location("agent101_dagger", HERE / "dagger.py")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _features():
  """The schema dagger.py builds: the robot's own features plus `intervention`."""
  from lerobot.utils.constants import ACTION, OBS_STR
  from lerobot.utils.feature_utils import hw_to_dataset_features

  observation = {f"{m}.pos": float for m in MOTORS} | {"front": (H, W, 3), "grip": (H, W, 3)}
  action = {f"{m}.pos": float for m in MOTORS}
  features = {
    **hw_to_dataset_features(observation, OBS_STR),
    **hw_to_dataset_features(action, ACTION),
  }
  features["intervention"] = dict(_dagger().INTERVENTION_FEATURE)
  return features


def _frame(step: int, intervention: bool) -> dict:
  q = np.full(6, float(step), dtype=np.float32)
  return {
    "observation.state": q,
    "observation.images.front": np.full((H, W, 3), step % 256, np.uint8),
    "observation.images.grip": np.full((H, W, 3), (step * 3) % 256, np.uint8),
    "action": q + (1.0 if intervention else 0.0),
    "task": "Put the cap into the red cup",
    "intervention": np.array([intervention], dtype=bool),
  }


def record_session(root: pathlib.Path, repo_id: str) -> tuple[list[list[bool]], range]:
  """Write two kept episodes and one discarded one.

  Returns the kept episodes' flags and the range of `observation.state` values the
  discarded episode carried, so a test can look for them rather than assume where they
  would have landed.
  """
  from lerobot.datasets import VideoEncodingManager
  from lerobot.datasets.lerobot_dataset import LeRobotDataset

  dataset = LeRobotDataset.create(
    repo_id, fps=30, features=_features(), root=str(root),
    robot_type="so101_follower", use_videos=True,
    image_writer_processes=0, image_writer_threads=8,
  )

  # Episode 0: the policy drives, you take over in the middle, it drives again.
  # Episode 1: all yours. Episode 2: recorded, then thrown away (left arrow).
  kept = [
    [False] * 10 + [True] * 6 + [False] * 4,
    [True] * 8,
  ]
  discarded = [False] * 5 + [True] * 5

  step = 0
  with VideoEncodingManager(dataset):
    for flags in kept:
      for intervention in flags:
        dataset.add_frame(_frame(step, intervention))
        step += 1
      dataset.save_episode()
    dropped_from = step
    for intervention in discarded:
      dataset.add_frame(_frame(step, intervention))
      step += 1
    dataset.clear_episode_buffer()
  return kept, range(dropped_from, step)


def reopen(root: pathlib.Path, repo_id: str):
  from lerobot.datasets.lerobot_dataset import LeRobotDataset

  return LeRobotDataset(repo_id, root=str(root))


def test_a_recorded_session_can_be_reopened():
  """The one that would have caught the missing finalize."""
  with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp) / "ds"
    repo_id = "test/rollout_dagger_save"
    kept, _ = record_session(root, repo_id)
    dataset = reopen(root, repo_id)  # this is what used to fail, at the Hub

    assert dataset.num_episodes == len(kept), dataset.num_episodes
    assert dataset.num_frames == sum(len(f) for f in kept), dataset.num_frames
    assert "intervention" in dataset.features


def test_the_discarded_episode_left_nothing_behind():
  """A failed attempt teaches a trajectory that did not work; it must not survive."""
  with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp) / "ds"
    repo_id = "test/rollout_dagger_discard"
    kept, dropped = record_session(root, repo_id)
    dataset = reopen(root, repo_id)

    assert dataset.num_episodes == 2, f"the discarded episode was kept: {dataset.num_episodes}"
    assert dataset.num_frames == sum(len(f) for f in kept)
    # Each frame's state carries its step number, so the discarded ones are findable.
    states = {float(np.asarray(dataset[i]["observation.state"]).reshape(-1)[0])
              for i in range(dataset.num_frames)}
    survivors = states & set(map(float, dropped))
    assert not survivors, f"frames of the discarded episode survived: {sorted(survivors)}"


def test_intervention_round_trips_per_frame():
  """Per frame, not per episode: the column is the label a training run weights on."""
  with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp) / "ds"
    repo_id = "test/rollout_dagger_flags"
    kept, _ = record_session(root, repo_id)
    dataset = reopen(root, repo_id)

    expected = [flag for episode in kept for flag in episode]
    got = [bool(np.asarray(dataset[i]["intervention"]).reshape(-1)[0])
           for i in range(dataset.num_frames)]
    assert got == expected, f"first mismatch at {next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)}"


def test_the_video_is_really_there():
  """use_videos=True means the frames live in an encoded file, not just in the parquet."""
  with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp) / "ds"
    repo_id = "test/rollout_dagger_video"
    record_session(root, repo_id)
    videos = list(root.rglob("*.mp4"))
    assert videos, f"no video written under {root}"
    assert all(v.stat().st_size > 0 for v in videos)

    dataset = reopen(root, repo_id)
    frame = dataset[0]["observation.images.front"]
    array = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    assert array.size > 0 and array.ndim == 3, array.shape


def main() -> int:
  shutil.rmtree("/tmp/lerobot_dagger_save_test", ignore_errors=True)
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  for test in tests:
    test()
    print(f"ok  {test.__name__}", flush=True)
  print(f"\n{len(tests)} passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
