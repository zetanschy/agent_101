#!/usr/bin/env python3
"""Render a DAgger episode to MP4, with the handovers drawn on it.

    ./robot dagger-video --dataset zetanschy/rollout_cap_to_cup_dagger
    ./robot dagger-video --dataset ... --episode 3 --out outputs/dagger/ep3.mp4

A DAgger dataset is a normal LeRobotDataset plus one bool column, `intervention`, and
that column is the whole story of the session: which stretches the policy drove and
which ones you took. It is also the thing you cannot see in the raw video, because a
correction looks exactly like autonomous execution -- the same arm moving through the
same scene. So this burns it in:

  * a banner across the top, green while the POLICY drives and amber while YOU do;
  * a timeline along the bottom with every intervention of the episode marked, and a
    playhead, so a glance says how often you had to step in and where;
  * both cameras side by side, since a takeover usually happens because of what the
    wrist camera saw.

Why this is worth a script rather than a glance at the parquet: after a session you
want to decide which episodes to keep, and "how did that recovery actually look" is the
question. `lerobot-dataset-viz` shows the data; this shows the handover.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess

import numpy as np

# Colours are the two states, everywhere: green = policy, amber = human.
POLICY = (46, 160, 67)
HUMAN = (219, 143, 0)
BANNER_H = 44
TIMELINE_H = 26
PAD = 8


def _font(size: int):
  from PIL import ImageFont

  for path in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  ):
    if pathlib.Path(path).exists():
      return ImageFont.truetype(path, size)
  return ImageFont.load_default()


def frame_image(dataset, index: int, cameras: list[str]) -> np.ndarray:
  """The cameras of one frame, side by side, as uint8 HWC."""
  item = dataset[index]
  panels = []
  for cam in cameras:
    img = item[cam]
    array = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if array.ndim == 3 and array.shape[0] in (1, 3):  # CHW -> HWC
      array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:  # lerobot hands back float [0,1]
      array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    panels.append(array)
  height = max(p.shape[0] for p in panels)
  panels = [
    np.pad(p, ((0, height - p.shape[0]), (0, 0), (0, 0)), constant_values=0) for p in panels
  ]
  return np.concatenate(panels, axis=1)


def compose(base: np.ndarray, *, intervention: bool, flags: np.ndarray, position: int,
            episode: int, cameras: list[str]) -> np.ndarray:
  """Draw the banner and the timeline around one frame."""
  from PIL import Image, ImageDraw

  h, w = base.shape[:2]
  canvas = Image.new("RGB", (w, BANNER_H + h + TIMELINE_H + PAD), (17, 17, 17))
  canvas.paste(Image.fromarray(base), (0, BANNER_H))
  draw = ImageDraw.Draw(canvas)

  colour = HUMAN if intervention else POLICY
  label = "HUMAN CORRECTION  (teleop)" if intervention else "AUTONOMOUS  (policy)"
  draw.rectangle([0, 0, w, BANNER_H], fill=colour)
  big, small = _font(22), _font(14)
  draw.text((PAD, 10), label, fill=(255, 255, 255), font=big)

  # The counter goes right-aligned, and ONLY if it clears the label -- at 2x320 the two
  # used to print on top of each other, which is worse than not showing the frame number.
  right = f"ep {episode}   {position + 1}/{len(flags)}"
  left_end = PAD + draw.textlength(label, font=big)
  if w - PAD - draw.textlength(right, font=small) > left_end + PAD:
    draw.text((w - PAD - draw.textlength(right, font=small), 15), right,
              fill=(255, 255, 255), font=small)

  # Camera names belong on their panels, not in the banner: short, and where the eye
  # already is when it wonders which view it is looking at.
  x = 0
  for camera in cameras:
    name = camera.rsplit(".", 1)[-1]
    width = base.shape[1] // max(len(cameras), 1)
    draw.text((x + PAD, BANNER_H + PAD), name, fill=(235, 235, 235), font=small,
              stroke_width=2, stroke_fill=(0, 0, 0))
    x += width

  # The timeline: one column per frame, amber where you were driving.
  top = BANNER_H + h + PAD // 2
  draw.rectangle([0, top, w, top + TIMELINE_H], fill=(38, 38, 38))
  for x in range(w):
    i = min(int(x / max(w - 1, 1) * (len(flags) - 1)), len(flags) - 1)
    if flags[i]:
      draw.line([(x, top), (x, top + TIMELINE_H)], fill=HUMAN)
  playhead = int(position / max(len(flags) - 1, 1) * (w - 1))
  draw.line([(playhead, top - 3), (playhead, top + TIMELINE_H + 3)], fill=(255, 255, 255), width=2)
  return np.asarray(canvas)


def episode_range(dataset, episode: int) -> tuple[int, int]:
  """[from, to) row indices of one episode.

  `episode_data_index` is the classic accessor and le101 0.6.1 no longer has it, so the
  fallback reads the episode_index column, which every version writes.
  """
  index = getattr(dataset, "episode_data_index", None)
  if index is not None:
    return int(index["from"][episode]), int(index["to"][episode])
  column = np.asarray(dataset.hf_dataset["episode_index"])
  rows = np.flatnonzero(column == episode)
  if rows.size == 0:
    raise SystemExit(f"episode {episode} is not in this dataset")
  return int(rows[0]), int(rows[-1]) + 1


def render(dataset, episode: int, out: pathlib.Path, fps: int, cameras: list[str]) -> int:
  """Write one episode's MP4. Returns the number of frames drawn."""
  from_idx, to_idx = episode_range(dataset, episode)
  flags = np.array(
    [bool(np.asarray(dataset[i]["intervention"]).reshape(-1)[0]) for i in range(from_idx, to_idx)]
  )

  out.parent.mkdir(parents=True, exist_ok=True)
  first = compose(frame_image(dataset, from_idx, cameras), intervention=flags[0], flags=flags,
                  position=0, episode=episode, cameras=cameras)
  h, w = first.shape[:2]
  # Piped to ffmpeg rather than encoded in-process: one dependency fewer, and the
  # yuv420p/even-dimension dance is what every player expects.
  proc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{w - w % 2}x{h - h % 2}", "-r", str(fps), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
    stdin=subprocess.PIPE,
  )
  try:
    for position, i in enumerate(range(from_idx, to_idx)):
      image = compose(frame_image(dataset, i, cameras), intervention=flags[position],
                      flags=flags, position=position, episode=episode, cameras=cameras)
      proc.stdin.write(image[: h - h % 2, : w - w % 2].tobytes())
  finally:
    proc.stdin.close()
    proc.wait()
  return len(flags)


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--dataset", required=True, help="repo_id of a DAgger dataset")
  p.add_argument("--root", default=None, help="its root, if not the HF cache")
  p.add_argument("--episode", type=int, default=None, help="default: every episode")
  p.add_argument("--out", default="outputs/dagger", help="file, or directory for several")
  p.add_argument("--cameras", nargs="+", default=None, help="default: all of them")
  args = p.parse_args(argv)

  from lerobot.datasets.lerobot_dataset import LeRobotDataset

  dataset = LeRobotDataset(args.dataset, root=args.root)
  if "intervention" not in dataset.features:
    raise SystemExit(
      f"{args.dataset} has no `intervention` column, so it is not a DAgger dataset -- "
      "nothing to draw. Record one with ./robot dagger."
    )
  if dataset.num_episodes == 0:
    raise SystemExit(f"{args.dataset} has no episodes yet")

  cameras = args.cameras or [k for k in dataset.features if k.startswith("observation.images.")]
  fps = int(dataset.fps)
  episodes = [args.episode] if args.episode is not None else list(range(dataset.num_episodes))
  out = pathlib.Path(args.out)

  for episode in episodes:
    target = out if out.suffix == ".mp4" else out / f"{args.dataset.split('/')[-1]}_ep{episode}.mp4"
    frames = render(dataset, episode, target, fps, cameras)
    start, stop = episode_range(dataset, episode)
    human = sum(
      bool(np.asarray(dataset[i]["intervention"]).reshape(-1)[0]) for i in range(start, stop)
    )
    print(f"{target}  {frames} frames, {human} of them yours "
          f"({100 * human / max(frames, 1):.0f}%), {frames / fps:.1f}s", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
