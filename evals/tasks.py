"""The benchmarks this bench runs, embodiment-agnostic by construction.

A Task in Inspect Robots is scenes + scorer + horizon and knows nothing about the
robot, which is the property the whole comparison rests on: the same task object is
handed to the LLM agent and to the fine-tuned VLA, so a difference in score is a
difference in policy and not in benchmark.

WHY THE INSTRUCTION IS WORDED THE WAY IT IS. "Put the cap into the red cup" is not a
description of a pick-and-place task in general, it is verbatim the instruction
zetanschy/pi05_lora_cap_tu_cup was fine-tuned on -- the DEFAULT_POLICY in webui/app.py
and the task string scripts/robot/infer.sh defaults to. A VLA is only obliged to
follow the instruction it was trained on, so changing a word here would hand the
comparison to the LLM agent by default. If the benchmark is ever retargeted at a
different manipuland, the pi0.5 side needs retraining, not rewording.

SCORING IS BY OPERATOR, because this rig has no success detector. The arm reports
joint positions and two camera streams; nothing on it can tell whether a cap ended up
in a cup. `success_at_end` would be worse than useless here -- it counts only
embodiment-detected "success" terminations, so it scores every operator-ended trial as
a failure. A person watches and answers y/n/partial per trial.
"""

from __future__ import annotations

from inspect_robots import Scene, Task, operator_scorer

# The pi0.5 checkpoint's own instruction. See the module docstring before editing.
INSTRUCTION = "Put the cap into the red cup"

# 20 s at 30 Hz is 600 steps. The horizon is in SECONDS rather than steps because the
# two policies step at wildly different rates -- a chunked VLA runs at camera cadence
# while an LLM agent waits on a frontier model for every action -- and a step budget
# would silently hand one of them many times more wall-clock than the other.
DEFAULT_SECONDS = 120.0


def pick_and_place(
  layouts: int = 5,
  *,
  seconds: float = DEFAULT_SECONDS,
  epochs: int = 1,
) -> Task:
  """Put the cap into the red cup, from several starting layouts.

  Each scene is one arrangement the OPERATOR sets up by hand before the trial: the
  scene's instruction is what the policy is told, and its metadata is what the person
  running the bench is told. There is no setup hook because nothing on this rig can
  place a cap.

  `layouts` scenes x `epochs` trials each. Keep the product small at first: every
  trial is a physical reset and a human verdict, so 5 x 1 is a 20-minute session and
  5 x 3 is an hour.
  """
  places = (
    ("near", "cap ~15 cm in front of the base, cup to its right"),
    ("left", "cap left of centre, cup centre-right"),
    ("right", "cap right of centre, cup centre-left"),
    ("far", "cap at the far edge of the reachable area, cup near"),
    ("close-pair", "cap and cup within 10 cm of each other, centre"),
  )
  if not 1 <= layouts <= len(places):
    raise ValueError(f"layouts must be 1..{len(places)}, got {layouts}")
  return Task(
    name="so101-cap-to-cup",
    scenes=[
      Scene(
        id=f"layout-{name}",
        instruction=INSTRUCTION,
        metadata={"operator_setup": description},
      )
      for name, description in places[:layouts]
    ],
    scorer=operator_scorer(),
    max_seconds=seconds,
    epochs=epochs,
  )
