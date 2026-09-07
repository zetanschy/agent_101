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

# SECONDS ARE NOT WALL-CLOCK SECONDS. eval() converts max_seconds to a step count
# using the embodiment's DECLARED control_hz (30), so 120 s means 3600 steps -- true
# for the chunked VLA, which really does run near camera cadence, and badly wrong for
# the LLM agent, which measured 5.5 steps/s and would have taken 11 minutes to spend a
# "120 second" budget. That mismatch cut the first Astra trial off at step 1685 of
# 3600 and made it look like a failure rather than an interruption.
#
# So the horizon is chosen per policy in evals/run.py: seconds for the VLA, and for the
# agent a step budget derived from its DECISION budget, which is the quantity that
# actually governs both its runtime and its bill.
DEFAULT_SECONDS = 120.0

# Control steps to allow per LLM decision. Measured on the first Astra trial: 1685
# steps over 38 calls is 44 on average, with single moves interpolated over as many as
# 190. 60 leaves headroom without letting the step horizon become the binding limit --
# the decision budget should be what ends a trial, because that is the one the agent is
# told about and can plan against.
STEPS_PER_DECISION = 60


def pick_and_place(
  layouts: int = 5,
  *,
  seconds: float | None = None,
  steps: int | None = None,
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

  Exactly one of `seconds` or `steps` sets the horizon, and which one you want depends
  on the policy -- see the DEFAULT_SECONDS note.
  """
  if (seconds is None) == (steps is None):
    seconds = DEFAULT_SECONDS if steps is None else None
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
    epochs=epochs,
    **({"max_seconds": seconds} if seconds is not None else {"max_steps": steps}),
  )
