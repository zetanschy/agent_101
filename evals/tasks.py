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

TWO BENCHMARKS, ONE INSTRUCTION. `pick_and_place` pins both objects per scene and
asks whether the skill works at all; `cap_quadrants` pins the cap to one lower table
quadrant and has the operator re-draw the CUP before every trial, which turns a scene's
epochs into a success rate over cup positions instead of a repeat of one arrangement.
Both carry the same INSTRUCTION and the same scorer, so their scores are comparable and
neither favours a policy by wording.

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


def _horizon(seconds: float | None, steps: int | None) -> dict[str, float | int]:
  """Resolve the one horizon a Task is allowed to declare.

  Task refuses both and refuses neither, so a caller that passes neither gets the
  seconds default and a caller that passes both gets steps -- the tighter, policy-
  derived limit -- rather than an error from inside the framework.
  """
  if (seconds is None) == (steps is None):
    seconds = DEFAULT_SECONDS if steps is None else None
  return {"max_seconds": seconds} if seconds is not None else {"max_steps": steps}


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
    **_horizon(seconds, steps),
  )


# The quadrant the cap sits in, and how the cup is drawn for every trial of it. The
# frame is the OPERATOR'S, standing behind the arm and looking down the table -- the
# same view the `front` camera has -- so "lower right" is the near-right corner of the
# reachable area, not the far one. Stated here because it is the only thing that keeps
# ten hand-set trials in one frame of reference, and because a mirrored setup would
# read as a policy failure.
QUADRANTS = (
  ("lower-right", "near-right"),
  ("lower-left", "near-left"),
)

# How far the cup has to stay from the cap. Under ~10 cm the two objects are one blob
# in the front camera and a policy can succeed without ever having localised the cup,
# which is the thing this benchmark is asking about.
CUP_CLEARANCE_CM = 10


def cap_quadrants(
  layouts: int = 2,
  *,
  seconds: float | None = None,
  steps: int | None = None,
  epochs: int = 5,
) -> Task:
  """Put the cap into the red cup, cap pinned to one lower quadrant, cup random.

  The SAME instruction and the same scorer as pick_and_place -- see the module
  docstring; the pi0.5 checkpoint is only obliged to follow the string it was trained
  on, and this task asks a different question about that one skill rather than asking
  for a new one.

  WHAT IT MEASURES, that pick_and_place does not. pick_and_place pins both objects per
  scene, so a policy that has memorised five arrangements scores well on it. Here the
  cap is fixed to a table quadrant and the CUP MOVES EVERY TRIAL, re-placed at a fresh
  random spot by the operator. So the five epochs of one scene are five genuinely
  different scenes to the policy, and the per-scene score becomes a success RATE over
  cup positions rather than a verdict on one arrangement. The two quadrants then split
  that rate by where the reach started from, which is the part of the workspace the arm
  is least symmetric about.

  HOW TO DRAW THE CUP, so that "random" is random and not a preference. Before every
  trial, place the cup anywhere in the reachable area, at least CUP_CLEARANCE_CM from
  the cap, and NOT where it was on the previous trial -- physically move it, do not
  nudge it. Vary distance as well as side; a run whose ten cup spots all landed
  mid-table has measured one cup position five times per quadrant.

  Defaults are the session the bench was designed around: 2 quadrants x 5 epochs = 10
  trials, ~40 minutes with a physical reset and a human verdict on each.
  """
  if not 1 <= layouts <= len(QUADRANTS):
    raise ValueError(f"layouts must be 1..{len(QUADRANTS)}, got {layouts}")
  return Task(
    name="so101-cap-to-cup-quadrants",
    scenes=[
      Scene(
        id=f"quadrant-{name}",
        instruction=INSTRUCTION,
        metadata={
          "operator_setup": (
            f"cap in the {corner} quadrant of the reachable area (operator's view); "
            f"RE-DRAW the red cup before EVERY trial -- anywhere reachable, "
            f">{CUP_CLEARANCE_CM} cm from the cap, never the previous spot"
          )
        },
      )
      for name, corner in QUADRANTS[:layouts]
    ],
    scorer=operator_scorer(),
    epochs=epochs,
    **_horizon(seconds, steps),
  )


# The benchmarks `./robot eval --task` can select, by their CLI name. Each builder
# takes the same (layouts, seconds/steps, epochs) keywords and carries its own
# defaults, so run.py forwards only what the operator actually asked for.
TASKS = {
  "cap-to-cup": pick_and_place,
  "cap-quadrants": cap_quadrants,
}
