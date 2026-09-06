# Evals — Inspect Robots on the SO-ARM101

[Inspect Robots](https://github.com/robocurve/inspect-robots) is an eval harness for
physical AI: define a benchmark once, then run any policy (an LLM agent, a VLA)
against any compatible embodiment. This directory wires it to this bench through
[inspect-robots-so101](https://github.com/robocurve/inspect-robots-so101), which
provides both halves of the SO-ARM + lerobot stack.

The first question it exists to answer: **can a frontier LLM driving the arm through
tool calls do pick-and-place, and how does that compare with the π0.5 checkpoint
fine-tuned for it?**

```bash
./robot eval --policy agent --dry-run                        # no arm, no motion
./robot eval-preflight --dry-run                             # prove the contract
./robot eval --policy agent --model openai/gpt-6-astra       # the LLM agent
./robot eval-openpi                                          # the working pi0.5
```

Both runs get the **same `Task` object** and the **same embodiment**, which is the
only reason their scores can be put side by side.

## The two policies, and why one needs its own container

| | policy | checkpoint | image | units |
|---|---|---|---|---|
| LLM agent | `agent` | `openai/gpt-6-astra` | `lerobot` | degrees |
| π0.5 | `openpi` | `/checkpoints/openpi_pi05_lora_cap_to_cup_200` | `openpi` | degrees |

The working π0.5 on this bench is an **openpi orbax checkpoint** (`params/` +
`assets/`), not a lerobot one — so `inspect-robots-so101`'s own `LeRobotPolicy` cannot
load it, and [openpi_policy.py](openpi_policy.py) wraps openpi's JAX policy as an
Inspect Robots `Policy` instead. It reuses `scripts/openpi/evaluate.py` by path for the
observation layout and config discovery, exactly as `webui/openpi_worker.py` does, so
there is one definition of an openpi observation rather than three. openpi pins jax and
its own lerobot, so it keeps its own image and its own `./robot` verb.

`--policy lerobot --checkpoint <hub-id>` still exists for lerobot-format checkpoints;
it is not the comparison target.

### Four settings must match webui/openpi_worker.py

That worker is the configuration known to drive this checkpoint well here, in both rtc
and sync. All three of these were wrong in the first version and the arm moved
strangely for all three reasons at once:

| | value | why |
|---|---|---|
| units | **degrees** | `openpi_worker.py` defaults `--units degrees`; `evaluate.py` defaults `normalized`. The webui default wins because it has evidence behind it. |
| replan interval | **15 of 50** | `--actions 15`: the last 35 actions of each chunk are normally discarded and re-planned. Inspect Robots' `DefaultController` plays the *whole* chunk when `replan_interval` is `None`. |
| settling | **off** | Waiting for the arm to arrive changes chunk-replay cadence, which this policy was tuned against. `inspect-robots-so101` ships it off for this reason. |
| slew limit | **none** | Neither `evaluate.py` nor `openpi_worker.py` sets `max_relative_target`. lerobot clamps against the *measured* position, so under grasping load a lagging servo drags the command with it and the arm creeps — it moves, but cannot close on the object. |

Because `SOArmConfig` refuses `home_pose` without a slew limit, no slew limit also
means **no auto-homing** on the openpi path. Run `./robot home` between trials so each
starts from the same pose — that is the closed-loop homing this repo already uses, and
it is what the webui workflow does anyway.

Units and replan interval live in [openpi_policy.py](openpi_policy.py), settling in
[run.py](run.py); all three are keyed off the selected policy so they cannot drift.

None of these fail loudly. The embodiment commands policy output verbatim after
clamping, and Inspect Robots compares state *keys*, not units — so the wrong unit is a
silent per-joint rescaling (a factor of `half_span/100`, 1.69 on `wrist_roll`), and a
missing replan interval is 35 stale open-loop actions per chunk.

`--actions N` overrides the replan interval.

## What is measured, and by whom

`so101-cap-to-cup` is five hand-set layouts of the instruction *"Put the cap into the
red cup"* — worded verbatim from the π0.5 checkpoint's fine-tuning task, because a VLA
is only obliged to follow the instruction it was trained on. Rewording it would hand
the comparison to the LLM by default.

Scoring is **by operator**. Nothing on this rig can tell whether a cap landed in a cup:
the arm reports joint positions and two camera streams, and that is all. A person
answers `y/n/partial/skip` after each trial. Do not switch this to `success_at_end` —
it counts only embodiment-detected `"success"` terminations and would score every
operator-ended trial a failure.

The horizon is in **seconds, not steps** (120 s default). A chunked VLA steps at camera
cadence while an LLM agent waits on a frontier model for every action; a step budget
would silently give one of them many times more wall-clock than the other.

## The safety clamp is computed, not typed

`inspect-robots-so101` clips every command to `joint_low/high` inside `step()`, beneath
any approver — and ships ±180° placeholders as defaults. `rig.py` derives the real ones
from this arm's own calibration file, so recalibrating moves the clamp with it:

| joint | limit (deg) | joint | limit (deg) |
|---|---|---|---|
| shoulder_pan | ±104.13 | wrist_flex | ±104.09 |
| shoulder_lift | ±104.31 | wrist_roll | ±168.57 |
| elbow_flex | ±97.89 | gripper | 0–100 |

From lerobot's own `MotorsBus._normalize`: `degrees = (raw - mid) * 360 / max_res`,
`mid = (range_min + range_max) / 2`, `max_res = 4095` for the STS3215. Two independent
checks that this is the right formula rather than merely a formula: the URDF's limits
(±110, ±100, −100..90, ±95, ±160) bracket these within a couple of degrees, and
`config/home_pose.json`'s recorded `shoulder_lift` is −104.31, i.e. the arm was homed
sitting exactly on the computed limit.

That coincidence is also a trap. The recorded home is −104.31 and the computed limit is
−104.3077, so the home pose is **0.0023° outside the clamp** and `SOArmConfig` rejects
it outright. `rig.HOME_INSET` pulls the home pose in rather than widening the clamp to
admit it — a safety bound should not be relaxed to fit a rounded number, and 0.0023° is
well under one encoder count (0.088°).

Three layers sit between a model's output and the motors, and they are not redundant:
Inspect Robots' `ClampApprover`, the embodiment's own clamp, and lerobot's
`max_relative_target` slew limit (10 counts ≈ 0.88°/step, the same leash
`scripts/robot/policy_bridge.py` puts on a sim-trained policy).

## About the lerobot version pin

`inspect-robots-so101` declares `lerobot[feetech]>=0.5,<0.6`; this image carries the
le101 fork at **0.6.1**. The Dockerfile installs the package **without** its `lerobot`
extra, so the fork stays put.

The cap is conservative rather than load-bearing here. The adapter's entire lerobot
seam is seven symbols, and all seven resolve against 0.6.1:

```
lerobot.robots.so_follower:SOFollower                       OK
lerobot.robots.so_follower.config_so_follower:SOFollowerRobotConfig  OK
lerobot.policies.factory:get_policy_class                   OK
lerobot.policies.factory:make_pre_post_processors           OK
lerobot.utils.feature_utils:hw_to_dataset_features          OK
lerobot.datasets.feature_utils:hw_to_dataset_features       missing (fallback arm)
lerobot.datasets.utils:hw_to_dataset_features               missing (fallback arm)
lerobot.async_inference.helpers:raw_observation_to_observation  OK
```

The two misses are the later arms of a `try/except` chain whose first arm resolves, so
the paths the cap was written for are not the paths taken. If a future lerobot moves
that seam, the policy breaks loudly at import rather than silently mid-eval.

## Before a hardware session

1. `./robot eval --policy agent --dry-run` — the whole chain (registry, compat check,
   task, policy, API credentials) against the dependency-free CubePick mock world.
   `--dry-run` only exercises the agent path: CubePick is a 2-D eef-delta world and the
   lerobot policy declares the 6-D joint contract, so the runner refuses that pairing
   and points at the preflight instead.
2. `./robot eval-preflight --dry-run` — action dim, control mode, cameras and state
   keys line up, with no motion.
3. `OPENAI_API_KEY` in `.env.local` for the agent path. Model strings are
   OpenRouter-style `provider/model`; `openai/*` resolves against `OPENAI_API_KEY`
   directly, anything else falls through to `OPENROUTER_API_KEY`.

A green preflight proves the dims and names line up. It does **not** prove the joint
values mean the same thing on both sides — confirm with a single slow jog before
trusting a checkpoint.

## Verified, and not

Verified: the rig config in both unit modes (limits, home-pose clamping, the
degrees→normalized conversion checked against the ratio), task construction, the full
eval loop end to end on the CubePick mock world, `inspect-robots-so101-preflight`
reporting the 6-D contract compatible in the built image, and the openpi checkpoint
loading through the adapter and returning a 50×6 action chunk (its range, −61…+56, is
consistent with normalized rather than degrees).

NOT verified: anything that moves the arm. No eval has driven hardware. The first
session should be the two dry runs above, then a single layout, before a full five.
