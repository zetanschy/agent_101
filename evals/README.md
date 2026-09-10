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
./robot eval-openpi --task cap-quadrants                     # pi0.5, quadrant + random cup
./robot eval-openpi --mode sync                              # the pre-RTC code path
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

### Five settings must match webui/openpi_worker.py

That worker is the configuration known to drive this checkpoint well here. Four of
these were wrong in the first version and the arm moved strangely for all four reasons
at once; the fifth — the execution mode — was wrong for longer, and is what
[rtc.py](rtc.py) fixes:

| | value | why |
|---|---|---|
| units | **degrees** | `openpi_worker.py` defaults `--units degrees`; `evaluate.py` defaults `normalized`. The webui default wins because it has evidence behind it. |
| replan interval | **15 of 50** | `--actions 15`: the last 35 actions of each chunk are normally discarded and re-planned. Inspect Robots' `DefaultController` plays the *whole* chunk when `replan_interval` is `None`. |
| settling | **off** | Waiting for the arm to arrive changes chunk-replay cadence, which this policy was tuned against. `inspect-robots-so101` ships it off for this reason. |
| slew limit | **none** | Neither `evaluate.py` nor `openpi_worker.py` sets `max_relative_target`. lerobot clamps against the *measured* position, so under grasping load a lagging servo drags the command with it and the arm creeps — it moves, but cannot close on the object. |
| execution mode | **rtc** | `openpi_worker.py` overlaps inference with execution (`--mode async`, or `rtc` with the seam pinned). `DefaultController` cannot: it calls `act()`, waits, *then* plays the chunk. See below. |

Because `SOArmConfig` refuses `home_pose` without a slew limit, no slew limit also
means **no auto-homing** on the openpi path. Run `./robot home` between trials so each
starts from the same pose — that is the closed-loop homing this repo already uses, and
it is what the webui workflow does anyway.

Units and replan interval live in [openpi_policy.py](openpi_policy.py), settling in
[run.py](run.py); all three are keyed off the selected policy so they cannot drift.

## Real-time chunking, and where the loop lives

Until now this bench ran the π0.5 checkpoint **synchronously**, and nothing said so.
`DefaultController` calls `act()`, waits for the chunk, then plays 15 actions of it —
and `SOArmEmbodiment` is *self-paced*, sleeping to `control_hz` measured from the end
of the previous step. So an inference did not just cost time, it cost **motion**: the
arm held its last commanded position for the whole inference, once every 15 actions.
The webui worker has never done that.

`--mode` picks the loop:

| `--mode` | inference | the seam between chunks | who runs it |
|---|---|---|---|
| `rtc` (default) | overlapped | **pinned**: the new chunk is sampled under a constraint reproducing the actions committed while it was being sampled | [rtc.py](rtc.py) |
| `async` | overlapped | spliced cold — the arm jumps | [rtc.py](rtc.py) |
| `sync` | blocking | no seam; the arm holds still and restarts from the chunk's head | the framework's `DefaultController` |

`sync` is kept, and kept the same — same controller, same `act()`, same blocking
inference — because every eval log recorded before this change was produced that way,
and those scores have to stay comparable to new ones. One thing did change for it: the
JIT compile is now paid at load (see the warmup below) instead of inside the first
trial, which moves wall clock around without changing an action.

### The one number that has to be right

RTC's soft constraint is only as good as `d`, the count of actions the arm will have
executed by the time the new chunk takes over. A loop that swaps chunks **on arrival**
cannot know it and has to predict its own latency, typically from a rolling maximum of
measured delays. This loop retires each chunk at a **fixed index**, so `d` is exact and known at
submit time. If the inference overruns, the loop stalls *at* that index and still sends
no more than promised: a slow inference costs motion, not correctness.

Nothing downstream can catch a wrong `d`. It produces smooth, plausible motion pinned
to the wrong instant — so the arithmetic is defined once, in
[../scripts/openpi/chunk_loop.py](../scripts/openpi/chunk_loop.py), and **shared with
the web UI**: `webui/openpi_worker.py` drives the same `ChunkSchedule` with its own
thread and its own 30 Hz pacing. One definition of `d`, the same way there is one
definition of an openpi observation.

```bash
python3 scripts/openpi/chunk_loop_test.py    # the arithmetic: no deps, no GPU, no arm
python -m evals.rtc_test                     # the framework seam, on the mock world
```

The algorithm itself is pinned to LeRobot's, numerically, by a third suite that runs
the reference implementation rather than a transcription of it:

```bash
./robot rtc-parity                           # dumps le101's RTCProcessor, checks openpi's
```

Last run: **56/56** prefix-weight cases (four schedules x a (d, s, H) grid including the
degenerate ones) and **7/7** guided denoise steps match — every one of them via
`jacobian=identity`, which is itself the evidence that the reference's autograd
correction reduces to the identity.

Both loop suites pass too. The first counts the actions a simulated loop actually sends and
asserts it equals the `d` that was promised, including when the inference takes 40
ticks in a 15-tick window. The second runs the real `eval()` over the CubePick mock
world and checks that each trial opens with its own blocking, unguided inference —
i.e. that no chunk survives a trial boundary.

### How wide the seam is, and why it is not `--actions`

`s` — the index where prefix influence reaches zero — defaults to **the whole overlap
with the chunk being retired**, and that is a correction. It used to be
`d + --actions`, which at H=50 and a 15-action window is 29 of 50 rather than 35, so
the constraint released six steps early and blended over less of the trajectory than
the method intends. PI's own kinetix eval computes it as `action_chunk_size −
execute_horizon`, which is the identity

```
s == action_horizon − replan_interval == len(previous chunk's leftover)
```

i.e. every step for which a previous plan exists to agree with. It follows `--actions`
on its own, so nothing has to be told the replan interval — a literal is only ever
right for one (LeRobot's `RTCConfig` default of 10 is one such literal, and the
`d + --actions` we had was another). `--rtc-horizon N` still releases it early on
purpose, and either way it is clipped to the real leftover.

Credit where due: this is the one place another JAX implementation of RTC reads the
paper more carefully than we did — its config documents the identity above and defaults
the horizon to "the whole leftover" for the same reason.

Two defaults we did **not** adopt from it, deliberately:

| | ours | theirs | why ours stays |
|---|---|---|---|
| Jacobian | `identity` | true VJP | `identity` is what LeRobot's `denoise_step` actually computes (its `requires_grad_` lands after the denoiser call), it is what `rtc_parity.py` pins us to, and the VJP roughly doubles per-step cost — on this rig that eats the window RTC exists to hide. `--rtc-jacobian full` to measure it. |
| `beta_max` | 5.0 | 10.0 | 5.0 is PI's kinetix eval value; 10.0 is LeRobot's config default. `--rtc-max-guidance` to A/B. |

Both are now flags rather than constants, so a session can test one against the other
without editing code. Which is better on this arm is a measurement nobody has taken.

### Three parts, three files

| | |
|---|---|
| `openpi.models.rtc` | the algorithm: prefix weights, guidance weight, guided velocity in the flow sampler |
| `openpi.policies.rtc` | `RealTimeChunker`: which previous actions to hand it, and the clipping of `d` and `s` |
| `chunk_loop.py` + [rtc.py](rtc.py) + [openpi_policy.py](openpi_policy.py) | the loop: when to ask, what `d` is, and who owns the in-flight request |

The policy owns the inference thread rather than the controller, for one reason:
`reset()` has to **drain before it resets the chunker**. A trial's last inference is
usually still in flight when the horizon or the operator ends the trial, and if it
lands after the reset, `RealTimeChunker` keeps *its* chunk as the prefix — so the next
trial's first guided sample is pinned to actions from the previous trial, on a table
that has been rearranged in between.

### Two things that would otherwise bite mid-trial

**The guided sampler is a second JAX trace.** Passing a prefix changes the argument
tree, so it compiles separately. Without a warmup the first guided call is the first
replan of the first trial — mid-motion, arm holding its last command, for as long as
the compile takes — tens of seconds, on the torch equivalent of this path, which is
long enough to end an episode. `OpenPiPolicy` now warms up **both** traces at load, on synthetic
pixels shaped from the declared observation space, with no action sent.

**Delta actions would need re-anchoring.** openpi's `extra_delta_transform` encodes
actions as deltas from the first state of each chunk, so a prefix from a chunk anchored
at an older state is not comparable to the one being sampled. `--mode rtc` **refuses**
such a checkpoint rather than guiding on an offset trajectory;
`pi05_soarm101_lora_cap_to_cup` sets it `False`, so this bench's checkpoint is fine.

### What a run now records

Per inference, in the policy transcript the report renders: `d`, `s`, the index the
prefix was taken from, and the chunk's **normalized `absmax`** — RTC's guidance is
additive and can push values past the ±1 the model was trained in, which the arm would
only show as a joint sitting on its clamp.

Per trial, in the log's `trial_metadata.openpi`: mode, inference count, how many were
guided, mean/max latency, `late_seams`, `stall_s`, and the worst `absmax`. `late_seams`
is the one to read first — a run whose seams are late is a run whose effective control
rate dropped below the 30 Hz the checkpoint was tuned at, and the fix is a longer
window (`--actions`), not a different mode.

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

### Two benchmarks, one instruction

`--task` selects one; both carry the *same* instruction and the same operator scorer,
so their scores stay comparable and neither one favours a policy by wording.

| `--task` | scenes × epochs | what moves between trials | asks |
|---|---|---|---|
| `cap-to-cup` (default) | 5 × 1 | nothing — both objects pinned per scene | does the skill work at all? |
| `cap-quadrants` | 2 × 5 = **10** | the **cup**, re-drawn at random every trial | does it work wherever the cup is? |

`cap-quadrants` pins the cap to one **lower table quadrant** — near-right for
`quadrant-lower-right`, near-left for `quadrant-lower-left`, in the *operator's* frame
(standing behind the arm, which is also the `front` camera's view) — and the operator
re-places the **red cup at a fresh random spot before every trial**. So a scene's five
epochs are five genuinely different scenes to the policy, and its score is a success
*rate* over cup positions rather than a verdict on one arrangement. The two quadrants
then split that rate by where the reach started, which is where this arm is least
symmetric.

Drawing the cup well is part of the method, not a detail:

* anywhere in the reachable area, **> 10 cm from the cap** — closer than that and the
  two objects are one blob in the front camera, so a policy can succeed without ever
  having localised the cup;
* **never the previous spot** — physically move it, do not nudge it;
* vary **distance** as much as side. Ten cup spots all landed mid-table means one cup
  position measured five times per quadrant.

`--layouts`/`--epochs` default to whatever the selected task declares, so
`./robot eval-openpi --task cap-quadrants` is the full 10-trial session; pass
`--layouts 1 --epochs 1` for a single trial first. Both benchmarks print their
per-scene setup before the arm moves, because the framework has no pre-trial hook —
the verdict prompt only echoes the setup once the trial is already over.

Budget it like a physical session: 10 trials × (120 s horizon + reset + `./robot home`
+ a verdict) is roughly 40 minutes at the arm.

Scoring is **by operator**. Nothing on this rig can tell whether a cap landed in a cup:
the arm reports joint positions and two camera streams, and that is all. A person
answers `y/n/partial/skip` after each trial. Do not switch this to `success_at_end` —
it counts only embodiment-detected `"success"` terminations and would score every
operator-ended trial a failure.

### The horizon means different things to the two policies

`max_seconds` is converted to a step count using the embodiment's **declared**
`control_hz` (30). The chunked VLA really does run near that. The LLM agent measured
**5.5 steps/s**, so a "120 second" budget was 3600 steps ≈ 11 minutes of wall clock —
and the first Astra trial was cut off at step 1685 of 3600, which read as a failure
rather than an interruption.

So the horizon follows the policy:

| policy | horizon | why |
|---|---|---|
| `openpi` | `--seconds` (120) | steps at 30 Hz are real for a chunked VLA |

One caveat that only `--mode rtc`/`async` removes: under `sync` those 3600 steps took
*longer* than 120 s of wall clock, because the embodiment absorbs each inference into
the control period. Overlapped, a 120 s budget is about 120 s at the arm.
| `agent` | `--decisions` (40) | the LLM call budget governs runtime *and* the bill |

`--decisions` sets the plugin's `max_llm_calls`, which is enforced **and** written into
the agent's system prompt, so the model plans against it. The step horizon is derived
from it (60 steps per decision, measured: 1685 steps over 38 calls averaged 44, with
single interpolated moves as long as 190) and sized so the decision budget is what
ends a trial.

Measured per decision on this rig: **~8 s** (6.0 s mean LLM latency plus the arm
executing the move) and **$0.064**, rising through a trial as the conversation
accumulates images — input tokens grew 1,477 → 9,641 across 38 calls. The plugin's own
default of 100 calls would be ~13 minutes and ~$6 per trial.

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

### Nothing else bounds a trial in time

Every other limit is a **count**: `max_steps`/`max_seconds` are step counts,
`max_llm_calls` is decisions. That is fine at ~8 s per decision and not fine when the
provider is degraded but not erroring — the agent's client waits 120 s per request and
retries three times, so one decision can burn six minutes without failing.

`--max-minutes` (default 10, `0` disables) wraps the policy in
[deadline.py](deadline.py) and ends the trial the way the framework already lets a
policy end one: an action carrying `request_stop`. It logs as
`termination_reason: "wall_clock_timeout"` with run status `success` — a truncation,
distinguishable at read time from an error and from your Ctrl-C. The stop action is
the **current pose**, so the guard can never itself become a motion.

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

### Why the report needs the policy to talk

The HTML report's camera player is built from the **policy transcript**, not from the
frames directory: it looks for a text part `camera '<name>' (step N):` followed by
`[image omitted: streamed camera frame]`, and resolves each to
`<frames_dir>/<trial_prefix>_<name>_<step:06d>.npy`. A policy that reports nothing
renders an *empty* flipbook however many frames the embodiment stored — which is what
the first openpi runs did. `OpenPiPolicy.transcript()` now emits one turn per
inference, naming both camera frames and the state, so π0.5's trials show imagery on
the same footing as the LLM agent's.

MP4s are a separate path that ignores transcripts entirely:

```bash
./robot eval-video          # writes <camera>.mp4 per trial into the frames dir
```

### Stored frames cost 3.1 GiB per minute of arm time

`store_frames` writes one **uncompressed** `.npy` per camera per control step —
640×480×3 uint8 is 0.88 MiB, twice a step, thirty steps a second, so **55 MB/s**. A
ten-trial session is 20–40 GB, and this bench has filled its disk on one.

The report never shows those pixels: `_html.py` sets `_FRAME_MAX_SIDE = 448` and
decimates anything larger with `array[::stride, ::stride]` on every render, so a
640×480 frame is displayed at **320×240** whether it was stored that way or not.
Storing what gets displayed is therefore free:

```bash
./robot eval-shrink --dry-run     # what it would save, touching nothing
./robot eval-shrink               # newest log: encode MP4s, then decimate 4x
./robot eval-shrink --stride 4    # 16x, at a flipbook coarser than the report's own
./robot eval-shrink --watch       # alongside a live session, as each trial finishes
```

Stride 2 is byte-identical in the report — same decimation, done once on disk instead
of on every render — and keeps every frame, so the flipbook and the transcript images
are unchanged. MP4s are encoded **first** by default, because the encoder reads the
same files and a run shrunk before encoding yields a smaller video forever.

What it does not fix is **peak** usage: the eval writes full-size frames while it runs,
so a session still needs its 20–40 GB at the moment it needs it. `--no-frames` avoids
that entirely; `--watch` is the middle ground, since a trial's frames stop changing
when the next trial starts, bounding the peak at roughly one trial.

`evals/shrink_test.py` pins the claim that makes this safe: a shrunk frame equals what
`_load_frame` would have rendered from the original, pixel for pixel.

Verified: the rig config in both unit modes (limits, home-pose clamping, the
degrees→normalized conversion checked against the ratio), task construction, the full
eval loop end to end on the CubePick mock world, `inspect-robots-so101-preflight`
reporting the 6-D contract compatible in the built image, and the openpi checkpoint
loading through the adapter and returning a 50×6 action chunk (its range, −61…+56, is
consistent with normalized rather than degrees).

Also verified, for the chunk loop: both suites above, plus the shared `ChunkSchedule`
loading and producing the same `d` inside `webui/openpi_worker.py`'s own loader.

### `--mode rtc` on hardware, first session

`so101-cap-to-cup`, 5 layouts × 2 epochs, 2026-09-10
(`so101-cap-to-cup_e648db86.json` against `so101-cap-to-cup_f750e49a.json`):

| | sync | rtc |
|---|---|---|
| operator | 0.8 | **0.9** |
| session | 841 s | **342 s** |
| arm-time (steps ÷ 30 Hz) | 309 s | **144 s** |
| time frozen at seams | ~270 s (618 × 0.43 s) | **0.0 s** |
| late seams | n/a | **0**, every trial |
| mean inference | 433 ms | 374 ms |
| guided | — | 28/29, 25/26, … (all but each trial's first) |
| normalized absmax | — | 1.02–1.14 |

One verdict apart is not a score difference worth defending at n=10 — near went
`[y,n] → [y,y]` and close-pair `[n,y] → [y,n]`. The process numbers are the result:
**no seam ever had to wait**, and the same ten trials took less than half the arm-time,
which is what removing a 430 ms freeze from every fifteenth action buys. `guided` being
one short of `inferences` in every trial is the design showing through — a trial's first
chunk has no prefix to honour.

The 374 ms against sync's 433 ms is mostly the warmup: the JIT compile used to land on
the first inference of the run and inflate its mean.

Guidance does push past the trained range, mildly — absmax 1.02–1.14 where 1.0 is the
edge. Worth watching per checkpoint, and the reason it is reported per trial.

NOT verified: `--mode async` has never driven the arm (it exists to isolate what the
guidance is worth, not to be run), and no RTC session has yet run `cap-quadrants`, whose
only hardware result is the 0.2 recorded on the synchronous path.
