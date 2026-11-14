# Two separate gaps sank the RGB student

The 88% vs 27% comparison does not measure distillation. The teacher and the student were scored in different simulators, at different effective delays, through different image pipelines, and neither has been scored in the other's environment. Your eval records split the drop into two parts of roughly equal size.

**Inside mjlab, an offline-BC student kept 57 of its teacher's 92 points on the same 100 stages.** That ~35-point in-domain gap is typical of offline behaviour cloning: rebar insertion scored 68.9% with pure BC vs 96.1% with DAgger, and ResiP saturated at 73–80% against a 98% teacher. The closest analogue to your setup, NVIDIA's SO-101 distillation tutorial, uses DAgger and nearly closes the gap (99.0% vs 99.6%).

**The rest is environment shift.** Your student's dataset contains exactly one dead time (1.0 frame). The live twin adds 12–31 ms of real-time pipeline latency, float64 CPU contact physics and a distort-blur-undistort image chain the student never saw. Your own model_2800 teacher **lost 31 points (92 → 61) when one frame of dead time was added at evaluation**, and it failed in the same ways the student does.

**Inside the distillation gap, the main suspect is information the student cannot see.** Its action is a delta on an integrated joint target that it never observes. It also has no joint velocity or history to infer the command still in flight under dead time. Validation MSE rose from 0.053 to 0.090 once dead time entered the data, which fits this explanation but does not prove it. The coarse encoder (a 6×5 feature map), single-step deltas and MSE averaging are plausible amplifiers that have not been isolated.

**The cheapest path has three steps:**
1. Get the two missing numbers: the teacher scored in the twin, and the student scored closed-loop in mjlab. Then run a delay sweep.
2. Retrain with dead time randomized over 1–3 frames, and with joint velocity and the integrated target as inputs. Both are already saved in your dataset shards.
3. Add teacher-relabelled DAgger, using the rsl_rl `Distillation` code already in your mjlab venv.

All file paths below are relative to `/mnt/storage/Desktop/agent_101`.

## The 88-versus-27 comparison crosses two environments

Every score on record is listed below. The two most informative cells are empty: nobody has run the teacher in the live twin, and nobody has run `bc_dead3700` closed-loop in mjlab. Confidence intervals are 95% Wilson intervals. At n=100 they span roughly ±9 points, so differences under ~15 points between single runs are not meaningful.

| Policy | Scored in | Dead time at eval | Success (95% CI) | Record in `outputs/sim_eval/` |
|---|---|---|---|---|
| teacher model_2800 | mjlab, lockstep | none | **92/100** (85–96) | `mjlab_teacher_2800` |
| teacher model_2800 | mjlab, lockstep | 1.0 frame | **61/100** (51–70) | `mjlab_teacher_2800_dead1` |
| teacher model_3700 (trained with 1–3 frames) | mjlab, lockstep | 1.0 frame | **88/100** (80–93) | `mjlab_teacher_dead3700` |
| student bc_2800 (flat mujoco_warp renders, no dead time) | mjlab, lockstep | none | **57/100** (47–66) | `mjlab_student_bc_2800` |
| student bc_2800 | live twin, real time | twin | **0/20** (0–16) | `mjlab_student_bc_2800_livetwin` |
| student bc_dead3700 (CPU textured renders + visual DR, 1.0 frame) | live twin, real time | twin servo + pipeline | **27/100** (19–36) | `mjlab_student_bc_dead3700_livetwin` |
| teacher model_3700 | live twin | — | never measured | — |
| student bc_dead3700 | mjlab, closed loop | — | never measured | — |

The bc_2800 pair is a clean distillation-only measurement. It shows the student kept **62% of its teacher's success** (57/92) in the teacher's own environment, on the same stages, under the same protocol. Assume bc_dead3700 kept the same fraction in mjlab. It would then score about 55, putting roughly **33 points of the 61-point drop on distillation and roughly 28 points on the move to the live twin**.

That split is arithmetic, not a measurement. bc_dead3700 differs from bc_2800 in four ways: textured renders, visual DR, dead-time data, and a different teacher. Its in-mjlab score could land well away from 55. The table below separates what your data confirms from what is inferred.

| Component | Evidence in your data | Literature signal | Status |
|---|---|---|---|
| (a) Evaluation/environment mismatch: other sim, real-time loop, delay, renderer | The bc_2800-type student fell from 57 in mjlab to 0/20 in the twin. Teacher_2800 fell 92 → 61 with one frame of dead time. The twin pipeline adds 12 ms median and 22 ms p95 latency | Fixed-timing policies break at 2× the trained latency ([Sandha et al.](https://proceedings.mlr.press/v155/sandha21a.html)). Without latency modelling, controllers fail outright ([Tan et al. 2018](https://arxiv.org/abs/1804.10332)) | Direction **confirmed**; size for bc_dead3700 **estimated (~28 pts)** |
| (b) Pure distillation gap: offline BC, compounding error | 92 → 57 in mjlab on the same stages, despite DART-style noise in the data | Pure BC 68.9% vs DAgger 96.1% ([Sun et al. 2026](https://arxiv.org/abs/2609.20477)); off-policy 16 vs on-policy 64 ([LBC](https://arxiv.org/abs/1912.12294)) | 35-pt aggregate **confirmed**; (b), (c) and (d) are entangled inside it |
| (c) Imitation gap: missing velocity, target/last action, contact, cap_up | Val MSE 0.053 → 0.090 once dead time entered the data (confounded). Open-side-up caps: student 50% vs 62% closed-side-up in mjlab (not significant); teacher ~90% for both | BC 0.16 / DAgger 0.34 vs 0.88 for a student-aware teacher in drawer opening ([SITT](https://arxiv.org/html/2412.09149v2)) | **Hypothesis**, consistent with the data, testable offline |
| (d) Student design: encoder resolution, single-step deltas, MSE, proprio shortcut, augmentation | Nothing isolated yet | Shallow conv encoder −25 to −62% relative ([robomimic](https://arxiv.org/abs/2108.03298)); chunking 1% → 44% ([ACT](https://arxiv.org/abs/2304.13705)) | **Hypothesis**, second-order |

## One unmodelled frame of delay already costs your teacher 31 points

Your own records hold the clearest evidence for component (a). The model_2800 teacher scores 92 without dead time. When the same checkpoint is scored with the 1.0-frame dead time (`mjlab_teacher_2800_dead1`), it drops to **61**. Its failures shift to 23 caps knocked off the table, 9 never picked up and 7 mugs knocked over. That failure mix resembles the BC students' failures, which means **failure categories alone cannot tell a timing error from a distillation error**.

model_3700 was trained over dead times of 1–3 frames and holds 88 at 1.0 frame. The student, however, saw exactly one value. `cap_to_mug_cfgs.py` sets `dead_range = (dead, dead) if play else …RANGES["dead_time_frames"]`, and `collect.py` builds its env with `play=True`. **The student is therefore the most delay-fragile policy in the stack**, while the teacher is the one that was robustified.

In the twin, `scripts/openpi/sim_eval.py` logs a median per-episode latency of **12 ms**, a median p95 of **22 ms** and a worst p95 of **31 ms** for the student's observe-plus-infer call. Add these to the identified 1.0-frame servo dead time and the student acts at an effective **1.4–1.9 frames**, a delay its data never contained. The loop also adds jitter. On every third tick (`TRACE_EVERY = 3`), the evaluator calls `sim.status()` and a full `robot.get_observation()` for the video before the student requests its own observation. That pushes those actions later in the real-time frame with a 3-tick rhythm.

The literature treats this kind of mismatch as decisive:
- Tan et al. found that without latency modelling, the learned controllers did not work on the real robot at all ([Tan et al. 2018](https://arxiv.org/abs/1804.10332)).
- Sandha et al. found that policies trained at fixed timing failed once execution latency reached twice the trained value ([Sandha et al.](https://proceedings.mlr.press/v155/sandha21a.html)).
- Vision-locomotion students model latency during the **distillation phase**, not only during teacher RL. Agarwal et al. account for 10 ± 10 ms of socket latency in phase 2 ([Agarwal et al. 2022](https://arxiv.org/abs/2211.07638)). Imai et al. found that camera–proprio asynchrony alone causes large performance drops, and they model a separate delay per modality ([Imai et al.](https://arxiv.org/abs/2109.14549)).

Your dataset renders image and joints from the same qpos snapshot, so the student never saw the image-vs-joint skew a real-time loop creates.

Physics differs as well. mujoco_warp runs in float32 and CPU MuJoCo in float64. The MuJoCo team warns that solver iterations, collision detection and small friction values can be sensitive to that difference. Some colliders return different maximum contact counts, and GPU results are not deterministic ([MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/)). Isaac Lab's guidance for cross-engine comparisons applies directly to grasping: identical friction coefficients in two engines do not give identical slip, grasp stability or contact counts ([Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/concepts/solver_differences.html)). mjlab lists cross-simulator portability as an explicit non-goal ([mjlab](https://arxiv.org/abs/2601.22074)).

The image pipelines differ too:
- **Training frames** (`collect.py`): pinhole renders, 4× supersampled and box-averaged.
- **Twin frames**: a 640×480 pinhole render goes through the fitted lens distortion (bilinear remap, black outside) and a Gaussian "lens softness" blur (`Real2Sim._warped`). `student_policy.py` then undistorts it with bilinear interpolation and replicated borders and area-resizes it.

That is two extra resamplings plus a blur the student never trained on. Small pixel-statistics changes matter for CNNs. Different resize implementations alone produce very different images ([Parmar et al.](https://arxiv.org/abs/2104.11222)), and a few-pixel translation can change a CNN's output drastically ([Azulay & Weiss](https://arxiv.org/abs/1805.12177)). SIMPLER found that visual matching closed the sim–real gap only when background, objects and robot texture were matched jointly (gap 0.272 → 0.136). Octo scored 0% with an untuned arm texture vs 29.3% with a tuned one ([SIMPLER](https://arxiv.org/abs/2405.05941)). Your flat-render student's 0/20 in the twin is the same phenomenon.

**Verdict on (a).** That it hurts is confirmed. How much it costs bc_dead3700 is unknown until the teacher runs in the twin. If model_3700 also collapses there, most of the 88 → 27 is environment, and no student change will fix it.

## Offline cloning explains most of the in-domain 35 points

Ross et al. bound the cost of behaviour cloning at **T²ε**. Each mistake moves the student into states the teacher never visited, where its error is uncorrected. DAgger trains on the student's own state distribution and achieves a bound linear in T ([Ross et al. 2011](https://proceedings.mlr.press/v15/ross11a.html)).

Every privileged-teacher distillation paper that ablates BC against on-policy relabelling finds a large gap:

| Work | Student input | Offline BC | On-policy (DAgger) |
|---|---|---|---|
| Learning by Cheating, CARLA ([Chen et al. 2019](https://arxiv.org/abs/1912.12294)) | RGB | 16% | 64% |
| Rebar insertion, arm ([Sun et al. 2026](https://arxiv.org/abs/2609.20477)) | multi-view RGB | 68.9% | 96.1% (β = 0.5 mixture) |
| Vision quadrotor ([Xing et al. 2024](https://arxiv.org/abs/2403.12203)) | pixels | 0 / 0 / 0 | 63 / 57 / 64 |
| Asymmetric actor-critic, real Pick ([Pinto et al. 2017](https://arxiv.org/abs/1710.06542)) | images | 0/5 | 3/5 |
| Isaac Lab SO-101 vial pick-place ([NVIDIA tutorial](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/06-distilled-policy.html)) | 2 stacked 64×48 wrist frames + joint pos, vel, target, previous action | — | 99.02% vs teacher 99.61% |

Both of your datasets already use DART-style noise (levels 0/0.1/0.3 per episode, per `meta.json`), and the 35-point gap persisted anyway. That matches what DART is designed for. It injects noise into the supervisor so the data contains recoveries from noise-like deviations, and it reported a 62% gain over BC on real grasping ([Laskey et al. 2017](https://arxiv.org/abs/1703.09327)). It does not produce the student's *systematic* errors, such as a consistent depth offset on approach, so those states never get labels.

More offline data also does not help much. ResiP's state student went from 78% at 10k trajectories to only 80% at 100k against a 98% teacher, which the authors trace to a limitation of the distillation process itself rather than to the change of modality ([Ankile et al. 2024](https://arxiv.org/abs/2407.16677)). The counterexample is DemoStart: offline BC matched a smooth teacher (99.0% vs 99.6%) but reached only 20.4% from a jerky SAC-X teacher ([Bauza et al. 2024](https://arxiv.org/abs/2409.06613)). A PPO teacher acting through 1–3 frames of dead time deserves an action-smoothness check before you trust offline BC alone.

Your twin breakdown fits compounding error but does not prove it. Success falls to **13% when the mug is 26–30 cm away**, and the longer transport gives drift more frames to accumulate. The closest published analogue uses your arm, your wrist resolution and a CNN smaller than yours, and still gets within 0.6 points of the teacher. It differs from your student in three ways: on-policy relabelling, two stacked frames, and velocity/target/previous-action inputs. The evidence therefore says the training procedure and the inputs matter more than the encoder.

The tooling is already installed. `thirdparty/mjlab/.venv` ships rsl_rl 5.5.0 with `algorithms/distillation.py`, `runners/distillation_runner.py` and a `CNNModel`. In that code the student acts, the teacher only labels, and storage is cleared after every update (online DAgger without aggregation) ([rsl_rl](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/distillation.py)). `scripts/openpi/dagger.py` is a different tool: human-in-the-loop DAgger for the openpi policy on the real arm, not teacher relabelling.

## The student must act on a target it cannot see

The student outputs a delta that `IntegratedJointPositionAction` (`cap_to_mug_mdp.py`) adds to a joint target. The target can sit up to the clip limit away from the measured joints, and a slew limit and 1-frame dead time sit between them. The teacher sees `last_action` and `joint_vel`, so it can tell where the command already in the pipe will take the arm. The student sees one frame and six joint positions. **Two states with the same image and joints but a different pending target get different teacher labels.** MSE then returns their average, which means creeping, overshooting or half-committed closures near a small cap.

This is the "imitation gap": when the teacher decides with information the student lacks, that information is averaged away during imitation ([Weihs et al., ADVISOR](https://arxiv.org/abs/2007.12173)). Recent distillation work describes the same aliasing: "identical student observations can correspond to contradictory teacher actions" ([RAIL](https://arxiv.org/html/2611.12251)).

Delay theory reaches the same conclusion from another direction. Any delay beyond one step breaks the Markov property, and the fix is to append at least the last K sent actions, where K is the maximum delay. SAC with a one-step action delay and no action buffer performed near random ([Bouteiller et al. 2021](https://arxiv.org/abs/2010.02966)). The positions-only setting has been tested directly. With velocities removed from MuJoCo observations, single-frame BC scored 363 on Walker2d and −38 on HalfCheetah. Adding one past frame raised these to 592 and 820, against experts at 2428 and 1336 ([Wen et al. 2020](https://arxiv.org/abs/2010.14876)).

Your data contains one signal that points this way, with a caveat:

| Dataset | Dead time in data | Val MSE |
|---|---|---|
| bc_2800 | none | 0.053 |
| bc_dead3700 | 1.0 frame | 0.090 (plateaued from epoch ~20) |

The rise fits labels that now depend on an unseen in-flight command. It is confounded by visual DR, textured renders and a different teacher.

The other privileged inputs matter less or are harder to recover:

| Input | Why it may be missing for the student | Your data |
|---|---|---|
| `fingers_on_cap` (two contact sensors) | Likely drives the timing of grasp closure. Scaffolder notes that proprioception and vision rarely predict touch, and that BC breaks when the optimal student and teacher act differently ([Hu et al. 2024](https://arxiv.org/abs/2405.14853)) | Not tested |
| `cap_up` | A single bit that may or may not be resolvable at 96×72 | Student: 20/40 open-side-up vs 37/60 closed-side-up in mjlab; 9/40 vs 18/60 in the twin. Teacher ~88–93% on both. A weak, non-significant hint |

DAgger alone does not fix missing information. RAIL's Corners task went BC 0.24 → DAgger 0.38 → 1.00 only with a teacher restricted to student-recoverable information ([RAIL](https://arxiv.org/html/2611.12251)). SITT's drawer opening went BC 0.16 → DAgger 0.34 → 0.88 with a student-aware teacher, and 0.74 when the KL penalty was removed ([Messikommer et al. 2025](https://arxiv.org/html/2412.09149v2)).

Two remedies follow:
- **Give the student the missing state.** Lee et al. ([2020](https://arxiv.org/abs/2010.11251)) and RMA ([Kumar et al. 2021](https://arxiv.org/abs/2107.04034)) feed proprioception and past targets or actions.
- **Make the teacher imitable.** Drop or noise the inputs the student cannot infer, as A2D ([Warrington et al.](https://arxiv.org/abs/2012.15566)) and SITT do.

Adding past actions has a known trap: **copycat**. A BC student given history learns to repeat the last action. It fits held-out demos well and then fails at changepoints such as starting to close the jaw. Naive history dropped Tool-Hang from 0.51 to 0.00 ([Torne et al. 2025](https://arxiv.org/html/2505.09561)). DAgger removes copycat: with 1k queries, Walker reached 2170 vs the expert's 2428 ([Wen et al. 2020](https://arxiv.org/abs/2010.14876)).

The practical resolution for your action space:
- Feed **target − joint_pos** and joint velocity rather than the raw last delta. These are what the Markov state actually needs.
- Pair them with DAgger.
- Watch on-policy action error at the moment the jaw closes.

Every hypothesis in this section can be tested offline today. Each shard written by `collect.py` already stores `joint_vel`, `target` (the integrated target) and `privileged` (the teacher's own 35-D observation) beside the images. `scripts/openpi/student_policy.py` already keeps `self.target` at deployment, so feeding it back costs nothing at run time.

## A 6×5 feature map and per-step deltas amplify every error

**Encoder.** Four 3×3 stride-2 layers reduce the 96×72 front image to about 6×5 cells (16 px per cell) and the 64×48 grip image to about 4×3. A single linear layer then has to decode the cap position from that grid. Precision-manipulation encoders keep more spatial detail:
- The DrQ/SAC-AE encoder strides only its first layer ([DrQ](https://arxiv.org/abs/2004.13649)).
- Spatial-softmax keypoints localized objects to **1.30 cm vs 2.59–4.75 cm** for the other architectures ([Levine et al. 2016](https://arxiv.org/abs/1504.00702)).
- Swapping robomimic's ResNet-18 + spatial softmax for a shallow conv net cost 25–62% relative ([robomimic](https://arxiv.org/abs/2108.03298)).
- Repeated downsampling erodes fine spatial detail, worst at low input resolution ([PRISM 2026](https://arxiv.org/abs/2606.15232)).

The SO-101 tutorial's small CNN [16,32,32] at 64×48 still reached 99% with DAgger and history, so the encoder ranks below the inputs and the training procedure.

**Action representation.** A per-step delta on a hidden integrated target behaves like velocity control. Diffusion Policy found position control beats velocity control, and velocity control suffers more from latency through compounding error ([Chi et al. 2023](https://arxiv.org/abs/2303.04137)). ACT saw degraded performance with delta joint positions and gained **1% → 44%** from action chunking (k = 1 → 100) ([Zhao et al. 2023](https://arxiv.org/abs/2304.13705)). Once the target is an input, though, delta vs absolute is a reparametrization. The missing anchor is the real problem, and the target input supplies it.

Chunking adds commitment across dead time, but costs reactivity. Under DAgger it also needs chunk labels from teacher rollouts that start at the student's states, not just one teacher query per state.

**Loss.** MSE regresses to the conditional mean whenever labels are aliased. KL to the teacher's action distribution beat L2 in all four seeds in DextrAH-RGB ([Singh et al.](https://arxiv.org/abs/2412.01791)), and ACT uses L1 for precision. A diffusion head would represent "close now" and "keep approaching" as separate modes, but it cannot choose between them without the missing signal.

**Proprioception shortcut.** Policies lean on proprioception in motion-transition phases because it lowers the loss fastest, which suppresses vision exactly at approach and grasp ([Lu et al., ICLR 2026](https://arxiv.org/abs/2602.12032)). robomimic found that extra proprioception hurt image agents by 2–29% ([robomimic](https://arxiv.org/abs/2108.03298)).

**Augmentation.** A ±4 px shift is standard (DrQ ±4 on 84 px; Diffusion Policy crops ~0.9) and is unlikely to be the problem. Removing shift augmentation costs 35–47% (relative) in robomimic.

**Auxiliary losses.** These are the cheapest precision lever. James et al.'s cube grasping went **84% → 100%** in the real world with auxiliary cube and gripper position heads, and the loss without them was concentrated at the grasp ([James et al. 2017](https://arxiv.org/abs/1707.02267)). DextrAH-G uses action L2 plus 0.1 × object-position L2 ([Lum et al. 2024](https://arxiv.org/abs/2407.02274)).

**Checkpoint selection.** Validation MSE is a poor guide: it correlates at r = 0.308 with real success ([SIMPLER](https://arxiv.org/abs/2405.05941)), and robomimic's val-loss checkpoint selection was 10–100% worse than the best checkpoint. Your 0.090 plateau says little about which epoch drives best.

## Measure first, then fix in this order

Run the diagnostics first. They are cheap, and D1 and D2 decide whether to spend effort on the twin or on the student. Effort figures are my estimates.

| # | Diagnostic | Where | What it decides | Effort |
|---|---|---|---|---|
| D1 | Score teacher model_3700 in the live twin on seeds 1000–1099, building its 35-D actor observation from the twin's MuJoCo state | `scripts/openpi/sim_eval.py` (new teacher stack beside the `.ts` student path) | Teacher ≥ ~80 in the twin → the environment is minor; fix the student. Teacher ≤ ~60 → fix delay, physics and render parity first | ~1 day |
| D2 | Score bc_dead3700 closed-loop in mjlab, rendering each step from current qpos with `collect.py`'s CPU renderer plus the twin's distort/blur/undistort chain, at 1.0-frame dead time, on the same stages | `thirdparty/mjlab/scripts/so101_twin/eval_teacher.py` env + `collect.py` renderer | Student(mjlab) vs teacher(mjlab) = distillation gap; student(mjlab) vs student(twin) = transfer gap. This harness becomes the DAgger engine | ~1 day |
| D3 | Delay sweep at 0, 1, 1.5, 2 and 3 frames for teacher and student in mjlab lockstep (`dead_time_frames` accepts fractions). Log per-step obs, action and sim timestamps in the twin | `cap_to_mug_cfgs.py`, `sim_eval.py` | Whether the student is more delay-fragile than the teacher, and the twin's real delay distribution | hours |
| D4 | On-policy action error: on every student rollout step (D2 and twin), compute the teacher's action from privileged state; plot \|a_s − a_t\| against time aligned to first contact and lift | D2 harness; twin state | Whether errors start at one phase (jaw-close timing) or accumulate (compounding) | hours |
| D5 | Render parity: the same qpos through `collect.py`'s pipeline vs the twin's pipeline; pixel difference and student action difference | `collect.py`, `Real2Sim._warped`, `student_policy.py` | Whether image preprocessing alone moves the actions | ~1 hour |
| D6 | Offline input ablation on existing shards: val MSE for joints-only, +`joint_vel`, +(`target` − `joint_pos`), +`fingers_on_cap`, +full `privileged` | `thirdparty/mjlab/scripts/so101_twin/student.py` | Which hidden variables the labels depend on (imitation gap) and whether proprioception carries the loss (shortcut) | a few GPU-hours |

Then apply the fixes in order of expected gain per unit of effort. F1–F3 can share one recollection and one retrain.

| # | Fix | Files | Evidence | Effort |
|---|---|---|---|---|
| F1 | Randomize dead time during collection: 1–3 frames, or the twin's measured 1.4–1.9 from D3. On sim-eval trace ticks, reuse the observation already captured instead of observing twice | `cap_to_mug_cfgs.py` (`dead_range` under `play`), `collect.py`, `sim_eval.py` | Your teacher lost 31 points at +1 frame; delay is modelled during distillation in [Agarwal et al.](https://arxiv.org/abs/2211.07638) and [Imai et al.](https://arxiv.org/abs/2109.14549) | hours + one recollection |
| F2 | Add joint velocity and target − joint_pos (optionally a 2-frame stack) as student inputs; feed velocity at deployment by finite difference | `student.py`, `student_policy.py` (already holds `self.target`) | Same inputs as the 99% [SO-101 tutorial](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/06-distilled-policy.html); [Bouteiller](https://arxiv.org/abs/2010.02966); [Wen](https://arxiv.org/abs/2010.14876). Copycat risk until DAgger | hours, no recollection |
| F3 | Auxiliary head predicting cap_from_ee, rim_from_ee and cap_up from image features, weight ~0.1 | `student.py` (labels in the shards' `privileged`) | Grasp 84 → 100% ([James](https://arxiv.org/abs/1707.02267)); [DextrAH-G](https://arxiv.org/abs/2407.02274) | hours |
| F4 | DAgger with teacher relabelling. Roll out the student with the D2 harness, label every visited state with `teacher(obs["actor"])`, drive half the envs with the teacher (α = 0.5), randomize delay, aggregate with the existing shards, fine-tune from bc_dead3700, repeat for a few rounds. Later, scale up with rsl_rl `Distillation` + mjlab `CameraSensor` (`src/mjlab/sensor/camera_sensor.py`), but only after D5-style checks show mujoco_warp renders match the twin, because the flat-render student scored 0/20 | D2 harness, `student.py`; `thirdparty/mjlab/.venv` rsl_rl 5.5.0 | +27 pts ([rebar](https://arxiv.org/abs/2609.20477)), +48 ([LBC](https://arxiv.org/abs/1912.12294)); α = 0.5 from [VIRAL](https://arxiv.org/abs/2511.15200); removes copycat | 2–4 days |
| F5 | Choose checkpoints and hyperparameters by closed-loop success on fixed seeded stages, not val MSE | D2 harness | [SIMPLER](https://arxiv.org/abs/2405.05941), [robomimic](https://arxiv.org/abs/2108.03298) | trivial once D2 exists |
| F6 | Encoder: stride 2 only in the first one or two layers plus spatial softmax (or ResNet-18 + GroupNorm + spatial softmax); raise grip resolution | `student.py`, `collect.py` `CAMS`, `student_policy.py` | [Levine](https://arxiv.org/abs/1504.00702), [robomimic](https://arxiv.org/abs/2108.03298), [Diffusion Policy](https://arxiv.org/abs/2303.04137) | ~1 day + recollection if resolution changes |
| F7 | Loss and head: L1/Huber or Gaussian NLL; optionally 4–8-step chunks of absolute targets with temporal ensembling | `student.py`, `student_policy.py` | [ACT](https://arxiv.org/abs/2304.13705), [DextrAH-RGB](https://arxiv.org/abs/2412.01791) | ~1 day |
| F8 | Proprio dropout or a narrower joint embedding, if D6 shows joints-only nearly matches 0.090 | `student.py` | [GAP](https://arxiv.org/abs/2602.12032), [Hsu et al.](https://arxiv.org/abs/2203.12677) | hours |
| F9 | Teacher side: retrain or fine-tune without `fingers_on_cap` (and with a noised `cap_up` if D6 shows images cannot resolve it), or add a SITT-style KL penalty; then redistil | `cap_to_mug_cfgs.py` `_state_terms` | [SITT](https://arxiv.org/html/2412.09149v2) 0.74 → 0.88; [RAIL](https://arxiv.org/html/2611.12251); [A2D](https://arxiv.org/abs/2012.15566) | one RL run + redistil |
| F10 | Student RL fine-tuning: PPO with an asymmetric critic on the teacher's observation, critic warm-up with the actor frozen, and a decaying BC/KL term | in-loop rendering at scale | [DoorMan](https://arxiv.org/abs/2512.01061) 50–70 → 80.8–85.8; [Xing](https://arxiv.org/abs/2403.12203) 57–64 → 76–85; naive BC → PPO collapsed to 4–21 | a week+ |

F10 comes last for a reason. Vanilla bootstrapping (DAgger init plus standard PPO) first improved and then degraded in Xing et al. RL fine-tuning is most useful once DAgger has reached its plateau and the remaining failures look observability-bound.

## Conclusion

The most useful change is in how the problem is framed. "My RGB student is 61 points worse than my teacher" is really two separate problems:
- **The student was trained for a world with exactly one delay, one physics engine and one resampling chain, and was tested in another.** Your own model_2800 run shows that one frame of timing mismatch costs about a third of a strong teacher's success.
- **An offline-cloned student loses about a third of its teacher in its home environment.** The literature attributes this to covariate shift, which DAgger fixes, and to unobservable state, which better inputs or a more imitable teacher fix.

Your dataset already contains the hidden variables: the integrated target, joint velocity and the full privileged vector. So the imitation-gap question can be settled offline in an afternoon, before any new rollout. The D2 harness (the student in mjlab with per-step CPU rendering) does double duty: it is the missing diagnostic and the DAgger rollout engine.

Two open questions will set the ceiling, and nothing here answers them yet:
- **How much of the twin gap is physics rather than timing?** D1 plus the delay ladder settles it.
- **Is jaw-closure timing driven by contact the student cannot sense?** D4 aligned to first contact, plus the D6 contact ablation, settles it. If so, the cheaper fix is a teacher retrained without contact sensors, not a cleverer student.

## Sources

| Short name | Citation |
|---|---|
| Ross 2011 | Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011. [link](https://proceedings.mlr.press/v15/ross11a.html) |
| DART | Laskey et al. DART: Noise Injection for Robust Imitation Learning. CoRL 2017. [arXiv:1703.09327](https://arxiv.org/abs/1703.09327) |
| LBC | Chen et al. Learning by Cheating. CoRL 2019. [arXiv:1912.12294](https://arxiv.org/abs/1912.12294) |
| Rebar | Sun et al. 2026, privileged-teacher distillation for rebar insertion. [arXiv:2609.20477](https://arxiv.org/abs/2609.20477) |
| Xing 2024 | Xing et al. Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight. CoRL 2024. [arXiv:2403.12203](https://arxiv.org/abs/2403.12203) |
| AAC | Pinto et al. Asymmetric Actor Critic for Image-Based Robot Learning. 2017. [arXiv:1710.06542](https://arxiv.org/abs/1710.06542) |
| VIRAL | He et al. (NVIDIA) 2025. [arXiv:2511.15200](https://arxiv.org/abs/2511.15200) |
| DoorMan | NVIDIA 2025, humanoid door opening. [arXiv:2512.01061](https://arxiv.org/abs/2512.01061) |
| UniDexGrasp++ | Wan et al. ICCV 2023. [arXiv:2304.00464](https://arxiv.org/abs/2304.00464) |
| Isaac Lab SO-101 | NVIDIA, Getting Started with Isaac Lab: distilled policy. [docs](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/06-distilled-policy.html) |
| rsl_rl | leggedrobotics/rsl_rl, `algorithms/distillation.py`. [GitHub](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/distillation.py) |
| DemoStart | Bauza et al. 2024. [arXiv:2409.06613](https://arxiv.org/abs/2409.06613) |
| ResiP | Ankile et al. From Imitation to Refinement. 2024. [arXiv:2407.16677](https://arxiv.org/abs/2407.16677) |
| DextrAH-G | Lum et al. CoRL 2024. [arXiv:2407.02274](https://arxiv.org/abs/2407.02274) |
| DextrAH-RGB | Singh et al. 2024. [arXiv:2412.01791](https://arxiv.org/abs/2412.01791) |
| ADVISOR | Weihs et al. Bridging the Imitation Gap by Adaptive Insubordination. NeurIPS 2021. [arXiv:2007.12173](https://arxiv.org/abs/2007.12173) |
| A2D | Warrington et al. Robust Asymmetric Learning in POMDPs. ICML 2021. [arXiv:2012.15566](https://arxiv.org/abs/2012.15566) |
| Scaffolder | Hu et al. Privileged Sensing Scaffolds Reinforcement Learning. ICLR 2024. [arXiv:2405.14853](https://arxiv.org/abs/2405.14853) |
| SITT | Messikommer et al. Student-Informed Teacher Training. ICLR 2025. [arXiv:2412.09149](https://arxiv.org/html/2412.09149v2) |
| RAIL | Mammadov et al. 2026. [arXiv:2611.12251](https://arxiv.org/html/2611.12251) |
| Lee 2020 | Lee et al. Learning Quadrupedal Locomotion over Challenging Terrain. Science Robotics 2020. [arXiv:2010.11251](https://arxiv.org/abs/2010.11251) |
| RMA | Kumar et al. RMA: Rapid Motor Adaptation. RSS 2021. [arXiv:2107.04034](https://arxiv.org/abs/2107.04034) |
| Fighting Copycat | Wen et al. Fighting Copycat Agents in Behavioral Cloning from Observation Histories. NeurIPS 2020. [arXiv:2010.14876](https://arxiv.org/abs/2010.14876) |
| PTP | Torne et al. Past-Token Prediction. 2025. [arXiv:2505.09561](https://arxiv.org/html/2505.09561) |
| Bouteiller | Bouteiller et al. Reinforcement Learning with Random Delays. ICLR 2021. [arXiv:2010.02966](https://arxiv.org/abs/2010.02966) |
| Tan 2018 | Tan et al. Sim-to-Real: Learning Agile Locomotion for Quadruped Robots. RSS 2018. [arXiv:1804.10332](https://arxiv.org/abs/1804.10332) |
| Sandha | Sandha et al. Sim2Real Transfer for Deep RL with Stochastic State Transition Delays (Time-in-State RL). CoRL 2020. [PMLR v155](https://proceedings.mlr.press/v155/sandha21a.html) |
| Imai (MMDR) | Imai et al. Vision-Guided Quadrupedal Locomotion in the Wild with Multi-Modal Delay Randomization. [arXiv:2109.14549](https://arxiv.org/abs/2109.14549) |
| Agarwal 2022 | Agarwal et al. Legged Locomotion in Challenging Terrains using Egocentric Vision. CoRL 2022. [arXiv:2211.07638](https://arxiv.org/abs/2211.07638) |
| MJWarp | MuJoCo Warp documentation. [docs](https://mujoco.readthedocs.io/en/latest/mjwarp/) |
| Isaac Lab solvers | Isaac Lab, Solver differences. [docs](https://isaac-sim.github.io/IsaacLab/main/source/concepts/solver_differences.html) |
| mjlab | Zakka et al. 2026. [arXiv:2601.22074](https://arxiv.org/abs/2601.22074) |
| SIMPLER | Li et al. Evaluating Real-World Robot Manipulation Policies in Simulation. 2024. [arXiv:2405.05941](https://arxiv.org/abs/2405.05941) |
| Parmar | Parmar, Zhang, Zhu. On Aliased Resizing and Surprising Subtleties in GAN Evaluation. CVPR 2022. [arXiv:2104.11222](https://arxiv.org/abs/2104.11222) |
| Azulay & Weiss | Why do deep convolutional networks generalize so poorly to small image transformations? JMLR 2019. [arXiv:1805.12177](https://arxiv.org/abs/1805.12177) |
| robomimic | Mandlekar et al. What Matters in Learning from Offline Human Demonstrations. CoRL 2021. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298) |
| Diffusion Policy | Chi et al. RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) |
| ACT | Zhao et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) |
| Levine 2016 | Levine et al. End-to-End Training of Deep Visuomotor Policies. JMLR 2016. [arXiv:1504.00702](https://arxiv.org/abs/1504.00702) |
| PRISM | Chen et al. Rethinking Implicit Spatial Representation in Visuomotor Policy Learning. 2026. [arXiv:2606.15232](https://arxiv.org/abs/2606.15232) |
| DrQ | Kostrikov, Yarats, Fergus. Image Augmentation Is All You Need. 2020. [arXiv:2004.13649](https://arxiv.org/abs/2004.13649) |
| James 2017 | James, Davison, Johns. Transferring End-to-End Visuomotor Control from Simulation to Real World. CoRL 2017. [arXiv:1707.02267](https://arxiv.org/abs/1707.02267) |
| Hsu 2022 | Hsu et al. Vision-Based Manipulators Need to Also See from Their Hands. ICLR 2022. [arXiv:2203.12677](https://arxiv.org/abs/2203.12677) |
| GAP | Lu et al. When would Vision-Proprioception Policies Fail in Robotic Manipulation? ICLR 2026. [arXiv:2602.12032](https://arxiv.org/abs/2602.12032) |
