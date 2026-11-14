# Privileged-teacher to vision-student distillation: BC (offline) vs DAgger (on-policy), and RL fine-tuning of the student

## Q1. Which major works distill a privileged teacher into a vision or partial-observation student, what method do they use, and which report BC-only vs DAgger numbers?

### Takeaway
Nearly every high-profile privileged-teacher-to-vision-student system uses on-policy, DAgger-style distillation: the student acts and the teacher relabels the states it visits. The exceptions are DemoStart, ResiP, Lin et al. 2025 and RialTo's first stage, which use offline BC on large teacher datasets. Every paper that reports a BC-only vs DAgger ablation finds a large gap in DAgger's favour:
- LBC: 16% → 64%.
- Rebar insertion 2026: 68.9% → 96.1%.
- Vision quadrotor 2024: 0% → 57–64%.
- AAC real Pick task: 0/5 → 3/5.

Offline BC only matches the teacher when the teacher is smooth and learnable and the data volume is large (DemoStart, 99.0% vs 99.6%). Even then it can saturate well below the teacher: ResiP reached 73–80% against a 98% teacher, even at 100k trajectories.

### Cited Findings
**Explicit BC (offline) vs on-policy (DAgger) ablations**
- **Learning by Cheating (Chen et al., CoRL 2019).** Table 1 ablation on the CoRL2017 CARLA "navigation" condition (test town, test weather), success %:
  - "Direct" one-stage imitation: 20.
  - Two-stage, off-policy, no white-box: 16.
  - Two-stage, on-policy only: 64.
  - Two-stage, white-box only (supervising all command branches): 96.
  - White-box + on-policy: 100.

  The vision student sees a 384×160 RGB image. It is first trained on the teacher's trajectories, then "online via DAgger, using the privileged agent as an oracle". The authors state that the DAgger stage "alone – without pre-training on the original trajectories – works equally well in the final accuracy" but is slower. On NoCrash (CARLA 0.9.5, test town) LBC reaches 100/94/85% (Empty/Regular/Dense). The authors also note that prior direct imitation approaches "heavily relied on trajectory noise injection". — [Chen et al. 2019, arXiv:1912.12294](https://arxiv.org/abs/1912.12294)
- **Rebar insertion, arm, multi-view RGB (Sun et al., arXiv Sept 2026, McGill/Princeton/UW).**
  - Recipe: a privileged teacher is distilled with a hybrid of online DAgger and BC. The loss is MSE over a mixture of observation distributions ρ = β·ρ_teacher + (1−β)·ρ_student, with β = 0.5 "following the ratio ablation of" VIRAL.
  - Ablation: pure BC (no DAgger) gives 68.9% sim success vs 96.1% for the full recipe, "a drop we attribute to the covariate shift of a student that is never asked to act from its own states."
  - Context: teachers converge to 96–98%; real-world success is 91.3% over 150 rollouts; removing appearance randomization collapses sim success to 2.8%.

  — [Sun et al. 2026, arXiv:2609.20477](https://arxiv.org/abs/2609.20477)
- **Vision-based agile quadrotor flight (Xing et al., CoRL 2024).** Three stages: state RL teacher → IL student → RL fine-tuning. Success rates (Figure-8 / SplitS / Kidney tracks), with pixel inputs:
  - BC: 0 / 0 / 0 (crash).
  - DAgger: 63 / 57 / 64.
  - PIRLNav (BC init + RL): 21 / 4 / 18.
  - Their method (DAgger + adaptive RL): 85 / 76 / 82.

  With gate-corner inputs, BC is again 0/0/0 and DAgger is 57/52/60. — [Xing et al. 2024, arXiv:2403.12203](https://arxiv.org/abs/2403.12203)
- **Asymmetric Actor-Critic (Pinto et al., 2017).** Real-robot Table I, image-based sim-to-real:
  - Pick: Asym HER 5/5, vanilla BC 0/5, DAgger BC 3/5.
  - Forward Push: 5/5, 1/5, 0/5.
  - Block Move: 5/5, 0/5, 0/5.

  In simulation, DAgger "learns much faster initially" but "saturates in performance and is lower than" asymmetric HER. The authors suggest that "behaviour cloning would fail if the expert policy depends on information contained in the full state but not in the partial observation." — [Pinto et al. 2017, arXiv:1710.06542](https://arxiv.org/abs/1710.06542)
- **VIRAL (He et al., NVIDIA; arXiv Nov 2025; CVPR 2026), humanoid loco-manipulation, RGB 108×192 student.**
  - Method: DAgger + BC mixture, where α = the "fraction of environments that follow the teacher policy during data collection." α=0 is pure DAgger, α=1 is pure BC, and the default is α=0.5.
  - Ablation is qualitative in the text: pure BC gives "fast loss reduction but produces a brittle policy" that does poorly in Isaac-to-MuJoCo and real tests. α=0.5 "slows optimization slightly but substantially improves deployment success rate." Exact numbers are only in Fig. 11, which I could not read.
  - Results: teacher ~95% in sim; real robot 54/59 consecutive successes.

  — [He et al. 2025, arXiv:2511.15200](https://arxiv.org/abs/2511.15200)
- **UniDexGrasp++ (Wan et al., ICCV 2023)** reports that UniDexGrasp had shown "DAgger outperforms GAIL and DAPG for policy distillation". Its Table 1 gives success % (train / test seen categories / test unseen categories):
  - State teacher: 87.9 / 84.3 / 83.1.
  - Vision student after DAgger distillation: 77.4 / 72.6 / 68.8.
  - After vision-based RL fine-tuning (iterative generalist-specialist learning): 85.4 / 79.6 / 76.7.

  — [Wan et al. 2023, arXiv:2304.00464](https://arxiv.org/abs/2304.00464)

**Works that use DAgger and report teacher-student gaps but no BC ablation**
- **RMA (Kumar et al., RSS 2021).** The phase-2 adaptation module is trained "with on-policy data (similar to Ross et al.)". The base policy is unrolled with ẑ from the current, initially random module, and the loop repeats until convergence. The stated rationale is that a dataset from the expert-with-true-z "will contain examples of only good trajectories", so the module "would not be robust to deviations from the expert trajectory." — [Kumar et al. 2021, arXiv:2107.04034](https://arxiv.org/abs/2107.04034)
- **Lee et al. 2020 (Science Robotics, quadruped).** "We employ the dataset aggregation strategy (DAgger)... training data is generated by rolling out trajectories by the student policy. For each visited state, the teacher policy computes its embedding and action vectors." The loss is MSE on actions plus a latent-embedding term. — [Lee et al. 2020, arXiv:2010.11251](https://arxiv.org/abs/2010.11251)
- **A System for General In-Hand Object Re-Orientation (Chen, Xu, Agrawal, CoRL 2021).**
  - Method: the student is distilled with DAgger by minimizing KL(π_E‖π_S). "While Dagger typically keep all the state-action pairs... we do Dagger in an online fashion where π_S only learns from the latest rollout data."
  - Scale: 40K parallel envs for state policies, 360 envs for the point-cloud vision student.
  - Results: per-object vision students exceed 80% success on non-symmetric objects and 50% on symmetric ones.

  — [Chen et al. 2021, arXiv:2111.03043](https://arxiv.org/abs/2111.03043)
- **Visual Dexterity (Chen et al., Science Robotics 2023).** The student (depth point cloud) is trained "using DAGGER to imitate the teacher." Training with rendered point clouds took "20+ days on an NVIDIA V100." They therefore used a two-stage student: synthetic complete point clouds first (<3 days, a 7× speedup), then fine-tuning with rendered point clouds (~1 day). "The student policies are able to track the teacher policy's performance closely" (ECDF curves, no single scalar). — [Chen et al. 2023, arXiv:2211.11744](https://arxiv.org/abs/2211.11744)
- **HORA (Qi et al., CoRL 2022).** Student = adaptation module on 30-step proprioceptive history. It is trained by executing the policy with the *predicted* extrinsics, storing the ground truth, and iterating (on-policy), with 16384 envs in IsaacGym. Sim results within the training distribution:
  - Student vs teacher: RotR 222.27 vs 233.71; TTF 0.82 vs 0.85.
  - Out-of-distribution: RotR 160.60 vs 165.07; TTF 0.68 vs 0.71.
  - Baselines: domain-randomization-only 176.12 / 0.81; NoAdapt 90.89 / 0.65.

  No BC baseline. — [Qi et al. 2022, arXiv:2210.04887](https://arxiv.org/abs/2210.04887)
- **DextrAH-G (Lum et al., CoRL 2024)**, depth student, 160×120 depth image.
  - Method: "online version of Dagger", loss L = ||â − a||² + β||x̂_obj − x_obj||² with β = 0.1, an auxiliary object-position head.
  - Sim results: 99% per-batch success, matching the teacher; per object, 80% for the student vs 85% for the privileged teacher.

  — [Lum et al. 2024, arXiv:2407.02274](https://arxiv.org/abs/2407.02274)
- **DextrAH-RGB (Singh et al., 2024/ICRA 2025)**, stereo RGB student.
  - Method: "online implementation of DAgger", loss = KL(π_student‖π_teacher) + L2 auxiliary object-position loss. Across 4 seeds, "KL-trained policies always out-performed their L2 counterparts."
  - Best student (fine-tuned ResNet-18, stereo, transformer fusion, LSTM): 0.89 of teacher performance in sim, 2.5 cm position error. A frozen ResNet gives 0.71; another ablation is reported as 0.83 (the extracted text is ambiguous about whether it is monocular or scratch-ResNet).
  - Compute: teacher 62 h on 8×H100; student 53 h on 4×L40S with Isaac Lab tiled rendering.
  - Real bin-packing: 77% (stereo) vs 87% for depth-based DextrAH-G. Sim-to-real is "highly variable" across seeds with similar sim metrics.
  - No BC vs DAgger ablation.

  — [Singh et al., arXiv:2412.01791](https://arxiv.org/abs/2412.01791)
- **DoorMan (NVIDIA, arXiv Dec 2025), humanoid door opening.**
  - Method: RGB student distilled "using DAgger... compared to behavioral cloning which only covers the teacher distribution."
  - Gap: "When teacher policies can consistently achieve 80–90% success rate, the initial student policy performance stales at 50–70%, suggesting a non-recoverable observability gap."
  - After GRPO fine-tuning, the student reaches 80.8–85.8%.

  — [DoorMan, arXiv:2512.01061](https://arxiv.org/abs/2512.01061)
- **RialTo (Torne et al., RSS 2024)**, point-cloud student, arm manipulation.
  - Stage 1 is offline BC distillation on a mixture sampled at ¼ each: 15000 full-point-cloud teacher trajectories, 5000 camera-view trajectories, 2000 trajectories with distractors, and 15 real demos.
  - Stage 2 is "a step of DAgger" (roll out the student, relabel with the state teacher), with the DAgger data, distractor sim data and real data sampled at ⅓ each.
  - Distillation takes about 24 h on an RTX 2080/3090.

  — [Torne et al. 2024, arXiv:2403.03949](https://arxiv.org/abs/2403.03949)
- **Isaac Lab SO-101 tutorial (NVIDIA, the same arm as the user's).**
  - Task: pick a vial and place it in a rack.
  - Student inputs: wrist RGB rendered at 80×60 and projected to 64×48, two stacked frames (6×48×64), plus 24-D proprioception (joint pos, joint vel, joint target, previous action).
  - Architecture: CNN [16,32,32] + MLP [256,128].
  - Training: rsl_rl DAgger-style distillation (the student acts, the frozen teacher labels), MSE loss, 1,024 envs, 1,600 iterations × 32 steps/env, 4 epochs, lr 5e-4, about 28 minutes on an RTX PRO 6000 Blackwell Max-Q.
  - Result over 1,024 episodes: teacher 99.61% vs student 99.02%, a 0.59-point gap. The 1,024 episodes repeat 128 home starts 8×.

  — [NVIDIA Isaac Lab tutorial](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/06-distilled-policy.html)

**Works that use offline BC distillation (no DAgger)**
- **DemoStart (Bauza et al., DeepMind, 2024).**
  - Method: "Instead of interactively querying the teacher policy... we generate an offline dataset of trajectories from the teacher, and use behavior cloning". The data is "thousands of successful episodes" with physics, perturbation and visual domain randomization.
  - Sim plug insertion: teacher 99.6% → RGB student 99.0%.
  - Distilling a jerky SAC-X teacher (99.2%) instead gives only 20.4%, "as the distilled policy struggles to learn from the jerky, non-smooth behaviors of SAC-X policy." Plug lift: 99.3% (DemoStart) and 98.3% (SAC-X).
  - Real robot: DemoStart distillation 97% / 64% / 97% vs SAC-X distillation 20% / 1%.

  — [Bauza et al. 2024, arXiv:2409.06613](https://arxiv.org/abs/2409.06613)
- **ResiP (Ankile et al., 2024/ICRA 2025).** Offline BC of successful RL-teacher trajectories into an RGB student:
  - ~1,000 trajectories give 73% real success on one-leg assembly vs a 98% teacher.
  - A *state*-based student scales from 78% at 10k trajectories to only 80% at 100k. "The modality shift is not the primary cause of the performance gap", pointing to "a fundamental limitation in the policy distillation process itself."
  - They note "DAgger-style online distillation might improve performance" but did not test it for the distillation.
  - Separately, BC of a diffusion policy saturates at 56% with 10,000 demos on round-table assembly, and DP-DAgger plateaus at 71%.

  — [Ankile et al. 2024, arXiv:2407.16677](https://arxiv.org/abs/2407.16677)
- **Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids (Lin, Sachdev, Fan, Malik, Zhu, 2025).**
  - Method: specialist PPO teachers use an asymmetric actor-critic. Each is rolled out for 5000 steps × 100 envs, successful trajectories are filtered and saved, and they are distilled offline into a Diffusion Policy. The student sees joint states, 3D object position and egocentric depth.
  - Gaps: no BC vs DAgger ablation and no teacher-vs-student numbers.
  - Real ablation: depth + object position gives 10/10 pickup on all tasks, while depth only gives 0–2/10.
  - Divide-and-conquer granularity, real success: mix 90.0%, shape 63.3%, single 40.0%, all 23.3%.

  — [Lin et al. 2025, arXiv:2502.20396](https://arxiv.org/abs/2502.20396)
- **DexPoint (Qin et al., CoRL 2022)** is *not* a teacher-student method in its main pipeline. It trains PPO directly on observed point clouds augmented with an "imagined" hand point cloud (only an appendix mentions "distilling experiments"). It is an alternative to distillation: direct RL on a geometric observation. — [Qin et al. 2022, arXiv:2211.09423](https://arxiv.org/abs/2211.09423)

### Inferences
- The user's 27% vs 88% (and earlier 57% vs 92%) gap is in line with published offline-BC outcomes: ResiP 73% vs 98%, DoorMan's pre-RL 50–70% vs 80–90%, rebar pure BC 68.9% vs 96%. It is far from DAgger-trained results: the SO-101 Isaac Lab tutorial 99.0% vs 99.6%, DextrAH-G 80% vs 85%, rebar 96.1%.
- The closest published analogue (the Isaac Lab SO-101 vial pick-and-place) differs from the user's student in four ways:
  - DAgger instead of offline BC.
  - Two stacked frames instead of one.
  - Joint velocity, joint target and previous action in the input; the user's student has none of these.
  - A wrist camera at the same 64×48 resolution.

  This suggests the user's resolution is not the main limitation, and that adding history and velocity / last action plus on-policy relabelling is a low-risk first change.
- DemoStart's SAC-X result (99.2% teacher → 20.4% student) shows that teacher action smoothness matters a lot for offline BC. A PPO teacher with high-frequency or bang-bang actions may be hard to clone from one RGB frame.
- Auxiliary privileged-state prediction heads are used in DextrAH-G (object position, β=0.1), DextrAH-RGB, and AAC's "bottleneck" full-state regression. They are a common, cheap addition. For the user they could predict cap position, cap upside flag, and mug rim position.

### Gaps
- VIRAL's quantitative BC vs DAgger vs mixture numbers are only in Fig. 11, which I could not extract.
- I found no BC-vs-DAgger ablation in DextrAH-G, DextrAH-RGB, Lin et al. 2025, Visual Dexterity, HORA, RMA or Lee 2020. They simply use DAgger, or offline diffusion BC in Lin et al.
- I found no peer-reviewed SO-100/SO-101 or mjlab teacher-student paper. The closest is the NVIDIA Isaac Lab SO-101 tutorial, which is documentation, not peer-reviewed, and evaluated on 128 home starts.
- UniDexGrasp (Xu et al., CVPR 2023): its claim that DAgger beats GAIL/DAPG for distillation is cited via UniDexGrasp++. I did not read the UniDexGrasp table directly.

## Q2. Why does offline BC from a teacher underperform DAgger, and does DART-style noise injection close the gap?

### Takeaway
Offline BC's error bound grows quadratically with horizon T (T²ε), because once the student drifts off the teacher's state distribution it is never corrected. DAgger trains on the student's own state distribution and gets a bound roughly linear in T (uTε). DART narrows the gap only for errors that look like the injected noise near the teacher's trajectories. It cannot fix systematic drift from partial observability (the "imitation gap"). In GPU sim, where the teacher is free to query, DART's main advantage over DAgger (no human relabelling) disappears.

### Cited Findings
- **Ross, Gordon & Bagnell (AISTATS 2011).**
  - For BC (supervised training on expert states) with error ε on the expert distribution, J(π) ≤ J(π*) + T²ε. A learner "can make as many as T²ε mistakes in expectation", and this "behaves as Θ(T²ε) for small ε."
  - With the forward algorithm or DAgger: J(π) ≤ J(π*) + uTε (Theorem 2.2). For DAgger, there exists π̂ in the sequence with J(π̂) ≤ J(π*) + uTε_N + O(1) (Theorem 3.2), where u bounds how much one deviation costs (u is O(1) when the expert recovers quickly).
  - DAgger algorithm: "Aggregate datasets: D ← D ∪ D_i. Train classifier π̂_{i+1} on D".

  — [Ross et al. 2011](https://proceedings.mlr.press/v15/ross11a.html)
- **DART (Laskey et al., CoRL 2017)** injects noise into the *supervisor's* policy during collection, so the data includes recovery, and tunes the noise level to match the trained robot policy's error.
  - On MuJoCo it is up to 3× faster in computation time than DAgger on Humanoid.
  - It lowers the supervisor's cumulative reward by only 5% during collection, whereas DAgger's executed policies have 80% less cumulative reward than the supervisor's.
  - On real grasping in clutter (Toyota HSR, human supervisors) it gives "a 62% performance increase over Behavior Cloning."

  — [Laskey et al. 2017, arXiv:1703.09327](https://arxiv.org/abs/1703.09327)
- **LBC** notes that prior direct imitation approaches "heavily relied on trajectory noise injection". Their own ablation shows on-policy student rollouts adding +48 points (16 → 64) on top of off-policy two-stage training. — [Chen et al. 2019](https://arxiv.org/abs/1912.12294)
- **RMA's rationale for on-policy data:** expert-only trajectories "will contain examples of only good trajectories... would not be robust to deviations from the expert trajectory, which will happen often during deployment." — [Kumar et al. 2021](https://arxiv.org/abs/2107.04034)
- **The "imitation gap" (ADVISOR, Weihs et al., NeurIPS 2021):** "when the teaching agent makes decisions with access to privileged information that is unavailable to the student, this information is marginalized during imitation learning, resulting in an 'imitation gap' and, potentially, poor results." — [Weihs et al., arXiv:2007.12173](https://arxiv.org/abs/2007.12173)
- **A2D (Warrington et al., ICML 2021)** shows that plain asymmetric imitation can learn a bad partially observed policy (their example: always crossing a lake despite being unable to see weak ice). It proposes adaptive asymmetric DAgger, which jointly trains the expert so it becomes imitable by the trainee. — [Warrington et al., arXiv:2012.15566](https://arxiv.org/abs/2012.15566)
- **Scaffolder (Hu et al., ICLR 2024)** notes that the BC objective "assumes the optimal student and teacher have the same action distributions", and that this "is violated" in Wrist-Camera Pick-and-Place, Occluded Pick-and-Place and Blind Pick. DreamerV3+BC is outperformed by other privileged-sensing methods. — [Hu et al., arXiv:2405.14853](https://arxiv.org/abs/2405.14853)
- **ResiP's** offline distillation saturates (state student 78% at 10k → 80% at 100k trajectories vs 98% teacher). This shows more offline data alone does not close the gap. — [Ankile et al. 2024](https://arxiv.org/abs/2407.16677)
- **DoorMan** describes the residual 50–70% vs 80–90% gap after DAgger as "a non-recoverable observability gap", which RL fine-tuning then closes. — [DoorMan, arXiv:2512.01061](https://arxiv.org/abs/2512.01061)

### Inferences
- The user's student has a single frame, no joint velocity, no last action and no history. The teacher conditions on velocities, a cap upside/downside flag and finger-cap contact sensors. Several of these are not inferable from one 64×48 / 96×72 frame. This creates an imitation gap that neither DART nor DAgger fixes; only better student observations (frame stack, velocity, last action, recurrence), auxiliary state prediction, or RL on the student can.
- The validation-MSE plateau (0.09 vs 0.053 train) fits both explanations: label ambiguity, where identical images map to different teacher actions, and over-fitting to 2000 episodes.
- The user's DART noise is per-episode fixed at std 0/0.1/0.3, not fitted to the learner's error. DART's design assumes the noise approximates the robot policy's error distribution. Systematic student errors (e.g., approaching the cap at a consistent offset because depth is ambiguous) are not Gaussian perturbations around the teacher, so DART covers them poorly. This is consistent with the user's failures concentrating in approach/grasp (cap pushed away, mug knocked over).
- In GPU sim the teacher is a cheap function call. DART's main motivation (human supervisors find DAgger relabelling tedious, plus DAgger's poor reward during collection) mostly disappears, so DAgger or a DAgger+BC mixture is the default choice in all recent GPU-sim works.

### Gaps
- I found no paper that directly compares DART vs DAgger for privileged-state → RGB distillation in GPU sim.
- DART's MuJoCo numbers come from the abstract. I did not extract per-task success tables.

## Q3. Practical DAgger recipes for distillation: β schedules and mixing, relabelling, aggregation vs on-policy-only, iterations, and GPU rendering cost

### Takeaway
The modern GPU-sim recipe is "online DAgger":
- The student (sometimes a fixed fraction of envs run the teacher, α≈0.5) acts in thousands of parallel envs.
- The teacher labels every visited state.
- The student is trained on just that rollout with MSE, Huber or KL, plus optional auxiliary state losses.
- The data is discarded, so there is no aggregation, and there is no explicit β decay schedule.

An optional offline-BC warm start speeds early training. Rendering is the cost driver: depth/RGB students take tens of GPU-hours (DextrAH-RGB 53 h on 4×L40S; VIRAL 64 GPUs; Visual Dexterity 20+ days on a V100 before switching to synthetic point clouds). The small SO-101 case is quick (28 min on one GPU at 64×48).

### Cited Findings
- **rsl_rl `Distillation` (used by Isaac Lab):**
  - The student acts (`self.transition.actions = self.student(obs, stochastic_output=True)`) and the teacher only labels (`privileged_actions = self.teacher(obs)`).
  - Loss is MSE by default, Huber optional.
  - It runs `num_learning_epochs` over the rollout, with `gradient_length` setting truncated BPTT for recurrent students.
  - `self.storage.clear()` after every update, so there is no replay or aggregation across iterations.
  - There is no β mixing; the docstring cites Ross et al. 2011.

  — [rsl_rl distillation.py](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/distillation.py)
- **Isaac Lab SO-101 distillation settings:** 1,024 envs, 32 steps/env per iteration, 1,600 iterations, 4 learning epochs, lr 5e-4, MSE, CNN [16,32,32] → MLP [256,128], 2 stacked 64×48 wrist frames. About 28 minutes on an RTX PRO 6000 Blackwell Max-Q. Student 99.02% vs teacher 99.61%. — [NVIDIA tutorial](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/06-distilled-policy.html)
- **Online-only (no aggregation) is common:** Chen et al. 2021: "we do Dagger in an online fashion where π_S only learns from the latest rollout data"; vision student with 360 envs, update every 50 rollout steps, batch size 30. — [Chen et al. 2021](https://arxiv.org/abs/2111.03043)
- **Mixture instead of a β schedule:**
  - VIRAL uses α = 0.5, meaning 50% of envs run the teacher and 50% the student, and all states are labelled by the teacher with MSE. They chose this because pure BC was brittle. Student training used 65,536 envs on 64 L40S GPUs, lr 2e-4, RGB 108×192. "Scaling from 1 to 64 GPUs consistently accelerates convergence" and improves stability.
  - The rebar paper copies β = 0.5 and measures +27.2 points over pure BC.

  — [VIRAL, arXiv:2511.15200](https://arxiv.org/abs/2511.15200); [Sun et al. 2026](https://arxiv.org/abs/2609.20477)
- **BC warm start then DAgger:**
  - LBC pre-trains on teacher trajectories, then runs DAgger. DAgger alone reaches the same final accuracy but slower; they also oversample high-loss "critical states" following Bastani et al.
  - RialTo does BC on 22k teacher trajectories, then one DAgger round.

  — [Chen et al. 2019](https://arxiv.org/abs/1912.12294); [Torne et al. 2024](https://arxiv.org/abs/2403.03949)
- **White-box / dense supervision helps even off-policy:** LBC's "white-box" supervision of all command branches gives 16 → 96 off-policy, which they describe as acting "as powerful data augmentation". The analogue for a manipulation student is querying the teacher on extra, counterfactual inputs. — [Chen et al. 2019](https://arxiv.org/abs/1912.12294)
- **Loss choice:** DextrAH-RGB found KL(π_student‖π_teacher) "always out-performed" L2 across 4 seeds; Chen 2021 also uses KL. DextrAH-G uses L2 actions + 0.1 × L2 object position. — [DextrAH-RGB](https://arxiv.org/abs/2412.01791); [DextrAH-G](https://arxiv.org/abs/2407.02274); [Chen et al. 2021](https://arxiv.org/abs/2111.03043)
- **Encoder choice (DextrAH-RGB):** a fine-tuned ImageNet ResNet-18 scored 0.89 of teacher performance vs 0.71 frozen; a training-from-scratch ablation was also lower (the extracted text gives 0.83 but is ambiguous). The rebar paper found pretraining gives 4–6× fewer distillation samples and a final 90–93% vs 81–85% from scratch. — [DextrAH-RGB](https://arxiv.org/abs/2412.01791); [Sun et al. 2026](https://arxiv.org/abs/2609.20477)
- **Rendering cost and throughput:**
  - **Isaac Lab paper (2025):**
    - State-based DextrAH teacher: more than 900,000 FPS on 8 GPUs with 16384 envs.
    - Perceptive DextrAH at 64×64: throughput "remains lower overall compared to state-based tasks due to the additional rendering cost."
    - RayCasterCamera is faster on a single GPU; TiledCamera parallelizes better and overtakes it at multi-GPU, large env counts.

    — [Isaac Lab, arXiv:2511.04831](https://arxiv.org/abs/2511.04831)
  - **ManiSkill3:** simulation + rendering "up to 30,000+ FPS" depending on task, with 2–3× less GPU memory than other platforms. At 128 envs, 3.5 GB vs Isaac Lab's 14.1 GB. ManiSkill3 also reports sim-to-real PPO directly from one 128×128 RGB camera for PickCube, with no teacher. — [Tao et al., arXiv:2410.00425](https://arxiv.org/abs/2410.00425)
  - **mjlab:** "High-fidelity RGB rendering is out of scope". It offers depth and raycast sensors plus "an experimental tiled-rendering camera". The recommended path is to train privileged policies and distill to vision "using external rendering". — [mjlab, arXiv:2601.22074](https://arxiv.org/abs/2601.22074)
  - **MuJoCo Warp:** a batch renderer described as a GPU ray-tracer over Warp BVH for many worlds, with per-camera resolution and RGB/depth output. The docs (as summarized by search) call it a "single hit raycaster optimized for high throughput and low fidelity". — [MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/)
  - **Visual Dexterity:** rendered point clouds made student training "20+ days on an NVIDIA V100". Synthetic mesh-sampled point clouds gave a 7× speedup (under 3 days), followed by about 1 day of fine-tuning on rendered clouds. — [Chen et al. 2023](https://arxiv.org/abs/2211.11744)

### Inferences
- For the user's mjlab setup, the cheapest on-policy route is a batched in-loop renderer: mjlab's experimental tiled camera or the MuJoCo Warp batch renderer, at the already-small 96×72 / 64×48 resolutions. The SO-101 Isaac Lab tutorial suggests about 1k envs at 64×48 suffices for a DAgger run of tens of minutes on one modern GPU.
- If in-loop rendering is too slow, a cheaper approximation:
  1. Roll out the current student on CPU MuJoCo renders (or a small env count).
  2. Relabel the visited states with the teacher.
  3. Aggregate with the existing 180k frames.

  That is classic DAgger with aggregation, as in RialTo's single DAgger round.
- Given VIRAL's α=0.5 and rebar's β=0.5, a reasonable default is to run half the envs with the teacher and half with the student, with all states teacher-labelled. Start from the current BC checkpoint, as LBC and RialTo do.

### Gaps
- I found no published absolute FPS for Isaac Lab tiled RGB at 64×64. Values are only in log-scale figures.
- I found no published throughput numbers for the mjlab tiled camera or the MJWarp batch renderer.
- None of the GPU-sim works report an explicit β-decay schedule (Ross's original β_i = p^(i−1)). All use either a pure student rollout or a fixed mixture, so I cannot cite schedule comparisons.

## Q4. RL fine-tuning of the student after distillation, and asymmetric actor-critic as an alternative

### Takeaway
RL on the student after DAgger consistently recovers much of the residual teacher-student gap when the gap comes from partial observability:
- DoorMan: 50–70% → 80.8–85.8% with GRPO.
- Quadrotor: DAgger 57–64% → 76–85% with PPO, an asymmetric critic and adaptive learning rate/clip.
- UniDexGrasp++: 77.4/72.6/68.8 → 85.4/79.6/76.7.

Naive BC-init followed by PPO tends to collapse (catastrophic forgetting): PIRLNav gave 4–21% where DAgger gave 57–64%. Use critic warm-up, conservative updates, or a retained imitation term. Asymmetric actor-critic alone (RL from pixels with a privileged critic) works for simple tasks (AAC, ManiSkill3 PickCube) but is sample-hungry. Scaffolder and other privileged-sensing methods suggest combining privileged signals in several places beats any single route.

### Cited Findings
- **Xing et al. (CoRL 2024): three stages (state RL teacher → DAgger student → RL fine-tuning with an asymmetric critic).**
  - The challenges they name: "the critic function requires interactions to adapt the pre-trained actor, necessitating a 'warm-up' process", and updates must be kept small "to prevent catastrophic forgetting." Their algorithm ties learning rates and PPO clip range to rollout performance.
  - Success: Figure-8 / SplitS / Kidney 85 / 76 / 82 vs DAgger 63 / 57 / 64. PIRLNav (BC init + RL, non-adaptive) reaches only 21 / 4 / 18. RL from scratch on pixels is 0%.
  - "Vanilla bootstrap" (DAgger init + standard RL) improved at first "but deteriorated with increasing timesteps."
  - With a fixed 10M-sample budget, a 60% IL / 40% RL split was best.
  - Under perceptual disturbance at 5%, success is 91% vs 33% for IL only.

  — [Xing et al. 2024, arXiv:2403.12203](https://arxiv.org/abs/2403.12203)
- **DoorMan (2025):** DAgger student, then GRPO, an actor-only PPO variant without a value function, with a binary success signal. Student sub-task success goes from 50–70% to 80.8–85.8%, plateauing "at the teacher upper bound". The paper says GRPO lets the student learn behaviours such as "keeping manipulated objects centered in view". — [DoorMan, arXiv:2512.01061](https://arxiv.org/abs/2512.01061)
- **UniDexGrasp++ (ICCV 2023):** a DAgger-based distillation that *also* trains a critic, with loss = MSE(teacher action, student action) + MSE(V, GAE return). This lets the vision student be fine-tuned with PPO afterwards. Vision results go from 77.4/72.6/68.8 to 85.4/79.6/76.7 after iterative vision-based fine-tuning. Baselines PPO+DAgger reach only 20.6/17.2/15.0 and DAPG+DAgger 17.9/15.2/13.9. — [Wan et al. 2023](https://arxiv.org/abs/2304.00464)
- **TGRL (Shenfeld, Hong, Tamar, Agrawal, ICML 2023):** RL plus a teacher-imitation term whose weight is adjusted "dynamically and automatically". It compares the agent to a counterfactual reward-only agent, raising the teacher weight when imitation helps and lowering it otherwise. The abstract claims it outperforms baselines "without hyper-parameter tuning". — [Shenfeld et al., arXiv:2307.03186](https://arxiv.org/abs/2307.03186)
- **ADVISOR (NeurIPS 2021)** dynamically weights imitation and RL losses per state to bridge the imitation gap. It "outperforms pure imitation, pure reinforcement learning, as well as their sequential and parallel combinations" in gridworlds, multi-agent particle environments and 3D simulators. — [Weihs et al.](https://arxiv.org/abs/2007.12173)
- **Asymmetric Actor-Critic (Pinto et al. 2017):** the critic gets the full state and the actor gets images. Asymmetric HER is 5/5 on all three real tasks, where symmetric HER and asymmetric DDPG are 0/5. DAgger BC saturates below it in sim. An auxiliary bottleneck regressing the full state from images "improve[s] the stability and speed of training". — [Pinto et al. 2017](https://arxiv.org/abs/1710.06542)
- **Scaffolder (ICLR 2024):** uses privileged sensors in the critic, world model, exploration policy and representation. It "bridges 79% of the gap" between target-only and privileged observations. With 250K–5M samples it beats or matches AAC and RMA trained with 100M samples in 9 of 10 tasks. Plain DreamerV3 on target sensors outperforms some privileged methods "like AAC", and RMA "completely fails in Blind Pick". — [Hu et al. 2024](https://arxiv.org/abs/2405.14853)
- **Asymmetric critic in teachers:** Lin et al. 2025 and DextrAH-RGB both use asymmetric actor-critic PPO teachers, where the critic gets extra privileged state. — [Lin et al. 2025](https://arxiv.org/abs/2502.20396); [DextrAH-RGB](https://arxiv.org/abs/2412.01791)
- **Residual RL:** ResiP trains a small single-step residual policy with PPO on top of a frozen chunked BC policy. In sim this lifts precise-assembly success to about 98% for one-leg, where BC/DAgger saturate at 56%/71% on round-table. It was done on *state*, with vision obtained by offline distillation (73% real vs 98% teacher). — [Ankile et al. 2024](https://arxiv.org/abs/2407.16677)
- **Direct pixel RL without a teacher:** ManiSkill3 reports sim-to-real PPO from one 128×128 RGB camera + joint positions on PickCube, using GPU-parallel sim + rendering. — [Tao et al. 2024](https://arxiv.org/abs/2410.00425)

### Inferences
- For the user, a staged plan follows the published evidence:
  1. Switch to on-policy DAgger, or a 50/50 DAgger/BC env mixture.
  2. Give the student history, velocity and last action.
  3. Then fine-tune the student with PPO. Use an asymmetric critic that gets the teacher's privileged observations (warm the critic up first while the actor is frozen) and a decaying BC/KL-to-teacher term or conservative clip/learning rate. This guards against the collapse seen with vanilla bootstrap and PIRLNav.
- RL fine-tuning should mainly help on failures that come from observability rather than imitation error: approach and grasp, cap pushed away, mug knocked over. DoorMan reports that GRPO teaches the student to keep objects in view, which matches front-camera / wrist-camera occlusion during approach.
- A cheap alternative to full RL is UniDexGrasp++'s trick: train a value head during DAgger so the student starts PPO with a warmed-up critic.

### Gaps
- I found no quantitative TGRL results for privileged-teacher → vision-student manipulation; only the abstract-level claim was verified.
- I found no published DAPG-style (BC-augmented policy gradient) fine-tuning of a distilled RGB student on an arm pick-and-place with numbers. PIRLNav, used as a baseline in Xing et al., is the closest.
- I did not verify PIRLNav's original ObjectNav numbers. The figures above are as reproduced in Xing et al.'s quadrotor setting.
