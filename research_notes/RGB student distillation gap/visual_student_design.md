# Visual student design: encoder, resolution, augmentation, losses and action representation for precise-grasp RGB BC / distilled students

Context for the report writer: the user's student uses a 96x72 front and a 64x48 wrist RGB image. Each image goes through a 4-layer 3x3 stride-2 CNN (32-64-64-64), is flattened, then passed to Linear-256 + LayerNorm. Joints go through Linear-64. These are concatenated into a 512-256 MLP that outputs a 6-D per-frame delta on an integrated joint target. Training uses MSE, DrQ shift of 4 px, and per-episode DR, on 2000 teacher episodes (~180k frames) of offline BC. Train MSE is 0.053 vs val 0.09; closed-loop success is 27% vs the teacher's 88%. Failures are approach and grasp precision. Where a note says "recalled", the fact comes from the cited paper but its text was not re-fetched in this session.

## Q1. Encoder: flatten-CNN vs spatial softmax / keypoints vs ResNet-18+GN+spatial softmax vs pretrained (R3M, VC-1, DINOv2, CLIP)

### Takeaway
For precise manipulation from images, a ResNet-18 trained end to end with spatial-softmax pooling (and GroupNorm) is the most consistently strong choice in the evidence found. Shallow conv encoders lose 25-62% (relative) in robomimic. Spatial-softmax keypoints localise objects about 2-3.5x more accurately than alternatives in Levine et al. Frozen pretrained encoders (ImageNet, R3M, frozen CLIP) underperform end-to-end training. Pretrained encoders become competitive or best only when fine-tuned.

### Cited Findings
- robomimic (Mandlekar et al. 2021) uses ResNet-18 followed by a spatial-softmax layer per camera. Replacing it with a shallow conv net cut success by 25%-62% relative, which the authors read as showing "large-capacity visual encoders are crucial for visuomotor imitation." Using a GMM head with a ResNet encoder "appears to be uniformly better." — [robomimic, arXiv 2108.03298](https://arxiv.org/abs/2108.03298)
- robomimic image BC-RNN reaches Lift 100%, Can 98%, Square 82%, Transport 72%, Tool Hang 67.3% (proficient-human data). Image policies can match or beat low-dim state policies, e.g. Tool Hang 67.3% (image) vs 19.3% (low-dim). — [robomimic](https://arxiv.org/abs/2108.03298)
- Diffusion Policy uses "a standard ResNet-18 (without pretraining)", replacing global average pooling with spatial softmax and BatchNorm with GroupNorm. — [Diffusion Policy, arXiv 2303.04137](https://arxiv.org/abs/2303.04137)
- Diffusion Policy encoder ablation (sim Square-PH success rate):
  - ResNet-18: scratch 0.94, frozen ImageNet-21k 0.58, finetuned 0.92.
  - ResNet-34: scratch 0.92, frozen 0.40, finetuned 0.94.
  - CLIP ViT-B/16: scratch 0.22, frozen 0.70, finetuned 0.98.
  - The authors find that "finetuning the pretrained vision encoder with a small learning rate ... gives the best performance overall." — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- On real Push-T (IoU / success), end-to-end reaches 0.80 / 0.95, R3M 0.66 / 0.80, and ImageNet 0.24 / 0.15. The authors conclude that "end-to-end training is still the most effective way to incorporate visual observation." — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- Levine et al. 2016 (end-to-end visuomotor policies) use 3 conv layers with no pooling, with 32 maps of 109x109. A spatial softmax plus expected-position layer turns these into 32 (x,y) feature points, concatenated with robot configuration, then FC 40-40 to torques (~92k params). For object-pose regression, spatial softmax + feature points reached 1.30 ± 0.73 cm test error, vs 2.59, 4.75 and 3.71 cm for the other architectures compared in their Table 3. The two-stage pose-prediction baseline "achieves poor performance" and was successful only on the coat-hanger task. Conv layers were pretrained by regressing target-object pose (~1000 images) "to learn the appearance of the objects prior to learning the behavior"; no with/without ablation is reported. — [Levine et al., arXiv 1504.00702](https://arxiv.org/abs/1504.00702)
- A 2026 study (Chen et al., "Rethinking Implicit Spatial Representation in Visuomotor Policy Learning") finds that spatial-softmax pooling gives "compact, stable spatial representations" that outperform feature-value representations with far fewer dimensions. It also finds that repeated downsampling weakens fine spatial information, especially at low input resolution, which limits spatial softmax. Their multiscale encoder (PRISM) raised low-resolution, high-precision ToolHang success from 5.0% to 13.4%. — [arXiv 2606.15232](https://arxiv.org/abs/2606.15232)
- Hansen et al. 2022 report that a shallow ConvNet trained from scratch with data augmentation is "surprisingly competitive" with PVR, MVP and R3M. They attribute the pretrained models' shortfall to "a significant domain gap between the pre-training datasets and current benchmarks", which is "alleviated by finetuning." Their benchmark used 84x84 RGB and upsampled to 224 for pretrained encoders. — [Hansen et al., arXiv 2212.05749](https://arxiv.org/abs/2212.05749)
- DextrAH-RGB (a sim-distilled RGB student for dexterous grasping) uses an ImageNet-pretrained ResNet-18 with the last two layers removed, **finetuned**, at 320x240 stereo. The best student (finetuned stereo ResNet + cross-attention) reaches 0.89 of teacher performance, vs 0.83 for mono. Object-position error is 2.5 cm (stereo) vs 3.3 cm (mono). — [DextrAH-RGB, arXiv 2412.01791](https://arxiv.org/abs/2412.01791)
- (Recalled) The DrQ / SAC-AE pixel encoder uses 4 conv layers, 3x3, 32 filters, with **stride 2 only in the first layer and stride 1 afterwards**. On 84x84 input this keeps a ~35x35 spatial map before the linear + LayerNorm projection. — [Yarats et al. SAC-AE, arXiv 1910.01741](https://arxiv.org/abs/1910.01741); [DrQ, arXiv 2004.13649](https://arxiv.org/abs/2004.13649)
- (Recalled) Finn et al. 2016 (deep spatial autoencoders) learn spatial-softmax feature points with an autoencoder objective and use them as the state for RL on real manipulation tasks. — [Finn et al., arXiv 1509.06113](https://arxiv.org/abs/1509.06113)

### Inferences
- The user's encoder looks like the DrQ encoder, but with stride 2 in all 4 layers. On 96x72 that leaves roughly a 6x5 (padded) or 5x3 (unpadded) x64 map, i.e. ~16 px per cell, before a flatten + linear. A ~3 cm cap is likely only a few pixels in the front view. Precise localisation then has to be decoded from a very coarse map by a single linear layer. Both robomimic's shallow-vs-ResNet result and Levine's spatial-softmax localisation numbers point to this as a likely precision bottleneck.
- Cheap changes that follow from the evidence, highest expected value first:
  1. Keep spatial resolution: stride 2 only in the first layer, or 1-2 stride-2 layers.
  2. Replace flatten with spatial softmax (keypoints) or add a coordinate channel.
  3. Move to ResNet-18 + GroupNorm + spatial softmax (robomimic / Diffusion Policy default). An ImageNet-pretrained version is fine if it is fine-tuned, not frozen.
- Frozen generic pretrained features (R3M, frozen CLIP, ImageNet) are unlikely to help a sim-rendered, low-resolution, small-object task. The evidence consistently favours end-to-end or fine-tuned encoders.

### Gaps
- I found no source that ablates "flatten vs spatial softmax" on the same backbone with closed-loop success numbers. Levine's numbers are pose-regression error; PRISM's abstract gives no per-pooling success numbers.
- I did not fetch VC-1 or DINOv2 manipulation numbers. (Recalled, not verified: the VC-1 paper, [arXiv 2303.18240](https://arxiv.org/abs/2303.18240), reports that no pretrained visual representation is universally dominant across embodied-AI tasks.)
- The R3M-vs-scratch comparison found is on Push-T, not on small-object grasping.

## Q2. Input resolution and wrist-camera importance

### Takeaway
I found no clean study of 84 vs 128 vs 224 resolution for small-object BC. Benchmarks use 84x84 (robomimic, sim) up to 240x240 (ToolHang) and 320x240 (real tasks). Higher resolution helps only if the encoder keeps spatial detail. A wrist camera reliably helps precise grasping and out-of-distribution generalization, often more than resolution: robomimic drops 9-43% without it, and Hsu et al. show 92 vs 64 OOD success (DAgger) and 52 vs 20 (real BC).

### Cited Findings
- Diffusion Policy input / crop sizes:
  - Lift, Can, Square: 84x84, randomly cropped to 76x76.
  - Push-T sim: 96x96 → 84x84.
  - ToolHang (the high-precision task): 240x240 → 216x216.
  - Real tasks: 320x240 → 288x216.
  - At test time they use a static centre crop of the same size. — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- ACT uses 4 cameras at 480x640 each, with ResNet-18 producing 15x20x512 feature maps per image (no pooling; features go into a transformer). It reaches fine bimanual tasks with 50 demos per task. — [ACT, arXiv 2304.13705](https://arxiv.org/abs/2304.13705)
- PRISM (2026): "repeated downsampling weakens fine-grained spatial information, especially with low-resolution observations"; ToolHang at low resolution is cited as a failure case (5.0% → 13.4% with their encoder). The paper notes that high-resolution images or additional camera views can alleviate this but add system complexity. — [arXiv 2606.15232](https://arxiv.org/abs/2606.15232)
- Two weaker results from search snippets only:
  - One visual-manipulation pretraining paper found that adapting resolution 112→224 beat fixed 112, while fixed 224 did worse (attributed to overfitting); 224 cost ~24% more compute. — [arXiv 2401.09038](https://arxiv.org/abs/2401.09038)
  - Coarse-to-fine IL found that higher resolution "was not necessary" because the camera ends very close to the object. — [arXiv 2105.06411](https://arxiv.org/abs/2105.06411)
- Removing robomimic's wrist camera drops Square by 9% and Transport by 43% relative. On real-world Can, the policy without a wrist camera gets 43.3% vs the proficient full policy. — [robomimic](https://arxiv.org/abs/2108.03298)
- Hsu et al. 2022 ("Vision-based manipulators need to also see from their hands"), OOD success hand-centric vs third-person:
  - Sim cube grasping (mean): DAgger 92.2 vs 64.4; DrQ 94.7 vs 46.6. In-distribution DAgger performance was "essentially identical" for the two views.
  - Real-robot BC sponge grasping (360 demos): 52.0 vs 20.0 OOD mean. In-distribution was 85 vs 85. With the table +5 cm higher, 65 vs 0.
  - Meta-World DrQ-v2 OOD mean: hand 69.6, third-person 55.6, both views 67.4. Both views with a variational information bottleneck (VIB) on the third-person stream: **84.4** (IQM 66.3 → 87.7, ~64% fewer OOD failures). An ℓ2 penalty instead of the VIB gave 70.3, and VIB on both streams 51.8.
  - Caveat from the paper: hand-centric only helps when the hand view contains enough task information; otherwise the third-person view is needed for learning. — [Hsu et al., arXiv 2203.12677](https://arxiv.org/abs/2203.12677)

### Inferences
- The user's wrist camera (64x48) is the main source of grasp-precision information near contact. Making it the primary stream may help more than raising front-camera resolution: give it more resolution (e.g. 96x72 or 128x96) or a less-downsampling encoder. Regularising the front stream (VIB, dropout) is a cheap, evidence-backed lever (Hsu: +17 points mean OOD).
- Front-camera resolution matters mostly for approach and localisation. At 96x72, a 3 cm cap probably spans a handful of pixels. Raising it to ~128-160 px wide, while also reducing encoder downsampling, is consistent with what precision benchmarks use (ToolHang 240 px).

### Gaps
- I found no direct 84 vs 128 vs 224 closed-loop success ablation for small-object grasping in BC. robomimic's main paper does not report a resolution sweep in the parts read.
- I did not check whether Hsu et al. applied random shift per view; the appendix was not read.

## Q3. Augmentation: random shift / crop (DrQ, RAD), fixed vs wrist cameras, colour / visual domain randomisation

### Takeaway
Random shift / crop is the single most important augmentation for pixel BC and RL. Removing it costs 26-47% in robomimic, and DrQ training stopped converging without it in Hsu et al. Typical magnitude is ~±4 px on 84 px, or a ~90% crop (DP: 76/84, 84/96, 216/240, 288/320). Colour jitter and appearance DR are used on top in sim-to-real students (DextrAH-RGB), but I found no source isolating their effect on grasp precision.

### Cited Findings
- robomimic: removing pixel-shift randomisation gives relative drops of 47% (Square) and 35% (Transport). On real Can, the policy without it reaches 26.7% vs the proficient full policy. — [robomimic](https://arxiv.org/abs/2108.03298)
- Diffusion Policy applies a random crop in training (crop ratio ~0.9, see Q2) and a fixed centre crop at inference. — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- Hsu et al.: for DrQ, removing random-shift augmentation stopped convergence, even with more interaction, for **both** hand-centric and third-person views. — [Hsu et al.](https://arxiv.org/abs/2203.12677)
- (Recalled) DrQ: pad the 84x84 frame by 4 px (replicate) and take a random 84x84 crop, i.e. ±4 px shift. RAD: across DMControl, random crop was the most effective single augmentation among those tested. — [DrQ, arXiv 2004.13649](https://arxiv.org/abs/2004.13649); [RAD, arXiv 2004.14990](https://arxiv.org/abs/2004.14990)
- DextrAH-RGB student augmentation and DR:
  - Random background p=0.5, colour jitter p=1.0, random blur p=0.1, HDRI dome lights at 30%.
  - Per-episode randomisation of albedo, roughness, metallic and specular for robot, table and object, plus texture randomisation.
  - Result: 0.89 of teacher performance with stereo RGB. — [DextrAH-RGB](https://arxiv.org/abs/2412.01791)
- James et al. 2017 trained a sim-to-real visuomotor controller for cube pick-and-place-into-basket from domain-randomised sim images only. Results: 100% real success with auxiliaries; ~200k images sufficed in sim, but ~4x more were needed for similar real-world success. — [James, Davison, Johns, CoRL 2017, arXiv 1707.02267](https://arxiv.org/abs/1707.02267)

### Inferences
- The user's 4 px shift is a larger fraction of the wrist image (4/64 = 6.3%) than of the front image (4/96 = 4.2%), but it is still within the ~±5-10% used by DrQ and DP, so the shift magnitude itself is unlikely to be the problem.
- Shift augmentation is applied on top of per-episode camera-pose jitter. Both make absolute pixel position a noisier cue for the front (fixed) camera. With a coarse flatten encoder this may worsen the localisation of a few-pixel object. One test: train with front-camera shift reduced (e.g. 2 px) or camera-pose jitter narrowed, and compare closed-loop success. This is a hypothesis, not a sourced finding.
- Heavy appearance DR enlarges the visual distribution the student must cover. With only 180k frames, that competes with learning precise localisation, consistent with the 0.053 vs 0.09 train / val gap.

### Gaps
- I found no source that separately ablates shift augmentation for fixed vs wrist cameras in BC.
- I found no quantitative ablation of colour / lighting DR on grasp precision for a distilled student.

## Q4. Auxiliary losses in teacher-student vision distillation (predicting object pose / privileged state)

### Takeaway
Predicting privileged state (object / gripper position) from images is standard in vision students: James 2017, DextrAH-RGB, the end-to-end RL dexterous student, ADEPT, and Levine's pose pretraining. Its one clean real-world ablation found shows 84% → 100% full-task success (James 2017). Its cost is tiny when the sim provides the labels.

### Cited Findings
- James et al. 2017: a CNN (8 conv layers, stride 2) plus joint angles feeds an LSTM that outputs joint velocities and a gripper action. Auxiliary outputs predict 3D cube position and 3D gripper position, and all loss terms are equally weighted. Real world, 32 trials, 1M sim images (~4000 episodes):
  - With auxiliaries: reached cube 100%, grasped 100%, full task 100%.
  - Without auxiliaries: reached cube 100%, **grasped 84%, full task 84%**.
  - The loss without auxiliaries is concentrated at the grasp stage. — [James et al. 2017](https://arxiv.org/abs/1707.02267)
- DextrAH-RGB adds an MLP [512,256] head predicting 3D object position (L_aux = ‖x̂_obj − x_obj‖), trained jointly with the action loss under online DAgger. Table I has no "aux removed" row, so its isolated effect is not reported. — [DextrAH-RGB](https://arxiv.org/abs/2412.01791)
- End-to-end RL dexterous grasping (2025): the distilled student also has an MLP head predicting object position. — [arXiv 2509.16434](https://arxiv.org/abs/2509.16434)
- ADEPT (2026; search snippet only): adds an auxiliary keypoint-pose loss to the BC loss, adopted from DextrAH-RGB. The motivation is that action cloning alone does not reliably teach the student to perceive the peg's orientation. — [ADEPT, arXiv 2608.19182](https://arxiv.org/abs/2608.19182)
- Levine et al. 2016 pretrained the conv layers by regressing target-object pose before policy training (Q1). — [Levine et al.](https://arxiv.org/abs/1504.00702)

### Inferences
- For the cap-into-mug task the natural auxiliary targets come for free from the teacher's privileged state: cap position (in the camera or robot frame), mug position, gripper / TCP position, and possibly cap-relative-to-gripper offset. A position-error readout (in cm) is also a much more interpretable diagnostic than action MSE. If the aux head cannot localise the cap to ~1 cm, the grasp will not be precise.
- The James 2017 failure split (approach always fine, grasp is where the no-aux model fails) mirrors the user's failures. This supports trying an aux head early.

### Gaps
- Apart from James 2017, I found no published ablation that isolates the aux-loss effect on success in a distillation (state teacher → RGB student) setting. The DextrAH-RGB and ADEPT ablations of the aux term are not reported in the parts read.
- I did not fetch Learning by Cheating or Visual Dexterity numbers.

## Q5. Proprioception shortcut: students over-relying on joint positions and ignoring vision

### Takeaway
There is solid evidence that proprioception can dominate optimisation and suppress visual learning, especially in the motion-transition / target-localisation phases. That is exactly the approach-and-grasp phase where the user's student fails. Documented remedies:
- phase-guided gradient down-weighting of the proprio branch (GAP, ICLR 2026);
- heavy proprio dropout (informal evidence only);
- information-bottlenecking the less informative stream (Hsu et al.).

### Cited Findings
- Lu et al. (ICLR 2026), "When would Vision-Proprioception Policies Fail in Robotic Manipulation?":
  - Diagnosis: in sub-phases where motion transitions and target localisation is needed, vision plays a limited role. The policy "naturally gravitates toward concise proprioceptive signals" because they reduce training loss faster, which suppresses vision learning.
  - Fix (GAP): scale down the proprioception gradient by the estimated probability that a timestep is in a motion-transition phase. Reported to work in sim and real, single and dual arm, and in VLAs.
  - A search summary said their concatenation baseline was ~15% worse than vision-only; the abstract does not confirm the number. — [arXiv 2602.12032](https://arxiv.org/abs/2602.12032)
- robomimic: adding extra proprioception (end-effector velocity, joint info) hurt low-dim agents by 49-88% relative and image agents by 2-29%. — [robomimic](https://arxiv.org/abs/2108.03298)
- A 2026 VLA study notes that state can serve as an action-correlation shortcut in BC, citing De Haan et al. 2019 (causal confusion) and Wen et al. 2020 (copycat agents). — [arXiv 2608.03052](https://arxiv.org/abs/2608.03052); [De Haan et al., arXiv 1905.11979](https://arxiv.org/abs/1905.11979); [Wen et al., arXiv 2010.14876](https://arxiv.org/abs/2010.14876)
- An informal blog experiment (not peer reviewed) reports:
  - Masking proprio with p=0.8 removed the shortcut: the dropout-trained model barely noticed when state was masked, while the baseline collapsed.
  - But it hurt overall performance: "dropout kills the shortcut without teaching the replacement."
  - Keeping state and rebalancing it with visual cues did better. — [blog: state shortcut](https://mcobzarenco-fontaine-blog.static.hf.space/papers/state-shortcut.html)
- Hsu et al.: regularising one input stream with a VIB raised OOD success from 67.4 to 84.4 mean, while an ℓ2 penalty gave only 70.3. — [Hsu et al.](https://arxiv.org/abs/2203.12677)

### Inferences
- The user's action is a delta on an integrated joint target, and the current joints are an input. In teacher data, the next delta is highly predictable from the recent joint trajectory (smooth teacher motion), much as copycat agents copy from history. The network can lower MSE substantially from joints alone, and so under-learn where the cap is. A cheap diagnostic: train a joints-only student. If its val MSE is close to 0.09, proprio is carrying most of the loss reduction.
- Remedies ranked by cost:
  1. Proprio dropout (p≈0.3-0.8) or noise injection on joints.
  2. A smaller or bottlenecked proprio embedding.
  3. Aux position losses (Q4), which force vision to carry localisation.
  4. GAP-style gradient scaling.

### Gaps
- I did not retrieve GAP's quantitative gains or a peer-reviewed controlled comparison of proprio-dropout rates for manipulation BC.

## Q6. Action representation and policy head: per-step delta vs absolute targets vs chunking, temporal ensembling, diffusion / flow vs MSE; offline val MSE vs closed-loop success

### Takeaway
Absolute / position targets beat deltas or velocity in both ACT and Diffusion Policy; velocity is more latency-sensitive "due to compounding error". Action chunking is the largest single effect reported: ACT goes from 1% at k=1 to 44% at k=100. Diffusion heads beat LSTM-GMM / MSE-style heads by large margins on hard tasks. Offline validation MSE is a weak predictor of rollout success (SIMPLER r=0.308; robomimic selection by val loss is 10-100% worse than the best checkpoint).

### Cited Findings
- ACT:
  - Chunking: "performance improves drastically from 1% at k=1 to 44% at k=100", averaged over 4 settings; it dips at k=200-400 (near open-loop).
  - Temporal ensembling: +3.3% for ACT and +4% for BC-ConvMLP.
  - Action space: the authors "noted degraded performance when using delta joint positions" instead of target joint positions (no numbers).
  - Loss: L1 because "L1 loss leads to more precise modeling of the action sequence."
  - CVAE matters on human data (35.3% → 2% without it) but is negligible on scripted data.
  - BC's core problem is compounding error: errors "accumulate and cause the robot to drift off of its training distribution." — [ACT](https://arxiv.org/abs/2304.13705)
- Diffusion Policy:
  - Action space: "a position-control action space consistently outperforms ... velocity control." Velocity control "is more affected by latency than position control, likely due to compounding error effects."
  - Horizon: an action horizon of 8 steps is "optimal for most tasks", and longer horizons react too slowly. Observation horizon 2 is a good compromise for vision policies.
  - Heads (max / avg success): Square-PH LSTM-GMM 0.82/0.59 vs DP-C 0.98/0.92; Transport-PH 0.88/0.62 vs 1.00/0.93; ToolHang-PH 0.68/0.49 vs 0.95/0.73; IBC near 0. Real Push-T: 95% vs 0-20% for IBC / LSTM-GMM.
  - Average improvement is 46.9%. — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- robomimic: a GMM head plus ResNet was uniformly better; dropping the GMM cost up to 58% on multi-human data. Longer RNN sequences (10/30/50) did not help significantly. — [robomimic](https://arxiv.org/abs/2108.03298)
- robomimic: choosing the checkpoint by validation loss or taking the last one gives a policy "significantly worse than the best one (10% to 100% decrease)"; offline policy selection "is non-trivial." — [robomimic](https://arxiv.org/abs/2108.03298)
- SIMPLER (Li et al. 2024), Table I, Google Robot: validation MSE gives Pearson r = 0.308, MMRV 0.375, vs SIMPLER-VisMatch r = 0.924, MMRV 0.056. The authors conclude "validation MSE is *not* a good proxy for a policy's real-world performance." — [SIMPLER, arXiv 2405.05941](https://arxiv.org/abs/2405.05941)
- AutoEval (search snippet): a weak correlation between AutoEval success and validation MSE on 400 BridgeData trajectories, "consistent with the findings in SIMPLER." — [AutoEval, arXiv 2503.24278](https://arxiv.org/abs/2503.24278)
- DextrAH-RGB: a KL loss between student and teacher action distributions beat L2 across all 4 seeds tested. — [DextrAH-RGB](https://arxiv.org/abs/2412.01791)

### Inferences
- A per-frame delta on an integrated target is the "velocity-like" representation that ACT and DP found worse. Any bias in predicted deltas integrates into a drifting target with no absolute anchor. Near a 3 cm cap, a few mm of drift becomes "cap pushed away" or "mug knocked over".
- Two options:
  1. Predict an absolute joint target (or a delta relative to current measured joints, not to the integrated target).
  2. Predict a short chunk (e.g. 8-16 steps) with receding-horizon execution or temporal ensembling.
- Teacher data from a single RL policy is close to unimodal and deterministic (like ACT's "scripted" setting, where the CVAE did not matter), so MSE / L1 is probably acceptable. A diffusion or flow head is a second-order change compared with encoder, chunking and absolute targets. L1 instead of MSE is a trivial change ACT recommends for precision.
- Do not tune on val MSE (0.09). Select checkpoints and hyperparameters by closed-loop success on a fixed set of ~50-100 seeded sim episodes.

### Gaps
- Exact velocity-vs-position numbers in Diffusion Policy are only in its Fig. 4 (not text-extractable here). ACT gives no numbers for delta vs absolute.
- I found no study on a joint-space "delta on integrated target" representation specifically.

## Q7. Data scale and DAgger in state-to-vision distillation

### Takeaway
Purely offline BC on 180k teacher frames is in the range James 2017 found sufficient in sim (~200k images), but well below online-DAgger distillation pipelines, which run thousands of parallel envs for tens of GPU-hours. DAgger reliably adds several points over non-DAgger training. Distilling from a state teacher has an inherent information-asymmetry ceiling that vision teachers reduce.

### Cited Findings
- James 2017: 1M images (~4000 episodes) for the real-world result. 200k images worked well in sim, but ~4x more data was needed for similar real-world success. They did not use DAgger; they perturbed initial joint angles to reduce compounding errors. — [James et al. 2017](https://arxiv.org/abs/1707.02267)
- DextrAH-RGB: online DAgger (teacher and student queried at every step), distillation for 53 h on one 4x L40S node (the teacher took 62 h on 8x H100). Student is 0.89 of teacher; frames and env counts are not reported. Real-world bin-packing: RGB student 77% vs depth-based DextrAH-G 87%. — [DextrAH-RGB](https://arxiv.org/abs/2412.01791)
- End-to-end RL dexterous grasping (2025):
  - Vision-in-the-loop training at 160x120 runs 4096-8400 concurrent envs on 4x L40S; at 320x240, 1024-2100.
  - Students "perform better when the teacher policy is also vision-based", because "there is less of an information asymmetry."
  - Real bin-packing: state-teacher RGB student 77% vs depth-teacher student 87-93%. — [arXiv 2509.16434](https://arxiv.org/abs/2509.16434)
- GA-DDPG (6D grasping with point clouds, search snippet): full method vs "No DAgger" was 91.3 vs 83.9 (ShapeNet) and 88.2 vs 85.3 (YCB). — [GA-DDPG, arXiv 2010.00824](https://arxiv.org/abs/2010.00824)
- Hsu et al. used DAgger as one of the main students: 92.2% OOD with hand-centric view in sim cube grasping. — [Hsu et al.](https://arxiv.org/abs/2203.12677)
- robomimic: easy tasks (Lift, Can) reach 75-100% with 20% of the data; harder tasks (Square, Transport) gain substantially from more data. — [robomimic](https://arxiv.org/abs/2108.03298)
- VIRAL (humanoid loco-manipulation, search snippet): trains students with an α-mixture of online DAgger and BC. Teacher rollouts "imprint strong priors" while student rollouts expose out-of-distribution states. — [VIRAL, arXiv 2511.15200](https://arxiv.org/abs/2511.15200)

### Inferences
- A 27% vs 88% gap with failures concentrated at approach and grasp is the classic covariate-shift signature: small localisation errors take the student to states the teacher never visited. Teacher-only data has no recovery examples there. DAgger relabelling of student rollouts by the state teacher (cheap in sim) targets this directly. Even a few DAgger rounds adding ~20-50% more frames from student-visited states is a standard, well-supported step.
- The train / val gap (0.053 vs 0.09) suggests the data is not yet saturating the model. Under heavy DR, more episodes (several x 2000) are plausibly useful. Better inductive bias (Q1) and aux losses (Q4) may give more per frame than raw data volume.
- Part of the gap may be irreducible if the teacher uses state not inferable from 96x72 / 64x48 images (e.g. exact cap pose under occlusion). Aux-head localisation error quantifies how large this ceiling is.

### Gaps
- I found no published frame counts for DextrAH-RGB, and no clean BC-only vs DAgger ablation for a state-teacher → RGB-student grasping task. GA-DDPG is point-cloud RL and its "No DAgger" is not pure BC.
- Learning by Cheating, Visual Dexterity, and UniDexGrasp++ data scales were not fetched.
