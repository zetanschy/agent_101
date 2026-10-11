# DAgger rounds: cap to mug

The command log of the cap-to-mug policy ("Put the cap into the red cup"):
- two starting policies trained on MuJoCo demos;
- then [Sirius](https://arxiv.org/abs/2211.08416) DAgger rounds on the real SO-ARM101.

Every command below is the one that was run, so the next round is a copy with the
numbers changed. The method itself (frame classes, weights, memory management) is in
[README.md › Sirius rounds](README.md#sirius-rounds).

## Where each step runs

| step | machine |
|---|---|
| collect, clean, build, upload, evaluate | the workstation with the arm (RTX 3060, 12 GB) |
| train | the training PC (RTX 5090), or a Vast.ai box through the `vast-train` skill |

- **Datasets built on the workstation** land under `soarm101/`, its `HF_USER`.
  `./robot data upload` pushes them under the Hugging Face login, `zetanschy/`.
- **Rollout datasets** keep the name given to `./robot dagger --dataset`
  (`zetanschy/rollout_...`).
- **Models** are downloaded to `~/Downloads/hf_models/`, which the containers mount as
  `/checkpoints/`.

## Results so far

- **Sim:** 100 seeded MuJoCo stages (seeds 1000–1099), graded by the sim.
- **Real arm:** the first 50 of those stages (seeds 1000–1049), graded by the operator on
  the Eval page.

| model | trained on | sim | real arm |
|---|---|---|---|
| `openpi_pi05_lora_cap_to_mug_sim_50` | 50 MuJoCo demos | 82/100 | 7/49 |
| `openpi_pi05_lora_cap_to_mug_dr_50` | 50 MuJoCo demos, domain-randomized | 37/100 | 10/49 |
| `openpi_pi05_lora_cap_to_mug_sirius_r1` | DR demos + round 1 | 59/100 | 29/50 |
| `openpi_pi05_lora_cap_to_mug_sirius_r2` | DR demos + rounds 1–2 | 76/100 | 39/50 |
| `openpi_pi05_lora_cap_to_mug_sirius_r3` | DR demos + rounds 1–3 | 70/100 | 45/50 |

## The rounds

| round | deploys | rollouts | stage seeds | rollout episodes (frames) | training set, episodes (frames) |
|---|---|---|---|---|---|
| 1 | `dr_50` | `zetanschy/rollout_cap_to_mug_dagger_r1` | 5000+ | 25 (18,486), after deleting episode 9 | `cap_to_mug_sirius_r1`, 75 (47,979) |
| 2 | `sirius_r1` | `zetanschy/rollout_cap_to_mug_dagger_r2` | 20000+ | 25 (15,190) | `cap_to_mug_sirius_r2`, 100 (63,169) |
| 3 | `sirius_r2` | `zetanschy/rollout_cap_to_mug_dagger_r3` | 30000+ | 25 (17,010) | `cap_to_mug_sirius_r3`, 125 (80,179) |

Each round:
- deploys the latest model;
- trains from scratch, from `pi05_base` with no `--init-from`, on the DR demos plus every
  rollout so far;
- uses config `pi05_soarm101_lora_cap_to_cup`: 30k steps at batch 16.

## Round 0: the starting policies

### Sim demos, no randomization: `cap_to_mug_sim_50`

Two recording sessions on the MuJoCo twin, driven by the real leader arm, 50 episodes kept:

```bash
./robot record --sim mujoco --name cap_to_mug_sim  --task "Put the cap into the red cup" --episodes N
./robot record --sim mujoco --name cap_to_mug_sim2 --task "Put the cap into the red cup" --episodes N
./robot data delete --name sim_mujoco_cap_to_mug_sim_20261001_015227 --episodes 7,9
./robot data merge --name cap_to_mug_sim_all \
    --from sim_mujoco_cap_to_mug_sim_20261001_015227,sim_mujoco_cap_to_mug_sim2_20261002_044936
./robot data upload --name cap_to_mug_sim_all --to cap_to_mug_sim_50
```

It was trained on a Vast.ai RTX 5090 through the `vast-train` skill (2026-10-02, W&B
`5c3eduko`). The same run on the training PC would be:

```bash
./robot openpi-train pi05_soarm101_lora_cap_to_cup --exp-name=openpi_pi05_lora_cap_to_mug_sim_50 \
    --data.repo-id=zetanschy/cap_to_mug_sim_50 --overwrite
```

### Sim demos with domain randomization: `cap_to_mug_dr_50`

`--sim-dr all` randomizes, per episode:
- **the visuals:** light, mat, cap colour, camera mounts;
- **the physics:** gains, friction, masses, dead time, sensor noise.

Each draw is logged to `sim/outputs/real2sim/live/dr_log.jsonl`. After the bad episodes
were deleted, one more session brought the set back to 50:

```bash
./robot record --sim mujoco --sim-dr all --name cap_to_mug_dr   --task "Put the cap into the red cup" --episodes 50
./robot record --sim mujoco --sim-dr all --name cap_to_mug_done --task "Put the cap into the red cup" --episodes 1
./robot data merge --name cap_to_mug_dr_50 \
    --from sim_mujoco_dr_cap_to_mug_dr_20261004_234622,sim_mujoco_dr_cap_to_mug_done_20261005_004141
./robot data upload --name cap_to_mug_dr_50
```

Trained on the training PC (2026-10-04, W&B `ckyh2h3c`):

```bash
./robot openpi-train pi05_soarm101_lora_cap_to_cup --exp-name=openpi_pi05_lora_cap_to_mug_dr_50 \
    --data.repo-id=zetanschy/cap_to_mug_dr_50 --overwrite
```

The final save was killed for lack of memory. The run was finished from the step-25000
checkpoint:
1. Delete the half-written `29999.orbax-checkpoint-tmp-*` folder.
2. Run:

```bash
./robot openpi-train pi05_soarm101_lora_cap_to_cup --exp-name=openpi_pi05_lora_cap_to_mug_dr_50 \
    --data.repo-id=zetanschy/cap_to_mug_dr_50 --resume
```

**The Sirius rounds start from `dr_50`**, not `sim_50`: the DR demos are the `demo` class
of every round.

## Rounds 1 and 2, as run

On the training PC, every round ran from the repo on `main`, in its own tmux session.
`openpi-dagger-stats` is an optional check of the sampling weights; it doesn't use the GPU.

**Round 1** (deploys `dr_50`):

```bash
# workstation
OPENPI_WEBUI_PORT=8011 ./robot dagger --stages \
    --policy /checkpoints/openpi_pi05_lora_cap_to_mug_dr_50 \
    --dataset zetanschy/rollout_cap_to_mug_dagger_r1
./robot data delete --repo-id zetanschy/rollout_cap_to_mug_dagger_r1 --episodes 9   # ended away from home
./robot sirius-build --name cap_to_mug_sirius_r1 --demos soarm101/cap_to_mug_dr_50 \
    --rollouts zetanschy/rollout_cap_to_mug_dagger_r1
./robot data upload --name cap_to_mug_sirius_r1

# training PC
huggingface-cli download --repo-type dataset zetanschy/cap_to_mug_sirius_r1 \
    --local-dir ~/.cache/huggingface/lerobot/zetanschy/cap_to_mug_sirius_r1
./robot openpi-dagger-stats --dataset zetanschy/cap_to_mug_sirius_r1 --scheme sirius
./robot openpi-sirius-train --data.repo-id=zetanschy/cap_to_mug_sirius_r1 \
    --exp-name=openpi_pi05_lora_cap_to_mug_sirius_r1 --overwrite
```

**Round 2** (deploys `sirius_r1`). Before training, the model-host server on the 5090 was
stopped to free the GPU.

```bash
# workstation
OPENPI_WEBUI_PORT=8011 ./robot dagger --stages --stage-seed 20000 \
    --policy /checkpoints/openpi_pi05_lora_cap_to_mug_sirius_r1 \
    --dataset zetanschy/rollout_cap_to_mug_dagger_r2
./robot sirius-build --name cap_to_mug_sirius_r2 --demos soarm101/cap_to_mug_dr_50 \
    --rollouts zetanschy/rollout_cap_to_mug_dagger_r1,zetanschy/rollout_cap_to_mug_dagger_r2
./robot data upload --name cap_to_mug_sirius_r2

# training PC
huggingface-cli download --repo-type dataset zetanschy/cap_to_mug_sirius_r2 \
    --local-dir ~/.cache/huggingface/lerobot/zetanschy/cap_to_mug_sirius_r2
./robot openpi-dagger-stats --dataset zetanschy/cap_to_mug_sirius_r2 --scheme sirius
./robot openpi-sirius-train --data.repo-id=zetanschy/cap_to_mug_sirius_r2 \
    --exp-name=openpi_pi05_lora_cap_to_mug_sirius_r2 --overwrite
```

## Round 3, step by step

Round 3 deploys `sirius_r2`. Steps 1–4 are done: 25 episodes, built and uploaded. For round 4,
change these:
- `r3` → `r4` in every name;
- the deployed model → `sirius_r3`;
- `--stage-seed 30000` → `40000`;
- add `zetanschy/rollout_cap_to_mug_dagger_r4` to `--rollouts`.

### 1. Collect (workstation)

Stop the web UI first: it holds the GPU (7.9 GB) and the follower's serial port.

```bash
OPENPI_WEBUI_PORT=8010 ./robot dagger --stages --stage-seed 30000 \
    --policy /checkpoints/openpi_pi05_lora_cap_to_mug_sirius_r2 \
    --dataset zetanschy/rollout_cap_to_mug_dagger_r3
```

**The placement page** is http://localhost:8010/place.
- It shows the MuJoCo stage for the next episode, with outlines on the live overhead camera.
- Episode k uses stage seed 30000 + k.
- "Another stage" jumps by 1000, so the next round's block starts 10000 higher.
- Never use seeds 1000–1099: those are the eval stages.

**The keys**, in the terminal:

| key | action |
|---|---|
| space | pause / resume the policy |
| tab | take over / stop correcting |
| → | save the episode |
| ← | discard it (the stage stays the same) |
| esc | quit |

- `--resume` appends to an existing rollout dataset.
- Home the arm between episodes with the placement page's Home button. It works only
  while paused and not recording.

### 2. Check and clean (workstation)

Render the episodes to MP4:

```bash
./robot dagger-video --dataset zetanschy/rollout_cap_to_mug_dagger_r3
```

Delete any episode that doesn't end at home with the arm still:

```bash
./robot data delete --repo-id zetanschy/rollout_cap_to_mug_dagger_r3 --episodes <i,j>
```

### 3. Build the training set (workstation)

Pass the rollouts oldest first:

```bash
./robot sirius-build --name cap_to_mug_sirius_r3 --demos soarm101/cap_to_mug_dr_50 \
    --rollouts zetanschy/rollout_cap_to_mug_dagger_r1,zetanschy/rollout_cap_to_mug_dagger_r2,zetanschy/rollout_cap_to_mug_dagger_r3
./robot openpi-dagger-stats --dataset soarm101/cap_to_mug_sirius_r3 --scheme sirius   # optional
```

- It writes `soarm101/cap_to_mug_sirius_r3`; `--force` overwrites it.
- It prints the class mix (demo / robot / intv / preintv) and the intervention-to-demo
  ratio. The paper ended a round at about a third.

### 4. Upload, and push the code (workstation)

```bash
./robot data upload --name cap_to_mug_sirius_r3      # -> zetanschy/cap_to_mug_sirius_r3
git push origin main
```

### 5. Train (training PC)

Run from the repo on `main`, in a tmux session. Stop anything else on the GPU first.

```bash
git pull
huggingface-cli download --repo-type dataset zetanschy/cap_to_mug_sirius_r3 \
    --local-dir ~/.cache/huggingface/lerobot/zetanschy/cap_to_mug_sirius_r3
./robot openpi-dagger-stats --dataset zetanschy/cap_to_mug_sirius_r3 --scheme sirius
./robot openpi-sirius-train --data.repo-id=zetanschy/cap_to_mug_sirius_r3 \
    --exp-name=openpi_pi05_lora_cap_to_mug_sirius_r3 --overwrite
```

- At 1.69 s/step on the 5090, 30k steps take about 14 h.
- `train.sh` computes the normalization stats and pushes the final checkpoint to
  `zetanschy/openpi_pi05_lora_cap_to_mug_sirius_r3`.
- On Vast.ai instead: the `vast-train` skill, with `--train-arg --sirius`.

### 6. Download the model (workstation)

```bash
cd ~/Downloads/hf_models && env -u PYTHONPATH uvx --from huggingface_hub hf download \
    zetanschy/openpi_pi05_lora_cap_to_mug_sirius_r3 \
    --include 'params/*' --include 'assets/*' --include 'agent101_config.json' --include '_CHECKPOINT_METADATA' \
    --local-dir openpi_pi05_lora_cap_to_mug_sirius_r3
```

`env -u PYTHONPATH` keeps ROS's Python path out of `uvx`.

### 7. Evaluate (workstation)

**Sim.** Results land in `outputs/sim_eval/<name>/`:

```bash
./robot sim-eval --policy /checkpoints/openpi_pi05_lora_cap_to_mug_sirius_r3 --episodes 100 \
    --name openpi_pi05_lora_cap_to_mug_sirius_r3
./robot sim-eval --compare outputs/sim_eval/openpi_pi05_lora_cap_to_mug_sirius_r3 \
    outputs/sim_eval/openpi_pi05_lora_cap_to_mug_sirius_r2
```

**Real arm:**

```bash
OPENPI_WEBUI_PORT=8010 ./robot real-eval        # then http://localhost:8010/eval
```

| Eval page field | value |
|---|---|
| Policy | `/checkpoints/openpi_pi05_lora_cap_to_mug_sirius_r3` |
| Task | Put the cap into the red cup |
| Trials | 50 |
| Seed | 1000 |
| Timeout | 45 s |
| Actions | 15 |

The results land in `outputs/real_eval/openpi_pi05_lora_cap_to_mug_sirius_r3_real_seed1000_n50/`.
Compare stage by stage with the previous round:

```bash
./robot sim-eval --compare outputs/real_eval/openpi_pi05_lora_cap_to_mug_sirius_r3_real_seed1000_n50 \
    outputs/real_eval/openpi_pi05_lora_cap_to_mug_sirius_r2_real_seed1000_n50
```

## How to correct

- **Take over as soon as it goes wrong.** Sirius drops only the 2 s before a takeover
  (`preintv`). Anything the policy did before that is kept as `robot` frames, so a late
  takeover trains on more of the mistake.
- **Hand the arm back once it's fixed:** tab to stop correcting (the arm pauses), then
  space to resume the policy. In round 2, every takeover ran to the end of the episode,
  and the policy's own frames were only 3.3% of the samples.
- **Correct the joint that's wrong, not the whole motion.** Of the 24 round-1 takeovers at
  the grasp, 11 moved mainly the elbow and 8 the wrist flex. After an overshoot, the
  round-1 model made the same one-joint correction by itself.
- **End every episode at home, with the arm still.** In round 1 the operator held the arm
  still once the cap was placed. The round-1 model then made one or two more reaches at
  most, and stayed still after that.

## Known problems

- **On the workstation, port 8001 is taken** by Omniverse Nucleus (`omni-thumbnails`).
  Hence `OPENPI_WEBUI_PORT=8010` (or 8011) on `dagger` and `real-eval`.
- **The server started by `dagger --stages` used to serve the whole web UI.** Pressing Load
  at `/` put a second policy on the 12 GB card, which ran out of memory. Now it serves only
  `/place`.
- **openpi's final save can run out of host memory** on the training PC. Resume from the
  last 5k checkpoint, as with `dr_50` above.
