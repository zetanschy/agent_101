# agent_101 — SO-ARM101 robot learning

LeRobot in Docker for the SO-ARM101 (leader + follower). All the flags you'd
otherwise repeat — ports, ids, camera configs — live once in [.env](.env);
thin wrappers in [scripts/](scripts/) assemble the full commands.

## Hardware (this machine)

| Part            | Value                          |
|-----------------|--------------------------------|
| Leader port     | `/dev/ttyACM0` (`zetans_leader`)   |
| Follower port   | `/dev/ttyACM1` (`zetans_follower`) |
| Cameras         | front=`video0`, grip=`video2`, side=`video4` |
| GPU             | RTX 3060 (CUDA)                |

If a port or camera index changes, edit [.env](.env) — nothing else.

## First run

```bash
./robot build                 # build the image (installs lerobot[feetech])
./robot login                 # Hugging Face + Weights & Biases tokens, once
```

`./robot login` prompts for both tokens and stores them in `.env.local`
(gitignored, `chmod 600`), then verifies each against the API. Every container and
`scripts/robot/train.sh` read that one file, so credentials survive `--rm` runs — unlike
a `wandb login` inside a container, which dies with it. `./robot login --status`
shows what's configured without printing secrets. To move to a cloud GPU box,
copy that single file: `scp .env.local user@box:agent_101/`.

Arm calibration already ships in [calibration/](calibration/) (`zetans_follower`,
`zetans_leader`) and is mounted into the container, so you can teleop straight
away. Only re-run `./robot calibrate follower` / `leader` if you re-cable or
swap servos.

## Everyday use

```bash
./robot teleop                # teleop, 2 cameras (front + grip)
./robot teleop --cams 3       # add the side camera
./robot teleop --no-display   # skip the rerun GUI

./robot record --name cap_pick_and_place \
  --task "Pick the cap and place it into the red mug" \
  --episodes 50               # add --cams 3 / --push as needed

./robot shell                 # container shell
./robot run lerobot-train ... # any raw lerobot command
```

Every wrapper prints the exact command it runs before executing, so you can
copy or tweak it.

## Correcting a policy by hand (DAgger)

```bash
./robot dagger --dataset zetanschy/rollout_cap_to_cup_dagger
#   space  pause / resume the policy     tab    take over / hand back
#   right  task complete: save it          left   the attempt failed: discard it
#   esc    end the session
```

The openpi checkpoint drives; when failure looks imminent you pause, take the **leader**
arm, recover the arm to a state the policy knows, correct, and hand it back — as often
as you like within one episode. The trajectory stays continuous and the episode ends
when the *task* is done, which is the protocol in
[le101's HIL guide](thirdparty/le101/docs/source/hil_data_collection.mdx). Frames are
written as a LeRobotDataset with an `intervention` column, the same one
`lerobot-rollout --strategy.type=dagger` writes, so it trains like any other dataset.

Both segments are recorded by default, as that guide describes; `--corrections-only`
records just your windows, one episode each, which is what lerobot's *code* defaults to.
The two readings differ and `intervention` is what lets a training run take either.

If **your own correction** goes wrong, keep correcting: recovering the arm from a mess
you made is the same data as recovering it from one the policy made. Only drop the
episode when the attempt itself is spoiled. There is deliberately no "undo the last
correction" — the episode would jump from the frame before it to wherever the arm now
is, and a trajectory with a teleport in it is worse than no trajectory. Afterwards,
`lerobot-edit-dataset --operation.type delete_episodes` removes whole episodes, which is
the same granularity for the same reason.

Episode control is lerobot's, keys included: **→** ends the attempt and keeps it, **←**
throws it away (`clear_episode_buffer()`, the same call `lerobot-record` makes for a
re-record), **esc** ends the session. A failed attempt is worse than no attempt — it
teaches a trajectory that did not work — so discarding is one keypress, and
`./robot dagger-save-test` proves it leaves nothing behind.

le101 ships that strategy already, and if your checkpoint is in **lerobot** format you
should use it directly (`lerobot-rollout --strategy.type=dagger`, with a leader teleop).
This wrapper exists because the checkpoint that works on this arm is an openpi **orbax**
directory, which lerobot's policy factory cannot load — see
[scripts/openpi/dagger.py](scripts/openpi/dagger.py).

## A joint slipped (a crash during teleop)

A collision does not decalibrate an encoder — a magnetic absolute encoder does not
forget. It moves the **metal**: the horn slips on the spline and the link sits a few
degrees from where the servo thinks it is. Check that first, it takes a second:

```bash
git status calibration/     # empty = the stored calibration is untouched, as expected
./robot joint-check         # read both arms, see which joint disagrees and by how much
```

Every checkpoint here was trained on **absolute joint targets**, so what a policy relies
on is exactly one mapping:

```
physical pose  ->  reported degrees
```

Restoring *that* is the job. Two fixes do it and keep every trained model:

1. **Reseat the horn** (preferred). Power the joint so it holds, loosen the horn screw,
   rotate the link back to where it belongs for that reading, retighten. The arm ends up
   matching both the old frame *and* the URDF, which the sim2real work and the computed
   safety clamp both assume.
2. **Shift that one joint's `homing_offset`** by the measured delta:

   ```bash
   ./robot joint-offset --joint wrist_flex --degrees 12.5          # dry run
   ./robot joint-offset --joint wrist_flex --degrees 12.5 --apply
   ```

   Same mapping restored, in software, and the safety clamp is **not** affected:
   `range_min`/`range_max` are recorded from `Present_Position`, i.e. *after* the
   offset, so they shift with it and the span `evals/rig.py` derives the clamp from
   never changes. What this does not fix is the arm's geometry, which still disagrees
   with the URDF by the slip — so sim2real and solved camera extrinsics stay off until
   the horn is reseated.

And one fix that looks right and is not:

> **Do not run `./robot calibrate`.** A full recalibration redefines every joint's frame
> from scratch, so the same physical pose reports different degrees than it did when
> `cap_to_cup_200` was recorded — and every checkpoint trained on it is then aiming at a
> frame that no longer exists. Nothing warns you; the arm simply starts missing.

Measuring the delta: put both arms in the *same physical pose* (folded against their
mechanical stops is the easiest reference) and read `./robot joint-check`, or put the
follower physically at its home pose and see which joint does not read ~0 against
`config/home_pose.json`.

## Training

```bash
./robot data upload --name cap_pick_and_place       # dataset to the Hub first
./robot train --dataset soarm101/cap_pick_and_place --name pi05_cap
./robot train --dataset soarm101/cap_pick_and_place --name pi05_cap --push
```

LoRA fine-tune of `pi05_base` (bf16 + gradient checkpointing, fits ~24GB) via
[docker-compose.train.yml](docker-compose.train.yml) — GPU only, no arms needed.
Checkpoints land in `outputs/train/<name>/` every 5000 steps; `--push` also
uploads to `hf.co/<you>/<name>`. Resume a crashed run with the same `--name` plus
`--resume true`. Tune with `--steps` / `--batch` / `--rank`.

**Weights & Biases** is on automatically whenever a key is stored (`./robot
login`), logging to project `agent_101` under a run named after `--name` —
resuming re-attaches to the same run. `--no-wandb` opts out per run,
`--wandb-offline` logs to disk for a later `wandb sync`, and `--wandb-project` /
`--wandb-entity` override the destination. Set the default project in
[.env](.env).

For a rented GPU box, [scripts/setup/setup_cloud.sh](scripts/setup/setup_cloud.sh) installs
the same stack without Docker; then call `bash scripts/robot/train.sh` with identical flags.

## Notes

- **Prerequisite:** the host needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  for GPU access (`nvidia-ctk`). Serial/camera/X11 passthrough is already in
  [docker-compose.yml](docker-compose.yml).
- **Display:** `--display_data=true` opens a rerun window via X11. `./robot`
  runs `xhost +local:root` for you. On Wayland or SSH, use `--no-display`.
- **New camera / extra port:** add the `/dev/videoN` line to
  [docker-compose.yml](docker-compose.yml) and the index to [.env](.env).
- Datasets and models persist in `~/.cache/huggingface` (mounted into the
  container), so they survive `--rm` runs.
