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
`scripts/train.sh` read that one file, so credentials survive `--rm` runs — unlike
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

For a rented GPU box, [scripts/setup-cloud.sh](scripts/setup-cloud.sh) installs
the same stack without Docker; then call `bash scripts/train.sh` with identical flags.

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
