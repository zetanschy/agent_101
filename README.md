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
./robot login                 # optional: cache your HF token in the volume
```

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
