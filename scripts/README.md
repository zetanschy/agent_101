# scripts/

Nothing here is meant to be run directly. `./robot <command>` assembles the flags and
picks the right execution mode; these are the bodies it dispatches to. Four folders,
split by *what runs them*, because that is what decides how a script is written:

| | runs | root is |
|---|---|---|
| `robot/` | inside the lerobot container, driving real hardware | irrelevant — env vars carry the config |
| `sim/` | natively, in the Isaac Sim conda env (GPU + shader cache) | `parents[2]` |
| `openpi/` | inside the openpi (JAX) container, or natively on a cloud box | `$(dirname "$0")/../..` |
| `setup/` | on the bare host, before anything else works | `$(dirname "$0")/../..` |

## robot/ — the real arm

`common.sh` holds the shared flag groups and lives here because these six are its only
consumers; each wrapper `cd`s to this directory and sources it.

    common.sh              robot_args / teleop_args / camera JSON / run()
    calibrate.sh           ./robot calibrate {follower|leader}
    teleop.sh              ./robot teleop
    record.sh              ./robot record
    infer.sh               ./robot infer
    viz.sh                 ./robot viz — rerun timeline for one episode
    data.sh                ./robot data {list|viz|upload|delete|repair}
    train.sh               ./robot train — lerobot training, wandb, HF push
    merge_datasets.py      ./robot data merge
    repair_timestamps.py   ./robot data repair

## sim/ — Isaac Sim

These do NOT go through Docker: Isaac Sim is a native conda install that needs the GPU
and a large cache in `$HOME`. `sim.sh` selects the environment and runs the rest.

    sim.sh                 the launcher; sources .env then .env.local
    convert_assets.py      ./robot sim-assets — printed CAD (STL) -> USD
    play.py                ./robot sim-play — build the push-T scene and check it
    teleop.py              ./robot sim-teleop — leader arm drives the sim
    record.py              ./robot sim-record — ...and write a LeRobot dataset
    train.py               ./robot sim-train — rsl_rl PPO
    policy.py              ./robot sim-policy — run a checkpoint, score its tracking
    demo_reach.py          ./robot sim-demo — one arm, a goal you drag, policy chasing
    camera_check.py        ./robot sim-camera-check — sim camera model vs the real ones
    calibrate_cameras.py   ./robot sim-calibrate — intrinsics, from a checkerboard
    calibrate_extrinsics.py  ./robot calib-* — where the cameras are, and the leader
                           joint publisher the sim commands consume
    compare_cameras.py     ./robot sim-compare-cameras — render next to live capture
    print_targets.py       ./robot print-targets — checkerboard, charuco, push-T goal

## openpi/ — the JAX stack

    train.sh               ./robot openpi-train
    evaluate.py            ./robot openpi-eval; webui/openpi_worker.py imports it
                           by path, so there is one definition of an observation
    rtc_parity.py          ./robot rtc-parity — openpi's real-time chunking vs
                           lerobot's reference, run in both containers
    setup_cloud.sh         a rented GPU box, without Docker

## setup/ — before anything works

    login.sh               ./robot login — HF + wandb keys into .env.local
    doctor.sh              ./robot doctor — is this box wired up
    preflight.sh           GPU, VRAM, disk, and an optional 20-step smoke train
    setup_cloud.sh         install the lerobot stack natively on a cloud box
    install_camera_udev.sh stable /dev/camera-* names (needs sudo)
