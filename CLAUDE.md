# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

SO-ARM101 robot learning: a leader + follower arm, LeRobot and openpi pi0.5 in Docker, and a
calibrated MuJoCo/Isaac twin of the bench. The current project is the cap-to-mug task ("Put the
cap into the red cup"): sim demos → real-arm DAgger/Sirius rounds → benchmarks in sim and on the
arm. [DAGGER.md](DAGGER.md) holds every command of those rounds as run; [README.md](README.md)
covers setup, training and the openpi vs lerobot comparison.

## How the repo is driven

Everything goes through `./robot <verb>` (bash). It assembles flags from `.env` and runs the
body in the right place, printing the exact command first. `./robot` with no verb lists them all.

- **Config:** `.env` holds ports, arm ids, cameras, `HF_USER`, episode timings. `.env.local`
  (gitignored) holds secrets: `HF_TOKEN`, `WANDB_API_KEY`, `VAST_AI_API_KEY`, and local
  overrides such as `OPENPI_WEBUI_PORT`. Change a port or camera in `.env`, nowhere else.
- **Three Docker stacks:**
  - `docker-compose.yml`: lerobot on the real arm (serial, cameras, X11). Verbs: teleop,
    record, infer, data.
  - `docker-compose.openpi.yml`: the openpi (JAX) image. Verbs: openpi-train, openpi-webui,
    sim-eval, real-eval, dagger, sirius-build. It mounts `~/Downloads/hf_models` as `/checkpoints`.
  - `docker-compose.train.yml`: lerobot training (`./robot train`).
- **Native mode:** `./robot` detects Docker. A box without it, such as Vast.ai, runs natively
  (`ROBOT_MODE=native|docker` forces it), set up by `scripts/openpi/setup_cloud.sh`.
- **`scripts/` is split by who runs it** (see `scripts/README.md`):
  - `robot/`: the lerobot container;
  - `openpi/`: the openpi container or a cloud box;
  - `sim/`: the Isaac conda env;
  - `setup/`: the bare host.

  Nothing there is meant to be run directly.
- **Submodules** (`thirdparty/`):
  - `le101`: the lerobot 0.6.1 fork `zetanschy/le101`. It is mounted live over `/opt/le101`, so
    an edit takes effect without a rebuild. Commit it in the fork and bump the pointer here.
  - `openpi`: fork, at `/opt/openpi`.
  - `mjlab`: the fork `zetanschy/mjlab`, branch `feat/so101-push-t`. Fetch it only when
    needed (`git submodule update --init thirdparty/mjlab`); **never
    `git clone --recursive`**, which also pulls the large Isaac submodules.
  - `sim2real_so101`.

## Architecture worth knowing before editing

- **Live sim as a robot** (`sim/real2sim/live/`): `server.py` serves a MuJoCo (or Isaac)
  follower over a unix socket. `--sim mujoco` on teleop/record/infer/openpi-webui/sim-eval
  starts it (`sim_up` in `./robot`) and points the lerobot plugin at it.
  - **Domain randomization:** `--sim-dr off|visual|physics|all`. The ranges live in `dr.py`;
    every draw is logged to `sim/outputs/real2sim/live/dr_log.jsonl`.
  - **Layouts are seeded:** `reset(layout, seed)`.
- **Stages:** stage seed S always gives the same cap/mug layout. `webui/eval_stage.py` renders
  it through a separate lockstep sim on `stage.sock`. That is how the sim benchmark, the real
  Eval page and the DAgger placement page show the same stages.
  - **Seeds 1000–1099 are the eval stages. Never collect on them.**
  - DAgger rounds used 5000+, 20000+ and 30000+; the real demos used 50000+.
- **Web UI** (`webui/app.py`, FastAPI):
  - `/`: control panel;
  - `/eval`: real-arm benchmark with operator grading;
  - `/place`: DAgger placement page.

  `openpi_worker.py` is the persistent inference process: RTC, executing 15 of every 50
  actions, about 400 ms per inference on the RTX 3060.
  - `scripts/openpi/sim_eval.py` reuses that same control loop, so sim and real scores are
    comparable.
  - Under `dagger --stages`, app.py runs with `PLACE_ONLY=1`, which serves `/place` only.
- **openpi checkpoints:** config `pi05_soarm101_lora_cap_to_cup` (in
  `thirdparty/openpi/src/openpi/training/config.py`): LoRA, batch 16, 30k steps, degrees units.
  - Norm stats are keyed by dataset repo id. `evaluate.load_policy` falls back to the single set
    a checkpoint carries, so always load through it.
  - The benchmarks load openpi checkpoints only, not lerobot ones.
- **DAgger / Sirius:**
  - `scripts/openpi/dagger.py` records takeovers with an `intervention` column.
  - `scripts/robot/sirius_build.py` merges demos and rollouts, adding a `demonstration` column.
  - `scripts/openpi/dagger_weights.py` turns the classes into sampling weights:
    - takeovers 50%;
    - the 2 s before each takeover 0;
    - demos at their natural share.
  - `train.sh --sirius` retrains from scratch.

## RL on the digital twin (mjlab)

The cap-to-mug task as an RL environment, `Mjlab-CapToMug-So101-Twin`, in the mjlab fork.

- **Twin entities:** `asset_zoo/robots/so_arm101/twin.py` builds mjlab entities from
  `sim/real2sim` without editing it. The arm is the twin's own MjSpec, taken through
  `build()`'s `spec_hook`.
- **Task:** `tasks/manipulation/config/so101/cap_to_mug_{mdp,cfgs}.py`.
- **Check:** `scripts/so101_twin/check_twin.py` shows the entities equal the twin:
  - every physical parameter is identical;
  - the twin's 6 recorded grasps are reproduced.

**Measured settings** (`twin.RL_*`):
- dt 1/900 s, so 30 substeps per 30 Hz action;
- 20 solver iterations;
- `nconmax` 96, `njmax` 1500.

1/600 s and coarser go NaN with the twin's contacts.

The current task is the **state teacher**: its actor and critic see privileged sim state,
including finger-cap contact sensors. A two-camera student is to be distilled from it
later. The trained teacher is `zetanschy/mjlab_so101_cap_to_mug_teacher` on the Hub:
94.3% of 1024 episodes end with the cap in the mug and the arm home.

### On another machine

The twin reads three inputs besides its code. Two are force-tracked under the
gitignored `sim/outputs/`: the CoACD finger pieces (plain git) and the mat look (Git LFS).
The third is the calibration dataset the rest pose comes from, on the Hub; the task
reads it as `episodes.npz`, which `real2sim core extract` builds (needs Docker).

```bash
git pull && git lfs pull                      # needs git-lfs; the look is LFS
git submodule update --init thirdparty/mjlab
(cd thirdparty/mjlab && env -u PYTHONPATH uv sync --frozen)   # mjlab's own venv
D=testingt_real2sim_20260927_152936
huggingface-cli download --repo-type dataset zetanschy/$D \
    --local-dir ~/.cache/huggingface/lerobot/soarm101/$D      # where the twin reads it
./robot real2sim core extract                 # -> sim/outputs/real2sim/$D/episodes.npz
cd thirdparty/mjlab
env -u PYTHONPATH MUJOCO_GL=egl uv run --frozen --no-sync python scripts/so101_twin/check_twin.py static
```

`twin.py` finds the repo as the parent of `thirdparty/mjlab`; set `AGENT101_ROOT` if
mjlab lives elsewhere. Every line of `check_twin.py static` must say `ok`. Do not
regenerate the finger pieces with `./robot real2sim mujoco fingers`: the decomposition
is not reproducible, and the check compares them exactly.

### Training the teacher

Hand the user the command; don't launch it (see Gotchas). In tmux, from
`thirdparty/mjlab`:

```bash
env -u PYTHONPATH MUJOCO_GL=egl uv run --frozen --no-sync train Mjlab-CapToMug-So101-Twin --env.scene.num-envs 4096
```

- **Run settings:** 3000 iterations, a checkpoint every 100, saved to
  `logs/rsl_rl/so101_cap_to_mug_teacher/<date>/`. Logging goes to W&B project `mjlab`.
- **Measured on the RTX 3060:**
  - 15.4 s per iteration at 4096 envs;
  - peak GPU memory 11.3 of 12.3 GB, so nothing else may use the GPU;
  - about 13 h for 3000 iterations.
- **Measured on the RTX 5090:** 3.0-4.4 s per iteration at 4096 envs, using about
  12 of 32 GB, so 8192 envs should fit.
- **Watch, in W&B:**
  - `Episode_Reward/reach`, `grasp`, `lift` and `carry` should rise in that order;
  - success is `Episode_Metrics/cap_in_mug` and `Episode_Metrics/cap_in_mug_and_home`
    (the cap placed and the arm back at rest, at the episode's last step);
  - `Sim/nan_env_recoveries` should stay small.
- **Go/no-go:** if `grasp` is still flat at iteration 300–500, stop the run and retune
  the reward stages rather than wait.

## Common commands

```bash
./robot teleop | record --name N --task "..." --episodes 50     # real arm (add --sim mujoco for the twin)
./robot data list | viz | upload --name N | delete --repo-id U/D --episodes 3,5 | merge --name OUT --from A,B
./robot sim-eval --policy /checkpoints/<model> --episodes 100 --name <model>   # -> outputs/sim_eval/<model>/
./robot sim-eval --compare outputs/sim_eval/A outputs/sim_eval/B              # paired sign test (also real_eval dirs)
OPENPI_WEBUI_PORT=8010 ./robot real-eval                                       # http://localhost:8010/eval
OPENPI_WEBUI_PORT=8010 ./robot dagger --stages --stage-seed S --policy /checkpoints/<model> --dataset zetanschy/rollout_X
./robot sirius-build --name SET --demos soarm101/cap_to_mug_dr_50 --rollouts U/R1,U/R2   # then data upload
./robot openpi-train pi05_soarm101_lora_cap_to_cup --exp-name=RUN --data.repo-id=U/D --overwrite
./robot openpi-sirius-train --data.repo-id=U/SET --exp-name=RUN --overwrite
./robot train --dataset U/D --name RUN [--push]       # lerobot; openpi-matched flags in README "openpi vs lerobot"
```

Download a model for evaluation into `~/Downloads/hf_models`, which the containers see as
`/checkpoints`:

```bash
cd ~/Downloads/hf_models && env -u PYTHONPATH uvx --from huggingface_hub hf download zetanschy/<model> \
  --include 'params/*' --include 'assets/*' --include 'agent101_config.json' --include '_CHECKPOINT_METADATA' --local-dir <model>
```

## Tests

There is no single runner. Each suite runs where its code runs:

```bash
bash sim/real2sim/live/run.sh test                    # live sim + domain randomization (mjlab uv env)
bash sim/real2sim/run.sh test [-k name]               # real2sim core (host python3, needs pytest)
docker compose -f docker-compose.openpi.yml run --rm -T openpi python scripts/openpi/dagger_weights_test.py
docker compose -f docker-compose.openpi.yml run --rm -T openpi python scripts/openpi/dagger_test.py
./robot dagger-save-test                              # DAgger dataset write/reopen, no arm or GPU
docker compose -f docker-compose.train.yml run --rm -T -w /opt/le101 train python -m pytest -q tests/<file>   # le101
```

The `scripts/**/*_test.py` files run themselves; they have their own `main`. The repo's
`.venv`, which VS Code activates, shadows `python3` and has no pytest. Drop it from `PATH` before
running `sim/real2sim/run.sh test`. The system Python's MuJoCo 3.6 is too new for
`sim/real2sim/live` tests; use their runner.

## Gotchas

- **Ports:** 8001 and 8002 are held by Omniverse Nucleus (`omni-thumbnails`). Set
  `OPENPI_WEBUI_PORT=8010` (or 8011) for openpi-webui, real-eval and dagger.
- **ROS `PYTHONPATH` breaks `uv`/`uvx`.** Prefix them with `env -u PYTHONPATH`.
- **The GPU is a 12 GB RTX 3060.** A loaded web UI holds about 7.9 GB and the follower's serial
  port, so stop it before `dagger`/`sim-eval`. If inference is slow, check `nvidia-smi` for
  other projects' jobs and ask before killing them.
- **Hugging Face namespaces:** datasets built here land under `soarm101/` (`HF_USER`);
  `./robot data upload` pushes them to the login namespace `zetanschy/`. Rollouts keep the name
  passed to `--dataset`.
- **`outputs/` is partly root-owned** (the containers run as root).
- **Training is not launched locally.** Hand over the command. Long runs happen on the user's
  RTX 5090 PC (push the repo and the dataset, give the commands) or on Vast.ai through the
  `vast-train` skill (search, confirm with the user, then launch).
- **Git:** commit with the configured identity, `zetanschy <janssv.200604@gmail.com>`, never
  the session email. The training PC clones `main`, so push before handing over commands.
