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
  - `mjlab`: pinned to a commit that is not on its public remote, so **never
    `git clone --recursive`**.
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
