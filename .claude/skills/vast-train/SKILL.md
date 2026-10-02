---
name: vast-train
description: Rent a Vast.ai GPU and run an openpi pi0.5 LoRA fine-tune on a LeRobot dataset end to end. Covers searching offers, confirming with the user, launching, monitoring, pushing the checkpoint to the Hub and tearing the box down. Use when the user asks to train or fine-tune pi05 on vast.ai, or to "find a machine and run my job" for a dataset.
---

# vast-train: a pi0.5 LoRA run on Vast.ai, from dataset to pushed checkpoint

The user gives a dataset (e.g. `zetanschy/cap_to_mug_sim_50`), and optionally a step
count and a run name. You do the rest in this order: **search → confirm → launch → report
the step time → let it finish → verify**.

**Never rent anything before the user confirms** with AskUserQuestion. Once they confirm,
do everything yourself, as the user asked on 2026-10-02.

Defaults:
- config `pi05_soarm101_lora_cap_to_cup` (batch 16)
- 30k steps
- exp name `openpi_pi05_lora_<dataset name>`
- teardown `destroy`

## 0. Preflight

- **Keys in `.env.local`:**
  - `VAST_AI_API_KEY` (cloud.vast.ai/manage-keys)
  - `HF_TOKEN`
  - `WANDB_API_KEY`
- **Push first.** `git diff --quiet origin/main -- scripts/openpi` must be clean, because
  the box clones `main`.
- **The CLI.** Use this helper (ROS's PYTHONPATH breaks `uvx`, so it is unset):
  ```bash
  V() { env -u PYTHONPATH uvx -q vastai --api-key "$(grep -m1 '^VAST_AI_API_KEY=' .env.local | cut -d= -f2-)" "$@"; }
  ```
- **Credit:** `V show user --raw` gives `.credit`. If the run's cost goes past the credit,
  the box stops before the checkpoint is pushed.

## 1. Search

Price every offer with 100 GB of disk (`--storage 100`). Rank offers by **the cost of
the whole run, not by $/h**:

`cost = steps × s/step ÷ 3600 × dph_total + 0.5 h × dph_total`

The 0.5 h covers setup, normalization stats, the compile and the ~9 GB push.

```bash
BASE='disk_space>=100 cuda_vers>=12.4 cpu_cores_effective>=12 cpu_ram>=32 inet_down>=500 reliability>0.99 verified=true rentable=true'
for q in "num_gpus=1 gpu_name=RTX_5090 cuda_vers>=12.8" "num_gpus=1 gpu_name=RTX_6000Ada" \
         "num_gpus=1 gpu_name=A100_SXM4" "num_gpus=1 gpu_name=L40S" "num_gpus=1 gpu_name=H100_SXM" \
         "num_gpus=2 gpu_name=RTX_4090" "num_gpus=2 gpu_name=RTX_3090"; do
  V search offers "$q $BASE" --storage 100 -o dph_total --raw   # take the first few of each
done
```

**What can hold the job.** openpi documents 22.5 GB, and JAX takes only 90% of a card.
- **One 32 GB or larger card works.**
- **A single 24 GB card runs out of memory.**
- **Two 24 GB cards work:** data parallel halves each card's batch.
- **Skip these:**
  - Turing cards such as the Quadro RTX 8000: no bf16.
  - CMP mining cards.
  - "48 GB" RTX 4090s: unofficial board mods.

**Step times of this job (openpi pi05 LoRA, batch 16).** A measured time beats any
benchmark. Scale anything unmeasured by DLPerf from the 6000 Ada row, as
`2.55 × 113 / dlperf`, and label it an estimate.

| GPU | s/step | source |
|---|---|---|
| 1× RTX 6000 Ada | 2.55 | Vast.ai, 2026-10-02, `openpi_pi05_lora_cap_to_mug_sim_50` |
| 2× RTX 4090 | 2.61 | W&B `openpi_pi05_lora_cap_to_cup_200`, 2026-07-30 |
| 1× RTX 5090 | 1.69 | W&B `dagger_r1`, 2026-09-19 (the user's own 5090, not rented) |

After every run, add its measured step time to this table and commit it.

The dataset size changes only the number of passes (`steps × 16 / frames`), not the cost.
Read the frame count from the dataset's `meta/info.json`.

## 2. Confirm (AskUserQuestion, one call)

1. **Machine:** the 3–4 cheapest runs. For each, give the offer id, GPU, $/h, s/step
   (marked measured or estimated), hours, run cost, and whether it fits the credit. Mark
   the cheapest run that fits as Recommended.
2. **Steps:** 30k, or fewer. If 30k doesn't fit the credit, give the step count that does.
3. **After the push:** destroy (recommended), stop, or keep.

If nothing fits the credit, say how much to add.

## 3. Launch

Offers vanish within minutes. If the confirmed one is gone, take the same GPU model at
no more than $0.05/h above it, and say so. Otherwise ask again.

```bash
V create instance <OFFER> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 100 \
  --ssh --direct --label agent101-<EXP> --raw          # -> new_contract = instance id
V show ssh-keys --raw                                  # empty? V create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
V show instance <ID> --raw                             # poll every 15 s until actual_status == running
```

The SSH target is `root@<public_ipaddr>`, on port `ports["22/tcp"][0].HostPort`.

Then, on the box:
```bash
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -p <PORT> root@<IP>"
$SSH 'cd /root && git clone -q https://github.com/zetanschy/agent_101'      # main, NOT --recursive
grep -E '^(HF_TOKEN|WANDB_API_KEY)=' .env.local | $SSH 'umask 077; cat > /root/agent_101/.env.local'
scp -P <PORT> -i ~/.ssh/id_ed25519 .claude/skills/vast-train/remote_run.sh root@<IP>:/root/
$SSH "tmux new-session -d -s train 'bash /root/remote_run.sh --exp-name <EXP> --dataset <DS> [--steps N] --teardown destroy'"
```

- **Tokens:** send them only over SSH stdin. Never pass them through `--env`, where Vast.ai
  would store them.
- **`--steps N` below 30k** also shortens the learning-rate decay to N, so the run anneals
  over its own length.
- **What `remote_run.sh` does:**
  1. `setup_cloud.sh`, which builds a Python 3.12 venv: the image has 3.11, and le101 needs 3.12.
  2. `train.sh`: normalization stats on this dataset, the training, the push to `<hf user>/<EXP>`.
  3. Checks that `params/` is on the Hub.
  4. **Destroys the box itself**, using the instance key in PID 1's environment.
  5. On anything short of a verified push, it **stops** the box instead, keeping the disk
     and its checkpoint.

The user can watch with `ssh -p <PORT> root@<IP> -t tmux attach -t train` (detach with
Ctrl-b, then d).

## 4. Report the step time, then let it run

Run a background Bash until-loop that exits on any of these in `/root/run.log`:
- openpi's `Progress on: <N>it/30.0kit rate:<r>s/it` with N ≥ 100;
- `EXIT <rc>`;
- `Traceback`.

It takes about 15 min to get there. Then report:
- the step time;
- the hours left;
- the projected extra cost at `dph_total`;
- whether that fits the credit.

If it doesn't fit, stop and ask the user: add credit, or restart with fewer steps.

After that, the box finishes and tears itself down on its own. If you keep watching,
re-arm a background loop for no more than 2 h at a time. A run already in progress
without `remote_run.sh` can be given the self-teardown with
`bash /root/remote_run.sh --finish-only --exp-name <EXP>`, run in its own tmux session.

## 5. Verify and close

- The Hub repo `<hf user>/<EXP>` holds `params/`.
- The instance is gone from `V show instances --raw`.
- If it was stopped instead, read the tail of `/root/run.log`. Restart it with
  `V start instance <ID>`, then run `train.sh ... --resume`.
- Tell the user the model URL, the total hours and the cost (from `V show invoices`).
- Add the measured step time to the table above.

## Learned the hard way (2026-10-02)

- **A filter of `gpu_ram>=45` hid every 32 GB card.** The 5090 is the cheapest run.
- **`compute_norm_stats.py` takes only `--config-name`.** `train.sh` now swaps the dataset in
  for it, so `--data.repo-id` works.
- **`$?` after a `$(date)` in the same `echo` is the date's exit code.** Save `rc=$?` first.
- **The host bills downloads and uploads.** It was $0.17 for setup alone.
