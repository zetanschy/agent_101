#!/usr/bin/env bash
# Pre-flight for a rented GPU box: check everything that can waste an hour of
# rent — GPU kernels, VRAM, RAM, disk, vCPU, python version, AV1 decoder —
# before a 13GB download and a 30k-step run.
#
#   bash scripts/preflight.sh                       # checks only, ~10s
#   bash scripts/preflight.sh --smoke               # + 20 real training steps
#   bash scripts/preflight.sh --smoke --batch 32 --dataset <id> --base <model>
#
# The kernel check matters most on new silicon (RTX 5090 = Blackwell = sm_120):
# a torch wheel built without your card's compute capability imports fine and
# reports cuda.is_available() == True, then dies with "no kernel image is
# available for execution on the device" once the model reaches the GPU.
#
# --smoke is the real proof: it runs the actual policy, LoRA, dataset and video
# decode for a few steps, so you know your --batch fits before committing.
set -euo pipefail
cd "$(dirname "$0")/.."

smoke=0; batch=16; dataset="zetanschy/raw_cap_to_cup_199"; base="lerobot/pi05_base"; workers=4
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)   smoke=1; shift ;;
    --batch)   batch="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --base)    base="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

python - <<'PY'
import os
import shutil
import subprocess
import sys
from pathlib import Path

GB = 2**30
fails: list[str] = []
warns: list[str] = []


def line(status: str, label: str, detail: str) -> None:
    print(f"  [{status:4s}] {label:22s} {detail}")


def check(cond_ok: bool, cond_warn: bool, label: str, detail: str, fix: str = "") -> None:
    if cond_ok:
        line("ok", label, detail)
    elif cond_warn:
        line("WARN", label, detail)
        warns.append(f"{label}: {detail}" + (f" — {fix}" if fix else ""))
    else:
        line("FAIL", label, detail)
        fails.append(f"{label}: {detail}" + (f" — {fix}" if fix else ""))


print("== box ==")

# Python: the le101 fork requires 3.12+, and many rented images still ship 3.10/3.11.
v = sys.version_info
check(
    v >= (3, 12), False,
    "python", f"{v.major}.{v.minor}.{v.micro}",
    "uv venv --python 3.12 && source .venv/bin/activate, then re-run setup-cloud.sh",
)

# vCPU: these datasets are AV1, decoded on CPU (torchcodec here is a CPU build),
# so cores are what feed the GPU. Too few and the card idles between steps.
cpus = os.cpu_count() or 0
check(cpus >= 8, cpus >= 4, "vCPU", f"{cpus} cores", "AV1 decode is CPU-bound; the GPU will starve")

# RAM: model load plus one video decoder per dataloader worker.
try:
    info = {
        k.strip(): int(val.split()[0]) * 1024
        for k, _, val in (ln.partition(":") for ln in Path("/proc/meminfo").read_text().splitlines())
    }
    total, avail = info["MemTotal"] / GB, info["MemAvailable"] / GB
    check(
        avail >= 12, avail >= 6,
        "RAM", f"{avail:.0f} GB available of {total:.0f} GB",
        "lower --num_workers if the run gets OOM-killed",
    )
except Exception as e:
    line("WARN", "RAM", f"could not read /proc/meminfo: {e}")

# Disk: base model ~13GB + dataset ~1GB + checkpoints. The HF cache and the repo
# can sit on different mounts on rented boxes, so check both, and require the sum
# only when they share a filesystem.
hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
probe = hf_home if hf_home.exists() else hf_home.parent if hf_home.parent.exists() else Path.home()
repo = Path.cwd()
same_fs = os.stat(probe).st_dev == os.stat(repo).st_dev
hf_free = shutil.disk_usage(probe).free / GB
repo_free = shutil.disk_usage(repo).free / GB
need = 20 if same_fs else 16
detail = f"{hf_free:.0f} GB free for downloads" + ("" if same_fs else f", {repo_free:.0f} GB for outputs")
check(hf_free >= need, hf_free >= 15, "disk", detail, f"need ~{need} GB (base model is ~13 GB)")
if not same_fs:
    check(repo_free >= 5, repo_free >= 2, "disk (outputs)", f"{repo_free:.0f} GB free", "checkpoints land here")

# AV1 decoder: these datasets are av1; without dav1d the fallback is far slower.
try:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True, timeout=20
    ).stdout
    has = "libdav1d" in out
    check(has, True, "av1 decode", "libdav1d" if has else "no libdav1d — decode will be slow")
except Exception:
    line("WARN", "av1 decode", "ffmpeg not found; pyav ships its own decoders, probably fine")

print()
print("== gpu ==")
try:
    import torch
except ImportError:
    line("FAIL", "torch", "not installed — run scripts/setup-cloud.sh first")
    print("\nFAILED")
    sys.exit(1)

print(f"  torch {torch.__version__}  built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    line("FAIL", "cuda", "torch.cuda.is_available() is False", )
    fails.append("cuda: not available")
else:
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    name = torch.cuda.get_device_name(0)
    line("ok", "device", f"{name}  capability {cap[0]}.{cap[1]} ({sm})")
    # A newer card can fall back to PTX JIT off an older sm_XX, which is slow and
    # not always present — require an exact match rather than trusting that.
    check(
        sm in archs, False,
        "kernels", f"{sm} in wheel [{', '.join(archs)}]",
        "pip install --index-url https://download.pytorch.org/whl/cu128 -U torch torchvision",
    )
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        (a @ a).sum().item()
        torch.cuda.synchronize()
        line("ok", "bf16 matmul", "executed on the GPU")
    except Exception as e:
        line("FAIL", "bf16 matmul", f"{type(e).__name__}: {e}")
        fails.append("bf16 matmul failed on the GPU")

    vram = torch.cuda.mem_get_info()[1] / GB
    # Starting points for pi0/pi05 LoRA + bf16 + gradient checkpointing, not measured
    # limits — --smoke is what actually settles it.
    suggested = 4 if vram < 12 else 8 if vram < 16 else 16 if vram < 26 else 24 if vram < 36 else 32
    check(vram >= 15, vram >= 10, "VRAM", f"{vram:.1f} GB  -> try --batch {suggested}")

print()
if fails:
    print("FAILED — fix these before training:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
if warns:
    print("usable, with caveats:")
    for w in warns:
        print(f"  - {w}")
else:
    print("all checks passed")
sys.exit(0)
PY

if [ "$smoke" = 0 ]; then
  echo
  echo "now prove the full path:  bash scripts/preflight.sh --smoke --batch <n>"
  exit 0
fi

echo
echo "== smoke: 20 real steps ($base on $dataset, batch $batch) =="
echo "   first run downloads the base model (~13GB)"
rm -rf outputs/train/gpu_smoke
bash scripts/train.sh --dataset "$dataset" --base "$base" --name gpu_smoke \
  --steps 20 --batch "$batch" --num_workers "$workers" --no-wandb

echo
echo "smoke OK — batch $batch fits and the whole path works."
echo "  for peak VRAM, watch \`nvidia-smi\` in a second shell while it runs"
echo "  clean up:  rm -rf outputs/train/gpu_smoke"
