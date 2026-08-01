# Unified image for BOTH the robot workflow and training, built from your
# vendored lerobot fork (thirdparty/le101, v0.6.1) — LoRA/PEFT, RL (hilserl),
# rollout/DAgger, newest VLAs. Installed editable so you can patch the fork and
# see changes live (source is mounted over /opt/le101 at runtime).
#
#   robot:    docker-compose.yml        (devices + cameras + X11)
#   training: docker-compose.train.yml  (GPU only, no devices)
# The previous pinned-0.4.1 Dockerfile is in git history if ever needed.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    WGPU_BACKEND=vulkan

# System deps: build tools, ffmpeg/av, feetech serial, audio/geos (lerobot deps),
# v4l-utils for cameras, plus Mesa software Vulkan + X libs for the rerun viewer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl cmake pkg-config ninja-build \
        ffmpeg v4l-utils libglib2.0-0 libgl1 libegl1 libegl1-mesa-dev \
        libusb-1.0-0-dev speech-dispatcher libgeos-dev portaudio19-dev \
        libvulkan1 mesa-vulkan-drivers \
        libx11-6 libxext6 libxrender1 libxkbcommon0 \
        libxcb1 libxcb-render0 libxcb-shape0 libxcb-xfixes0 libxcb-cursor0 libxkbcommon-x11-0 \
    && mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# CUDA torch first (the fork targets cu128; wheels bundle their own CUDA runtime,
# GPU comes from the nvidia container runtime). Pinned to the fork's range.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7,<2.12" "torchvision>=0.22,<0.27"

# Editable install of the fork with the extras we use:
#   feetech      -> SO-ARM motors        core_scripts -> dataset+hardware+viz (teleop/record/calibrate)
#   training     -> trainer deps         pi -> pi0/pi05    peft -> LoRA adapters
COPY thirdparty/le101 /opt/le101
RUN pip install --no-cache-dir -e "/opt/le101[feetech,core_scripts,training,pi,peft]"

# Web UI backend (small; last layer so it doesn't invalidate the lerobot layer).
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

# openpi (JAX) alongside torch, so ONE web UI can load either stack's policies and
# switch between them per model. This is safe despite appearances: jax's CUDA deps are
# all >= bounds and torch cu128's exact pins satisfy every one of them
# (cudnn 9.19.0.56 vs >=9.1,<10.0; nccl 2.28.9 vs >=2.18.1), verified by running a
# torch bf16 GPU matmul and a jax GPU matmul in the same process.
#
# --no-deps for openpi itself: its dependency set pins lerobot to an UPSTREAM commit
# (which would replace the le101 fork), torch==2.7.1, and private i2rt. The list here
# is what openpi.training.config and policy_config actually import — pytest included,
# since openpi imports it at module level in models_pytorch/gemma_pytorch.py.
# transformers is deliberately NOT pinned to openpi's 4.53.2, to leave lerobot's own
# version alone; the JAX path only needs it for the tokenizer.
RUN pip install --no-cache-dir "jax[cuda12]==0.5.3" \
        "flax==0.10.2" "orbax-checkpoint==0.11.13" "jaxtyping==0.2.36" \
        "ml_collections==1.0.0" "beartype==0.19.0" \
        "augmax>=0.3.4" "equinox>=0.11.8" "etils[epath]" "treescope>=0.1.7" \
        "numpydantic>=1.6.6" "dm-tree>=0.1.8" "flatbuffers>=24.3.25" \
        "fsspec[gcs]>=2024.6.0" "tqdm-loggable>=0.2" "tyro>=0.9.5" \
        "sentencepiece>=0.2.0" "einops>=0.8.0" optax chex pytest
COPY thirdparty/openpi /opt/openpi
RUN pip install --no-cache-dir --no-deps -e /opt/openpi \
    && pip install --no-cache-dir --no-deps -e /opt/openpi/packages/openpi-client \
    && python -c "import jax, torch; from openpi.training import config; \
print('jax', jax.__version__, '| torch', torch.__version__, '| openpi importable')"

WORKDIR /workspace
CMD ["bash"]
