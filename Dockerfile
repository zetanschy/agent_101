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

WORKDIR /workspace
CMD ["bash"]
