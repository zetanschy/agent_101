FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: python, video/ffmpeg, serial (feetech bus), and the X libs
# the rerun native viewer needs for --display_data.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev \
        build-essential linux-libc-dev \
        git ffmpeg v4l-utils libusb-1.0-0 \
        libgl1 libglib2.0-0 \
        libx11-6 libxext6 libxrender1 libxkbcommon0 libegl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

# LeRobot + Feetech motor bus support (SO-ARM101 uses STS3215 servos).
# Pinned to match the host install (0.4.1): later versions renamed the
# calibration subfolders (so101_leader -> so_leader), which invalidates the
# existing calibration files and forces a re-calibration on connect.
RUN pip install --no-cache-dir "lerobot[feetech]==0.4.1"

# rerun viewer (wgpu) rendering deps, added last so they don't invalidate the
# torch/lerobot layer above. Mesa software Vulkan (lavapipe) lets the GUI render
# over X11 without NVIDIA Vulkan passthrough; libxcb*/libxkbcommon-x11 are the
# winit/X11 libs the viewer window needs. XDG_RUNTIME_DIR silences a viewer warning.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers \
        libxcb1 libxcb-render0 libxcb-shape0 libxcb-xfixes0 libxcb-cursor0 \
        libxkbcommon-x11-0 \
    && mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["bash"]
