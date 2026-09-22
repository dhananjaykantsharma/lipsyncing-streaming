import modal

musetalk_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "curl",
    )
    # matches MuseTalk's README exactly (torch 2.0.1 + cu118) — mmcv only
    # ships prebuilt wheels for specific torch/cuda combos; anything else
    # forces a from-source build that needs a full CUDA devel toolchain
    # we don't have here, and fails.
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "opencv-python==4.9.0.80",
        "numpy==1.23.5",
        "librosa==0.11.0",
        "soundfile==0.12.1",
        "einops==0.8.1",
        "omegaconf",
        # MuseTalk's own requirements.txt pins diffusers==0.30.2, but that
        # version unconditionally touches torch.xpu at import time, which
        # doesn't exist before torch 2.4 -> crashes on our torch 2.0.1.
        # 0.27.2 predates that and is otherwise API-compatible for our use
        # (just loading an AutoencoderKL checkpoint).
        "diffusers==0.27.2",
        "accelerate==0.28.0",
        "transformers==4.39.2",
        # diffusers==0.27.2 still imports the deprecated `cached_download`
        # helper, which was removed in huggingface_hub 0.26+ — pin below
        # that so both diffusers and our own snapshot_download/hf_hub_download
        # calls in download_weights() keep working.
        "huggingface_hub==0.23.4",
        "imageio",
        "imageio-ffmpeg",
        "ffmpeg-python",
        "moviepy",
        "decord",
        "gdown",
        "fastapi",
        "uvicorn",
        "websockets",
    )
    # MuseTalk's landmark/pose detection (used inside get_landmark_and_bbox)
    # depends on the OpenMMLab stack, which their README installs via `mim`
    # rather than plain pip.
    .pip_install("openmim")
    .run_commands(
        "mim install mmengine",
        'mim install "mmcv==2.0.1"',
        'mim install "mmdet==3.1.0"',
        'mim install "mmpose==1.1.0"',
    )
    .run_commands(
        "git clone https://github.com/TMElyralab/MuseTalk.git /root/MuseTalk"
    )
    # PyAV bundles libx264; used for low-latency H.264 streaming (h264util.py).
    # No hard deps, so it won't disturb the numpy/torch pins above.
    .pip_install("av==17.1.0")
    .add_local_python_source("image", "volume", "h264util", "blendutil")
)


app = modal.App("musetalk-lipsync", image=musetalk_image)