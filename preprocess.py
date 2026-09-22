from image import app, musetalk_image
from volume import volume, MODEL_DIR, AVATAR_DIR


@app.function(
    image=musetalk_image,
    volumes={"/data": volume},
    timeout=1800,
)
def download_weights():
    """One-off: download every checkpoint MuseTalk's pipeline needs from
    HuggingFace (+ two files that live outside HF) into the Volume, laid
    out exactly like MuseTalk's own download_weights.sh expects (models/musetalk,
    models/musetalkV15, models/sd-vae, models/whisper, models/dwpose,
    models/syncnet, models/face-parse-bisent), so the live inference
    container never has to fetch anything at request time."""
    import os
    import shutil
    import subprocess
    from huggingface_hub import snapshot_download, hf_hub_download

    target = MODEL_DIR
    marker = f"{target}/.download_complete"

    if os.path.exists(marker):
        print("Weights already present, skipping.")
        return

    # earlier run left an incorrectly-nested models/musetalk/musetalk/...
    # layout — wipe it so this run starts clean with the correct structure
    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)

    print("Downloading MuseTalk (v1.0 + v1.5)...")
    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=target,
        allow_patterns=[
            "musetalk/musetalk.json",
            "musetalk/pytorch_model.bin",
            "musetalkV15/musetalk.json",
            "musetalkV15/unet.pth",
        ],
    )

    print("Downloading SD VAE...")
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        local_dir=f"{target}/sd-vae",
        allow_patterns=["config.json", "diffusion_pytorch_model.bin"],
    )

    print("Downloading Whisper...")
    snapshot_download(
        repo_id="openai/whisper-tiny",
        local_dir=f"{target}/whisper",
        allow_patterns=["config.json", "pytorch_model.bin", "preprocessor_config.json"],
    )

    print("Downloading DWPose...")
    os.makedirs(f"{target}/dwpose", exist_ok=True)
    hf_hub_download(
        repo_id="yzd-v/DWPose",
        filename="dw-ll_ucoco_384.pth",
        local_dir=f"{target}/dwpose",
    )

    print("Downloading SyncNet...")
    os.makedirs(f"{target}/syncnet", exist_ok=True)
    hf_hub_download(
        repo_id="ByteDance/LatentSync",
        filename="latentsync_syncnet.pt",
        local_dir=f"{target}/syncnet",
    )

    print("Downloading face-parse-bisent (not on HF)...")
    face_dir = f"{target}/face-parse-bisent"
    os.makedirs(face_dir, exist_ok=True)
    subprocess.run(
        [
            "gdown",
            "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
            "-O", f"{face_dir}/79999_iter.pth",
        ],
        check=True,
    )
    subprocess.run(
        [
            "curl", "-L",
            "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            "-o", f"{face_dir}/resnet18-5c106cde.pth",
        ],
        check=True,
    )

    with open(marker, "w") as f:
        f.write("ok")

    volume.commit()
    print(f"All weights downloaded to {target}")


@app.function(
    image=musetalk_image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=1800,
)
def preprocess_avatar(
    avatar_id: str = "avatar_1",
    avatar_filename: str = "avatar.mp4",
    version: str = "v15",
):
    """One-off: reuse MuseTalk's own `Avatar` (from scripts/realtime_inference.py)
    to do avatar preprocessing (face detect, crop, VAE-encode into latents,
    mask generation) on the fixed sample avatar video, and cache the result
    to the Volume so the live inference container just loads it instead of
    recomputing per request."""
    import os
    import shutil
    import sys
    import argparse
    import torch

    repo = "/root/MuseTalk"
    os.chdir(repo)
    sys.path.insert(0, repo)

    avatar_video_path = f"{AVATAR_DIR}/{avatar_filename}"
    if not os.path.exists(avatar_video_path):
        raise FileNotFoundError(
            f"{avatar_video_path} not found — upload the avatar video to "
            f"the volume first (modal volume put musetalk-data <local> "
            f"avatars/{avatar_filename})."
        )

    # MuseTalk writes/reads relative "./results/..." and "./models/...".
    # Symlink both to our persistent Volume paths so nothing it does is lost
    # when the container exits, and so it finds the weights we downloaded.
    results_target = f"{AVATAR_DIR}/results"
    os.makedirs(results_target, exist_ok=True)
    if not os.path.lexists(f"{repo}/results"):
        os.symlink(results_target, f"{repo}/results")
    if not os.path.lexists(f"{repo}/models"):
        os.symlink(MODEL_DIR, f"{repo}/models")

    from musetalk.utils.utils import load_all_model
    from musetalk.utils.face_parsing import FaceParsing
    import scripts.realtime_inference as rt

    if version == "v15":
        model_dir = f"{repo}/models/musetalkV15"
        unet_model_path = f"{model_dir}/unet.pth"
    else:
        model_dir = f"{repo}/models/musetalk"
        unet_model_path = f"{model_dir}/pytorch_model.bin"
    unet_config = f"{model_dir}/musetalk.json"

    device = torch.device("cuda")
    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type="sd-vae",
        unet_config=unet_config,
        device=device,
    )
    fp = (
        FaceParsing(left_cheek_width=90, right_cheek_width=90)
        if version == "v15"
        else FaceParsing()
    )

    # Avatar (inside scripts/realtime_inference.py) reads these as
    # MODULE-LEVEL globals rather than constructor args — inject them
    # before instantiating it.
    rt.args = argparse.Namespace(version=version, extra_margin=10, parsing_mode="jaw")
    rt.vae = vae
    rt.fp = fp

    # avoid Avatar.init()'s interactive "re-create? (y/n)" input() prompt,
    # which would hang forever in a non-interactive container — just wipe
    # any previous attempt for this avatar_id and let it build fresh.
    avatar_dir = (
        f"{results_target}/{version}/avatars/{avatar_id}"
        if version == "v15"
        else f"{results_target}/avatars/{avatar_id}"
    )
    if os.path.exists(avatar_dir):
        shutil.rmtree(avatar_dir)

    rt.Avatar(
        avatar_id=avatar_id,
        video_path=avatar_video_path,
        bbox_shift=0,
        batch_size=8,
        preparation=True,
    )

    volume.commit()
    print(f"Avatar '{avatar_id}' preprocessed and cached under {avatar_dir}")


@app.local_entrypoint()
def main():
    download_weights.remote()
    preprocess_avatar.remote()
