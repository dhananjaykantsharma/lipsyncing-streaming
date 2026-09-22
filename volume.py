import modal

# one shared persistent disk for model weights + avatar cache
volume = modal.Volume.from_name("musetalk-data", create_if_missing=True)

MODEL_DIR = "/data/models"
AVATAR_DIR = "/data/avatars"
