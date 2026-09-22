from image import app, musetalk_image


@app.function(image=musetalk_image, timeout=300)
def inspect_repo():
    import os

    root = "/root/MuseTalk"
    print(f"--- top-level contents of {root} ---")
    for name in sorted(os.listdir(root)):
        print(name)

    # find likely download scripts and README files
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        # skip deep/vendor dirs to keep output short
        if dirpath.count("/") - root.count("/") > 2:
            continue
        for f in filenames:
            lower = f.lower()
            if "download" in lower or lower.startswith("readme"):
                candidates.append(os.path.join(dirpath, f))

    print("\n--- candidate files (download scripts / readmes) ---")
    for path in candidates:
        print(path)

    print("\n--- contents ---")
    for path in candidates:
        try:
            size = os.path.getsize(path)
            if size > 20_000:
                print(f"\n### {path} (skipped, {size} bytes, too large) ###")
                continue
            with open(path, "r", errors="ignore") as fh:
                content = fh.read()
            print(f"\n### {path} ###")
            print(content)
        except Exception as e:
            print(f"could not read {path}: {e}")


@app.function(image=musetalk_image, timeout=300)
def inspect_inference():
    import os

    root = "/root/MuseTalk"

    for sub in ["scripts", "configs"]:
        d = os.path.join(root, sub)
        print(f"\n--- contents of {d} ---")
        for dirpath, _, filenames in os.walk(d):
            for f in filenames:
                print(os.path.join(dirpath, f))

    # things most likely to define the avatar-preprocessing / realtime API
    interesting = []
    for dirpath, _, filenames in os.walk(root):
        if dirpath.count("/") - root.count("/") > 3:
            continue
        for f in filenames:
            lower = f.lower()
            if "realtime" in lower or lower == "inference.sh" or (
                "inference" in lower and (lower.endswith(".py") or lower.endswith(".yaml"))
            ):
                interesting.append(os.path.join(dirpath, f))

    print("\n--- interesting files found ---")
    for p in interesting:
        print(p)

    print("\n--- contents ---")
    for path in interesting:
        try:
            size = os.path.getsize(path)
            if size > 30_000:
                print(f"\n### {path} (skipped, {size} bytes, too large) ###")
                continue
            with open(path, "r", errors="ignore") as fh:
                content = fh.read()
            print(f"\n### {path} ###")
            print(content)
        except Exception as e:
            print(f"could not read {path}: {e}")


@app.local_entrypoint()
def main():
    inspect_inference.remote()


@app.function(image=musetalk_image, timeout=120)
def show_requirements():
    with open("/root/MuseTalk/requirements.txt") as f:
        print(f.read())


@app.function(image=musetalk_image, timeout=120)
def grep_readme_torch():
    with open("/root/MuseTalk/README.md") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "torch" in line.lower() or "cuda" in line.lower() or "pip install" in line.lower():
            print(f"{i}: {line}", end="")


@app.function(image=musetalk_image, timeout=120)
def readme_lines():
    with open("/root/MuseTalk/README.md") as f:
        lines = f.readlines()
    for i, line in enumerate(lines[145:200], start=145):
        print(f"{i}: {line}", end="")


@app.function(image=musetalk_image, timeout=60)
def list_sample_audio():
    import os
    d = "/root/MuseTalk/data/audio"
    print(os.listdir(d) if os.path.exists(d) else f"{d} does not exist")


@app.function(image=musetalk_image, timeout=60)
def get_sample_audio() -> bytes:
    with open("/root/MuseTalk/data/audio/eng.wav", "rb") as f:
        return f.read()


@app.local_entrypoint()
def fetch_audio():
    import pathlib
    data = get_sample_audio.remote()
    out = pathlib.Path("test_audio.wav")
    out.write_bytes(data)
    print(f"Saved {len(data)} bytes to {out}")


@app.function(image=musetalk_image, timeout=60)
def show_blending():
    print(open("/root/MuseTalk/musetalk/utils/blending.py").read())
