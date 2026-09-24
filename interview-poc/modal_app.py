"""The interview backend + frontend, deployed on Modal (CPU only, no GPU).

    modal deploy modal_app.py      (from interview-poc/)

gives a permanent HTTPS URL (https://<workspace>--interview-backend-web.modal.run)
serving the same FastAPI app as `uvicorn server:app` locally. HTTPS matters:
browsers only allow the microphone on https:// (or localhost).

Keys come from the Modal secret "interview-backend" (OPENAI_API_KEY,
DEEPGRAM_API_KEY, MODAL_WS_URL = the *deployed* GPU app's /ws-stream URL),
not from backend/.env, which is never uploaded.

Latency logs go to the "interview-latency-logs" Volume:
    modal volume get interview-latency-logs / ./backend/logs
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    # same versions as the local backend venv the code was tested with
    .pip_install(
        "fastapi==0.141.1",
        "uvicorn[standard]==0.53.0",
        "python-dotenv==1.2.3",
        "httpx==0.28.1",
        "httpx2==2.13.0",
        "openai==3.17.0",
        "websockets==16.1.1",
    )
    .env({"LATENCY_LOG_DIR": "/logs"})
    .add_local_dir(
        HERE / "backend",
        "/app/backend",
        ignore=["venv", "**/__pycache__", "logs", ".env"],
    )
    .add_local_dir(HERE / "frontend", "/app/frontend")
)

app = modal.App("interview-backend", image=image)
logs = modal.Volume.from_name("interview-latency-logs", create_if_missing=True)


@app.function(
    secrets=[modal.Secret.from_name("interview-backend")],
    volumes={"/logs": logs},
    # scale to zero like the GPU app: stop after 3 min with no open session
    scaledown_window=180,
    # one interview = one websocket = one long-running request
    timeout=3600,
)
# many interviews share one small CPU container (each still gets its own GPU container)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import os
    import sys

    os.chdir("/app/backend")
    sys.path.insert(0, "/app/backend")
    from server import app as fastapi_app

    return fastapi_app
