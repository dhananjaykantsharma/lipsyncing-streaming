"""Browser-facing WebSocket server + static file server for the POC frontend.

Protocol (browser <-> this server, over one WebSocket at /session):
  The candidate's answer is uploaded WHILE they speak, not after:
  browser -> server   {"type": "answer_start"}                  "Start Answer" clicked
  browser -> server   binary: WebM/Opus chunk (every 250 ms)    appended to a buffer
  browser -> server   {"type": "answer_end", "chunks", "bytes"} "Done" -> buffer goes to STT
  browser -> server   {"type": "answer_cancel"}                 mic failed, drop the buffer
  (a binary message with no answer_start before it = a whole answer in one blob)
  server  -> browser  {"type": "answer_text", "text": ...}      STT result
  server  -> browser  {"type": "question_text", "text": ...}    next question
  server  -> browser  binary: [0x01][MP3 bytes]                 TTS audio
  server  -> browser  {"type": "video_start", "width", "height", "fps"}
  server  -> browser  binary: [0x02][<Modal's own frame packet, untouched>]
  server  -> browser  {"type": "done"}                          turn finished
  server  -> browser  {"type": "error", "message": ...}

Latency reporting (see latency_log.py; one JSON per session in backend/logs/):
  browser -> server   {"type": "ping", "id"}  ->  server -> browser {"type": "pong", "id"}
  browser -> server   {"type": "client_metrics", "turn", "metrics": {...}}
  (question_text / answer_text carry "turn" so the browser can tag its metrics)
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import http_clients
from session import Session


@asynccontextmanager
async def lifespan(app):
    yield
    await http_clients.aclose()


app = FastAPI(lifespan=lifespan)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.websocket("/session")
async def session_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def send_json(data):
        await websocket.send_json(data)

    async def send_binary(data):
        await websocket.send_bytes(data)

    session = Session(send_json, send_binary)
    try:
        await session.start()
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))
            if msg.get("bytes") is not None:
                if not session.add_answer_chunk(msg["bytes"]):
                    await session.handle_answer(msg["bytes"])
            elif msg.get("text") is not None:
                await session.handle_client_message(json.loads(msg["text"]))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send_json({"type": "error", "message": repr(e)})
        except Exception:
            pass
    finally:
        await session.close()
        try:
            await websocket.close()
        except Exception:
            pass  # already closed (e.g. client disconnected)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """Make the browser revalidate index.html / app.js on every load (a cheap
    304 when unchanged). Without it, after a redeploy the browser kept running
    its cached app.js (a test session ran the old 128 kbps recorder)."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


# serve the plain HTML/CSS/JS frontend at http://localhost:8000/
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
