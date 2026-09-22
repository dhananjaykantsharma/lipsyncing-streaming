"""Browser-facing WebSocket server + static file server for the POC frontend.

Protocol (browser <-> this server, over one WebSocket at /session):
  browser -> server   binary: the candidate's recorded answer (one blob, sent
                       once the "I'm done answering" button is pressed)
  server  -> browser  {"type": "answer_text", "text": ...}      STT result

  The LLM's answer is streamed and split into sentences (see session.py);
  each sentence gets its own TTS + Modal lip-sync turn, pipelined with the
  next sentence's LLM/TTS work. So the group below repeats once per
  sentence, with "final": true only on the last one:
  server  -> browser  {"type": "question_text", "text": ..., "final": bool}
  server  -> browser  binary: [0x01][WAV bytes]                 TTS audio for that sentence
  server  -> browser  {"type": "video_start", "width", "height", "fps"}
  server  -> browser  binary: [0x02][<Modal's own frame packet, untouched>]
  server  -> browser  {"type": "done", "final": bool}           this sentence finished

  server  -> browser  {"type": "error", "message": ...}
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from session import Session

app = FastAPI()

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
            audio_bytes = await websocket.receive_bytes()
            await session.handle_answer(audio_bytes)
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


# serve the plain HTML/CSS/JS frontend at http://localhost:8000/
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
