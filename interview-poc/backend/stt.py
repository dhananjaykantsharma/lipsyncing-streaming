"""Speech-to-text via Deepgram's prerecorded (batch) API.

Push-to-talk means we already have the candidate's whole answer as one blob
by the time we call this, so the simple REST endpoint is enough — no need
for Deepgram's streaming/websocket API."""
from config import DEEPGRAM_API_KEY
from http_clients import deepgram, traced


async def transcribe(audio_bytes: bytes, content_type: str = "audio/webm", metrics: dict | None = None) -> str:
    """`metrics`, if given, is filled with the network phase split and the
    answer's audio duration."""
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not set (backend/.env)")

    print(f"[stt] received {len(audio_bytes)} bytes, content_type={content_type}")

    params = {"model": "nova-2", "smart_format": "true", "punctuate": "true"}
    headers = {"Content-Type": content_type}

    # shared client (http_clients.py): the TLS connection is reused across turns
    with traced() as trace:
        resp = await deepgram.post("/v1/listen", params=params, headers=headers, content=audio_bytes)
    resp.raise_for_status()
    data = resp.json()

    print(f"[stt] deepgram response: {data}")

    if metrics is not None:
        meta = data.get("metadata", {})
        metrics["audio_bytes"] = len(audio_bytes)
        metrics["audio_duration_s"] = meta.get("duration")
        metrics["net"] = trace.phases()

    try:
        return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError):
        return ""
