"""Speech-to-text via Deepgram's prerecorded (batch) API.

Push-to-talk means we already have the candidate's whole answer as one blob
by the time we call this, so the simple REST endpoint is enough — no need
for Deepgram's streaming/websocket API."""
import httpx

from config import DEEPGRAM_API_KEY

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


async def transcribe(audio_bytes: bytes, content_type: str = "audio/webm") -> str:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not set (backend/.env)")

    print(f"[stt] received {len(audio_bytes)} bytes, content_type={content_type}")

    params = {"model": "nova-2", "smart_format": "true", "punctuate": "true"}
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": content_type}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(DEEPGRAM_URL, params=params, headers=headers, content=audio_bytes)
        resp.raise_for_status()
        data = resp.json()

    print(f"[stt] deepgram response: {data}")

    try:
        return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError):
        return ""
