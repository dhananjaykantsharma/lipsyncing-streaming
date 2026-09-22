"""Question text -> speech, via OpenAI TTS. WAV out (what MuseTalk's audio
pipeline expects), not the default mp3."""
from openai import AsyncOpenAI

from config import OPENAI_API_KEY

_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


async def synthesize(text: str, voice: str = "alloy") -> bytes:
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not set (backend/.env)")

    response = await _client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="wav",
    )
    return response.content
