"""Question text -> speech, via OpenAI TTS. WAV out (what MuseTalk's audio
pipeline expects), not the default mp3.

OpenAI TTS cannot accept incrementally-growing text (see llm.py's docstring),
but it CAN stream the audio bytes of a single call as they're generated
instead of making us wait for the whole clip — so `with_streaming_response`
gets us the completed WAV sooner than the plain non-streaming call would,
even though we still need every byte before handing it to Modal (which
needs a complete, valid WAV file)."""
from openai import AsyncOpenAI

from config import OPENAI_API_KEY

_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


async def synthesize(text: str, voice: str = "alloy") -> bytes:
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not set (backend/.env)")

    chunks = []
    async with _client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="wav",
    ) as response:
        async for chunk in response.iter_bytes():
            chunks.append(chunk)
    return b"".join(chunks)
