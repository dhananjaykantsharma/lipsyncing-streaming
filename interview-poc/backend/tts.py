"""Question text -> speech, via OpenAI TTS.

MP3, not WAV: the same bytes are downloaded from OpenAI, uploaded to Modal
and sent to the browser, and 24 kHz 16-bit WAV is ~384 kbps. MuseTalk loads
it with librosa (decodes MP3 fine) and the browser's decodeAudioData does too."""
import struct

from http_clients import openai_client as _client, traced
from latency_log import header_ms

TTS_FORMAT = "mp3"


def _wav_duration_s(wav: bytes):
    """Duration from the fmt chunk + byte count (OpenAI's streamed WAV header
    has a placeholder data size, so the header's own length can't be trusted)."""
    try:
        channels, sample_rate = struct.unpack_from("<HI", wav, 22)
        bits = struct.unpack_from("<H", wav, 34)[0]
        return round((len(wav) - 44) / (sample_rate * channels * bits // 8), 3), sample_rate
    except struct.error:
        return None, None


async def synthesize(text: str, voice: str = "alloy", metrics: dict | None = None) -> bytes:
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not set (backend/.env)")

    with traced() as trace:
        raw = await _client.audio.speech.with_raw_response.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format=TTS_FORMAT,
        )
        audio = raw.content

    if metrics is not None:
        metrics["net"] = trace.phases()
        metrics["server_processing_ms"] = header_ms(raw.headers, "openai-processing-ms")
        metrics["format"] = TTS_FORMAT
        metrics["chars"] = len(text)
        metrics["audio_bytes"] = len(audio)
        if TTS_FORMAT == "wav":
            metrics["audio_duration_s"], metrics["sample_rate"] = _wav_duration_s(audio)
        # for mp3 the duration is in modal_server.audio_duration_s / client.audio_duration_s

    return audio
