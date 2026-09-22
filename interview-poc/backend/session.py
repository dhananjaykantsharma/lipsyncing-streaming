"""One interview session = one browser connection = one Modal connection.

Orchestrates STT -> LLM (streamed, sentence-by-sentence) -> TTS -> Modal ->
relay frames. A multi-sentence question is pipelined: sentence 2's TTS
starts generating (as a background task) the moment the LLM finishes
streaming it, while sentence 1 is still being lip-synced by Modal — so by
the time Modal is ready for sentence 2, its audio is usually already there."""
import asyncio

import llm
import stt
import tts
from modal_client import ModalRelay

TAG_TTS_AUDIO = 0x01
TAG_VIDEO_FRAME = 0x02


class Session:
    def __init__(self, send_json, send_binary):
        """send_json(dict) / send_binary(bytes): both async, write to the browser's WebSocket."""
        self.send_json = send_json
        self.send_binary = send_binary
        self.history: list[dict] = []
        self.modal = ModalRelay()

    async def start(self):
        """No candidate answer yet — ask the opening question."""
        await self._respond()

    async def handle_answer(self, audio_bytes: bytes, content_type: str = "audio/webm"):
        text = await stt.transcribe(audio_bytes, content_type=content_type)
        self.history.append({"role": "user", "content": text})
        await self.send_json({"type": "answer_text", "text": text})
        await self._respond()

    async def _respond(self):
        """Pulls sentences off llm.stream_question() and, for each one, kicks
        off its TTS immediately (as a background task) rather than waiting —
        so LLM+TTS for the NEXT sentence overlap with the CURRENT sentence's
        Modal lip-sync. A queue keeps them in order for Modal/the browser,
        which both need sentences delivered sequentially."""
        queue: asyncio.Queue = asyncio.Queue()
        parts: list[str] = []

        async def produce():
            async for sentence, is_final in llm.stream_question(self.history):
                parts.append(sentence)
                tts_task = asyncio.create_task(tts.synthesize(sentence))
                await queue.put((sentence, is_final, tts_task))
            await queue.put(None)  # sentinel: no more sentences

        producer = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                sentence, is_final, tts_task = item
                await self._ask_sentence(sentence, is_final, tts_task)
        finally:
            await producer  # propagate any exception raised while streaming the LLM

        if parts:
            self.history.append({"role": "assistant", "content": " ".join(parts)})

    async def _ask_sentence(self, text: str, is_final: bool, tts_task: asyncio.Task):
        await self.send_json({"type": "question_text", "text": text, "final": is_final})

        audio_bytes = await tts_task
        await self.send_binary(bytes([TAG_TTS_AUDIO]) + audio_bytes)

        async def on_json(data):
            if data.get("type") == "start":
                await self.send_json({
                    "type": "video_start",
                    "width": data.get("width"),
                    "height": data.get("height"),
                    "fps": data.get("fps"),
                })
            elif data.get("type") == "done":
                await self.send_json({"type": "done", "final": is_final})

        async def on_binary(msg: bytes):
            await self.send_binary(bytes([TAG_VIDEO_FRAME]) + msg)

        try:
            await self.modal.stream_lipsync(audio_bytes, on_json, on_binary)
        except Exception as e:
            await self.send_json({"type": "error", "message": f"lipsync failed: {e!r}"})

    async def close(self):
        await self.modal.close()
