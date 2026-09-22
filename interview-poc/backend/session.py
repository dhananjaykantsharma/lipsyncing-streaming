"""One interview session = one browser connection = one Modal connection.
Orchestrates STT -> LLM -> (send text) -> TTS -> Modal -> (relay frames)."""
import json

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
        question = await llm.next_question(self.history)
        self.history.append({"role": "assistant", "content": question})
        await self._ask(question)

    async def handle_answer(self, audio_bytes: bytes, content_type: str = "audio/webm"):
        text = await stt.transcribe(audio_bytes, content_type=content_type)
        self.history.append({"role": "user", "content": text})
        await self.send_json({"type": "answer_text", "text": text})

        question = await llm.next_question(self.history)
        self.history.append({"role": "assistant", "content": question})
        await self._ask(question)

    async def _ask(self, question_text: str):
        await self.send_json({"type": "question_text", "text": question_text})

        audio_bytes = await tts.synthesize(question_text)
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
                await self.send_json({"type": "done"})

        async def on_binary(msg: bytes):
            await self.send_binary(bytes([TAG_VIDEO_FRAME]) + msg)

        try:
            await self.modal.stream_lipsync(audio_bytes, on_json, on_binary)
        except Exception as e:
            await self.send_json({"type": "error", "message": f"lipsync failed: {e!r}"})

    async def close(self):
        await self.modal.close()
