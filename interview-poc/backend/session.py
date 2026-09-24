"""One interview session = one browser connection = one Modal connection.
Orchestrates STT -> LLM -> (send text) -> TTS -> Modal -> (relay frames),
timing every step into a per-session JSON log (see latency_log.py)."""
import asyncio
import time

import http_clients
import llm
import stt
import tts
from latency_log import SessionLog, ms_since
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
        self.log = SessionLog()
        self._keep_warm = None
        print(f"[session] latency log: {self.log.path}")

    async def start(self):
        """No candidate answer yet — ask the opening question."""
        # keeps the pooled Deepgram/OpenAI connections open between turns
        self._keep_warm = asyncio.create_task(http_clients.keep_warm())
        turn = self.log.new_turn("opening")
        try:
            await self._respond(turn)
        except Exception as e:
            turn["errors"].append(repr(e))
            raise
        finally:
            _derive_network(turn)
            self.log.save()

    async def handle_answer(self, audio_bytes: bytes, content_type: str = "audio/webm"):
        turn = self.log.new_turn("answer")
        try:
            t = time.perf_counter()
            text = await stt.transcribe(audio_bytes, content_type=content_type, metrics=turn["stt"])
            _finish_vendor_call(turn["stt"], t)
            self.log.mark(turn, "stt_done")
            turn["answer_text"] = text

            self.history.append({"role": "user", "content": text})
            await self.send_json({"type": "answer_text", "text": text, "turn": turn["turn"]})
            self.log.mark(turn, "answer_text_sent")

            await self._respond(turn)
        except Exception as e:
            turn["errors"].append(repr(e))
            raise
        finally:
            _derive_network(turn)
            self.log.save()

    async def handle_client_message(self, data: dict):
        """Browser -> backend control messages (latency reporting)."""
        if data.get("type") == "ping":
            await self.send_json({"type": "pong", "id": data.get("id")})
        elif data.get("type") == "client_metrics":
            turn = self.log.turn(data.get("turn"))
            if turn is None:
                return
            client = data.get("metrics") or {}
            turn["client"] = client
            net = turn["network"]
            net["browser_rtt_ms"] = client.get("ws_rtt_ms")
            # browser measured send -> answer_text; backend measured
            # receive -> answer_text sent. The difference is the answer
            # upload + one downlink hop.
            a, b = client.get("answer_text_after_send_ms"), turn["timeline_ms"].get("answer_text_sent")
            if a is not None and b is not None:
                net["browser_answer_upload_ms"] = round(a - b, 1)
            self.log.save()

    async def close(self):
        if self._keep_warm is not None:
            self._keep_warm.cancel()
        self.log.close()
        await self.modal.close()

    async def _respond(self, turn: dict):
        t = time.perf_counter()
        question = await llm.next_question(self.history, metrics=turn["llm"])
        _finish_vendor_call(turn["llm"], t)
        self.log.mark(turn, "llm_done")
        turn["question_text"] = question

        self.history.append({"role": "assistant", "content": question})
        await self._ask(question, turn)

    async def _ask(self, question_text: str, turn: dict):
        await self.send_json({"type": "question_text", "text": question_text, "turn": turn["turn"]})
        self.log.mark(turn, "question_text_sent")

        t = time.perf_counter()
        audio_bytes = await tts.synthesize(question_text, metrics=turn["tts"])
        _finish_vendor_call(turn["tts"], t)
        self.log.mark(turn, "tts_done")

        await self.send_binary(bytes([TAG_TTS_AUDIO]) + audio_bytes)
        self.log.mark(turn, "tts_audio_sent")

        first_frame_marked = False

        async def on_json(data):
            if data.get("type") == "start":
                self.log.mark(turn, "modal_start_msg")
                await self.send_json({
                    "type": "video_start",
                    "width": data.get("width"),
                    "height": data.get("height"),
                    "fps": data.get("fps"),
                })
            elif data.get("type") == "done":
                self.log.mark(turn, "modal_done")
                await self.send_json({"type": "done"})

        async def on_binary(msg: bytes):
            nonlocal first_frame_marked
            if not first_frame_marked:
                first_frame_marked = True
                self.log.mark(turn, "modal_first_packet")
            await self.send_binary(bytes([TAG_VIDEO_FRAME]) + msg)

        try:
            self.log.mark(turn, "modal_request_start")
            server_stats = await self.modal.stream_lipsync(audio_bytes, on_json, on_binary, metrics=turn["modal"])
            turn["modal_server"] = server_stats
        except Exception as e:
            turn["errors"].append(f"lipsync: {e!r}")
            await self.send_json({"type": "error", "message": f"lipsync failed: {e!r}"})


def _finish_vendor_call(m: dict, t0: float):
    """total_ms, and the part of it that was NOT the vendor's own compute."""
    m["total_ms"] = ms_since(t0)
    if m.get("server_processing_ms") is not None:
        m["network_overhead_ms"] = round(m["total_ms"] - m["server_processing_ms"], 1)


def _derive_network(turn: dict):
    """Network share of each hop. For Modal: backend-observed time minus
    Modal's own server-side time (audio upload + packet download + Modal's
    websocket proxy). For STT/LLM/TTS: total minus the vendor's compute."""
    m, s, net = turn["modal"], turn["modal_server"] or {}, turn["network"]
    net["modal_ping_rtt_ms"] = m.get("ping_rtt_ms")
    if m.get("first_packet_ms") is not None and s.get("server_first_packet_s") is not None:
        net["modal_first_packet_overhead_ms"] = round(m["first_packet_ms"] - s["server_first_packet_s"] * 1000, 1)
    if m.get("done_ms") is not None and s.get("server_total_s") is not None:
        net["modal_total_overhead_ms"] = round(m["done_ms"] - s["server_total_s"] * 1000, 1)
    for k in ("stt", "llm", "tts"):
        overhead = turn[k].get("network_overhead_ms")
        if overhead is None and k == "stt":
            n = turn["stt"].get("net") or {}
            # everything except the server_wait (which includes Deepgram compute)
            parts = [n.get(p) for p in ("tcp_connect_ms", "tls_handshake_ms", "upload_ms", "download_ms")]
            overhead = round(sum(p for p in parts if p), 1) if n else None
        net[f"{k}_network_overhead_ms"] = overhead
