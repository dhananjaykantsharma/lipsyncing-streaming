"""One persistent connection to Modal's MuseTalk /ws-stream endpoint, reused
across every turn of one interview session (Modal's own server already loops
`while True: audio_bytes = await websocket.receive_bytes()`, so we don't
need to reconnect per turn — that also keeps the GPU container warm)."""
import asyncio
import json
import time

import websockets

from config import MODAL_WS_URL
from latency_log import ms_since

# gap between two consecutive video packets above which playback would have
# to hold a frame (25 fps => 40 ms per frame)
FRAME_INTERVAL_MS = 40


class ModalRelay:
    def __init__(self):
        self._ws = None

    async def connect(self) -> float | None:
        """Returns how long the handshake took (ms), or None if already connected."""
        if self._ws is not None:
            return None
        t0 = time.perf_counter()
        # open_timeout=600: a cold Modal container can take 40-90s to
        # boot (model loading) before it even accepts the handshake —
        # the library's 10s default times out well before that.
        # ping_interval=None: a slow link can queue data ahead of the
        # pong and trip the keepalive.
        # (both match tools/ws_h264_view.py, which hits the same server)
        self._ws = await websockets.connect(
            MODAL_WS_URL, max_size=None, ping_interval=None, open_timeout=600
        )
        return ms_since(t0)

    async def close(self):
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def ping_ms(self) -> float | None:
        """One websocket ping/pong round trip to Modal = pure network RTT."""
        try:
            pong = await self._ws.ping()
            return round(await asyncio.wait_for(pong, timeout=10) * 1000, 1)
        except Exception:
            return None

    async def stream_lipsync(self, audio_bytes: bytes, on_json, on_binary, metrics: dict | None = None):
        """Sends audio_bytes, then relays everything Modal sends back until
        its {"type": "done"} arrives. `on_json` / `on_binary` are async
        callbacks invoked per message, in order. Returns the "done" message's
        stats dict (or None). `metrics`, if given, is filled with the
        backend-side timings (relative to the moment the audio is sent)."""
        m = metrics if metrics is not None else {}
        m["connect_ms"] = await self.connect()  # null => connection reused (warm)
        m["ping_rtt_ms"] = await self.ping_ms()

        t0 = time.perf_counter()
        await self._ws.send(audio_bytes)
        m["audio_upload_bytes"] = len(audio_bytes)
        m["send_ms"] = ms_since(t0)

        packets = 0
        video_bytes = 0
        relay_ms = 0.0
        last = None
        gaps = []
        while True:
            msg = await self._ws.recv()
            now = time.perf_counter()
            if isinstance(msg, (bytes, bytearray)):
                if packets == 0:
                    m["first_packet_ms"] = ms_since(t0)
                else:
                    gaps.append((now - last) * 1000)
                last = now
                packets += 1
                video_bytes += len(msg)
                t_relay = time.perf_counter()
                await on_binary(msg)
                relay_ms += time.perf_counter() - t_relay
                continue
            data = json.loads(msg)
            if data.get("type") == "start":
                m["start_msg_ms"] = ms_since(t0)
            await on_json(data)
            if data.get("type") == "done":
                m["done_ms"] = ms_since(t0)
                if last is not None:
                    m["last_packet_ms"] = round((last - t0) * 1000, 1)
                m["packets"] = packets
                m["video_bytes"] = video_bytes
                # time spent pushing frames on to the browser socket
                m["browser_relay_total_ms"] = round(relay_ms * 1000, 1)
                if gaps:
                    m["packet_gap_max_ms"] = round(max(gaps), 1)
                    m["packet_gap_mean_ms"] = round(sum(gaps) / len(gaps), 1)
                    m["packet_gaps_over_frame_interval"] = sum(g > FRAME_INTERVAL_MS for g in gaps)
                return data.get("stats")
