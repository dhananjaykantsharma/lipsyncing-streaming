"""One persistent connection to Modal's MuseTalk /ws-stream endpoint, reused
across every turn of one interview session (Modal's own server already loops
`while True: audio_bytes = await websocket.receive_bytes()`, so we don't
need to reconnect per turn — that also keeps the GPU container warm)."""
import json

import websockets

from config import MODAL_WS_URL


class ModalRelay:
    def __init__(self):
        self._ws = None

    async def connect(self):
        if self._ws is None:
            # open_timeout=600: a cold Modal container can take 40-90s to
            # boot (model loading) before it even accepts the handshake —
            # the library's 10s default times out well before that.
            # ping_interval=None: a slow link can queue data ahead of the
            # pong and trip the keepalive.
            # (both match tools/ws_h264_view.py, which hits the same server)
            self._ws = await websockets.connect(
                MODAL_WS_URL, max_size=None, ping_interval=None, open_timeout=600
            )

    async def close(self):
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def stream_lipsync(self, audio_bytes: bytes, on_json, on_binary):
        """Sends audio_bytes, then relays everything Modal sends back until
        its {"type": "done"} arrives. `on_json` / `on_binary` are async
        callbacks invoked per message, in order. Returns the "done" message's
        stats dict (or None)."""
        await self.connect()
        await self._ws.send(audio_bytes)
        while True:
            msg = await self._ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                await on_binary(msg)
                continue
            data = json.loads(msg)
            await on_json(data)
            if data.get("type") == "done":
                return data.get("stats")
