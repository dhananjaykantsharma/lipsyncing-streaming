"""Per-session latency log: one JSON file per interview session under
backend/logs/, rewritten (atomically) after every update so it can be read
while the session is still running.

All durations are milliseconds unless the key says otherwise. Each turn
records:
  timeline_ms   when each step finished, relative to the turn's start
                (answer received by the backend / session start for turn 0)
  stt/llm/tts   per-service total + server-side vs network split
  modal         backend-side view of the Modal websocket (RTT, first packet, ...)
  modal_server  Modal's own stats (whisper, GPU, blend, encode)
  network       derived: time that is neither our code nor a vendor's compute
  client        browser-side view (what the candidate actually experienced)
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def header_ms(headers, name: str):
    try:
        return float(headers.get(name))
    except (TypeError, ValueError):
        return None


class HttpTrace:
    """httpx `trace` extension callback -> network phase timings.

    Pass as `extensions={"trace": trace}`; after the response body is read,
    phases() splits the request into connect / TLS / upload / server wait /
    download. `server_wait_ms` is the vendor's processing time + one RTT."""

    def __init__(self):
        self.events: dict[str, float] = {}

    async def __call__(self, name: str, info: dict):
        # "http11.send_request_body.complete" -> "send_request_body.complete"
        for prefix in ("http11.", "http2.", "connection."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        self.events[name] = time.perf_counter()

    def _span(self, start: str, end: str):
        a, b = self.events.get(start), self.events.get(end)
        return round((b - a) * 1000, 1) if a is not None and b is not None else None

    def phases(self) -> dict:
        return {
            # false => pooled connection reused, no TCP/TLS handshake on this call
            "new_connection": "connect_tcp.started" in self.events,
            "tcp_connect_ms": self._span("connect_tcp.started", "connect_tcp.complete"),
            "tls_handshake_ms": self._span("start_tls.started", "start_tls.complete"),
            "upload_ms": self._span("send_request_headers.started", "send_request_body.complete"),
            "server_wait_ms": self._span("send_request_body.complete", "receive_response_headers.complete"),
            "download_ms": self._span("receive_response_headers.complete", "receive_response_body.complete"),
        }


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


class SessionLog:
    SUMMARY_SECTIONS = ("timeline_ms", "stt", "llm", "tts", "modal", "modal_server", "network", "client")

    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        self.session_id = uuid.uuid4().hex[:8]
        self.path = LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}_{self.session_id}.json"
        self.data = {
            "session_id": self.session_id,
            "started_at": _now_iso(),
            "ended_at": None,
            "summary": {},
            "turns": [],
        }
        self.save()

    def new_turn(self, kind: str) -> dict:
        turn = {
            "turn": len(self.data["turns"]),
            "kind": kind,  # "opening" | "answer"
            "started_at": _now_iso(),
            "timeline_ms": {},
            "stt": {},
            "llm": {},
            "tts": {},
            "modal": {},
            "modal_server": None,
            "network": {},
            "client": None,
            "errors": [],
        }
        turn["_t0"] = time.perf_counter()
        self.data["turns"].append(turn)
        return turn

    @staticmethod
    def mark(turn: dict, name: str):
        turn["timeline_ms"][name] = ms_since(turn["_t0"])

    def turn(self, idx):
        try:
            return self.data["turns"][int(idx)]
        except (TypeError, ValueError, IndexError):
            return None

    def _summarize(self):
        """mean / min / max of every numeric metric across turns."""
        values: dict[str, list] = {}
        for t in self.data["turns"]:
            for section in self.SUMMARY_SECTIONS:
                if isinstance(t.get(section), dict):
                    for k, v in _flatten(t[section], section + ".").items():
                        values.setdefault(k, []).append(v)
        self.data["summary"] = {
            k: {"n": len(v), "mean": round(sum(v) / len(v), 1), "min": min(v), "max": max(v)}
            for k, v in values.items()
        }

    def save(self):
        self._summarize()
        data = dict(self.data)
        data["turns"] = [{k: v for k, v in t.items() if not k.startswith("_")} for t in self.data["turns"]]
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)

    def close(self):
        self.data["ended_at"] = _now_iso()
        self.save()
