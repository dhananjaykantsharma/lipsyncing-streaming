"""Process-wide HTTP clients for Deepgram (STT) and OpenAI (LLM + TTS), so the
TCP + TLS handshake happens once and every later call reuses the connection.

Two things would otherwise force a fresh handshake on every turn:
  - a new client per call (stt.py used `async with httpx.AsyncClient()`)
  - httpx's default keepalive_expiry of 5 s: a turn is 1-2 minutes apart,
    so even a shared pool would have dropped the idle connection by then.
So: one client per vendor, a long keepalive_expiry, and keep_warm() pings
both vendors periodically so the vendor side doesn't close it either.

Per-call trace: wrap a call in `with traced() as trace:` and trace.phases()
says whether that call opened a new connection (tcp_connect_ms / tls_handshake_ms
are null when the connection was reused)."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

import httpx
import httpx2  # the OpenAI SDK's own fork of httpx; its http_client must be one of these
from openai import AsyncOpenAI

from config import DEEPGRAM_API_KEY, OPENAI_API_KEY
from latency_log import HttpTrace

KEEPALIVE_EXPIRY_S = 600
# how often keep_warm() touches each vendor while a session is open
WARM_INTERVAL_S = 30

_current_trace: ContextVar[HttpTrace | None] = ContextVar("_current_trace", default=None)


@contextmanager
def traced():
    trace = HttpTrace()
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        _current_trace.reset(token)


def _tracing_transport(mod):
    """AsyncHTTPTransport (of httpx or httpx2) that attaches the current
    traced() callback to each request — the OpenAI SDK gives no per-request
    hook for httpx `extensions`, so it's injected at the transport."""

    class TracingTransport(mod.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            trace = _current_trace.get()
            if trace is not None:
                request.extensions["trace"] = trace
            return await super().handle_async_request(request)

    return TracingTransport(
        limits=mod.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=KEEPALIVE_EXPIRY_S)
    )


deepgram = httpx.AsyncClient(
    base_url="https://api.deepgram.com",
    headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
    timeout=60,
    transport=_tracing_transport(httpx),
)

openai_client = (
    AsyncOpenAI(api_key=OPENAI_API_KEY, http_client=httpx2.AsyncClient(transport=_tracing_transport(httpx2)))
    if OPENAI_API_KEY
    else None
)


async def warm_up():
    """Cheap authenticated GETs that open (or refresh) one pooled connection
    per vendor. The response itself doesn't matter — any status warms it."""

    async def dg():
        if DEEPGRAM_API_KEY:
            await deepgram.get("/v1/projects")

    async def oa():
        if openai_client is not None:
            await openai_client.models.with_raw_response.list()

    results = await asyncio.gather(dg(), oa(), return_exceptions=True)
    for name, r in zip(("deepgram", "openai"), results):
        if isinstance(r, Exception):
            print(f"[http] warm-up {name} failed: {r!r}")


async def keep_warm():
    """Runs for the life of a session (Session.start / Session.close)."""
    while True:
        await warm_up()
        await asyncio.sleep(WARM_INTERVAL_S)


async def aclose():
    await deepgram.aclose()
    if openai_client is not None:
        await openai_client.close()
