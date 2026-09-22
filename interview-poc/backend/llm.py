"""Next interview question via OpenAI chat completions — streamed and split
into sentences, so the caller can hand each sentence to TTS/Modal as soon as
it's complete instead of waiting for the whole (possibly multi-sentence)
response. OpenAI's TTS API cannot accept incrementally-growing text itself
(it needs one complete string per call), so sentence-at-a-time is the
practical unit of streaming across LLM -> TTS -> Modal."""
import re

from openai import AsyncOpenAI

from config import OPENAI_API_KEY, INTERVIEWER_SYSTEM_PROMPT

_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# a sentence ends at . ! or ? followed by whitespace (or end of text, handled
# by the final flush). Good enough for spoken interview questions; doesn't
# need to be bulletproof against abbreviations/decimals for this POC.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


async def stream_question(history: list[dict]):
    """history: [{"role": "user"|"assistant", "content": "..."}], oldest first.
    Empty history -> the opening question.

    Yields (sentence_text, is_final) as soon as each sentence is complete,
    is_final=True only on the last one. The caller is responsible for
    joining them back together for the conversation history."""
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not set (backend/.env)")

    messages = [{"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT}] + history
    stream = await _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=150,
        stream=True,
    )

    # We only find out a sentence was the LAST one when the stream ends (or
    # nothing follows it) — so the most recently completed sentence is held
    # back as `pending` until we know whether another one follows.
    buffer = ""
    pending = None
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        buffer += delta
        while True:
            m = _SENTENCE_BOUNDARY.search(buffer)
            if not m:
                break
            sentence, buffer = buffer[: m.start()], buffer[m.end():]
            sentence = sentence.strip()
            if not sentence:
                continue
            if pending is not None:
                yield pending, False
            pending = sentence

    tail = buffer.strip()
    if tail:
        if pending is not None:
            yield pending, False
        pending = tail

    if pending is not None:
        yield pending, True
