"""Next interview question via OpenAI chat completions."""
from config import INTERVIEWER_SYSTEM_PROMPT
from http_clients import openai_client as _client, traced
from latency_log import header_ms


async def next_question(history: list[dict], metrics: dict | None = None) -> str:
    """history: [{"role": "user"|"assistant", "content": "..."}], oldest first.
    Empty history -> the opening question. `metrics`, if given, gets OpenAI's
    own processing time (openai-processing-ms header) and token counts."""
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not set (backend/.env)")

    messages = [{"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT}] + history
    with traced() as trace:
        raw = await _client.chat.completions.with_raw_response.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
    resp = raw.parse()

    if metrics is not None:
        metrics["net"] = trace.phases()
        metrics["server_processing_ms"] = header_ms(raw.headers, "openai-processing-ms")
        metrics["model"] = resp.model
        if resp.usage:
            metrics["prompt_tokens"] = resp.usage.prompt_tokens
            metrics["completion_tokens"] = resp.usage.completion_tokens

    return resp.choices[0].message.content.strip()
