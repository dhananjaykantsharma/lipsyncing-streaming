"""Next interview question via OpenAI chat completions."""
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, INTERVIEWER_SYSTEM_PROMPT

_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


async def next_question(history: list[dict]) -> str:
    """history: [{"role": "user"|"assistant", "content": "..."}], oldest first.
    Empty history -> the opening question."""
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not set (backend/.env)")

    messages = [{"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT}] + history
    resp = await _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()
