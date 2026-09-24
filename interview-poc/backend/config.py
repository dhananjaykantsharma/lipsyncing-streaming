import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
MODAL_WS_URL = os.environ.get("MODAL_WS_URL", "")
# close the Modal websocket after this long without a turn, so an idle
# session (e.g. a tab left open) doesn't keep the GPU container running
MODAL_IDLE_CLOSE_S = float(os.environ.get("MODAL_IDLE_CLOSE_S", "180"))

INTERVIEWER_SYSTEM_PROMPT = (
    "You are a friendly, professional interviewer conducting a spoken job interview. "
    "Ask exactly one question at a time, in plain conversational language suitable for "
    "text-to-speech (no markdown, no bullet points, no headings). Keep each question to "
    "1-2 sentences. Start with a warm opening question. Ask natural follow-up questions "
    "based on what the candidate says."
)
