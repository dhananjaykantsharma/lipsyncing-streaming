import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
MODAL_WS_URL = os.environ.get("MODAL_WS_URL", "")

INTERVIEWER_SYSTEM_PROMPT = (
    "You are a friendly, professional interviewer conducting a spoken job interview. "
    "Ask exactly one question at a time, in plain conversational language suitable for "
    "text-to-speech (no markdown, no bullet points, no headings). Keep each question to "
    "1-2 sentences. Start with a warm opening question. Ask natural follow-up questions "
    "based on what the candidate says."
)
