# AI Interview Avatar — MuseTalk on Modal + Realtime Interview App

A GPU-backed lip-sync avatar pipeline, plus a proof-of-concept spoken-interview
application built on top of it. An interviewer avatar asks questions out loud
with synced lip movement, listens to the candidate's spoken answer, and asks
a natural follow-up — end to end, in near real time.

The project has two parts that live in the same repo but run as separate
processes:

1. **The lip-sync service** (root of this repo) — runs [MuseTalk](https://github.com/TMElyralab/MuseTalk)
   on a [Modal](https://modal.com) A100 GPU. Given an audio clip and a fixed
   avatar video, it streams back an H.264 video of the avatar speaking that
   audio, frame by frame, over a WebSocket.
2. **The interview POC application** (`interview-poc/`) — a separate Python
   backend (STT → LLM → TTS → the lip-sync service) and a plain HTML/CSS/JS
   frontend that runs the actual interview in a browser: it records the
   candidate's answer, transcribes it, generates the next question, and
   plays back the avatar asking it — synced audio, video, and text.

---

## Architecture

```
                         ┌────────────────────────────────────────┐
                         │  Modal.com — A100 40GB GPU container    │
                         │  MuseTalk lip-sync (this repo's root)   │
                         │  /ws-stream  (audio in -> H.264 frames  │
                         │               out, streamed)            │
                         └───────────────▲──────────────┬──────────┘
                                         │ audio         │ H.264 frames
                                         │               │
┌────────────────┐   answer audio  ┌────┴───────────────▼────┐   question text
│                │ ───────────────►│                           │   + TTS audio
│  Browser       │                 │  interview-poc/backend    │   + video frames
│  (candidate's  │◄─────────────── │  (FastAPI, own WebSocket) │
│  screen)       │  question text  │  STT: Deepgram             │
│  WebCodecs +   │  + audio        │  LLM: OpenAI (streamed)    │
│  Web Audio API │  + avatar video │  TTS: OpenAI (streamed)    │
└────────────────┘                 └────────────────────────────┘
```

The interview backend never talks to the GPU model directly — it's a thin
orchestrator that calls Deepgram/OpenAI and relays whatever the Modal service
streams back, over its own WebSocket, to the browser.

### One turn, step by step

1. Candidate clicks "Start Answer", speaks, clicks "Done Answering". The
   browser sends the recorded clip to the backend over its own WebSocket.
2. Backend transcribes it (Deepgram), and streams the next question from the
   LLM (OpenAI, `stream=True`) — **split into sentences as they complete**,
   not the whole answer at once.
3. For each sentence: TTS starts generating audio for it immediately, and the
   *next* sentence's LLM/TTS work runs concurrently in the background — so by
   the time sentence 1 finishes playing, sentence 2 is usually already ready.
4. Each sentence's audio goes to the Modal lip-sync service, which streams
   back H.264 video frames as they're generated (faster than real time on a
   warm GPU).
5. The backend relays question text, TTS audio, and video frames to the
   browser as they arrive. The browser reveals text/audio/video **together**
   (not text-first), and paces playback to a fixed 25fps clock driven by the
   audio, so multiple sentences play back-to-back with no overlap or gaps.

See [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) for the full build history,
every bug hit along the way, and detailed latency measurements (in Hinglish).

---

## Project structure

```
avatar gpu test/
├── image.py              Modal container image definition (torch, MuseTalk, mmcv, ffmpeg, ...)
├── volume.py              Modal Volume (persistent storage) config
├── preprocess.py          One-off: download model weights + preprocess the avatar video
├── server.py              The GPU service: loads MuseTalk once, serves /ws and /ws-stream
├── blendutil.py           Fast (bit-exact) replacement for MuseTalk's frame-blending step
├── h264util.py            H.264 encode/decode helpers (PyAV + libx264)
├── inspect_musetalk.py    Small helper scripts used to inspect MuseTalk's own source/config
├── test.py                Minimal GPU sanity check
├── sample.mp4             The avatar's source video
├── tools/
│   └── ws_h264_view.py    Standalone test client/player (audio + 25fps video, for debugging)
│
└── interview-poc/
    ├── backend/
    │   ├── server.py       FastAPI app: browser WebSocket (/session) + static file server
    │   ├── session.py      Orchestrates STT -> LLM -> TTS -> Modal relay per turn
    │   ├── stt.py           Deepgram (prerecorded/batch transcription)
    │   ├── llm.py           OpenAI chat, streamed + split into sentences
    │   ├── tts.py            OpenAI TTS, streamed
    │   ├── modal_client.py   Persistent WebSocket connection to the Modal lip-sync service
    │   ├── config.py         Loads .env
    │   ├── requirements.txt
    │   └── .env.example      Copy to .env and fill in API keys
    └── frontend/
        ├── index.html
        ├── style.css
        └── app.js            WebSocket client, mic recording, WebCodecs H.264 decode,
                               Web Audio playback clock, all UI state
```

---

## Prerequisites

- A [Modal](https://modal.com) account, with the CLI installed and authenticated.
- An [OpenAI](https://platform.openai.com) API key (used for both the LLM and TTS).
- A [Deepgram](https://deepgram.com) API key (used for speech-to-text).
- Python 3.10, `ffmpeg`, and (for audio playback in test tools) `libportaudio2`.
- A Chromium-based browser (Chrome/Edge) for the interview frontend — it
  relies on the [WebCodecs API](https://www.w3.org/TR/webcodecs/), which has
  limited support elsewhere.

---

## Setup

### 1. The lip-sync service (repo root)

```bash
cd "avatar gpu test"
python3 -m venv venv && source venv/bin/activate
pip install modal websockets av opencv-python numpy sounddevice

modal token new                                    # authenticate with Modal

# upload the avatar video, then download weights + preprocess it (one-off)
modal volume put musetalk-data sample.mp4 avatars/avatar.mp4
modal run preprocess.py
```

### 2. The interview backend

```bash
cd interview-poc/backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in OPENAI_API_KEY, DEEPGRAM_API_KEY
# MODAL_WS_URL should already point at the `modal serve` URL below
```

---

## Running it

Two long-running processes, each in its own terminal:

```bash
# Terminal 1 — the GPU lip-sync service
cd "avatar gpu test"
source venv/bin/activate
modal serve server.py

# Terminal 2 — the interview backend (also serves the frontend)
cd "avatar gpu test/interview-poc/backend"
source venv/bin/activate
python3 -m uvicorn server:app --port 8091
```

Then open **http://localhost:8091** in Chrome/Edge, click "Start Interview",
grant microphone access, and go.

`modal serve` bills GPU time while it's running — stop it (`Ctrl-C`, or
`pkill -INT -f "[v]env/bin/modal serve server.py"`) when you're done, and
confirm with `modal app list` that no app is left running.

---

## Status

This is a proof of concept, not a production deployment. What works:

- End-to-end pipeline: recorded answer → transcript → next question →
  synced avatar video+audio, sentence-by-sentence streaming throughout.
- Real-time-capable generation: the GPU service produces frames faster than
  25fps once warm, so playback never stalls waiting on it.

What's not done yet:

- **Cold start**: the first request after the GPU container spins up takes
  ~30-40s (model loading). Not yet mitigated with a keep-warm container.
- **No auth**: the Modal WebSocket endpoint and the interview backend are
  both unauthenticated — fine for local testing, not for exposing publicly.
- **Region**: the Modal container currently runs in US West; for candidates
  in India this adds meaningful network latency to the first frame of each
  turn (generation itself is not the bottleneck — see `PROJECT_SUMMARY.md`
  §5 for the actual measurements).
- **`modal deploy`**: everything has been run via `modal serve` (dev mode)
  so far, not deployed as a permanent service.

For the detailed build log — every bug hit, every measurement taken, and the
reasoning behind each design decision — see
[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).
