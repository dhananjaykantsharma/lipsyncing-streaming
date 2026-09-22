// Interview POC frontend.
//
// The backend streams the LLM's answer sentence-by-sentence (each sentence
// gets its own TTS + Modal lip-sync turn, pipelined with the next sentence's
// LLM/TTS work) rather than waiting for the whole (possibly multi-sentence)
// answer before doing anything. So a single logical "turn" is really a
// sequence of one or more of these cycles, each with its own `final` flag
// on the last message of the cycle:
//
// Protocol (matches backend/server.py):
//   we send    binary: candidate's recorded answer (one blob, webm/opus)
//   we receive {"type":"answer_text","text":...}          STT result (debug display)
//   -- one cycle per sentence, repeated until final:true --
//   we receive {"type":"question_text","text":...,"final":bool}
//   we receive binary [0x01][WAV bytes]                    TTS audio for that sentence
//   we receive {"type":"video_start","width","height","fps"}
//   we receive binary [0x02][<Modal's frame packet>]        = [1B flags][4B seq][H.264 Annex-B access unit]
//   we receive {"type":"done","final":bool}                 this sentence finished generating
//   -- /cycle --
//   we receive {"type":"error","message":...}
//
// The H.264 stream is Annex-B with SPS/PPS repeated on every keyframe (verified
// against the actual encoder output), so VideoDecoder.configure() is called
// WITHOUT a `description` — per the WebCodecs spec, no description means the
// decoder treats the bitstream as Annex-B and reads SPS/PPS from it directly.
//
// Playback pacing mirrors tools/ws_h264_view.py: the TTS audio is the master
// clock, frames are decoded/drawn to keep up with a fixed 25 fps schedule,
// held on the last frame if the next packet hasn't arrived yet, and skipped
// (but still decoded, since H.264 frames depend on their predecessors) if we
// ever fall behind. Sentences are queued and played strictly one at a time
// (see playQueue/drivePlayQueue) so a fast-arriving sentence 2 never starts
// playing over sentence 1 — only DATA can arrive ahead, not audio.

const FPS = 25;
const PREBUFFER_S = 0.5;
// specific to our server's encoder settings (1280x720, libx264 defaults) —
// see interview-poc's implementation notes if the avatar resolution changes.
const H264_CODEC = "avc1.64001F";

const TAG_TTS_AUDIO = 0x01;
const TAG_VIDEO_FRAME = 0x02;

const el = {
  status: document.getElementById("status"),
  question: document.getElementById("question"),
  answer: document.getElementById("answer"),
  startBtn: document.getElementById("startBtn"),
  recordBtn: document.getElementById("recordBtn"),
  canvas: document.getElementById("avatar"),
  loadingOverlay: document.getElementById("loadingOverlay"),
  loadingText: document.getElementById("loadingText"),
  processingSpinner: document.getElementById("processingSpinner"),
};
const ctx = el.canvas.getContext("2d");

function setStatus(text) {
  el.status.textContent = text;
}

// Shown from "Start Interview" until the first turn is actually ready to
// play (covers cold start, whatever it ends up costing) — and reused as the
// "generating your next question" cover in between turns too.
function showLoading(text) {
  el.loadingText.textContent = text;
  el.loadingOverlay.classList.remove("hidden");
}
function hideLoading() {
  el.loadingOverlay.classList.add("hidden");
}

// Small spinner next to the buttons: shown while the candidate's answer is
// being processed (STT -> LLM -> TTS -> Modal), hidden once the next
// question is ready to reveal.
function showProcessing() {
  el.processingSpinner.classList.remove("hidden");
}
function hideProcessing() {
  el.processingSpinner.classList.add("hidden");
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ---------------------------------------------------------------------
// Audio master clock: plays one WAV clip, exposes position() in seconds.
// ---------------------------------------------------------------------
class AudioClock {
  constructor(audioCtx) {
    this.ctx = audioCtx;
    this.t0 = null;
    this.duration = 0;
    this.source = null;
  }

  async start(arrayBuffer) {
    const buf = await this.ctx.decodeAudioData(arrayBuffer.slice(0));
    this.duration = buf.duration;
    this.source = this.ctx.createBufferSource();
    this.source.buffer = buf;
    this.source.connect(this.ctx.destination);
    // output latency: seconds between scheduling a sample and it being audible
    const latency = (this.ctx.outputLatency || this.ctx.baseLatency || 0);
    this.t0 = this.ctx.currentTime - latency;
    this.source.start();
  }

  position() {
    if (this.t0 === null) return null;
    return this.ctx.currentTime - this.t0;
  }

  stop() {
    if (this.source) {
      try { this.source.stop(); } catch (e) { /* already stopped */ }
    }
  }
}

// ---------------------------------------------------------------------
// One turn's video: receives H.264 packets, decodes + draws at 25 fps
// paced by an AudioClock.
// ---------------------------------------------------------------------
class TurnPlayer {
  constructor(audioCtx, onReveal) {
    this.audioCtx = audioCtx;
    this.onReveal = onReveal; // called right as audio+video actually start, together
    this.reset();
  }

  reset() {
    this.decoder = null;
    this.packets = []; // {data: Uint8Array, isKey: bool}
    this.nRecv = 0;
    this.decodedCount = 0;
    this.latestFrame = null;
    this.total = null;
    this.done = false;
    this.clock = null;
    this.ttsArrayBuffer = null;
    this._rafId = null;
    this.final = false; // is this the last sentence of the current answer?
  }

  onVideoStart(msg) {
    this.decoder = new VideoDecoder({
      output: (frame) => {
        if (this.latestFrame) this.latestFrame.close();
        this.latestFrame = frame;
      },
      error: (e) => console.error("VideoDecoder error:", e),
    });
    this.decoder.configure({
      codec: H264_CODEC,
      codedWidth: msg.width,
      codedHeight: msg.height,
      optimizeForLatency: true,
    });
  }

  onVideoFrame(payload) {
    // payload = [1B flags][4B seq, big-endian][H.264 access unit]
    const flags = payload[0];
    const isKey = (flags & 1) !== 0;
    const data = payload.slice(5);
    this.packets.push({ data, isKey });
    this.nRecv++;
  }

  onTtsAudio(arrayBuffer) {
    this.ttsArrayBuffer = arrayBuffer;
  }

  onDone(isFinal) {
    this.done = true;
    this.total = this.nRecv;
    this.final = isFinal;
  }

  // Waits for a small prebuffer, starts audio, then paces frames to it.
  // Resolves once the whole turn has finished playing.
  async play() {
    const prebufferFrames = Math.round(PREBUFFER_S * FPS);
    while (this.nRecv < prebufferFrames && !this.done) {
      await sleep(5);
    }
    if (this.ttsArrayBuffer === null) {
      // TTS audio should have arrived well before the first video packet;
      // wait a little longer just in case of reordering/slow network.
      for (let i = 0; i < 100 && this.ttsArrayBuffer === null; i++) await sleep(20);
    }

    this.clock = new AudioClock(this.audioCtx);
    await this.clock.start(this.ttsArrayBuffer);
    if (this.onReveal) this.onReveal();

    return new Promise((resolve) => {
      const tick = () => {
        const pos = this.clock.position();
        const target = pos >= 0 ? Math.floor(pos * FPS) : -1;
        const newest = this.nRecv - 1;
        const want = this.total === null ? target : Math.min(target, this.total - 1);
        const upto = Math.min(want, newest);

        while (this.decodedCount <= upto && this.packets.length > 0) {
          const pkt = this.packets.shift();
          this.decoder.decode(new EncodedVideoChunk({
            type: pkt.isKey ? "key" : "delta",
            timestamp: this.decodedCount * (1e6 / FPS),
            data: pkt.data,
          }));
          this.decodedCount++;
        }

        if (this.latestFrame) {
          ctx.drawImage(this.latestFrame, 0, 0, el.canvas.width, el.canvas.height);
        }

        const finishedAt = Math.max((this.total || 0) / FPS, this.clock.duration) + 0.15;
        if (this.done && this.total !== null && pos >= finishedAt) {
          this.clock.stop();
          if (this.decoder && this.decoder.state !== "closed") this.decoder.close();
          resolve();
          return;
        }
        this._rafId = requestAnimationFrame(tick);
      };
      this._rafId = requestAnimationFrame(tick);
    });
  }
}

// ---------------------------------------------------------------------
// Mic recording (push-to-talk).
// ---------------------------------------------------------------------
class Recorder {
  constructor() {
    this.mediaRecorder = null;
    this.chunks = [];
    this.stream = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: "audio/webm;codecs=opus" });
    this.chunks = [];
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
    this.mediaRecorder.start();
  }

  stop() {
    return new Promise((resolve) => {
      this.mediaRecorder.onstop = () => {
        this.stream.getTracks().forEach((t) => t.stop());
        resolve(new Blob(this.chunks, { type: "audio/webm" }));
      };
      this.mediaRecorder.stop();
    });
  }
}

// ---------------------------------------------------------------------
// Session: drives the WebSocket + UI state machine.
// ---------------------------------------------------------------------
let ws = null;
let audioCtx = null;
let recorder = null;
let recording = false;

// The backend now streams the answer sentence-by-sentence (LLM -> TTS ->
// Modal pipelined per sentence), so a single logical "turn" can involve
// several question_text/video_start/done cycles in a row. We keep two
// separate notions of "current":
//   receivingPlayer — whichever sentence's data is arriving from the
//                      backend right now (can run ahead of playback)
//   playQueue        — sentences waiting to be PLAYED, strictly in order,
//                      one at a time, so sentence 2's audio never starts
//                      before sentence 1 has finished (no overlap)
let receivingPlayer = null;
let playQueue = [];
let playing = false;
let answerInProgress = false;

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/session`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("Connected. Waiting for the first question...");

  ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      await handleControl(msg);
    } else {
      const bytes = new Uint8Array(event.data);
      const tag = bytes[0];
      const payload = bytes.slice(1);
      if (tag === TAG_TTS_AUDIO) {
        receivingPlayer && receivingPlayer.onTtsAudio(payload.buffer);
      } else if (tag === TAG_VIDEO_FRAME) {
        receivingPlayer && receivingPlayer.onVideoFrame(payload);
      }
    }
  };

  ws.onclose = () => setStatus("Disconnected.");
  ws.onerror = () => setStatus("Connection error.");
}

// Plays queued sentences strictly one after another. Only one of these
// loops ever runs at a time (guarded by `playing`).
async function drivePlayQueue() {
  while (playQueue.length > 0) {
    const player = playQueue.shift();
    await player.play();
    if (player.final) {
      answerInProgress = false;
      onTurnFinished();
    }
  }
  playing = false;
}

async function handleControl(msg) {
  switch (msg.type) {
    case "answer_text":
      el.answer.textContent = `You said: "${msg.text}"`;
      break;

    case "question_text": {
      // don't show the text yet — held back until audio+video are actually
      // about to play, so everything appears together (TurnPlayer.onReveal)
      if (!answerInProgress) {
        el.question.textContent = "";
        answerInProgress = true;
      }
      el.answer.textContent = "";
      el.recordBtn.disabled = true;
      const sentenceText = msg.text;
      receivingPlayer = new TurnPlayer(audioCtx, () => {
        el.question.textContent += (el.question.textContent ? " " : "") + sentenceText;
        hideLoading();
        hideProcessing();
        setStatus("Interviewer is speaking...");
      });
      playQueue.push(receivingPlayer);
      if (!playing) {
        playing = true;
        drivePlayQueue();
      }
      break;
    }

    case "video_start":
      receivingPlayer.onVideoStart(msg);
      break;

    case "done":
      receivingPlayer.onDone(msg.final);
      break;

    case "error":
      setStatus(`Error: ${msg.message}`);
      hideLoading();
      hideProcessing();
      answerInProgress = false;
      el.recordBtn.disabled = false;
      break;
  }
}

function onTurnFinished() {
  setStatus("Your turn. Click 'Start Answer' when ready.");
  el.recordBtn.disabled = false;
}

async function onStart() {
  el.startBtn.disabled = true;
  el.startBtn.style.display = "none";
  showLoading("Creating your interview session...");
  audioCtx = new AudioContext();
  recorder = new Recorder();
  connect();
}

async function onRecordClick() {
  if (!recording) {
    try {
      await recorder.start();
    } catch (e) {
      setStatus(`Microphone error: ${e}`);
      return;
    }
    recording = true;
    el.recordBtn.textContent = "Done Answering";
    el.recordBtn.classList.add("recording");
    setStatus("Recording your answer...");
  } else {
    recording = false;
    el.recordBtn.textContent = "Start Answer";
    el.recordBtn.classList.remove("recording");
    el.recordBtn.disabled = true;
    setStatus("Processing your answer...");
    showProcessing();
    const blob = await recorder.stop();
    console.log(`recorded answer: ${blob.size} bytes, type=${blob.type}`);
    const buf = await blob.arrayBuffer();
    ws.send(buf);
  }
}

el.startBtn.addEventListener("click", onStart);
el.recordBtn.addEventListener("click", onRecordClick);
