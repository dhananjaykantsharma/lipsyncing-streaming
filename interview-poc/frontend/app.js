// Interview POC frontend.
//
// Protocol (matches backend/server.py):
//   we send    {"type":"answer_start"}, then binary WebM/Opus chunks every
//              ANSWER_TIMESLICE_MS while the candidate speaks, then
//              {"type":"answer_end","chunks","bytes"} on "Done" — so the
//              answer is already uploaded by the time they click Done
//   we receive {"type":"answer_text","text":...}          STT result (debug display)
//   we receive {"type":"question_text","text":...}        next question
//   we receive binary [0x01][MP3 bytes]                    TTS audio for that question
//   we receive {"type":"video_start","width","height","fps"}
//   we receive binary [0x02][<Modal's frame packet>]        = [1B flags][4B seq][H.264 Annex-B access unit]
//   we receive {"type":"done"}                              turn finished
//   we receive {"type":"error","message":...}
//   latency reporting (backend writes it to backend/logs/session_*.json):
//   we send    {"type":"ping","id"} -> we receive {"type":"pong","id"}
//   we send    {"type":"client_metrics","turn","metrics":{...}} once a turn finishes playing
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
// ever fall behind.

const FPS = 25;
const PREBUFFER_S = 0.5;
// specific to our server's encoder settings (1280x720, libx264 defaults) —
// see interview-poc's implementation notes if the avatar resolution changes.
const H264_CODEC = "avc1.64001F";

// Opus bitrate for the candidate's answer. Chrome's default is ~128 kbps; at
// 32 kbps the upload is ~4x smaller and Deepgram's transcript was identical
// (tools/stt_bitrate_test.py: 0% WER vs 128k, confidence 0.999).
const ANSWER_BITRATE_BPS = 32000;
// how often the recorder hands over a chunk to upload while recording
const ANSWER_TIMESLICE_MS = 250;

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
// Per-turn latency metrics. t0 = the click that started the turn ("Done
// Answering", or "Start Interview" for the opening question); every *_ms
// mark is milliseconds since then. Sent to the backend when the turn ends.
// ---------------------------------------------------------------------
function newTurnMetrics(extra) {
  return { t0: performance.now(), values: { ...extra } };
}
function markT(m, name, overwrite = false) {
  if (!m || (!overwrite && m.values[name] !== undefined)) return;
  m.values[name] = Math.round((performance.now() - m.t0) * 10) / 10;
}

// ---------------------------------------------------------------------
// Audio master clock: plays one TTS clip (MP3), exposes position() in seconds.
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
    this.metrics = null; // set by handleControl("question_text")
    this.turnId = null;
    this.stallMs = 0;
    this.stallEvents = 0;
  }

  onVideoStart(msg) {
    this.decoder = new VideoDecoder({
      output: (frame) => {
        markT(this.metrics, "first_frame_decoded_ms");
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
    markT(this.metrics, "first_packet_ms");
    markT(this.metrics, "last_packet_ms", true);
  }

  onTtsAudio(arrayBuffer) {
    markT(this.metrics, "tts_audio_ms");
    this.ttsArrayBuffer = arrayBuffer;
  }

  onDone() {
    markT(this.metrics, "done_ms");
    this.done = true;
    this.total = this.nRecv;
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
    markT(this.metrics, "prebuffer_ready_ms");

    this.clock = new AudioClock(this.audioCtx);
    const tDecode = performance.now();
    await this.clock.start(this.ttsArrayBuffer);
    if (this.metrics) {
      this.metrics.values.audio_decode_ms = Math.round((performance.now() - tDecode) * 10) / 10;
      this.metrics.values.audio_duration_s = Math.round(this.clock.duration * 1000) / 1000;
    }
    // THE headline number: click -> avatar audibly/visibly speaking
    markT(this.metrics, "avatar_speaking_ms");
    if (this.onReveal) this.onReveal();

    let lastTick = performance.now();
    let starving = false;
    return new Promise((resolve) => {
      const tick = () => {
        const now = performance.now();
        const pos = this.clock.position();
        const target = pos >= 0 ? Math.floor(pos * FPS) : -1;
        const newest = this.nRecv - 1;
        const want = this.total === null ? target : Math.min(target, this.total - 1);
        const upto = Math.min(want, newest);

        // stall = the audio clock wants a frame that hasn't arrived yet, so
        // the avatar freezes on its last frame while the audio keeps going
        if (upto < want) {
          if (!starving) this.stallEvents++;
          this.stallMs += now - lastTick;
        }
        starving = upto < want;
        lastTick = now;

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
          if (this.metrics) {
            markT(this.metrics, "playback_end_ms");
            Object.assign(this.metrics.values, {
              packets_received: this.nRecv,
              frames_decoded: this.decodedCount,
              video_duration_s: Math.round((this.total / FPS) * 1000) / 1000,
              stall_ms: Math.round(this.stallMs),
              stall_events: this.stallEvents,
            });
          }
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
    this.stream = null;
    this.nChunks = 0;
    this.nBytes = 0;
  }

  // onChunk(blob) is called every ANSWER_TIMESLICE_MS while recording, and
  // once more with the tail when stop() is called
  async start(onChunk) {
    // mono: the bitrate then goes to one voice channel (as in the bitrate test)
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    this.mediaRecorder = new MediaRecorder(this.stream, {
      mimeType: "audio/webm;codecs=opus",
      audioBitsPerSecond: ANSWER_BITRATE_BPS,
    });
    this.nChunks = 0;
    this.nBytes = 0;
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size === 0) return;
      this.nChunks++;
      this.nBytes += e.data.size;
      onChunk(e.data);
    };
    this.mediaRecorder.start(ANSWER_TIMESLICE_MS);
  }

  // resolves after the final chunk has been handed to onChunk
  stop() {
    return new Promise((resolve) => {
      this.mediaRecorder.onstop = () => {
        this.stream.getTracks().forEach((t) => t.stop());
        resolve();
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
let currentPlayer = null;
let recorder = null;
let recording = false;
let pendingMetrics = null; // the turn being waited on (created at the click)
let recordStartedAt = null;
const pendingPongs = new Map();

// browser <-> backend websocket round trip (sent while the backend is idle)
function measureRtt() {
  return new Promise((resolve) => {
    const id = Math.random().toString(36).slice(2);
    const t = performance.now();
    pendingPongs.set(id, () => resolve(Math.round((performance.now() - t) * 10) / 10));
    ws.send(JSON.stringify({ type: "ping", id }));
    setTimeout(() => { if (pendingPongs.delete(id)) resolve(null); }, 5000);
  });
}

async function reportMetrics(player) {
  if (!player.metrics || player.turnId === null) return;
  const values = { ...player.metrics.values, ws_rtt_ms: await measureRtt() };
  ws.send(JSON.stringify({ type: "client_metrics", turn: player.turnId, metrics: values }));
  console.log(`turn ${player.turnId} latency`, values);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/session`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    markT(pendingMetrics, "ws_open_ms");
    setStatus("Connected. Waiting for the first question...");
  };

  ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      await handleControl(msg);
    } else {
      const bytes = new Uint8Array(event.data);
      const tag = bytes[0];
      const payload = bytes.slice(1);
      if (tag === TAG_TTS_AUDIO) {
        currentPlayer && currentPlayer.onTtsAudio(payload.buffer);
      } else if (tag === TAG_VIDEO_FRAME) {
        currentPlayer && currentPlayer.onVideoFrame(payload);
      }
    }
  };

  ws.onclose = () => setStatus("Disconnected.");
  ws.onerror = () => setStatus("Connection error.");
}

async function handleControl(msg) {
  switch (msg.type) {
    case "answer_text":
      if (pendingMetrics) {
        markT(pendingMetrics, "answer_text_ms");
        // from the moment the blob was handed to the socket (excludes recorder finalize)
        pendingMetrics.values.answer_text_after_send_ms =
          pendingMetrics.values.answer_text_ms - pendingMetrics.values.answer_sent_ms;
      }
      el.answer.textContent = `You said: "${msg.text}"`;
      break;

    case "question_text":
      // don't show the text yet — held back until audio+video are actually
      // about to play, so everything appears together (TurnPlayer.onReveal)
      el.answer.textContent = "";
      el.recordBtn.disabled = true;
      markT(pendingMetrics, "question_text_ms");
      currentPlayer = new TurnPlayer(audioCtx, () => {
        el.question.textContent = msg.text;
        hideLoading();
        hideProcessing();
        setStatus("Interviewer is speaking...");
      });
      currentPlayer.metrics = pendingMetrics;
      currentPlayer.turnId = msg.turn ?? null;
      pendingMetrics = null;
      break;

    case "video_start":
      markT(currentPlayer.metrics, "video_start_ms");
      currentPlayer.onVideoStart(msg);
      // don't await here: play() resolves only once the turn's video has
      // fully finished, which is exactly when we want to unlock the mic
      {
        const player = currentPlayer;
        player.play().then(() => {
          onTurnFinished();
          reportMetrics(player);
        });
      }
      break;

    case "done":
      currentPlayer.onDone();
      break;

    case "pong": {
      const done = pendingPongs.get(msg.id);
      if (done) {
        pendingPongs.delete(msg.id);
        done();
      }
      break;
    }

    case "error":
      setStatus(`Error: ${msg.message}`);
      hideLoading();
      hideProcessing();
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
  pendingMetrics = newTurnMetrics({ kind: "opening" });
  showLoading("Creating your interview session...");
  audioCtx = new AudioContext();
  recorder = new Recorder();
  connect();
}

async function onRecordClick() {
  if (!recording) {
    ws.send(JSON.stringify({ type: "answer_start" }));
    try {
      // each chunk goes straight onto the socket (WebSocket.send keeps order)
      await recorder.start((blob) => ws.send(blob));
    } catch (e) {
      ws.send(JSON.stringify({ type: "answer_cancel" }));
      setStatus(`Microphone error: ${e}`);
      return;
    }
    recording = true;
    recordStartedAt = performance.now();
    el.recordBtn.textContent = "Done Answering";
    el.recordBtn.classList.add("recording");
    setStatus("Recording your answer...");
  } else {
    recording = false;
    const metrics = newTurnMetrics({
      kind: "answer",
      recording_s: Math.round(performance.now() - recordStartedAt) / 1000,
      // bytes recorded but not yet sent when Done was clicked (0 => upload kept up)
      ws_backlog_at_done_bytes: ws.bufferedAmount,
    });
    el.recordBtn.textContent = "Start Answer";
    el.recordBtn.classList.remove("recording");
    el.recordBtn.disabled = true;
    setStatus("Processing your answer...");
    showProcessing();
    await recorder.stop(); // the tail chunk is on the socket after this
    const { nChunks, nBytes } = recorder;
    console.log(`recorded answer: ${nBytes} bytes in ${nChunks} chunks (streamed while recording)`);
    metrics.values.answer_blob_bytes = nBytes;
    metrics.values.answer_chunks = nChunks;
    // actual bitrate the browser produced (should be ~32 when the setting is honoured)
    metrics.values.answer_kbps = Math.round((nBytes * 8) / metrics.values.recording_s / 1000);
    pendingMetrics = metrics;
    ws.send(JSON.stringify({ type: "answer_end", chunks: nChunks, bytes: nBytes }));
    markT(metrics, "answer_sent_ms"); // click -> answer_end sent (tail chunk + recorder finalize)
  }
}

el.startBtn.addEventListener("click", onStart);
el.recordBtn.addEventListener("click", onRecordClick);
