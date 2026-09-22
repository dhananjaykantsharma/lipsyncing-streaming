"""Live player for the H.264 streaming endpoint (/ws-stream): fixed 25 fps, with audio.

  pip install websockets av opencv-python sounddevice numpy
  python tools/ws_h264_view.py <audio.wav> [--url wss://.../ws-stream] [--runs 1]
         [--prebuffer 0.5] [--av-offset-ms 0] [--overlay] [--no-display] [--no-audio] [--mute]

The server generates frames faster than real time, so they pile up as small H.264
packets and are drawn on a fixed 25 fps clock instead of "as soon as they arrive".
The audio is the master clock: each tick the player draws the frame that matches the
audio position, holds the last frame if the next one has not arrived yet, and skips
frames if it ever falls behind. Press q / Esc in the video window to quit.

--prebuffer   seconds of video to collect before playback starts (jitter cushion)
--av-offset-ms  fine-tune lip-sync: positive = video later, negative = video earlier
--mute        run the whole audio path but play silence (for testing)
--no-audio    no sound device at all, plain 25 fps clock
--no-display  no window (prints the timing numbers only)"""
import argparse
import asyncio
import json
import os
import struct
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import websockets  # noqa: E402

from h264util import H264Decoder  # noqa: E402

DEFAULT_URL = "wss://dhananjay-sharma--musetalk-lipsync-musetalkinference-web-dev.modal.run/ws-stream"
FPS = 25
WINDOW = "MuseTalk H.264 stream"


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def load_audio(path, rate=48000):
    """Decode any audio file to mono int16 at `rate` (PyAV)."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    chunks = []
    with av.open(path) as c:
        for frame in c.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(chunks), rate


class SilentClock:
    """No sound device: a plain clock that starts when start() is called."""

    def __init__(self, offset_s=0.0):
        self.t_first, self.offset = None, offset_s

    def start(self):
        self.t_first = time.perf_counter()

    def position(self):
        return None if self.t_first is None else time.perf_counter() - self.t_first - self.offset

    def stop(self):
        pass


class AudioClock:
    """Plays the audio and exposes `position()` = seconds of audio audible right now."""

    def __init__(self, samples, rate, mute=False, offset_s=0.0):
        import sounddevice as sd

        self.sd, self.samples, self.rate = sd, samples, rate
        self.mute, self.offset = mute, offset_s
        self.pos, self.t_first, self.latency, self.stream = 0, None, 0.0, None

    def _callback(self, outdata, frames, time_info, status):
        if self.t_first is None:
            # the first block we hand over becomes audible `latency` seconds from now
            lat = time_info.outputBufferDacTime - time_info.currentTime
            self.latency = lat if 0 <= lat < 1.0 else float(self.stream.latency)
            self.t_first = time.perf_counter()
        n = max(0, min(frames, len(self.samples) - self.pos))
        outdata[:] = 0
        if n and not self.mute:
            outdata[:n, 0] = self.samples[self.pos:self.pos + n]
        self.pos += n

    def start(self):
        self.stream = self.sd.OutputStream(samplerate=self.rate, channels=1, dtype="int16",
                                           callback=self._callback)
        self.stream.start()

    def position(self):
        if self.t_first is None:
            return None
        return time.perf_counter() - self.t_first - self.latency - self.offset

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


class State:
    def __init__(self):
        self.packets = deque()  # small H.264 packets waiting to be decoded + drawn
        self.n_recv = 0
        self.done = False
        self.total = None
        self.stats = None
        self.error = None
        self.arrivals = []
        self.wire_bytes = 0


async def receiver(ws, st, t_req):
    try:
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "done":
                    st.stats = data.get("stats")
                    return
                continue
            struct.unpack(">BI", msg[:5])  # flags, frame no. (frames arrive in order)
            st.packets.append(msg[5:])
            st.n_recv += 1
            st.wire_bytes += len(msg)
            st.arrivals.append(time.perf_counter() - t_req)
    except Exception as e:  # connection dropped etc.
        st.error = repr(e)
    finally:
        st.total = st.n_recv
        st.done = True


async def player(st, clock, dec, cv2, prebuffer_frames, audio_dur, overlay):
    while not (st.n_recv >= prebuffer_frames or st.done):
        await asyncio.sleep(0.005)
    if st.n_recv == 0:
        return None

    clock.start()
    while clock.position() is None:
        await asyncio.sleep(0.002)
    t_start = time.perf_counter()

    decoded, last_shown, latest = 0, -1, None
    shown = skipped = late = 0
    hold_s, max_depth = 0.0, 0
    errs, draws = [], []
    last_loop, quit_ = time.perf_counter(), False

    while True:
        now = time.perf_counter()
        dt, last_loop = now - last_loop, now
        pos = clock.position()
        target = int(pos * FPS) if pos >= 0 else -1
        newest = st.n_recv - 1
        want = target if st.total is None else min(target, st.total - 1)

        # decode forward (the H.264 chain must be decoded in order, even for frames we skip)
        upto = min(want, newest)
        while decoded <= upto:
            for img in dec.decode(st.packets.popleft()):
                latest = img
            decoded += 1
        max_depth = max(max_depth, len(st.packets))

        idx = decoded - 1
        if latest is not None and idx > last_shown:
            err = pos - idx / FPS  # how long after its due time this frame is drawn
            errs.append(err)
            late += err > 1.5 / FPS
            skipped += idx - last_shown - 1
            last_shown = idx
            shown += 1
            draws.append(time.perf_counter())
            if cv2 is not None:
                if overlay:
                    cv2.putText(latest, f"t={pos:6.2f}s  buffered={len(st.packets):4d} pkts", (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(WINDOW, latest)

        if target >= 0 and target > newest and not st.done:
            hold_s += dt  # the frame that is due has not arrived yet: showing the previous one

        if cv2 is not None:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                quit_ = True
                break
        if st.done and st.total is not None and pos >= max(st.total / FPS, audio_dur) + 0.15:
            break
        await asyncio.sleep(0.002)

    clock.stop()
    return dict(t_start=t_start, shown=shown, skipped=skipped, late=late, hold_s=hold_s,
                errs=errs, draws=draws, max_depth=max_depth, last_idx=last_shown, quit=quit_)


async def one_run(ws, audio_bytes, args, cv2, label):
    st = State()
    if args.no_audio:
        clock, audio_dur = SilentClock(args.av_offset_ms / 1000), 0.0
    else:
        samples, rate = load_audio(args.audio)
        clock = AudioClock(samples, rate, mute=args.mute, offset_s=args.av_offset_ms / 1000)
        audio_dur = len(samples) / rate

    t_req = time.perf_counter()
    await ws.send(audio_bytes)
    if cv2 is not None:
        # the first imshow() builds the window (~0.4s): do it now, while we wait for the
        # first packet, not after the audio clock has started (that would cost skipped frames)
        cv2.imshow(WINDOW, np.zeros((720, 1280, 3), np.uint8))
        cv2.waitKey(30)
    rtask = asyncio.create_task(receiver(ws, st, t_req))
    try:
        res = await player(st, clock, H264Decoder(), cv2, int(args.prebuffer * FPS), audio_dur, args.overlay)
    finally:
        if not rtask.done():
            rtask.cancel()
        await asyncio.gather(rtask, return_exceptions=True)

    print(f"\n===== {label} =====")
    n = st.n_recv
    if n == 0 or res is None:
        print(f"no frames received (error: {st.error})")
        return True
    span = max(1e-9, st.arrivals[-1] - st.arrivals[0])
    print(f"network : {n} frames, first packet after {st.arrivals[0]:.2f}s, arrived at {(n - 1) / span:.1f} fps "
          f"(faster than {FPS} = we are ahead of playback), {st.wire_bytes * 8 / (n / FPS) / 1e6:.2f} Mbps of video")
    errs, draws = res["errs"], res["draws"]
    play_s = max(1e-9, draws[-1] - draws[0])
    print(f"playback: started {res['t_start'] - t_req:.2f}s after sending the audio "
          f"(first frame + first sound, incl. {args.prebuffer:.1f}s prebuffer)")
    print(f"          clock advanced {res['last_idx'] / play_s:.2f} fps over {play_s:.1f}s; "
          f"{res['shown']} frames drawn, {res['skipped']} skipped (fell behind), "
          f"{res['late']} drawn late (>1.5 frames)")
    print(f"          held on last frame (next one not there yet): {res['hold_s']:.2f}s | "
          f"max packets buffered ahead: {res['max_depth']} ({res['max_depth'] / FPS:.1f}s)")
    if errs:
        e = [x * 1000 for x in errs]
        print(f"          draw lateness vs due time: p50 {pct(e, .5):.1f}ms  p95 {pct(e, .95):.1f}ms  max {max(e):.1f}ms "
              f"(one frame = {1000 / FPS:.0f}ms)")
    if st.error:
        print(f"          stream ended with: {st.error}")
    if st.stats:
        s = st.stats
        print(f"server  : total {s.get('server_total_s')}s, first packet {s.get('server_first_packet_s')}s, "
              f"{s.get('video_bitrate_mbps')} Mbps, error={s.get('error')}")
    return not res["quit"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--prebuffer", type=float, default=0.5)
    ap.add_argument("--av-offset-ms", type=float, default=0.0)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--mute", action="store_true")
    args = ap.parse_args()

    cv2 = None
    if not args.no_display:
        import cv2 as _cv2

        cv2 = _cv2

    with open(args.audio, "rb") as f:
        audio_bytes = f.read()

    # ping_interval=None: a slow link can queue data ahead of the pong and trip the keepalive
    async with websockets.connect(args.url, max_size=None, open_timeout=600,
                                  close_timeout=600, ping_interval=None) as ws:
        for i in range(1, args.runs + 1):
            if not await one_run(ws, audio_bytes, args, cv2, f"RUN {i}"):
                break

    if cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
