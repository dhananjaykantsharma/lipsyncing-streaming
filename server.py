import modal

from image import app, musetalk_image
from volume import volume, MODEL_DIR, AVATAR_DIR
from blendutil import fast_blend
from h264util import H264Encoder

AVATAR_ID = "avatar_1"
VERSION = "v15"


def _audio_suffix(audio_bytes: bytes) -> str:
    """Temp-file extension for the incoming audio: librosa/soundfile choose
    the decoder from it, so MP3 bytes saved as .wav would fail to load."""
    if audio_bytes[:3] == b"ID3" or (len(audio_bytes) > 1 and audio_bytes[0] == 0xFF and audio_bytes[1] & 0xE0 == 0xE0):
        return ".mp3"
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    return ".wav"


def _summ(xs):
    """count / mean / p50 / p95 / max / total of a list of seconds, in ms."""
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(1000 * sum(s) / n, 2),
        "p50_ms": round(1000 * s[n // 2], 2),
        "p95_ms": round(1000 * s[min(n - 1, int(n * 0.95))], 2),
        "max_ms": round(1000 * s[-1], 2),
        "total_s": round(sum(s), 2),
    }


@app.cls(
    image=musetalk_image,
    volumes={"/data": volume},
    gpu="A100",
    # blend + x264 are CPU-bound; a default container gets very few cores
    cpu=8,
    timeout=3600,
    scaledown_window=300,
)
class MuseTalkInference:

    @modal.enter()
    def load(self):
        """Runs once per container start: loads all models + the cached
        avatar (built by preprocess_avatar.py) into GPU memory, so a live
        request only has to run inference, not preprocessing."""
        import os
        import sys
        import argparse
        import torch
        from transformers import WhisperModel

        repo = "/root/MuseTalk"
        os.chdir(repo)
        sys.path.insert(0, repo)

        if not os.path.lexists(f"{repo}/results"):
            os.symlink(f"{AVATAR_DIR}/results", f"{repo}/results")
        if not os.path.lexists(f"{repo}/models"):
            os.symlink(MODEL_DIR, f"{repo}/models")

        from musetalk.utils.utils import load_all_model
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.audio_processor import AudioProcessor
        import scripts.realtime_inference as rt

        if VERSION == "v15":
            model_dir = f"{repo}/models/musetalkV15"
            unet_model_path = f"{model_dir}/unet.pth"
        else:
            model_dir = f"{repo}/models/musetalk"
            unet_model_path = f"{model_dir}/pytorch_model.bin"
        unet_config = f"{model_dir}/musetalk.json"

        device = torch.device("cuda")
        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type="sd-vae",
            unet_config=unet_config,
            device=device,
        )
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
        weight_dtype = unet.model.dtype

        import soundfile
        # TTS audio arrives as MP3; without soundfile MP3 support librosa
        # falls back to the (much slower) audioread/ffmpeg path
        print(f"soundfile {soundfile.__libsndfile_version__}: MP3 decode = {'MP3' in soundfile.available_formats()}")

        whisper_dir = f"{repo}/models/whisper"
        audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        whisper = WhisperModel.from_pretrained(whisper_dir)
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)

        fp = (
            FaceParsing(left_cheek_width=90, right_cheek_width=90)
            if VERSION == "v15"
            else FaceParsing()
        )

        # scripts.realtime_inference.Avatar reads these as MODULE-LEVEL
        # globals (same gotcha as in preprocess_avatar.py) — inject them
        # before instantiating Avatar or calling .inference().
        rt.args = argparse.Namespace(
            version=VERSION,
            extra_margin=10,
            parsing_mode="jaw",
            audio_padding_length_left=2,
            audio_padding_length_right=2,
            skip_save_images=False,
        )
        rt.vae = vae
        rt.unet = unet
        rt.pe = pe
        rt.device = device
        rt.weight_dtype = weight_dtype
        rt.whisper = whisper
        rt.audio_processor = audio_processor
        rt.fp = fp
        rt.timesteps = torch.tensor([0], device=device)

        self.rt = rt

        # preparation=False -> loads latents.pt / coords.pkl / mask files
        # we already built in preprocess_avatar.py, instead of recomputing.
        self.avatar = rt.Avatar(
            avatar_id=AVATAR_ID,
            video_path="",
            bbox_shift=0,
            # MuseTalk's own realtime_inference.py defaults to 20 (better
            # GPU utilization per batch); we'd started conservative at 8.
            batch_size=20,
            preparation=False,
        )

    def _generate(self, audio_bytes: bytes, fps: int = 25) -> bytes:
        """audio bytes in -> lip-synced mp4 bytes out. v1: generates the
        whole clip then returns it in one shot (MuseTalk is fast enough on
        A100 that this is usable); true frame-by-frame streaming is a
        follow-up optimization, not needed to get end-to-end working."""
        import os
        import tempfile
        import uuid

        with tempfile.NamedTemporaryFile(suffix=_audio_suffix(audio_bytes), delete=False) as f:
            f.write(audio_bytes)
            audio_path = f.name

        out_name = uuid.uuid4().hex
        video_path = os.path.join(self.avatar.video_out_path, out_name + ".mp4")
        try:
            self.avatar.inference(audio_path, out_name, fps, skip_save_images=False)
            with open(video_path, "rb") as f:
                video_bytes = f.read()
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(video_path):
                os.remove(video_path)

        return video_bytes

    def _run_h264_pipeline(self, audio_bytes: bytes, emit, stop, stats: dict, fps: int = 25, n_blend: int = 2):
        """audio -> [GPU: lip shapes] -> [thread pool: paste onto avatar frame]
        -> [encoder thread: H.264] -> emit(seq, is_keyframe, packet_bytes).

        The stages overlap like an assembly line: while the GPU works on
        batch N+1, the pool is blending batch N and the encoder is encoding
        batch N-1. Blocking, so call it from a worker thread. `stop` (a
        threading.Event) aborts early, e.g. if the client disconnects.
        Per-stage timings (seconds) are recorded into `stats`."""
        import os
        import queue
        import tempfile
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        import cv2
        import numpy as np
        import torch

        avatar = self.avatar
        rt = self.rt
        h, w = avatar.frame_list_cycle[0].shape[:2]
        w, h = w - w % 2, h - h % 2  # yuv420p needs even dimensions
        encoder = H264Encoder(w, h, fps=fps)

        stats.update(
            gpu_batch=[], gpu_per_frame=[], blend=[], encode=[],
            packet_bytes=[], pending_depth=[], keyframes=0,
        )

        def blend(idx, res_frame):
            t = time.perf_counter()
            bbox = avatar.coord_list_cycle[idx % len(avatar.coord_list_cycle)]
            ori_frame = avatar.frame_list_cycle[idx % len(avatar.frame_list_cycle)].copy()
            x1, y1, x2, y2 = bbox
            try:
                face = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                # degenerate bbox: keep the untouched avatar frame so the frame
                # count (and therefore lip-sync timing) stays intact
                stats["blend"].append(time.perf_counter() - t)
                return np.ascontiguousarray(ori_frame[:h, :w])
            mask = avatar.mask_list_cycle[idx % len(avatar.mask_list_cycle)]
            mask_box = avatar.mask_coords_list_cycle[idx % len(avatar.mask_coords_list_cycle)]
            # bit-exact with rt.get_image_blending, ~8x cheaper (crop only, no full-frame PIL round trip)
            out = fast_blend(ori_frame, face, bbox, mask, mask_box)
            stats["blend"].append(time.perf_counter() - t)
            return np.ascontiguousarray(out[:h, :w])

        pending = queue.Queue(maxsize=48)  # futures, in frame order
        SENTINEL = object()
        errors = []

        def record(seq, packets):
            for data, key in packets:
                emit(seq, key, data)
                seq += 1
                stats["packet_bytes"].append(len(data))
                stats["keyframes"] += int(key)
            return seq

        def encode_loop():
            seq = 0
            try:
                while True:
                    fut = pending.get()
                    if fut is SENTINEL:
                        break
                    stats["pending_depth"].append(pending.qsize())
                    frame = fut.result()
                    if stop.is_set():
                        continue
                    t = time.perf_counter()
                    packets = encoder.encode(frame)
                    stats["encode"].append(time.perf_counter() - t)
                    seq = record(seq, packets)
                if not stop.is_set():
                    record(seq, encoder.flush())
            except Exception as e:  # surface it and unblock the GPU thread
                errors.append(e)
                stop.set()

        def put_pending(item):
            while not stop.is_set():
                try:
                    pending.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        with tempfile.NamedTemporaryFile(suffix=_audio_suffix(audio_bytes), delete=False) as f:
            f.write(audio_bytes)
            audio_path = f.name

        pool = ThreadPoolExecutor(max_workers=n_blend)
        enc_thread = threading.Thread(target=encode_loop, daemon=True)
        enc_thread.start()
        try:
            t_start = time.perf_counter()
            t_audio = t_start
            # CPU: librosa load (+ resample to 16 kHz) and mel features
            whisper_input_features, librosa_length = rt.audio_processor.get_audio_feature(
                audio_path, weight_dtype=rt.weight_dtype
            )
            stats["audio_load_mel_s"] = round(time.perf_counter() - t_audio, 3)
            stats["audio_duration_s"] = round(librosa_length / 16000, 3)
            t_whisper = time.perf_counter()
            # GPU: whisper encoder over the WHOLE clip, before any frame is made
            whisper_chunks = rt.audio_processor.get_whisper_chunk(
                whisper_input_features,
                rt.device,
                rt.weight_dtype,
                rt.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=rt.args.audio_padding_length_left,
                audio_padding_length_right=rt.args.audio_padding_length_right,
            )
            torch.cuda.synchronize()
            stats["whisper_s"] = round(time.perf_counter() - t_whisper, 3)
            stats["audio_features_s"] = round(time.perf_counter() - t_audio, 3)
            stats["total_frames_expected"] = len(whisper_chunks)

            gen = rt.datagen(whisper_chunks, avatar.input_latent_list_cycle, avatar.batch_size)
            t_loop = time.perf_counter()

            idx = 0
            for whisper_batch, latent_batch in gen:
                if stop.is_set():
                    break
                t_gpu = time.perf_counter()
                audio_feature_batch = rt.pe(whisper_batch.to(rt.device))
                latent_batch = latent_batch.to(device=rt.device, dtype=rt.unet.model.dtype)
                pred_latents = rt.unet.model(
                    latent_batch, rt.timesteps, encoder_hidden_states=audio_feature_batch
                ).sample
                pred_latents = pred_latents.to(device=rt.device, dtype=rt.vae.vae.dtype)
                recon = rt.vae.decode_latents(pred_latents)
                torch.cuda.synchronize()
                gpu_s = time.perf_counter() - t_gpu
                stats["gpu_batch"].append(gpu_s)
                stats["gpu_per_frame"].append(gpu_s / max(1, len(recon)))
                if "first_batch_ready_s" not in stats:
                    # pipeline start -> first batch of frames off the GPU
                    stats["first_batch_ready_s"] = round(time.perf_counter() - t_start, 3)
                    stats["first_batch_frames"] = len(recon)

                for res_frame in recon:
                    if not put_pending(pool.submit(blend, idx, res_frame)):
                        break
                    idx += 1
            # all GPU batches, incl. time blocked on a full blend/encode queue
            stats["gpu_loop_s"] = round(time.perf_counter() - t_loop, 3)
        finally:
            while enc_thread.is_alive():
                try:
                    pending.put(SENTINEL, timeout=0.5)
                    break
                except queue.Full:
                    continue
            enc_thread.join()
            pool.shutdown(wait=True)
            os.remove(audio_path)

        if errors:
            raise errors[0]

    @modal.method()
    def test_generate(self, sample_audio_path: str = "data/audio/eng.wav", fps: int = 25) -> bytes:
        """Smoke test using one of MuseTalk's own bundled sample audio
        clips, so we can validate the whole pipeline before wiring up real
        audio from the interview app."""
        import os

        full_path = os.path.join("/root/MuseTalk", sample_audio_path)
        with open(full_path, "rb") as f:
            audio_bytes = f.read()
        return self._generate(audio_bytes, fps)

    @modal.asgi_app()
    def web(self):
        import asyncio
        import threading
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        @web_app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    audio_bytes = await websocket.receive_bytes()
                    # _generate() blocks on the GPU for 100+ seconds — run
                    # it in a thread so the event loop stays free to answer
                    # WebSocket ping/pong, otherwise the connection times
                    # out mid-generation.
                    video_bytes = await asyncio.to_thread(self._generate, audio_bytes)
                    await websocket.send_bytes(video_bytes)
            except WebSocketDisconnect:
                pass

        @web_app.websocket("/ws-stream")
        async def ws_stream_endpoint(websocket: WebSocket):
            """audio in -> H.264 packets out, each sent the moment it exists.

            Protocol per request:
              text   {"type": "start", "codec": "h264", "format": "annexb", "width", "height", "fps"}
              binary [1B flags: bit0 = keyframe][4B frame number, big-endian][H.264 access unit]  (x N)
              text   {"type": "done", "stats": {...}}
            The pipeline runs on a worker thread; frames are handed to this
            async handler through a queue so the event loop stays free for
            WebSocket pings while the GPU is busy."""
            import json
            import struct
            import time

            await websocket.accept()
            loop = asyncio.get_event_loop()
            try:
                while True:
                    audio_bytes = await websocket.receive_bytes()
                    t_recv = time.perf_counter()
                    out_q: asyncio.Queue = asyncio.Queue()
                    stop = threading.Event()
                    gen_stats: dict = {}
                    err: list = []

                    def emit(seq, key, data, out_q=out_q):
                        loop.call_soon_threadsafe(out_q.put_nowait, (seq, key, data))

                    def run(audio_bytes=audio_bytes, out_q=out_q, stop=stop, gen_stats=gen_stats, err=err):
                        try:
                            self._run_h264_pipeline(audio_bytes, emit, stop, gen_stats)
                        except Exception as e:
                            err.append(repr(e))
                        finally:
                            loop.call_soon_threadsafe(out_q.put_nowait, None)

                    threading.Thread(target=run, daemon=True).start()

                    fh, fw = self.avatar.frame_list_cycle[0].shape[:2]
                    await websocket.send_text(json.dumps({
                        "type": "start", "codec": "h264", "format": "annexb",
                        "width": fw - fw % 2, "height": fh - fh % 2, "fps": 25,
                    }))

                    sent = 0
                    first_packet_s = None
                    ws_send_s = 0.0
                    try:
                        while True:
                            item = await out_q.get()
                            if item is None:
                                break
                            seq, key, data = item
                            t_send = time.perf_counter()
                            await websocket.send_bytes(struct.pack(">BI", 1 if key else 0, seq) + data)
                            ws_send_s += time.perf_counter() - t_send
                            sent += 1
                            if first_packet_s is None:
                                first_packet_s = time.perf_counter() - t_recv
                    finally:
                        stop.set()  # client gone (or finished): stop the GPU work

                    pb = gen_stats.get("packet_bytes", [])
                    depth = gen_stats.get("pending_depth", [])
                    server_total_s = time.perf_counter() - t_recv
                    audio_dur = gen_stats.get("audio_duration_s")
                    server_stats = {
                        "server_total_s": round(server_total_s, 3),
                        "server_first_packet_s": round(first_packet_s or 0, 3),
                        "audio_upload_bytes": len(audio_bytes),
                        "audio_duration_s": audio_dur,
                        # CPU: audio decode + resample + mel
                        "audio_load_mel_s": gen_stats.get("audio_load_mel_s"),
                        # GPU: whisper encoder over the whole clip
                        "whisper_s": gen_stats.get("whisper_s"),
                        "audio_features_s": gen_stats.get("audio_features_s"),
                        # pipeline start -> first GPU batch done (includes whisper)
                        "first_batch_ready_s": gen_stats.get("first_batch_ready_s"),
                        "first_batch_frames": gen_stats.get("first_batch_frames"),
                        "gpu_loop_s": gen_stats.get("gpu_loop_s"),
                        # < 1.0 => generated faster than it plays back
                        "realtime_factor": round(server_total_s / audio_dur, 3) if audio_dur else None,
                        # time blocked pushing packets into the socket (slow downlink shows up here)
                        "ws_send_total_s": round(ws_send_s, 3),
                        "frames_expected": gen_stats.get("total_frames_expected"),
                        "packets_sent": sent,
                        "keyframes": gen_stats.get("keyframes"),
                        "avg_packet_kb": round(sum(pb) / max(1, len(pb)) / 1024, 2),
                        "video_bitrate_mbps": round(sum(pb) * 8 / (max(1, len(pb)) / 25) / 1e6, 3),
                        "gpu_per_batch": _summ(gen_stats.get("gpu_batch", [])),
                        "gpu_per_frame": _summ(gen_stats.get("gpu_per_frame", [])),
                        "blend_per_frame": _summ(gen_stats.get("blend", [])),
                        "encode_per_frame": _summ(gen_stats.get("encode", [])),
                        # ~48 => encoder is the bottleneck, ~0 => GPU/blend can't keep up
                        "frames_waiting_for_encoder_mean": round(sum(depth) / max(1, len(depth)), 1),
                        "error": err[0] if err else None,
                    }
                    print("SERVER STATS:", json.dumps(server_stats))
                    await websocket.send_text(json.dumps({"type": "done", "stats": server_stats}))
            except WebSocketDisconnect:
                pass

        return web_app


@app.local_entrypoint()
def main():
    import pathlib

    inference = MuseTalkInference()
    video_bytes = inference.test_generate.remote()
    out = pathlib.Path("test_output.mp4")
    out.write_bytes(video_bytes)
    print(f"Wrote {len(video_bytes)} bytes to {out}")
