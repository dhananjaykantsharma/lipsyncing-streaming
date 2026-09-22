"""Low-latency H.264 (Annex-B) encode/decode helpers shared by the server and
the test clients. Every packet is one complete access unit, so it can be fed
straight to a decoder (PyAV here, WebCodecs in a browser) the moment it arrives."""
from fractions import Fraction


class H264Encoder:
    """One BGR frame in -> the packets it produced out (SPS/PPS in-band on keyframes)."""

    def __init__(self, width, height, fps=25, crf=26, maxrate_kbps=2500, gop=None, preset="veryfast"):
        import av

        if width % 2 or height % 2:
            raise ValueError("yuv420p needs even width/height")
        self.width, self.height = width, height
        ctx = av.CodecContext.create("libx264", "w")
        ctx.width = width
        ctx.height = height
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, fps)
        ctx.framerate = Fraction(fps, 1)
        ctx.gop_size = gop or fps
        ctx.options = {
            "preset": preset,
            "tune": "zerolatency",
            "crf": str(crf),
            "maxrate": f"{maxrate_kbps}k",
            "bufsize": f"{maxrate_kbps // 2}k",
        }
        ctx.open()
        self.ctx = ctx
        self.n = 0

    def encode(self, bgr):
        import av

        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p")
        frame.pts = self.n
        self.n += 1
        return [(bytes(p), bool(p.is_keyframe)) for p in self.ctx.encode(frame)]

    def flush(self):
        return [(bytes(p), bool(p.is_keyframe)) for p in self.ctx.encode(None)]


class H264Decoder:
    """Feed it one packet at a time; get back BGR ndarrays."""

    def __init__(self):
        import av

        self.ctx = av.CodecContext.create("h264", "r")

    def decode(self, data):
        import av

        return [f.to_ndarray(format="bgr24") for f in self.ctx.decode(av.Packet(data))]

    def flush(self):
        return [f.to_ndarray(format="bgr24") for f in self.ctx.decode(None)]
