"""Fast replacement for MuseTalk's get_image_blending (musetalk/utils/blending.py)."""
import numpy as np


def fast_blend(frame, face, face_box, mask, crop_box):
    """Same output as get_image_blending, bit for bit, but it only touches the
    crop around the face. The original converts the whole 1280x720 frame to a
    PIL image and back on every call, which is where nearly all of its time
    (and its GIL contention with the other pipeline threads) goes.

    frame:    HxWx3 uint8 BGR. A private copy: it is modified in place and returned.
    face:     hxwx3 uint8 BGR, exactly the size of face_box.
    face_box: (x, y, x1, y1) of the generated face inside the frame.
    mask:     uint8 blend weights (0..255), HxW or HxWx3, exactly the size of crop_box.
    crop_box: (x_s, y_s, x_e, y_e); may stick out of the frame.
    """
    H, W = frame.shape[:2]
    x, y, x1, y1 = face_box
    xs, ys, xe, ye = crop_box

    # part of the crop that actually lies inside the frame
    ix0, iy0, ix1, iy1 = max(xs, 0), max(ys, 0), min(xe, W), min(ye, H)
    if ix1 <= ix0 or iy1 <= iy0:
        return frame

    region = frame[iy0:iy1, ix0:ix1]  # view into frame
    top = region.copy()  # "face_large": the crop with the generated face pasted in
    fx0, fy0, fx1, fy1 = max(x, ix0), max(y, iy0), min(x1, ix1), min(y1, iy1)
    if fx1 > fx0 and fy1 > fy0:
        top[fy0 - iy0:fy1 - iy0, fx0 - ix0:fx1 - ix0] = face[fy0 - y:fy1 - y, fx0 - x:fx1 - x]

    m = mask[..., 0] if mask.ndim == 3 else mask
    m = m[iy0 - ys:iy1 - ys, ix0 - xs:ix1 - xs].astype(np.uint16)[..., None]

    # Pillow's masked paste: DIV255(base*(255-m) + top*m), DIV255(a) = (((a+128)>>8) + (a+128)) >> 8
    t = region.astype(np.uint16) * (255 - m) + top.astype(np.uint16) * m + 128
    region[...] = ((t >> 8) + t) >> 8
    return frame
