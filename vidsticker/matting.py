"""Alpha matting: chroma keying, neural matting, and the hybrid of the two.

The hybrid is the interesting part. On a green-screen source a chroma key gives
pixel-accurate edges but happily keeps every non-green thing in the background
(lens flares, sparkle overlays, light rays). A neural matte knows what the
subject *is* and drops those, but its edges are soft and it tends to leave a
halo of screen-coloured fringe. Taking the per-pixel minimum of the two alphas
keeps the strengths of both: the key defines the edge, the network defines what
counts as subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Chroma key colour expressed as a (Cr, Cb) pair - the luma-independent part of
# YCrCb, so a screen that is lit unevenly still matches on a single key.
KeyColor = Tuple[float, float]

MODES = ("auto", "hybrid", "chroma", "ai")


# Fallbacks for when the ramp cannot be calibrated against a neural matte.
# A saturated key sits roughly 64 chroma units from neutral grey, so the ramp
# has to span a large part of that distance - a narrow one reaches full opacity
# midway through the screen-to-subject blend and keeps the backdrop's half of it.
DEFAULT_TOLERANCE = 10.0
DEFAULT_SOFTNESS = 45.0

# Bounds on what the per-frame calibration may choose.
MIN_TOLERANCE, MAX_TOLERANCE = 6.0, 30.0
MIN_SOFTNESS, MAX_SOFTNESS = 15.0, 120.0


@dataclass
class MatteConfig:
    mode: str = "auto"
    model: str = "isnet-general-use"
    key: Optional[KeyColor] = None
    tolerance: Optional[float] = None   # None = calibrate per frame
    softness: Optional[float] = None    # None = calibrate per frame
    despill: float = 0.8
    feather: float = 0.0
    shrink: float = 0.0
    ai_gain: float = 1.0

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.softness is not None and self.softness <= 0:
            raise ValueError("softness must be > 0")
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        if not 0.0 <= self.despill <= 1.0:
            raise ValueError("despill must be within 0..1")


def _ycrcb(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)


def border_pixels(img: np.ndarray, width: int = 8) -> np.ndarray:
    """Flattened list of pixels from the image's outer frame."""
    w = max(1, min(width, min(img.shape[:2]) // 2))
    return np.concatenate([
        img[:w].reshape(-1, img.shape[2]),
        img[-w:].reshape(-1, img.shape[2]),
        img[:, :w].reshape(-1, img.shape[2]),
        img[:, -w:].reshape(-1, img.shape[2]),
    ])


def detect_key(rgb: np.ndarray) -> Optional[KeyColor]:
    """Guess a chroma key colour from the image border.

    Returns None when the border does not look like a solid coloured screen -
    either it is too desaturated (a real-world background) or too varied (a
    busy scene running off the edge of frame).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    edge = border_pixels(hsv)
    sat = float(np.median(edge[:, 1]))
    if sat < 60:
        return None

    # Hue is circular, so measure spread on the unit circle rather than with a
    # plain standard deviation (which breaks across the 179 -> 0 wrap).
    ang = edge[:, 0].astype(np.float32) * (2 * np.pi / 180.0)
    coherence = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    if coherence < 0.85:
        return None

    edge_ycc = border_pixels(_ycrcb(rgb))
    return float(np.median(edge_ycc[:, 1])), float(np.median(edge_ycc[:, 2]))


def chroma_alpha(rgb: np.ndarray, key: KeyColor, tolerance: float, softness: float) -> np.ndarray:
    """Soft alpha from chroma distance to the key colour.

    Pixels within `tolerance` of the key are fully transparent; alpha ramps to
    opaque over the next `softness` units of distance, which is what gives
    antialiased edges rather than a stair-stepped cutout.
    """
    ycc = _ycrcb(rgb)
    dist = np.hypot(ycc[..., 1] - key[0], ycc[..., 2] - key[1])
    return np.clip((dist - tolerance) / softness, 0.0, 1.0)


def calibrate_ramp(rgb: np.ndarray, key: KeyColor, ai: np.ndarray) -> Optional[Tuple[float, float]]:
    """Fit the key's ramp to the colour separation this frame actually shows.

    A fixed ramp is guesswork: how far the subject's colours sit from the key
    depends entirely on the footage. Using the neural matte to label confident
    foreground and background, we can measure it instead - put the transparent
    end past the screen's own spread, and reach full opacity right where the
    subject's colours begin.

    Getting this wrong is what leaves a halo on motion blur. A blurred edge is a
    genuine mix of subject and screen, so it lands midway between the two in
    chroma. If the ramp saturates before that midpoint, every blurred edge pixel
    is written out as fully opaque backdrop.
    """
    ycc = _ycrcb(rgb)
    dist = np.hypot(ycc[..., 1] - key[0], ycc[..., 2] - key[1])

    # Erode both sides: we want pixels that are unambiguously one or the other,
    # not the soft boundary between them.
    solid = np.ones((15, 15), np.uint8)
    fg = cv2.erode((ai > 0.9).astype(np.uint8), solid).astype(bool)
    bg = cv2.erode((ai < 0.1).astype(np.uint8), solid).astype(bool)
    if fg.sum() < 500 or bg.sum() < 500:
        return None

    # p90 rather than the max: an animated backdrop carries overlays (rays,
    # sparkles) that are nowhere near the key and would otherwise set the floor.
    spread = float(np.percentile(dist[bg], 90))
    subject = float(np.percentile(dist[fg], 1))

    tolerance = float(np.clip(spread * 1.5, MIN_TOLERANCE, MAX_TOLERANCE))
    softness = float(np.clip(subject - tolerance, MIN_SOFTNESS, MAX_SOFTNESS))
    return tolerance, softness


def ai_alpha(session, rgb: np.ndarray) -> np.ndarray:
    """Subject alpha from a rembg segmentation model."""
    from PIL import Image
    from rembg import remove

    mask = remove(Image.fromarray(rgb), session=session,
                  only_mask=True, post_process_mask=False)
    return np.asarray(mask).astype(np.float32) / 255.0


def key_to_rgb(key: KeyColor, luma: float = 128.0) -> np.ndarray:
    """A representative RGB colour for a (Cr, Cb) key, at the given luma."""
    px = np.array([[[luma, key[0], key[1]]]], dtype=np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_YCrCb2RGB)[0, 0]


def spill_channel(key: Optional[KeyColor]) -> Optional[int]:
    """Which RGB channel the screen spills into: 0=red, 1=green, 2=blue."""
    if key is None:
        return None
    return int(np.argmax(key_to_rgb(key)))


def despill(rgb: np.ndarray, channel: Optional[int] = 1, amount: float = 0.8) -> np.ndarray:
    """Pull back screen colour bleeding onto the subject.

    Edge pixels on a green screen pick up green from the backdrop. Clamping the
    key's channel toward the strongest of the other two removes the tint without
    touching pixels that lead in that channel by a large margin.
    """
    f = rgb.astype(np.float32)
    if amount <= 0 or channel is None:
        return f
    others = [c for c in (0, 1, 2) if c != channel]
    limit = np.maximum(f[..., others[0]], f[..., others[1]])
    tinted = f[..., channel] > limit
    out = f.copy()
    out[..., channel] = np.where(
        tinted, limit + (f[..., channel] - limit) * (1.0 - amount), f[..., channel])
    return out


def screen_rgb(rgb: np.ndarray, alpha: np.ndarray, thresh: float = 0.1) -> Optional[np.ndarray]:
    """The screen's own colour, as the median of the confidently transparent pixels."""
    bg = alpha < thresh
    if bg.sum() < 200:
        return None
    return np.median(rgb[bg], axis=0).astype(np.float32)


def unmix_screen(rgb: np.ndarray, alpha: np.ndarray, screen: np.ndarray,
                 amount: float = 1.0, floor: float = 0.15) -> np.ndarray:
    """Recover the subject's own colour from a pixel that is part screen.

    A partly covered pixel - an antialiased outline, and far more of them on
    motion blur - is literally `C = a*F + (1-a)*S`. Solving that for F removes
    exactly the screen's share, where despill only approximates it by clamping a
    channel. On a blurred edge, which can be half screen, the difference is a
    visible green fringe versus none.
    """
    if amount <= 0:
        return rgb.astype(np.float32)
    a = np.maximum(alpha, floor)[..., None]
    fg = (rgb.astype(np.float32) - (1.0 - a) * screen.reshape(1, 1, 3)) / a
    return np.clip(rgb.astype(np.float32) * (1.0 - amount) + fg * amount, 0.0, 255.0)


def _postprocess(alpha: np.ndarray, cfg: MatteConfig) -> np.ndarray:
    if cfg.shrink > 0:
        k = int(cfg.shrink) * 2 + 1
        alpha = cv2.erode(alpha, np.ones((k, k), np.uint8))
    if cfg.feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), cfg.feather)
    return np.clip(alpha, 0.0, 1.0)


def compute_alpha(rgb: np.ndarray, cfg: MatteConfig, session=None,
                  key: Optional[KeyColor] = None) -> np.ndarray:
    """Alpha channel for one frame, following the configured strategy."""
    use_chroma = cfg.mode in ("chroma", "hybrid") or (cfg.mode == "auto" and key is not None)
    use_ai = cfg.mode in ("ai", "hybrid") or (cfg.mode == "auto")

    alpha = None
    if use_ai:
        if session is None:
            raise ValueError("a rembg session is required for 'ai' and 'hybrid' modes")
        alpha = np.clip(ai_alpha(session, rgb) * cfg.ai_gain, 0.0, 1.0)

    if use_chroma:
        if key is None:
            if cfg.mode == "chroma":
                raise ValueError("no chroma key colour found; pass --key or use --mode ai")
        else:
            tolerance, softness = cfg.tolerance, cfg.softness
            if (tolerance is None or softness is None) and alpha is not None:
                fitted = calibrate_ramp(rgb, key, alpha)
                if fitted:
                    tolerance = cfg.tolerance if cfg.tolerance is not None else fitted[0]
                    softness = cfg.softness if cfg.softness is not None else fitted[1]
            ck = chroma_alpha(rgb, key,
                              DEFAULT_TOLERANCE if tolerance is None else tolerance,
                              DEFAULT_SOFTNESS if softness is None else softness)
            alpha = ck if alpha is None else np.minimum(alpha, ck)

    if alpha is None:
        raise ValueError(f"no matting strategy applies for mode {cfg.mode!r}")
    return _postprocess(alpha, cfg)


def temporal_despeckle(prev: Optional[np.ndarray], cur: np.ndarray,
                       nxt: Optional[np.ndarray]) -> np.ndarray:
    """Drop coverage that only this frame claims, without inventing any.

    A per-frame neural matte flickers: pixels blink opaque for a single frame. A
    median of three consecutive alphas removes that.

    But a plain median also *adds* coverage, and on fast motion that is ruinous.
    Where a limb is moving, the two neighbouring frames can agree with each other
    while the current frame disagrees, so the median paints the limb's other
    position onto this frame — over backdrop, which then ships as opaque screen
    colour. On the example clip that turned 18 stray pixels into 15,282.

    Clamping to the current frame keeps the despeckling and drops the invention:
    a pixel can never come out more opaque than this frame's own matte made it,
    so it still had to convince both the key and the network *here*.
    """
    if prev is None or nxt is None:
        return cur
    return np.minimum(cur, np.median(np.stack([prev, cur, nxt], axis=0), axis=0))


def resize_rgba(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize straight-alpha RGBA without dragging the backdrop into the edge.

    Resampling RGB and alpha independently mixes the colour of transparent
    pixels into their opaque neighbours - and a transparent pixel here still
    holds the screen's colour, so the sticker picks up a green rim on the way
    down to output size. Premultiplying first weights each pixel's colour by its
    own coverage, so transparent ones contribute nothing.
    """
    w, h = size
    a = arr[..., 3].astype(np.float32) / 255.0
    interp = cv2.INTER_AREA if (w < arr.shape[1] or h < arr.shape[0]) else cv2.INTER_LANCZOS4
    pre = cv2.resize(arr[..., :3].astype(np.float32) * a[..., None], (w, h), interpolation=interp)
    a_out = cv2.resize(a, (w, h), interpolation=interp)
    rgb = pre / np.maximum(a_out, 1e-4)[..., None]
    return np.dstack([np.clip(rgb, 0, 255), a_out * 255.0]).astype(np.uint8)


def content_bbox(alpha: np.ndarray, threshold: float = 0.06) -> Optional[Tuple[int, int, int, int]]:
    """Tight (left, top, right, bottom) box around non-transparent pixels."""
    mask = alpha > threshold
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
