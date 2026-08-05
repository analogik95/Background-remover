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


@dataclass
class MatteConfig:
    mode: str = "auto"
    model: str = "isnet-general-use"
    key: Optional[KeyColor] = None
    tolerance: float = 8.0
    softness: float = 20.0
    despill: float = 0.8
    feather: float = 0.0
    shrink: float = 0.0
    ai_gain: float = 1.0

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.softness <= 0:
            raise ValueError("softness must be > 0")
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
            ck = chroma_alpha(rgb, key, cfg.tolerance, cfg.softness)
            alpha = ck if alpha is None else np.minimum(alpha, ck)

    if alpha is None:
        raise ValueError(f"no matting strategy applies for mode {cfg.mode!r}")
    return _postprocess(alpha, cfg)


def temporal_median(prev: Optional[np.ndarray], cur: np.ndarray,
                    nxt: Optional[np.ndarray]) -> np.ndarray:
    """Median of three consecutive alphas.

    A per-frame neural matte flickers slightly from frame to frame; a 3-tap
    median removes single-frame speckle without smearing genuine motion the way
    an average would.
    """
    stack = [a for a in (prev, cur, nxt) if a is not None]
    if len(stack) < 3:
        return cur
    return np.median(np.stack(stack, axis=0), axis=0)


def content_bbox(alpha: np.ndarray, threshold: float = 0.06) -> Optional[Tuple[int, int, int, int]]:
    """Tight (left, top, right, bottom) box around non-transparent pixels."""
    mask = alpha > threshold
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
