"""Writers for the animated sticker formats.

GIF is the awkward one: 256 colours and a *single* fully-transparent palette
index, so soft alpha has to be thresholded. WebP and APNG carry real 8-bit
alpha, so the same frames come out noticeably cleaner there - which is why the
pipeline offers them alongside the GIF rather than instead of it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Sequence

from PIL import Image

from .video import ffmpeg_bin

# GIF delays are stored in hundredths of a second, so only these rates play back
# at their true speed. Anything else gets rounded by the decoder.
GIF_EXACT_FPS = (50.0, 25.0, 20.0, 12.5, 10.0, 5.0)


def gif_timing_error(fps: float) -> float:
    """Relative speed error a GIF decoder introduces at this frame rate."""
    if fps <= 0:
        return 0.0
    delay_cs = max(1, round(100.0 / fps))
    return abs((100.0 / delay_cs) - fps) / fps


def nearest_gif_fps(fps: float) -> float:
    """Closest frame rate that GIF can represent exactly."""
    return min(GIF_EXACT_FPS, key=lambda c: abs(c - fps))


def write_gif(frames: Sequence[Path], out: Path, fps: float,
              alpha_threshold: int = 128, dither: str = "sierra2_4a",
              colors: int = 255, loop: int = 0) -> Path:
    """Encode RGBA PNG frames to a transparent GIF via ffmpeg's palette filters.

    `reserve_transparent` keeps one palette slot free for transparency, and
    `alpha_threshold` decides which soft-alpha pixels claim it.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames[0].parent / "%05d.png")

    use = f"paletteuse=alpha_threshold={alpha_threshold}"
    use += f":dither={dither}" if dither and dither != "none" else ":dither=none"

    cmd = [
        ffmpeg_bin(), "-y", "-v", "error",
        "-framerate", f"{fps}", "-start_number", "0", "-i", pattern,
        "-filter_complex",
        f"[0:v]split[a][b];"
        f"[a]palettegen=max_colors={colors}:reserve_transparent=1:stats_mode=full[p];"
        f"[b][p]{use}",
        # Disable both frame-cropping and transparency-based inter-frame diffing.
        # The diff optimisation reuses the transparent index to mean "unchanged",
        # which conflicts with real transparency and smears the subject into a
        # trail of every previous frame.
        "-gifflags", "-offsetting-transdiff",
        "-loop", str(loop),
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"GIF encode failed:\n{proc.stderr.strip()}")
    return out


def _durations(count: int, fps: float) -> List[int]:
    """Per-frame durations in ms that sum to the true animation length.

    Distributing the rounding error keeps the total duration exact instead of
    letting a fractional frame time drift over a long clip.
    """
    step = 1000.0 / fps
    marks = [round(i * step) for i in range(count + 1)]
    return [max(10, marks[i + 1] - marks[i]) for i in range(count)]


def write_webp(frames: Sequence[Path], out: Path, fps: float,
               quality: int = 90, lossless: bool = False, loop: int = 0,
               method: int = 4) -> Path:
    """Animated WebP - full 8-bit alpha, and far smaller than the GIF.

    `method` trades encode time for compression; 6 costs several minutes on a
    long clip for a few percent of size, so 4 is the default.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.open(p).convert("RGBA") for p in frames]
    imgs[0].save(
        out, save_all=True, append_images=imgs[1:],
        duration=_durations(len(imgs), fps), loop=loop,
        lossless=lossless, quality=quality, method=method,
    )
    return out


def write_apng(frames: Sequence[Path], out: Path, fps: float, loop: int = 0) -> Path:
    """Animated PNG - lossless with full alpha, the highest-fidelity output."""
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.open(p).convert("RGBA") for p in frames]
    imgs[0].save(
        out, save_all=True, append_images=imgs[1:],
        duration=_durations(len(imgs), fps), loop=loop,
        disposal=2, default_image=False,
    )
    return out


def write_preview_sheet(frames: Sequence[Path], out: Path, count: int = 5,
                        height: int = 260, checker: int = 10) -> Path:
    """Contact sheet of sample frames over a checkerboard, for eyeballing edges."""
    import numpy as np

    picks = [frames[round(i * (len(frames) - 1) / max(1, count - 1))]
             for i in range(min(count, len(frames)))]
    tiles = []
    for p in picks:
        im = Image.open(p).convert("RGBA")
        scale = height / im.height
        im = im.resize((max(1, round(im.width * scale)), height), Image.LANCZOS)
        arr = np.asarray(im).astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:im.height, 0:im.width]
        board = np.where(((xx // checker + yy // checker) % 2) == 0, 0.90, 0.72)
        rgb = arr[..., :3] * arr[..., 3:] + board[..., None] * (1 - arr[..., 3:])
        tiles.append(Image.fromarray((rgb * 255).astype("uint8")))

    gap = 8
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), height), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + gap
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
