"""End-to-end video -> transparent sticker pipeline.

Four passes, deliberately staged through a temp directory so peak memory stays
flat regardless of clip length:

  1. decode   - ffmpeg writes RGB frames to disk
  2. matte    - per-frame alpha, saved as 8-bit PNGs
  3. compose  - temporal smoothing, shared crop box, RGBA frames
  4. encode   - GIF / WebP / APNG

The shared crop box is why passes 2 and 3 are separate: the sticker has to be
cropped to the union of every frame's content, which is only known after the
whole clip has been matted.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from . import encoders, matting, video
from .matting import MatteConfig

ProgressFn = Callable[[str, int, int], None]

FORMATS = ("gif", "webp", "apng")


@dataclass
class StickerOptions:
    fps: Optional[float] = None
    size: int = 512
    work_size: int = 1280
    crop: bool = True
    pad: int = 6
    square: bool = False
    smooth: bool = True
    formats: Sequence[str] = ("gif", "webp")
    start: Optional[float] = None
    duration: Optional[float] = None
    loop: int = 0
    alpha_threshold: int = 128
    dither: str = "sierra2_4a"
    colors: int = 255
    webp_quality: int = 90
    preview: bool = True
    matte: MatteConfig = field(default_factory=MatteConfig)

    def validate(self) -> None:
        self.matte.validate()
        bad = [f for f in self.formats if f not in FORMATS]
        if bad:
            raise ValueError(f"unknown output format(s) {bad}; choose from {list(FORMATS)}")
        if not self.formats:
            raise ValueError("at least one output format is required")
        if self.size < 16:
            raise ValueError("size must be at least 16 px")
        if not 1 <= self.alpha_threshold <= 255:
            raise ValueError("alpha-threshold must be within 1..255")
        # One palette slot is reserved for transparency, so 255 is the ceiling.
        if not 2 <= self.colors <= 255:
            raise ValueError("colors must be within 2..255")


@dataclass
class StickerResult:
    outputs: Dict[str, Path]
    frames: int
    fps: float
    width: int
    height: int
    mode: str
    key: Optional[matting.KeyColor]
    preview: Optional[Path] = None

    @property
    def summary(self) -> str:
        dims = f"{self.width}x{self.height}"
        files = ", ".join(
            f"{k} {p.stat().st_size / 1e6:.2f} MB" for k, p in sorted(self.outputs.items())
        )
        return (f"{self.frames} frames @ {self.fps:g} fps, {dims}, "
                f"matte={self.mode} -> {files}")


def _choose_fps(info: video.VideoInfo, opts: StickerOptions) -> float:
    if opts.fps:
        return opts.fps
    src = info.fps or 25.0
    # GIF can only express delays in whole centiseconds. A 24 fps source asks
    # for 4.1666 cs and every decoder rounds it differently, so when GIF is one
    # of the targets we resample to the nearest rate GIF represents exactly.
    if "gif" in opts.formats and encoders.gif_timing_error(src) > 0.02:
        return encoders.nearest_gif_fps(src)
    return src


def _sample_key(frames: Sequence[Path], cfg: MatteConfig) -> Optional[matting.KeyColor]:
    """Detect the chroma key from a few frames spread across the clip.

    Sampling several frames matters when the backdrop is animated: one frame may
    be washed out by an effect, but the median across samples lands on the
    screen's true colour.
    """
    if cfg.key is not None:
        return cfg.key
    if cfg.mode == "ai":
        return None

    picks = [frames[round(i * (len(frames) - 1) / 4)] for i in range(5)]
    found = []
    for p in picks:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        k = matting.detect_key(rgb)
        if k is not None:
            found.append(k)
    if len(found) < 3:
        return None
    arr = np.array(found)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


def _resolve_mode(cfg: MatteConfig, key: Optional[matting.KeyColor]) -> str:
    if cfg.mode != "auto":
        return cfg.mode
    return "hybrid" if key is not None else "ai"


def _pad_box(box, w: int, h: int, pad: int, square: bool):
    left, top, right, bottom = box
    left, top = max(0, left - pad), max(0, top - pad)
    right, bottom = min(w, right + pad), min(h, bottom + pad)
    if square:
        # Centre a square on the content and let it run past the frame edge if
        # it has to. PIL fills out-of-bounds crop area with transparent pixels,
        # which is what a square sticker canvas wants anyway - clamping to the
        # frame here would silently return a non-square box.
        side = max(right - left, bottom - top)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        left, top = round(cx - side / 2), round(cy - side / 2)
        right, bottom = left + side, top + side
    return int(left), int(top), int(right), int(bottom)


def _fit(size: tuple, target: int) -> tuple:
    w, h = size
    longest = max(w, h)
    if longest <= target:
        return w, h
    scale = target / longest
    return max(1, round(w * scale)), max(1, round(h * scale))


def create_sticker(src: Path, out_dir: Path, opts: Optional[StickerOptions] = None,
                   progress: Optional[ProgressFn] = None,
                   stem: Optional[str] = None,
                   keep_work: bool = False) -> StickerResult:
    opts = opts or StickerOptions()
    opts.validate()
    src, out_dir = Path(src), Path(out_dir)
    if not src.is_file():
        raise FileNotFoundError(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or src.stem

    def tick(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    work = Path(tempfile.mkdtemp(prefix="vidsticker-"))
    try:
        # 1. decode -------------------------------------------------------
        tick("probe", 0, 1)
        info = video.probe(src)
        fps = _choose_fps(info, opts)
        tick("probe", 1, 1)

        tick("decode", 0, 1)
        raw = video.extract_frames(
            src, work / "raw", fps=fps, start=opts.start,
            duration=opts.duration, max_side=opts.work_size,
        )
        tick("decode", len(raw), len(raw))

        # 2. matte --------------------------------------------------------
        cfg = opts.matte
        key = _sample_key(raw, cfg)
        mode = _resolve_mode(cfg, key)
        if mode == "chroma" and key is None:
            raise ValueError("chroma mode requested but no key colour was detected; pass --key")

        session = None
        if mode in ("ai", "hybrid"):
            from rembg import new_session
            tick("model", 0, 1)
            session = new_session(cfg.model)
            tick("model", 1, 1)

        run_cfg = MatteConfig(**{**cfg.__dict__, "mode": mode})
        # Re-detect the key on every frame when the user has not pinned one. An
        # animated backdrop (light rays, flicker, a pulsing gradient) shifts the
        # screen colour as the clip runs, and a single clip-wide key leaves
        # patches of backdrop behind on the frames it does not fit.
        track_key = key is not None and cfg.key is None
        alpha_dir = work / "alpha"
        alpha_dir.mkdir()
        for i, fp in enumerate(raw):
            rgb = np.asarray(Image.open(fp).convert("RGB"))
            frame_key = key
            if track_key:
                frame_key = matting.detect_key(rgb) or key
            a = matting.compute_alpha(rgb, run_cfg, session=session, key=frame_key)
            Image.fromarray((a * 255).astype(np.uint8)).save(alpha_dir / fp.name)
            tick("matte", i + 1, len(raw))

        # 3. compose ------------------------------------------------------
        alphas = sorted(alpha_dir.glob("*.png"))

        def load_alpha(i: int) -> Optional[np.ndarray]:
            if not 0 <= i < len(alphas):
                return None
            return np.asarray(Image.open(alphas[i])).astype(np.float32) / 255.0

        # Smoothed mattes go back to disk rather than into a list: holding a
        # whole clip of full-resolution float alpha in memory is hundreds of MB
        # on a short video and unbounded on a long one.
        final_dir = work / "matte"
        final_dir.mkdir()
        box = None
        shape = None
        for i in range(len(alphas)):
            a = load_alpha(i)
            if opts.smooth:
                a = matting.temporal_despeckle(load_alpha(i - 1), a, load_alpha(i + 1))
            shape = a.shape
            Image.fromarray((a * 255).astype(np.uint8)).save(final_dir / alphas[i].name)
            b = matting.content_bbox(a)
            if b:
                box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                             max(box[2], b[2]), max(box[3], b[3]))
            tick("analyse", i + 1, len(alphas))

        h, w = shape
        if box is None:
            raise ValueError(
                "the matte is empty - no subject was found. Try --mode ai, or "
                "--tolerance/--softness if the subject shares the backdrop's colour."
            )
        # --square applies to the output canvas either way, so it still takes
        # effect when the subject crop is turned off.
        region = box if opts.crop else (0, 0, w, h)
        crop = _pad_box(region, w, h, opts.pad if opts.crop else 0, opts.square)
        out_size = _fit((crop[2] - crop[0], crop[3] - crop[1]), opts.size)

        spill = matting.spill_channel(key)
        rgba_dir = work / "rgba"
        rgba_dir.mkdir()
        for i, fp in enumerate(raw):
            rgb = np.asarray(Image.open(fp).convert("RGB"))
            alpha = np.asarray(Image.open(final_dir / fp.name)).astype(np.float32)
            # Unmixing needs both the matte and a reading of the screen, so it
            # only applies once there is a key; otherwise fall back to despill.
            screen = matting.screen_rgb(rgb, alpha / 255.0) if key is not None else None
            if screen is not None:
                clean = matting.unmix_screen(rgb, alpha / 255.0, screen, run_cfg.despill)
            else:
                clean = matting.despill(rgb, spill, run_cfg.despill)
            frame = np.dstack([clean, alpha])
            im = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGBA").crop(crop)
            if im.size != out_size:
                im = Image.fromarray(matting.resize_rgba(np.asarray(im), out_size), "RGBA")
            im.save(rgba_dir / fp.name)
            tick("compose", i + 1, len(raw))

        rgba = sorted(rgba_dir.glob("*.png"))

        # 4. encode -------------------------------------------------------
        outputs: Dict[str, Path] = {}
        for n, fmt in enumerate(opts.formats):
            tick("encode", n, len(opts.formats))
            if fmt == "gif":
                outputs["gif"] = encoders.write_gif(
                    rgba, out_dir / f"{stem}.gif", fps,
                    alpha_threshold=opts.alpha_threshold, dither=opts.dither,
                    colors=opts.colors, loop=opts.loop)
            elif fmt == "webp":
                outputs["webp"] = encoders.write_webp(
                    rgba, out_dir / f"{stem}.webp", fps,
                    quality=opts.webp_quality, loop=opts.loop)
            elif fmt == "apng":
                outputs["apng"] = encoders.write_apng(
                    rgba, out_dir / f"{stem}.png", fps, loop=opts.loop)
        tick("encode", len(opts.formats), len(opts.formats))

        preview = None
        if opts.preview:
            preview = encoders.write_preview_sheet(rgba, out_dir / f"{stem}-preview.png")

        return StickerResult(
            outputs=outputs, frames=len(rgba), fps=fps,
            width=out_size[0], height=out_size[1],
            mode=mode, key=key, preview=preview,
        )
    finally:
        if keep_work:
            print(f"[vidsticker] work dir kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
