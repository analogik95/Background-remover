"""Thin ffmpeg/ffprobe wrappers for decoding source video into frames."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegMissing(
            "ffmpeg not found on PATH. Install it (apt install ffmpeg / brew install ffmpeg) "
            "or `pip install imageio-ffmpeg` and put its binary on PATH."
        )
    return exe


def ffprobe_bin() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise FFmpegMissing("ffprobe not found on PATH (ships with ffmpeg).")
    return exe


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"no video stream found in {path.name}")
    st = streams[0]

    num, _, den = st.get("r_frame_rate", "0/1").partition("/")
    try:
        fps = float(num) / float(den) if float(den) else 0.0
    except (TypeError, ValueError):
        fps = 0.0

    duration = float(st.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb = int(st.get("nb_frames") or 0) or int(round(fps * duration))
    return VideoInfo(int(st["width"]), int(st["height"]), fps or 25.0, duration, nb)


def extract_frames(src: Path, dest_dir: Path, fps: Optional[float] = None,
                   start: Optional[float] = None, duration: Optional[float] = None,
                   max_side: Optional[int] = None) -> List[Path]:
    """Decode `src` into numbered PNG frames inside `dest_dir`."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg_bin(), "-y", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(src)]

    filters = []
    if fps:
        filters.append(f"fps={fps}")
    if max_side:
        # Downscale only, and only if the long edge exceeds max_side. Matting
        # accuracy tracks input resolution, so we never upscale to reach it.
        filters.append(
            f"scale=w='if(gt(iw,ih),min(iw,{max_side}),-2)':"
            f"h='if(gt(iw,ih),-2,min(ih,{max_side}))':flags=lanczos"
        )
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-fps_mode", "passthrough", "-start_number", "0", str(dest_dir / "%05d.png")]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {src.name}:\n{proc.stderr.strip()}")

    frames = sorted(dest_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {src.name}")
    return frames
