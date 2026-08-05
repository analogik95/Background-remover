"""Tests for vidsticker.

The end-to-end tests build a synthetic green-screen clip and run it through in
`chroma` mode, so they need ffmpeg but no model download and no network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence

from vidsticker import encoders, matting, video
from vidsticker.matting import MatteConfig
from vidsticker.pipeline import StickerOptions, create_sticker

W, H, N, FPS = 160, 200, 12, 20
SCREEN = (0, 177, 64)
SUBJECT = (222, 96, 40)


def make_frame(i: int, screen=SCREEN) -> Image.Image:
    """Green screen with an orange square tracking across it."""
    img = Image.new("RGB", (W, H), screen)
    a = np.asarray(img).copy()
    x = 30 + i * 4
    a[70:130, x:x + 50] = SUBJECT
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("clip")
    for i in range(N):
        make_frame(i).save(d / f"{i:05d}.png")
    out = d / "clip.mp4"
    subprocess.run(
        [video.ffmpeg_bin(), "-y", "-v", "error", "-framerate", str(FPS),
         "-i", str(d / "%05d.png"), "-pix_fmt", "yuv420p", "-crf", "12", str(out)],
        check=True, capture_output=True,
    )
    return out


# --- matting units --------------------------------------------------------

def test_detect_key_finds_the_screen():
    key = matting.detect_key(np.asarray(make_frame(0)))
    assert key is not None
    ycc = matting._ycrcb(np.full((4, 4, 3), SCREEN, dtype=np.uint8))
    assert key == pytest.approx((ycc[0, 0, 1], ycc[0, 0, 2]), abs=2)


def test_detect_key_declines_on_a_plain_backdrop():
    grey = np.full((80, 80, 3), 130, dtype=np.uint8)
    assert matting.detect_key(grey) is None


def test_detect_key_declines_on_a_busy_backdrop():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    assert matting.detect_key(noise) is None


def test_chroma_alpha_separates_subject_from_screen():
    rgb = np.asarray(make_frame(0))
    key = matting.detect_key(rgb)
    a = matting.chroma_alpha(rgb, key, tolerance=8, softness=20)
    assert a[10, 10] == 0.0          # screen
    assert a[100, 60] == 1.0         # subject


def test_chroma_alpha_ramps_between_tolerance_and_softness():
    key = (100.0, 100.0)
    # Construct pixels at known chroma distances by walking Cr away from the key.
    ycc = np.zeros((1, 3, 3), np.uint8)
    ycc[0, :, 0] = 128
    ycc[0, :, 1] = [100, 118, 160]   # distance 0, 18, 60
    ycc[0, :, 2] = 100
    import cv2
    rgb = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    a = matting.chroma_alpha(rgb, key, tolerance=8, softness=20)[0]
    assert a[0] == 0.0
    assert 0.0 < a[1] < 1.0
    assert a[2] == 1.0


def test_despill_pulls_green_back_without_touching_neutrals():
    px = np.array([[[40, 200, 60], [128, 128, 128]]], dtype=np.uint8)
    out = matting.despill(px, channel=1, amount=1.0)
    assert out[0, 0, 1] == pytest.approx(60)          # clamped to max(r, b)
    assert out[0, 1].tolist() == [128, 128, 128]      # unchanged


def test_despill_off_is_a_no_op():
    px = np.array([[[40, 200, 60]]], dtype=np.uint8)
    assert matting.despill(px, amount=0.0).tolist() == px.astype(np.float32).tolist()
    assert matting.despill(px, channel=None).tolist() == px.astype(np.float32).tolist()


@pytest.mark.parametrize("screen, channel", [
    ((0, 177, 64), 1),      # green screen
    ((0, 71, 187), 2),      # blue screen
    ((200, 20, 20), 0),     # red screen
])
def test_spill_channel_follows_the_screen_colour(screen, channel):
    key = matting.detect_key(np.asarray(make_frame(0, screen=screen)))
    assert key is not None
    assert matting.spill_channel(key) == channel


def test_spill_channel_is_none_without_a_key():
    assert matting.spill_channel(None) is None


def test_temporal_median_kills_a_one_frame_spike():
    zero, one = np.zeros((4, 4), np.float32), np.ones((4, 4), np.float32)
    assert matting.temporal_median(zero, one, zero).max() == 0.0
    # With a neighbour missing there is nothing to vote against, so it passes through.
    assert matting.temporal_median(None, one, zero).max() == 1.0


def test_content_bbox_is_tight():
    a = np.zeros((50, 60), np.float32)
    a[10:20, 30:45] = 1.0
    assert matting.content_bbox(a) == (30, 10, 45, 20)


def test_content_bbox_of_empty_matte_is_none():
    assert matting.content_bbox(np.zeros((10, 10), np.float32)) is None


def test_compute_alpha_hybrid_takes_the_stricter_of_the_two(monkeypatch):
    rgb = np.asarray(make_frame(0))
    key = matting.detect_key(rgb)

    # A neural matte that claims everything is subject must not override the key.
    monkeypatch.setattr(matting, "ai_alpha", lambda s, r: np.ones(r.shape[:2], np.float32))
    a = matting.compute_alpha(rgb, MatteConfig(mode="hybrid"), session=object(), key=key)
    assert a[10, 10] == 0.0
    assert a[100, 60] == 1.0


def test_compute_alpha_ai_ignores_the_screen_colour(monkeypatch):
    rgb = np.asarray(make_frame(0))
    monkeypatch.setattr(matting, "ai_alpha", lambda s, r: np.ones(r.shape[:2], np.float32))
    a = matting.compute_alpha(rgb, MatteConfig(mode="ai"), session=object(),
                              key=matting.detect_key(rgb))
    assert a.min() == 1.0


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        MatteConfig(mode="nonsense").validate()


# --- encoder units --------------------------------------------------------

def test_gif_fps_snapping():
    assert encoders.gif_timing_error(25) == 0.0
    assert encoders.gif_timing_error(24) > 0.02
    assert encoders.nearest_gif_fps(24) == 25.0
    assert encoders.nearest_gif_fps(30) == 25.0


def test_durations_sum_to_the_true_length():
    d = encoders._durations(30, 24)
    assert sum(d) == 1250            # 30 frames at 24 fps
    assert set(d) <= {41, 42}        # rounding spread across frames, not accumulated


# --- end to end -----------------------------------------------------------

def test_probe_reads_the_clip(clip):
    info = video.probe(clip)
    assert (info.width, info.height) == (W, H)
    assert info.fps == pytest.approx(FPS, abs=0.5)


def test_end_to_end_produces_a_transparent_cropped_gif(clip, tmp_path):
    opts = StickerOptions(formats=["gif", "webp"], size=128, smooth=False,
                          preview=False, matte=MatteConfig(mode="chroma"))
    res = create_sticker(clip, tmp_path, opts, stem="s")

    assert res.mode == "chroma"
    assert res.frames == N
    assert set(res.outputs) == {"gif", "webp"}
    assert all(p.stat().st_size > 0 for p in res.outputs.values())

    gif = Image.open(res.outputs["gif"])
    assert gif.n_frames == N
    assert "transparency" in gif.info

    frames = [f.convert("RGBA") for f in ImageSequence.Iterator(gif)]
    alpha = np.asarray(frames[0])[..., 3]
    assert alpha[0, 0] == 0                                  # corner cut away
    assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255   # subject kept
    # Every frame keeps its own silhouette rather than accumulating a trail.
    coverage = [np.asarray(f)[..., 3].mean() for f in frames]
    assert max(coverage) - min(coverage) < 0.15 * max(coverage)

    # Cropping tracks the square as it moves, so the canvas spans its full travel.
    assert res.width <= 128 and res.height <= 128
    assert res.width > res.height     # subject sweeps horizontally


def test_square_option_pads_to_a_square_canvas(clip, tmp_path):
    opts = StickerOptions(formats=["gif"], size=96, square=True, smooth=False,
                          preview=False, matte=MatteConfig(mode="chroma"))
    res = create_sticker(clip, tmp_path, opts, stem="sq")
    assert res.width == res.height


def test_square_stays_square_when_the_subject_overflows_the_frame(clip, tmp_path):
    # Without cropping, the square canvas has to extend past the 160x200 source.
    opts = StickerOptions(formats=["gif"], size=96, square=True, crop=False,
                          smooth=False, preview=False, matte=MatteConfig(mode="chroma"))
    res = create_sticker(clip, tmp_path, opts, stem="sqfull")
    assert res.width == res.height


def test_colors_above_the_palette_ceiling_is_rejected():
    with pytest.raises(ValueError, match="colors must be"):
        StickerOptions(colors=256).validate()


def test_no_crop_keeps_the_source_framing(clip, tmp_path):
    opts = StickerOptions(formats=["gif"], size=1000, crop=False, smooth=False,
                          preview=False, matte=MatteConfig(mode="chroma"))
    res = create_sticker(clip, tmp_path, opts, stem="full")
    assert (res.width, res.height) == (W, H)


def test_trimming_shortens_the_clip(clip, tmp_path):
    opts = StickerOptions(formats=["gif"], size=96, duration=0.3, smooth=False,
                          preview=False, matte=MatteConfig(mode="chroma"))
    res = create_sticker(clip, tmp_path, opts, stem="trim")
    assert 0 < res.frames < N


def test_chroma_mode_without_a_key_fails_clearly(tmp_path):
    d = tmp_path / "grey"
    d.mkdir()
    for i in range(4):
        Image.new("RGB", (W, H), (130, 130, 130)).save(d / f"{i:05d}.png")
    clip = tmp_path / "grey.mp4"
    subprocess.run(
        [video.ffmpeg_bin(), "-y", "-v", "error", "-framerate", str(FPS),
         "-i", str(d / "%05d.png"), "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True,
    )
    opts = StickerOptions(formats=["gif"], preview=False, matte=MatteConfig(mode="chroma"))
    with pytest.raises(ValueError, match="no key colour"):
        create_sticker(clip, tmp_path / "o", opts)


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="unknown output format"):
        StickerOptions(formats=["mp4"]).validate()


def test_progress_reports_every_stage(clip, tmp_path):
    seen = []
    opts = StickerOptions(formats=["gif"], size=96, preview=False,
                          matte=MatteConfig(mode="chroma"))
    create_sticker(clip, tmp_path, opts, stem="p",
                   progress=lambda s, d, t: seen.append(s))
    assert {"probe", "decode", "matte", "analyse", "compose", "encode"} <= set(seen)


def test_work_directory_is_cleaned_up(clip, tmp_path):
    import tempfile
    before = set(Path(tempfile.gettempdir()).glob("vidsticker-*"))
    opts = StickerOptions(formats=["gif"], size=96, preview=False,
                          matte=MatteConfig(mode="chroma"))
    create_sticker(clip, tmp_path, opts, stem="clean")
    assert set(Path(tempfile.gettempdir()).glob("vidsticker-*")) == before
