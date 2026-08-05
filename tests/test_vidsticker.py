"""Tests for vidsticker.

The end-to-end tests build a synthetic green-screen clip and run it through in
`chroma` mode, so they need ffmpeg but no model download and no network.
"""

from __future__ import annotations

import subprocess
import threading
import time
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


def test_screen_rgb_reads_the_backdrop():
    rgb = np.asarray(make_frame(0))
    alpha = _perfect_ai_mask()
    assert matting.screen_rgb(rgb, alpha).tolist() == pytest.approx(list(SCREEN), abs=1)


def test_screen_rgb_is_none_without_enough_backdrop():
    rgb = np.asarray(make_frame(0))
    assert matting.screen_rgb(rgb, np.ones((H, W), np.float32)) is None


def test_unmix_recovers_the_subject_colour_from_a_half_covered_pixel():
    screen = np.array(SCREEN, np.float32)
    subject = np.array(SUBJECT, np.float32)
    # Exactly what a 50%-covered edge pixel looks like.
    observed = (0.5 * subject + 0.5 * screen).astype(np.uint8).reshape(1, 1, 3)
    alpha = np.full((1, 1), 0.5, np.float32)

    out = matting.unmix_screen(observed, alpha, screen)[0, 0]
    assert out == pytest.approx(subject, abs=2)

    # Despill only clamps a channel, so it cannot get back there.
    approx = matting.despill(observed, channel=1, amount=1.0)[0, 0]
    assert np.abs(approx - subject).max() > np.abs(out - subject).max()


def test_unmix_leaves_a_fully_covered_pixel_alone():
    screen = np.array(SCREEN, np.float32)
    px = np.array(SUBJECT, np.uint8).reshape(1, 1, 3)
    out = matting.unmix_screen(px, np.ones((1, 1), np.float32), screen)[0, 0]
    assert out == pytest.approx(np.array(SUBJECT, np.float32), abs=1)


def test_unmix_amount_scales_the_correction():
    screen = np.array(SCREEN, np.float32)
    observed = np.array([[[111, 136, 52]]], np.uint8)
    alpha = np.full((1, 1), 0.5, np.float32)
    full = matting.unmix_screen(observed, alpha, screen, amount=1.0)[0, 0]
    half = matting.unmix_screen(observed, alpha, screen, amount=0.5)[0, 0]
    none = matting.unmix_screen(observed, alpha, screen, amount=0.0)[0, 0]
    assert none == pytest.approx(observed[0, 0], abs=1)
    assert np.all(np.abs(half - observed[0, 0]) < np.abs(full - observed[0, 0]))


def test_temporal_despeckle_kills_a_one_frame_spike():
    zero, one = np.zeros((4, 4), np.float32), np.ones((4, 4), np.float32)
    assert matting.temporal_despeckle(zero, one, zero).max() == 0.0
    # With a neighbour missing there is nothing to vote against, so it passes through.
    assert matting.temporal_despeckle(None, one, zero).max() == 1.0


def test_temporal_despeckle_never_adds_coverage():
    """The failure mode this replaced: neighbours out-voting the current frame.

    On fast motion the frames either side agree about where a limb is while the
    current frame disagrees, and a plain median paints that limb onto this
    frame's backdrop.
    """
    zero, one = np.zeros((4, 4), np.float32), np.ones((4, 4), np.float32)
    assert matting.temporal_despeckle(one, zero, one).max() == 0.0
    assert np.median(np.stack([one, zero, one]), axis=0).max() == 1.0   # a median would

    # It also never exceeds the current frame anywhere, for any input.
    rng = np.random.default_rng(0)
    p, c, n = (rng.random((16, 16)).astype(np.float32) for _ in range(3))
    assert np.all(matting.temporal_despeckle(p, c, n) <= c + 1e-6)


def test_resize_rgba_does_not_drag_the_backdrop_into_the_edge():
    """Downscaling must not tint opaque pixels with transparent neighbours."""
    arr = np.zeros((32, 32, 4), np.uint8)
    arr[..., :3] = SCREEN            # transparent pixels still hold screen colour
    arr[5:27, 5:27, :3] = SUBJECT
    arr[5:27, 5:27, 3] = 255

    def worst_shift(img):
        """How far the kept pixels drifted from the subject's true colour."""
        kept = img[..., 3] > 128
        assert kept.any()
        return np.abs(img[..., :3][kept].astype(float) - np.array(SUBJECT, float)).max()

    out = matting.resize_rgba(arr, (11, 11))
    naive = np.asarray(Image.fromarray(arr, "RGBA").resize((11, 11), Image.LANCZOS))

    assert worst_shift(out) == 0.0            # nothing but subject in the kept pixels
    assert worst_shift(naive) > 20            # straight-alpha pulls the screen in


def test_resize_rgba_preserves_size_and_alpha():
    arr = np.zeros((10, 20, 4), np.uint8)
    arr[..., 3] = 255
    out = matting.resize_rgba(arr, (10, 5))
    assert out.shape == (5, 10, 4)
    assert out[..., 3].min() == 255


def test_content_bbox_is_tight():
    a = np.zeros((50, 60), np.float32)
    a[10:20, 30:45] = 1.0
    assert matting.content_bbox(a) == (30, 10, 45, 20)


def test_content_bbox_of_empty_matte_is_none():
    assert matting.content_bbox(np.zeros((10, 10), np.float32)) is None


def _perfect_ai_mask() -> np.ndarray:
    """The matte the network would produce for make_frame(0), exactly."""
    m = np.zeros((H, W), np.float32)
    m[70:130, 30:80] = 1.0
    return m


def test_calibrate_ramp_reaches_opacity_where_the_subject_begins():
    rgb = np.asarray(make_frame(0))
    key = matting.detect_key(rgb)
    tolerance, softness = matting.calibrate_ramp(rgb, key, _perfect_ai_mask())

    ycc = matting._ycrcb(rgb)
    dist = np.hypot(ycc[..., 1] - key[0], ycc[..., 2] - key[1])
    subject = float(np.percentile(dist[70:130, 30:80], 1))

    assert tolerance >= matting.MIN_TOLERANCE
    # The ramp must top out right at the subject, not before it: saturating early
    # is what keeps the backdrop's half of a blurred edge. (This synthetic subject
    # is orange against green — far enough apart to hit the softness ceiling.)
    ceiling = tolerance + matting.MAX_SOFTNESS
    assert tolerance + softness == pytest.approx(min(subject, ceiling), abs=1.0)


def test_calibrate_ramp_stays_tight_when_the_subject_resembles_the_screen():
    """A subject close to the key in colour must not widen the ramp onto itself."""
    near_screen = (20, 170, 70)        # a hair away from SCREEN in chroma
    rgb = np.asarray(make_frame(0)).copy()
    rgb[70:130, 30:80] = near_screen

    key = matting.detect_key(rgb)
    tolerance, softness = matting.calibrate_ramp(rgb, key, _perfect_ai_mask())
    assert softness <= matting.MIN_SOFTNESS + 1
    assert tolerance + softness < 60      # nowhere near the wide ramp above


def test_calibrate_ramp_declines_without_enough_of_either_side():
    rgb = np.asarray(make_frame(0))
    key = matting.detect_key(rgb)
    assert matting.calibrate_ramp(rgb, key, np.ones((H, W), np.float32)) is None
    assert matting.calibrate_ramp(rgb, key, np.zeros((H, W), np.float32)) is None


def test_calibrated_ramp_drops_a_half_blended_edge_pixel(monkeypatch):
    """A 50/50 mix of screen and subject must not come out opaque."""
    rgb = np.asarray(make_frame(0)).copy()
    blend = ((np.array(SCREEN, float) + np.array(SUBJECT, float)) / 2).astype(np.uint8)
    rgb[100, 85:95] = blend                      # a blurred edge beside the square

    monkeypatch.setattr(matting, "ai_alpha", lambda s, r: _perfect_ai_mask_with_edge())
    a = matting.compute_alpha(rgb, MatteConfig(mode="hybrid"), session=object(),
                              key=matting.detect_key(rgb))
    assert a[100, 90] < 0.7                      # the blend is not full subject
    assert a[100, 60] == 1.0                     # the subject still is

    # The narrow fixed ramp this replaced wrote the same pixel out fully opaque,
    # which is the halo that showed up along motion-blurred edges.
    tight = matting.compute_alpha(rgb, MatteConfig(mode="hybrid", tolerance=8, softness=20),
                                  session=object(), key=matting.detect_key(rgb))
    assert tight[100, 90] == 1.0


def _perfect_ai_mask_with_edge() -> np.ndarray:
    m = _perfect_ai_mask()
    m[100, 85:95] = 1.0
    return m


def test_explicit_ramp_values_override_the_calibration(monkeypatch):
    rgb = np.asarray(make_frame(0)).copy()
    blend = ((np.array(SCREEN, float) + np.array(SUBJECT, float)) / 2).astype(np.uint8)
    rgb[100, 85:95] = blend
    key = matting.detect_key(rgb)
    monkeypatch.setattr(matting, "ai_alpha", lambda s, r: _perfect_ai_mask_with_edge())

    auto = matting.compute_alpha(rgb, MatteConfig(mode="hybrid"), session=object(), key=key)

    fixed = MatteConfig(mode="hybrid", tolerance=8.0, softness=20.0)
    pinned = matting.compute_alpha(rgb, fixed, session=object(), key=key)
    assert np.allclose(pinned[100, 85:95], matting.chroma_alpha(rgb, key, 8.0, 20.0)[100, 85:95])

    # Setting only one end still leaves the other calibrated.
    half = matting.compute_alpha(rgb, MatteConfig(mode="hybrid", softness=20.0),
                                 session=object(), key=key)
    assert half[100, 90] != auto[100, 90]


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="tolerance must be"):
        MatteConfig(tolerance=-1).validate()


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


def test_no_screen_colour_survives_in_the_output(clip, tmp_path):
    """Nothing recognisably backdrop-coloured may ship as an opaque pixel.

    The source is h264, so its edges carry real screen/subject blend pixels -
    the same thing motion blur produces, just narrower.
    """
    opts = StickerOptions(formats=["gif"], size=128, preview=False,
                          matte=MatteConfig(mode="chroma"))
    res = create_sticker(clip, tmp_path, opts, stem="green")

    leaked = 0
    for f in ImageSequence.Iterator(Image.open(res.outputs["gif"])):
        a = np.asarray(f.convert("RGBA")).astype(int)
        green_led = a[..., 1] - np.maximum(a[..., 0], a[..., 2])
        leaked += int(((green_led > 25) & (a[..., 3] > 128)).sum())
    assert leaked == 0


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


# --- web app ---------------------------------------------------------------

@pytest.fixture
def client():
    from vidsticker import web
    web.JOBS.clear()
    return web.create_app().test_client()


def _upload(client, clip, **fields):
    data = {"mode": "chroma", "size": "96", "formats": "gif",
            "video": (clip.open("rb"), "my clip.mp4")}
    data.update(fields)
    return client.post("/api/jobs", data=data, content_type="multipart/form-data")


def test_index_renders(client):
    assert client.get("/").status_code == 200


def test_upload_without_a_file_is_a_400(client):
    assert client.post("/api/jobs", data={}).get_json()["error"]


def test_bad_option_is_rejected_before_any_work(client, clip):
    r = _upload(client, clip, mode="nonsense")
    assert r.status_code == 400
    assert "mode must be one of" in r.get_json()["error"]


def test_upload_converts_and_serves_the_result(client, clip):
    from vidsticker import web
    job = _upload(client, clip).get_json()
    for _ in range(600):
        state = client.get(f"/api/jobs/{job['id']}").get_json()
        if state["state"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert state["state"] == "done", state.get("error")
    assert state["progress"] == 1.0

    files = state["result"]["files"]
    assert [f["format"] for f in files] == ["gif"]
    body = client.get(files[0]["url"])
    assert body.status_code == 200 and body.data[:6] == b"GIF89a"


def test_downloads_cannot_escape_the_job_directory(client, tmp_path):
    from vidsticker import web
    job = web.Job(id="esc", name="n", dir=tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "secret.txt").write_text("no")
    web.JOBS["esc"] = job
    assert client.get("/api/jobs/esc/file/secret.txt").status_code == 404
    assert client.get("/api/jobs/esc/file/..%2Fsecret.txt").status_code == 404


def test_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/events").status_code == 404


def test_jobs_beyond_the_concurrency_cap_wait_their_turn(client, clip, monkeypatch):
    """A busy host must queue extra uploads, not run them all at once."""
    from vidsticker import web

    running = []
    peak = 0
    lock = threading.Lock()

    def slow(*a, **kw):
        nonlocal peak
        with lock:
            running.append(1)
            peak = max(peak, len(running))
        time.sleep(0.4)
        with lock:
            running.pop()
        raise RuntimeError("stopped before encoding")   # we only care about scheduling

    monkeypatch.setattr(web, "create_sticker", slow)
    monkeypatch.setattr(web, "_slots", threading.Semaphore(2))

    jobs = [_upload(client, clip).get_json() for _ in range(5)]
    deadline = time.time() + 30
    while time.time() < deadline:
        states = [client.get(f"/api/jobs/{j['id']}").get_json()["state"] for j in jobs]
        if all(s in ("done", "error") for s in states):
            break
        time.sleep(0.1)
    assert peak <= 2, f"{peak} conversions ran at once despite a cap of 2"
    assert len(jobs) == 5


def test_work_directory_is_cleaned_up(clip, tmp_path):
    import tempfile
    before = set(Path(tempfile.gettempdir()).glob("vidsticker-*"))
    opts = StickerOptions(formats=["gif"], size=96, preview=False,
                          matte=MatteConfig(mode="chroma"))
    create_sticker(clip, tmp_path, opts, stem="clean")
    assert set(Path(tempfile.gettempdir()).glob("vidsticker-*")) == before
