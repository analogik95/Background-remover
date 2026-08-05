# vidsticker ✂️

Turn a video into a transparent animated sticker — GIF, WebP or APNG — with the
background cut away.

Comes as a CLI and a small browser UI.

```bash
vidsticker clip.mp4            # -> out/clip.gif + out/clip.webp
vidsticker --serve             # -> browser UI on http://127.0.0.1:8000
```

<p align="center">
  <img src="examples/luffy-sticker-preview.png" width="820" alt="Frames of the finished sticker on a checkerboard">
</p>

---

## How the background removal works

Most tools pick one of two strategies. Each fails on its own:

| | edges | knows what the subject is |
|---|---|---|
| **Chroma key** | pixel-accurate | ✗ — keeps *anything* that isn't the screen colour: light rays, lens flares, sparkle overlays |
| **Neural matte** | soft, approximate | ✓ |

So vidsticker runs both and takes the **per-pixel minimum** of the two alphas.
The key defines the edge; the network defines what counts as subject. A pixel
has to convince both to survive.

The rest of the pipeline exists because real footage breaks the naive version of
that idea:

- **The key is re-detected on every frame.** An animated backdrop — light rays,
  a flicker, a pulsing gradient — shifts the screen colour as the clip runs. One
  clip-wide key leaves patches of backdrop behind on the frames it doesn't fit.
- **The key's ramp is fitted to the footage, not guessed.** How far the subject's
  colours sit from the key depends on the clip, so vidsticker measures it: the
  neural matte labels confident foreground and background, and the ramp is set to
  clear the screen's own spread and reach full opacity right where the subject's
  colours begin. This is what makes **motion blur** work. A blurred edge is a
  genuine mix of subject and screen and lands midway between them in colour; a
  ramp that saturates before that midpoint writes every blurred edge out as
  fully opaque backdrop.
- **The screen is unmixed out of partly-covered pixels, not just despilled.** An
  antialiased outline — and far more of a motion-blurred one — is literally
  `C = α·F + (1−α)·S`. Solving that for `F` removes exactly the screen's share.
  Despill only approximates it by clamping a channel, which on a half-screen
  blurred edge is the difference between a green fringe and none.
- **Temporal despeckling never adds coverage.** A per-frame network makes pixels
  blink opaque for a single frame, and a 3-frame median removes that — but a
  plain median also *invents* coverage. Where a limb moves fast, the frames
  either side agree with each other while the current frame disagrees, so the
  median paints the limb's other position onto this frame's backdrop. Clamping
  the median to the current frame keeps the despeckling and drops the invention.
- **Downscaling happens in premultiplied alpha.** Resampling colour and alpha
  independently mixes transparent pixels — which still hold the screen's colour —
  into their opaque neighbours, re-tinting the edge on the way to output size.
- **One crop box for the whole clip**, taken from the union of every frame's
  content, so the sticker is tight but the subject never drifts or jitters
  inside the frame.

On a source without a coloured screen, `auto` detects that there's no key and
falls back to the neural matte alone.

## Install

```bash
pip install -r requirements.txt        # or: pip install -e .
```

Needs **ffmpeg** on `PATH` (`apt install ffmpeg` / `brew install ffmpeg`).
Segmentation models download themselves on first use (~180 MB, cached in
`~/.u2net`).

## CLI

```bash
vidsticker clip.mp4                          # GIF + WebP, auto matte, 512 px
vidsticker clip.mp4 -o stickers --size 320
vidsticker clip.mp4 --formats gif webp apng --square
vidsticker clip.mp4 --mode ai                # ignore the backdrop colour
vidsticker clip.mp4 --start 2 --duration 4   # trim
```

| flag | what it does |
|---|---|
| `--size N` | longest edge in px (default 512) |
| `--fps N` | output rate; defaults to the source, snapped for GIF (see below) |
| `--formats` | any of `gif webp apng` (default `gif webp`) |
| `--mode` | `auto` · `hybrid` · `chroma` · `ai` |
| `--model` | `isnet-general-use` (default), `isnet-anime`, `u2net_human_seg`, `birefnet-general`, … |
| `--square` | pad the crop to a square canvas |
| `--no-crop` | keep the original framing |
| `--tolerance` / `--softness` | chroma radius kept transparent / width of the edge ramp (both fitted per frame by default) |
| `--despill` | strength of the screen-colour removal, `0`–`1` (default 0.8) |
| `--shrink` / `--feather` | erode / blur the matte edge |
| `--no-smooth` | disable the temporal despeckling |
| `--alpha-threshold` | GIF alpha cutoff, `1`–`255` (default 128) |

`vidsticker --help` lists the rest.

### Picking a model

`isnet-general-use` is a good default. Switch when the subject is unusual:

- **`isnet-anime`** — illustration and anime. Noticeably better at *interior*
  gaps: the slot between an arm and a sleeve, gaps between fingers. Roughly 3×
  slower.
- **`u2net_human_seg`** — a person filling the frame.
- **`birefnet-general`** — the best edges available here, and the slowest by a
  wide margin.

## Browser UI

```bash
vidsticker --serve --port 8000
```

Drag a video in, watch the stages stream past, download the result. Conversions
run on a worker thread and report progress over server-sent events, so a
multi-minute job stays responsive.

`--serve` is Flask's development server. To put it on the internet, see below.

## Hosting it

The matte runs a segmentation model and shells out to ffmpeg, so this needs a
real server — there's no static-hosting version. The repo ships a `Dockerfile`
with the models baked in, and configs for a few hosts in [`deploy/`](deploy/):

```bash
docker build -t vidsticker .
docker run -p 8000:7860 vidsticker      # http://localhost:8000
```

**Free option: Hugging Face Spaces.** Free CPU Spaces get 2 vCPU and 16 GB RAM —
the only free tier here that clears the memory requirement. Copy
`deploy/huggingface/README.md` over the Space's `README.md` (its YAML header is
what selects the Docker build and the port), then push.

Fly.io (`deploy/fly.toml`) and Render (`deploy/render.yaml`) are set up too.
Skip Render's free tier — 512 MB gets killed loading the model.

Whatever you pick: **2 GB RAM minimum**, budget about a second per frame per
core, and run a **single worker** with threads. Jobs and their progress streams
live in process memory, so a second worker fields requests for jobs it can't
see. [`deploy/DEPLOY.md`](deploy/DEPLOY.md) has the details, including what to
change before exposing it publicly — there is no authentication in front of it.

## Python API

```python
from pathlib import Path
from vidsticker import MatteConfig, StickerOptions, create_sticker

result = create_sticker(
    Path("clip.mp4"), Path("out"),
    StickerOptions(size=512, formats=["gif", "webp"],
                   matte=MatteConfig(mode="auto", model="isnet-anime")),
    progress=lambda stage, done, total: print(stage, done, total),
)
print(result.summary)
```

## Format notes

**GIF carries only one bit of alpha** and 256 colours, so soft edges get
thresholded at `--alpha-threshold`. It's also limited to delays in whole
hundredths of a second — a 24 fps source wants 4.1666 cs and every decoder
rounds it differently, so when GIF is a target vidsticker resamples to the
nearest rate GIF can represent exactly (25 fps here). Clip length is preserved.

**WebP and APNG carry real 8-bit alpha**, so soft edges — and motion blur, which
is just a very wide soft edge — survive instead of being cut to a hard 1-bit
stencil. WebP is smaller than the GIF as well (5.28 MB vs 6.69 MB on the example
below, and the gap widens as you drop `--webp-quality`). APNG is lossless and
much larger. If whatever you're pasting into takes WebP, prefer it — GIF is the
compatibility option, not the good one.

## Worked example

`examples/` holds the output for a 6 s, 1080×1920 clip of an anime character on
a green screen — with animated gold light rays and sparkles drifting across the
backdrop, which is exactly the case a plain chroma key gets wrong.

```bash
vidsticker luffy.mp4 -o examples --model isnet-anime --formats gif webp
```

```
Matte: hybrid (chroma key Cr/Cb 84/101)
Output: 150 frames @ 25 fps, 288x512 px
  gif   examples/luffy-sticker.gif   (6.69 MB)
  webp  examples/luffy-sticker.webp  (5.28 MB)
```

Counting opaque pixels that still read as backdrop-coloured, over every output
frame:

| output | frames | opaque px | backdrop-coloured |
|---|---:|---:|---:|
| GIF | 150 | 8,320,453 | 8,915 (0.107%) |
| WebP | 144 | 7,986,949 | **144 (0.002%)** |

The GIF's remaining tenth of a percent is all on the fast-swinging arms, and it
is the 1-bit alpha, not the matte: motion blur is real partial coverage, and
forcing it through a hard threshold both stipples the arm and keeps the
backdrop's half of the pixels it rounds up. WebP, from the identical frames,
carries the same blur as actual translucency.

What each stage contributed:

- **Chroma key alone** kept every gold ray and sparkle — they aren't green, so
  the key has no reason to drop them.
- **Neural matte alone** dropped the sparkles but left a green fringe hugging
  the silhouette.
- **Per-frame keying** mattered here: the rays wash the backdrop out as the clip
  runs, moving the key from Cr/Cb 59/93 to 86/103. Held at one clip-wide key,
  the opening frames' backdrop stayed 80% opaque.
- **Fitting the ramp** is what made the motion-blur frames work. On this clip the
  background sits ≤11 chroma units from the key, blurred edges land near 29, and
  the subject starts at 61 — so a ramp saturating at 28, as a fixed 8/20 does,
  falls squarely in the blur and writes it out as opaque backdrop.
- **Clamping the temporal median** to the current frame. Left as a plain median,
  the frames either side out-voted the current one wherever an arm was swinging,
  painting its other position onto the backdrop: 18 stray pixels became 15,282.
- **`isnet-anime`** over the default model closed the interior gaps — the slot
  between a sleeve and a forearm, which the general model filled in solid.

A compact variant for messaging apps with size limits — `examples/luffy-sticker-small.gif`,
180×320 at 1.29 MB:

```bash
vidsticker luffy.mp4 --size 320 --fps 12.5 --colors 160
```

## Tuning a stubborn clip

| symptom | try |
|---|---|
| green rim around the subject | raise `--despill`, or `--shrink 1` |
| holes in the subject | lower `--tolerance`; the subject may share the backdrop's colour |
| background survives in patches | `--mode ai`, or a model better matched to the subject |
| background survives in *interior gaps* | a stronger model — `isnet-anime`, `birefnet-general` |
| jagged GIF edges, or motion blur breaking up | lower `--alpha-threshold` (~96), or output WebP |
| file too big | `--size 320`, `--fps 15`, `--colors 128` |
| edges flicker between frames | ensure `--no-smooth` is *not* set |

## Tests

```bash
python -m pytest tests/ -q
```

They build a synthetic green-screen clip and run it end to end in `chroma` mode —
ffmpeg required, no model download, no network.

## License

MIT
