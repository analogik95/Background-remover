# vidsticker (Android)

A fully on-device port of the desktop `vidsticker` tool: pick a video, cut the
background out, get back a transparent animated GIF — no server, no upload,
works offline once the matting model is on the device.

## Why on-device is a different shape than the desktop tool

The desktop pipeline decodes the whole clip, holds every frame's alpha and
RGBA in memory, and hands frames to ffmpeg's palette filters for the GIF. None
of that fits a phone's memory budget, so several things are rebuilt rather
than ported line-for-line:

- **Two decode passes over disk-backed intermediates, not one pass in memory.**
  Pass 1 decodes each frame, runs the matting model, and writes an 8-bit alpha
  mask to a cache file. Pass 1.5 walks those files as a 3-wide sliding window
  to despeckle and find the crop box the whole clip needs. Pass 2 re-decodes,
  unmixes the screen colour back out, crops, resizes, and streams straight
  into the GIF encoder. Peak memory is one frame, not the whole clip — the
  trade is decoding the video twice, which is cheap next to running the
  matting model once per frame.
- **A hand-written GIF encoder**, quantizing each frame's palette independently
  (median-cut) rather than one palette shared by the whole clip via ffmpeg.
  Costs a little file size; means the encoder never holds more than one frame.
- **`MediaMetadataRetriever.getFrameAtTime`, not a MediaCodec+Surface decode
  loop.** Slower in principle, but a Surface decode's colour format varies by
  device/encoder in ways that are a frequent source of subtly wrong pixels;
  `getFrameAtTime` always hands back correct ARGB. Not the bottleneck here
  regardless — matting one frame costs far more than decoding it.
- **The matting model is downloaded on first use, not bundled.** At ~175 MB
  each, shipping isnet-general-use *and* isnet-anime in the APK would roughly
  double install size for checkpoints most people only need one of. Fetched
  from the same public release the desktop tool's `rembg` dependency uses, via
  a foreground service so a download this size survives the user switching
  apps — an ordinary background coroutine gets killed by Doze within minutes.

Everything else — the hybrid matte (chroma key `min`'d with the neural mask),
per-frame ramp calibration, screen unmixing, clamped-median despeckling,
premultiplied-alpha resize — is a direct port of the same math, in
`matting/Matting.kt`. See the desktop README for why each of those exists;
the reasoning doesn't change on a phone, only the plumbing around it does.

## Requirements

- Android Studio Koala+ / a JDK 17 toolchain
- Android SDK platform 34, build-tools 34.0.0
- A device or emulator on **API 26+** (adaptive icons, modern Bitmap APIs);
  animated GIF *preview* inside the app needs **API 28+**
  ([`AnimatedImageDrawable`](https://developer.android.com/reference/android/graphics/drawable/AnimatedImageDrawable));
  below that the app still produces a correct file, it just shows the first
  frame in the result card.

## Build

```bash
./gradlew assembleDebug      # app/build/outputs/apk/debug/app-debug.apk
./gradlew testDebugUnitTest  # matting math + GIF encoder, no device needed
```

The debug APK bundles ONNX Runtime's native libraries for all four ABIs
(~80 MB unstripped). A release build should split by ABI or ship an App
Bundle instead — that wasn't wired up here.

## Architecture

```
matting/
  Matting.kt          pure math: chroma key, ramp calibration, unmixing,
                       despeckling, premultiplied resize - zero Android deps,
                       unit tested directly on the JVM (MattingTest.kt)
  MattingModel.kt      ONNX Runtime wrapper; preprocessing ported line-for-line
                       from rembg's BaseSession.normalize/DisSession.predict
video/
  VideoFrameExtractor.kt   MediaMetadataRetriever-based frame decode
encode/
  GifLzwEncoder.kt     GIF's variable-width LZW compressor
  MedianCutQuantizer.kt   per-frame palette quantization
  GifEncoder.kt        GIF89a container: NETSCAPE loop extension, graphic
                       control extension (disposal=2, one transparent index),
                       local colour table per frame
model/
  ModelManager.kt          download + MD5 verification against rembg's
                           published checksums
  ModelDownloadService.kt  foreground service wrapping the download
pipeline/
  StickerPipeline.kt   orchestrates the passes described above
ui/
  StickerViewModel.kt, MainScreen.kt   Compose UI: picker, options, progress,
                                       result preview + save/share
```

## What's verified vs. what isn't

Verified in this environment (no device or emulator was available to run the
actual app):

- The full Gradle build compiles and packages a working APK, with ONNX
  Runtime's native libraries present for all four ABIs.
- `Matting.kt`'s math runs as plain JUnit tests on the JVM (27 tests), mirroring
  the desktop test suite: chroma key detection accepts/declines correctly,
  the ramp calibration doesn't saturate before the subject, screen unmixing
  recovers subject colour from a half-covered pixel, temporal despeckling
  never adds coverage beyond the current frame, premultiplied resize doesn't
  bleed backdrop colour into an opaque edge.
- The GIF encoder was round-tripped through two independent decoders before
  ever compiling into the app: Pillow, via a throwaway `kotlinc` script, and a
  from-scratch reference decoder now living in `GifEncoderTest.kt`. That
  caught a real bug worth naming: the encoder's code-width growth trigger
  (`nextCode >= 1 shl codeSize`) was one insertion too early relative to any
  standard decoder, which any test frame with more than a couple hundred
  colours corrupts on decode. Small, flat-colour test frames never reached
  that code path and passed regardless — the fix and the reasoning are in
  `GifLzwEncoder.kt`'s doc comment, and the regression tests specifically use
  enough random colour to force both that boundary and a mid-stream table
  reset past 4096 codes.
- The exact preprocessing constants (resize to 1024×1024, normalize by the
  frame's own max pixel value, per-checkpoint channel means, min-max output
  normalization) were confirmed against `rembg`'s actual source, and the MD5
  checksums in `ModelManager.kt` were checked against the real downloaded
  model files.

Not verified — there was no Android device/emulator in this environment:

- The app has never actually run. Video picking, the download service,
  end-to-end conversion of a real video, and the GIF preview/save/share flow
  are implemented against the documented APIs but untested at runtime.
- On-device performance is unmeasured. Budget considerably more than the
  desktop tool's ~1s/frame/core — no GPU delegate is configured beyond a
  best-effort NNAPI attempt that silently falls back to CPU if unavailable.
- **Notifications require a runtime permission on Android 13+ that this app
  never requests.** The download still runs either way; only the progress
  notification would silently fail to appear. Add a
  `POST_NOTIFICATIONS` permission request before shipping.

## Known gaps

- **GIF only.** No WebP or APNG — animated WebP needs libwebp via JNI, and
  wiring up NDK/native bindings was out of scope here. The `Encoder`-shaped
  boundary in `pipeline/StickerPipeline.kt` is where a second encoder would
  plug in.
- **No trim / start-time controls**, unlike the desktop CLI's `--start`/`--duration`.
- **No advanced matting overrides** (tolerance/softness/despill) in the UI —
  the pipeline supports them via `MatteConfig`, just not exposed as controls.
