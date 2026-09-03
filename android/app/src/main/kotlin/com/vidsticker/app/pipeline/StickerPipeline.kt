package com.vidsticker.app.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.vidsticker.app.encode.GifEncoder
import com.vidsticker.app.matting.KeyColor
import com.vidsticker.app.matting.MatteConfig
import com.vidsticker.app.matting.MatteMode
import com.vidsticker.app.matting.Matting
import com.vidsticker.app.matting.MattingModel
import com.vidsticker.app.matting.MattingModelKind
import com.vidsticker.app.matting.RgbImage
import com.vidsticker.app.video.VideoFrameExtractor
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class StickerOptions(
    val fps: Double? = null,
    val outputSize: Int = 512,
    val workMaxSide: Int = 768,
    val crop: Boolean = true,
    val pad: Int = 6,
    val square: Boolean = false,
    val alphaThreshold: Int = 128,
    val colors: Int = 255,
    val loopCount: Int = 0,
    val matte: MatteConfig = MatteConfig(),
)

data class StickerResult(
    val outFile: File,
    val frames: Int,
    val fps: Double,
    val width: Int,
    val height: Int,
    val mode: MatteMode,
    val key: KeyColor?,
)

class StickerPipelineException(message: String) : Exception(message)

/**
 * End-to-end video -> transparent sticker pipeline, entirely on-device.
 *
 * A direct port of the desktop tool's approach - same hybrid matte (chroma key
 * `min`'d with a neural mask), same per-frame ramp calibration, same
 * screen-unmixing and clamped-median despeckling - staged through two decode
 * passes and disk-backed intermediates instead of one decode pass and
 * in-memory arrays. That trade only matters here: a phone doesn't have the
 * headroom to hold a few hundred full-resolution RGBA frames at once, but
 * re-decoding a video twice (cheap) instead of once is an easy price to avoid
 * ever holding more than one frame's pixels in memory at a time.
 *
 * Pass 1 decodes each frame, runs the matting model, combines it with the
 * chroma key, and writes the resulting alpha to a cache file. Pass 1.5 walks
 * those files as a 3-wide sliding window to despeckle and to find the crop
 * box that fits every frame's content. Pass 2 re-decodes, unmixes the screen
 * colour back out, crops, resizes, and streams frames straight into the GIF
 * encoder - never holding the whole clip's pixels at once.
 */
class StickerPipeline(private val context: Context) {

    fun interface Progress {
        fun onProgress(stage: String, done: Int, total: Int)
    }

    fun createSticker(
        uri: Uri,
        modelFile: File?,
        modelKind: MattingModelKind,
        opts: StickerOptions,
        outFile: File,
        progress: Progress,
    ): StickerResult {
        val extractor = VideoFrameExtractor(context)
        val workDir = File(context.cacheDir, "sticker-work-${System.currentTimeMillis()}")
        val alphaDir = File(workDir, "alpha").apply { mkdirs() }
        val matteDir = File(workDir, "matte").apply { mkdirs() }

        try {
            progress.onProgress("probe", 0, 1)
            val info = extractor.probe(uri)
            if (info.width <= 0 || info.height <= 0) throw StickerPipelineException("could not read this video")
            val sourceFps = 25.0 // MediaMetadataRetriever exposes no reliable fps; a GIF-safe default anyway.
            val fps = opts.fps ?: VideoFrameExtractor.nearestGifFps(sourceFps)
            progress.onProgress("probe", 1, 1)

            val (workW, workH) = fitLongestEdge(info.width, info.height, opts.workMaxSide)
            val frameCount = estimateFrameCount(info.durationMs, fps)
            if (frameCount <= 0) throw StickerPipelineException("clip is too short to extract a frame")

            // --- key sampling: a handful of frames spread across the clip -----
            val globalKey = sampleGlobalKey(extractor, uri, info.durationMs)
            val mode = when (opts.matte.mode) {
                MatteMode.AUTO -> if (globalKey != null) MatteMode.HYBRID else MatteMode.AI
                else -> opts.matte.mode
            }
            if (mode == MatteMode.CHROMA && globalKey == null) {
                throw StickerPipelineException("no chroma key colour was detected in this clip")
            }
            val runCfg = opts.matte.copy(mode = mode)
            val trackKey = globalKey != null && opts.matte.mode != MatteMode.CHROMA

            val model = if (mode == MatteMode.AI || mode == MatteMode.HYBRID) {
                requireNotNull(modelFile) { "a matting model is required for mode $mode" }
                progress.onProgress("model", 0, 1)
                MattingModel.load(modelFile, modelKind).also { progress.onProgress("model", 1, 1) }
            } else null

            try {
                // --- pass 1: decode + matte, alpha to disk ---------------------
                var actualCount = 0
                extractor.extractFrames(uri, fps, opts.workMaxSide, durationMs = info.durationMs) { i, bitmap ->
                    val img = bitmapToRgbImage(bitmap)
                    val aiAlpha = model?.predictAlpha(bitmap)
                    val frameKey = if (trackKey) Matting.detectKey(img) ?: globalKey else globalKey
                    val alpha = Matting.computeAlpha(img, runCfg, aiAlpha, frameKey)
                    writeAlpha(File(alphaDir, frameName(i)), alpha)
                    actualCount = i + 1
                    progress.onProgress("matte", actualCount, frameCount)
                    true
                }
                if (actualCount == 0) throw StickerPipelineException("no frames could be decoded from this clip")

                // --- pass 1.5: despeckle + crop-box union ----------------------
                var box: IntArray? = null
                for (i in 0 until actualCount) {
                    val prev = if (i > 0) readAlpha(File(alphaDir, frameName(i - 1)), workW * workH) else null
                    val cur = readAlpha(File(alphaDir, frameName(i)), workW * workH)
                    val next = if (i < actualCount - 1) readAlpha(File(alphaDir, frameName(i + 1)), workW * workH) else null
                    val despeckled = Matting.temporalDespeckle(prev, cur, next)
                    writeAlpha(File(matteDir, frameName(i)), despeckled)

                    val b = Matting.contentBoundingBox(despeckled, workW, workH)
                    if (b != null) {
                        box = if (box == null) b else intArrayOf(
                            min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]),
                        )
                    }
                    progress.onProgress("analyse", i + 1, actualCount)
                }
                val contentBox = box ?: throw StickerPipelineException(
                    "no subject was found in this clip - try a different matte mode",
                )

                val region = if (opts.crop) contentBox else intArrayOf(0, 0, workW, workH)
                val crop = padBox(region, workW, workH, if (opts.crop) opts.pad else 0, opts.square)
                val cropW = crop[2] - crop[0]
                val cropH = crop[3] - crop[1]
                val (outW, outH) = fitLongestEdge(cropW, cropH, opts.outputSize)

                // --- pass 2: re-decode, unmix, crop, resize, encode ------------
                outFile.parentFile?.mkdirs()
                outFile.outputStream().use { fos ->
                    val gif = GifEncoder(fos, outW, outH, opts.loopCount)
                    val delayCs = max(1, (100.0 / fps).roundToInt())
                    var encoded = 0

                    extractor.extractFrames(uri, fps, opts.workMaxSide, durationMs = info.durationMs) { i, bitmap ->
                        val img = bitmapToRgbImage(bitmap)
                        val alpha = readAlpha(File(matteDir, frameName(i)), workW * workH)
                        val screen = if (globalKey != null) Matting.screenRgb(img, alpha) else null
                        val cleaned = if (screen != null) Matting.unmixScreen(img, alpha, screen, runCfg.despill) else img

                        val argb = cropAndComposite(cleaned, alpha, workW, workH, crop, outW, outH)
                        gif.writeFrame(argb, delayCs, opts.alphaThreshold, opts.colors)

                        encoded = i + 1
                        progress.onProgress("encode", encoded, actualCount)
                        true
                    }
                    gif.finish()
                }

                return StickerResult(outFile, actualCount, fps, outW, outH, mode, globalKey)
            } finally {
                model?.close()
            }
        } finally {
            workDir.deleteRecursively()
        }
    }

    // ---- key sampling ---------------------------------------------------------

    /**
     * Detects a chroma key from a handful of frames spread across the clip,
     * taking the median so a single washed-out sample can't derail it. Grabs
     * one frame at a time via [VideoFrameExtractor.extractFrames] with a
     * one-frame window rather than a dedicated single-frame API - simplest
     * way to reuse the same decode path key sampling only needs briefly.
     */
    private fun sampleGlobalKey(extractor: VideoFrameExtractor, uri: Uri, durationMs: Long): KeyColor? {
        val sampleCount = 5
        val keys = mutableListOf<KeyColor>()
        for (i in 0 until sampleCount) {
            val t = if (sampleCount <= 1) 0L else durationMs * i / (sampleCount - 1)
            var picked: Bitmap? = null
            extractor.extractFrames(uri, 1000.0, 480, startMs = t, durationMs = 1) { _, b -> picked = b; false }
            picked?.let { bmp -> Matting.detectKey(bitmapToRgbImage(bmp))?.let { keys += it } }
        }
        if (keys.size < 3) return null
        val crs = keys.map { it.cr }.sorted()
        val cbs = keys.map { it.cb }.sorted()
        fun med(l: List<Float>) = if (l.size % 2 == 1) l[l.size / 2] else (l[l.size / 2 - 1] + l[l.size / 2]) / 2f
        return KeyColor(med(crs), med(cbs))
    }

    // ---- pixel helpers ----------------------------------------------------

    private fun bitmapToRgbImage(bitmap: Bitmap): RgbImage {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return RgbImage.fromArgb(pixels, bitmap.width, bitmap.height)
    }

    /** Crops [img]/[alpha] to [crop] (which may extend past the source - those pixels come out transparent), then resizes to (outW, outH). */
    private fun cropAndComposite(
        img: RgbImage, alpha: FloatArray, srcW: Int, srcH: Int,
        crop: IntArray, outW: Int, outH: Int,
    ): IntArray {
        val cropW = crop[2] - crop[0]
        val cropH = crop[3] - crop[1]
        val cr = FloatArray(cropW * cropH); val cg = FloatArray(cropW * cropH)
        val cb = FloatArray(cropW * cropH); val ca = FloatArray(cropW * cropH)
        for (y in 0 until cropH) {
            val sy = crop[1] + y
            for (x in 0 until cropW) {
                val sx = crop[0] + x
                val di = y * cropW + x
                if (sx in 0 until srcW && sy in 0 until srcH) {
                    val si = sy * srcW + sx
                    cr[di] = img.r[si]; cg[di] = img.g[si]; cb[di] = img.b[si]; ca[di] = alpha[si] * 255f
                } // else stays 0 - transparent, colour irrelevant
            }
        }
        val resized = Matting.resizeRgbaPremultiplied(cr, cg, cb, ca, cropW, cropH, outW, outH)
        val out = IntArray(outW * outH)
        for (i in out.indices) {
            val a = resized.a[i].roundToInt().coerceIn(0, 255)
            val r = resized.r[i].roundToInt().coerceIn(0, 255)
            val g = resized.g[i].roundToInt().coerceIn(0, 255)
            val b = resized.b[i].roundToInt().coerceIn(0, 255)
            out[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }
        return out
    }

    // ---- crop geometry (mirrors the desktop pipeline's _pad_box / _fit) ------

    private fun padBox(box: IntArray, w: Int, h: Int, pad: Int, square: Boolean): IntArray {
        var left = max(0, box[0] - pad)
        var top = max(0, box[1] - pad)
        var right = min(w, box[2] + pad)
        var bottom = min(h, box[3] + pad)
        if (square) {
            val side = max(right - left, bottom - top)
            val cx = (left + right) / 2.0
            val cy = (top + bottom) / 2.0
            left = (cx - side / 2.0).roundToInt()
            top = (cy - side / 2.0).roundToInt()
            right = left + side
            bottom = top + side
        }
        return intArrayOf(left, top, right, bottom)
    }

    private fun fitLongestEdge(w: Int, h: Int, target: Int): Pair<Int, Int> {
        val longest = max(w, h)
        if (longest <= target) return w to h
        val scale = target.toFloat() / longest
        return max(1, (w * scale).roundToInt()) to max(1, (h * scale).roundToInt())
    }

    private fun estimateFrameCount(durationMs: Long, fps: Double): Int =
        max(1, (durationMs / 1000.0 * fps).roundToInt())

    // ---- raw alpha cache files ----------------------------------------------

    private fun frameName(i: Int) = "%05d.raw".format(i)

    private fun writeAlpha(file: File, alpha: FloatArray) {
        val bytes = ByteArray(alpha.size)
        for (i in alpha.indices) bytes[i] = (alpha[i] * 255f).roundToInt().coerceIn(0, 255).toByte()
        file.writeBytes(bytes)
    }

    private fun readAlpha(file: File, expectedSize: Int): FloatArray {
        val bytes = RandomAccessFile(file, "r").use { raf ->
            ByteArray(expectedSize).also { raf.readFully(it) }
        }
        val out = FloatArray(expectedSize)
        for (i in out.indices) out[i] = (bytes[i].toInt() and 0xFF) / 255f
        return out
    }
}
