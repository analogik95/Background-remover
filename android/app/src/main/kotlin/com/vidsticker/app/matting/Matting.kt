package com.vidsticker.app.matting

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Alpha matting: chroma keying, neural matting, and the hybrid of the two.
 *
 * Ported from the desktop pipeline's `matting.py`. The interesting idea is the
 * hybrid: on a green-screen source a chroma key gives pixel-accurate edges but
 * happily keeps anything non-green (light rays, sparkle overlays), while a
 * neural matte knows what the subject is but has soft edges and a
 * screen-coloured fringe. Taking the per-pixel minimum of the two alphas keeps
 * the strengths of both - the key defines the edge, the network defines what
 * counts as subject.
 *
 * Everything here works on flat arrays addressed as `y * width + x`, not on
 * [android.graphics.Bitmap], so the hot per-pixel loops avoid both Bitmap's
 * per-call JNI overhead and an intermediate object per pixel.
 */

/** A chroma key colour, in OpenCV-style YCrCb (Cr, Cb only - Y is discarded). */
data class KeyColor(val cr: Float, val cb: Float)

enum class MatteMode { AUTO, HYBRID, CHROMA, AI }

data class MatteConfig(
    val mode: MatteMode = MatteMode.AUTO,
    /** Null means "fit per frame" - see [Matting.calibrateRamp]. */
    val tolerance: Float? = null,
    val softness: Float? = null,
    val despill: Float = 0.8f,
    val aiGain: Float = 1.0f,
)

/** Bounds on what per-frame calibration may choose. Mirrors matting.py. */
private const val MIN_TOLERANCE = 6.0f
private const val MAX_TOLERANCE = 30.0f
private const val MIN_SOFTNESS = 15.0f
private const val MAX_SOFTNESS = 120.0f
const val DEFAULT_TOLERANCE = 10.0f
const val DEFAULT_SOFTNESS = 45.0f

/** RGB pixels plus dimensions - the shape every function here operates on. */
class RgbImage(val width: Int, val height: Int, val r: FloatArray, val g: FloatArray, val b: FloatArray) {
    companion object {
        /** Extracts RGB planes from an ARGB_8888 pixel buffer ([android.graphics.Bitmap.getPixels]). */
        fun fromArgb(pixels: IntArray, width: Int, height: Int): RgbImage {
            val r = FloatArray(pixels.size)
            val g = FloatArray(pixels.size)
            val b = FloatArray(pixels.size)
            for (i in pixels.indices) {
                val p = pixels[i]
                r[i] = ((p shr 16) and 0xFF).toFloat()
                g[i] = ((p shr 8) and 0xFF).toFloat()
                b[i] = (p and 0xFF).toFloat()
            }
            return RgbImage(width, height, r, g, b)
        }
    }
}

object Matting {

    // ---- colour space -----------------------------------------------------

    /**
     * OpenCV-compatible RGB -> Y/Cr/Cb, verified pixel-for-pixel against
     * `cv2.cvtColor(_, cv2.COLOR_RGB2YCrCb)`: Y is rounded to an integer
     * *before* Cr/Cb are derived from it, matching OpenCV's fixed-point path -
     * skipping that rounding measurably shifts the chroma distance used below.
     */
    fun rgbToYCrCb(r: Float, g: Float, b: Float): Triple<Float, Float, Float> {
        val y = (0.299f * r + 0.587f * g + 0.114f * b).roundToInt().toFloat()
        val cr = (r - y) * 0.713f + 128f
        val cb = (b - y) * 0.564f + 128f
        return Triple(y, cr, cb)
    }

    private fun crChannel(img: RgbImage, out: FloatArray, outCb: FloatArray) {
        for (i in out.indices) {
            val (_, cr, cb) = rgbToYCrCb(img.r[i], img.g[i], img.b[i])
            out[i] = cr
            outCb[i] = cb
        }
    }

    // ---- key detection ------------------------------------------------------

    private fun borderIndices(width: Int, height: Int, margin: Int): IntArray {
        val m = max(1, min(margin, min(width, height) / 2))
        val idx = ArrayList<Int>()
        for (y in 0 until m) for (x in 0 until width) idx += y * width + x
        for (y in height - m until height) for (x in 0 until width) idx += y * width + x
        for (y in m until height - m) {
            for (x in 0 until m) idx += y * width + x
            for (x in width - m until width) idx += y * width + x
        }
        return idx.toIntArray()
    }

    private fun median(values: FloatArray): Float {
        val sorted = values.copyOf().also { it.sort() }
        val n = sorted.size
        return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2f
    }

    /**
     * Guesses a chroma key colour from the image border. Returns null when the
     * border does not look like a solid coloured screen: either too
     * desaturated (a real backdrop) or too varied in hue (a busy scene running
     * to the edge of frame).
     */
    fun detectKey(img: RgbImage): KeyColor? {
        val border = borderIndices(img.width, img.height, 8)
        if (border.isEmpty()) return null

        // Saturation from RGB max/min (HSV-style), to gate on "colourful enough".
        val sats = FloatArray(border.size)
        val hues = FloatArray(border.size)
        for ((k, i) in border.withIndex()) {
            val r = img.r[i]; val g = img.g[i]; val b = img.b[i]
            val mx = max(r, max(g, b)); val mn = min(r, min(g, b))
            sats[k] = if (mx <= 0f) 0f else (mx - mn) / mx * 255f
            hues[k] = hueDegrees(r, g, b, mx, mn)
        }
        if (median(sats) < 60f) return null

        // Hue is circular; measure spread on the unit circle, not with std-dev.
        var sumCos = 0.0; var sumSin = 0.0
        for (h in hues) {
            val rad = h * (2.0 * Math.PI / 360.0)
            sumCos += Math.cos(rad); sumSin += Math.sin(rad)
        }
        val coherence = hypot(sumCos / hues.size, sumSin / hues.size)
        if (coherence < 0.85) return null

        val crVals = FloatArray(border.size)
        val cbVals = FloatArray(border.size)
        for ((k, i) in border.withIndex()) {
            val (_, cr, cb) = rgbToYCrCb(img.r[i], img.g[i], img.b[i])
            crVals[k] = cr; cbVals[k] = cb
        }
        return KeyColor(median(crVals), median(cbVals))
    }

    /** Hue in [0, 360), matching OpenCV/colorsys conventions closely enough for the coherence check. */
    private fun hueDegrees(r: Float, g: Float, b: Float, mx: Float, mn: Float): Float {
        val d = mx - mn
        if (d <= 0f) return 0f
        val h = when (mx) {
            r -> 60f * (((g - b) / d).mod(6f))
            g -> 60f * (((b - r) / d) + 2f)
            else -> 60f * (((r - g) / d) + 4f)
        }
        return if (h < 0f) h + 360f else h
    }

    // ---- chroma alpha ---------------------------------------------------------

    /**
     * Soft alpha from chroma distance to the key. Pixels within [tolerance] of
     * the key are fully transparent; alpha ramps to opaque over the next
     * [softness] units, which is what antialiases the edge instead of a
     * stair-stepped cutout.
     */
    fun chromaAlpha(img: RgbImage, key: KeyColor, tolerance: Float, softness: Float): FloatArray {
        val alpha = FloatArray(img.r.size)
        for (i in alpha.indices) {
            val (_, cr, cb) = rgbToYCrCb(img.r[i], img.g[i], img.b[i])
            val dist = hypot((cr - key.cr).toDouble(), (cb - key.cb).toDouble()).toFloat()
            alpha[i] = ((dist - tolerance) / softness).coerceIn(0f, 1f)
        }
        return alpha
    }

    // ---- ramp calibration -------------------------------------------------

    /**
     * Fits the key's ramp to the colour separation this frame actually shows,
     * using the neural matte to label confident foreground/background.
     *
     * A fixed ramp is guesswork: how far the subject's colours sit from the key
     * depends entirely on the footage. Getting this wrong is what leaves a halo
     * on motion blur - a blurred edge is a genuine ~50/50 mix of subject and
     * screen, landing midway between them in chroma, so a ramp that saturates
     * before that midpoint writes every blurred edge pixel out as opaque
     * backdrop. This mirrors `matting.calibrate_ramp` in the desktop tool,
     * which is exactly the bug that motivated it there.
     */
    fun calibrateRamp(img: RgbImage, key: KeyColor, aiAlpha: FloatArray): Pair<Float, Float>? {
        val w = img.width; val h = img.height
        val fgMask = BooleanArray(aiAlpha.size) { aiAlpha[it] > 0.9f }
        val bgMask = BooleanArray(aiAlpha.size) { aiAlpha[it] < 0.1f }
        erode3x3(fgMask, w, h, radius = 7)
        erode3x3(bgMask, w, h, radius = 7)

        val fgCount = fgMask.count { it }
        val bgCount = bgMask.count { it }
        if (fgCount < 500 || bgCount < 500) return null

        val fgDist = FloatArray(fgCount)
        val bgDist = FloatArray(bgCount)
        var fi = 0; var bi = 0
        for (i in aiAlpha.indices) {
            val (_, cr, cb) = rgbToYCrCb(img.r[i], img.g[i], img.b[i])
            val d = hypot((cr - key.cr).toDouble(), (cb - key.cb).toDouble()).toFloat()
            if (fgMask[i]) fgDist[fi++] = d
            if (bgMask[i]) bgDist[bi++] = d
        }

        // p90 rather than the max: an animated backdrop can carry overlays
        // (rays, sparkles) nowhere near the key that would otherwise set the floor.
        val spread = percentile(bgDist, 90f)
        val subject = percentile(fgDist, 1f)

        val tolerance = (spread * 1.5f).coerceIn(MIN_TOLERANCE, MAX_TOLERANCE)
        val softness = (subject - tolerance).coerceIn(MIN_SOFTNESS, MAX_SOFTNESS)
        return tolerance to softness
    }

    private fun percentile(values: FloatArray, p: Float): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.copyOf().also { it.sort() }
        val rank = (p / 100f) * (sorted.size - 1)
        val lo = rank.toInt().coerceIn(0, sorted.size - 1)
        val hi = (lo + 1).coerceIn(0, sorted.size - 1)
        val frac = rank - lo
        return sorted[lo] * (1 - frac) + sorted[hi] * frac
    }

    /** Erodes a boolean mask in place by [radius] using a square structuring element (cheap separable pass). */
    private fun erode3x3(mask: BooleanArray, w: Int, h: Int, radius: Int) {
        if (radius <= 0) return
        // Two 1-D passes (horizontal then vertical) approximate a square erosion
        // in O(w*h*radius) instead of O(w*h*radius^2), which matters at 720p+.
        val tmp = BooleanArray(mask.size)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var v = true
                var dx = -radius
                while (dx <= radius && v) {
                    val xx = x + dx
                    v = v && xx in 0 until w && mask[y * w + xx]
                    dx++
                }
                tmp[y * w + x] = v
            }
        }
        for (x in 0 until w) {
            for (y in 0 until h) {
                var v = true
                var dy = -radius
                while (dy <= radius && v) {
                    val yy = y + dy
                    v = v && yy in 0 until h && tmp[yy * w + x]
                    dy++
                }
                mask[y * w + x] = v
            }
        }
    }

    // ---- combine ------------------------------------------------------------

    /** Per-pixel minimum of two alphas - the hybrid's actual decision rule. */
    fun combineMin(a: FloatArray, b: FloatArray): FloatArray {
        val out = FloatArray(a.size)
        for (i in out.indices) out[i] = min(a[i], b[i])
        return out
    }

    // ---- screen removal (unmixing) ------------------------------------------

    /** The screen's own colour, as the median RGB of confidently-transparent pixels. */
    fun screenRgb(img: RgbImage, alpha: FloatArray, threshold: Float = 0.1f): FloatArray? {
        val idx = ArrayList<Int>(img.r.size / 4)
        for (i in alpha.indices) if (alpha[i] < threshold) idx += i
        if (idx.size < 200) return null
        val rs = FloatArray(idx.size); val gs = FloatArray(idx.size); val bs = FloatArray(idx.size)
        for ((k, i) in idx.withIndex()) { rs[k] = img.r[i]; gs[k] = img.g[i]; bs[k] = img.b[i] }
        return floatArrayOf(median(rs), median(gs), median(bs))
    }

    /**
     * Recovers the subject's own colour from a pixel that is part screen.
     *
     * A partly covered pixel is literally `C = a*F + (1-a)*S`; solving for F
     * removes exactly the screen's share. This is what a plain "despill" clamp
     * only approximates - on a half-screen blurred edge, the difference is a
     * visible green fringe versus none.
     */
    fun unmixScreen(img: RgbImage, alpha: FloatArray, screen: FloatArray, amount: Float, floor: Float = 0.15f): RgbImage {
        if (amount <= 0f) return img
        val outR = FloatArray(img.r.size); val outG = FloatArray(img.r.size); val outB = FloatArray(img.r.size)
        for (i in outR.indices) {
            val a = max(alpha[i], floor)
            val fr = (img.r[i] - (1 - a) * screen[0]) / a
            val fg = (img.g[i] - (1 - a) * screen[1]) / a
            val fb = (img.b[i] - (1 - a) * screen[2]) / a
            outR[i] = (img.r[i] * (1 - amount) + fr * amount).coerceIn(0f, 255f)
            outG[i] = (img.g[i] * (1 - amount) + fg * amount).coerceIn(0f, 255f)
            outB[i] = (img.b[i] * (1 - amount) + fb * amount).coerceIn(0f, 255f)
        }
        return RgbImage(img.width, img.height, outR, outG, outB)
    }

    // ---- temporal despeckling -------------------------------------------------

    /**
     * Drops coverage that only this frame claims, without ever inventing any.
     *
     * A per-frame neural matte flickers: a pixel blinks opaque for one frame,
     * and a median of three consecutive alphas removes that. But a plain median
     * also *adds* coverage - on a fast-moving limb the neighbouring frames can
     * agree with each other while the current frame disagrees, and the median
     * paints the limb's other position onto this frame's backdrop, which then
     * ships as solid screen colour. Clamping to the current frame keeps the
     * despeckling and drops the invention: a pixel can never come out more
     * opaque than this frame's own matte made it.
     */
    fun temporalDespeckle(prev: FloatArray?, cur: FloatArray, next: FloatArray?): FloatArray {
        if (prev == null || next == null) return cur
        val out = FloatArray(cur.size)
        for (i in out.indices) {
            val lo = min(prev[i], next[i])
            val hi = max(prev[i], next[i])
            val med = if (cur[i] < lo) lo else if (cur[i] > hi) hi else cur[i]
            out[i] = min(cur[i], med)
        }
        return out
    }

    // ---- crop / bbox --------------------------------------------------------

    /** Tight (left, top, right, bottom) box around alpha above [threshold], or null if empty. */
    fun contentBoundingBox(alpha: FloatArray, width: Int, height: Int, threshold: Float = 0.06f): IntArray? {
        var left = width; var right = -1; var top = height; var bottom = -1
        for (y in 0 until height) {
            val rowBase = y * width
            for (x in 0 until width) {
                if (alpha[rowBase + x] > threshold) {
                    if (x < left) left = x
                    if (x > right) right = x
                    if (y < top) top = y
                    if (y > bottom) bottom = y
                }
            }
        }
        return if (right < 0) null else intArrayOf(left, top, right + 1, bottom + 1)
    }

    // ---- premultiplied resize -------------------------------------------------

    /**
     * Resizes straight-alpha RGBA without dragging the backdrop into the edge.
     *
     * Resampling colour and alpha independently mixes a transparent pixel's
     * colour - which still holds the screen's colour - into its opaque
     * neighbours, re-tinting the edge on the way to output size.
     * Premultiplying first weights each pixel's colour by its own coverage, so
     * transparent ones contribute nothing.
     *
     * A plain box filter (rather than bilinear/Lanczos) is used here since it
     * is both correct for downscaling and simple to get right without an image
     * library; upscaling (bigger output than the crop) falls back to bilinear
     * on the premultiplied planes, which is exact for solid regions and only
     * approximate right at the edge - an acceptable trade for a mobile CPU path.
     */
    fun resizeRgbaPremultiplied(
        r: FloatArray, g: FloatArray, b: FloatArray, a: FloatArray,
        srcW: Int, srcH: Int, dstW: Int, dstH: Int,
    ): Quad {
        val pr = FloatArray(r.size); val pg = FloatArray(r.size); val pb = FloatArray(r.size)
        for (i in r.indices) { val av = a[i]; pr[i] = r[i] * av; pg[i] = g[i] * av; pb[i] = b[i] * av }

        val outPr = resizePlane(pr, srcW, srcH, dstW, dstH)
        val outPg = resizePlane(pg, srcW, srcH, dstW, dstH)
        val outPb = resizePlane(pb, srcW, srcH, dstW, dstH)
        val outA = resizePlane(a, srcW, srcH, dstW, dstH)

        val outR = FloatArray(outA.size); val outG = FloatArray(outA.size); val outB = FloatArray(outA.size)
        for (i in outA.indices) {
            val av = max(outA[i], 1e-4f)
            outR[i] = (outPr[i] / av).coerceIn(0f, 255f)
            outG[i] = (outPg[i] / av).coerceIn(0f, 255f)
            outB[i] = (outPb[i] / av).coerceIn(0f, 255f)
        }
        return Quad(outR, outG, outB, outA)
    }

    data class Quad(val r: FloatArray, val g: FloatArray, val b: FloatArray, val a: FloatArray)

    /** Separable bilinear resize of a single plane (area-averaging is left to the box a caller pre-applies, if any). */
    private fun resizePlane(src: FloatArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int): FloatArray {
        if (srcW == dstW && srcH == dstH) return src.copyOf()

        // Horizontal pass.
        val horiz = FloatArray(dstW * srcH)
        val xScale = srcW.toFloat() / dstW
        for (dx in 0 until dstW) {
            val sx = (dx + 0.5f) * xScale - 0.5f
            val x0 = sx.toInt().coerceIn(0, srcW - 1)
            val x1 = (x0 + 1).coerceIn(0, srcW - 1)
            val frac = (sx - x0).coerceIn(0f, 1f)
            for (y in 0 until srcH) {
                val base = y * srcW
                horiz[y * dstW + dx] = src[base + x0] * (1 - frac) + src[base + x1] * frac
            }
        }
        // Vertical pass.
        val out = FloatArray(dstW * dstH)
        val yScale = srcH.toFloat() / dstH
        for (dy in 0 until dstH) {
            val sy = (dy + 0.5f) * yScale - 0.5f
            val y0 = sy.toInt().coerceIn(0, srcH - 1)
            val y1 = (y0 + 1).coerceIn(0, srcH - 1)
            val frac = (sy - y0).coerceIn(0f, 1f)
            for (x in 0 until dstW) {
                out[dy * dstW + x] = horiz[y0 * dstW + x] * (1 - frac) + horiz[y1 * dstW + x] * frac
            }
        }
        return out
    }

    // ---- full per-frame alpha -------------------------------------------------

    /**
     * Computes one frame's alpha per [MatteConfig.mode], mirroring
     * `matting.compute_alpha`. [aiAlpha] is required for AI/HYBRID/AUTO-with-model.
     */
    fun computeAlpha(img: RgbImage, cfg: MatteConfig, aiAlpha: FloatArray?, key: KeyColor?): FloatArray {
        val useAi = cfg.mode == MatteMode.AI || cfg.mode == MatteMode.HYBRID ||
            (cfg.mode == MatteMode.AUTO)
        val useChroma = cfg.mode == MatteMode.CHROMA || cfg.mode == MatteMode.HYBRID ||
            (cfg.mode == MatteMode.AUTO && key != null)

        var alpha: FloatArray? = null
        if (useAi) {
            requireNotNull(aiAlpha) { "AI alpha requested but none was computed" }
            alpha = FloatArray(aiAlpha.size) { (aiAlpha[it] * cfg.aiGain).coerceIn(0f, 1f) }
        }
        if (useChroma && key != null) {
            var tolerance = cfg.tolerance
            var softness = cfg.softness
            if ((tolerance == null || softness == null) && alpha != null) {
                val fitted = calibrateRamp(img, key, alpha)
                if (fitted != null) {
                    if (tolerance == null) tolerance = fitted.first
                    if (softness == null) softness = fitted.second
                }
            }
            val ck = chromaAlpha(img, key, tolerance ?: DEFAULT_TOLERANCE, softness ?: DEFAULT_SOFTNESS)
            alpha = if (alpha == null) ck else combineMin(alpha, ck)
        } else if (useChroma && cfg.mode == MatteMode.CHROMA) {
            error("chroma mode requested but no key colour was detected")
        }
        return alpha ?: error("no matting strategy applies for mode ${cfg.mode}")
    }
}
