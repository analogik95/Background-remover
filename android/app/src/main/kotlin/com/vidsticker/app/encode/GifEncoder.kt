package com.vidsticker.app.encode

import java.io.OutputStream
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.max

/**
 * Writes an animated, transparent GIF89a, one frame at a time.
 *
 * Each frame carries its own local colour table (quantized independently)
 * rather than one global palette shared by the whole clip. That costs a
 * little file size - each frame repeats its palette - but it means the
 * encoder only ever holds one frame in memory, not the whole clip, which is
 * the difference between this running on a phone and not. Disposal method 2
 * ("restore to background") is used on every frame so transparent pixels
 * never show a smeared trail of earlier frames through them.
 */
class GifEncoder(
    private val out: OutputStream,
    private val width: Int,
    private val height: Int,
    private val loopCount: Int = 0,
) {
    private var started = false
    private var finished = false

    private fun ensureStarted() {
        if (started) return
        started = true
        writeAscii("GIF89a")
        writeLE16(width)
        writeLE16(height)
        out.write(0x00) // no global colour table
        out.write(0x00) // background colour index
        out.write(0x00) // pixel aspect ratio
        writeLoopExtension()
    }

    private fun writeLoopExtension() {
        out.write(0x21); out.write(0xFF); out.write(0x0B)
        writeAscii("NETSCAPE2.0")
        out.write(0x03); out.write(0x01)
        writeLE16(loopCount)
        out.write(0x00)
    }

    /**
     * Appends one frame. [argb] is `width*height` packed ARGB pixels
     * (Android's [android.graphics.Bitmap.getPixels] layout); any pixel with
     * alpha below [alphaThreshold] becomes the frame's single transparent
     * index - GIF has no partial alpha, so soft coverage is thresholded here.
     */
    fun writeFrame(argb: IntArray, delayCentiseconds: Int, alphaThreshold: Int, maxColors: Int = 255) {
        require(!finished) { "GIF already finished" }
        require(argb.size == width * height) { "frame size does not match encoder dimensions" }
        ensureStarted()

        val cappedColors = maxColors.coerceIn(1, 255)
        val n = argb.size
        val opaque = BooleanArray(n)
        var opaqueCount = 0
        for (i in 0 until n) {
            if ((argb[i] ushr 24) and 0xFF >= alphaThreshold) { opaque[i] = true; opaqueCount++ }
        }

        val palette: Array<IntArray>
        if (opaqueCount == 0) {
            palette = arrayOf(intArrayOf(0, 0, 0))
        } else {
            val rs = IntArray(opaqueCount); val gs = IntArray(opaqueCount); val bs = IntArray(opaqueCount)
            var k = 0
            for (i in 0 until n) {
                if (opaque[i]) {
                    val p = argb[i]
                    rs[k] = (p shr 16) and 0xFF; gs[k] = (p shr 8) and 0xFF; bs[k] = p and 0xFF
                    k++
                }
            }
            palette = MedianCutQuantizer.quantize(rs, gs, bs, cappedColors)
        }

        val transparentIndex = palette.size
        val neededEntries = palette.size + 1
        val paddedSize = nextPowerOfTwoAtLeast(max(4, neededEntries)).coerceAtMost(256)
        val bits = log2Exact(paddedSize)
        val minCodeSize = bits.coerceIn(2, 8)

        // Real colours' cache: most frames here are flat cartoon-style art with
        // long runs of identical pixels, so caching by exact RGB turns an
        // O(pixels * paletteSize) nearest-colour search into one lookup per
        // *distinct* colour.
        val nearestCache = HashMap<Int, Int>()
        val indices = IntArray(n)
        for (i in 0 until n) {
            if (!opaque[i]) { indices[i] = transparentIndex; continue }
            val p = argb[i]
            val rgb = p and 0xFFFFFF
            indices[i] = nearestCache.getOrPut(rgb) {
                MedianCutQuantizer.nearest(palette, (p shr 16) and 0xFF, (p shr 8) and 0xFF, p and 0xFF)
            }
        }

        writeGraphicControlExtension(delayCentiseconds, transparentIndex)
        writeImageDescriptorAndData(palette, paddedSize, bits - 1, minCodeSize, indices)
    }

    fun finish() {
        if (finished) return
        ensureStarted()
        out.write(0x3B)
        finished = true
    }

    // ---- section writers ----------------------------------------------------

    private fun writeGraphicControlExtension(delayCentiseconds: Int, transparentIndex: Int) {
        out.write(0x21); out.write(0xF9); out.write(0x04)
        val disposalRestoreToBackground = 2
        val packed = (disposalRestoreToBackground shl 2) or 0x01 // transparent-colour flag set
        out.write(packed)
        writeLE16(delayCentiseconds.coerceIn(1, 0xFFFF))
        out.write(transparentIndex and 0xFF)
        out.write(0x00)
    }

    private fun writeImageDescriptorAndData(
        palette: Array<IntArray>, paddedSize: Int, tableSizeField: Int, minCodeSize: Int, indices: IntArray,
    ) {
        out.write(0x2C)
        writeLE16(0); writeLE16(0)
        writeLE16(width); writeLE16(height)
        out.write(0x80 or tableSizeField) // local colour table present

        for (i in 0 until paddedSize) {
            val c = if (i < palette.size) palette[i] else intArrayOf(0, 0, 0)
            out.write(c[0]); out.write(c[1]); out.write(c[2])
        }

        out.write(minCodeSize)
        val compressed = GifLzwEncoder.encode(indices, minCodeSize)
        var offset = 0
        while (offset < compressed.size) {
            val len = (compressed.size - offset).coerceAtMost(255)
            out.write(len)
            out.write(compressed, offset, len)
            offset += len
        }
        out.write(0x00)
    }

    // ---- small helpers --------------------------------------------------------

    private fun writeAscii(s: String) { for (c in s) out.write(c.code) }
    private fun writeLE16(v: Int) { out.write(v and 0xFF); out.write((v shr 8) and 0xFF) }

    private fun nextPowerOfTwoAtLeast(v: Int): Int {
        var p = 1
        while (p < v) p = p shl 1
        return p
    }

    private fun log2Exact(p: Int): Int = ceil(ln(p.toDouble()) / ln(2.0)).toInt().coerceAtLeast(1)
}
