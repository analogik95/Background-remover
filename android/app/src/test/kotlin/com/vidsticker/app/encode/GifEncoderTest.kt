package com.vidsticker.app.encode

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import kotlin.random.Random

/**
 * A from-scratch GIF89a/LZW decoder, used only to verify [GifEncoder]'s
 * output without depending on a platform image codec in a JVM unit test.
 *
 * This exists because the bug it would have caught shipped once already: a
 * one-insertion timing offset between [GifLzwEncoder] and any standard
 * decoder that only shows up once an image has enough distinct colours to
 * grow the code width mid-stream. A handful of flat-colour test frames never
 * exercised that path and passed regardless - these tests specifically use
 * enough random colour to force it, and one large enough to force a
 * mid-stream table reset past 4096 codes.
 */
private object ReferenceGifDecoder {
    class Frame(val width: Int, val height: Int, val argb: IntArray, val delayCentiseconds: Int)

    fun decode(data: ByteArray): List<Frame> {
        var pos = 6 // "GIF89a"
        val screenW = le16(data, pos); val screenH = le16(data, pos + 2); pos += 4
        val lsdPacked = data[pos].toInt() and 0xFF; pos += 1
        pos += 2 // background index, pixel aspect
        if (lsdPacked and 0x80 != 0) pos += (2 shl (lsdPacked and 7)) * 3

        val frames = mutableListOf<Frame>()
        var transparentIndex = -1
        var delay = 0
        while (pos < data.size) {
            when (val marker = data[pos].toInt() and 0xFF) {
                0x21 -> {
                    val label = data[pos + 1].toInt() and 0xFF
                    pos += 2
                    if (label == 0xF9) {
                        val blockSize = data[pos].toInt() and 0xFF
                        val packed = data[pos + 1].toInt() and 0xFF
                        delay = le16(data, pos + 2)
                        transparentIndex = if (packed and 1 != 0) data[pos + 4].toInt() and 0xFF else -1
                        pos += blockSize + 2 // +1 for the size byte itself, +1 for the terminator
                    } else {
                        pos += (data[pos].toInt() and 0xFF) + 1
                        while (true) {
                            val n = data[pos].toInt() and 0xFF; pos += 1
                            if (n == 0) break
                            pos += n
                        }
                    }
                }
                0x2C -> {
                    pos += 1
                    pos += 8 // left, top, width, height (we trust the LSD size)
                    val w = le16(data, pos - 4); val h = le16(data, pos - 2)
                    val idPacked = data[pos].toInt() and 0xFF; pos += 1
                    var palette = IntArray(0)
                    if (idPacked and 0x80 != 0) {
                        val size = 2 shl (idPacked and 7)
                        palette = IntArray(size)
                        for (i in 0 until size) {
                            val r = data[pos].toInt() and 0xFF; val g = data[pos + 1].toInt() and 0xFF; val b = data[pos + 2].toInt() and 0xFF
                            palette[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                            pos += 3
                        }
                    }
                    val minCodeSize = data[pos].toInt() and 0xFF; pos += 1
                    val (subBlocks, newPos) = readSubBlocks(data, pos)
                    pos = newPos
                    val indices = lzwDecode(subBlocks, minCodeSize, w * h)
                    val argb = IntArray(w * h) { i ->
                        val idx = indices[i]
                        if (idx == transparentIndex) 0 else palette[idx]
                    }
                    frames += Frame(w, h, argb, delay)
                }
                0x3B -> return frames
                else -> error("unexpected GIF block marker 0x${marker.toString(16)} at $pos")
            }
        }
        return frames
    }

    private fun le16(d: ByteArray, i: Int) = (d[i].toInt() and 0xFF) or ((d[i + 1].toInt() and 0xFF) shl 8)

    private fun readSubBlocks(data: ByteArray, start: Int): Pair<ByteArray, Int> {
        var pos = start
        val out = ByteArrayOutputStream()
        while (true) {
            val n = data[pos].toInt() and 0xFF; pos += 1
            if (n == 0) break
            out.write(data, pos, n)
            pos += n
        }
        return out.toByteArray() to pos
    }

    /**
     * Growth uses the naive `nextCode >= (1 shl codeSize)`, deliberately
     * *not* matching [GifLzwEncoder]'s `>` - this decoder skips inserting a
     * table entry for the first code after every Clear (there is nothing yet
     * to pair it with), which puts its own code count one behind the
     * encoder's throughout the stream. That natural lag is exactly what the
     * encoder's `>` compensates for; mirroring `>` here on top of the same
     * lag would double-compensate and desync the two.
     */
    private fun lzwDecode(data: ByteArray, minCodeSize: Int, expectedPixels: Int): IntArray {
        val clear = 1 shl minCodeSize
        val eoi = clear + 1
        var codeSize = minCodeSize + 1
        var nextCode = eoi + 1
        var table = HashMap<Int, IntArray>().apply { for (i in 0 until clear) put(i, intArrayOf(i)) }

        var bitBuffer = 0; var bitCount = 0; var pos = 0
        fun readCode(): Int? {
            while (bitCount < codeSize) {
                if (pos >= data.size) return null
                bitBuffer = bitBuffer or ((data[pos].toInt() and 0xFF) shl bitCount)
                pos++; bitCount += 8
            }
            val code = bitBuffer and ((1 shl codeSize) - 1)
            bitBuffer = bitBuffer ushr codeSize
            bitCount -= codeSize
            return code
        }

        val out = ArrayList<Int>(expectedPixels)
        var prev: Int? = null
        while (true) {
            val code = readCode() ?: error("ran out of bits before EOI")
            if (code == clear) {
                table = HashMap<Int, IntArray>().apply { for (i in 0 until clear) put(i, intArrayOf(i)) }
                nextCode = eoi + 1; codeSize = minCodeSize + 1; prev = null
                continue
            }
            if (code == eoi) break

            val entry = table[code] ?: run {
                require(code == nextCode && prev != null) { "bad code $code, nextCode=$nextCode" }
                table.getValue(prev!!) + table.getValue(prev!!)[0]
            }
            out.addAll(entry.toList())
            if (prev != null && nextCode <= 4095) {
                table[nextCode] = table.getValue(prev) + entry[0]
                nextCode++
                if (nextCode >= (1 shl codeSize) && codeSize < 12) codeSize++
            }
            prev = code
        }
        return out.toIntArray()
    }
}

class GifEncoderTest {

    private fun encodeOne(argb: IntArray, w: Int, h: Int, alphaThreshold: Int = 128): ByteArray {
        val out = ByteArrayOutputStream()
        val enc = GifEncoder(out, w, h)
        enc.writeFrame(argb, delayCentiseconds = 10, alphaThreshold = alphaThreshold)
        enc.finish()
        return out.toByteArray()
    }

    @Test fun `round-trips a simple two-colour transparent frame exactly`() {
        val w = 20; val h = 20
        val argb = IntArray(w * h) { i ->
            val x = i % w; val y = i / w
            if (x in 5 until 15 && y in 5 until 15) (0xFF shl 24) or (220 shl 16) or (40 shl 8) or 40
            else 0 // fully transparent
        }
        val bytes = encodeOne(argb, w, h)
        val frames = ReferenceGifDecoder.decode(bytes)
        assertEquals(1, frames.size)
        val f = frames[0]
        for (i in argb.indices) {
            val expectedOpaque = (argb[i] ushr 24) and 0xFF >= 128
            val actualOpaque = (f.argb[i] ushr 24) and 0xFF != 0
            assertEquals("pixel $i transparency", expectedOpaque, actualOpaque)
            if (expectedOpaque) assertEquals("pixel $i colour", argb[i] and 0xFFFFFF, f.argb[i] and 0xFFFFFF)
        }
    }

    @Test fun `round-trips a highly random frame that forces code-width growth`() {
        // Real footage almost always exceeds 255 distinct quantized colours,
        // which is exactly the regime the shipped-then-fixed bug lived in -
        // a flat two-colour test frame never reaches it.
        val rnd = Random(1)
        val w = 200; val h = 150
        val argb = IntArray(w * h) {
            if (rnd.nextInt(10) == 0) 0
            else (0xFF shl 24) or (rnd.nextInt(256) shl 16) or (rnd.nextInt(256) shl 8) or rnd.nextInt(256)
        }
        val bytes = encodeOne(argb, w, h)
        val frames = ReferenceGifDecoder.decode(bytes)
        assertEquals(1, frames.size)
        assertEquals(w * h, frames[0].argb.size)
    }

    @Test fun `round-trips a frame large enough to force a mid-stream table reset`() {
        // >4096 LZW codes forces at least one Clear-and-reset mid-stream; this
        // is the other boundary a small test frame can never reach.
        val rnd = Random(7)
        val w = 300; val h = 300
        val argb = IntArray(w * h) {
            (0xFF shl 24) or (rnd.nextInt(256) shl 16) or (rnd.nextInt(256) shl 8) or rnd.nextInt(256)
        }
        val bytes = encodeOne(argb, w, h)
        val frames = ReferenceGifDecoder.decode(bytes)
        assertEquals(w * h, frames[0].argb.size)
    }

    @Test fun `multi-frame animation preserves frame count and delay`() {
        val w = 16; val h = 16
        val out = ByteArrayOutputStream()
        val enc = GifEncoder(out, w, h, loopCount = 0)
        repeat(4) { i ->
            val argb = IntArray(w * h) { (0xFF shl 24) or ((i * 40) shl 16) }
            enc.writeFrame(argb, delayCentiseconds = 8, alphaThreshold = 128)
        }
        enc.finish()
        val frames = ReferenceGifDecoder.decode(out.toByteArray())
        assertEquals(4, frames.size)
        frames.forEach { assertEquals(8, it.delayCentiseconds) }
    }

    @Test fun `a fully transparent frame decodes to zero opaque pixels`() {
        val w = 10; val h = 10
        val bytes = encodeOne(IntArray(w * h), w, h)
        val frames = ReferenceGifDecoder.decode(bytes)
        assertTrue(frames[0].argb.all { (it ushr 24) and 0xFF == 0 })
    }
}
