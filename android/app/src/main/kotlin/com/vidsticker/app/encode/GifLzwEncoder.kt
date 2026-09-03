package com.vidsticker.app.encode

import java.io.ByteArrayOutputStream

/**
 * GIF's variable-width LZW compressor, applied to a stream of palette indices.
 *
 * This is the one place in a GIF encoder where a one-bit mistake produces a
 * file that *looks* fine in a hex dump but decodes to garbage: the code width
 * has to grow at exactly the point every decoder expects.
 *
 * That point is *not* the naive "the next code no longer fits in the current
 * width" - `nextCode >= (1 shl codeSize)`. A decoder only learns a new table
 * entry exists after reading the *following* code (it has nothing to pair
 * the first code of a block with), so its own code count trails the
 * encoder's by exactly one throughout the stream. Growing on `nextCode >
 * (1 shl codeSize)` - one step later than the naive check - is what makes the
 * encoder's transition line up with where a standard decoder independently
 * arrives at its own, correctly-lagged count. Verified by round-tripping
 * through a from-scratch reference decoder before this ever ran on a device;
 * the naive `>=` version passes on small inputs (never enough distinct
 * colours to exercise a width change) and corrupts anything larger.
 */
object GifLzwEncoder {

    /**
     * Compresses [indices] (values in `0 until 2^minCodeSize`) into a
     * byte-aligned LZW stream, GIF-style: Clear and End-of-Information codes
     * bracket the data, and the code table resets on overflow past 12 bits.
     */
    fun encode(indices: IntArray, minCodeSize: Int): ByteArray {
        val clearCode = 1 shl minCodeSize
        val eoiCode = clearCode + 1
        val firstFreeCode = eoiCode + 1

        val out = ByteArrayOutputStream()
        val bits = BitPacker(out)

        var codeSize = minCodeSize + 1
        var nextCode = firstFreeCode
        // Key = (prefixCode shl 8) or pixelValue. Pixel values are small
        // (<=255), and prefix codes never exceed 4095, so this never collides.
        val table = HashMap<Long, Int>()

        bits.writeCode(clearCode, codeSize)

        var prefix = -1
        for (k in indices) {
            if (prefix < 0) { prefix = k; continue }

            val key = (prefix.toLong() shl 8) or k.toLong()
            val existing = table[key]
            if (existing != null) {
                prefix = existing
                continue
            }

            bits.writeCode(prefix, codeSize)

            if (nextCode <= 4095) {
                table[key] = nextCode
                nextCode++
                if (nextCode > (1 shl codeSize) && codeSize < 12) {
                    codeSize++
                }
            } else {
                bits.writeCode(clearCode, codeSize)
                table.clear()
                codeSize = minCodeSize + 1
                nextCode = firstFreeCode
            }
            prefix = k
        }
        if (prefix >= 0) bits.writeCode(prefix, codeSize)
        bits.writeCode(eoiCode, codeSize)
        bits.flush()

        return out.toByteArray()
    }

    /** Packs variable-width codes LSB-first into bytes, GIF's bit order. */
    private class BitPacker(private val out: ByteArrayOutputStream) {
        private var bitBuffer = 0
        private var bitCount = 0

        fun writeCode(code: Int, width: Int) {
            bitBuffer = bitBuffer or (code shl bitCount)
            bitCount += width
            while (bitCount >= 8) {
                out.write(bitBuffer and 0xFF)
                bitBuffer = bitBuffer ushr 8
                bitCount -= 8
            }
        }

        fun flush() {
            if (bitCount > 0) {
                out.write(bitBuffer and 0xFF)
                bitBuffer = 0
                bitCount = 0
            }
        }
    }
}
