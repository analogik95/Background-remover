package com.vidsticker.app.encode

import kotlin.math.max

/**
 * Reduces an image's colours to a palette of at most [maxColors], via the
 * standard median-cut algorithm: repeatedly split the largest-range colour
 * box along its longest axis until there are enough boxes, then average each
 * box's pixels into one palette entry.
 */
object MedianCutQuantizer {

    class Palette(val colors: Array<IntArray>, val pixelToIndex: (r: Int, g: Int, b: Int) -> Int)

    private class Box(val pixels: MutableList<Int>, val r: IntArray, val g: IntArray, val b: IntArray) {
        fun rangeAxis(): Int {
            var rMin = 255; var rMax = 0; var gMin = 255; var gMax = 0; var bMin = 255; var bMax = 0
            for (p in pixels) {
                if (r[p] < rMin) rMin = r[p]; if (r[p] > rMax) rMax = r[p]
                if (g[p] < gMin) gMin = g[p]; if (g[p] > gMax) gMax = g[p]
                if (b[p] < bMin) bMin = b[p]; if (b[p] > bMax) bMax = b[p]
            }
            val rr = rMax - rMin; val gr = gMax - gMin; val br = bMax - bMin
            return if (rr >= gr && rr >= br) 0 else if (gr >= br) 1 else 2
        }
    }

    /**
     * Builds a palette from the given RGB samples (parallel arrays, one entry
     * per sample pixel - callers typically pass only the *opaque* pixels of an
     * image, skipping fully-transparent ones since those never need a slot).
     */
    fun quantize(r: IntArray, g: IntArray, b: IntArray, maxColors: Int): Array<IntArray> {
        require(maxColors in 1..256)
        if (r.isEmpty()) return arrayOf(intArrayOf(0, 0, 0))

        val allPixels = (0 until r.size).toMutableList()
        var boxes = mutableListOf(Box(allPixels, r, g, b))

        while (boxes.size < maxColors) {
            val splitIdx = boxes.indices.maxByOrNull { boxSpread(boxes[it]) } ?: break
            val box = boxes[splitIdx]
            if (box.pixels.size <= 1) break

            val axis = box.rangeAxis()
            val channel = when (axis) { 0 -> r; 1 -> g; else -> b }
            val sorted = box.pixels.sortedBy { channel[it] }
            val mid = sorted.size / 2
            val left = Box(sorted.subList(0, mid).toMutableList(), r, g, b)
            val right = Box(sorted.subList(mid, sorted.size).toMutableList(), r, g, b)

            boxes.removeAt(splitIdx)
            boxes.add(left)
            boxes.add(right)
        }

        return boxes.map { box ->
            var sr = 0L; var sg = 0L; var sb = 0L
            for (p in box.pixels) { sr += r[p]; sg += g[p]; sb += b[p] }
            val n = max(1, box.pixels.size)
            intArrayOf((sr / n).toInt(), (sg / n).toInt(), (sb / n).toInt())
        }.toTypedArray()
    }

    private fun boxSpread(box: Box): Int {
        if (box.pixels.size <= 1) return -1
        var rMin = 255; var rMax = 0; var gMin = 255; var gMax = 0; var bMin = 255; var bMax = 0
        for (p in box.pixels) {
            if (box.r[p] < rMin) rMin = box.r[p]; if (box.r[p] > rMax) rMax = box.r[p]
            if (box.g[p] < gMin) gMin = box.g[p]; if (box.g[p] > gMax) gMax = box.g[p]
            if (box.b[p] < bMin) bMin = box.b[p]; if (box.b[p] > bMax) bMax = box.b[p]
        }
        return max(rMax - rMin, max(gMax - gMin, bMax - bMin))
    }

    /** Index of the palette entry closest to (r, g, b) by squared Euclidean distance. */
    fun nearest(palette: Array<IntArray>, r: Int, g: Int, b: Int): Int {
        var best = 0; var bestDist = Int.MAX_VALUE
        for (i in palette.indices) {
            val c = palette[i]
            val dr = c[0] - r; val dg = c[1] - g; val db = c[2] - b
            val dist = dr * dr + dg * dg + db * db
            if (dist < bestDist) { bestDist = dist; best = i }
        }
        return best
    }
}
