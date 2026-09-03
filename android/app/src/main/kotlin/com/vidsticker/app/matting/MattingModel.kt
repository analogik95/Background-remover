package com.vidsticker.app.matting

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

enum class MattingModelKind(val assetName: String, val downloadUrl: String, val meanR: Float, val meanG: Float, val meanB: Float) {
    GENERAL(
        assetName = "isnet-general-use.onnx",
        downloadUrl = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        meanR = 0.5f, meanG = 0.5f, meanB = 0.5f,
    ),
    ANIME(
        assetName = "isnet-anime.onnx",
        downloadUrl = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
        // ImageNet mean, used by the anime-tuned checkpoint - closes interior
        // gaps (a sleeve/forearm slot) noticeably better than the general model.
        meanR = 0.485f, meanG = 0.456f, meanB = 0.406f,
    ),
}

/**
 * Runs an isnet segmentation model (the same ones the desktop tool's `rembg`
 * dependency uses) via ONNX Runtime, and returns a per-pixel alpha mask.
 *
 * Preprocessing is ported line-for-line from rembg's `BaseSession.normalize`
 * and `DisSession.predict` so the two implementations agree on what the model
 * is actually being asked: resize to 1024x1024, divide by the resized image's
 * *own* maximum pixel value (not a fixed 255 - unusual, but that is what the
 * checkpoint was exported expecting), subtract the channel mean, run the
 * model, then min-max normalize the raw output back to a 0..1 mask.
 */
class MattingModel private constructor(
    private val session: OrtSession,
    private val env: OrtEnvironment,
    private val kind: MattingModelKind,
) : AutoCloseable {

    companion object {
        private const val TAG = "MattingModel"
        private const val INPUT_SIZE = 1024

        fun load(modelFile: File, kind: MattingModelKind): MattingModel {
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            try {
                // Hardware acceleration where the device offers it; CPU is the
                // guaranteed fallback and onnxruntime handles the fallback itself
                // if NNAPI rejects a node in the graph.
                options.addNnapi()
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI unavailable, using CPU execution provider", e)
            }
            val session = env.createSession(modelFile.absolutePath, options)
            return MattingModel(session, env, kind)
        }
    }

    /**
     * Runs the model on [bitmap] and returns a [width]x[height] alpha mask
     * (0f..1f), matching the source image's own resolution.
     */
    fun predictAlpha(bitmap: Bitmap): FloatArray {
        val srcW = bitmap.width
        val srcH = bitmap.height

        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        if (scaled !== bitmap) scaled.recycle()

        val chw = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        val plane = INPUT_SIZE * INPUT_SIZE

        // rembg normalizes by the resized image's own max pixel value, not a
        // fixed 255 - reproduced here rather than assumed away, since a very
        // dark frame would otherwise be preprocessed differently than the
        // reference implementation.
        var maxVal = 1e-6f
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            if (r > maxVal) maxVal = r.toFloat()
            if (g > maxVal) maxVal = g.toFloat()
            if (b > maxVal) maxVal = b.toFloat()
        }

        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / maxVal
            val g = ((p shr 8) and 0xFF) / maxVal
            val b = (p and 0xFF) / maxVal
            chw[i] = (r - kind.meanR)                 // R plane, std == 1.0
            chw[plane + i] = (g - kind.meanG)          // G plane
            chw[2 * plane + i] = (b - kind.meanB)      // B plane
        }

        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputName = session.inputNames.iterator().next()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { result ->
                val raw = result[0].value as Array<Array<Array<FloatArray>>>
                // Shape is [1, 1, 1024, 1024]; flatten and min-max normalize,
                // exactly mirroring `(pred - pred.min()) / (pred.max() - pred.min())`.
                val flat = FloatArray(plane)
                var mn = Float.MAX_VALUE
                var mx = -Float.MAX_VALUE
                var idx = 0
                for (y in 0 until INPUT_SIZE) {
                    val row = raw[0][0][y]
                    for (x in 0 until INPUT_SIZE) {
                        val v = row[x]
                        flat[idx++] = v
                        if (v < mn) mn = v
                        if (v > mx) mx = v
                    }
                }
                val range = (mx - mn).let { if (it < 1e-6f) 1e-6f else it }
                for (i in flat.indices) flat[i] = (flat[i] - mn) / range

                return resizeMaskBilinear(flat, INPUT_SIZE, INPUT_SIZE, srcW, srcH)
            }
        }
    }

    /** Plain bilinear resize for a single-channel mask - no premultiplication needed, it isn't RGBA. */
    private fun resizeMaskBilinear(src: FloatArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int): FloatArray {
        if (srcW == dstW && srcH == dstH) return src
        val out = FloatArray(dstW * dstH)
        val xScale = srcW.toFloat() / dstW
        val yScale = srcH.toFloat() / dstH
        for (dy in 0 until dstH) {
            val sy = ((dy + 0.5f) * yScale - 0.5f).coerceIn(0f, (srcH - 1).toFloat())
            val y0 = sy.toInt().coerceIn(0, srcH - 1)
            val y1 = (y0 + 1).coerceIn(0, srcH - 1)
            val fy = sy - y0
            for (dx in 0 until dstW) {
                val sx = ((dx + 0.5f) * xScale - 0.5f).coerceIn(0f, (srcW - 1).toFloat())
                val x0 = sx.toInt().coerceIn(0, srcW - 1)
                val x1 = (x0 + 1).coerceIn(0, srcW - 1)
                val fx = sx - x0
                val top = src[y0 * srcW + x0] * (1 - fx) + src[y0 * srcW + x1] * fx
                val bot = src[y1 * srcW + x0] * (1 - fx) + src[y1 * srcW + x1] * fx
                out[dy * dstW + dx] = top * (1 - fy) + bot * fy
            }
        }
        return out
    }

    override fun close() {
        session.close()
    }
}
