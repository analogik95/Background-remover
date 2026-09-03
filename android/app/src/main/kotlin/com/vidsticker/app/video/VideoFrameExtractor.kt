package com.vidsticker.app.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong

data class VideoInfo(val width: Int, val height: Int, val durationMs: Long, val rotation: Int)

/**
 * Decodes a video into a sequence of frame [Bitmap]s at a target frame rate.
 *
 * Uses [MediaMetadataRetriever.getFrameAtTime] rather than a manual
 * MediaCodec+Surface decode loop. That is the faster path in principle, but
 * its output colour format varies by device/encoder in ways that are a
 * frequent source of subtly wrong pixels; `getFrameAtTime` always hands back a
 * correct ARGB_8888 bitmap. It is not the bottleneck here regardless - matting
 * a single 1024x1024 frame through the model costs far more than decoding it.
 */
class VideoFrameExtractor(private val context: Context) {

    fun probe(uri: Uri): VideoInfo {
        withDataSource(uri) { mmr ->
            val w = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val h = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val duration = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val rotation = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            return VideoInfo(w, h, duration, rotation)
        }
    }

    /**
     * Extracts frames at [fps], each capped to [maxSide] on its longest edge
     * (downscale only - matting accuracy tracks input resolution, so this
     * never upscales to reach the cap). [onFrame] is called once per frame in
     * order; returning false from it stops extraction early.
     */
    fun extractFrames(
        uri: Uri,
        fps: Double,
        maxSide: Int,
        startMs: Long = 0,
        durationMs: Long? = null,
        onFrame: (index: Int, bitmap: Bitmap) -> Boolean,
    ) {
        val info = probe(uri)
        val clipDuration = durationMs ?: (info.durationMs - startMs)
        val frameCount = max(1, (clipDuration / 1000.0 * fps).roundToLong().toInt())
        val stepUs = (1_000_000.0 / fps).roundToLong()
        val startUs = startMs * 1000

        withDataSource(uri) { mmr ->
            // A pipeline stage that fails to decode index i and skips it would
            // leave a gap in the numbered cache files a later pass reads by
            // index - repeating the last good frame keeps the sequence dense
            // instead. This only ever triggers on a handful of pathological
            // timestamps (usually right at EOF); it does not mask genuine
            // decode failures, since the first frame still has to succeed.
            var lastGood: Bitmap? = null
            try {
                for (i in 0 until frameCount) {
                    val timeUs = startUs + i * stepUs
                    val decoded = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                        ?: mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                    val frame = decoded ?: lastGood
                        ?: throw IllegalStateException("could not decode the first frame of this clip")

                    val bounded = boundToMaxSide(frame, maxSide)
                    val keepGoing = onFrame(i, bounded)
                    if (bounded !== frame) bounded.recycle()
                    if (decoded != null && decoded !== lastGood) {
                        lastGood?.recycle()
                        lastGood = decoded
                    }
                    if (!keepGoing) break
                }
            } finally {
                lastGood?.recycle()
            }
        }
    }

    private fun boundToMaxSide(bitmap: Bitmap, maxSide: Int): Bitmap {
        val longest = max(bitmap.width, bitmap.height)
        if (longest <= maxSide) return bitmap
        val scale = maxSide.toFloat() / longest
        val w = max(1, (bitmap.width * scale).roundToLong().toInt())
        val h = max(1, (bitmap.height * scale).roundToLong().toInt())
        return Bitmap.createScaledBitmap(bitmap, w, h, true)
    }

    /** Opens a retriever on [uri] for the duration of [block], always releasing it - regardless of Android API level. */
    private inline fun <T> withDataSource(uri: Uri, block: (MediaMetadataRetriever) -> T): T {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(context, uri)
            block(mmr)
        } finally {
            mmr.release()
        }
    }

    companion object {
        /** Clamps a requested fps to something a GIF's whole-centisecond delays can represent exactly. */
        private val GIF_EXACT_FPS = doubleArrayOf(50.0, 25.0, 20.0, 12.5, 10.0, 5.0)

        fun nearestGifFps(fps: Double): Double = GIF_EXACT_FPS.minByOrNull { kotlin.math.abs(it - fps) } ?: 25.0
    }
}
