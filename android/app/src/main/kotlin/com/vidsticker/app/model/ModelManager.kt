package com.vidsticker.app.model

import android.content.Context
import com.vidsticker.app.matting.MattingModelKind
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Downloads and verifies the on-device matting model.
 *
 * The model is not bundled into the APK - at ~175 MB it would roughly double
 * the app's install size for something only half of users need (most people
 * will pick one of the two checkpoints, not both). Instead it is fetched on
 * first use from the same public release rembg itself downloads from, and
 * cached under the app's files directory, matching the desktop tool's own
 * `~/.u2net` cache.
 */
class ModelManager(private val context: Context) {

    // Checksums as published by rembg (github.com/danielgatis/rembg) - matched
    // against the actual files during the desktop side of this project, so
    // pinning them here catches a corrupted or wrong download outright rather
    // than letting a broken model fail mysteriously on the first frame.
    private val checksums = mapOf(
        MattingModelKind.GENERAL to "fc16ebd8b0c10d971d3513d564d01e29",
        MattingModelKind.ANIME to "6f184e756bb3bd901c8849220a83e38e",
    )

    fun modelFile(kind: MattingModelKind): File =
        File(File(context.filesDir, "models"), kind.assetName)

    fun isDownloaded(kind: MattingModelKind): Boolean = modelFile(kind).let {
        it.isFile && it.length() > 0
    }

    sealed class Progress {
        data class Downloading(val bytesRead: Long, val totalBytes: Long) : Progress()
        data class Verifying(val kind: MattingModelKind) : Progress()
        data class Done(val kind: MattingModelKind) : Progress()
        data class Failed(val kind: MattingModelKind, val message: String) : Progress()
    }

    /**
     * Downloads [kind] if not already present, streaming progress through
     * [onProgress]. Safe to call when the model already exists - returns
     * immediately with a single [Progress.Done].
     */
    suspend fun ensureDownloaded(kind: MattingModelKind, onProgress: (Progress) -> Unit) {
        if (isDownloaded(kind)) {
            onProgress(Progress.Done(kind))
            return
        }
        withContext(Dispatchers.IO) {
            val dest = modelFile(kind)
            dest.parentFile?.mkdirs()
            val tmp = File(dest.parentFile, "${dest.name}.part")

            try {
                val connection = (URL(kind.downloadUrl).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 15_000
                    readTimeout = 30_000
                    instanceFollowRedirects = true
                }
                connection.connect()
                if (connection.responseCode !in 200..299) {
                    throw IOException("server returned HTTP ${connection.responseCode}")
                }
                val total = connection.contentLengthLong
                val digest = MessageDigest.getInstance("MD5")

                connection.inputStream.use { input ->
                    tmp.outputStream().use { output ->
                        val buffer = ByteArray(64 * 1024)
                        var readSoFar = 0L
                        var lastReportedPercent = -1
                        while (true) {
                            val n = input.read(buffer)
                            if (n < 0) break
                            output.write(buffer, 0, n)
                            digest.update(buffer, 0, n)
                            readSoFar += n
                            // Throttle to whole percent points - at 64 KB chunks over a
                            // 176 MB file that is still ~2700 callbacks without this.
                            val percent = if (total > 0) (readSoFar * 100 / total).toInt() else -1
                            if (percent != lastReportedPercent) {
                                lastReportedPercent = percent
                                onProgress(Progress.Downloading(readSoFar, total))
                            }
                        }
                    }
                }

                onProgress(Progress.Verifying(kind))
                val actual = digest.digest().joinToString("") { "%02x".format(it) }
                val expected = checksums[kind]
                if (expected != null && !actual.equals(expected, ignoreCase = true)) {
                    tmp.delete()
                    onProgress(Progress.Failed(kind, "downloaded file failed verification - please retry"))
                    return@withContext
                }

                if (!tmp.renameTo(dest)) {
                    // Cross-filesystem fallback (renameTo can fail across volumes).
                    tmp.copyTo(dest, overwrite = true)
                    tmp.delete()
                }
                onProgress(Progress.Done(kind))
            } catch (e: IOException) {
                tmp.delete()
                onProgress(Progress.Failed(kind, e.message ?: "network error"))
            }
        }
    }

    fun delete(kind: MattingModelKind) {
        modelFile(kind).delete()
    }
}
