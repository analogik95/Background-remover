package com.vidsticker.app.model

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.vidsticker.app.MainActivity
import com.vidsticker.app.VidStickerApp
import com.vidsticker.app.matting.MattingModelKind
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Runs the ~175 MB model download as a foreground service.
 *
 * A download this size left as a plain background coroutine gets killed the
 * moment the user switches away from the app - Doze and the various OEM
 * battery managers all treat an ordinary background process as fair game.
 * Foreground-with-notification is the one mechanism Android reliably leaves
 * alone.
 */
class ModelDownloadService : Service() {

    private val scope = CoroutineScope(SupervisorJob())
    private lateinit var manager: ModelManager

    override fun onCreate() {
        super.onCreate()
        manager = ModelManager(applicationContext)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val kindName = intent?.getStringExtra(EXTRA_MODEL_KIND)
        val kind = kindName?.let { runCatching { MattingModelKind.valueOf(it) }.getOrNull() }
            ?: MattingModelKind.ANIME

        startForegroundCompat(buildNotification(0, indeterminate = true))
        ModelDownloadState.reset(kind)

        scope.launch {
            manager.ensureDownloaded(kind) { progress ->
                ModelDownloadState.update(progress)
                when (progress) {
                    is ModelManager.Progress.Downloading -> {
                        val pct = if (progress.totalBytes > 0)
                            (progress.bytesRead * 100 / progress.totalBytes).toInt() else 0
                        notify(buildNotification(pct, indeterminate = progress.totalBytes <= 0))
                    }
                    is ModelManager.Progress.Verifying -> notify(buildNotification(100, indeterminate = true))
                    is ModelManager.Progress.Done, is ModelManager.Progress.Failed -> stopSelf()
                }
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private fun notify(n: Notification) {
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, n)
    }

    private fun startForegroundCompat(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildNotification(percent: Int, indeterminate: Boolean): Notification {
        val openApp = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, VidStickerApp.MODEL_DOWNLOAD_CHANNEL)
            .setContentTitle("Downloading matting model")
            .setContentText(if (indeterminate) "Starting…" else "$percent%")
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(100, percent, indeterminate)
            .setOngoing(true)
            .setContentIntent(openApp)
            .build()
    }

    companion object {
        const val EXTRA_MODEL_KIND = "model_kind"
        private const val NOTIFICATION_ID = 42

        fun start(context: android.content.Context, kind: MattingModelKind) {
            val intent = Intent(context, ModelDownloadService::class.java)
                .putExtra(EXTRA_MODEL_KIND, kind.name)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }
    }
}
