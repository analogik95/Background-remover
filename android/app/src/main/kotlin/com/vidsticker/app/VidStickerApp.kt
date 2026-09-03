package com.vidsticker.app

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build

class VidStickerApp : Application() {
    override fun onCreate() {
        super.onCreate()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                MODEL_DOWNLOAD_CHANNEL, "Model download", NotificationManager.IMPORTANCE_LOW,
            ).apply { description = "Progress while the on-device matting model downloads" }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    companion object {
        const val MODEL_DOWNLOAD_CHANNEL = "model_download"
    }
}
