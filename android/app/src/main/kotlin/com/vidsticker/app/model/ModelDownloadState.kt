package com.vidsticker.app.model

import com.vidsticker.app.matting.MattingModelKind
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * In-process pub/sub between [ModelDownloadService] and the UI.
 *
 * The service and the activity share a process, so a plain observable holder
 * is simpler and just as reliable as a bound-service/Messenger round trip for
 * something this app never needs across processes.
 */
object ModelDownloadState {
    private val _progress = MutableStateFlow<ModelManager.Progress?>(null)
    val progress: StateFlow<ModelManager.Progress?> = _progress.asStateFlow()

    fun reset(kind: MattingModelKind) {
        _progress.value = ModelManager.Progress.Downloading(0, 0)
    }

    fun update(progress: ModelManager.Progress) {
        _progress.value = progress
    }
}
