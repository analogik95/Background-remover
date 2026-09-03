package com.vidsticker.app.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.vidsticker.app.matting.MatteConfig
import com.vidsticker.app.matting.MatteMode
import com.vidsticker.app.matting.MattingModelKind
import com.vidsticker.app.model.ModelDownloadService
import com.vidsticker.app.model.ModelDownloadState
import com.vidsticker.app.model.ModelManager
import com.vidsticker.app.pipeline.StickerOptions
import com.vidsticker.app.pipeline.StickerPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

data class UiOptions(
    val size: Int = 512,
    val modelKind: MattingModelKind = MattingModelKind.ANIME,
    val mode: MatteMode = MatteMode.AUTO,
    val square: Boolean = false,
)

sealed class JobState {
    data object Idle : JobState()
    data class NeedsModel(val kind: MattingModelKind) : JobState()
    data class Downloading(val percent: Int) : JobState()
    data class Running(val stage: String, val fraction: Float) : JobState()
    data class Done(val file: File, val frames: Int, val fps: Double, val width: Int, val height: Int, val mode: MatteMode) : JobState()
    data class Error(val message: String) : JobState()
}

/** Rough share of total runtime per stage, so the bar advances evenly rather than sitting at 5% through the long matte pass. */
private val STAGE_WEIGHT = mapOf("probe" to 1, "model" to 2, "matte" to 70, "analyse" to 8, "encode" to 19)
private val STAGE_ORDER = listOf("probe", "model", "matte", "analyse", "encode")

class StickerViewModel(app: Application) : AndroidViewModel(app) {

    private val modelManager = ModelManager(app)
    private val pipeline = StickerPipeline(app)

    private val _selectedVideo = MutableStateFlow<Uri?>(null)
    val selectedVideo: StateFlow<Uri?> = _selectedVideo.asStateFlow()

    private val _options = MutableStateFlow(UiOptions())
    val options: StateFlow<UiOptions> = _options.asStateFlow()

    private val _jobState = MutableStateFlow<JobState>(JobState.Idle)
    val jobState: StateFlow<JobState> = _jobState.asStateFlow()

    init {
        viewModelScope.launch {
            ModelDownloadState.progress.collect { p ->
                when (p) {
                    is ModelManager.Progress.Downloading -> {
                        val pct = if (p.totalBytes > 0) (p.bytesRead * 100 / p.totalBytes).toInt() else 0
                        _jobState.value = JobState.Downloading(pct)
                    }
                    is ModelManager.Progress.Verifying -> _jobState.value = JobState.Downloading(100)
                    is ModelManager.Progress.Done -> if (_jobState.value is JobState.Downloading) {
                        _jobState.value = JobState.Idle
                    }
                    is ModelManager.Progress.Failed -> _jobState.value = JobState.Error(p.message)
                    null -> Unit
                }
            }
        }
    }

    fun onVideoPicked(uri: Uri) {
        _selectedVideo.value = uri
        _jobState.value = JobState.Idle
    }

    fun updateOptions(transform: (UiOptions) -> UiOptions) {
        _options.value = transform(_options.value)
    }

    fun startDownload() {
        ModelDownloadService.start(getApplication(), _options.value.modelKind)
    }

    fun makeSticker() {
        val uri = _selectedVideo.value ?: return
        val opts = _options.value

        val needsModel = opts.mode != MatteMode.CHROMA
        if (needsModel && !modelManager.isDownloaded(opts.modelKind)) {
            _jobState.value = JobState.NeedsModel(opts.modelKind)
            return
        }

        viewModelScope.launch(Dispatchers.Default) {
            _jobState.value = JobState.Running("probe", 0f)
            try {
                val outFile = File(getApplication<Application>().cacheDir, "stickers/sticker-${System.currentTimeMillis()}.gif")
                val modelFile = if (needsModel) modelManager.modelFile(opts.modelKind) else null

                val result = pipeline.createSticker(
                    uri = uri,
                    modelFile = modelFile,
                    modelKind = opts.modelKind,
                    opts = StickerOptions(
                        outputSize = opts.size,
                        square = opts.square,
                        matte = MatteConfig(mode = opts.mode),
                    ),
                    outFile = outFile,
                    progress = { stage, done, total ->
                        _jobState.value = JobState.Running(stage, overallFraction(stage, done, total))
                    },
                )
                _jobState.value = JobState.Done(result.outFile, result.frames, result.fps, result.width, result.height, result.mode)
            } catch (e: Exception) {
                _jobState.value = JobState.Error(e.message ?: "conversion failed")
            }
        }
    }

    private fun overallFraction(stage: String, done: Int, total: Int): Float {
        val totalWeight = STAGE_WEIGHT.values.sum()
        val idx = STAGE_ORDER.indexOf(stage).coerceAtLeast(0)
        val before = STAGE_ORDER.take(idx).sumOf { STAGE_WEIGHT[it] ?: 0 }
        val frac = if (total > 0) done.toFloat() / total else 1f
        return ((before + (STAGE_WEIGHT[stage] ?: 0) * frac) / totalWeight).coerceIn(0f, 1f)
    }
}
