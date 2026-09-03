package com.vidsticker.app.ui

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.drawable.AnimatedImageDrawable
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.widget.ImageView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.vidsticker.app.matting.MatteMode
import com.vidsticker.app.matting.MattingModelKind
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(vm: StickerViewModel = viewModel()) {
    val context = LocalContext.current
    val video by vm.selectedVideo.collectAsState()
    val options by vm.options.collectAsState()
    val job by vm.jobState.collectAsState()

    val pickVideo = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let(vm::onVideoPicked)
    }

    Scaffold { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("vidsticker", style = MaterialTheme.typography.headlineMedium)
            Text(
                "Cut the background out of a video, on this device - no upload, no server.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Button(onClick = { pickVideo.launch("video/*") }, modifier = Modifier.fillMaxWidth()) {
                Text(if (video == null) "Choose a video" else "Choose a different video")
            }
            video?.let { Text(it.lastPathSegmentOrSelf(), style = MaterialTheme.typography.bodySmall) }

            OptionsCard(options = options, onChange = vm::updateOptions)

            Button(
                onClick = vm::makeSticker,
                enabled = video != null && job !is JobState.Running && job !is JobState.Downloading,
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Make sticker") }

            JobStatus(job = job, onDownload = vm::startDownload)
        }
    }
}

@Composable
private fun OptionsCard(options: UiOptions, onChange: ((UiOptions) -> UiOptions) -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Sticker options", style = MaterialTheme.typography.titleSmall)

            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Size", modifier = Modifier.width(60.dp))
                listOf(320, 512, 768).forEach { s ->
                    FilterChip(
                        selected = options.size == s,
                        onClick = { onChange { it.copy(size = s) } },
                        label = { Text("${s}px") },
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Model", modifier = Modifier.width(60.dp))
                MattingModelKind.values().forEach { k ->
                    FilterChip(
                        selected = options.modelKind == k,
                        onClick = { onChange { it.copy(modelKind = k) } },
                        label = { Text(if (k == MattingModelKind.ANIME) "Anime" else "General") },
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Matte", modifier = Modifier.width(60.dp))
                MatteMode.values().forEach { m ->
                    FilterChip(
                        selected = options.mode == m,
                        onClick = { onChange { it.copy(mode = m) } },
                        label = { Text(m.name.lowercase().replaceFirstChar(Char::uppercase)) },
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = options.square,
                    onClick = { onChange { it.copy(square = !it.square) } },
                    label = { Text("Square canvas") },
                )
            }
        }
    }
}

private val STAGE_LABELS = mapOf(
    "probe" to "Reading video", "model" to "Loading model", "matte" to "Removing background",
    "analyse" to "Smoothing matte", "encode" to "Encoding sticker",
)

@Composable
private fun JobStatus(job: JobState, onDownload: () -> Unit) {
    when (job) {
        is JobState.Idle -> Unit
        is JobState.NeedsModel -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("This mode needs the ${job.kind.name.lowercase()} matting model (~175 MB), downloaded once and cached on-device.")
                Button(onClick = onDownload) { Text("Download model") }
            }
        }
        is JobState.Downloading -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Downloading model… ${job.percent}%")
                LinearProgressIndicator(progress = { job.percent / 100f }, modifier = Modifier.fillMaxWidth())
            }
        }
        is JobState.Running -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(STAGE_LABELS[job.stage] ?: job.stage)
                LinearProgressIndicator(progress = { job.fraction }, modifier = Modifier.fillMaxWidth())
            }
        }
        is JobState.Error -> Card(Modifier.fillMaxWidth()) {
            Text(
                "Couldn't make a sticker: ${job.message}",
                modifier = Modifier.padding(16.dp),
                color = MaterialTheme.colorScheme.error,
            )
        }
        is JobState.Done -> ResultCard(job)
    }
}

@Composable
private fun ResultCard(result: JobState.Done) {
    val context = LocalContext.current
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Sticker ready", style = MaterialTheme.typography.titleSmall)
            CheckerboardGifPreview(
                result.file,
                Modifier.fillMaxWidth().aspectRatio(result.width.toFloat() / result.height),
            )
            Text(
                "${result.frames} frames · ${"%.1f".format(result.fps)}fps · ${result.width}×${result.height} · matte: ${result.mode.name.lowercase()}",
                style = MaterialTheme.typography.bodySmall,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { saveToGallery(context, result.file) }) { Text("Save") }
                TextButton(onClick = { shareFile(context, result.file) }) { Text("Share") }
            }
        }
    }
}

/** Checkerboard-backed GIF preview: real animation on API 28+, first frame only below that. */
@Composable
private fun CheckerboardGifPreview(file: File, modifier: Modifier = Modifier) {
    Box(
        modifier.background(CheckerBrushColor),
        contentAlignment = Alignment.Center,
    ) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            AndroidView(factory = { ctx ->
                ImageView(ctx).apply {
                    val source = ImageDecoder.createSource(file)
                    val drawable = ImageDecoder.decodeDrawable(source)
                    setImageDrawable(drawable)
                    (drawable as? AnimatedImageDrawable)?.start()
                }
            })
        } else {
            val bmp = remember(file) { BitmapFactory.decodeFile(file.absolutePath) }
            AndroidView(factory = { ctx -> ImageView(ctx).apply { setImageBitmap(bmp) } })
        }
    }
}

private val CheckerBrushColor = androidx.compose.ui.graphics.Color(0xFFCCCCCC)

private fun Uri.lastPathSegmentOrSelf(): String = lastPathSegment ?: toString()

private fun saveToGallery(context: Context, file: File) {
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, file.name)
        put(MediaStore.Images.Media.MIME_TYPE, "image/gif")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/vidsticker")
        }
    }
    val resolver = context.contentResolver
    val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
    resolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
}

private fun shareFile(context: Context, file: File) {
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "image/gif"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "Share sticker"))
}
