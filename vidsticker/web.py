"""Browser UI: drop a video in, watch it matte, download the sticker.

Jobs run on a worker thread and stream progress over server-sent events, so a
multi-minute conversion reports live instead of hanging on a request.
"""

from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from .matting import MatteConfig
from .pipeline import FORMATS, StickerOptions, create_sticker

MAX_UPLOAD_MB = 256
JOB_TTL_SECONDS = 60 * 60
STAGES = ["probe", "decode", "model", "matte", "analyse", "compose", "encode"]
# Rough share of total runtime per stage, so the bar advances evenly rather than
# sitting at 5% through the long matting pass.
STAGE_WEIGHT = {"probe": 1, "decode": 6, "model": 3, "matte": 65,
                "analyse": 6, "compose": 12, "encode": 7}


@dataclass
class Job:
    id: str
    name: str
    dir: Path
    created: float = field(default_factory=time.time)
    state: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[dict] = None
    listeners: List[queue.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        return {"id": self.id, "state": self.state, "stage": self.stage,
                "progress": round(self.progress, 3), "error": self.error,
                "result": self.result}

    def emit(self) -> None:
        payload = self.snapshot()
        with self.lock:
            for q in list(self.listeners):
                q.put(payload)


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _overall(stage: str, done: int, total: int) -> float:
    total_weight = sum(STAGE_WEIGHT.values())
    before = sum(STAGE_WEIGHT[s] for s in STAGES[:STAGES.index(stage)]) if stage in STAGES else 0
    frac = (done / total) if total else 1.0
    return min(1.0, (before + STAGE_WEIGHT.get(stage, 0) * frac) / total_weight)


def _reap_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with JOBS_LOCK:
        stale = [j for j in JOBS.values() if j.created < cutoff and j.state in ("done", "error")]
        for job in stale:
            JOBS.pop(job.id, None)
            shutil.rmtree(job.dir, ignore_errors=True)


def _run(job: Job, src: Path, opts: StickerOptions) -> None:
    def progress(stage: str, done: int, total: int) -> None:
        job.stage = stage
        job.progress = _overall(stage, done, total)
        job.emit()

    try:
        job.state = "running"
        job.emit()
        res = create_sticker(src, job.dir / "out", opts, progress=progress, stem=job.name)
        job.result = {
            "frames": res.frames, "fps": res.fps, "width": res.width, "height": res.height,
            "mode": res.mode,
            "key": [round(res.key[0]), round(res.key[1])] if res.key else None,
            "files": [
                {"format": fmt, "name": p.name, "bytes": p.stat().st_size,
                 "url": f"/api/jobs/{job.id}/file/{p.name}"}
                for fmt, p in sorted(res.outputs.items())
            ],
            "preview": f"/api/jobs/{job.id}/file/{res.preview.name}" if res.preview else None,
        }
        job.state, job.stage, job.progress = "done", "done", 1.0
    except Exception as exc:
        job.state, job.error = "error", str(exc)
    finally:
        job.emit()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=MAX_UPLOAD_MB)

    @app.post("/api/jobs")
    def submit():
        _reap_old_jobs()
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            return jsonify(error="no video file was uploaded"), 400

        form = request.form
        formats = [f for f in form.getlist("formats") if f in FORMATS] or ["gif", "webp"]

        def num(name, cast, default):
            raw = form.get(name)
            if raw in (None, ""):
                return default
            try:
                return cast(raw)
            except ValueError:
                return default

        opts = StickerOptions(
            size=num("size", int, 512),
            fps=num("fps", float, None),
            formats=formats,
            square=form.get("square") == "on",
            crop=form.get("crop", "on") == "on",
            smooth=form.get("smooth", "on") == "on",
            alpha_threshold=num("alpha_threshold", int, 128),
            matte=MatteConfig(
                mode=form.get("mode", "auto"),
                model=form.get("model", "isnet-general-use"),
                tolerance=num("tolerance", float, 8.0),
                softness=num("softness", float, 20.0),
                despill=num("despill", float, 0.8),
            ),
        )
        try:
            opts.validate()
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        job_id = uuid.uuid4().hex[:12]
        job_dir = Path(tempfile.mkdtemp(prefix=f"vidsticker-job-{job_id}-"))
        src = job_dir / "input" / Path(upload.filename).name
        src.parent.mkdir(parents=True)
        upload.save(src)

        # Only the stem is reused for output names; the extension and any path
        # components from the client are discarded.
        job = Job(id=job_id, name=Path(upload.filename).stem or "sticker", dir=job_dir)
        with JOBS_LOCK:
            JOBS[job_id] = job
        threading.Thread(target=_run, args=(job, src, opts), daemon=True).start()
        return jsonify(job.snapshot()), 202

    @app.get("/api/jobs/<job_id>")
    def status(job_id):
        job = JOBS.get(job_id) or abort(404)
        return jsonify(job.snapshot())

    @app.get("/api/jobs/<job_id>/events")
    def events(job_id):
        job = JOBS.get(job_id) or abort(404)

        def stream():
            q: queue.Queue = queue.Queue()
            with job.lock:
                job.listeners.append(q)
            try:
                yield f"data: {json.dumps(job.snapshot())}\n\n"
                while True:
                    try:
                        payload = q.get(timeout=15)
                    except queue.Empty:
                        yield ": keepalive\n\n"  # stops proxies closing an idle stream
                        continue
                    yield f"data: {json.dumps(payload)}\n\n"
                    if payload["state"] in ("done", "error"):
                        return
            finally:
                with job.lock:
                    if q in job.listeners:
                        job.listeners.remove(q)

        return Response(stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/<job_id>/file/<path:name>")
    def download(job_id, name):
        job = JOBS.get(job_id) or abort(404)
        out_dir = (job.dir / "out").resolve()
        target = (out_dir / name).resolve()
        # Reject anything that escapes the job's output directory.
        if not target.is_file() or out_dir not in target.parents:
            abort(404)
        return send_file(target, download_name=target.name)

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    app = create_app()
    print(f"vidsticker UI  ->  http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
