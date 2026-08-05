# Putting vidsticker online

The background removal runs a segmentation model and shells out to ffmpeg, so it
needs a real server — there is no static-hosting version of this. Everything
here builds the same `Dockerfile` at the repo root.

## What it actually needs

| | |
|---|---|
| RAM | **2 GB minimum.** onnxruntime plus an isnet session sits around 1–1.5 GB. 512 MB tiers will OOM. |
| CPU | Matting is the whole cost: roughly **1 s per frame per core**. A 6 s clip at 25 fps is ~2 min on 2 vCPU. |
| Disk | ~2.5 GB image (the models are baked in), plus a few hundred MB of scratch per job. |
| Port | Reads `$PORT`, defaults to 7860. |

Run one worker. Jobs and their progress streams live in process memory, so a
second worker fields requests for jobs it cannot see. Scale with `--threads`
(already 16) and `VIDSTICKER_MAX_JOBS`, not with workers.

## Hugging Face Spaces — free, and the best fit

Free CPU Spaces get 2 vCPU and 16 GB RAM, which is the only free tier listed
here that comfortably clears the memory requirement.

1. Create a Space at <https://huggingface.co/new-space> → SDK **Docker** → blank.
2. Copy `deploy/huggingface/README.md` to the Space's `README.md`. The YAML
   header at the top is what tells Spaces to build the Dockerfile and which port
   to expose — without it the Space will not start.
3. Push this repo to the Space:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space HEAD:main
   ```

The first build takes ~10 minutes, most of it downloading the models. Spaces
sleep when idle and wake on the next request.

## Fly.io

```bash
fly launch --no-deploy --copy-config --config deploy/fly.toml
fly deploy
```

`deploy/fly.toml` asks for a 2 GB machine — below that it OOMs mid-matte. It
also sets `auto_stop_machines`, so an idle instance costs nothing and cold-starts
on the next request.

## Render

```bash
# Blueprint -> point at deploy/render.yaml
```

Note the plan in `deploy/render.yaml` is `standard`, not `free`: Render's free
tier is 512 MB and will be killed loading the model. If you want a free host,
use Spaces.

## Anywhere else

```bash
docker build -t vidsticker .
docker run -p 8000:7860 -e PORT=7860 vidsticker
```

That image runs unchanged on Cloud Run, Railway, a VPS, or your own machine.

## Before exposing it publicly

The app was written for local use and the defaults reflect that:

- **No authentication.** Anyone with the URL can burn your CPU. Put it behind an
  authenticating proxy, or your host's access control, if that matters.
- **Jobs are in-memory and unpersisted.** A restart loses in-flight work; results
  are reaped an hour after they finish (`VIDSTICKER_JOB_TTL`).
- **Uploads are capped at 256 MB** (`VIDSTICKER_MAX_UPLOAD_MB`) and **2 jobs run
  at once** (`VIDSTICKER_MAX_JOBS`). Both are worth lowering on a small box —
  each concurrent job wants a core and a few hundred MB.
- **Nothing is scanned.** Uploads are handed to ffmpeg by path; the filename from
  the client is reduced to its stem and never used to build a filesystem path.
