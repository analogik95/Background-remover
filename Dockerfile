FROM python:3.11-slim

# ffmpeg does all the decoding; libgl/libglib are what opencv links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Most managed hosts (Hugging Face Spaces among them) run the container as a
# non-root uid, so everything the app writes has to live somewhere it owns.
RUN useradd -m -u 1000 app
ENV HOME=/home/app \
    U2NET_HOME=/home/app/.u2net \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY pyproject.toml README.md ./
COPY vidsticker ./vidsticker
RUN pip install --no-cache-dir .

RUN chown -R app:app /home/app /app
USER app

# Bake the models into the image. Fetched at run time they are a ~350 MB stall
# on whoever submits the first video.
RUN python -c "from rembg import new_session; \
    [new_session(m) for m in ('isnet-general-use', 'isnet-anime')]"

EXPOSE 7860

# One worker, many threads, on purpose: jobs and the progress streams that
# follow them live in process memory, so a second worker would field requests
# for jobs it cannot see. Threads are the right axis anyway - the work is a
# subprocess (ffmpeg) or numpy/onnx, both of which drop the GIL.
# --timeout 0 because a progress stream is a deliberately long-lived request.
CMD gunicorn --bind "0.0.0.0:$PORT" \
    --workers 1 --threads 16 --worker-class gthread --timeout 0 \
    "vidsticker.web:create_app()"
