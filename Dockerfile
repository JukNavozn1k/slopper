# CUDA 12.8 + cuDNN runtime base so torch (cu128) and faster-whisper's
# CTranslate2 backend can run on the GPU. cu128 covers both Pascal (GTX 1080,
# sm_61) and Blackwell (RTX 5060, sm_120).
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 (deadsnakes), FFmpeg, OpenCV deps, Node.js (yt-dlp JS challenges)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# uv for dependency management (matches the local uv workflow)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1
ENV UV_PYTHON=python3.12
# Keep the venv outside /app so the compose bind-mount (.:/app) doesn't shadow it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (better layer caching) from the lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Always upgrade yt-dlp to latest (YouTube bot-detection changes frequently)
RUN uv pip install --upgrade yt-dlp

# Copy application code
COPY . .

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN mkdir -p /app/uploads /app/output /tmp/Ultralytics \
    && chown -R appuser:appuser /app /opt/venv /tmp/Ultralytics
USER appuser

# Pre-download YOLO model on build (running as appuser)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
