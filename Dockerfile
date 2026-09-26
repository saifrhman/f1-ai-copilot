# F1 AI Copilot: one image for the API (default command) and the web UI.
#
# Run it with Docker Compose from the project folder: docker-compose.yml runs the api and ui services
# from this image next to a qdrant/qdrant server; the commands are in its header and in README
# "Docker Compose". Settings and the API key are read from .env at runtime (.dockerignore keeps .env
# out of the image).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg decodes M4A/AAC radio clips (and is required by optional Whisper transcription).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-lock.txt ./
RUN pip install -r requirements-lock.txt

# UID 1000 matches the first user on most Linux hosts, so bind-mounted data/, outputs/ and .cache/
# stay writable; override with --build-arg APP_UID=$(id -u).
ARG APP_UID=1000
RUN useradd --create-home --uid "${APP_UID}" app

# Code is owned by root (read-only for the app user); only data, index, cache and output dirs are writable.
COPY . .
RUN mkdir -p data outputs .cache .qdrant \
    && chown app:app data outputs .cache .qdrant
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
