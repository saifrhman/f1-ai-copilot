# F1 AI Copilot: one image for the API (default command) and the web UI.
#
# Run it with Docker Compose from the project folder: docker-compose.yml starts the api, ui and qdrant
# services from this image, and the UI's fix steps use the same Compose commands.
#
#   mkdir -p data/fia_docs .cache outputs          # bind-mounted; create them before the first run
#   docker compose up -d --build                    # UI http://127.0.0.1:8501, API http://127.0.0.1:8000/docs
#   docker compose run --rm api python scripts/fetch_fia_regulations.py
#   docker compose run --rm api python scripts/build_fia_index.py --dry-run   # then without --dry-run
#   docker compose down                             # stop (the index stays in the qdrant volume)
#
# Settings and the API key are read from .env at runtime (.dockerignore keeps .env out of the image).
# Bind-mounted host folders must be writable by APP_UID: on Linux build with your user id
# (APP_UID=$(id -u) docker compose build) if it is not 1000.
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
