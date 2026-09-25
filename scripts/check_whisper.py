#!/usr/bin/env python3
"""Transcribe a local audio file with the configured Whisper model and run the full emotion analysis.

    WHISPER_MODEL=tiny python scripts/check_whisper.py path/to/radio.wav
    WHISPER_MODEL=base WHISPER_CACHE_DIR=/srv/whisper-models python scripts/check_whisper.py radio.flac

Needs openai-whisper (pip install -r requirements-whisper.txt) and ffmpeg on PATH. The first run
downloads the model (tiny is about 72 MB, base about 139 MB) into WHISPER_CACHE_DIR, or into
Whisper's default ~/.cache/whisper; later runs load it from there.

The script prints the configuration, the model load time, Whisper's transcription of the file
with its wall-clock time, the time of a first acoustic-only analysis, and the full
classify_emotion_detailed(path, transcribe=True, allow_local_paths=True) result as JSON. The
first analysis in a process pays one-time costs: librosa imports most of its modules lazily
(pulling in scipy.signal and scipy.stats, usually several seconds) and numba compiles a few
kernels. The model is loaded and those costs are paid by then, so the last time is the latency
of one analysis request with transcription on a warmed-up server.

.env is loaded unless F1_COPILOT_LOAD_DOTENV=0; variables that are already set always win.

Exit status: 0 success (including an empty transcript, reported as such); 1 the audio was
rejected, or loading the model or transcribing failed; 2 usage error or missing file; 3 Whisper
or ffmpeg is unavailable, WHISPER_MODEL is not an official model name, or the model cache
directory (WHISPER_CACHE_DIR or Whisper's default) cannot be used.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from core_modules.driver_emotion.emotion_classifier import (  # noqa: E402
    TranscriptionError,
    classify_emotion_detailed,
    get_transcriber,
)

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_UNAVAILABLE = 0, 1, 2, 3


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _whisper_version() -> str:
    try:
        return importlib.metadata.version("openai-whisper")
    except importlib.metadata.PackageNotFoundError:
        return "unknown version"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="local audio file (WAV, FLAC, OGG, MP3; M4A and others via ffmpeg)")
    args = parser.parse_args(argv)

    if os.getenv("F1_COPILOT_LOAD_DOTENV", "1") != "0":
        load_dotenv(PROJECT_ROOT / ".env")  # never overrides variables that are already set

    audio = args.audio.expanduser()
    if not audio.is_file():
        _error(f"{audio} is not an existing file")
        return EXIT_USAGE

    transcriber = get_transcriber()
    settings = transcriber.settings
    try:
        cache_dir = str(settings.model_cache_dir())
        if settings.cache_dir is None:
            cache_dir += " (Whisper's default; set WHISPER_CACHE_DIR to change it)"
    except ValueError as exc:  # availability() below reports it as the error
        cache_dir = f"{settings.cache_dir or 'Whisper default'} (unusable: {exc})"
    print(f"Whisper model:   {settings.model_name}")
    print(f"Model cache dir: {cache_dir}")
    print(f"Audio file:      {audio} ({audio.stat().st_size} bytes)")

    available, reason = transcriber.availability()
    if not available:
        _error(f"Whisper transcription is unavailable: {reason}")
        return EXIT_UNAVAILABLE

    try:
        started = time.perf_counter()
        model = transcriber.load()
        device = getattr(model, "device", "unknown device")
        print(f"Model load:      {time.perf_counter() - started:.2f} s on {device} "
              f"(openai-whisper {_whisper_version()}; includes the download on first use)")

        started = time.perf_counter()
        transcript = transcriber.transcribe(str(audio))
        print(f"Transcription:   {time.perf_counter() - started:.2f} s")
        print(f"Transcript:      {transcript!r}" if transcript else "Transcript:      (Whisper returned no text)")

        started = time.perf_counter()
        classify_emotion_detailed(str(audio), transcribe=False, allow_local_paths=True)
        print(f"Acoustic only:   {time.perf_counter() - started:.2f} s (first analysis in this process; includes "
              "librosa's one-time lazy imports of scipy and numba compilation)")

        started = time.perf_counter()
        result = classify_emotion_detailed(str(audio), transcribe=True, allow_local_paths=True)
        elapsed = time.perf_counter() - started
    except TranscriptionError as exc:
        _error(f"{exc}" + (f": {exc.__cause__!r}" if exc.__cause__ is not None else ""))
        return EXIT_FAILED
    except ValueError as exc:
        _error(f"the emotion analysis rejected the audio: {exc}")
        return EXIT_FAILED

    print(f"Full analysis:   {elapsed:.2f} s (classify_emotion_detailed, transcribe=True, warmed up)")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
