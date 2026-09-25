"""Driver-radio emotion analysis: input security, decoding, honest features, Whisper status, text keywords."""

import base64
import io
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import types
import urllib.error
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from core_modules.driver_emotion import emotion_classifier as ec
from core_modules.driver_emotion.emotion_classifier import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_INPUT_CHARS,
    EmotionType,
    TextEmotionEvidence,
    TranscriptionError,
    classify_emotion_detailed,
    classify_text_emotion,
    combine_evidence,
)
from core_modules.driver_emotion.schemas import AudioFeatures, EmotionRequest, EmotionResponse

EMOTION_VALUES = {emotion.value for emotion in EmotionType}


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def speech_like(sr, seconds=1.5, f0=160.0, amplitude=0.3, snr_db=None, seed=0, channels=1, syllables_per_s=3.0):
    """Harmonic voice with slow intonation drift, syllable amplitude modulation and pauses.

    ``syllables_per_s=0`` gives continuous voicing without pauses.
    """

    rng = np.random.default_rng(seed)
    t = np.arange(int(round(sr * seconds))) / sr
    frequency = f0 * (1.0 + 0.06 * np.sin(2 * math.pi * 0.7 * t))
    phase = 2 * math.pi * np.cumsum(frequency) / sr
    voice = sum(0.6 ** k * np.sin(k * phase) for k in range(1, 8))
    envelope = np.clip(np.sin(2 * math.pi * syllables_per_s * t), 0.0, None) ** 0.5 if syllables_per_s else 1.0
    y = amplitude * voice * envelope / np.max(np.abs(voice))
    if snr_db is not None:
        signal_rms = np.sqrt(np.mean(y ** 2))
        y = y + rng.standard_normal(y.size) * signal_rms / 10 ** (snr_db / 20)
    if channels == 2:
        y = np.stack([y, 0.8 * y + 0.002 * rng.standard_normal(y.size)], axis=1)
    return np.clip(y, -1.0, 1.0)


def encode(y, sr, fmt="WAV"):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def wav_bytes(y, sr):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    return buffer.getvalue()


def assert_finite_features(result):
    features = result["audio_features"]
    assert "tempo" not in features
    assert all(math.isfinite(value) for value in features.values()), features
    assert features["mean_pitch"] > 0 and features["rms_energy"] > 0
    assert "voiced_fraction" not in features  # replaced by names that state their denominator
    voiced_of_clip = features["voiced_fraction_of_clip"]
    non_silent_of_clip = features["non_silent_fraction_of_clip"]
    voiced_of_non_silent = features["voiced_fraction_of_non_silent"]
    assert 0.0 < voiced_of_clip <= non_silent_of_clip <= 1.0
    assert ec.MIN_VOICED_SHARE <= voiced_of_non_silent <= 1.0
    assert voiced_of_clip == pytest.approx(voiced_of_non_silent * non_silent_of_clip)
    assert result["emotion"] in EMOTION_VALUES
    assert 0.0 <= result["confidence"] <= 0.95
    EmotionResponse.model_validate(result)  # documented response contract, extra keys forbidden
    json.dumps(result, allow_nan=False)


@pytest.fixture(autouse=True)
def _fresh_transcriber(monkeypatch):
    monkeypatch.setattr(ec, "_transcriber", None)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_CACHE_DIR", raising=False)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# whisper.available_models() of openai-whisper 20250625
OFFICIAL_WHISPER_MODELS = [
    "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
    "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo", "turbo",
]
# File names openai-whisper 20250625 stores the models under (the basename of each download URL).
WHISPER_CHECKPOINT_FILES = {
    **{name: f"{name}.pt" for name in OFFICIAL_WHISPER_MODELS},
    "large": "large-v3.pt",
    "turbo": "large-v3-turbo.pt",
}


def install_fake_whisper(monkeypatch, tmp_path, transcribe, load_failures=0, device="cpu"):
    """Stand-in for openai-whisper (real models are 72 MB+ downloads) plus an ffmpeg on PATH.

    It follows the real package's contract: ``load_model`` rejects names missing from
    ``available_models()``, runs ``os.makedirs(download_root, exist_ok=True)`` and reads the model
    file named in ``_MODELS`` from ``download_root`` (or $XDG_CACHE_HOME/whisper, redirected into
    tmp_path, when none is given), "downloading" it first when it is missing. The first
    ``load_failures`` loads fail like an offline download (urllib's URLError). Every load prints
    a line to stdout, so tests can count loads and see when they happen relative to other output.
    Like the real model, it runs in FP16 unless asked for FP32 (recorded in ``precisions``) and,
    on the CPU, warns when it was not asked for FP32.
    """

    module = types.ModuleType("whisper")
    remaining_failures = [load_failures]

    class _Model:
        def __init__(self):
            self.device = types.SimpleNamespace(type=device)
            self.precisions = []

        def transcribe(self, path, **options):
            fp16 = options.get("fp16", True)
            if fp16 and device == "cpu":
                warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            self.precisions.append("fp16" if fp16 and device != "cpu" else "fp32")
            return transcribe(path)

    def load_model(name, device=None, download_root=None, in_memory=False):
        if name not in OFFICIAL_WHISPER_MODELS:
            raise RuntimeError(f"Model {name} not found; available models = {OFFICIAL_WHISPER_MODELS}")
        print(f"fake whisper: loading model {name!r}")
        if remaining_failures[0] > 0:
            remaining_failures[0] -= 1
            raise urllib.error.URLError(socket.gaierror(-3, "Temporary failure in name resolution"))
        if download_root is None:
            download_root = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "whisper")
        os.makedirs(download_root, exist_ok=True)
        checkpoint = Path(download_root) / module._MODELS[name].rsplit("/", 1)[-1]
        if not checkpoint.is_file():
            checkpoint.write_bytes(b"checkpoint")
        checkpoint.read_bytes()
        return _Model()

    module._MODELS = {name: f"https://models.invalid/whisper/{file}" for name, file in WHISPER_CHECKPOINT_FILES.items()}
    module.available_models = lambda: list(OFFICIAL_WHISPER_MODELS)
    module.load_model = load_model
    monkeypatch.setitem(sys.modules, "whisper", module)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    ffmpeg = bin_dir / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\nexit 0\n")
    ffmpeg.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))


# ---------------------------------------------------------------------------
# Realistic audio succeeds with finite, meaningful features
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sr", [16000, 22050, 44100, 48000])
def test_speech_like_audio_is_analysed_at_common_sample_rates(sr):
    result = classify_emotion_detailed(encode(speech_like(sr), sr))

    assert_finite_features(result)
    assert result["audio_features"]["mean_pitch"] == pytest.approx(160.0, rel=0.05)
    assert result["duration"] == pytest.approx(1.5, abs=0.01)
    assert result["source_sample_rate"] == sr
    assert result["transcription"] is None
    assert result["transcription_status"] == "not_requested"
    assert result["transcription_unavailable_reason"] is None
    assert result["evidence_combination"] == "acoustic_only"
    assert "heuristic" in result["classifier"] and "Heuristic" in result["disclaimer"]


@pytest.mark.parametrize(
    "fmt, sr, channels, snr_db",
    [("WAV", 16000, 2, 5.0), ("FLAC", 44100, 2, 10.0), ("OGG", 48000, 1, 5.0), ("MP3", 44100, 1, 10.0)],
)
def test_noisy_stereo_and_compressed_speech_is_analysed(fmt, sr, channels, snr_db):
    audio = speech_like(sr, seconds=2.0, f0=140.0, snr_db=snr_db, channels=channels, seed=sr)
    result = classify_emotion_detailed(encode(audio, sr, fmt))

    assert_finite_features(result)
    assert result["audio_features"]["mean_pitch"] == pytest.approx(140.0, rel=0.1)
    assert result["duration"] == pytest.approx(2.0, abs=0.1)


def test_pitch_feature_follows_the_speaker():
    low = classify_emotion_detailed(encode(speech_like(22050, f0=110.0), 22050))
    high = classify_emotion_detailed(encode(speech_like(22050, f0=260.0), 22050))

    assert low["audio_features"]["mean_pitch"] == pytest.approx(110.0, rel=0.05)
    assert high["audio_features"]["mean_pitch"] == pytest.approx(260.0, rel=0.05)


def test_half_second_pure_tone_is_still_accepted():
    t = np.arange(int(22050 * 0.5)) / 22050
    result = classify_emotion_detailed(encode(0.15 * np.sin(2 * math.pi * 220 * t), 22050))
    assert result["audio_features"]["mean_pitch"] == pytest.approx(220.0, rel=0.03)


def test_voiced_fractions_state_their_denominators():
    # Reviewer repro: 3 s of digital silence + 1 s of continuous voicing. Only ~25% of the clip is voiced;
    # the old single "voiced_fraction" key reported the non-silent share (~0.9) as if it were the clip share.
    sr = 22050
    voice = speech_like(sr, seconds=1.0, f0=180.0, syllables_per_s=0)
    alone = classify_emotion_detailed(encode(voice, sr))["audio_features"]
    padded = classify_emotion_detailed(encode(np.concatenate([np.zeros(3 * sr), voice]), sr))["audio_features"]

    assert alone["voiced_fraction_of_clip"] > 0.9
    assert padded["voiced_fraction_of_clip"] == pytest.approx(0.25, abs=0.03)
    assert padded["non_silent_fraction_of_clip"] == pytest.approx(0.25, abs=0.03)
    # The share of the non-silent audio that is voiced (the noise gate) is unaffected by silence padding.
    assert padded["voiced_fraction_of_non_silent"] > 0.9
    assert padded["voiced_fraction_of_non_silent"] == pytest.approx(alone["voiced_fraction_of_non_silent"], abs=0.05)
    # Clip-wide averages are diluted by silence, as the schema documents; pitch (voiced frames only) is not.
    assert padded["rms_energy"] == pytest.approx(alone["rms_energy"] / 4, rel=0.15)
    assert padded["mean_pitch"] == pytest.approx(alone["mean_pitch"], rel=0.01)


def test_response_schema_documents_every_feature_denominator():
    fields = AudioFeatures.model_fields
    assert "NON-SILENT" in fields["voiced_fraction_of_non_silent"].description
    assert "all frames of the clip" in fields["voiced_fraction_of_clip"].description
    assert "voiced frames only" in fields["mean_pitch"].description
    assert "ALL frames" in fields["rms_energy"].description
    assert all(field.description for field in fields.values())

    features = classify_emotion_detailed(encode(speech_like(22050), 22050))["audio_features"]
    assert set(features) == set(fields)
    for bad in ({**features, "voiced_fraction": 0.5}, {**features, "voiced_fraction_of_clip": 1.5}, {**features, "rms_energy": math.nan}):
        with pytest.raises(ValidationError):
            AudioFeatures.model_validate(bad)


def test_concurrent_requests_match_sequential_results():
    clips = [encode(speech_like(sr, f0=f0), sr) for sr, f0 in [(16000, 120.0), (22050, 180.0), (44100, 230.0), (48000, 150.0)]]
    sequential = [classify_emotion_detailed(clip) for clip in clips]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(classify_emotion_detailed, clips))
    assert concurrent == sequential


# ---------------------------------------------------------------------------
# Honest rejection instead of a "calm" label
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "audio",
    [np.zeros(22050 * 2), 1e-5 * np.random.default_rng(0).standard_normal(22050)],
    ids=["digital-silence", "near-silence"],
)
def test_silent_audio_is_rejected(audio):
    with pytest.raises(ValueError, match="silent"):
        classify_emotion_detailed(encode(audio, 22050))


@pytest.mark.parametrize("sr", [16000, 44100])
def test_noise_without_voiced_speech_is_rejected(sr):
    noise = 0.3 * np.random.default_rng(sr).standard_normal(sr * 3)
    with pytest.raises(ValueError, match="voiced speech"):
        classify_emotion_detailed(encode(noise, sr))


def test_a_few_voiced_frames_inside_long_noise_are_rejected_by_the_voiced_share_gate():
    sr = 22050
    voice = speech_like(sr, seconds=0.6, f0=180.0, syllables_per_s=0)
    noise = 0.05 * np.random.default_rng(3).standard_normal(sr * 6)
    buried = np.concatenate([noise[: 3 * sr], voice, noise[3 * sr:]])

    with pytest.raises(ValueError, match="voiced speech") as error:
        ec.AudioFeatureExtractor.features_from_waveform(buried, sr)
    voiced_frames, share_percent = map(int, re.search(r"(\d+) voiced frames \((\d+)%", str(error.value)).groups())
    # Enough voiced frames for the frame-count gate: only the share gate rejects this clip.
    assert voiced_frames >= ec.MIN_VOICED_FRAMES
    assert share_percent < ec.MIN_VOICED_SHARE * 100

    # The same voiced segment inside less noise clears the share gate and is analysed.
    features = ec.AudioFeatureExtractor.features_from_waveform(np.concatenate([noise[: sr // 2], voice, noise[sr // 2: sr]]), sr)
    assert features["voiced_fraction_of_non_silent"] >= ec.MIN_VOICED_SHARE
    assert features["mean_pitch"] == pytest.approx(180.0, rel=0.1)


@pytest.mark.parametrize("samples", [1, 220, int(22050 * 0.3), int(22050 * 0.49)])
def test_short_clips_are_rejected(samples):
    t = np.arange(samples) / 22050
    with pytest.raises(ValueError, match="too short"):
        classify_emotion_detailed(encode(0.2 * np.sin(2 * math.pi * 200 * t), 22050))


def test_out_of_band_features_never_score_positively():
    classifier = ec.get_emotion_classifier()
    assert ec._band_similarity(0.0, 90, 210) == 0.0
    assert ec._band_similarity(420.0, 90, 210) == 0.0
    silence_like = {"mean_pitch": 0.0, "pitch_std": 0.0, "rms_energy": 0.0, "energy_std": 0.0}
    with pytest.raises(ValueError, match="mean_pitch"):
        classifier.profile_scores(silence_like)
    with pytest.raises(ValueError, match="rms_energy"):
        classifier.profile_scores({"mean_pitch": 150.0, "pitch_std": 10.0, "rms_energy": 0.0, "energy_std": 0.0})


# ---------------------------------------------------------------------------
# Input decoding and security
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [b"hello", np.random.default_rng(7).bytes(4096), b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(64), b"\x89PNG\r\n\x1a\n" + bytes(512)],
    ids=["text", "random", "fake-riff", "png"],
)
def test_base64_of_non_audio_is_rejected_with_value_error(payload):
    with pytest.raises(ValueError, match="Could not decode audio"):
        classify_emotion_detailed(base64.b64encode(payload).decode())


def test_data_uris_must_be_base64_audio():
    wav = encode(speech_like(22050), 22050)
    for media_type in ("text/plain", "image/png", "application/octet-stream", ""):
        with pytest.raises(ValueError, match=r"audio/\*"):
            classify_emotion_detailed(f"data:{media_type};base64,{wav}")
    with pytest.raises(ValueError, match="base64"):
        classify_emotion_detailed(f"data:audio/wav,{wav}")

    assert_finite_features(classify_emotion_detailed(f"data:audio/wav;base64,{wav}"))
    assert_finite_features(classify_emotion_detailed(f"DATA:Audio/WAV;codecs=1;BASE64,{wav}"))


def test_line_wrapped_base64_is_accepted():
    raw = wav_bytes(speech_like(22050), 22050)
    mime_style = base64.encodebytes(raw).decode()  # 76-column lines, as produced by GNU base64
    assert "\n" in mime_style
    assert_finite_features(classify_emotion_detailed(mime_style))
    assert_finite_features(classify_emotion_detailed("data:audio/wav;base64,\r\n" + mime_style.replace("\n", "\r\n")))


@pytest.mark.parametrize("value", ["", "   \n", "this-is-not-a-path-or-base64", "abc", "%%%%"])
def test_invalid_strings_are_rejected(value):
    with pytest.raises(ValueError):
        classify_emotion_detailed(value)


def test_server_paths_are_never_opened_by_default(tmp_path, monkeypatch):
    existing = tmp_path / "radio.wav"
    existing.write_bytes(wav_bytes(speech_like(22050), 22050))
    missing = tmp_path / "ghost.wav"
    messages = []
    for candidate in (existing, missing, tmp_path, "/etc/passwd"):
        with pytest.raises(ValueError) as error:
            classify_emotion_detailed(str(candidate))
        messages.append(str(error.value))
    assert len(set(messages)) == 1, messages  # identical: no file-existence oracle
    assert "file paths are not accepted" in messages[0]

    # A relative name that is also valid base64 is decoded as base64, never looked up on disk.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "radiowav").write_bytes(existing.read_bytes())
    oracle = []
    for name in ("radiowav", "ghostwav"):
        with pytest.raises(ValueError) as error:
            classify_emotion_detailed(name)
        oracle.append(str(error.value))
    assert oracle[0] == oracle[1]
    assert "Could not decode audio" in oracle[0]
    assert "radiowav" not in oracle[0] and str(tmp_path) not in oracle[0]


def test_local_paths_work_only_when_explicitly_allowed(tmp_path):
    path = tmp_path / "radio.flac"
    sf.write(path, speech_like(44100, f0=150.0), 44100, format="FLAC")

    result = classify_emotion_detailed(str(path), allow_local_paths=True)
    assert_finite_features(result)
    assert result["audio_features"]["mean_pitch"] == pytest.approx(150.0, rel=0.05)
    assert ec.classify_emotion(str(path), allow_local_paths=True) in EMOTION_VALUES
    with pytest.raises(ValueError):
        ec.classify_emotion(str(path))

    for unusable in (tmp_path / "missing.wav", tmp_path):
        with pytest.raises(ValueError, match="existing audio file path"):
            classify_emotion_detailed(str(unusable), allow_local_paths=True)

    not_audio = tmp_path / "notes.wav"
    not_audio.write_text("definitely not audio")
    with pytest.raises(ValueError, match="Could not decode audio"):
        classify_emotion_detailed(str(not_audio), allow_local_paths=True)

    huge = tmp_path / "huge.wav"
    with open(huge, "wb") as handle:  # sparse file: no disk space used
        handle.truncate(MAX_AUDIO_BYTES + 1)
    with pytest.raises(ValueError, match="too large"):
        classify_emotion_detailed(str(huge), allow_local_paths=True)


def test_oversized_input_is_rejected_before_decoding():
    with pytest.raises(ValueError, match="too large"):
        classify_emotion_detailed("A" * (MAX_AUDIO_INPUT_CHARS + 1))
    oversized = base64.b64encode(bytes(MAX_AUDIO_BYTES + 3)).decode()
    assert len(oversized) < MAX_AUDIO_INPUT_CHARS  # only the decoded-size limit applies here
    with pytest.raises(ValueError, match="too large"):
        classify_emotion_detailed(oversized)


def test_clips_longer_than_the_duration_limit_are_rejected():
    audio = speech_like(8000, seconds=ec.MAX_DURATION_S + 1.0)
    with pytest.raises(ValueError, match="longer than"):
        classify_emotion_detailed(encode(audio, 8000))


def test_low_sample_rates_are_rejected():
    with pytest.raises(ValueError, match="sample rate"):
        classify_emotion_detailed(encode(speech_like(4000, seconds=1.0, f0=150.0), 4000))


# ---------------------------------------------------------------------------
# Decompression bombs: declared stream parameters are checked before decoding
# ---------------------------------------------------------------------------
def silent_flac(sr, channels, seconds):
    buffer = io.BytesIO()
    with sf.SoundFile(buffer, "w", samplerate=sr, channels=channels, format="FLAC", subtype="PCM_16") as handle:
        block = np.zeros((sr // 4, channels), dtype=np.int16)
        for _ in range(int(seconds * 4)):
            handle.write(block)
    return buffer.getvalue()


@pytest.fixture
def decoder_calls(monkeypatch):
    """Record (and refuse) every librosa decode, to prove a rejection happened before decoding."""

    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("audio was decoded before its declared parameters were checked")

    monkeypatch.setattr(ec.librosa, "load", refuse)
    return calls


def test_tiny_high_rate_multichannel_flac_is_rejected_without_decoding():
    # Reviewer repro: <1 MB of FLAC at 655350 Hz x 8 channels x 119 s made librosa.load(sr=None)
    # allocate ~3 GB. Two seconds are enough to show it: ~11 kB that decode to ~250 MB and ~2 s of work.
    data = silent_flac(655350, 8, 2.0)
    assert len(data) < 50_000
    payload = base64.b64encode(data).decode()

    started = time.perf_counter()
    with pytest.raises(ValueError, match="8 channels; only mono or stereo"):
        classify_emotion_detailed(payload)
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize(
    "data, message",
    [
        (lambda: silent_flac(655350, 1, 1.0), "655350 Hz is above the supported maximum of 96000 Hz"),
        (lambda: silent_flac(192000, 2, 1.0), "192000 Hz is above the supported maximum"),
        (lambda: silent_flac(48000, 3, 1.0), "3 channels"),
        (lambda: wav_bytes(np.zeros(4000), 4000), "4000 Hz is below the supported minimum"),
        (lambda: wav_bytes(np.zeros(8000 * 122), 8000), "longer than the 120 s limit"),
    ],
    ids=["655k-mono", "192k-stereo", "3-channels", "4k", "122s"],
)
def test_declared_stream_parameters_are_rejected_before_decoding(decoder_calls, data, message):
    with pytest.raises(ValueError, match=message):
        classify_emotion_detailed(base64.b64encode(data()).decode())
    assert decoder_calls == []


def test_stereo_at_the_maximum_sample_rate_is_accepted():
    result = classify_emotion_detailed(encode(speech_like(ec.MAX_SOURCE_SAMPLE_RATE, channels=2), ec.MAX_SOURCE_SAMPLE_RATE))
    assert_finite_features(result)
    assert result["source_sample_rate"] == ec.MAX_SOURCE_SAMPLE_RATE


MP4_HEADER = b"\x00\x00\x00\x18ftypM4A \x00\x00\x02\x00" + bytes(64)  # routed to ffmpeg, unreadable by soundfile


class _FakeFFmpegReader:
    """Stand-in for audioread's FFmpegAudioFile: the stream header ffmpeg would report, then PCM blocks."""

    def __init__(self, samplerate, channels, duration, blocks=()):
        self.samplerate, self.channels, self.duration = samplerate, channels, duration
        self.blocks_read, self.closed = 0, False
        self._blocks = blocks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def __iter__(self):
        for block in self._blocks:
            self.blocks_read += 1
            yield block


def _install_fake_ffmpeg(monkeypatch, reader):
    import audioread.ffdec

    monkeypatch.setattr(audioread.ffdec, "FFmpegAudioFile", lambda path: reader)


def test_ffmpeg_containers_are_checked_before_their_pcm_is_read(monkeypatch):
    def refuse():
        raise AssertionError("PCM was read before the declared stream was checked")
        yield  # pragma: no cover

    reader = _FakeFFmpegReader(655350, 8, 119.0, refuse())
    _install_fake_ffmpeg(monkeypatch, reader)
    with pytest.raises(ValueError, match="8 channels"):
        classify_emotion_detailed(base64.b64encode(MP4_HEADER).decode())
    assert reader.closed and reader.blocks_read == 0

    reader = _FakeFFmpegReader(44100, 2, 300.0, refuse())
    _install_fake_ffmpeg(monkeypatch, reader)
    with pytest.raises(ValueError, match="longer than"):
        classify_emotion_detailed(base64.b64encode(MP4_HEADER).decode())


def test_ffmpeg_read_is_byte_limited_when_the_duration_is_not_declared(monkeypatch):
    # WebM from browsers often reports "Duration: N/A" (0); the read itself must stop at the limit.
    block = (0.1 * 32767 * np.sin(2 * math.pi * 150 * np.arange(2048) / 8000)).astype("<i2").tobytes()

    def endless():
        while True:
            yield block

    reader = _FakeFFmpegReader(8000, 1, 0.0, endless())
    _install_fake_ffmpeg(monkeypatch, reader)
    with pytest.raises(ValueError, match="longer than"):
        classify_emotion_detailed(base64.b64encode(MP4_HEADER).decode())
    limit_bytes = (ec.MAX_DURATION_S + 0.5) * 8000 * 2
    assert reader.blocks_read <= limit_bytes / len(block) + 1
    assert reader.closed


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_real_ffmpeg_containers_are_decoded_or_rejected_by_their_stream_header(tmp_path):
    def m4a(audio, sr):
        source, target = tmp_path / "in.wav", tmp_path / "out.m4a"
        sf.write(source, audio, sr)
        done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:a", "aac", str(target)], capture_output=True)
        if done.returncode != 0:
            pytest.skip("this ffmpeg build has no AAC encoder")
        return base64.b64encode(target.read_bytes()).decode()

    stereo = classify_emotion_detailed(m4a(speech_like(44100, channels=2), 44100))
    assert_finite_features(stereo)
    assert stereo["source_sample_rate"] == 44100
    assert stereo["audio_features"]["mean_pitch"] == pytest.approx(160.0, rel=0.05)

    eight = np.tile(speech_like(48000)[:, None], (1, 8))
    with pytest.raises(ValueError, match="8 channels"):
        classify_emotion_detailed(m4a(eight, 48000))


def test_temporary_files_are_always_removed(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(ec.tempfile, "tempdir", str(scratch))

    classify_emotion_detailed(encode(speech_like(22050), 22050))
    for bad in (base64.b64encode(b"hello").decode(), encode(np.zeros(22050), 22050), encode(np.zeros(10), 22050)):
        with pytest.raises(ValueError):
            classify_emotion_detailed(bad)
    assert list(scratch.iterdir()) == []


# ---------------------------------------------------------------------------
# Whisper: explicit status, never a fabricated transcript
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attributes", [{}, {"load_model": lambda name: None}], ids=["graphite-whisper", "no-available-models"])
def test_unrelated_whisper_package_is_reported_unavailable(monkeypatch, attributes):
    # The PyPI "whisper" package (Graphite's database) installs a module without load_model.
    module = types.ModuleType("whisper")
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "whisper", module)
    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)

    assert result["transcription_status"] == "unavailable"
    assert "load_model" in result["transcription_unavailable_reason"]
    assert result["transcription"] is None
    assert result["text_emotion"] is None and result["text_confidence"] is None
    assert result["emotion"] == result["acoustic_emotion"]
    assert result["evidence_combination"] == "acoustic_only"
    EmotionResponse.model_validate(result)


def test_missing_whisper_is_reported_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper", None)  # makes "import whisper" raise ImportError
    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "unavailable"
    assert "not installed" in result["transcription_unavailable_reason"]
    assert result["transcription"] is None


def test_missing_ffmpeg_is_reported_unavailable(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "should never be used"})
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "unavailable"
    assert "ffmpeg" in result["transcription_unavailable_reason"]
    assert result["transcription"] is None


def test_whisper_runtime_failure_is_raised_not_swallowed(monkeypatch, tmp_path):
    def broken(path):
        raise RuntimeError("CUDA out of memory")

    install_fake_whisper(monkeypatch, tmp_path, broken)
    with pytest.raises(TranscriptionError, match="Whisper transcription failed"):
        classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)


def test_whisper_transcript_is_reported_and_combined(monkeypatch, tmp_path):
    seen = []

    def transcribe(path):
        seen.append((path, os.path.getsize(path)))
        return {"text": "  No grip at the rear, the car is terrible and I'm so frustrated.  "}

    install_fake_whisper(monkeypatch, tmp_path, transcribe)
    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)

    assert result["transcription_status"] == "completed"
    assert result["transcription"] == "No grip at the rear, the car is terrible and I'm so frustrated."
    assert result["text_emotion"] == "frustrated"
    assert sorted(result["text_keyword_hits"]["frustrated"]) == ["frustrated", "no grip", "terrible"]
    assert "Whisper" in result["classifier"]
    assert seen and seen[0][1] > 0  # Whisper received the real decoded upload
    assert not os.path.exists(seen[0][0])  # ... which was removed afterwards
    EmotionResponse.model_validate(result)


def test_empty_whisper_transcript_is_reported_as_empty(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "   "})
    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "empty"
    assert result["transcription"] is None
    assert result["text_emotion"] is None
    assert result["emotion"] == result["acoustic_emotion"]


def test_single_keyword_does_not_override_acoustic_label(monkeypatch, tmp_path):
    clip = encode(speech_like(22050), 22050)
    acoustic = classify_emotion_detailed(clip)["emotion"]
    keyword = "furious" if acoustic != "angry" else "brilliant"
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": f"I am {keyword}"})

    result = classify_emotion_detailed(clip, transcribe=True)
    assert result["text_emotion"] not in (None, acoustic, "neutral")
    assert result["emotion"] == acoustic
    assert result["evidence_combination"] == "acoustic_kept_text_disagrees"
    assert result["confidence"] < result["acoustic_confidence"]


def test_transcribe_flag_must_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe="false")


def test_confidences_and_profile_scores_are_rounded_to_three_decimals(monkeypatch, tmp_path):
    # Reviewer repro: 0.2 + 0.15 * 3 keyword hits was reported as 0.6499999999999999.
    assert classify_text_emotion("brilliant, amazing, fantastic").confidence == 0.65
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Brilliant, amazing, fantastic!"})
    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)

    assert result["text_emotion"] == "excited" and result["text_confidence"] == 0.65
    scores = [result["confidence"], result["acoustic_confidence"], result["text_confidence"]]
    scores += list(result["acoustic_profile_scores"].values())
    assert all(score == round(score, 3) for score in scores), scores
    for evidence in (TextEmotionEvidence(EmotionType.ANGRY, 0.65, 3), TextEmotionEvidence(EmotionType.CALM, 0.35, 1)):
        for acoustic in (0.1234567, 0.6543219):
            confidence = combine_evidence(EmotionType.CALM, acoustic, evidence)[1]
            assert confidence == round(confidence, 3)


# ---------------------------------------------------------------------------
# Whisper configuration: WHISPER_MODEL and WHISPER_CACHE_DIR
# ---------------------------------------------------------------------------
def test_whisper_settings_are_read_from_the_environment(tmp_path):
    assert ec.WhisperSettings.from_env({}) == ec.WhisperSettings("base", None)
    assert ec.WhisperSettings.from_env({"WHISPER_MODEL": "  ", "WHISPER_CACHE_DIR": ""}) == ec.WhisperSettings("base", None)

    settings = ec.WhisperSettings.from_env({"WHISPER_MODEL": " tiny.en ", "WHISPER_CACHE_DIR": " models/whisper "})
    assert settings.model_name == "tiny.en"
    assert settings.download_root() == PROJECT_ROOT / "models" / "whisper"  # relative: the project root
    assert ec.WhisperSettings(cache_dir="~/whisper-models").download_root() == Path.home() / "whisper-models"
    assert ec.WhisperSettings(cache_dir=str(tmp_path / "new")).download_root() == tmp_path / "new"  # created on download
    assert ec.WhisperSettings().download_root() is None  # Whisper's own default


def test_configured_model_is_downloaded_into_whisper_cache_dir(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Copy, understood."})
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(tmp_path / "models"))

    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "completed"
    assert result["transcription"] == "Copy, understood."
    assert sorted(path.name for path in (tmp_path / "models").iterdir()) == ["tiny.pt"]
    assert not (tmp_path / "xdg-cache").exists()  # Whisper's default cache was not used
    assert ec.get_transcriber().model_name == "tiny"


def test_default_model_and_cache_are_used_when_unset(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "All good"})
    monkeypatch.setenv("WHISPER_MODEL", "")  # an empty value (e.g. "WHISPER_MODEL=" in .env) counts as unset

    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "completed"
    assert sorted(path.name for path in (tmp_path / "xdg-cache" / "whisper").iterdir()) == ["base.pt"]


def test_unknown_whisper_model_is_reported_unavailable_without_downloading(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "should never be used"})
    monkeypatch.setenv("WHISPER_MODEL", "bse")
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(tmp_path / "models"))

    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "unavailable"
    reason = result["transcription_unavailable_reason"]
    assert reason.startswith("WHISPER_MODEL 'bse' is not a Whisper model; use one of: tiny.en, tiny, base.en, base,")
    assert result["transcription"] is None and result["evidence_combination"] == "acoustic_only"
    EmotionResponse.model_validate(result)
    assert not (tmp_path / "models").exists()

    transcriber = ec.get_transcriber()
    assert transcriber.availability() == (False, reason)
    for action in (transcriber.load, lambda: transcriber.transcribe("unused.wav")):
        with pytest.raises(TranscriptionError, match="unavailable: WHISPER_MODEL 'bse'"):
            action()


def test_whisper_model_names_are_matched_exactly(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "unused"})
    for name in ("Base", "tiny en", "large_v3", "x" * 500):
        monkeypatch.setattr(ec, "_transcriber", None)
        monkeypatch.setenv("WHISPER_MODEL", name)
        available, reason = ec.get_transcriber().availability()
        assert not available and "is not a Whisper model" in reason
        assert len(reason) < 300  # long values are not echoed in full


@pytest.mark.parametrize(
    "value",
    ["{tmp}/custom-tiny.pt", "models/whisper/tiny", "custom.PT", "~/whisper/base.pt", "C:\\models\\tiny"],
    ids=["existing-checkpoint", "relative-path", "pt-suffix", "home-path", "windows-path"],
)
def test_whisper_model_file_paths_are_rejected_without_echoing_them(monkeypatch, tmp_path, value):
    # whisper.load_model() itself would load a local checkpoint; only official names are supported here,
    # and the configured path must not reach API clients through the unavailable reason.
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "should never be used"})
    (tmp_path / "custom-tiny.pt").write_bytes(b"checkpoint")
    value = value.replace("{tmp}", str(tmp_path))
    monkeypatch.setenv("WHISPER_MODEL", value)

    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "unavailable"
    reason = result["transcription_unavailable_reason"]
    assert reason.startswith("WHISPER_MODEL looks like a file path; local checkpoint files are not supported.")
    assert reason.endswith(", ".join(OFFICIAL_WHISPER_MODELS))
    assert value not in reason and str(tmp_path) not in reason and "custom" not in reason


def _unusable_cache_dir(tmp_path, kind):
    regular_file = tmp_path / "cache.txt"
    regular_file.write_text("not a directory")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    return {
        "regular-file": str(regular_file),
        "below-a-file": str(regular_file / "whisper-models"),
        "broken-symlink": str(tmp_path / "dangling"),
        "below-a-broken-symlink": str(tmp_path / "dangling" / "whisper-models"),
        "unknown-user-home": "~no-such-user-f1copilot/models",
    }[kind]


@pytest.mark.parametrize(
    "kind, message",
    [
        ("regular-file", "WHISPER_CACHE_DIR exists but is not a directory"),
        ("below-a-file", "WHISPER_CACHE_DIR cannot be created: part of its path is a file, not a directory"),
        ("broken-symlink", "WHISPER_CACHE_DIR is a broken symbolic link"),
        ("below-a-broken-symlink", "WHISPER_CACHE_DIR cannot be created: part of its path is a broken symbolic link"),
        ("unknown-user-home", "WHISPER_CACHE_DIR starts with '~' but that home directory cannot be determined"),
    ],
)
def test_unusable_whisper_cache_dir_is_reported_unavailable(monkeypatch, tmp_path, capsys, kind, message):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "should never be used"})
    monkeypatch.setenv("WHISPER_CACHE_DIR", _unusable_cache_dir(tmp_path, kind))

    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "unavailable"
    assert result["transcription_unavailable_reason"] == message  # no server path in the API response
    transcriber = ec.get_transcriber()
    with pytest.raises(TranscriptionError, match=f"unavailable: {re.escape(message)}"):
        transcriber.load()
    assert "fake whisper: loading" not in capsys.readouterr().out  # no load or download was attempted
    assert not (tmp_path / "xdg-cache").exists()


@contextmanager
def _permissions(path, mode, denied_access):
    """chmod ``path`` for the block; skip when the user bypasses permissions (root ignores them)."""

    original = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode)
    try:
        if os.access(path, denied_access):
            pytest.skip("file permissions are not enforced for this user")
        yield
    finally:
        path.chmod(original)


def test_whisper_cache_dir_below_a_read_only_directory_is_reported_unavailable(monkeypatch, tmp_path, capsys):
    # Reviewer repro: WHISPER_CACHE_DIR=/usr/share/f1-whisper-models was "ready" while every load failed.
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "should never be used"})
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(read_only / "models" / "whisper"))

    with _permissions(read_only, 0o555, os.W_OK):
        result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "unavailable"
    assert result["transcription_unavailable_reason"] == (
        "WHISPER_CACHE_DIR cannot be created: its nearest existing parent directory is not writable"
    )
    assert "fake whisper: loading" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "locked_part, message",
    [
        ("parent", "WHISPER_CACHE_DIR cannot be accessed (PermissionError)"),
        ("cache-dir", "WHISPER_CACHE_DIR cannot be accessed (the directory has no search permission)"),
    ],
)
def test_whisper_cache_dir_without_search_permission_is_reported_unavailable(monkeypatch, tmp_path, locked_part, message):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "should never be used"})
    parent = tmp_path / "parent"
    (parent / "models").mkdir(parents=True)
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(parent / "models"))
    locked = parent if locked_part == "parent" else parent / "models"

    with _permissions(locked, 0o600, os.X_OK):  # readable and writable, but not searchable
        available, reason = ec.get_transcriber().availability()  # must not raise PermissionError
    assert (available, reason) == (False, message)


def test_read_only_whisper_cache_dir_works_only_when_it_already_holds_the_model(monkeypatch, tmp_path, capsys):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Copy, understood."})
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(models))
    monkeypatch.setenv("WHISPER_MODEL", "turbo")
    clip = encode(speech_like(22050), 22050)

    with _permissions(models, 0o555, os.W_OK):
        missing = classify_emotion_detailed(clip, transcribe=True)
    assert missing["transcription_status"] == "unavailable"
    assert missing["transcription_unavailable_reason"] == (
        "WHISPER_CACHE_DIR is not writable and does not hold the model file large-v3-turbo.pt yet, "
        "so the model cannot be downloaded into it"
    )
    assert "fake whisper: loading" not in capsys.readouterr().out

    (models / "large-v3-turbo.pt").write_bytes(b"checkpoint")  # pre-populated, e.g. baked into an image
    with _permissions(models, 0o555, os.W_OK):
        result = classify_emotion_detailed(clip, transcribe=True)
    assert result["transcription_status"] == "completed"
    assert result["transcription"] == "Copy, understood."


def test_unusable_default_whisper_cache_is_reported_with_a_hint(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "should never be used"})
    regular_file = tmp_path / "not-a-dir"
    regular_file.write_text("")
    monkeypatch.setenv("XDG_CACHE_HOME", str(regular_file))  # Whisper would use <file>/whisper

    result = classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    assert result["transcription_status"] == "unavailable"
    assert result["transcription_unavailable_reason"] == (
        "Whisper's default model cache directory ($XDG_CACHE_HOME/whisper or ~/.cache/whisper) cannot be created: "
        "part of its path is a file, not a directory; set WHISPER_CACHE_DIR to a usable directory"
    )

    monkeypatch.setenv("WHISPER_CACHE_DIR", str(tmp_path / "models"))  # the documented way out
    monkeypatch.setattr(ec, "_transcriber", None)
    assert ec.get_transcriber().availability() == (True, None)


def test_model_load_failure_is_explicit_and_retried(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Brakes gone, no brakes!"}, load_failures=1)
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    clip = encode(speech_like(22050), 22050)

    with pytest.raises(TranscriptionError, match=r"Loading the Whisper model 'tiny' failed \(URLError\)") as error:
        classify_emotion_detailed(clip, transcribe=True)
    assert "downloading the model failed: the first use of a model needs network access" in str(error.value)
    # The failure is not cached: once the download works, the same process transcribes.
    result = classify_emotion_detailed(clip, transcribe=True)
    assert result["transcription_status"] == "completed"
    assert result["text_emotion"] == "panicked"


def test_unreadable_model_file_is_not_blamed_on_the_network(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "should never be used"})
    models = tmp_path / "models"
    models.mkdir()
    checkpoint = models / "tiny.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(models))
    monkeypatch.setenv("WHISPER_MODEL", "tiny")

    with _permissions(checkpoint, 0o000, os.R_OK):
        with pytest.raises(TranscriptionError) as error:
            classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)
    message = str(error.value)
    assert message.startswith("Loading the Whisper model 'tiny' failed (PermissionError); Whisper could not create, read")
    assert "check that directory's permissions" in message
    assert "network" not in message and str(tmp_path) not in message


def test_loaded_model_is_cached_and_reused(monkeypatch, tmp_path, capsys):
    # A real model load takes seconds (tiny) to minutes (large): it must happen once per process.
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Copy"})
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    clip = encode(speech_like(22050), 22050)

    for _ in range(3):
        assert classify_emotion_detailed(clip, transcribe=True)["transcription"] == "Copy"
    transcriber = ec.get_transcriber()
    model = transcriber.load()
    assert transcriber.load() is model
    assert model.precisions == ["fp32"] * 3  # this one model object ran all three transcriptions
    assert capsys.readouterr().out.count("fake whisper: loading model 'tiny'") == 1


def test_cpu_transcription_asks_for_fp32_instead_of_warning_on_every_call(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Copy"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            assert classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)["transcription"] == "Copy"
    assert not [w for w in caught if "FP16" in str(w.message)]
    assert ec.get_transcriber().load().precisions == ["fp32", "fp32"]


def test_gpu_transcription_keeps_whispers_fp16_default(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Copy"}, device="cuda")
    assert classify_emotion_detailed(encode(speech_like(22050), 22050), transcribe=True)["transcription"] == "Copy"
    assert ec.get_transcriber().load().precisions == ["fp16"]


# ---------------------------------------------------------------------------
# scripts/check_whisper.py
# ---------------------------------------------------------------------------
def _radio_file(tmp_path):
    path = tmp_path / "radio.wav"
    path.write_bytes(wav_bytes(speech_like(22050), 22050))
    return path


def test_check_whisper_script_prints_transcript_timing_and_full_result(monkeypatch, tmp_path, capsys):
    from scripts.check_whisper import main

    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": " No grip, the car is terrible. "})
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(tmp_path / "models"))

    assert main([str(_radio_file(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "Whisper model:   tiny\n" in out
    assert f"Model cache dir: {tmp_path / 'models'}\n" in out
    for timing in ("Model load", "Transcription", "Acoustic only", "Full analysis"):
        assert re.search(rf"^{timing}: +\d+\.\d\d s", out, re.M), timing
    # The model is loaded inside the timed "Model load" step (not during the transcription), and
    # only once: the transcription and the full analysis reuse it.
    assert out.count("fake whisper: loading model 'tiny'") == 1
    assert out.index("fake whisper: loading model 'tiny'") < out.index("Model load:")
    assert "lazy imports of scipy and numba compilation" in out
    assert "Transcript:      'No grip, the car is terrible.'" in out
    result = json.loads(out[out.index("\n{") + 1:])
    EmotionResponse.model_validate(result)
    assert result["transcription_status"] == "completed"
    assert result["transcription"] == "No grip, the car is terrible."
    assert result["text_emotion"] == "frustrated"
    assert (tmp_path / "models" / "tiny.pt").exists()


def test_check_whisper_script_fails_clearly_without_whisper_or_a_valid_model(monkeypatch, tmp_path, capsys):
    from scripts.check_whisper import main

    audio = str(_radio_file(tmp_path))
    monkeypatch.setitem(sys.modules, "whisper", None)
    assert main([audio]) == 3
    assert "Whisper transcription is unavailable: openai-whisper is not installed" in capsys.readouterr().err

    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "unused"})
    monkeypatch.setattr(ec, "_transcriber", None)
    monkeypatch.setenv("WHISPER_MODEL", "huge")
    assert main([audio]) == 3
    assert "WHISPER_MODEL 'huge' is not a Whisper model" in capsys.readouterr().err

    # Reviewer repro: WHISPER_CACHE_DIR=/etc/passwd/whisper-models used to fail late, with exit status 1.
    monkeypatch.setattr(ec, "_transcriber", None)
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("WHISPER_CACHE_DIR", _unusable_cache_dir(tmp_path, "below-a-file"))
    assert main([audio]) == 3
    captured = capsys.readouterr()
    assert "WHISPER_CACHE_DIR cannot be created: part of its path is a file, not a directory" in captured.err
    assert "fake whisper: loading" not in captured.out

    assert main([str(tmp_path / "missing.wav")]) == 2
    assert "is not an existing file" in capsys.readouterr().err


def test_check_whisper_script_reads_dotenv_only_when_enabled(monkeypatch, tmp_path, capsys):
    import scripts.check_whisper as script

    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Copy"})
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("WHISPER_MODEL=tiny\nF1_CHECK_WHISPER_DOTENV_PROBE=loaded\n")
    monkeypatch.setattr(script, "PROJECT_ROOT", project)  # never the developer's real .env
    for name in ("WHISPER_MODEL", "F1_CHECK_WHISPER_DOTENV_PROBE"):
        monkeypatch.setenv(name, "")  # records the original state, so teardown also removes what .env adds
        monkeypatch.delenv(name)
    audio = str(_radio_file(tmp_path))

    monkeypatch.setenv("F1_COPILOT_LOAD_DOTENV", "0")
    environment = dict(os.environ)
    assert script.main([audio]) == 0
    assert dict(os.environ) == environment
    assert "Whisper model:   base\n" in capsys.readouterr().out

    monkeypatch.setenv("F1_COPILOT_LOAD_DOTENV", "1")
    monkeypatch.setattr(ec, "_transcriber", None)
    assert script.main([audio]) == 0
    assert "Whisper model:   tiny\n" in capsys.readouterr().out
    assert os.environ["F1_CHECK_WHISPER_DOTENV_PROBE"] == "loaded"


def test_check_whisper_script_reports_rejected_audio_and_whisper_failures(monkeypatch, tmp_path, capsys):
    from scripts.check_whisper import main

    install_fake_whisper(monkeypatch, tmp_path, lambda path: {"text": "Thank you."})
    silent = tmp_path / "silent.wav"
    silent.write_bytes(wav_bytes(np.zeros(22050), 22050))
    assert main([str(silent)]) == 1
    captured = capsys.readouterr()
    assert "Transcript:      'Thank you.'" in captured.out  # what Whisper made of it is still shown
    assert "the emotion analysis rejected the audio: Audio is silent" in captured.err

    def broken(path):
        raise RuntimeError("CUDA out of memory")

    (tmp_path / "second").mkdir()
    install_fake_whisper(monkeypatch, tmp_path / "second", broken)
    monkeypatch.setattr(ec, "_transcriber", None)
    assert main([str(_radio_file(tmp_path))]) == 1
    assert "Whisper transcription failed (RuntimeError): RuntimeError('CUDA out of memory')" in capsys.readouterr().err


def test_check_whisper_script_runs_standalone_from_any_directory(tmp_path):
    # Without ffmpeg on PATH it must stop before any model download, whether or not Whisper is installed.
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    env = {**os.environ, "PATH": str(empty_bin), "F1_COPILOT_LOAD_DOTENV": "0", "XDG_CACHE_HOME": str(tmp_path / "xdg")}
    done = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "check_whisper.py"), str(_radio_file(tmp_path))],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 3, done.stderr
    assert "error: Whisper transcription is unavailable: " in done.stderr
    assert "Whisper model:   base" in done.stdout
    assert not (tmp_path / "xdg" / "whisper").exists()


# ---------------------------------------------------------------------------
# Transcript keyword heuristic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("I made it through turn one", "neutral"),  # "made" is not "mad"
        ("My eyes are watering in this heat", "neutral"),  # "eyes" is not "yes"
        ("That was helpful information", "neutral"),
        ("I'm mad, absolutely furious", "angry"),
        ("MAD!!", "angry"),
        ("Yes! Brilliant, get in there!", "excited"),
        ("No, I'm happy with it", "excited"),  # "No," closes its own clause
        ("No grip at the rear, the car is terrible", "frustrated"),
        ("No brakes! No brakes!", "panicked"),  # phrases that start with "no" are not negated
        ("Copy, understood, staying steady", "focused"),
        ("All good, tyres feel fine", "calm"),
        ("Great lap but the tyres are terrible", "neutral"),  # tie is ambiguous
        ("", "neutral"),
    ],
)
def test_text_keywords_use_whole_words_and_phrases(text, expected):
    assert classify_text_emotion(text).emotion.value == expected


@pytest.mark.parametrize(
    "text, negated",
    [
        ("I'm not happy with the balance", ["happy"]),
        ("I don't think it's great", ["great"]),
        ("Never calm in this traffic", ["calm"]),
        ("I’m not very happy", ["happy"]),
    ],
)
def test_negated_keywords_are_ignored(text, negated):
    evidence = classify_text_emotion(text)
    assert evidence.emotion is EmotionType.NEUTRAL
    assert evidence.confidence == 0.0
    assert evidence.negated_keywords == negated


def test_text_confidence_grows_with_net_hits_and_is_capped():
    one = classify_text_emotion("brilliant")
    two = classify_text_emotion("brilliant, amazing")
    many = classify_text_emotion("brilliant amazing fantastic awesome great yes happy")
    assert (one.net_hits, two.net_hits) == (1, 2)
    assert one.confidence < two.confidence < many.confidence <= 0.85


def test_evidence_combination_rules():
    single_angry = TextEmotionEvidence(EmotionType.ANGRY, 0.35, 1)
    strong_angry = TextEmotionEvidence(EmotionType.ANGRY, 0.5, 2)
    calm_text = TextEmotionEvidence(EmotionType.CALM, 0.35, 1)
    neutral_text = TextEmotionEvidence(EmotionType.NEUTRAL, 0.0, 0)

    assert combine_evidence(EmotionType.CALM, 0.5, None) == (EmotionType.CALM, 0.5, "acoustic_only")
    assert combine_evidence(EmotionType.CALM, 0.5, neutral_text) == (EmotionType.CALM, 0.5, "acoustic_only")
    # A single keyword never flips the label, even when its score beats the acoustic one.
    emotion, confidence, rule = combine_evidence(EmotionType.CALM, 0.3, single_angry)
    assert (emotion, rule) == (EmotionType.CALM, "acoustic_kept_text_disagrees")
    assert confidence == pytest.approx(0.125)
    emotion, confidence, rule = combine_evidence(EmotionType.CALM, 0.3, strong_angry)
    assert (emotion, rule) == (EmotionType.ANGRY, "text_overrides_acoustic")
    assert confidence == pytest.approx(0.35)
    emotion, _, rule = combine_evidence(EmotionType.CALM, 0.6, strong_angry)
    assert (emotion, rule) == (EmotionType.CALM, "acoustic_kept_text_disagrees")
    emotion, confidence, rule = combine_evidence(EmotionType.CALM, 0.5, calm_text)
    assert (emotion, rule) == (EmotionType.CALM, "text_agrees")
    assert confidence == pytest.approx(0.6)
    assert combine_evidence(EmotionType.CALM, 0.93, calm_text)[1] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
def test_emotion_request_schema_limits():
    request = EmotionRequest(audio_file="UklGRg==")
    assert request.transcribe is False

    with pytest.raises(ValidationError):
        EmotionRequest(audio_file="UklGRg==", transcribe=False, path="/etc/passwd")
    with pytest.raises(ValidationError):
        EmotionRequest(audio_file="")
    with pytest.raises(ValidationError):
        EmotionRequest(audio_file="A" * (MAX_AUDIO_INPUT_CHARS + 1))
    with pytest.raises(ValidationError):
        EmotionRequest(audio_file="UklGRg==", transcribe=[1])
    assert 27_000_000 < MAX_AUDIO_INPUT_CHARS < 29_000_000


@pytest.mark.parametrize("value", ["yes", "on", "true", "false", "1", 1, 0, 1.0, None])
def test_emotion_request_transcribe_accepts_only_json_booleans(value):
    # The router rejects non-booleans too; the endpoint used to coerce "yes"/1/"on" to True.
    with pytest.raises(ValidationError, match="transcribe"):
        EmotionRequest(audio_file="UklGRg==", transcribe=value)


def test_emotion_request_transcribe_booleans_round_trip():
    assert EmotionRequest(audio_file="UklGRg==", transcribe=True).transcribe is True
    assert EmotionRequest.model_validate_json('{"audio_file": "UklGRg==", "transcribe": false}').transcribe is False
    with pytest.raises(ValidationError):
        EmotionRequest.model_validate_json('{"audio_file": "UklGRg==", "transcribe": "yes"}')
    assert EmotionRequest.model_json_schema()["properties"]["transcribe"]["type"] == "boolean"


def test_rejected_non_audio_does_not_leak_file_handles(tmp_path):
    import gc
    import warnings

    from core_modules.driver_emotion.emotion_classifier import load_waveform

    # A fake WAV header or PNG bytes used to reach librosa's audioread fallback, whose raw
    # reader opens the file and raises without closing it.
    payloads = [b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(64), b"\x89PNG\r\n\x1a\n" + bytes(512)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for index, payload in enumerate(payloads):
            junk = tmp_path / f"junk{index}.audio"
            junk.write_bytes(payload)
            with pytest.raises(ValueError, match="Could not decode audio"):
                load_waveform(str(junk))
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


def test_an_exact_tie_for_the_best_profile_is_neutral_not_the_first_listed():
    tie = {ec.EmotionType.CALM: 1.0, ec.EmotionType.FOCUSED: 1.0, ec.EmotionType.ANGRY: 0.2}
    assert ec.EmotionClassifier.label_from_scores(tie) == (ec.EmotionType.NEUTRAL, 0.0)
    reordered = dict(reversed(list(tie.items())))
    assert ec.EmotionClassifier.label_from_scores(reordered) == (ec.EmotionType.NEUTRAL, 0.0)
    # A lead, however small, still names the profile: 0.45 x 1.0 + 0.55 x 0.01.
    emotion, confidence = ec.EmotionClassifier.label_from_scores({ec.EmotionType.CALM: 1.0, ec.EmotionType.FOCUSED: 0.99})
    assert emotion is ec.EmotionType.CALM and confidence == pytest.approx(0.4555, abs=1e-3)


def test_disclaimer_states_how_the_acoustic_confidence_is_computed():
    assert "0.45 x the best profile similarity + 0.55 x its lead over the runner-up" in ec.DISCLAIMER
    assert "not a probability" in ec.DISCLAIMER
