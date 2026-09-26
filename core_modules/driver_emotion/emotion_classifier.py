#!/usr/bin/env python3
"""Driver-radio emotion analysis: a transparent acoustic heuristic plus optional Whisper keywords.

This is NOT a trained or validated emotion model. The acoustic label comes from hand-set
pitch/energy bands (heuristic profiles) and the optional text label from keyword matching
on a local Whisper transcript (heuristic). Confidence values are heuristic scores built from
profile similarities and margins, not probabilities.

Input handling (security):
- ``allow_local_paths=False`` (the default, used by the HTTP API): only raw base64 or
  ``data:audio/<type>;base64,...`` URIs are accepted. The filesystem is never consulted, so
  every unusable input gets a message that depends only on the submitted string.
- ``allow_local_paths=True``: trusted Python callers may also pass an existing file path.

Submitted data is limited to ``MAX_AUDIO_BYTES``. Before any sample is decoded, the stream
parameters declared in the header are checked (at most ``MAX_CHANNELS`` channels,
``MIN_SOURCE_SAMPLE_RATE``-``MAX_SOURCE_SAMPLE_RATE`` Hz, at most ``MAX_DURATION_S`` long), so a
small compressed file cannot declare gigabytes of PCM. Clips shorter than ``MIN_DURATION_S``,
silent clips and clips without enough voiced speech are rejected with ``ValueError`` rather than
being given a label.
"""

import base64
import binascii
import errno
import http.client
import importlib
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import urllib.error
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import librosa
import numpy as np
import soundfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_AUDIO_BYTES = 20 * 1024 * 1024
_MAX_BASE64_CHARS = 4 * ((MAX_AUDIO_BYTES + 2) // 3)
# Allows MIME-style CRLF line breaks every 76 characters plus a data-URI header (~28.7M chars).
MAX_AUDIO_INPUT_CHARS = _MAX_BASE64_CHARS + 2 * (_MAX_BASE64_CHARS // 76) + 1024
MAX_DURATION_S = 120.0
MIN_DURATION_S = 0.5
MIN_SOURCE_SAMPLE_RATE = 8000
# Declared stream parameters are checked before decoding: together with MAX_DURATION_S they cap
# the decoded PCM at about 92 MB of float32, whatever the (compressed) upload size.
MAX_SOURCE_SAMPLE_RATE = 96000
MAX_CHANNELS = 2
_DURATION_READ_MARGIN_S = 0.5  # decode slightly past the limit so an over-long clip is detected
_MAX_LOCAL_PATH_CHARS = 4096

# ---------------------------------------------------------------------------
# Analysis parameters (heuristic)
# ---------------------------------------------------------------------------
ANALYSIS_SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 512
SILENCE_RMS_FLOOR = 1e-3  # about -60 dBFS
PITCH_FMIN_HZ = 60.0
PITCH_FMAX_HZ = 800.0
VOICING_NCCF_THRESHOLD = 0.5
VOICING_MAX_OCTAVE_JUMP = 0.1  # max f0 change between adjacent frames (octaves, ~1.2 semitones)
MIN_VOICED_FRAMES = 10  # about 0.23 s of voiced speech
MIN_VOICED_SHARE = 0.2  # share of non-silent frames that must be voiced
NEUTRAL_CONFIDENCE_THRESHOLD = 0.20
MAX_CONFIDENCE = 0.95
SCORE_DECIMALS = 3  # confidences and profile scores are heuristic margins; more digits are noise
# Acoustic confidence = weight x the best profile similarity + weight x its lead over the runner-up.
ACOUSTIC_SIMILARITY_WEIGHT = 0.45
ACOUSTIC_LEAD_WEIGHT = 0.55
ACOUSTIC_CONFIDENCE_RULE = (
    f"{ACOUSTIC_SIMILARITY_WEIGHT} × the best profile similarity + {ACOUSTIC_LEAD_WEIGHT} × its lead over the "
    f"runner-up, at most {MAX_CONFIDENCE}; below {NEUTRAL_CONFIDENCE_THRESHOLD:.2f}, or when two profiles tie for "
    "the best similarity, the acoustic label is neutral"
)

_INVALID_INPUT_MESSAGE = (
    "audio_file must be base64-encoded audio or a data:audio/<type>;base64,<data> URI "
    "(server file paths are not accepted)"
)
_INVALID_INPUT_MESSAGE_LOCAL = "audio_file must be an existing audio file path, base64-encoded audio or a data:audio/* URI"
_TOO_LARGE_MESSAGE = f"audio_file is too large: at most {MAX_AUDIO_BYTES // (1024 * 1024)} MiB of audio is accepted"
_DECODE_ERROR_MESSAGE = (
    "Could not decode audio: the data is not a supported, intact audio file "
    "(WAV, FLAC, OGG and MP3 are supported; other containers such as M4A need ffmpeg)"
)
_TOO_LONG_MESSAGE = f"Audio is longer than the {MAX_DURATION_S:.0f} s limit; send a shorter radio clip"
DISCLAIMER = (
    "Heuristic analysis: acoustic pitch/energy profiles and transcript keyword matching. "
    "Not a validated emotion model; each confidence is a heuristic score, not a probability "
    "(acoustic_confidence_rule and evidence_combination_rule say how it is computed)."
)


class EmotionType(Enum):
    CALM = "calm"
    ANGRY = "angry"
    PANICKED = "panicked"
    FOCUSED = "focused"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"


class TranscriptionError(RuntimeError):
    """Whisper was available but failed while loading the model or transcribing."""


@dataclass(frozen=True)
class TextEmotionEvidence:
    """Keyword evidence from a transcript (heuristic). ``net_hits`` = winner hits minus runner-up hits."""

    emotion: EmotionType
    confidence: float
    net_hits: int
    keyword_hits: Dict[str, List[str]] = field(default_factory=dict)
    negated_keywords: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input decoding
# ---------------------------------------------------------------------------
def _existing_local_file(value: str) -> Optional[str]:
    """Return ``value`` as a path if it names an existing regular file (trusted callers only)."""

    if value[:5].lower() == "data:" or len(value) > _MAX_LOCAL_PATH_CHARS:
        return None
    try:
        path = Path(value).expanduser()
        if not path.is_file():
            return None
        size = path.stat().st_size
    except (OSError, ValueError):
        return None
    if size > MAX_AUDIO_BYTES:
        raise ValueError(_TOO_LARGE_MESSAGE)
    return str(path)


def _decode_base64_audio(value: str, invalid_message: str) -> bytes:
    """Decode raw base64 or a ``data:audio/*;base64,`` URI (whitespace and line breaks ignored)."""

    payload = value
    if value[:5].lower() == "data:":
        header, separator, payload = value.partition(",")
        media_type, *params = [part.strip().lower() for part in header[5:].split(";")]
        if not separator or not media_type.startswith("audio/"):
            raise ValueError("Data URIs must have an audio/* media type, e.g. data:audio/wav;base64,<data>")
        if "base64" not in params:
            raise ValueError("Audio data URIs must be base64 encoded: data:audio/<type>;base64,<data>")

    compact = "".join(payload.split())
    if not compact:
        raise ValueError("audio_file contains no audio data")
    if len(compact) > _MAX_BASE64_CHARS:
        raise ValueError(_TOO_LARGE_MESSAGE)
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(invalid_message) from exc
    if not decoded:
        raise ValueError("audio_file contains no audio data")
    if len(decoded) > MAX_AUDIO_BYTES:
        raise ValueError(_TOO_LARGE_MESSAGE)
    return decoded


@contextmanager
def audio_input_file(audio_input: str, allow_local_paths: bool = False) -> Iterator[str]:
    """Yield a readable path for ``audio_input``; any temporary file is always removed afterwards."""

    if not isinstance(audio_input, str):
        raise ValueError("audio_file must be a string")
    value = audio_input.strip()
    if not value:
        raise ValueError("audio_file cannot be empty")
    if len(value) > MAX_AUDIO_INPUT_CHARS:
        raise ValueError(_TOO_LARGE_MESSAGE)

    if allow_local_paths:
        local_path = _existing_local_file(value)
        if local_path is not None:
            yield local_path
            return

    data = _decode_base64_audio(value, _INVALID_INPUT_MESSAGE_LOCAL if allow_local_paths else _INVALID_INPUT_MESSAGE)
    fd, temp_path = tempfile.mkstemp(prefix="f1-radio-", suffix=".audio")
    try:
        with open(fd, "wb") as handle:
            handle.write(data)
        yield temp_path
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temp_path)


def _check_sample_rate(sample_rate: int) -> None:
    if sample_rate < MIN_SOURCE_SAMPLE_RATE:
        raise ValueError(f"Audio sample rate {sample_rate} Hz is below the supported minimum of {MIN_SOURCE_SAMPLE_RATE} Hz")
    if sample_rate > MAX_SOURCE_SAMPLE_RATE:
        raise ValueError(f"Audio sample rate {sample_rate} Hz is above the supported maximum of {MAX_SOURCE_SAMPLE_RATE} Hz")


def _check_declared_stream(sample_rate: int, channels: int, duration: Optional[float]) -> None:
    """Reject header-declared parameters that would make decoding expensive, before any decoding.

    A few kilobytes of FLAC can declare minutes of 8-channel audio at 655 kHz (gigabytes of PCM),
    so the declared channel count, sample rate and duration are bounded first. ``duration`` is
    None when the container does not declare it; the duration-limited read then bounds the work.
    """

    if channels < 1:
        raise ValueError(_DECODE_ERROR_MESSAGE)
    if channels > MAX_CHANNELS:
        raise ValueError(f"Audio has {channels} channels; only mono or stereo (at most {MAX_CHANNELS}) is supported")
    _check_sample_rate(sample_rate)
    if duration is not None and duration > MAX_DURATION_S + _DURATION_READ_MARGIN_S:
        raise ValueError(_TOO_LONG_MESSAGE)


def _open_soundfile(path: str) -> Optional[soundfile.SoundFile]:
    try:
        return soundfile.SoundFile(path)
    except (RuntimeError, OSError):  # soundfile.LibsndfileError is a RuntimeError
        return None


def _needs_ffmpeg_container(path: str) -> bool:
    """MP4/M4A (``ftyp`` box), WebM/Matroska (EBML) or ADTS AAC, which only ffmpeg decodes here."""

    with open(path, "rb") as handle:
        head = handle.read(12)
    return head[4:8] == b"ftyp" or head[:4] == b"\x1a\x45\xdf\xa3" or head[:2] in (b"\xff\xf1", b"\xff\xf9")


@contextmanager
def _decoding_errors() -> Iterator[None]:
    """Report any decoder failure (soundfile, audioread/ffmpeg, EOFError, ...) as "not decodable"."""

    try:
        yield
    except Exception as exc:
        logger.debug("Audio decoding failed: %s", type(exc).__name__)
        raise ValueError(_DECODE_ERROR_MESSAGE) from exc


def _decode_with_soundfile(sound_file: soundfile.SoundFile, read_seconds: float) -> Tuple[np.ndarray, int]:
    with sound_file:
        rate = int(sound_file.samplerate)
        _check_declared_stream(rate, int(sound_file.channels), sound_file.frames / rate if rate > 0 else None)
        with _decoding_errors():
            # librosa reads from the checked handle itself, so no other decoder is consulted.
            return librosa.load(sound_file, sr=None, mono=True, duration=read_seconds)


def _decode_with_ffmpeg(path: str, read_seconds: float) -> Tuple[np.ndarray, int]:
    """Decode an ffmpeg-only container after checking the stream parameters ffmpeg reports.

    Decoding buffers PCM at the native sample rate and channel count (resampling happens later),
    so a lower target sample rate alone would not bound memory: the declared-stream checks and
    the byte limit of the read below do.
    """

    with _decoding_errors():
        from audioread.ffdec import FFmpegAudioFile  # librosa dependency; needs the ffmpeg binary

        reader = FFmpegAudioFile(path)  # starts ffmpeg and parses the stream header it prints
    with reader:  # closing kills ffmpeg, so an early stop costs no further decoding
        rate, channels = int(reader.samplerate), int(reader.channels)
        declared = float(reader.duration)  # 0 when the container does not declare a duration
        _check_declared_stream(rate, channels, declared if declared > 0 else None)
        frame_bytes = 2 * channels  # ffmpeg writes interleaved signed 16-bit little-endian PCM
        limit = int(read_seconds * rate) * frame_bytes
        pcm = bytearray()
        with _decoding_errors():
            for block in reader:
                pcm += block[: limit - len(pcm)]
                if len(pcm) >= limit:
                    break
    frames = np.frombuffer(pcm, dtype="<i2", count=(len(pcm) // frame_bytes) * channels)
    samples = frames.reshape(-1, channels).astype(np.float32).mean(axis=1) / np.float32(32768.0)
    return samples, rate


def load_waveform(path: str, target_sr: int = ANALYSIS_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Decode ``path`` to mono float samples at ``target_sr``; returns ``(samples, source_sample_rate)``.

    The declared stream parameters are checked before decoding (see ``_check_declared_stream``).
    Raises ValueError for undecodable data, more than ``MAX_CHANNELS`` channels, clips longer than
    ``MAX_DURATION_S`` or shorter than ``MIN_DURATION_S``, sample rates outside
    ``MIN_SOURCE_SAMPLE_RATE``-``MAX_SOURCE_SAMPLE_RATE`` and non-finite samples.
    """

    # Decode at most slightly more than the limit so over-long uploads cost bounded work.
    read_seconds = MAX_DURATION_S + _DURATION_READ_MARGIN_S
    sound_file = _open_soundfile(path)
    if sound_file is not None:
        samples, source_sr = _decode_with_soundfile(sound_file, read_seconds)
    elif _needs_ffmpeg_container(path):
        samples, source_sr = _decode_with_ffmpeg(path, read_seconds)
    else:
        # Arbitrary bytes never reach audioread (it leaks a file handle on garbage input).
        raise ValueError(_DECODE_ERROR_MESSAGE)

    source_sr = int(source_sr)
    if samples.size == 0 or source_sr <= 0:
        raise ValueError("Audio contains no samples")
    duration = samples.size / source_sr
    if duration > MAX_DURATION_S:
        raise ValueError(_TOO_LONG_MESSAGE)
    if duration < MIN_DURATION_S:
        raise ValueError(f"Audio clip is too short ({duration:.3f} s); at least {MIN_DURATION_S} s is required")
    _check_sample_rate(source_sr)
    if not np.all(np.isfinite(samples)):
        raise ValueError("Decoded audio contains non-finite samples")
    if source_sr != target_sr:
        samples = librosa.resample(samples, orig_sr=source_sr, target_sr=target_sr)
    return samples, source_sr


# ---------------------------------------------------------------------------
# Acoustic features
# ---------------------------------------------------------------------------
def _nccf_at_lags(frames: np.ndarray, lags: np.ndarray, window: int) -> np.ndarray:
    """Normalised cross-correlation of each frame's first ``window`` samples with itself shifted by its lag."""

    result = np.zeros(lags.size)
    columns = np.arange(window)
    for start in range(0, lags.size, 256):  # chunks keep peak memory small for 120 s clips
        chunk = slice(start, start + 256)
        block = frames[:, chunk].T
        head = block[:, :window]
        shifted = np.take_along_axis(block, lags[chunk, None] + columns[None, :], axis=1)
        numerator = np.einsum("ij,ij->i", head, shifted)
        denominator = np.sqrt(np.einsum("ij,ij->i", head, head) * np.einsum("ij,ij->i", shifted, shifted))
        np.divide(numerator, denominator, out=result[chunk], where=denominator > 0)
    return result


def voiced_pitch(y: np.ndarray, sr: int, frame_rms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame YIN f0 (Hz) and a heuristic voicing mask.

    A frame counts as voiced when (1) its normalised cross-correlation at the YIN period is at
    least ``VOICING_NCCF_THRESHOLD``, (2) its RMS is above the silence floor, (3) the f0 estimate
    is not pinned to the search limits and (4) an adjacent frame passes (1)-(3) with a similar f0.
    White noise and silence therefore yield (almost) no voiced frames.
    """

    f0 = librosa.yin(y, fmin=PITCH_FMIN_HZ, fmax=PITCH_FMAX_HZ, sr=sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    frames = librosa.util.frame(np.pad(y, FRAME_LENGTH // 2), frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    count = min(f0.size, frames.shape[1], frame_rms.size)
    f0, frames, frame_rms = f0[:count], frames[:, :count], frame_rms[:count]

    lags = np.rint(sr / f0).astype(int)
    nccf = _nccf_at_lags(frames, lags, FRAME_LENGTH // 2)
    inside_range = (f0 > PITCH_FMIN_HZ * 1.03) & (f0 < PITCH_FMAX_HZ / 1.03)
    periodic = (nccf >= VOICING_NCCF_THRESHOLD) & (frame_rms >= SILENCE_RMS_FLOOR) & inside_range

    similar = np.abs(np.diff(np.log2(f0))) <= VOICING_MAX_OCTAVE_JUMP
    has_neighbour = np.zeros(count, dtype=bool)
    has_neighbour[1:] |= periodic[:-1] & similar
    has_neighbour[:-1] |= periodic[1:] & similar
    return f0, periodic & has_neighbour


class AudioFeatureExtractor:
    """Extract the acoustic features used by the transparent heuristic classifier."""

    @staticmethod
    def features_from_waveform(samples: np.ndarray, sr: int) -> Dict[str, float]:
        """Features of a mono waveform; raises ValueError for short, silent or unvoiced audio.

        Features cover different frame sets (analysis frames are ``FRAME_LENGTH`` samples, hop ``HOP_LENGTH``):
        - ``mean_pitch``/``pitch_std``: voiced frames only.
        - ``rms_energy``, ``energy_std``, spectral centroid, MFCC and zero-crossing statistics: every
          frame of the clip, so pauses and leading/trailing silence lower the means.
        - ``non_silent_fraction_of_clip`` = non-silent frames / all frames; ``voiced_fraction_of_clip``
          = voiced frames / all frames; ``voiced_fraction_of_non_silent`` = voiced frames / non-silent
          frames (the share gated by ``MIN_VOICED_SHARE``; it is NOT the share of the clip that is speech).
        """

        y = np.asarray(samples, dtype=np.float32)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("Audio must be a non-empty mono waveform")
        if not np.all(np.isfinite(y)):
            raise ValueError("Audio contains non-finite samples")
        duration = y.size / sr
        if duration < MIN_DURATION_S:
            raise ValueError(f"Audio clip is too short ({duration:.3f} s); at least {MIN_DURATION_S} s is required")
        y = y - np.float32(np.mean(y))

        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        total_frames = int(rms.size)
        non_silent = int(np.count_nonzero(rms >= SILENCE_RMS_FLOOR))
        if non_silent == 0:
            raise ValueError("Audio is silent or near-silent (every frame is below -60 dBFS); there is no speech to analyse")

        f0, voiced = voiced_pitch(y, sr, rms)
        voiced_count = int(np.count_nonzero(voiced))
        voiced_share = voiced_count / non_silent
        if voiced_count < MIN_VOICED_FRAMES or voiced_share < MIN_VOICED_SHARE:
            raise ValueError(
                f"Audio does not contain enough voiced speech to classify: {voiced_count} voiced frames "
                f"({voiced_share:.0%} of the non-silent audio); at least {MIN_VOICED_FRAMES} frames and "
                f"{MIN_VOICED_SHARE:.0%} are required. Noise-only clips are not classified."
            )

        voiced_f0 = f0[voiced]
        spectral = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        features = {
            "mean_pitch": float(np.mean(voiced_f0)),
            "pitch_std": float(np.std(voiced_f0)),
            "rms_energy": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
            "spectral_centroid_mean": float(np.mean(spectral)),
            "spectral_centroid_std": float(np.std(spectral)),
            "mfcc_mean": float(np.mean(mfccs)),
            "mfcc_std": float(np.std(mfccs)),
            "zero_crossing_rate": float(np.mean(zcr)),
            "non_silent_fraction_of_clip": non_silent / total_frames,
            "voiced_fraction_of_clip": voiced_count / total_frames,
            "voiced_fraction_of_non_silent": float(voiced_share),
            "duration": float(duration),
        }
        if not all(np.isfinite(value) for value in features.values()):
            raise ValueError("Audio feature extraction produced non-finite values")
        return features


def _band_similarity(value: float, low: float, high: float) -> float:
    """1 inside [low, high]; decays linearly to 0 at value 0 (below) and at 2*high (above)."""

    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / low) if low > 0 else 0.0
    return max(0.0, 1.0 - (value - high) / high)


# ---------------------------------------------------------------------------
# Text keywords (heuristic)
# ---------------------------------------------------------------------------
_TEXT_KEYWORDS: Dict[EmotionType, Tuple[str, ...]] = {
    EmotionType.ANGRY: ("angry", "furious", "mad", "damn", "shit", "idiot", "stupid", "ridiculous"),
    EmotionType.FRUSTRATED: (
        "frustrated", "frustrating", "annoyed", "annoying", "upset", "struggling", "useless", "terrible",
        "no grip", "no traction", "no pace",
    ),
    EmotionType.PANICKED: ("panic", "panicking", "emergency", "urgent", "fire", "smoke", "help me", "no brakes", "brakes gone"),
    EmotionType.EXCITED: ("great", "amazing", "fantastic", "brilliant", "awesome", "happy", "yes", "get in there"),
    EmotionType.FOCUSED: ("focus", "focused", "careful", "steady", "concentrate", "copy", "understood"),
    EmotionType.CALM: ("calm", "relaxed", "smooth", "okay", "ok", "fine", "all good", "no worries", "no problem"),
}
_NEGATORS = frozenset(
    "not no never without hardly cannot cant dont doesnt didnt isnt arent wasnt werent wont wouldnt "
    "couldnt shouldnt havent hasnt aint".split()
)
_NEGATION_WINDOW = 3  # tokens before a keyword, within the same clause
# Longest phrases first so "no grip" is consumed before any single word inside it.
_KEYWORD_PATTERNS: List[Tuple[Tuple[str, ...], EmotionType]] = sorted(
    ((tuple(keyword.split()), emotion) for emotion, keywords in _TEXT_KEYWORDS.items() for keyword in keywords),
    key=lambda item: -len(item[0]),
)
_CLAUSE_SPLIT = re.compile(r"[.,;:!?\n]+")
_TOKEN = re.compile(r"[a-z0-9']+")
TEXT_OVERRIDE_MIN_NET_HITS = 2
TEXT_AGREEMENT_BONUS = 0.10
# What each rule of combine_evidence does (the response's evidence_combination_rule).
EVIDENCE_COMBINATION_RULES = {
    "acoustic_only": "The acoustic label is used: there was no transcript, or its keywords were neutral or tied.",
    "text_agrees": (
        f"Transcript keywords agree with the acoustic label; confidence = min({MAX_CONFIDENCE}, max(acoustic, text) "
        f"+ {TEXT_AGREEMENT_BONUS:.2f})."
    ),
    "text_overrides_acoustic": (
        f"Transcript keywords (at least {TEXT_OVERRIDE_MIN_NET_HITS} more hits for one emotion than for any other, "
        "and a higher score) override the acoustic label; confidence = text - acoustic / 2."
    ),
    "acoustic_kept_text_disagrees": (
        "The transcript points to another emotion but too weakly (e.g. a single keyword), so the acoustic label is "
        "kept; confidence = max(0, acoustic - text / 2)."
    ),
}


def _clause_tokens(text: str) -> Iterator[List[str]]:
    normalised = text.lower().replace("’", "'")
    for clause in _CLAUSE_SPLIT.split(normalised):
        tokens = [token.replace("'", "") for token in _TOKEN.findall(clause)]
        tokens = [token for token in tokens if token]
        if tokens:
            yield tokens


def classify_text_emotion(text: str) -> TextEmotionEvidence:
    """Keyword-based transcript emotion (heuristic).

    Words are matched as whole tokens ("made" is not "mad", "eyes" is not "yes"), multi-word
    phrases as token sequences, and a keyword preceded by a negator ("not", "no", "don't", ...)
    within three tokens of the same clause is ignored and reported in ``negated_keywords``.
    A tie between the top two emotions is ambiguous and yields NEUTRAL with confidence 0.
    Confidence = min(0.85, 0.2 + 0.15 * net_hits), rounded to ``SCORE_DECIMALS``.
    """

    hits: Dict[EmotionType, List[str]] = {emotion: [] for emotion in _TEXT_KEYWORDS}
    negated: List[str] = []
    for tokens in _clause_tokens(text or ""):
        used = [False] * len(tokens)
        for pattern, emotion in _KEYWORD_PATTERNS:
            size = len(pattern)
            for start in range(len(tokens) - size + 1):
                if tuple(tokens[start:start + size]) != pattern or any(used[start:start + size]):
                    continue
                used[start:start + size] = [True] * size
                preceding = tokens[max(0, start - _NEGATION_WINDOW):start]
                keyword = " ".join(pattern)
                if pattern[0] not in _NEGATORS and any(token in _NEGATORS for token in preceding):
                    negated.append(keyword)
                else:
                    hits[emotion].append(keyword)

    ranked = sorted(hits.items(), key=lambda item: len(item[1]), reverse=True)
    top_emotion, top_hits = ranked[0]
    net_hits = len(top_hits) - len(ranked[1][1])
    keyword_hits = {emotion.value: words for emotion, words in hits.items() if words}
    if net_hits <= 0:
        return TextEmotionEvidence(EmotionType.NEUTRAL, 0.0, 0, keyword_hits, negated)
    confidence = round(min(0.85, 0.2 + 0.15 * net_hits), SCORE_DECIMALS)
    return TextEmotionEvidence(top_emotion, confidence, net_hits, keyword_hits, negated)


def combine_evidence(
    acoustic_emotion: EmotionType, acoustic_confidence: float, text: Optional[TextEmotionEvidence]
) -> Tuple[EmotionType, float, str]:
    """Combine acoustic and transcript evidence (heuristic). Returns (emotion, confidence, rule).

    The acoustic label is the default; ``EVIDENCE_COMBINATION_RULES`` says what each rule does.
    The confidence is rounded to ``SCORE_DECIMALS``.
    """

    a = float(acoustic_confidence)
    if text is None or text.emotion is EmotionType.NEUTRAL:
        return acoustic_emotion, round(a, SCORE_DECIMALS), "acoustic_only"
    t = float(text.confidence)
    if text.emotion is acoustic_emotion:
        return acoustic_emotion, round(min(MAX_CONFIDENCE, max(a, t) + TEXT_AGREEMENT_BONUS), SCORE_DECIMALS), "text_agrees"
    if text.net_hits >= TEXT_OVERRIDE_MIN_NET_HITS and t > a:
        return text.emotion, round(max(0.0, t - a / 2.0), SCORE_DECIMALS), "text_overrides_acoustic"
    return acoustic_emotion, round(max(0.0, a - t / 2.0), SCORE_DECIMALS), "acoustic_kept_text_disagrees"


# ---------------------------------------------------------------------------
# Acoustic classifier
# ---------------------------------------------------------------------------
class EmotionClassifier:
    """Heuristic acoustic/text classifier. Confidence values are heuristic scores, not probabilities.

    Stateless after construction (the profile table is read-only), so it is safe to share across threads.
    """

    REQUIRED_FEATURES = ("mean_pitch", "pitch_std", "rms_energy", "energy_std")

    def __init__(self):
        self.emotion_thresholds = self._load_emotion_thresholds()

    @staticmethod
    def _load_emotion_thresholds() -> Dict[EmotionType, Dict[str, Tuple[float, float]]]:
        """Hand-set (heuristic, not fitted to data) feature bands per emotion profile."""

        return {
            EmotionType.CALM: {"mean_pitch": (90, 210), "pitch_std": (5, 35), "rms_energy": (0.02, 0.35), "energy_std": (0.0, 0.15)},
            EmotionType.ANGRY: {"mean_pitch": (180, 420), "pitch_std": (30, 100), "rms_energy": (0.25, 1.0), "energy_std": (0.08, 0.50)},
            EmotionType.PANICKED: {"mean_pitch": (250, 550), "pitch_std": (45, 140), "rms_energy": (0.30, 1.0), "energy_std": (0.12, 0.60)},
            EmotionType.FOCUSED: {"mean_pitch": (120, 280), "pitch_std": (10, 50), "rms_energy": (0.08, 0.50), "energy_std": (0.02, 0.20)},
            EmotionType.EXCITED: {"mean_pitch": (180, 400), "pitch_std": (25, 90), "rms_energy": (0.20, 0.85), "energy_std": (0.07, 0.35)},
            EmotionType.FRUSTRATED: {"mean_pitch": (150, 340), "pitch_std": (25, 85), "rms_energy": (0.15, 0.75), "energy_std": (0.08, 0.40)},
        }

    def profile_scores(self, features: Dict[str, float]) -> Dict[EmotionType, float]:
        """Mean band similarity per profile (rounded to ``SCORE_DECIMALS``).

        Out-of-band features (e.g. 0 Hz pitch) score 0, never positive.
        """

        for name in self.REQUIRED_FEATURES:
            value = features.get(name)
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f"Acoustic feature {name!r} is missing, negative or non-finite")
        if not PITCH_FMIN_HZ <= features["mean_pitch"] <= PITCH_FMAX_HZ:
            raise ValueError(
                f"mean_pitch {features['mean_pitch']:.1f} Hz is outside the tracked {PITCH_FMIN_HZ:.0f}-"
                f"{PITCH_FMAX_HZ:.0f} Hz range, so there is no voiced speech to classify"
            )
        if features["rms_energy"] < SILENCE_RMS_FLOOR:
            raise ValueError("rms_energy is below the silence floor; there is no speech to classify")

        scores: Dict[EmotionType, float] = {}
        for emotion, bands in self.emotion_thresholds.items():
            parts = [_band_similarity(float(features[name]), low, high) for name, (low, high) in bands.items()]
            scores[emotion] = round(float(np.mean(parts)), SCORE_DECIMALS)
        return scores

    @staticmethod
    def label_from_scores(scores: Dict[EmotionType, float]) -> Tuple[EmotionType, float]:
        """Best profile and its confidence (``ACOUSTIC_CONFIDENCE_RULE``).

        The confidence is rounded to ``SCORE_DECIMALS`` before the NEUTRAL threshold is applied, so
        the reported value is the one the threshold saw. An exact tie for the best score is ambiguous
        and yields NEUTRAL with confidence 0, as in the transcript classifier.
        """

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_emotion, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        if len(ordered) > 1 and best_score == second_score:
            return EmotionType.NEUTRAL, 0.0
        raw = ACOUSTIC_SIMILARITY_WEIGHT * best_score + ACOUSTIC_LEAD_WEIGHT * max(0.0, best_score - second_score)
        confidence = round(float(np.clip(raw, 0.0, MAX_CONFIDENCE)), SCORE_DECIMALS)
        if confidence < NEUTRAL_CONFIDENCE_THRESHOLD:
            return EmotionType.NEUTRAL, confidence
        return best_emotion, confidence


# ---------------------------------------------------------------------------
# Optional Whisper transcription
# ---------------------------------------------------------------------------
DEFAULT_WHISPER_MODEL = "base"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAX_ECHOED_SETTING_CHARS = 64
_DEFAULT_CACHE_LABEL = "Whisper's default model cache directory ($XDG_CACHE_HOME/whisper or ~/.cache/whisper)"
# whisper.load_model errors that mean the download failed (urllib wraps connection errors in URLError).
_DOWNLOAD_ERRORS = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)
# ... and errors that mean the cache directory or the model file in it could not be created, read or written.
_CACHE_ACCESS_ERRORS = (PermissionError, NotADirectoryError, IsADirectoryError, FileExistsError)


def _whisper_default_cache_dir() -> Path:
    """The directory whisper.load_model uses without ``download_root`` (the same expression as openai-whisper)."""

    default = os.path.join(os.path.expanduser("~"), ".cache")
    return Path(os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")).absolute()


def _checkpoint_file_name(whisper_module: Any, model_name: str) -> str:
    """The file name Whisper stores ``model_name`` under in its cache directory.

    openai-whisper names the file after the model's download URL in its private ``whisper._MODELS``
    table ("turbo" is stored as "large-v3-turbo.pt"); "<name>.pt" is assumed if that table is missing.
    """

    urls = getattr(whisper_module, "_MODELS", None)
    url = urls.get(model_name) if isinstance(urls, Mapping) else None
    return url.rstrip("/").rsplit("/", 1)[-1] if isinstance(url, str) and url.strip("/") else f"{model_name}.pt"


def _check_cache_dir(path: Path, label: str, checkpoint: Optional[str] = None) -> None:
    """Raise ValueError unless Whisper can use the absolute ``path`` as its model cache directory.

    whisper.load_model runs ``os.makedirs(path, exist_ok=True)`` and then reads the model file from
    ``path`` or downloads it there first. Therefore:
    - an existing directory must be searchable; it must also be writable unless it already holds
      the ``checkpoint`` file (a read-only, pre-populated cache works). Without ``checkpoint`` the
      write permission of an existing directory is not checked;
    - a missing directory is created on first use, so its nearest existing ancestor must be a
      writable and searchable directory;
    - a regular file, a broken symbolic link or a path below a regular file never works.
    These are permission checks (os.access), not a guarantee: a full disk or a model file that is
    corrupt or unreadable still makes the load fail, with an explicit TranscriptionError. The
    messages never contain the path, because they reach API clients.
    """

    try:
        info: Optional[os.stat_result] = os.stat(path)
    except FileNotFoundError:
        info = None
    except NotADirectoryError as exc:
        raise ValueError(f"{label} cannot be created: part of its path is a file, not a directory") from exc
    except OSError as exc:  # e.g. PermissionError: a parent directory cannot be searched
        raise ValueError(f"{label} cannot be accessed ({type(exc).__name__})") from exc

    if info is not None:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} exists but is not a directory")
        if not os.access(path, os.X_OK):
            raise ValueError(f"{label} cannot be accessed (the directory has no search permission)")
        if checkpoint is not None and not os.access(path, os.W_OK) and not (path / checkpoint).is_file():
            raise ValueError(
                f"{label} is not writable and does not hold the model file {checkpoint} yet, "
                "so the model cannot be downloaded into it"
            )
        return

    if os.path.islink(path):
        raise ValueError(f"{label} is a broken symbolic link")
    # Every existing component of a missing path is a directory (otherwise stat() raised
    # NotADirectoryError above), so only the permissions of the nearest existing one matter.
    ancestor = path.parent
    while True:
        try:
            os.stat(ancestor)
            break
        except FileNotFoundError:
            if os.path.islink(ancestor):
                raise ValueError(f"{label} cannot be created: part of its path is a broken symbolic link") from None
            ancestor = ancestor.parent  # terminates: the root directory always exists
        except OSError as exc:
            raise ValueError(f"{label} cannot be accessed ({type(exc).__name__})") from exc
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise ValueError(f"{label} cannot be created: its nearest existing parent directory is not writable")


def _unknown_model_reason(value: str, known_models: List[str]) -> str:
    """Reason for a WHISPER_MODEL that ``whisper.available_models()`` does not list."""

    names = ", ".join(known_models)
    if "/" in value or "\\" in value or value.startswith("~") or value.lower().endswith(".pt"):
        # whisper.load_model would also load a local checkpoint file, but that is not supported
        # here, and the value is not echoed because the reason reaches API clients and /health.
        return (
            "WHISPER_MODEL looks like a file path; local checkpoint files are not supported. Use an official "
            f"model name (a pre-downloaded official model is loaded from WHISPER_CACHE_DIR): {names}"
        )
    return f"WHISPER_MODEL {value[:_MAX_ECHOED_SETTING_CHARS]!r} is not a Whisper model; use one of: {names}"


def _load_failure_hint(exc: BaseException) -> str:
    """What a failed whisper.load_model most likely means (empty for errors without a known cause)."""

    if isinstance(exc, _DOWNLOAD_ERRORS):
        return (
            "; downloading the model failed: the first use of a model needs network access unless its "
            "file is already in the model cache directory (WHISPER_CACHE_DIR)"
        )
    if isinstance(exc, _CACHE_ACCESS_ERRORS) or getattr(exc, "errno", None) == errno.EROFS:
        return (
            "; Whisper could not create, read or write the model file in its cache directory "
            "(WHISPER_CACHE_DIR, or ~/.cache/whisper by default): check that directory's permissions"
        )
    if getattr(exc, "errno", None) == errno.ENOSPC:
        return "; there is no space left on the disk that holds the model cache directory"
    return ""


@dataclass(frozen=True)
class WhisperSettings:
    """Whisper configuration from ``WHISPER_MODEL`` and ``WHISPER_CACHE_DIR`` (empty values count as unset).

    - ``model_name``: an official Whisper model name such as "tiny", "base", "small" or "turbo"
      (default "base"). ``WhisperTranscriber.availability`` checks it against
      ``whisper.available_models()``, because only the installed package knows the valid names.
      Paths to local checkpoint files are not supported (see ``_unknown_model_reason``).
    - ``cache_dir``: the directory Whisper downloads model files to (on first use) and loads
      them from. None keeps Whisper's own default ($XDG_CACHE_HOME/whisper, usually
      ~/.cache/whisper). "~" is expanded and a relative path is resolved against the project root.
      It must be creatable or writable, or already hold the model file (see ``_check_cache_dir``).
    """

    model_name: str = DEFAULT_WHISPER_MODEL
    cache_dir: Optional[str] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "WhisperSettings":
        env = os.environ if env is None else env
        model_name = (env.get("WHISPER_MODEL") or "").strip()
        cache_dir = (env.get("WHISPER_CACHE_DIR") or "").strip()
        return cls(model_name=model_name or DEFAULT_WHISPER_MODEL, cache_dir=cache_dir or None)

    def download_root(self, checkpoint: Optional[str] = None) -> Optional[Path]:
        """The resolved WHISPER_CACHE_DIR, or None when it is unset (Whisper's default applies).

        Raises ValueError when Whisper cannot use the directory (see ``_check_cache_dir``);
        ``checkpoint`` is the model's file name, which lets a read-only directory that already
        holds it pass. A directory that does not exist yet is fine if it can be created.
        """

        if self.cache_dir is None:
            return None
        try:
            path = Path(self.cache_dir).expanduser()
        except RuntimeError as exc:  # "~name" for an unknown user
            raise ValueError("WHISPER_CACHE_DIR starts with '~' but that home directory cannot be determined") from exc
        path = path if path.is_absolute() else _PROJECT_ROOT / path
        _check_cache_dir(path, "WHISPER_CACHE_DIR", checkpoint)
        return path

    def model_cache_dir(self, checkpoint: Optional[str] = None) -> Path:
        """The directory the model is loaded from and downloaded to: WHISPER_CACHE_DIR or Whisper's default.

        Raises ValueError when it cannot be used, as ``download_root`` does; for Whisper's default
        directory the message suggests setting WHISPER_CACHE_DIR instead.
        """

        root = self.download_root(checkpoint)
        if root is not None:
            return root
        default = _whisper_default_cache_dir()
        try:
            _check_cache_dir(default, _DEFAULT_CACHE_LABEL, checkpoint)
        except ValueError as exc:
            raise ValueError(f"{exc}; set WHISPER_CACHE_DIR to a usable directory") from exc
        return default


class WhisperTranscriber:
    """Optional local OpenAI Whisper transcription. It never fabricates a fallback transcript.

    Settings come from ``WhisperSettings.from_env()`` when the transcriber is created (the shared
    one in ``get_transcriber`` is created on first use, so restart the process after changing
    them). The loaded model is cached and guarded by a lock: Whisper installs per-call hooks on
    the model, so concurrent transcriptions on one model object are serialised.
    """

    def __init__(self, settings: Optional[WhisperSettings] = None):
        self.settings = settings if settings is not None else WhisperSettings.from_env()
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    def availability(self) -> Tuple[bool, Optional[str]]:
        """(available, reason_if_unavailable); it never downloads or loads the model (see ``load``).

        Requires openai-whisper (with load_model and available_models), a WHISPER_MODEL listed by
        ``whisper.available_models()``, a usable model cache directory (WHISPER_CACHE_DIR or
        Whisper's default, see ``_check_cache_dir``) and ffmpeg.
        """

        try:
            module = importlib.import_module("whisper")
        except ImportError:
            return False, "openai-whisper is not installed (pip install -r requirements-whisper.txt)"
        except Exception as exc:  # a broken torch install can raise OSError/RuntimeError on import
            return False, f"the whisper module failed to import ({type(exc).__name__})"
        if not callable(getattr(module, "load_model", None)) or not callable(getattr(module, "available_models", None)):
            return False, "the installed 'whisper' module is not openai-whisper (it has no load_model/available_models)"
        try:
            known_models = [str(name) for name in module.available_models()]
        except Exception as exc:
            return False, f"whisper.available_models() failed ({type(exc).__name__})"
        if self.settings.model_name not in known_models:
            return False, _unknown_model_reason(self.settings.model_name, known_models)
        try:
            self.settings.model_cache_dir(_checkpoint_file_name(module, self.settings.model_name))
        except ValueError as exc:
            return False, str(exc)
        if shutil.which("ffmpeg") is None:
            return False, "ffmpeg was not found on PATH (Whisper needs it to read audio)"
        return True, None

    def _require_available(self) -> None:
        available, reason = self.availability()
        if not available:
            raise TranscriptionError(f"Whisper transcription is unavailable: {reason}")

    def _load_locked(self) -> Any:
        """Load and cache the model; the caller holds ``self._lock``. A failed load is retried next time."""

        if self._model is None:
            whisper = importlib.import_module("whisper")
            try:
                root = self.settings.download_root()
            except ValueError as exc:  # the directory changed after availability() checked it
                raise TranscriptionError(f"Whisper transcription is unavailable: {exc}") from exc
            try:
                self._model = whisper.load_model(self.model_name, download_root=None if root is None else str(root))
            except Exception as exc:
                logger.warning("Loading Whisper model %r failed", self.model_name, exc_info=True)
                raise TranscriptionError(
                    f"Loading the Whisper model {self.model_name!r} failed ({type(exc).__name__}){_load_failure_hint(exc)}"
                ) from exc
        return self._model

    def load(self) -> Any:
        """Load the configured model now (downloading it on first use) instead of on the first transcription.

        Raises TranscriptionError when Whisper is unavailable or the model cannot be loaded.
        """

        self._require_available()
        with self._lock:
            return self._load_locked()

    def transcribe(self, audio_file: str) -> Optional[str]:
        """Transcript text, or None if Whisper produced no text. Raises TranscriptionError on any failure."""

        self._require_available()
        with self._lock:
            model = self._load_locked()
            # Whisper defaults to FP16, and on CPU it warns and falls back to FP32 on every call.
            options = {"fp16": False} if getattr(getattr(model, "device", None), "type", None) == "cpu" else {}
            try:
                result = model.transcribe(audio_file, **options)
            except Exception as exc:
                logger.warning("Whisper transcription failed", exc_info=True)
                raise TranscriptionError(f"Whisper transcription failed ({type(exc).__name__})") from exc
        text = result.get("text", "") if isinstance(result, dict) else ""
        text = str(text or "").strip()
        return text or None


_emotion_classifier = EmotionClassifier()
_transcriber: Optional[WhisperTranscriber] = None
_transcriber_lock = threading.Lock()


def get_emotion_classifier() -> EmotionClassifier:
    return _emotion_classifier


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    with _transcriber_lock:
        if _transcriber is None:
            _transcriber = WhisperTranscriber()
        return _transcriber


def classify_emotion_detailed(audio_file: str, transcribe: bool = False, allow_local_paths: bool = False) -> Dict[str, Any]:
    """Classify driver-radio emotion (heuristic) and return a JSON-serialisable result.

    ``audio_file`` is base64 audio or a ``data:audio/*;base64,`` URI; existing file paths are
    accepted only with ``allow_local_paths=True`` (never enable this for HTTP input).

    ``transcription_status`` is "not_requested", "unavailable" (with
    ``transcription_unavailable_reason``: openai-whisper or ffmpeg missing, an invalid
    WHISPER_MODEL, or a model cache directory that cannot be used), "completed" or "empty"
    (Whisper returned no text).

    Raises ValueError for invalid, undecodable, oversized, too long/short, silent or unvoiced
    audio, and TranscriptionError (a RuntimeError) if Whisper is available but loading the model
    or transcribing fails.
    """

    if not isinstance(transcribe, bool):
        raise ValueError("transcribe must be a boolean")
    classifier = get_emotion_classifier()
    transcription: Optional[str] = None
    status, unavailable_reason = "not_requested", None
    with audio_input_file(audio_file, allow_local_paths) as path:
        samples, source_sr = load_waveform(path)
        features = AudioFeatureExtractor.features_from_waveform(samples, ANALYSIS_SAMPLE_RATE)
        profile_scores = classifier.profile_scores(features)
        acoustic_emotion, acoustic_confidence = classifier.label_from_scores(profile_scores)
        if transcribe:
            transcriber = get_transcriber()
            available, unavailable_reason = transcriber.availability()
            if not available:
                status = "unavailable"
            else:
                transcription = transcriber.transcribe(path)
                status = "completed" if transcription else "empty"

    text = classify_text_emotion(transcription) if transcription else None
    emotion, confidence, rule = combine_evidence(acoustic_emotion, acoustic_confidence, text)
    return {
        "emotion": emotion.value,
        "confidence": float(confidence),
        "acoustic_emotion": acoustic_emotion.value,
        "acoustic_confidence": float(acoustic_confidence),
        "acoustic_confidence_rule": ACOUSTIC_CONFIDENCE_RULE,
        "acoustic_profile_scores": {key.value: value for key, value in profile_scores.items()},
        "audio_features": features,
        "duration": features["duration"],
        "source_sample_rate": source_sr,
        "transcription": transcription,
        "transcription_status": status,
        "transcription_unavailable_reason": unavailable_reason,
        "text_emotion": text.emotion.value if text else None,
        "text_confidence": text.confidence if text else None,
        "text_keyword_hits": text.keyword_hits if text else None,
        "negated_keywords": text.negated_keywords if text else None,
        "evidence_combination": rule,
        "evidence_combination_rule": EVIDENCE_COMBINATION_RULES[rule],
        "classifier": "acoustic heuristic" + (" + Whisper transcript keyword heuristic" if text else ""),
        "disclaimer": DISCLAIMER,
    }
