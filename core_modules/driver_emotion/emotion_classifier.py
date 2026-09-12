#!/usr/bin/env python3
"""Driver radio emotion analysis from real audio features with optional Whisper transcription."""

import base64
import binascii
import logging
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np


class EmotionType(Enum):
    CALM = "calm"
    ANGRY = "angry"
    PANICKED = "panicked"
    FOCUSED = "focused"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    emotion: EmotionType
    confidence: float
    timestamp: Optional[str] = None
    duration: Optional[float] = None
    audio_features: Optional[Dict[str, float]] = None


def _decode_audio_input(audio_input: str) -> Tuple[str, Optional[str]]:
    """Return an audio path plus an optional temporary path to remove later."""

    value = audio_input.strip()
    if not value:
        raise ValueError("audio_file cannot be empty")

    # Raw base64 can be thousands of characters long. Do not pass such strings
    # to pathlib.stat(), which can raise ENAMETOOLONG before we get a chance to
    # decode them. Plausible filesystem paths are checked first.
    if not value.startswith("data:") and len(value) <= 1024:
        try:
            path = Path(value)
            if path.exists() and path.is_file():
                return str(path), None
        except OSError:
            pass

    payload = value
    suffix = ".wav"
    if payload.startswith("data:"):
        header, sep, payload = payload.partition(",")
        if not sep or ";base64" not in header:
            raise ValueError("Audio data URI must be base64 encoded")
        if "audio/mpeg" in header:
            suffix = ".mp3"
        elif "audio/ogg" in header:
            suffix = ".ogg"
        elif "audio/flac" in header:
            suffix = ".flac"

    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_file must be an existing path or valid base64 audio") from exc
    if not decoded:
        raise ValueError("Decoded audio is empty")

    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(decoded)
        handle.flush()
    finally:
        handle.close()
    return handle.name, handle.name


class AudioFeatureExtractor:
    """Extract acoustic features used by the transparent heuristic classifier."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract_features(self, audio_file: str) -> Dict[str, float]:
        y, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        if y.size == 0:
            raise ValueError("Audio file contains no samples")
        duration = float(librosa.get_duration(y=y, sr=sr))
        if duration <= 0:
            raise ValueError("Audio duration must be positive")

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        voiced = pitches[magnitudes > max(0.1, float(np.percentile(magnitudes, 75)))]
        mean_pitch = float(np.mean(voiced)) if voiced.size else 0.0
        pitch_std = float(np.std(voiced)) if voiced.size else 0.0
        rms = librosa.feature.rms(y=y)
        spectral = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
        zcr = librosa.feature.zero_crossing_rate(y)

        features = {
            "mean_pitch": mean_pitch,
            "pitch_std": pitch_std,
            "rms_energy": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
            "spectral_centroid_mean": float(np.mean(spectral)),
            "spectral_centroid_std": float(np.std(spectral)),
            "mfcc_mean": float(np.mean(mfccs)),
            "mfcc_std": float(np.std(mfccs)),
            "tempo": tempo_value,
            "zero_crossing_rate": float(np.mean(zcr)),
            "duration": duration,
        }
        if not all(np.isfinite(value) for value in features.values()):
            raise ValueError("Audio feature extraction produced non-finite values")
        return features


class EmotionClassifier:
    """Heuristic acoustic/text classifier. Confidence values are similarity scores, not probabilities."""

    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
        self.emotion_thresholds = self._load_emotion_thresholds()

    @staticmethod
    def _load_emotion_thresholds() -> Dict[EmotionType, Dict[str, Tuple[float, float]]]:
        return {
            EmotionType.CALM: {"mean_pitch": (90, 210), "pitch_std": (5, 35), "rms_energy": (0.02, 0.35), "energy_std": (0.0, 0.15)},
            EmotionType.ANGRY: {"mean_pitch": (180, 420), "pitch_std": (30, 100), "rms_energy": (0.25, 1.0), "energy_std": (0.08, 0.50)},
            EmotionType.PANICKED: {"mean_pitch": (250, 550), "pitch_std": (45, 140), "rms_energy": (0.30, 1.0), "energy_std": (0.12, 0.60)},
            EmotionType.FOCUSED: {"mean_pitch": (120, 280), "pitch_std": (10, 50), "rms_energy": (0.08, 0.50), "energy_std": (0.02, 0.20)},
            EmotionType.EXCITED: {"mean_pitch": (180, 400), "pitch_std": (25, 90), "rms_energy": (0.20, 0.85), "energy_std": (0.07, 0.35)},
            EmotionType.FRUSTRATED: {"mean_pitch": (150, 340), "pitch_std": (25, 85), "rms_energy": (0.15, 0.75), "energy_std": (0.08, 0.40)},
        }

    def classify_emotion(self, audio_file: str) -> EmotionResult:
        path, temporary = _decode_audio_input(audio_file)
        try:
            features = self.feature_extractor.extract_features(path)
            emotion, confidence = self._classify_from_features(features)
            return EmotionResult(emotion=emotion, confidence=confidence, duration=features.get("duration"), audio_features=features)
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _classify_from_features(self, features: Dict[str, float]) -> Tuple[EmotionType, float]:
        scores: Dict[EmotionType, float] = {}
        for emotion, thresholds in self.emotion_thresholds.items():
            parts = []
            for name, (minimum, maximum) in thresholds.items():
                if name not in features:
                    continue
                value = float(features[name])
                if minimum <= value <= maximum:
                    parts.append(1.0)
                elif value < minimum:
                    parts.append(max(0.0, 1.0 - (minimum - value) / max(abs(minimum), 1e-6)))
                else:
                    parts.append(max(0.0, 1.0 - (value - maximum) / max(abs(maximum), 1e-6)))
            if parts:
                scores[emotion] = float(np.mean(parts))

        if not scores:
            return EmotionType.NEUTRAL, 0.0
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_emotion, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        confidence = float(np.clip(0.45 * best_score + 0.55 * max(0.0, best_score - second_score), 0.0, 0.95))
        return (best_emotion, confidence) if confidence >= 0.20 else (EmotionType.NEUTRAL, confidence)

    @staticmethod
    def classify_emotion_from_text(text: str) -> EmotionResult:
        text_lower = text.lower()
        keywords = {
            EmotionType.ANGRY: ["angry", "furious", "mad", "damn", "shit"],
            EmotionType.FRUSTRATED: ["frustrated", "annoyed", "upset", "struggling", "no grip"],
            EmotionType.PANICKED: ["panic", "emergency", "help", "urgent", "quick"],
            EmotionType.EXCITED: ["great", "amazing", "fantastic", "brilliant", "yes"],
            EmotionType.FOCUSED: ["focus", "careful", "steady", "copy", "understood"],
            EmotionType.CALM: ["calm", "relaxed", "smooth", "okay", "ok"],
        }
        scores = {emotion: sum(1 for word in words if word in text_lower) for emotion, words in keywords.items()}
        best_emotion, hits = max(scores.items(), key=lambda item: item[1])
        if hits == 0:
            return EmotionResult(EmotionType.NEUTRAL, 0.0)
        return EmotionResult(best_emotion, float(min(0.85, 0.35 + 0.15 * hits)))


class WhisperTranscriber:
    """Optional OpenAI Whisper transcription. It never fabricates a fallback transcript."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        try:
            import whisper  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def transcribe(self, audio_file: str) -> Optional[str]:
        if not self.available:
            return None
        import whisper
        if self._model is None:
            self._model = whisper.load_model(self.model_name)
        result = self._model.transcribe(audio_file)
        text = str(result.get("text", "")).strip()
        return text or None


_emotion_classifier: Optional[EmotionClassifier] = None
_transcriber: Optional[WhisperTranscriber] = None


def get_emotion_classifier() -> EmotionClassifier:
    global _emotion_classifier
    if _emotion_classifier is None:
        _emotion_classifier = EmotionClassifier()
    return _emotion_classifier


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber


def classify_emotion(audio_file: str) -> str:
    return get_emotion_classifier().classify_emotion(audio_file).emotion.value


def classify_emotion_detailed(audio_file: str, transcribe: bool = False) -> Dict[str, Any]:
    path, temporary = _decode_audio_input(audio_file)
    try:
        classifier = get_emotion_classifier()
        features = classifier.feature_extractor.extract_features(path)
        acoustic_emotion, acoustic_confidence = classifier._classify_from_features(features)
        transcription = get_transcriber().transcribe(path) if transcribe else None
        final_emotion = acoustic_emotion
        final_confidence = acoustic_confidence
        text_result = EmotionResult(EmotionType.NEUTRAL, 0.0)
        if transcription:
            text_result = classifier.classify_emotion_from_text(transcription)
            if text_result.confidence > acoustic_confidence:
                final_emotion = text_result.emotion
            final_confidence = float(np.clip((acoustic_confidence + text_result.confidence) / 2.0, 0.0, 0.95))
        return {
            "emotion": final_emotion.value,
            "confidence": final_confidence,
            "transcription": transcription,
            "transcription_available": get_transcriber().available,
            "audio_features": features,
            "duration": features.get("duration"),
            "text_emotion": text_result.emotion.value if transcription else None,
            "text_confidence": text_result.confidence if transcription else None,
            "classifier": "acoustic heuristic" + (" + Whisper text" if transcription else ""),
        }
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                logging.warning("Could not remove temporary audio file %s", temporary)
