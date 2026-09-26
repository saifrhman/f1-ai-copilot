"""Pydantic v2 request/response models for the driver-radio emotion API."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from core_modules.driver_emotion.emotion_classifier import (
    ANALYSIS_SAMPLE_RATE,
    FRAME_LENGTH,
    HOP_LENGTH,
    MAX_AUDIO_BYTES,
    MAX_AUDIO_INPUT_CHARS,
    MAX_CHANNELS,
    MAX_CONFIDENCE,
    MAX_DURATION_S,
    MAX_SOURCE_SAMPLE_RATE,
    MIN_DURATION_S,
    MIN_SOURCE_SAMPLE_RATE,
    MIN_VOICED_SHARE,
    PITCH_FMAX_HZ,
    PITCH_FMIN_HZ,
    EmotionType,
)

TranscriptionStatus = Literal["not_requested", "unavailable", "completed", "empty"]
EvidenceCombination = Literal["acoustic_only", "text_agrees", "text_overrides_acoustic", "acoustic_kept_text_disagrees"]


class EmotionRequest(BaseModel):
    """Body of POST /api/emotion/classify."""

    model_config = ConfigDict(extra="forbid")

    audio_file: str = Field(
        ...,
        min_length=1,
        max_length=MAX_AUDIO_INPUT_CHARS,
        description=(
            "Base64-encoded audio or a data:audio/<type>;base64,<data> URI (line breaks allowed). "
            f"At most {MAX_AUDIO_BYTES // (1024 * 1024)} MiB decoded, {MIN_DURATION_S}-{MAX_DURATION_S:.0f} s long, "
            f"mono or stereo (at most {MAX_CHANNELS} channels), {MIN_SOURCE_SAMPLE_RATE}-{MAX_SOURCE_SAMPLE_RATE} Hz. "
            "WAV, FLAC, OGG and MP3 are decoded natively; M4A/AAC need ffmpeg on the server. "
            "Server file paths are not accepted."
        ),
    )
    transcribe: StrictBool = Field(
        False,
        description=(
            "JSON true/false only (strings such as 'yes' and numbers are rejected). "
            "Also transcribe with local OpenAI Whisper when it and ffmpeg are installed. "
            "The response's transcription_status says whether this happened."
        ),
    )


class AudioFeatures(BaseModel):
    """Acoustic inputs of the heuristic classifier. Each field states which analysis frames it covers.

    A frame is "silent" below -60 dBFS RMS and "voiced" when the heuristic pitch tracker accepts it
    (see ``non_silent_fraction_of_clip`` for the frame size).
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mean_pitch: float = Field(
        ge=PITCH_FMIN_HZ, le=PITCH_FMAX_HZ, description="Mean f0 (Hz) over voiced frames only"
    )
    pitch_std: float = Field(ge=0.0, description="Standard deviation of f0 (Hz) over voiced frames only")
    rms_energy: float = Field(
        ge=0.0,
        description="Mean frame RMS (full scale = 1) over ALL frames of the clip; pauses and silence lower it",
    )
    energy_std: float = Field(ge=0.0, description="Standard deviation of frame RMS over all frames of the clip")
    spectral_centroid_mean: float = Field(ge=0.0, description="Mean spectral centroid (Hz) over all frames of the clip")
    spectral_centroid_std: float = Field(ge=0.0, description="Standard deviation of the spectral centroid (Hz), all frames")
    mfcc_mean: float = Field(description="Mean of 13 MFCCs over all frames of the clip")
    mfcc_std: float = Field(ge=0.0, description="Standard deviation of the 13 MFCCs over all frames of the clip")
    zero_crossing_rate: float = Field(ge=0.0, le=1.0, description="Mean zero-crossing rate per sample, all frames")
    non_silent_fraction_of_clip: float = Field(
        gt=0.0,
        le=1.0,
        description=(
            "Non-silent frames / all frames of the clip (frames are "
            f"{FRAME_LENGTH}-sample windows every {HOP_LENGTH} samples at {ANALYSIS_SAMPLE_RATE} Hz)"
        ),
    )
    voiced_fraction_of_clip: float = Field(
        gt=0.0, le=1.0, description="Voiced frames / all frames of the clip (share of the whole clip that is voiced)"
    )
    voiced_fraction_of_non_silent: float = Field(
        gt=0.0,
        le=1.0,
        description=(
            "Voiced frames / NON-SILENT frames. Clips below "
            f"{MIN_VOICED_SHARE:.0%} are rejected as noise. Silence does not lower this value, so it is "
            "not the share of the clip that is speech (see voiced_fraction_of_clip)."
        ),
    )
    duration: float = Field(gt=0.0, description="Clip duration in seconds")


class EmotionResponse(BaseModel):
    """Result of classify_emotion_detailed (heuristic analysis; see ``disclaimer``)."""

    model_config = ConfigDict(extra="forbid")

    emotion: EmotionType = Field(description="Final label after combining acoustic and transcript evidence")
    confidence: float = Field(
        ge=0.0, le=MAX_CONFIDENCE, allow_inf_nan=False,
        description="Heuristic score (similarities and margins), not a probability (3 decimals, like every score in this response)",
    )
    acoustic_emotion: EmotionType
    acoustic_confidence: float = Field(ge=0.0, le=MAX_CONFIDENCE, allow_inf_nan=False)
    acoustic_confidence_rule: str = Field(description="How acoustic_confidence is computed from the profile scores")
    acoustic_profile_scores: Dict[str, float] = Field(description="Band-similarity score (0-1) of each heuristic emotion profile")
    audio_features: AudioFeatures = Field(description="Acoustic features (heuristic inputs); each field states its frame set")
    duration: float = Field(gt=0.0, allow_inf_nan=False, description="Clip duration in seconds")
    source_sample_rate: int = Field(
        ge=MIN_SOURCE_SAMPLE_RATE, le=MAX_SOURCE_SAMPLE_RATE, description="Sample rate of the submitted audio (Hz)"
    )
    transcription: Optional[str] = Field(description="Whisper transcript; null unless transcription_status is 'completed'")
    transcription_status: TranscriptionStatus
    transcription_unavailable_reason: Optional[str] = Field(description="Why Whisper could not run (status 'unavailable')")
    text_emotion: Optional[EmotionType] = None
    text_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    text_keyword_hits: Optional[Dict[str, List[str]]] = None
    negated_keywords: Optional[List[str]] = None
    evidence_combination: EvidenceCombination
    evidence_combination_rule: str = Field(
        description="What the evidence_combination rule does and how it computes confidence from the acoustic and text scores"
    )
    classifier: str
    disclaimer: str
