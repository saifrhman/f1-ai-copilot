"""Driver radio: coarse emotion label for a team-radio clip (POST /api/emotion/classify, heuristic)."""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from ui.api_client import ApiError, ApiUnavailable, get_client
from ui.components import (
    CLIP_SOURCES,
    MAX_AUDIO_TEXT,
    UPLOAD,
    clip_input,
    heuristic_badge,
    humanise,
    json_expander,
    md_text,
    oversize_problem,
    profile_tie_note,
    show_api_error,
    transcription_state,
)

RESULT_KEY = "radio_result"
# (label, unit, frames covered) per API feature; only the first four feed the emotion profiles.
FEATURES = {
    "mean_pitch": ("Mean pitch", "Hz", "voiced frames"),
    "pitch_std": ("Pitch variation (std)", "Hz", "voiced frames"),
    "rms_energy": ("Mean energy (RMS)", "full scale = 1", "all frames"),
    "energy_std": ("Energy variation (std)", "full scale = 1", "all frames"),
    "spectral_centroid_mean": ("Spectral centroid", "Hz", "all frames"),
    "spectral_centroid_std": ("Spectral centroid variation (std)", "Hz", "all frames"),
    "mfcc_mean": ("MFCC mean", "", "all frames"),
    "mfcc_std": ("MFCC variation (std)", "", "all frames"),
    "zero_crossing_rate": ("Zero-crossing rate", "per sample", "all frames"),
    "non_silent_fraction_of_clip": ("Non-silent share of the clip", "fraction", "all frames"),
    "voiced_fraction_of_clip": ("Voiced share of the clip", "fraction", "all frames"),
    "voiced_fraction_of_non_silent": ("Voiced share of the non-silent audio", "fraction", "non-silent frames"),
    "duration": ("Duration", "s", "whole clip"),
}
PROFILE_FEATURES = ("mean_pitch", "pitch_std", "rms_energy", "energy_std")


def render_transcription_notice(state: Dict[str, Any]) -> None:
    if state["available"] is False:
        st.info(
            f"**Transcription is unavailable on this API:** {md_text(state['reason'])}. The acoustic analysis works "
            "without it. Whisper is optional: install it (and ffmpeg) where the API runs, then restart the API; the "
            "check runs once per API process.",
            icon=":material/mic_off:",
        )
    elif state["available"] is None:
        st.caption(f"Transcription availability unknown: {md_text(state['reason'])}.")


def render_transcript(body: Dict[str, Any]) -> None:
    status = body.get("transcription_status")
    if status == "not_requested":
        st.caption("Transcription was not requested: the label uses the acoustic features only.")
    elif status == "unavailable":
        st.info(
            f"Transcription was requested but Whisper is unavailable on the API: "
            f"{md_text(body.get('transcription_unavailable_reason') or 'no reason given')}. The label uses the "
            "acoustic features only.",
            icon=":material/mic_off:",
        )
    elif status == "empty":
        st.caption("Whisper returned no text for this clip, so the label uses the acoustic features only.")
    elif status == "completed":
        st.markdown(f"> {md_text(body.get('transcription') or '')}")
        hits = body.get("text_keyword_hits") or {}
        text_emotion = body.get("text_emotion")
        text_confidence = body.get("text_confidence")
        score = "" if text_confidence is None else f" (keyword score {text_confidence:.3f})"
        st.markdown(f"Transcript keyword label: **{md_text(text_emotion or 'none')}**{score}")
        if hits:
            st.dataframe(
                [{"Emotion": emotion, "Keywords found": ", ".join(words)} for emotion, words in hits.items()], hide_index=True
            )
        if body.get("negated_keywords"):
            st.caption(f"Ignored as negated: {md_text(', '.join(body['negated_keywords']))}")


def acoustic_breakdown(body: Dict[str, Any], scores: List[Tuple[str, float]]) -> str:
    """How the API scored the acoustic label (its ``acoustic_confidence_rule``), with the similarities it used."""

    text = f"Acoustic confidence {body['acoustic_confidence']:.3f} = {md_text(body.get('acoustic_confidence_rule'))}."
    if len(scores) > 1:
        (best, best_score), (runner_up, runner_up_score) = scores[0], scores[1]
        text += (
            f" Here: best {md_text(best)} {best_score:.3f}, runner-up {md_text(runner_up)} {runner_up_score:.3f}, "
            f"lead {best_score - runner_up_score:.3f}."
        )
    return text


def render_result(result: Dict[str, Any]) -> None:
    body = result["response"]
    # Best first; the API's own order among equal scores (an exact tie for the best score gives neutral).
    scores = sorted((body.get("acoustic_profile_scores") or {}).items(), key=lambda item: -item[1])
    st.subheader("Result")
    st.caption(f"Clip: {md_text(result['label'])}")
    columns = st.columns(4)
    columns[0].metric("Emotion", str(body["emotion"]).capitalize())
    columns[1].metric(
        "Confidence",
        f"{body['confidence']:.3f}",
        help="Heuristic score, not a probability; the combination rule below says how it was computed",
    )
    columns[2].metric(  # the confidence as the delta line: label and number in one value are cut off
        "Acoustic label",
        str(body["acoustic_emotion"]).capitalize(),
        delta=f"{body['acoustic_confidence']:.3f}",
        delta_color="off",
        delta_arrow="off",
        delta_description="acoustic confidence",
        help="Label and acoustic confidence from the pitch and energy profiles alone: "
        + md_text(body.get("acoustic_confidence_rule")),
    )
    columns[3].metric("Duration", f"{body['duration']:.1f} s", help=f"Submitted at {body['source_sample_rate']} Hz")
    tie = profile_tie_note(body.get("acoustic_profile_scores") or {}, body.get("acoustic_emotion"))
    if tie:
        st.warning(tie, icon=":material/balance:")
    combination = body.get("evidence_combination")
    basis = (
        "the acoustic confidence"
        if combination == "acoustic_only"
        else "computed from the acoustic confidence and the transcript keyword score by the rule below"
    )
    st.caption(
        f"Confidence is {basis}: a heuristic score, not the probability that the driver feels this way. "
        f"**Combination rule:** {md_text(body.get('evidence_combination_rule') or combination)}"
    )
    st.caption(acoustic_breakdown(body, scores))

    st.markdown("**Transcript**")
    render_transcript(body)

    # One table under the other: side by side, the feature table's columns are cut off even on a desktop.
    st.markdown("**Emotion profile similarity**")
    st.dataframe(
        [{"Profile": name, "Band similarity": score} for name, score in scores],
        hide_index=True,
        column_config={"Band similarity": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")},
    )
    st.caption("Mean band similarity of the pitch and energy features to each hand-set profile (0-1).")
    st.markdown("**Acoustic features**")
    rows = []
    for key, value in (body.get("audio_features") or {}).items():
        name, unit, frames = FEATURES.get(key, (humanise(key), "", ""))
        rows.append(
            {
                "Feature": name,
                "Used for the label": key in PROFILE_FEATURES,
                "Value": round(value, 4),
                "Unit": unit,
                "Frames": frames,
            }
        )
    st.dataframe(  # all 13 features, without a scrollbar
        rows, hide_index=True, height="content", column_config={"Value": st.column_config.NumberColumn(format="%.4f")}
    )
    st.caption("Energy is measured on the recording as it is, so a louder or quieter recording can change the label.")
    st.caption(md_text(body.get("disclaimer", "")))
    json_expander(body)


st.title("Driver radio")
heuristic_badge("Heuristic · not a validated emotion model")
st.markdown(
    "Upload or record a team-radio clip. The API measures pitch and energy and compares them with hand-set "
    "emotion profiles; optionally, a local Whisper transcript is scanned for emotion keywords. The result is a "
    "coarse label for exploring radio clips, **not a validated emotion model**."
)
st.caption(
    f"Clips: 0.5-120 s of speech, mono or stereo, 8-96 kHz, at most {MAX_AUDIO_TEXT}. WAV, FLAC, OGG and MP3 are "
    "decoded natively; M4A and WebM need ffmpeg where the API runs. Silent or noise-only clips are rejected, not labelled."
)
st.caption(
    "**Labels depend on recording level:** the energy bands are absolute, so the same speech recorded louder or "
    "quieter (another microphone, gain setting or radio chain) can get a different label."
)
transcription = transcription_state()
render_transcription_notice(transcription)

source = st.segmented_control("Clip", CLIP_SOURCES, default=UPLOAD, required=True, key="radio_source")
with st.form("radio_form"):
    clip = clip_input(source, "radio")
    transcribe = st.toggle(
        "Transcribe with Whisper",
        disabled=transcription["available"] is False,
        key="radio_transcribe",
        help="Slower. Adds a keyword heuristic on the transcript; the combination rule is shown with the result.",
    )
    submitted = st.form_submit_button("Analyse clip", type="primary", icon=":material/graphic_eq:", key="radio_submit")

if submitted:
    st.session_state.pop(RESULT_KEY, None)
    audio: Optional[bytes] = clip.getvalue() if clip is not None else None
    if not audio:
        st.warning("Upload or record a clip first.", icon=":material/mic:")
    elif too_large := oversize_problem(audio):
        st.error(f"Nothing was sent to the API. {too_large}", icon=":material/data_alert:")
    else:
        try:
            with st.spinner("Analysing the clip..."):
                wanted = bool(transcribe) and transcription["available"] is not False
                body = get_client().classify_emotion(base64.b64encode(audio).decode("ascii"), transcribe=wanted)
        except (ApiUnavailable, ApiError) as exc:
            show_api_error(exc, "The analysis")
        else:
            label = clip.name if source == UPLOAD else "recording"
            st.session_state[RESULT_KEY] = {"label": label, "response": body}

if RESULT_KEY in st.session_state:
    render_result(st.session_state[RESULT_KEY])
