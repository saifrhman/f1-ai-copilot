"""Web UI pages for laps, radio clips and questions: Ghost car, Driver radio and Ask the copilot.

Every page runs in AppTest against the real FastAPI app (``ui_api``); the regulatory questions use
the real RAG pipeline over generated PDFs with an in-memory index (``installed_rag``), where only the
embedding and chat services are deterministic stand-ins.
"""

from __future__ import annotations

import base64
import io
import json
import math
from typing import Callable, List

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import streamlit.testing.v1.app_test as app_test_module
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

import app.main
import core_modules.rule_checker.fia_rag.pipeline as rag_pipeline
from core_modules.driver_emotion.emotion_classifier import get_transcriber
from tests.helpers import (
    DEFINITIONS_QUESTION,
    ScriptedChatModel,
    build_test_rag,
    make_definitions_rag,
    write_definitions_corpus,
)
from tests.ui_support import (  # noqa: F401 (pytest fixtures)
    assert_no_exception,
    column_formats,
    expander_labels,
    frame,
    markdown_text,
    metrics,
    page_text,
    record_calls,
    run_page,
    ui_api,
    ui_api_down,
)
from ui.components import documented_example, md_text, not_modelled_text

WHISPER_MISSING = "openai-whisper is not installed (pip install -r requirements-whisper.txt)"


@pytest.fixture
def served_file(monkeypatch) -> Callable[[str], bytes]:
    """Bytes behind an image or download URL of the last AppTest run (Streamlit's in-memory media store)."""

    stores: List[MemoryMediaFileStorage] = []

    class RecordingStorage(MemoryMediaFileStorage):
        def __init__(self, media_endpoint: str) -> None:
            super().__init__(media_endpoint)
            stores.append(self)

    monkeypatch.setattr(app_test_module, "MemoryMediaFileStorage", RecordingStorage)  # AppTest makes one per run
    return lambda url: stores[-1].get_file(url.rsplit("/", 1)[-1]).content


def _speech_wav(seconds: float = 1.5, rate: int = 22050, f0: float = 160.0, vibrato: float = 0.0, gain: float = 0.2) -> bytes:
    """A voice-like clip: a harmonic tone (pitch swinging by ``vibrato`` Hz) with a syllable-rate amplitude envelope."""

    t = np.arange(int(rate * seconds)) / rate
    phase = 2 * math.pi * np.cumsum(f0 + vibrato * np.sin(2 * math.pi * 2 * t)) / rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 6))
    envelope = 0.6 + 0.4 * np.sin(2 * math.pi * 3 * t)
    buffer = io.BytesIO()
    sf.write(buffer, gain * voice * envelope / 2.3, rate, format="WAV")
    return buffer.getvalue()


def _acoustic_confidence(at: AppTest) -> tuple:
    """The acoustic label metric's second line: (value, description)."""

    metric = next(metric for metric in at.metric if metric.label == "Acoustic label")
    return metric.proto.delta, metric.proto.delta_description


def _captions(at: AppTest) -> str:
    return "\n".join(str(element.value) for element in at.caption)


def _input_problems(at: AppTest) -> str:
    """The problems a page listed instead of sending its request (Markdown, as ``md_text`` escapes them)."""

    assert at.error[0].value == "Nothing was sent to the API. Please fix:"
    nodes = list(at.main)
    heading = next(index for index, node in enumerate(nodes) if node.type == "error")
    return str(nodes[heading + 1].value)  # the list follows its heading


# ------------------------------------------------------------------ ghost car


def test_ghost_example_laps_show_the_api_png_and_delta(ui_api, served_file, monkeypatch):
    compared = record_calls(monkeypatch, ui_api, "generate_ghost")
    fetched = record_calls(monkeypatch, ui_api, "fetch_ghost_image")
    at = run_page("ghost")
    assert_no_exception(at)
    assert "Synthetic example data, not recorded telemetry" in at.info[0].value
    assert at.title[0].value == "Ghost car" and not at.image
    assert ":orange-badge[:material/science: Heuristic zones · not validated]" in markdown_text(at)
    assert "Braking-zone matching and the loss/gain locations are heuristic." in markdown_text(at)

    at.button(key="ghost_submit").click().run()
    assert_no_exception(at)
    payload, body = compared[0]["args"][0], compared[0]["result"]
    assert payload["lap1_telemetry"]["lap_number"] == 14 and payload["track_section"] == "Example circuit"
    assert payload["alignment"] == "auto" and "distance" not in payload["lap1_telemetry"]
    # The example laps run from line to line, so auto alignment uses lap fractions without caveats.
    assert body["alignment"]["method"] == "lap_fraction" and body["warnings"] == []
    png = fetched[0]["result"]
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(at.image) == 1
    assert served_file(at.image[0].value[0]) == png  # the page shows the API's image unchanged
    summary = body["summary"]
    assert summary["final_delta_s"] > 0  # lap 15 is slower by construction
    assert metrics(at)["Gap at the end"] == f"{summary['final_delta_s']:+.3f} s"
    assert metrics(at)["Official lap-time delta"] == f"{body['lap_time_delta_s']:+.3f} s"
    loss, gain = summary["largest_time_loss"], summary["largest_time_gain"]
    assert metrics(at)["Largest loss (lap 2)"] == f"{loss['time_s']:.3f} s"
    assert metrics(at)["Largest gain (lap 2)"] == f"{gain['time_s']:.3f} s" and gain["time_s"] > 0  # turns 2 and 6
    captions = _captions(at)
    assert (
        f"Largest loss: {loss['time_s']:.3f} s between {loss['from_m']:.0f} m and {loss['to_m']:.0f} m; "
        f"largest gain: {gain['time_s']:.3f} s between {gain['from_m']:.0f} m and {gain['to_m']:.0f} m. "
        "Each is the largest net change of the gap over any stretch"
    ) in captions
    # The API's method note, with plain words for the response fields it names.
    assert (
        "Method: The largest loss or gain is the largest continuous change of the gap for lap 2; its start and end "
        "bound the stretch holding all but"
    ) in captions
    assert "are reported as none." in captions
    for field in ("time\\_s", "delta\\_time", "loss\\_gain", "null"):
        assert field not in captions, field
    text = page_text(at)
    assert "Compared: the synthetic example laps 14 and 15 (not recorded telemetry)" in text
    assert "Alignment: **lap fraction** (requested auto)" in text

    laps = frame(at, "Lap time (s)")
    for row, key in zip(laps.to_dict("records"), ("lap1", "lap2"), strict=True):
        lap = body["laps"][key]
        assert (row["Lap number"], row["Samples"]) == (lap["lap_number"], lap["samples"])
        assert row["Lap time (s)"] == lap["lap_time_s"] and row["Telemetry span (s)"] == lap["telemetry_duration_s"]
    sectors = frame(at, "Lap 2 - lap 1 (s)")
    assert sectors["Lap 1 (s)"].tolist() == body["laps"]["lap1"]["sector_times_s"]
    assert sectors["Lap 2 (s)"].tolist() == body["laps"]["lap2"]["sector_times_s"]
    assert sectors["Lap 2 - lap 1 (s)"].tolist() == body["sector_time_deltas_s"]
    braking, first_zone = frame(at, "Lap 1 braking from (m)"), body["braking"]["matched_zones"][0]
    assert braking.iloc[0].tolist() == [
        first_zone["lap1_start_m"], first_zone["lap2_start_m"], 0.0, first_zone["lap1_end_m"], first_zone["lap2_end_m"]
    ]
    # -0.04 m as displayed (one decimal): 0.0, not -0.0; a sign that means nothing is not shown.
    assert first_zone["brake_point_delta_m"] == -0.04
    deltas = braking["Brake point delta (m; > 0 = lap 2 later)"].tolist()
    assert deltas == [round(zone["brake_point_delta_m"], 1) + 0.0 for zone in body["braking"]["matched_zones"]]
    assert all(math.copysign(1.0, delta) > 0 for delta in deltas if delta == 0)
    # The method note: a field name that starts a sentence is replaced by a capitalised phrase.
    assert "greatest overlap first. The brake point delta \\> 0 means lap 2 starts braking later." in _captions(at)
    assert len(braking) == body["braking"]["matched_zone_count"] == 6  # one per example corner
    measures = frame(at, "Measure").set_index("Measure")
    throttle, gear, drs = body["throttle"], body["gear"], body["drs"]
    assert measures.loc["Full-throttle share of distance"].tolist() == [
        f"{throttle['lap1_full_throttle_fraction']:.1%}",
        f"{throttle['lap2_full_throttle_fraction']:.1%}",
    ]
    assert measures.loc["Highest gear"].tolist() == [str(gear["lap1_max_gear"]), str(gear["lap2_max_gear"])]
    assert measures.loc["Gear changes"].tolist() == [str(gear["lap1_gear_changes"]), str(gear["lap2_gear_changes"])]
    assert measures.loc["DRS open distance (m)"].tolist() == [
        f"{drs['lap1_open_distance_m']:.0f}",
        f"{drs['lap2_open_distance_m']:.0f}",
    ]
    traces = frame(at, "Gap lap 2 - lap 1 (s)")
    # Signed differences as displayed (3 decimals), so a tiny negative gap never shows as -0.000.
    assert traces["Gap lap 2 - lap 1 (s)"].tolist() == [round(value, 3) + 0.0 for value in body["traces"]["delta_time_s"]]
    # Every number column has one fixed number of decimals: seconds 3, metres and km/h 1.
    formats = {}
    for element in at.dataframe:
        formats.update(column_formats(element))
    assert {name: formats[name] for name in ("Lap 1 (s)", "Lap 2 - lap 1 (s)", "Lap time (s)", "Telemetry span (s)")} == {
        "Lap 1 (s)": "%.3f",
        "Lap 2 - lap 1 (s)": "%.3f",
        "Lap time (s)": "%.3f",
        "Telemetry span (s)": "%.3f",
    }
    assert formats["Lap 1 braking from (m)"] == formats["Brake point delta (m; > 0 = lap 2 later)"] == "%.1f"
    assert formats["Distance (m)"] == formats["Lap 2 speed (km/h)"] == "%.1f"


def test_ghost_template_round_trips_through_the_upload(ui_api, served_file, monkeypatch):
    compared = record_calls(monkeypatch, ui_api, "generate_ghost")
    at = run_page("ghost")
    at.segmented_control(key="ghost_source").set_value("Upload CSV files").run()
    assert_no_exception(at)
    template = pd.read_csv(io.BytesIO(served_file(at.download_button[0].proto.url)))
    assert list(template.columns) == ["timestamp", "speed", "throttle", "brake", "gear", "drs"]
    # Each lap's fields say which lap they belong to (screen readers read the label, not the heading).
    assert [at.file_uploader(key=f"ghost_{key}_file").label for key in ("lap1", "lap2")] == [
        "Reference lap CSV",
        "Comparison lap CSV",
    ]
    assert at.number_input(key="ghost_lap2_lap_time").label == "Comparison lap time in s (optional)"
    assert at.text_input(key="ghost_lap1_sectors").label == "Reference sector times in s (optional)"

    slower = template.rename(columns={"timestamp": "time"}).assign(speed=(template["speed"] * 0.99).round(1))
    slower["distance"] = range(len(slower))  # ignored: the API derives distance itself
    at.file_uploader(key="ghost_lap1_file").set_value(("reference.csv", template.to_csv(index=False).encode(), "text/csv"))
    at.file_uploader(key="ghost_lap2_file").set_value(("slower.csv", slower.to_csv(index=False).encode(), "text/csv"))
    at.number_input(key="ghost_lap1_lap_number").set_value(7)
    at.text_input(key="ghost_track_upload").input("Test track")
    at.selectbox(key="ghost_alignment").set_value("distance")
    at.button(key="ghost_submit").click().run()
    assert_no_exception(at)

    payload, body = compared[0]["args"][0], compared[0]["result"]
    lap1, lap2 = payload["lap1_telemetry"], payload["lap2_telemetry"]
    assert lap1["timestamps"] == template["timestamp"].tolist() and lap1["lap_number"] == 7
    assert lap2["drs"] == template["drs"].tolist() and isinstance(lap2["gear"][0], int)
    assert "lap_time" not in lap1 and "distance" not in lap2
    assert payload["track_section"] == "Test track" and payload["alignment"] == "distance"
    assert at.info[0].value == "Comparison lap (`slower.csv`): ignored columns: `distance` (not used by the comparison)"
    assert metrics(at)["Official lap-time delta"] == "not supplied"
    assert body["summary"]["final_delta_s"] > 0.5  # 1 % slower everywhere
    assert metrics(at)["Gap at the end"] == f"{body['summary']['final_delta_s']:+.3f} s"
    assert "Compared: `reference.csv` (reference) vs `slower.csv` (comparison)" in page_text(at)
    assert len(at.image) == 1


def test_ghost_reads_fastf1_style_columns(ui_api, monkeypatch):
    compared = record_calls(monkeypatch, ui_api, "generate_ghost")
    fastf1 = b"Time,Speed,nGear,Brake,RPM\n0.0,250,7,False,11000\n0.2,252,7,False,11100\n0.4,240,6,True,10500\n"
    at = run_page("ghost")
    at.segmented_control(key="ghost_source").set_value("Upload CSV files").run()
    at.file_uploader(key="ghost_lap1_file").set_value(("fastf1.csv", fastf1, "text/csv"))
    at.file_uploader(key="ghost_lap2_file").set_value(("fastf1_b.csv", fastf1.replace(b",7,", b",6,"), "text/csv"))
    at.number_input(key="ghost_lap1_lap_time").set_value(0.45)  # official time, longer than the 0.4 s of samples
    at.button(key="ghost_submit").click().run()
    assert_no_exception(at)
    payload, body = compared[0]["args"][0], compared[0]["result"]
    lap1 = payload["lap1_telemetry"]
    assert lap1["gear"] == [7, 7, 6] and lap1["brake"] == [0.0, 0.0, 1.0] and lap1["speed"] == [250.0, 252.0, 240.0]
    assert payload["lap2_telemetry"]["gear"] == [6, 6, 6] and lap1["lap_time"] == 0.45
    assert at.info[0].value == "Reference lap (`fastf1.csv`): ignored columns: `RPM` (not used by the comparison)"
    assert metrics(at)["Gap at the end"] == "+0.000 s"
    laps = frame(at, "Lap time (s)").set_index("Lap")
    assert laps.loc["Reference (lap 1)", "Lap time (s)"] == body["laps"]["lap1"]["lap_time_s"] == 0.45
    assert laps.loc["Reference (lap 1)", "Telemetry span (s)"] == 0.4
    assert laps["Lap time from"].tolist() == ["supplied", "telemetry span"]
    measures = frame(at, "Measure").set_index("Measure")
    assert measures.loc["Highest gear"].tolist() == ["7", "6"] and measures.loc["Gear changes"].tolist() == ["1", "0"]


def test_ghost_bad_csvs_are_explained_without_calling_the_api(ui_api, monkeypatch):
    compared = record_calls(monkeypatch, ui_api, "generate_ghost")
    at = run_page("ghost")
    at.segmented_control(key="ghost_source").set_value("Upload CSV files").run()
    at.button(key="ghost_submit").click().run()
    problems = _input_problems(at)
    assert md_text("Reference lap: upload a CSV file") in problems and md_text("Comparison lap: upload a CSV file") in problems
    at.file_uploader(key="ghost_lap1_file").set_value(("no_speed.csv", b"timestamp,throttle\n0,1\n0.1,1\n", "text/csv"))
    bad = b"time,speed,throttle\n0,200,100\n0.1,fast,0.5\n0.1,210,0\n"
    at.file_uploader(key="ghost_lap2_file").set_value(("bad.csv", bad, "text/csv"))
    at.text_input(key="ghost_lap2_sectors").input("28.1, 38.4")
    at.text_input(key="ghost_track_upload").input("Monza!")
    at.button(key="ghost_submit").click().run()
    assert_no_exception(at)

    assert compared == [] and not at.image
    error = _input_problems(at)
    for expected in (
        "Track or section name: use letters, digits, spaces, _ and -",
        "Reference lap (`no_speed.csv`): missing the `speed` column (columns found: `timestamp`, `throttle`)",
        "Comparison lap (`bad.csv`): column `speed`: data row 2: 'fast' is not a number",
        "Comparison lap (`bad.csv`): column `throttle`: data row 1: 100 is outside 0-1 (divide percentages by 100)",
        "Comparison lap: give exactly three positive sector times in seconds",
    ):
        assert md_text(expected) in error

    at.file_uploader(key="ghost_lap2_file").set_value(("stalled.csv", b"time,speed\n0,200\n0.5,201\n0.5,202\n", "text/csv"))
    at.file_uploader(key="ghost_lap1_file").set_value(("excel.csv", b"time;speed\n0;200\n1;200\n", "text/csv"))
    at.text_input(key="ghost_lap2_sectors").input("")
    at.text_input(key="ghost_track_upload").input("")
    at.button(key="ghost_submit").click().run()
    error = _input_problems(at)
    assert md_text("the file seems to use ';' between values: save it with commas") in error
    assert md_text("time must increase from row to row: data row 3 (0.5 s) does not come after data row 2 (0.5 s)") in error
    assert compared == []


def test_ghost_checks_documented_ranges_and_row_lengths_before_sending(ui_api, monkeypatch):
    compared = record_calls(monkeypatch, ui_api, "generate_ghost")
    at = run_page("ghost")
    at.segmented_control(key="ghost_source").set_value("Upload CSV files").run()
    fast = b"time,speed,gear\n0,450,7\n1,300,9\n"
    at.file_uploader(key="ghost_lap1_file").set_value(("fast.csv", fast, "text/csv"))
    # One value more per row than the header: pandas would silently use the time as a row index.
    shifted = b"time,speed\n" + b"".join(f"{i * 0.1:.1f},{200 + i},{i % 2}\n".encode() for i in range(20))
    at.file_uploader(key="ghost_lap2_file").set_value(("shifted.csv", shifted, "text/csv"))
    at.button(key="ghost_submit").click().run()
    assert_no_exception(at)
    error = _input_problems(at)
    assert md_text("Reference lap (`fast.csv`): column `speed`: data row 1: 450 is outside 0-400 km/h") in error
    assert md_text("Reference lap (`fast.csv`): column `gear`: data row 2: '9' is not a gear number 0-8") in error
    assert md_text(
        "Comparison lap (`shifted.csv`): data rows have more values than the header has column names: check the "
        "header row, and that values are separated by commas"
    ) in error

    fastf1_drs = b"time,speed,drs\n0,300,8\n1,310,12\n"
    at.file_uploader(key="ghost_lap1_file").set_value(("drs.csv", fastf1_drs, "text/csv"))
    at.file_uploader(key="ghost_lap2_file").set_value(("drs2.csv", b"time,speed,drs\n0,300,0\n1,310,yes\n", "text/csv"))
    at.button(key="ghost_submit").click().run()
    error = _input_problems(at)
    assert md_text("column `drs`: data row 1: '8' is not true/false or 1/0 (FastF1 DRS codes: convert with DRS >= 10)") in error
    assert md_text("column `drs`: data row 2: 'yes' is not true/false or 1/0") in error and error.count("FastF1") == 1
    assert compared == []


def test_ghost_shows_the_api_rejection_of_plausible_looking_csvs(ui_api):
    at = run_page("ghost")
    at.segmented_control(key="ghost_source").set_value("Upload CSV files").run()
    millimetres = b"time,speed,x,y\n0,200,2000000,0\n1,200,2055000,0\n"  # x/y must be metres (at most 1e6)
    at.file_uploader(key="ghost_lap1_file").set_value(("mm.csv", millimetres, "text/csv"))
    at.file_uploader(key="ghost_lap2_file").set_value(("ok.csv", b"time,speed\n0,200\n1,200\n", "text/csv"))
    at.button(key="ghost_submit").click().run()
    assert_no_exception(at)
    assert at.error[0].value.startswith("The comparison was rejected by the API (HTTP 422)")
    assert md_text("lap1_telemetry.x[0]: Input should be less than or equal to 1000000") in markdown_text(at)
    assert (
        "In these messages `lap1_telemetry` is the reference CSV and `lap2_telemetry` the comparison CSV; `[i]` is data row i + 1"
    ) in _captions(at)
    assert not at.image and not at.metric


# ------------------------------------------------------------------ driver radio


def test_radio_labels_a_synthetic_voice_clip(ui_api, monkeypatch):
    classified = record_calls(monkeypatch, ui_api, "classify_emotion")
    wav = _speech_wav()
    at = run_page("radio")
    assert_no_exception(at)
    assert "Heuristic · not a validated emotion model" in markdown_text(at)
    assert "**Labels depend on recording level:** the energy bands are absolute" in _captions(at)
    at.file_uploader(key="radio_file").set_value(("radio.wav", wav, "audio/wav"))
    at.button(key="radio_submit").click().run()
    assert_no_exception(at)

    (audio_file,), options = classified[0]["args"], classified[0]["kwargs"]
    assert base64.b64decode(audio_file) == wav and options == {"transcribe": False}
    body = classified[0]["result"]
    assert body["evidence_combination"] == "acoustic_only"
    assert metrics(at)["Emotion"] == body["emotion"].capitalize()
    assert metrics(at)["Confidence"] == f"{body['confidence']:.3f}"
    confidence = next(metric for metric in at.metric if metric.label == "Confidence")
    assert "not a probability" in confidence.help
    assert metrics(at)["Duration"] == "1.5 s"
    assert metrics(at)["Acoustic label"] == body["acoustic_emotion"].capitalize()
    assert _acoustic_confidence(at) == (f"{body['acoustic_confidence']:.3f}", "acoustic confidence")
    text = page_text(at)
    assert "Confidence is the acoustic confidence: a heuristic score, not the probability that the driver feels" in text
    # The API's formula, with the similarities it used; they reproduce its score.
    (best, best_score), (runner_up, runner_up_score) = sorted(body["acoustic_profile_scores"].items(), key=lambda i: -i[1])[:2]
    assert (
        f"Acoustic confidence {body['acoustic_confidence']:.3f} = 0.45 × the best profile similarity + 0.55 × its lead "
        "over the runner-up, at most 0.95; below 0.20, or when two profiles tie for the best similarity, the acoustic label is neutral. Here: best "
        f"{best} {best_score:.3f}, runner-up {runner_up} {runner_up_score:.3f}, lead {best_score - runner_up_score:.3f}."
    ) in _captions(at)
    assert body["acoustic_confidence"] == pytest.approx(0.45 * best_score + 0.55 * (best_score - runner_up_score), abs=1e-3)
    assert best_score > runner_up_score and not at.warning  # a clear lead: no tie warning
    assert "The acoustic label is used" in text and "Transcription was not requested" in text
    profiles, features = at.dataframe[0].value, at.dataframe[1].value
    assert profiles.iloc[0]["Profile"] == max(body["acoustic_profile_scores"], key=body["acoustic_profile_scores"].get)
    # The marker for the four label features comes right after the name, and those rows come first (narrow screens).
    assert list(features.columns[:2]) == ["Feature", "Used for the label"]
    assert features["Used for the label"].tolist() == [True] * 4 + [False] * (len(features) - 4)
    pitch = features.set_index("Feature").loc["Mean pitch"]
    assert pitch["Value"] == round(body["audio_features"]["mean_pitch"], 4) and bool(pitch["Used for the label"])
    assert 150 < pitch["Value"] < 170  # the clip's 160 Hz fundamental
    assert column_formats(at.dataframe[1]) == {"Value": "%.4f"}  # one number of decimals for every feature


def test_radio_and_assistant_explain_a_tie_for_the_best_profile(ui_api, monkeypatch):
    # A voice inside both the calm and the focused bands: the two profiles tie at 1.000. The audio does not
    # separate them, so the API reports a neutral acoustic label with confidence 0 instead of picking one.
    wav = _speech_wav(vibrato=25.0, gain=0.5)
    classified = record_calls(monkeypatch, ui_api, "classify_emotion")
    at = run_page("radio")
    at.file_uploader(key="radio_file").set_value(("radio.wav", wav, "audio/wav"))
    at.button(key="radio_submit").click().run()
    assert_no_exception(at)
    body = classified[0]["result"]
    scores = body["acoustic_profile_scores"]
    assert scores["calm"] == scores["focused"] == 1.0 and max(scores.values()) == 1.0
    assert (body["acoustic_emotion"], body["acoustic_confidence"]) == ("neutral", 0.0)
    tie = (
        "**Tied profiles:** calm and focused both score 1.000, so the audio does not separate them: the acoustic "
        "label is **neutral** with confidence 0."
    )
    assert [warning.value for warning in at.warning] == [tie]
    assert metrics(at)["Emotion"] == "Neutral" and metrics(at)["Acoustic label"] == "Neutral"  # the API's label, kept
    assert _acoustic_confidence(at) == ("0.000", "acoustic confidence")
    assert "Here: best calm 1.000, runner-up focused 1.000, lead 0.000." in _captions(at)

    monkeypatch.setattr(app.main, "_transcription_availability", lambda: (False, WHISPER_MISSING))
    at = run_page("assistant")
    at.file_uploader(key="assistant_file").set_value(("radio.wav", wav, "audio/wav"))
    _ask(at, "How does the driver sound on the team radio?")
    assert_no_exception(at)
    assert [warning.value for warning in at.warning] == [tie]
    assert "**Acoustic confidence:** 0.45 × the best profile similarity + 0.55 × its lead over the runner-up" in (
        _captions(at)
    )


def test_radio_shows_the_transcript_and_how_it_changed_the_score(ui_api, monkeypatch):
    monkeypatch.setattr(app.main, "_transcription_availability", lambda: (True, None))
    transcriber = get_transcriber()
    monkeypatch.setattr(transcriber, "availability", lambda: (True, None))
    monkeypatch.setattr(transcriber, "transcribe", lambda audio_file: "that was brilliant")
    classified = record_calls(monkeypatch, ui_api, "classify_emotion")
    at = run_page("radio")
    at.toggle(key="radio_transcribe").set_value(True)
    at.file_uploader(key="radio_file").set_value(("radio.wav", _speech_wav(), "audio/wav"))
    at.button(key="radio_submit").click().run()
    assert_no_exception(at)

    body = classified[0]["result"]
    assert classified[0]["kwargs"] == {"transcribe": True} and body["transcription_status"] == "completed"
    # One "excited" keyword against a calm voice: too weak to override, so it lowers the score instead.
    assert body["evidence_combination"] == "acoustic_kept_text_disagrees" and body["emotion"] == body["acoustic_emotion"]
    assert body["confidence"] < body["acoustic_confidence"]
    assert metrics(at)["Emotion"] == body["emotion"].capitalize()
    assert metrics(at)["Confidence"] == f"{body['confidence']:.3f}"
    assert metrics(at)["Acoustic label"] == body["acoustic_emotion"].capitalize()
    assert _acoustic_confidence(at) == (f"{body['acoustic_confidence']:.3f}", "acoustic confidence")
    markdown, captions = markdown_text(at), _captions(at)
    assert "> that was brilliant" in markdown
    assert f"Transcript keyword label: **{body['text_emotion']}** (keyword score {body['text_confidence']:.3f})" in markdown
    assert (
        "Confidence is computed from the acoustic confidence and the transcript keyword score by the rule below: a "
        "heuristic score, not the probability"
    ) in captions
    assert "the acoustic label is kept; confidence = max(0, acoustic - text / 2)" in captions
    keywords = frame(at, "Keywords found")
    assert dict(zip(keywords["Emotion"], keywords["Keywords found"], strict=True)) == {
        emotion: ", ".join(words) for emotion, words in body["text_keyword_hits"].items()
    }
    assert keywords.set_index("Emotion").loc["excited", "Keywords found"] == "brilliant"


def test_radio_explains_that_transcription_is_unavailable(ui_api, monkeypatch):
    monkeypatch.setattr(app.main, "_transcription_availability", lambda: (False, WHISPER_MISSING))
    at = run_page("radio")
    assert_no_exception(at)
    assert at.toggle(key="radio_transcribe").disabled
    notice = at.info[0].value
    assert f"**Transcription is unavailable on this API:** {WHISPER_MISSING}" in notice
    assert "The acoustic analysis works without it" in notice


def test_radio_reports_a_transcription_the_api_could_not_run(ui_api, monkeypatch):
    # /health reported Whisper as ready, but it is gone by the time the clip is analysed.
    monkeypatch.setattr(app.main, "_transcription_availability", lambda: (True, None))
    monkeypatch.setattr(get_transcriber(), "availability", lambda: (False, "ffmpeg is not installed"))
    classified = record_calls(monkeypatch, ui_api, "classify_emotion")
    at = run_page("radio")
    assert not at.toggle(key="radio_transcribe").disabled and not at.info
    at.toggle(key="radio_transcribe").set_value(True)
    at.file_uploader(key="radio_file").set_value(("radio.wav", _speech_wav(), "audio/wav"))
    at.button(key="radio_submit").click().run()
    assert_no_exception(at)
    assert classified[0]["kwargs"] == {"transcribe": True}
    assert classified[0]["result"]["transcription_status"] == "unavailable"
    assert "Transcription was requested but Whisper is unavailable on the API: ffmpeg is not installed" in at.info[0].value


def test_radio_refuses_an_oversized_clip_before_sending(ui_api, monkeypatch):
    classified = record_calls(monkeypatch, ui_api, "classify_emotion")
    at = run_page("radio")
    at.file_uploader(key="radio_file").set_value(("long.wav", b"\0" * (20 * 1024 * 1024 + 1), "audio/wav"))
    at.button(key="radio_submit").click().run()
    assert_no_exception(at)
    assert classified == [] and not at.metric
    assert at.error[0].value == (
        "Nothing was sent to the API. The clip is 20.0 MiB; the API accepts at most 20 MiB. Trim it, or save it in a "
        "compressed format (FLAC, OGG or MP3)."
    )


def test_radio_when_the_api_is_down(ui_api_down):
    at = run_page("radio")
    assert_no_exception(at)
    assert "Transcription availability unknown: the API status check failed (connection failed" in page_text(at)
    assert not at.toggle(key="radio_transcribe").disabled  # unknown is not reported as unavailable
    at.file_uploader(key="radio_file").set_value(("radio.wav", _speech_wav(), "audio/wav"))
    at.button(key="radio_submit").click().run()
    assert_no_exception(at)
    assert at.error[0].value.startswith(f"The analysis could not reach the API at `{ui_api_down.base_url}`")
    assert "uvicorn app.main:app --port" in at.info[0].value and not at.metric


def test_radio_shows_why_the_api_rejected_a_clip(ui_api):
    at = run_page("radio")
    at.file_uploader(key="radio_file").set_value(("notes.wav", b"these are not audio samples", "audio/wav"))
    at.button(key="radio_submit").click().run()
    assert_no_exception(at)
    assert at.error[0].value.startswith("The analysis was rejected by the API (HTTP 422)")
    assert "Could not decode audio" in markdown_text(at) and not at.metric


# ------------------------------------------------------------------ ask the copilot


def _ask(at: AppTest, question: str) -> AppTest:
    at.text_area(key="assistant_query").input(question)
    return at.button(key="assistant_ask").click().run()


def test_assistant_answers_a_strategy_question_with_the_documented_example(ui_api, monkeypatch):
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    at = run_page("assistant")
    assert_no_exception(at)
    at.pills(key="assistant_example").set_value("What pit stop strategy should I use?").run()
    assert at.text_area(key="assistant_query").value == "What pit stop strategy should I use?"
    at.button(key="assistant_add_strategy").click().run()
    assert_no_exception(at)
    assert "from the API's documented StrategyRequest example: example values, not your data" in at.info[0].value
    example = documented_example(ui_api.get_openapi(), "StrategyRequest")
    assert json.loads(at.text_area(key="assistant_context").value) == example

    at.button(key="assistant_ask").click().run()
    assert_no_exception(at)
    query, context = asked[0]["args"]
    assert query == "What pit stop strategy should I use?" and context == example
    body = asked[0]["result"]
    assert metrics(at)["Routed to"] == "Race strategy"
    assert metrics(at)["Score"] == "no score" and body["confidence"] is None
    assert metrics(at)["Data sources"] == "Strategy engine" and body["data_sources"] == ["strategy_engine"]
    sources = next(metric for metric in at.metric if metric.label == "Data sources")
    assert "(data_sources: strategy_engine)" in sources.help  # the API's id, to find it in the raw response
    assert md_text(body["answer"]) in markdown_text(at)
    # The answer stays marked as computed from example values after the helper's notice is gone.
    assert (
        f"Question: {md_text(query)} · context sent: "
        + md_text(", ".join(example) + " (all the API's documented example values, not your data)")
    ) in _captions(at)
    assert "query\\_type" not in _captions(at)  # the Routed to metric names the module
    plans = at.dataframe[0].value
    best = ui_api.generate_strategy(example)["strategies"][0]
    assert plans.iloc[0]["Plan"] == best["strategy_id"]
    assert plans.iloc[0]["Compounds"] == " → ".join(best["tire_compounds"])  # the strategy page's arrows
    assert column_formats(at.dataframe[0]) == {"Projected time (s)": "%.3f", "Behind best (s)": "%.3f"}
    not_modelled = body["additional_context"]["not_modelled_inputs"]
    assert not_modelled and not_modelled_text(not_modelled) in _captions(at)  # worded as on the strategy page
    assert "Heuristic · not validated" in markdown_text(at)
    assert "How the question was routed" in expander_labels(at)
    assert "Race strategy (strategy)" in frame(at, "Matched terms")["Module"].tolist()


def test_assistant_answers_a_setup_question_with_the_documented_example(ui_api, monkeypatch):
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    at = run_page("assistant")
    at.button(key="assistant_add_setup").click().run()
    _ask(at, "What setup should I run for this track?")
    assert_no_exception(at)
    body = asked[0]["result"]
    extra = body["additional_context"]
    assert body["query_type"] == "technical" and body["data_sources"] == ["setup_optimizer"]
    assert metrics(at)["Routed to"] == "Car setup" and metrics(at)["Multi-start agreement"] == f"{body['confidence']:.3f}"
    assert "Confidence" not in metrics(at)
    setup = frame(at, "Parameter").set_index("Parameter")
    for name in ("ride_height", "front_wing_angle", "rear_wing_angle", "brake_bias"):
        assert setup.loc[name.replace("_", " "), "Value"] == extra[name]
        assert setup.loc[name.replace("_", " "), "Unit"] == extra["units"][name]
    for name, value in extra["diff_settings"].items():
        assert setup.loc[f"diff {name}", "Value"] == value
    assert f"Multi-start agreement (the API's confidence): {md_text(extra['confidence_method'])}" in _captions(at)
    assert "Car setup (technical)" in frame(at, "Matched terms")["Module"].tolist()


def test_assistant_lap_question_restates_the_telemetry_with_its_caveats(ui_api, monkeypatch):
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    at = run_page("assistant")
    at.button(key="assistant_add_telemetry").click().run()
    at.pills(key="assistant_example").set_value("How does my latest lap compare with my best?").run()
    at.button(key="assistant_ask").click().run()
    assert_no_exception(at)
    body = asked[0]["result"]
    extra = body["additional_context"]
    assert body["query_type"] == "performance" and body["confidence"] > 0
    assert metrics(at)["Routed to"] == "Lap performance" and metrics(at)["Coverage"] == f"{body['confidence']:.3f}"
    evidence = frame(at, "Supplied value").set_index("Telemetry")
    assert json.loads(evidence.loc["lap_times", "Supplied value"]) == extra["evidence"]["lap_times"]
    captions = _captions(at)
    assert f"Method: {md_text(extra['method'])}" in captions and "no causal analysis" in extra["method"]
    assert f"Coverage: {md_text(extra['confidence_basis'])}" in captions and "not a statistical probability" in captions
    assert "context sent: telemetry (all the API's documented example values, not your data)" in captions

    context = json.loads(at.text_area(key="assistant_context").value)
    context["telemetry"]["lap_times"][-1] = 95.2  # now the user's own data
    at.text_area(key="assistant_context").input(json.dumps(context))
    at.button(key="assistant_ask").click().run()
    assert_no_exception(at)
    question = next(caption.value for caption in at.caption if caption.value.startswith("Question:"))
    assert question.endswith("context sent: telemetry") and len(asked) == 2


def test_assistant_context_helpers_merge_and_clear(ui_api):
    at = run_page("assistant")
    at.text_area(key="assistant_context").input("{not json")
    at.button(key="assistant_add_setup").click().run()
    assert "Fix the context JSON before adding an example: not valid JSON" in at.info[0].value
    at.button(key="assistant_clear").click().run()
    at.button(key="assistant_add_setup").click().run()
    at.button(key="assistant_add_telemetry").click().run()
    assert_no_exception(at)
    context = json.loads(at.text_area(key="assistant_context").value)
    assert set(context) == {"driver_preferences", "track_profile", "weather", "n_trials", "seed", "telemetry"}
    assert context["track_profile"]["track_name"] == "Silverstone Circuit"
    at.button(key="assistant_clear").click().run()
    assert at.text_area(key="assistant_context").value == ""


def test_assistant_answers_a_regulatory_question_with_citations(ui_api, installed_rag, monkeypatch):
    rag, llm = installed_rag
    rag.build_index()
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    at = _ask(run_page("assistant"), "What is the pit lane speed limit rule?")
    assert_no_exception(at)
    body = asked[0]["result"]
    extra = body["additional_context"]
    assert body["query_type"] == "regulatory" and extra["grounded"] and extra["citations"] == ["S1"]
    assert len(llm.calls) == 1  # one scripted answer; no real provider exists in tests
    assert metrics(at)["Routed to"] == "FIA regulations"
    assert metrics(at)["Evidence strength"] == f"{body['confidence']:.3f}" and body["confidence"] > 0
    assert metrics(at)["Data sources"] == "FIA regulation passages" and body["data_sources"] == ["fia_regulations"]
    assert at.success[0].value.startswith("Grounded answer: it cites the passages below")
    text = page_text(at)
    assert "80km/h \\[S1\\]" in text  # the answer, with its citation label shown literally
    assert "(similarity of the best cited regulation passage; not a probability)" in text
    # The cited passage as the FIA regulations page shows it: the same title, badges, text and source.
    score = extra["retrieved_passages"][0]["score"]
    assert f"S1 · cited · B1.6 · printed page B1 · PDF page 1 · similarity {score:.3f}" in expander_labels(at)
    assert ":blue-badge[S1] :green-badge[cited] :gray-badge[regulation passage]" in markdown_text(at)
    assert "A speed limit of\n> 80km/h will be imposed in the pit lane" in text
    assert "Source: section\\_b\\_sporting.pdf (no official URL recorded)" in _captions(at)


def test_assistant_shows_a_cited_definition_as_a_definition(ui_api, tmp_path, monkeypatch, qdrant):
    llm = ScriptedChatModel(
        "Speeding in the pit lane during a TTCS gives a drive through penalty [S1][S2]. TTCS include the Race session [S2]."
    )
    rag = make_definitions_rag(write_definitions_corpus(tmp_path / "fia_docs"), tmp_path, qdrant, llm=llm)
    rag.build_index()
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    at = _ask(run_page("assistant"), DEFINITIONS_QUESTION)
    assert_no_exception(at)
    extra = asked[0]["result"]["additional_context"]
    assert extra["grounded"] and extra["citations"] == ["S1", "S2"]
    labels = expander_labels(at)
    assert "S2 · cited · definition\\: Total Time Classified Session (TTCS) · printed page B85 · PDF page 2" in labels
    regulation = next(label for label in labels if label.startswith("S1"))
    assert "B1.6" in regulation and "similarity" in regulation  # definitions have no similarity score
    assert ":blue-badge[S2] :green-badge[cited] :violet-badge[definition]" in markdown_text(at)
    assert metrics(at)["Evidence strength"] == f"{asked[0]['result']['confidence']:.3f}"  # a regulation passage is cited


def test_assistant_answer_citing_only_definitions_has_no_evidence_strength(ui_api, tmp_path, monkeypatch, qdrant):
    llm = ScriptedChatModel("TTCS include the Race session [S2].")
    rag = make_definitions_rag(write_definitions_corpus(tmp_path / "fia_docs"), tmp_path, qdrant, llm=llm)
    rag.build_index()
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    at = _ask(run_page("assistant"), DEFINITIONS_QUESTION)
    assert_no_exception(at)
    body = asked[0]["result"]
    extra = body["additional_context"]
    assert body["query_type"] == "regulatory" and extra["grounded"] and extra["citations"] == ["S2"]
    assert body["confidence"] == 0.0  # the API counts cited regulation passages only
    # As on the FIA regulations page: no evidence strength, not a misleading 0.000.
    assert metrics(at)["Evidence strength"] == "–"
    captions = _captions(at)
    assert "Only official definitions were cited. Definitions are not retrieved by similarity" in captions
    assert "Evidence strength 0.000" not in captions


def test_assistant_explains_a_declined_regulatory_question(ui_api, installed_rag):
    rag, _ = installed_rag
    rag.build_index()
    at = _ask(run_page("assistant"), "What is the rule for the rear wing colour?")
    assert_no_exception(at)
    warning = at.warning[0].value
    assert warning == (
        "Declined (model\\_declined): The answer model found no answer in the evidence. The model read the retrieved "
        "passages and replied that they do not contain enough evidence. No unverified answer is shown."
    )
    assert "**Next step:** Check the evidence below: if it does not cover the question" in markdown_text(at)
    assert metrics(at)["Evidence strength"] == "0.000" and metrics(at)["Data sources"] == "none"
    assert "Evidence strength 0.000: a declined answer always scores 0 (not a probability)" in _captions(at)
    evidence = [label for label in expander_labels(at) if label.startswith("Evidence given to the model (")]
    assert len(evidence) == 1 and not any(label.startswith("S") for label in expander_labels(at))
    assert "cited by the rejected output" not in markdown_text(at)  # the model cited nothing
    assert not at.success


def test_assistant_never_presents_a_rejected_answers_citations_as_evidence(ui_api, tmp_path, monkeypatch):
    # The draft cites a real passage but invents the rule number, so it is declined with its citation kept.
    reply = "Under Article B9.9.9 the pit lane speed limit is 80km/h [S1]."
    rag, _, qdrant = build_test_rag(tmp_path / "fia_docs", reply=reply)
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    rag.build_index()
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    try:
        at = _ask(run_page("assistant"), "What is the pit lane speed limit rule?")
        assert_no_exception(at)
    finally:
        qdrant.close()
    extra = asked[0]["result"]["additional_context"]
    assert extra["decline_reason"] == "unsupported_rule_reference" and extra["referenced_rules"] == ["B9.9.9"]
    assert [p["label"] for p in extra["retrieved_passages"] if p["cited"]] == ["S1"]

    assert at.warning[0].value == (
        "Declined (unsupported\\_rule\\_reference): The answer named a rule that is not in its sources. The answer "
        "mentions an article or rule number that does not occur in the passages it cites, so it may be invented or "
        "misattributed. No unverified answer is shown."
    )
    assert "**Next step:** Rephrasing the question can help." in markdown_text(at)
    assert "Rules referenced" not in page_text(at) and "B9.9.9" not in _captions(at) + markdown_text(at)
    assert not any(label.startswith("S1") for label in expander_labels(at))  # no quote card for a rejected citation
    assert "Evidence given to the model (1)" in expander_labels(at)
    assert "Passages marked as cited were cited by the rejected output: they are not verified evidence." in _captions(at)
    assert ":blue-badge[S1] :orange-badge[cited by the rejected output]" in markdown_text(at)
    assert metrics(at)["Evidence strength"] == "0.000" and not at.success


def test_assistant_explains_a_question_without_matching_passages(ui_api, installed_rag):
    rag, llm = installed_rag
    rag.build_index()
    at = _ask(run_page("assistant"), "What is the rule for zebra crossings in the paddock kitchen?")
    assert_no_exception(at)
    assert llm.calls == []  # no passage reached the threshold, so the answer model was not called
    assert at.warning[0].value.startswith(
        "Declined (no\\_evidence\\_above\\_threshold): No passage was similar enough to the question. No regulation "
        "passage reached the similarity threshold, so the answer model was not called."
    )
    assert "**Next step:** Use the regulations' own terms" in markdown_text(at)
    assert "No passage passed the similarity threshold." in _captions(at)


def test_assistant_regulatory_question_without_an_index_is_service_unavailable(ui_api):
    at = run_page("assistant")
    assert ":red-badge[not configured] Regulation QA is not ready on this API" in markdown_text(at)
    _ask(at, "What is the pit lane speed limit rule?")
    assert_no_exception(at)
    assert at.warning[0].value.startswith("Service unavailable (HTTP 503): FIA RAG is unavailable")
    assert not at.metric and "Answer" not in [header.value for header in at.subheader]


def test_assistant_explains_how_to_start_an_unreachable_api(ui_api_down):
    at = _ask(run_page("assistant"), "What pit stop strategy should I use?")
    assert_no_exception(at)
    assert at.error[0].value.startswith(f"The question could not reach the API at `{ui_api_down.base_url}`")
    assert "uvicorn app.main:app --port" in at.info[0].value and not at.metric


def test_assistant_checks_the_context_before_sending(ui_api, monkeypatch):
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    at = run_page("assistant")
    for context, reason in (
        ('{"telemetry": }', "not valid JSON: Expecting value (line 1, column 15)"),
        ('{"telemetry": {"lap_times": [NaN]}}', "NaN is not a valid JSON number"),
        ("[1, 2]", "it must be a JSON object"),
        # Valid JSON that the request body cannot carry: numbers beyond float range and lone surrogates.
        ('{"telemetry": {"lap_times": [95.1, 1e400]}}', "1e400 is beyond the range of a 64-bit number"),
        ("[-1e999]", "-1e999 is beyond the range of a 64-bit number"),
        ('{"note": "\\ud800"}', "it contains text that is not valid Unicode"),
        ("[" * 100_000 + "]" * 100_000, "it is nested too deeply"),
    ):
        at.text_area(key="assistant_context").input(context)
        _ask(at, "How does my latest lap compare with my best?")
        assert_no_exception(at)
        assert md_text(f"Context: {reason}") in _input_problems(at)
    assert asked == []

    example = documented_example(ui_api.get_openapi(), "StrategyRequest")
    example["race_state"]["current_lap"] = 99
    example["car_status"]["engine_wear"] = 3
    at.text_area(key="assistant_context").input(json.dumps(example))
    _ask(at, "What pit stop strategy should I use?")
    assert_no_exception(at)
    assert at.error[0].value.startswith("The question was rejected by the API (HTTP 422)")
    # One line per error, located in the context, without pydantic's "Value error, " prefix.
    markdown = markdown_text(at)
    assert md_text("- context.car_status.engine_wear: Input should be less than or equal to 1 (got 3)") in markdown
    assert md_text("- context.race_state: current_lap (99) must be <= total_laps (57)") in markdown
    assert "Value error" not in markdown and "errors.pydantic.dev" not in markdown


def test_assistant_attached_clip_and_json_audio_conflict(ui_api, monkeypatch):
    monkeypatch.setattr(app.main, "_transcription_availability", lambda: (True, None))
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    at = run_page("assistant")
    at.file_uploader(key="assistant_file").set_value(("radio.wav", _speech_wav(), "audio/wav"))
    at.toggle(key="assistant_transcribe").set_value(True)
    at.text_area(key="assistant_context").input('{"audio_file": "UklGRg=="}')
    _ask(at, "How does the driver sound on the team radio?")
    assert_no_exception(at)
    assert md_text("Context: remove `audio_file` from the JSON, or detach the clip.") in _input_problems(at)
    assert asked == []

    monkeypatch.setattr(get_transcriber(), "availability", lambda: (False, "ffmpeg is not installed"))
    at.text_area(key="assistant_context").input("")
    _ask(at, "How does the driver sound on the team radio?")
    assert_no_exception(at)
    assert asked[0]["args"][1]["transcribe"] is True
    assert md_text("transcription unavailable: ffmpeg is not installed") in page_text(at)


def test_assistant_without_context_says_what_is_missing(ui_api):
    at = _ask(run_page("assistant"), "What setup should I run for this track?")
    assert_no_exception(at)
    assert metrics(at)["Routed to"] == "Car setup" and metrics(at)["Data sources"] == "none"
    # The API reports 0 for a module that did not run; that is not shown as a score.
    confidence = next(metric for metric in at.metric if metric.label == "Multi-start agreement")
    assert confidence.value == "not run" and confidence.help.startswith("The module did not run")
    assert md_text("I need driver_preferences, track_profile and weather in the query context") in markdown_text(at)
    assert at.info[0].value == (
        "No evidence was used: this module answers only from context you supply. Add it with **Setup example** "
        "above (example values) or your own JSON, and ask again."
    )

    _ask(at, "hello there")
    assert_no_exception(at)
    assert metrics(at)["Routed to"] == "No module" and metrics(at)["Score"] == "not run"


def test_assistant_radio_question_without_a_clip_points_to_the_clip_attachment(ui_api):
    at = _ask(run_page("assistant"), "How does the driver sound on the team radio?")
    assert_no_exception(at)
    assert metrics(at)["Routed to"] == "Driver radio" and metrics(at)["Heuristic score"] == "not run"
    info = at.info[0].value
    assert "Attach a clip under **Attach a driver-radio clip** above and ask again" in info
    assert "helpers" not in info and "Example" not in info
    assert "Attach a driver-radio clip (for radio questions)" in expander_labels(at)
    assert [link.proto.page for link in at.get("page_link")][-1] == "radio"


def test_assistant_sends_an_attached_clip_to_the_radio_heuristic(ui_api, monkeypatch):
    monkeypatch.setattr(app.main, "_transcription_availability", lambda: (False, WHISPER_MISSING))
    asked = record_calls(monkeypatch, ui_api, "natural_query")
    wav = _speech_wav()
    at = run_page("assistant")
    toggle = at.toggle(key="assistant_transcribe")
    assert toggle.disabled and toggle.help == f"Unavailable on this API: {WHISPER_MISSING}"
    at.file_uploader(key="assistant_file").set_value(("radio.wav", wav, "audio/wav"))
    _ask(at, "How does the driver sound on the team radio?")
    assert_no_exception(at)
    _, context = asked[0]["args"]
    assert base64.b64decode(context["audio_file"]) == wav and "transcribe" not in context
    body = asked[0]["result"]
    assert body["query_type"] == "emotion" and body["data_sources"] == ["driver_emotion"]
    assert metrics(at)["Routed to"] == "Driver radio"
    confidence = next(metric for metric in at.metric if metric.label == "Heuristic score")
    assert confidence.value == f"{body['confidence']:.3f}" and "combined with the transcript keyword score" in confidence.help
    text = page_text(at)
    assert "context sent: audio\\_file (attached clip)" in text
    assert "combination rule: acoustic only" in text
    assert "**Combination rule:** " + md_text("The acoustic label is used: there was no transcript") in text
    assert "Not a validated emotion model; each confidence is a heuristic score, not a probability" in text
