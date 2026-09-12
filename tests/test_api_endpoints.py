import base64
import io
import math

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _audio_base64() -> str:
    sample_rate = 22050
    t = np.arange(int(sample_rate * 0.35)) / sample_rate
    waveform = 0.12 * np.sin(2 * math.pi * 220 * t)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _lap(lap_number: int, offset: float = 0.0):
    return {
        "timestamps": [0.0, 0.5, 1.0, 1.5],
        "x": [0.0, 10.0, 20.0, 30.0],
        "y": [0.0, 2.0, 1.0, 0.0],
        "speed": [180 + offset, 200 + offset, 210 + offset, 190 + offset],
        "throttle": [0.7, 0.9, 1.0, 0.5],
        "brake": [0.0, 0.0, 0.0, 0.4],
        "steering": [0.0, 0.1, -0.1, 0.0],
        "drs": [False, True, True, False],
        "gear": [5, 6, 7, 5],
        "lap_time": 80.0 + offset / 100.0,
        "sector_times": [26.5, 26.7, 26.8],
        "lap_number": lap_number,
    }


def test_root_and_health_endpoints():
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["docs"] == "/docs"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["modules"]["strategy"] == "ready"


def test_penalty_endpoint_is_explicitly_heuristic():
    response = client.post(
        "/api/penalty/predict",
        json={
            "incident_type": "unsafe_release",
            "track_condition": "dry",
            "intent": "accidental",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method"] == "transparent heuristic triage"
    assert body["referenced_rule"] is None
    assert "not an FIA steward-decision predictor" in body["disclaimer"]


def test_emotion_endpoint_accepts_real_base64_audio():
    response = client.post(
        "/api/emotion/classify",
        json={"audio_file": _audio_base64(), "transcribe": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["duration"] > 0.3
    assert body["transcription"] is None
    assert 0.0 <= body["confidence"] <= 0.95


def test_emotion_endpoint_rejects_invalid_audio():
    response = client.post(
        "/api/emotion/classify",
        json={"audio_file": "this-is-not-a-path-or-base64", "transcribe": False},
    )
    assert response.status_code == 422


def test_ghost_endpoint_returns_served_artifact_url():
    response = client.post(
        "/api/ghost/generate",
        json={
            "lap1_telemetry": _lap(1),
            "lap2_telemetry": _lap(2, 2.0),
            "track_section": "silverstone",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["samples_compared"] == 4
    assert body["visualization_url"].startswith("/artifacts/")

    artifact = client.get(body["visualization_url"])
    assert artifact.status_code == 200
    assert artifact.headers["content-type"].startswith("image/png")


def test_ghost_endpoint_rejects_empty_telemetry():
    response = client.post(
        "/api/ghost/generate",
        json={
            "lap1_telemetry": {"timestamps": []},
            "lap2_telemetry": _lap(2),
            "track_section": "silverstone",
        },
    )
    assert response.status_code == 422


def test_fia_endpoint_does_not_fall_back_to_mock_when_unconfigured(monkeypatch):
    # CI intentionally has neither an OpenAI key nor checked-in FIA PDFs. The
    # endpoint must expose that state rather than returning a fabricated answer.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post(
        "/api/fia/query",
        json={"question": "What rule applies to an unsafe release?"},
    )
    assert response.status_code == 503
    assert "FIA RAG is unavailable" in response.json()["detail"]
