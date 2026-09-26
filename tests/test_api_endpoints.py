"""HTTP contract of every endpoint (FastAPI TestClient; the RAG uses a real in-memory index when configured)."""

import base64
import copy
import json
import io
import math

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app.main as main
import core_modules.rule_checker.fia_rag.pipeline as rag_pipeline
from app.main import DEFAULT_BODY_LIMIT, ROUTE_BODY_LIMITS, app
from core_modules.driver_emotion.emotion_classifier import TranscriptionError
from core_modules.rule_checker.fia_rag import DECLINE_ANSWER
from core_modules.setup_optimizer.schemas import SetupRequest
from tests.helpers import (
    DEFINITIONS_QUESTION,
    ScriptedChatModel,
    make_definitions_rag,
    write_definitions_corpus,
)

client = TestClient(app)


# ------------------------------------------------------------------ fixtures


def _speech_like_base64(seconds=1.0, rate=22050, f0=160.0):
    t = np.arange(int(rate * seconds)) / rate
    voice = sum(np.sin(2 * math.pi * f0 * k * t) / k for k in range(1, 6))
    envelope = 0.6 + 0.4 * np.sin(2 * math.pi * 3 * t)
    buffer = io.BytesIO()
    sf.write(buffer, 0.2 * voice * envelope / 2.3, rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _lap(speed_kmh, seconds=60.0, rate_hz=2.0, lap_number=1):
    n = int(seconds * rate_hz) + 1
    return {
        "timestamps": [i / rate_hz for i in range(n)],
        "speed": [float(speed_kmh)] * n,
        "brake": [0.0] * (n - 10) + [0.8] * 10,
        "drs": [False] * n,
        "lap_number": lap_number,
    }


STRATEGY_REQUEST = {
    "telemetry": {"lap_times": [95.6, 95.3, 95.9, 95.4], "braking_consistency": 0.75},
    "car_status": {"engine_wear": 0.3, "brake_wear": 0.4, "damage": {"front_wing": 0.1}},
    "driver_profile": {"tire_management": 0.7, "risk_tolerance": 0.6, "braking_consistency": 0.75, "throttle_aggressiveness": 0.7},
    "tire_data": {
        "soft": {"base_performance": 1.0, "degradation_rate": 0.004, "warm_up_laps": 2, "peak_performance_window": [2, 10], "pit_stop_delta": 22.0},
        "medium": {"base_performance": 0.992, "degradation_rate": 0.0025, "warm_up_laps": 3, "peak_performance_window": [3, 18], "pit_stop_delta": 22.0},
        "hard": {"base_performance": 0.985, "degradation_rate": 0.0015, "warm_up_laps": 4, "peak_performance_window": [4, 28], "pit_stop_delta": 22.0},
    },
    "race_state": {"current_lap": 18, "total_laps": 57, "weather": "dry", "track_temperature": 32.0,
                   "current_compound": "medium", "current_tire_age": 17},
    "competition": [{"driver_id": "HAM", "tire_compound": "medium", "tire_age": 21, "gap_to_leader": 4.4}],
}

# The documented request example with a small trial budget.
SETUP_REQUEST = {**copy.deepcopy(SetupRequest.model_config["json_schema_extra"]["examples"][0]), "n_trials": 24, "seed": 7}


# ------------------------------------------------------------------ service endpoints


def test_root_docs_and_openapi_describe_the_api():
    assert client.get("/").json()["docs"] == "/docs"
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert {"/api/fia/query", "/api/fia/retrieve", "/api/strategy/generate", "/api/ghost/generate"} <= set(schema["paths"])
    assert "FIAAnswerResponse" in schema["components"]["schemas"]


def test_health_reports_unconfigured_rag_as_degraded():
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["modules"]["fia_rag"] == "not_configured"
    assert any("OPENAI_API_KEY" in problem for problem in body["details"]["fia_rag"]["problems"])
    assert body["modules"]["emotion_transcription"] in {"ready", "unavailable"}


def test_health_survives_invalid_rag_settings(monkeypatch):
    monkeypatch.setenv("FIA_RAG_TOP_K", "five")
    body = client.get("/health").json()
    assert body["modules"]["fia_rag"] == "misconfigured" and body["status"] == "degraded"
    assert client.get("/api/fia/status").json()["state"] == "misconfigured"
    response = client.post("/api/fia/query", json={"question": "What is the pit lane speed limit?"})
    assert response.status_code == 503 and "FIA_RAG_TOP_K" in response.json()["detail"]


def test_health_is_healthy_when_the_index_is_current(installed_rag):
    rag, _ = installed_rag
    assert client.get("/health").json()["modules"]["fia_rag"] == "index_missing"
    rag.build_index()
    body = client.get("/health").json()
    assert body["modules"]["fia_rag"] == "ready" and body["status"] == "healthy"


def test_cors_allows_any_origin_without_credentials():
    response = client.options(
        "/api/penalty/predict", headers={"Origin": "https://example.org", "Access-Control-Request-Method": "POST"}
    )
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


MIB = 1024 * 1024
# The body limits the README documents: audio routes, ghost telemetry, and one route with the default.
DOCUMENTED_BODY_LIMITS = {
    "/api/emotion/classify": 40 * MIB,
    "/api/query/natural": 40 * MIB,
    "/api/ghost/generate": 8 * MIB,
    "/api/penalty/predict": MIB,
}


@pytest.mark.parametrize("route, limit", DOCUMENTED_BODY_LIMITS.items())
def test_each_route_rejects_a_streamed_body_one_byte_over_its_limit(route, limit):
    assert set(ROUTE_BODY_LIMITS) < set(DOCUMENTED_BODY_LIMITS)  # every route with its own limit is covered
    assert ROUTE_BODY_LIMITS.get(route, DEFAULT_BODY_LIMIT) == limit

    def body():  # streamed in 1 MiB chunks, never held at once
        remaining = limit + 1
        while remaining:
            chunk = min(remaining, MIB)
            remaining -= chunk
            yield b"x" * chunk

    response = client.post(route, content=body(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_unexpected_errors_are_json_500s_without_internals(monkeypatch):
    def broken(**_):
        raise KeyError("internal detail")

    monkeypatch.setattr(main, "predict_penalty", broken)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/penalty/predict", json={"incident_type": "collision", "track_condition": "dry", "intent": "accidental"}
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error", "error_type": "KeyError"}


# ------------------------------------------------------------------ FIA RAG


def test_fia_query_returns_503_not_a_mock_answer_when_unconfigured():
    response = client.post("/api/fia/query", json={"question": "What rule applies to an unsafe release?"})
    assert response.status_code == 503
    body = response.json()
    assert body["detail"].startswith("FIA RAG is unavailable") and "/home/" not in body["detail"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "x" * 2001}, {"question": "ok", "top_k": 0}, {"question": "ok", "extra": 1}, {"question": 5}],
)
def test_fia_query_validation(payload):
    assert client.post("/api/fia/query", json=payload).status_code == 422


def test_fia_query_with_missing_index_is_503(installed_rag):
    response = client.post("/api/fia/query", json={"question": "What is the pit lane speed limit?"})
    assert response.status_code == 503 and "build_fia_index.py" in response.json()["detail"]


def test_fia_query_returns_validated_citations_mapped_to_passages(installed_rag):
    rag, llm = installed_rag
    rag.build_index()
    body = client.post("/api/fia/query", json={"question": "What is the speed limit in the pit lane?"}).json()
    assert body["grounded"] is True and body["status"] == "answered"
    by_label = {p["label"]: p for p in body["retrieved_passages"]}
    assert body["citations"] and all(label in by_label for label in body["citations"])
    cited = by_label[body["citations"][0]]
    assert cited["cited"] and "80km/h" in cited["text"]
    assert (cited["source"], cited["page"], cited["page_label"], cited["section"]) == ("section_b_sporting.pdf", 1, "B1", None)
    assert len(llm.calls) == 1
    # Citation spans are offsets into the returned answer, also where normalising would change its length.
    llm.reply = "In the ﬁrst session the pit lane limit is 80km/h ［Ｓ１］."
    body = client.post("/api/fia/query", json={"question": "What is the speed limit in the pit lane?"}).json()
    assert body["grounded"] and body["answer"] == llm.reply
    assert [(body["answer"][s["start"] : s["end"]], s["labels"]) for s in body["citation_spans"]] == [("［Ｓ１］", ["S1"])]


def test_fia_query_declines_fabricated_citation(installed_rag):
    rag, llm = installed_rag
    rag.build_index()
    llm.reply = "The limit is 80km/h according to Article B9.9 [S9]."
    body = client.post("/api/fia/query", json={"question": "What is the speed limit in the pit lane?"}).json()
    assert body["grounded"] is False and body["answer"] == DECLINE_ANSWER
    assert body["decline_reason"] == "invalid_citation" and body["validation"]["invalid_citations"] == ["S9"]
    assert body["citations"] == [] and body["citation_spans"] == []


def test_fia_endpoints_expose_definition_passages(tmp_path, monkeypatch, qdrant):
    llm = ScriptedChatModel("Speeding in the pit lane during a TTCS gives a drive through penalty [S1]; TTCS include the Race session [S2].")
    rag = make_definitions_rag(write_definitions_corpus(tmp_path / "fia_docs"), tmp_path, qdrant, llm=llm)
    rag.build_index()
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    body = client.post("/api/fia/query", json={"question": DEFINITIONS_QUESTION}).json()
    assert body["grounded"] and body["retrieved_passages"][1]["kind"] == "definition"
    assert body["retrieved_passages"][1]["defined_term"] == "Total Time Classified Session (TTCS)"
    assert body["retrieved_passages"][0]["kind"] == "regulation" and body["retrieved_passages"][0]["defined_term"] is None
    retrieved = client.post("/api/fia/retrieve", json={"question": DEFINITIONS_QUESTION}).json()
    assert retrieved["definitions"][0]["kind"] == "definition"


def test_fia_retrieve_is_independent_of_generation(installed_rag):
    rag, llm = installed_rag
    rag.build_index()
    body = client.post("/api/fia/retrieve", json={"question": "pit lane speed limit", "top_k": 2, "min_score": 0.0}).json()
    assert len(body["passages"]) == 2 and body["top_k"] == 2
    assert body["passages"][0]["score"] >= body["passages"][1]["score"]
    assert llm.calls == []


def test_regulatory_natural_query_uses_the_rag(installed_rag):
    rag, _ = installed_rag
    rag.build_index()
    body = client.post("/api/query/natural", json={"query": "What is the pit lane speed limit rule?"}).json()
    assert body["query_type"] == "regulatory"
    assert body["additional_context"]["grounded"] is True and body["data_sources"] == ["fia_regulations"]


def test_regulatory_natural_query_is_503_when_rag_unconfigured():
    response = client.post("/api/query/natural", json={"query": "Which FIA regulation applies to an unsafe release?"})
    assert response.status_code == 503 and "FIA RAG is unavailable" in response.json()["detail"]


# ------------------------------------------------------------------ heuristic modules


def test_strategy_endpoint_returns_ranked_complete_plans():
    body = client.post("/api/strategy/generate", json=STRATEGY_REQUEST).json()
    assert body["heuristic"] is True and body["tire_state"] == "supplied"
    remaining = body["remaining_laps"]
    assert remaining == 57 - 18 + 1
    assert body["strategies"]
    for plan in body["strategies"]:
        assert sum(stint["laps"] for stint in plan["stint_breakdown"]) == remaining
        assert plan["pit_stops"] == len(plan["pit_laps"]) == len(plan["tire_compounds"]) - 1
    assert [p["projected_race_time"] for p in body["strategies"]] == sorted(p["projected_race_time"] for p in body["strategies"])


def test_strategy_endpoint_works_on_the_final_lap():
    payload = {**STRATEGY_REQUEST, "race_state": {**STRATEGY_REQUEST["race_state"], "current_lap": 57}}
    body = client.post("/api/strategy/generate", json=payload).json()
    assert body["remaining_laps"] == 1
    assert [plan["pit_stops"] for plan in body["strategies"]] == [0]  # only the no-further-stop plan fits


def test_strategy_endpoint_rejects_nan_literals():
    raw = json.dumps(STRATEGY_REQUEST).replace('"lap_times": [95.6', '"lap_times": [NaN')
    assert "NaN" in raw
    response = client.post("/api/strategy/generate", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["input"] == "nan"


def test_validation_errors_do_not_echo_huge_inputs():
    response = client.post("/api/fia/query", json={"question": "q" * 5000})
    assert response.status_code == 422
    assert len(response.text) < 2000


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["telemetry"].update(lap_times=[]),
        lambda p: p["tire_data"]["soft"].update(pit_stop_delta=-10),
        lambda p: p["race_state"].update(current_lap=5.5),
        lambda p: p["race_state"].update(weather="sunny"),
        lambda p: p["tire_data"]["soft"].update(compound="hard"),
    ],
)
def test_strategy_endpoint_rejects_invalid_input(mutate):
    payload = copy.deepcopy(STRATEGY_REQUEST)
    mutate(payload)
    assert client.post("/api/strategy/generate", json=payload).status_code == 422


def _calibration_example():
    from core_modules.strategy_optimizer.schemas import TyreCalibrationRequest

    return copy.deepcopy(TyreCalibrationRequest.model_config["json_schema_extra"]["examples"][0])


def test_tyre_calibration_endpoint_recovers_the_generating_parameters():
    # The documented example laps: soft peak 80.0 s until tyre age 5, medium 80.6 s until age 8.
    response = client.post("/api/strategy/calibrate-tyres", json=_calibration_example())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reference_compound"] == "soft" and body["estimated_base_lap_time_s"] == pytest.approx(80.0, abs=0.01)
    assert {c: v["status"] for c, v in body["compounds"].items()} == {"soft": "estimated", "medium": "estimated"}
    assert body["tire_data"]["soft"]["peak_performance_window"] == [1, 5]
    assert body["tire_data"]["medium"]["peak_performance_window"] == [1, 8]
    assert body["tire_data"]["medium"]["base_performance"] == pytest.approx(80.0 / 80.6, abs=1e-3)
    # Out-laps and in-laps are excluded, not fitted.
    assert {lap["reason"] for lap in body["excluded_laps"]} >= {"pit_out", "pit_in"}


def test_calibrated_tyre_data_is_accepted_by_the_strategy_endpoint():
    calibrated = client.post("/api/strategy/calibrate-tyres", json=_calibration_example()).json()["tire_data"]
    payload = copy.deepcopy(STRATEGY_REQUEST)
    payload["tire_data"].update(calibrated)
    response = client.post("/api/strategy/generate", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["strategies"]


def test_tyre_calibration_reports_insufficient_data_instead_of_guessing():
    request = _calibration_example()
    request["laps"] = [lap for lap in request["laps"] if lap["compound"] == "soft"][:5]  # 1 out-lap + 4 clean laps
    body = client.post("/api/strategy/calibrate-tyres", json=request).json()
    assert body["compounds"]["soft"]["status"] == "insufficient_data" and body["compounds"]["soft"]["reason"]
    assert body["tire_data"] == {} and body["reference_compound"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(laps=[]),
        lambda p: p.update(warm_up_laps={"hard": 2}),  # no hard laps supplied
        lambda p: p.update(fuel_correction_s_per_lap=0.05, laps=[{k: v for k, v in lap.items() if k != "race_lap"} for lap in p["laps"]]),
        lambda p: p["laps"][3].update(lap_time=5.0),
        lambda p: p.update(pit_stop_delta=-1.0),
        lambda p: p.update(unexpected=True),
    ],
)
def test_tyre_calibration_rejects_invalid_input(mutate):
    payload = _calibration_example()
    mutate(payload)
    assert client.post("/api/strategy/calibrate-tyres", json=payload).status_code == 422


def test_setup_endpoint_runs_a_reproducible_search():
    first = client.post("/api/setup/recommend", json=SETUP_REQUEST)
    second = client.post("/api/setup/recommend", json=SETUP_REQUEST)
    assert first.status_code == 200, first.text
    assert first.json() == second.json()
    bad = {**SETUP_REQUEST, "driver_preferences": {"preferred_ride_height": 5.0}}
    assert client.post("/api/setup/recommend", json=bad).status_code == 422


def test_ghost_endpoint_aligns_laps_and_serves_the_artifact():
    request = {"lap1_telemetry": _lap(200.0), "lap2_telemetry": _lap(190.0, seconds=63.2, lap_number=2), "track_section": "silverstone"}
    response = client.post("/api/ghost/generate", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["missing_channels"]["lap1"] == ["x", "y", "throttle", "steering", "gear"]
    assert body["throttle"] is None and body["sector_time_deltas_s"] is None
    assert body["summary"]["final_delta_s"] > 0  # the slower lap loses time
    artifact = client.get(body["visualization_url"])
    assert artifact.status_code == 200 and artifact.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    "request_body",
    [
        {"lap1_telemetry": {"timestamps": [0, 1, 2]}, "lap2_telemetry": _lap(200.0)},
        {"lap1_telemetry": _lap(200.0), "lap2_telemetry": _lap(200.0), "track_section": "../../etc/x"},
        {"lap1_telemetry": {**_lap(200.0), "timestamps": [0.0] * 121}, "lap2_telemetry": _lap(200.0)},
    ],
)
def test_ghost_endpoint_rejects_invalid_telemetry(request_body):
    assert client.post("/api/ghost/generate", json=request_body).status_code == 422


def test_emotion_endpoint_analyses_real_audio():
    response = client.post("/api/emotion/classify", json={"audio_file": _speech_like_base64(), "transcribe": False})
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.9 < body["duration"] < 1.1
    assert abs(body["audio_features"]["mean_pitch"] - 160.0) < 16.0
    assert body["transcription"] is None and body["transcription_status"] == "not_requested"


def test_emotion_endpoint_never_opens_server_paths():
    existing = client.post("/api/emotion/classify", json={"audio_file": "/etc/hostname"})
    missing = client.post("/api/emotion/classify", json={"audio_file": "/etc/does-not-exist-anywhere"})
    assert existing.status_code == missing.status_code == 422
    assert existing.json()["detail"] == missing.json()["detail"]


@pytest.mark.parametrize("audio", [base64.b64encode(b"hello world").decode(), "data:text/plain;base64,aGVsbG8=", "not base64 !!"])
def test_emotion_endpoint_rejects_non_audio(audio):
    assert client.post("/api/emotion/classify", json={"audio_file": audio}).status_code == 422


def test_penalty_endpoint_is_explicit_heuristic_triage():
    body = client.post(
        "/api/penalty/predict", json={"incident_type": "unsafe_release", "track_condition": "dry", "intent": "accidental"}
    ).json()
    assert body["method"] == "transparent heuristic triage" and body["referenced_rule"] is None
    assert "triage_category" in body and "predicted_penalty" not in body
    assert "not an FIA steward-decision predictor" in body["disclaimer"]
    invalid = {"incident_type": "unsafe_release", "track_condition": "sunny", "intent": "accidental"}
    assert client.post("/api/penalty/predict", json=invalid).status_code == 422


def test_natural_query_performance_uses_only_supplied_telemetry():
    body = client.post(
        "/api/query/natural",
        json={"query": "Why was my lap time slower?", "context": {"telemetry": {"lap_times": [80.0, 80.4], "braking_consistency": 0.8}}},
    ).json()
    assert body["query_type"] == "performance" and body["data_sources"] == ["telemetry"]
    assert "80.400" in body["answer"]
    assert client.post("/api/query/natural", json={"query": "  "}).status_code == 422


def test_natural_strategy_and_setup_routes_match_the_dedicated_endpoints():
    body = client.post("/api/query/natural", json={"query": "What pit stop strategy should I use?", "context": STRATEGY_REQUEST}).json()
    direct = client.post("/api/strategy/generate", json=STRATEGY_REQUEST).json()
    assert body["query_type"] == "strategy" and body["confidence"] is None
    assert body["additional_context"]["best_strategy_id"] == direct["best_strategy_id"]
    best = direct["strategies"][0]
    laps = [str(lap) for lap in best["pit_laps"]]
    assert f"pit laps {', '.join(laps[:-1])} and {laps[-1]}" in body["answer"]  # plain wording, not a Python list
    assert "[" not in body["answer"] and "stop(s)" not in body["answer"] and " → ".join(best["tire_compounds"]) in body["answer"]

    setup_context = {key: SETUP_REQUEST[key] for key in ("driver_preferences", "track_profile", "weather", "n_trials", "seed")}
    body = client.post("/api/query/natural", json={"query": "What setup should I run?", "context": setup_context}).json()
    assert body["query_type"] == "technical"
    assert body["additional_context"]["ride_height"] == client.post("/api/setup/recommend", json=SETUP_REQUEST).json()["ride_height"]

    bad = copy.deepcopy(STRATEGY_REQUEST)
    bad["tire_data"]["soft"].pop("peak_performance_window")
    assert client.post("/api/strategy/generate", json=bad).status_code == 422
    assert client.post("/api/query/natural", json={"query": "What pit stop strategy should I use?", "context": bad}).status_code == 422


def test_an_invalid_natural_query_context_gets_the_structured_422_of_its_endpoint():
    bad = copy.deepcopy(STRATEGY_REQUEST)
    bad["car_status"]["engine_wear"] = 3
    bad["race_state"]["current_lap"] = 99
    direct = client.post("/api/strategy/generate", json=bad).json()["detail"]
    response = client.post("/api/query/natural", json={"query": "What pit stop strategy should I use?", "context": bad})
    assert response.status_code == 422 and "errors.pydantic.dev" not in response.text
    detail = response.json()["detail"]
    assert [error["loc"] for error in detail] == [["body", "context", *error["loc"][1:]] for error in direct]
    assert [(error["msg"], error["input"]) for error in detail] == [(error["msg"], error["input"]) for error in direct]
    assert detail[1]["input"] == "<object with 6 keys>"  # summarised like every other 422, not echoed

    setup = {key: SETUP_REQUEST[key] for key in ("driver_preferences", "track_profile", "weather")}
    setup["driver_preferences"] = {**setup["driver_preferences"], "risk_tolerance": 7}
    response = client.post("/api/query/natural", json={"query": "What setup should I run?", "context": setup})
    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [["body", "context", "driver_preferences", "risk_tolerance"]]


def test_a_transcription_failure_is_a_503_on_both_audio_routes(monkeypatch):
    def failing(*args, **kwargs):
        raise TranscriptionError("Loading the Whisper model 'tiny' failed (URLError)")

    monkeypatch.setattr("app.main.classify_emotion_detailed", failing)
    monkeypatch.setattr("app.main.process_natural_query", failing)
    for route, payload in (
        ("/api/emotion/classify", {"audio_file": "UklGRg==", "transcribe": True}),
        ("/api/query/natural", {"query": "How does the driver sound on the radio?"}),
    ):
        response = client.post(route, json=payload)
        assert response.status_code == 503, route
        assert response.json() == {"detail": "Transcription failed: Loading the Whisper model 'tiny' failed (URLError)"}


def test_natural_query_without_required_context_declines_explicitly():
    body = client.post("/api/query/natural", json={"query": "What pit stop strategy should I use?"}).json()
    assert body["query_type"] == "strategy" and body["confidence"] == 0.0 and body["data_sources"] == []


# ------------------------------------------------------------------ hardening regressions


def test_chunked_upload_over_the_limit_is_rejected_while_streaming():
    def body():
        yield b'{"incident_type": "collision", "pad": "'
        for _ in range(3):
            yield b"x" * (512 * 1024)
        yield b'"}'

    response = client.post("/api/penalty/predict", content=body(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_body_limits_are_per_route():
    big = {"incident_type": "collision", "track_condition": "dry", "intent": "accidental", "pad": "x" * (2 * 1024 * 1024)}
    assert client.post("/api/penalty/predict", json=big).status_code == 413
    # the emotion route accepts large base64 audio (here: invalid, so 422 - but not 413)
    assert client.post("/api/emotion/classify", json={"audio_file": "A" * (2 * 1024 * 1024)}).status_code == 422


def test_413_carries_cors_headers():
    response = client.post(
        "/api/penalty/predict",
        content=b"x" * (2 * 1024 * 1024),
        headers={"content-type": "application/json", "Origin": "https://example.org"},
    )
    assert response.status_code == 413 and response.headers.get("access-control-allow-origin") == "*"


@pytest.mark.parametrize("raw", [b'{"question": "pit lane speed \\ud83d"}', b'{"question": "x", "top_k": true}'])
def test_hostile_json_is_a_422_not_a_500(raw):
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/fia/retrieve", content=raw, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422, response.text


def test_deeply_nested_json_is_a_client_error_not_a_500():
    raw = b'{"question": ' + b"[" * 1000 + b"]" * 1000 + b"}"
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/fia/retrieve", content=raw, headers={"content-type": "application/json"}
    )
    # Python 3.11's JSON decoder gives up on this depth inside FastAPI's body parser (400);
    # 3.12 parses it and validation rejects it (422). Either way it must not be a server error.
    assert response.status_code in (400, 422), response.text


def test_validation_errors_are_bounded_for_huge_invalid_objects():
    payload = {f"k{i}": i for i in range(20000)}
    response = client.post("/api/penalty/predict", content=json.dumps(payload), headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert len(response.content) < 10_000 and response.json()["more_errors"] > 0


def test_only_ghost_images_are_served_under_artifacts():
    (main.ARTIFACTS_DIR / "fia_rag_eval.json").parent.mkdir(parents=True, exist_ok=True)
    (main.ARTIFACTS_DIR / "fia_rag_eval.json").write_text("{}")
    assert client.get("/artifacts/fia_rag_eval.json").status_code == 404


@pytest.mark.parametrize(
    "value,expected",
    [("", ["*"]), ("*", ["*"]), ("https://a.example, https://b.example/", ["https://a.example", "https://b.example"])],
)
def test_cors_origins_are_normalised(monkeypatch, value, expected):
    monkeypatch.setenv("CORS_ORIGINS", value)
    assert main._cors_origins() == expected


@pytest.mark.parametrize("value", ["*,https://a.example", "a.example", "ftp://a.example"])
def test_invalid_cors_origins_fail_startup(monkeypatch, value):
    monkeypatch.setenv("CORS_ORIGINS", value)
    with pytest.raises(RuntimeError):
        main._cors_origins()


def test_health_reports_a_failing_provider(installed_rag):
    rag, llm = installed_rag
    rag.build_index()

    def broken(messages):
        raise TimeoutError("provider down")

    llm.reply = broken
    assert client.post("/api/fia/query", json={"question": "What is the speed limit in the pit lane?"}).status_code == 503
    body = client.get("/health").json()
    assert body["modules"]["fia_rag"] == "provider_failing" and body["status"] == "degraded"


def test_ghost_artifact_write_failure_is_a_503(monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("read-only artifacts directory")

    monkeypatch.setattr(main, "generate_ghost_comparison", fail)
    response = client.post("/api/ghost/generate", json={"lap1_telemetry": _lap(200.0), "lap2_telemetry": _lap(199.0)})
    assert response.status_code == 503 and "not writable" in response.json()["detail"]
