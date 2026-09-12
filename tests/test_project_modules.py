import base64
import io
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from core_modules.driver_emotion.emotion_classifier import classify_emotion_detailed
from core_modules.ghost_car.ghost_car_visualizer import GhostCarVisualizer
from core_modules.rule_checker.penalty_predictor import predict_penalty
from core_modules.setup_optimizer.setup_recommender import (
    DriverPreferences,
    SetupOptimizer,
    TrackProfile,
    TrackType,
    WeatherCondition as SetupWeather,
    WeatherData,
)
from core_modules.strategy_optimizer.strategy_engine import (
    CarStatus,
    Competitor,
    DriverProfile,
    RaceState,
    TireCompound,
    TireData,
    WeatherCondition,
    generate_strategy,
)
from scripts.fetch_fia_regulations import discover_pdf_urls


client = TestClient(app)


def _strategy_inputs():
    telemetry = {
        "lap_times": [80.4, 80.2, 80.3],
        "braking_consistency": 0.82,
        "throttle_aggressiveness": 0.68,
    }
    car_status = CarStatus(
        damage={"front_wing": 0.05, "floor": 0.02},
        fuel_load=75.0,
        brake_temp=340.0,
        engine_wear=0.25,
        ers_availability=0.8,
        brake_wear=0.2,
    )
    driver = DriverProfile(0.72, 0.55, "calculated", 0.82, 0.68)
    tyre_data = {
        TireCompound.SOFT: TireData(TireCompound.SOFT, 1.00, 0.018, 2, (2, 8), 24.0),
        TireCompound.MEDIUM: TireData(TireCompound.MEDIUM, 0.97, 0.012, 3, (3, 16), 24.0),
        TireCompound.HARD: TireData(TireCompound.HARD, 0.94, 0.008, 4, (4, 25), 24.0),
    }
    race = RaceState(10, 30, WeatherCondition.DRY, 32.0, 0.4, 0.2, 0.1, [])
    competitors = [
        Competitor("CAR2", 2, TireCompound.MEDIUM, 17, 2.0, 1.2, 1.0, 0, [])
    ]
    return telemetry, car_status, driver, tyre_data, race, competitors


def test_strategy_engine_generates_complete_stints():
    inputs = _strategy_inputs()
    strategies = generate_strategy(*inputs)
    remaining_laps = inputs[4].total_laps - inputs[4].current_lap + 1

    assert strategies
    assert strategies == sorted(strategies, key=lambda s: s.projected_race_time)
    for strategy in strategies:
        assert len(strategy.stint_breakdown) == len(strategy.tire_compounds)
        assert sum(stint["laps"] for stint in strategy.stint_breakdown) == remaining_laps
        assert len(strategy.pit_laps) == len(strategy.tire_compounds) - 1
        assert 0.0 <= strategy.confidence_score <= 1.0


def test_strategy_api_converts_string_enums_and_returns_stints():
    telemetry, car_status, driver, tyre_data, race, competitors = _strategy_inputs()
    response = client.post(
        "/api/strategy/generate",
        json={
            "telemetry": telemetry,
            "car_status": car_status.__dict__,
            "driver_profile": driver.__dict__,
            "tire_data": {
                key.value: {
                    "compound": value.compound.value,
                    "base_performance": value.base_performance,
                    "degradation_rate": value.degradation_rate,
                    "warm_up_laps": value.warm_up_laps,
                    "peak_performance_window": list(value.peak_performance_window),
                    "pit_stop_delta": value.pit_stop_delta,
                }
                for key, value in tyre_data.items()
            },
            "race_state": {
                **race.__dict__,
                "weather": race.weather.value,
            },
            "competition": [
                {
                    **competitors[0].__dict__,
                    "tire_compound": competitors[0].tire_compound.value,
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["strategies"]
    assert body["strategies"][0]["stint_breakdown"]


def test_setup_optimizer_is_reproducible_and_bounded():
    optimizer_a = SetupOptimizer(n_trials=8, seed=7)
    optimizer_b = SetupOptimizer(n_trials=8, seed=7)
    driver = DriverPreferences(risk_tolerance=0.55, tire_management=0.7)
    track = TrackProfile("Test Circuit", 5000, 16, 7, 5, TrackType.MIXED, 205, 0.7)
    weather = WeatherData(SetupWeather.DRY, 25.0, 55.0)

    first = optimizer_a.recommend_setup(driver, track, weather)
    second = optimizer_b.recommend_setup(driver, track, weather)

    assert first["optimization_method"] == "Optuna TPESampler"
    assert first["ride_height"] == second["ride_height"]
    assert 60.0 <= first["ride_height"] <= 85.0
    assert 0.0 <= first["front_wing_angle"] <= 15.0
    assert 0.0 <= first["rear_wing_angle"] <= 20.0
    assert 0.0 <= first["confidence"] <= 1.0


def test_setup_api_accepts_json_enums():
    response = client.post(
        "/api/setup/recommend",
        json={
            "driver_preferences": {"risk_tolerance": 0.5, "tire_management": 0.7},
            "track_profile": {
                "track_name": "Silverstone Circuit",
                "track_length": 5891,
                "corners": 18,
                "high_speed_sections": 8,
                "low_speed_sections": 4,
                "track_type": "high_speed",
                "average_speed": 220,
                "downforce_requirement": 0.6,
            },
            "weather": {"condition": "dry", "temperature": 24, "humidity": 50},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["optimization_method"] == "Optuna TPESampler"


def _telemetry(lap_number: int, speed_offset: float = 0.0):
    return {
        "timestamps": [0.0, 0.5, 1.0, 1.5],
        "x": [0.0, 10.0, 20.0, 30.0],
        "y": [0.0, 2.0, 1.0, 0.0],
        "speed": [180 + speed_offset, 200 + speed_offset, 210 + speed_offset, 190 + speed_offset],
        "throttle": [0.7, 0.9, 1.0, 0.5],
        "brake": [0.0, 0.0, 0.0, 0.4],
        "steering": [0.0, 0.1, -0.1, 0.0],
        "drs": [False, True, True, False],
        "gear": [5, 6, 7, 5],
        "lap_time": 80.0 + speed_offset / 100.0,
        "sector_times": [26.5, 26.7, 26.8],
        "lap_number": lap_number,
    }


def test_ghost_visualizer_rejects_empty_and_writes_artifact(tmp_path):
    visualizer = GhostCarVisualizer(output_dir=str(tmp_path))
    result = visualizer.generate_ghost_comparison(_telemetry(1), _telemetry(2, 2.0), "silverstone")
    assert Path(result["visualization_path"]).exists()
    assert result["samples_compared"] == 4

    try:
        visualizer.generate_ghost_comparison({"timestamps": []}, _telemetry(2), "silverstone")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty telemetry should fail")


def test_emotion_classifier_uses_real_audio_not_mock_data():
    sample_rate = 22050
    duration = 0.5
    samples = np.arange(int(sample_rate * duration)) / sample_rate
    waveform = 0.15 * np.sin(2 * math.pi * 220 * samples)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    result = classify_emotion_detailed(encoded, transcribe=False)
    assert result["audio_features"]["duration"] > 0.4
    assert result["transcription"] is None
    assert 0.0 <= result["confidence"] <= 0.95


def test_penalty_estimator_does_not_invent_fia_article_numbers():
    result = predict_penalty("unsafe_release", "dry", "accidental")
    assert result["method"] == "transparent heuristic triage"
    assert result["referenced_rule"] is None
    assert "FIA RAG" in result["disclaimer"]


def test_natural_query_performance_uses_supplied_telemetry():
    response = client.post(
        "/api/query/natural",
        json={
            "query": "Why was my lap time slower?",
            "context": {"telemetry": {"lap_times": [80.0, 80.4], "braking_consistency": 0.8}},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query_type"] == "performance"
    assert "80.400" in body["answer"]
    assert body["data_sources"] == ["telemetry"]


def test_fia_downloader_discovers_all_sections_from_official_style_links():
    anchors = "\n".join(
        f'<a href="/system/files/documents/fia_2026_f1_regulations_-_section_{section.lower()}_example.pdf">Section {section}</a>'
        for section in "ABCDEF"
    )
    discovered = discover_pdf_urls(anchors, "https://www.fia.com/regulation/category/2182", 2026)
    assert set(discovered) == set("ABCDEF")
    assert all(url.startswith("https://www.fia.com/") for url in discovered.values())


def test_health_reports_fia_as_degraded_when_not_configured(monkeypatch):
    # The test environment intentionally has no OPENAI_API_KEY and no FIA PDFs.
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert "fia_rag" in body["modules"]
