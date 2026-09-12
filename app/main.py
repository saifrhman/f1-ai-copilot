#!/usr/bin/env python3
"""FastAPI entry point for F1 AI Copilot."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core_modules.driver_emotion.emotion_classifier import classify_emotion_detailed
from core_modules.ghost_car.ghost_car_visualizer import generate_ghost_comparison
from core_modules.llm_query.natural_query import process_natural_query
from core_modules.rule_checker.fia_rag_agent import get_fia_knowledge_base, query_fia_regulations_detailed
from core_modules.rule_checker.penalty_predictor import predict_penalty
from core_modules.setup_optimizer.setup_recommender import recommend_setup
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


app = FastAPI(
    title="F1 AI Copilot",
    description="Formula 1 analysis demo with strategy, FIA RAG, setup, telemetry and radio-analysis modules",
    version="1.1.0",
)

cors_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [value.strip() for value in cors_raw.split(",") if value.strip()]
allow_credentials = cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("outputs").mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory="outputs", check_dir=False), name="artifacts")


class StrategyRequest(BaseModel):
    telemetry: Dict[str, Any]
    car_status: Dict[str, Any]
    driver_profile: Dict[str, Any]
    tire_data: Dict[str, Any]
    race_state: Dict[str, Any]
    competition: List[Dict[str, Any]]


class FIAQueryRequest(BaseModel):
    question: str


class PenaltyRequest(BaseModel):
    incident_type: str
    track_condition: str
    intent: str
    driver_history: Optional[Dict[str, Any]] = None


class NaturalQueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None


class EmotionRequest(BaseModel):
    audio_file: str
    transcribe: bool = False


class GhostCarRequest(BaseModel):
    lap1_telemetry: Dict[str, Any]
    lap2_telemetry: Dict[str, Any]
    track_section: str = "monaco"


class SetupRequest(BaseModel):
    driver_preferences: Dict[str, Any]
    track_profile: Dict[str, Any]
    weather: Dict[str, Any]


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "F1 AI Copilot API",
        "version": app.version,
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    fia_status = get_fia_knowledge_base().status()
    fia_ready = bool(fia_status.get("initialized"))
    return {
        "status": "healthy" if fia_ready else "degraded",
        "version": app.version,
        "modules": {
            "strategy": "ready",
            "fia_rag": "ready" if fia_ready else "not_configured",
            "penalty": "ready",
            "natural_query": "ready",
            "emotion": "ready",
            "ghost": "ready",
            "setup": "ready",
        },
        "fia_rag": fia_status,
    }


@app.post("/api/strategy/generate")
async def generate_race_strategy(request: StrategyRequest) -> Dict[str, Any]:
    try:
        race_data = dict(request.race_state)
        if isinstance(race_data.get("weather"), str):
            race_data["weather"] = WeatherCondition(race_data["weather"])

        tyre_data: Dict[TireCompound, TireData] = {}
        for key, raw in request.tire_data.items():
            compound = TireCompound(key)
            item = dict(raw)
            raw_compound = item.get("compound", key)
            item["compound"] = TireCompound(raw_compound) if isinstance(raw_compound, str) else raw_compound
            if "peak_performance_window" in item:
                item["peak_performance_window"] = tuple(item["peak_performance_window"])
            tyre_data[compound] = TireData(**item)

        competitors: List[Competitor] = []
        for raw in request.competition:
            item = dict(raw)
            if isinstance(item.get("tire_compound"), str):
                item["tire_compound"] = TireCompound(item["tire_compound"])
            competitors.append(Competitor(**item))

        strategies = generate_strategy(
            telemetry=request.telemetry,
            car_status=CarStatus(**request.car_status),
            driver_profile=DriverProfile(**request.driver_profile),
            tire_data=tyre_data,
            race_state=RaceState(**race_data),
            competition=competitors,
        )
        return {
            "strategies": [
                {
                    "strategy_id": strategy.strategy_id,
                    "projected_race_time": strategy.projected_race_time,
                    "confidence_score": strategy.confidence_score,
                    "risk_level": strategy.risk_level,
                    "pit_stops": len(strategy.pit_laps),
                    "tire_compounds": [compound.value for compound in strategy.tire_compounds],
                    "pit_laps": strategy.pit_laps,
                    "stint_breakdown": [
                        {
                            **{k: v for k, v in stint.items() if k != "tire_compound"},
                            "tire_compound": stint["tire_compound"].value,
                        }
                        for stint in strategy.stint_breakdown
                    ],
                    "undercut_opportunities": strategy.undercut_opportunities,
                    "overcut_opportunities": strategy.overcut_opportunities,
                    "notes": strategy.notes,
                }
                for strategy in strategies
            ]
        }
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid strategy input: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Strategy generation failed: {exc}") from exc


@app.get("/api/fia/status")
async def fia_rag_status() -> Dict[str, Any]:
    return get_fia_knowledge_base().status()


@app.post("/api/fia/query")
async def query_fia_rules(request: FIAQueryRequest) -> Dict[str, Any]:
    try:
        return query_fia_regulations_detailed(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"FIA query failed: {exc}") from exc


@app.post("/api/penalty/predict")
async def predict_incident_penalty(request: PenaltyRequest) -> Dict[str, Any]:
    try:
        return predict_penalty(
            incident_type=request.incident_type,
            track_condition=request.track_condition,
            intent=request.intent,
            driver_history=request.driver_history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Penalty prediction failed: {exc}") from exc


@app.post("/api/query/natural")
async def process_query(request: NaturalQueryRequest) -> Dict[str, Any]:
    try:
        return process_natural_query(request.query, request.context)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Natural query processing failed: {exc}") from exc


@app.post("/api/emotion/classify")
async def classify_driver_emotion(request: EmotionRequest) -> Dict[str, Any]:
    try:
        return classify_emotion_detailed(request.audio_file, transcribe=request.transcribe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Emotion classification failed: {exc}") from exc


@app.post("/api/ghost/generate")
async def generate_ghost_car(request: GhostCarRequest) -> Dict[str, Any]:
    try:
        return generate_ghost_comparison(
            lap1_telemetry=request.lap1_telemetry,
            lap2_telemetry=request.lap2_telemetry,
            track_section=request.track_section,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ghost car generation failed: {exc}") from exc


@app.post("/api/setup/recommend")
async def recommend_car_setup(request: SetupRequest) -> Dict[str, Any]:
    try:
        return recommend_setup(
            driver_preferences=request.driver_preferences,
            track_profile=request.track_profile,
            weather=request.weather,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid setup input: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Setup recommendation failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
