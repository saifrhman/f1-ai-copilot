#!/usr/bin/env python3
"""
F1 AI Copilot - Main Application
FastAPI backend for Formula 1 race engineering and strategy optimization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn

from core_modules.strategy_optimizer.strategy_engine import (
    generate_strategy, DriverProfile, CarStatus, TireData, RaceState, Competitor
)
from core_modules.rule_checker.fia_rag_agent import query_fia_regulations
from core_modules.rule_checker.penalty_predictor import predict_penalty
from core_modules.llm_query.natural_query import process_natural_query
from core_modules.driver_emotion.emotion_classifier import classify_emotion
from core_modules.ghost_car.ghost_car_visualizer import generate_ghost_comparison
from core_modules.setup_optimizer.setup_recommender import recommend_setup

app = FastAPI(
    title="F1 AI Copilot",
    description="AI-powered Formula 1 race engineering and strategy optimization system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
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
    audio_file: str  # Base64 encoded audio or file path

class GhostCarRequest(BaseModel):
    lap1_telemetry: Dict[str, Any]
    lap2_telemetry: Dict[str, Any]
    track_section: str

class SetupRequest(BaseModel):
    driver_preferences: Dict[str, Any]
    track_profile: Dict[str, Any]
    weather: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "F1 AI Copilot API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "modules": ["strategy", "fia", "emotion", "ghost", "setup"]}

@app.post("/api/strategy/generate")
async def generate_race_strategy(request: StrategyRequest):
    """
    Generate optimal race strategy based on current conditions
    """
    try:
        # Convert dict inputs to proper objects
        driver_profile = DriverProfile(**request.driver_profile)
        car_status = CarStatus(**request.car_status)
        race_state = RaceState(**request.race_state)
        
        # Convert tire data
        tire_data = {}
        for compound, data in request.tire_data.items():
            tire_data[compound] = TireData(**data)
        
        # Convert competition data
        competition = [Competitor(**comp) for comp in request.competition]
        
        strategies = generate_strategy(
            telemetry=request.telemetry,
            car_status=car_status,
            driver_profile=driver_profile,
            tire_data=tire_data,
            race_state=race_state,
            competition=competition
        )
        
        return {
            "strategies": [
                {
                    "strategy_id": s.strategy_id,
                    "projected_race_time": s.projected_race_time,
                    "confidence_score": s.confidence_score,
                    "risk_level": s.risk_level,
                    "pit_stops": len(s.pit_laps),
                    "tire_compounds": [c.value for c in s.tire_compounds],
                    "pit_laps": s.pit_laps,
                    "undercut_opportunities": s.undercut_opportunities,
                    "overcut_opportunities": s.overcut_opportunities,
                    "notes": s.notes
                }
                for s in strategies
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy generation failed: {str(e)}")

@app.post("/api/fia/query")
async def query_fia_rules(request: FIAQueryRequest):
    """
    Query FIA regulations using RAG system
    """
    try:
        answer = query_fia_regulations(request.question)
        return {"answer": answer, "source": "fia_rag_agent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FIA query failed: {str(e)}")

@app.post("/api/penalty/predict")
async def predict_incident_penalty(request: PenaltyRequest):
    """
    Predict penalty for an incident based on FIA rules and precedent
    """
    try:
        penalty = predict_penalty(
            incident_type=request.incident_type,
            track_condition=request.track_condition,
            intent=request.intent,
            driver_history=request.driver_history
        )
        return penalty
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Penalty prediction failed: {str(e)}")

@app.post("/api/query/natural")
async def process_query(request: NaturalQueryRequest):
    """
    Process natural language queries about race performance or regulations
    """
    try:
        result = process_natural_query(request.query, request.context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Natural query processing failed: {str(e)}")

@app.post("/api/emotion/classify")
async def classify_driver_emotion(request: EmotionRequest):
    """
    Classify driver emotion from radio communication
    """
    try:
        emotion = classify_emotion(request.audio_file)
        return {"emotion": emotion, "confidence": 0.85}  # Placeholder confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emotion classification failed: {str(e)}")

@app.post("/api/ghost/generate")
async def generate_ghost_car(request: GhostCarRequest):
    """
    Generate ghost car visualization for lap comparison
    """
    try:
        comparison = generate_ghost_comparison(
            lap1_telemetry=request.lap1_telemetry,
            lap2_telemetry=request.lap2_telemetry,
            track_section=request.track_section
        )
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ghost car generation failed: {str(e)}")

@app.post("/api/setup/recommend")
async def recommend_car_setup(request: SetupRequest):
    """
    Recommend optimal car setup based on conditions
    """
    try:
        setup = recommend_setup(
            driver_preferences=request.driver_preferences,
            track_profile=request.track_profile,
            weather=request.weather
        )
        return setup
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup recommendation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 