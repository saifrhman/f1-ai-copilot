#!/usr/bin/env python3
"""
F1 AI Copilot - API Schemas
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class WeatherCondition(str, Enum):
    DRY = "dry"
    WET = "wet"
    INTERMEDIATE = "intermediate"


class TireCompound(str, Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"


class IncidentType(str, Enum):
    TRACK_LIMITS = "track_limits"
    UNSAFE_RELEASE = "unsafe_release"
    COLLISION = "collision"
    BLOCKING = "blocking"
    DANGEROUS_DRIVING = "dangerous_driving"
    TECHNICAL_INFRINGEMENT = "technical_infringement"


class EmotionType(str, Enum):
    CALM = "calm"
    ANGRY = "angry"
    PANICKED = "panicked"
    FOCUSED = "focused"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"


# Strategy API Schemas
class TelemetryData(BaseModel):
    braking_zones: List[int] = Field(..., description="Braking zone numbers")
    throttle_trace: List[float] = Field(..., description="Throttle application percentages")
    lap_times: List[float] = Field(..., description="Recent lap times in seconds")
    braking_consistency: Optional[float] = Field(0.8, description="Driver braking consistency (0-1)")
    throttle_aggressiveness: Optional[float] = Field(0.5, description="Throttle aggressiveness (0-1)")


class CarStatusData(BaseModel):
    damage: Dict[str, float] = Field(..., description="Component damage levels")
    fuel_load: float = Field(..., description="Current fuel load in kg")
    brake_temp: float = Field(..., description="Brake temperature")
    engine_wear: float = Field(..., description="Engine wear percentage (0-1)")
    ers_availability: float = Field(..., description="ERS energy available (0-1)")
    brake_wear: float = Field(..., description="Brake wear percentage (0-1)")


class DriverProfileData(BaseModel):
    tire_management: float = Field(..., description="Tire saving ability (0-1)")
    risk_tolerance: float = Field(..., description="Risk tolerance level (0-1)")
    overtaking_style: str = Field(..., description="Overtaking style: aggressive/conservative/calculated")
    braking_consistency: float = Field(..., description="Consistency in braking zones (0-1)")
    throttle_aggressiveness: float = Field(..., description="Throttle application aggressiveness (0-1)")


class TireDataModel(BaseModel):
    compound: TireCompound
    base_performance: float = Field(..., description="Base performance multiplier")
    degradation_rate: float = Field(..., description="Performance loss per lap")
    warm_up_laps: int = Field(..., description="Laps needed for optimal performance")
    peak_performance_window: tuple = Field(..., description="(start_lap, end_lap) for peak performance")
    pit_stop_delta: float = Field(..., description="Time lost in pit stop")


class RaceStateData(BaseModel):
    current_lap: int = Field(..., description="Current lap number")
    total_laps: int = Field(..., description="Total race laps")
    weather: WeatherCondition = Field(..., description="Weather condition")
    track_temperature: float = Field(..., description="Track temperature in Celsius")
    track_evolution: float = Field(..., description="Track evolution factor (0-1)")
    safety_car_probability: float = Field(..., description="Probability of safety car (0-1)")
    yellow_flag_risk: float = Field(..., description="Risk of yellow flags (0-1)")
    weather_forecast: List[Dict[str, Any]] = Field(..., description="Future weather predictions")


class CompetitorData(BaseModel):
    driver_id: str = Field(..., description="Competitor driver ID")
    current_position: int = Field(..., description="Current race position")
    tire_compound: TireCompound = Field(..., description="Current tire compound")
    tire_age: int = Field(..., description="Laps on current tires")
    gap_to_leader: float = Field(..., description="Gap to race leader in seconds")
    gap_ahead: float = Field(..., description="Gap to car ahead in seconds")
    gap_behind: float = Field(..., description="Gap to car behind in seconds")
    pit_stops_completed: int = Field(..., description="Number of pit stops completed")
    estimated_strategy: List[Dict[str, Any]] = Field(default=[], description="Estimated competitor strategy")


class StrategyRequest(BaseModel):
    telemetry: TelemetryData
    car_status: CarStatusData
    driver_profile: DriverProfileData
    tire_data: Dict[str, TireDataModel]
    race_state: RaceStateData
    competition: List[CompetitorData]


class StrategyOption(BaseModel):
    strategy_id: str
    projected_race_time: float
    confidence_score: float
    risk_level: str
    pit_stops: int
    tire_compounds: List[str]
    pit_laps: List[int]
    undercut_opportunities: List[Dict[str, Any]]
    overcut_opportunities: List[Dict[str, Any]]
    notes: List[str]


class StrategyResponse(BaseModel):
    strategies: List[StrategyOption]


# FIA RAG API Schemas
class FIAQueryRequest(BaseModel):
    question: str = Field(..., description="Question about FIA regulations")


class FIAQueryResponse(BaseModel):
    answer: str
    source: str
    confidence: Optional[float] = None
    referenced_rules: Optional[List[str]] = None


# Penalty Prediction API Schemas
class PenaltyRequest(BaseModel):
    incident_type: IncidentType = Field(..., description="Type of incident")
    track_condition: str = Field(..., description="Track condition during incident")
    intent: str = Field(..., description="Driver intent: accidental/intentional/racing_incident")
    driver_history: Optional[Dict[str, Any]] = Field(None, description="Driver penalty history")


class PenaltyResponse(BaseModel):
    predicted_penalty: str = Field(..., description="Predicted penalty (e.g., '5s', 'grid_drop')")
    confidence: float = Field(..., description="Confidence in prediction (0-1)")
    referenced_rule: str = Field(..., description="Referenced FIA rule (e.g., Article 38.3)")
    reasoning: str = Field(..., description="Explanation of penalty prediction")


# Natural Query API Schemas
class NaturalQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class NaturalQueryResponse(BaseModel):
    answer: str
    query_type: str = Field(..., description="Type of query: performance/regulatory/technical")
    confidence: float
    data_sources: List[str] = Field(..., description="Data sources used")


# Emotion Classification API Schemas
class EmotionRequest(BaseModel):
    audio_file: str = Field(..., description="Base64 encoded audio or file path")


class EmotionResponse(BaseModel):
    emotion: EmotionType
    confidence: float
    timestamp: Optional[str] = None
    duration: Optional[float] = None


# Ghost Car API Schemas
class GhostCarRequest(BaseModel):
    lap1_telemetry: Dict[str, Any] = Field(..., description="Telemetry data for first lap")
    lap2_telemetry: Dict[str, Any] = Field(..., description="Telemetry data for second lap")
    track_section: str = Field(..., description="Track section for comparison")


class GhostCarResponse(BaseModel):
    frame_deltas: List[float] = Field(..., description="Frame-by-frame time deltas")
    speed_comparison: Dict[str, List[float]] = Field(..., description="Speed traces comparison")
    braking_zones: Dict[str, List[int]] = Field(..., description="Braking zones comparison")
    drs_usage: Dict[str, List[bool]] = Field(..., description="DRS usage comparison")
    visualization_url: Optional[str] = Field(None, description="URL to visualization")


# Setup Recommendation API Schemas
class DriverPreferences(BaseModel):
    preferred_ride_height: Optional[float] = Field(None, description="Preferred ride height")
    preferred_wing_angles: Optional[Dict[str, float]] = Field(None, description="Preferred wing angles")
    preferred_diff_settings: Optional[Dict[str, float]] = Field(None, description="Preferred differential settings")


class TrackProfile(BaseModel):
    track_name: str = Field(..., description="Track name")
    track_length: float = Field(..., description="Track length in meters")
    corners: int = Field(..., description="Number of corners")
    high_speed_sections: int = Field(..., description="Number of high-speed sections")
    low_speed_sections: int = Field(..., description="Number of low-speed sections")


class WeatherData(BaseModel):
    condition: WeatherCondition = Field(..., description="Weather condition")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    wind_speed: Optional[float] = Field(None, description="Wind speed in km/h")


class SetupRequest(BaseModel):
    driver_preferences: DriverPreferences
    track_profile: TrackProfile
    weather: WeatherData


class SetupResponse(BaseModel):
    ride_height: float = Field(..., description="Recommended ride height")
    front_wing_angle: float = Field(..., description="Front wing angle")
    rear_wing_angle: float = Field(..., description="Rear wing angle")
    diff_settings: Dict[str, float] = Field(..., description="Differential settings")
    confidence: float = Field(..., description="Confidence in setup recommendation")
    reasoning: str = Field(..., description="Explanation of setup choices")


# Health Check Schemas
class HealthResponse(BaseModel):
    status: str
    modules: List[str]
    version: str = "1.0.0"
    timestamp: Optional[str] = None 