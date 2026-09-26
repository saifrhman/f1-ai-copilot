"""Pydantic request models for the setup optimiser (usable directly as FastAPI request bodies).

Field bounds mirror the engine's validation constants; cross-field rules (section counts vs
corners) are delegated to the engine validator so they live in one place. Floats are strict
(no numeric strings or booleans) and must be finite. ``pydantic.ValidationError`` is a
``ValueError`` subclass, so callers can map both to HTTP 422.
"""

from __future__ import annotations

from typing import Annotated, Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from core_modules.setup_optimizer.setup_recommender import (
    AVERAGE_SPEED_RANGE_KPH,
    DEFAULT_N_TRIALS,
    DEFAULT_SEED,
    MAX_CORNERS,
    MAX_N_TRIALS,
    MAX_SEED,
    MAX_WIND_SPEED_MS,
    MIN_N_TRIALS,
    N_STARTUP_TRIALS,
    SETUP_BOUNDS,
    TEMPERATURE_RANGE_C,
    TRACK_LENGTH_RANGE_M,
    TRACK_NAME_MAX_LENGTH,
    DriverPreferences,
    SetupInputs,
    TrackProfile,
    TrackType,
    WeatherCondition,
    WeatherData,
    validate_track_profile,
)

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


def _bounded(name: str, description: str) -> Any:
    lo, hi = SETUP_BOUNDS[name]
    return Field(
        None, ge=lo, le=hi, description=f"{description} ({lo:g}-{hi:g}); pins this parameter exactly in the search"
    )


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WingAnglePreferences(_StrictModel):
    front: Optional[FiniteFloat] = _bounded("front_wing_angle", "Front wing angle in degrees")
    rear: Optional[FiniteFloat] = _bounded("rear_wing_angle", "Rear wing angle in degrees")


class DiffPreferences(_StrictModel):
    preload: Optional[FiniteFloat] = _bounded("diff_preload", "Differential preload, % lock")
    power: Optional[FiniteFloat] = _bounded("diff_power", "On-throttle differential lock, %")
    coast: Optional[FiniteFloat] = _bounded("diff_coast", "Off-throttle differential lock, %")


class DriverPreferencesModel(_StrictModel):
    preferred_ride_height: Optional[FiniteFloat] = _bounded("ride_height", "Ride height in mm")
    preferred_wing_angles: Optional[WingAnglePreferences] = Field(None, description="Pinned wing angles")
    preferred_diff_settings: Optional[DiffPreferences] = Field(None, description="Pinned differential settings")
    risk_tolerance: FiniteFloat = Field(0.5, ge=0.0, le=1.0, description="0 = wants a stable car, 1 = accepts a pointy car")
    tire_management: FiniteFloat = Field(0.5, ge=0.0, le=1.0, description="0 = ignore tyre wear, 1 = protect the tyres")

    def to_engine(self) -> DriverPreferences:
        wings = self.preferred_wing_angles.model_dump(exclude_none=True) if self.preferred_wing_angles else None
        diff = self.preferred_diff_settings.model_dump(exclude_none=True) if self.preferred_diff_settings else None
        return DriverPreferences(
            preferred_ride_height=self.preferred_ride_height,
            preferred_wing_angles=wings or None,
            preferred_diff_settings=diff or None,
            risk_tolerance=self.risk_tolerance,
            tire_management=self.tire_management,
        )


class TrackProfileModel(_StrictModel):
    track_name: Optional[str] = Field(
        None, min_length=1, max_length=TRACK_NAME_MAX_LENGTH, description="Label for the reasoning text only; not modelled"
    )
    track_length: FiniteFloat = Field(ge=TRACK_LENGTH_RANGE_M[0], le=TRACK_LENGTH_RANGE_M[1], description="Lap length in metres")
    corners: StrictInt = Field(ge=1, le=MAX_CORNERS, description="Number of corners")
    high_speed_sections: StrictInt = Field(ge=0, le=MAX_CORNERS, description="Corners taken at high speed (<= corners)")
    low_speed_sections: StrictInt = Field(ge=0, le=MAX_CORNERS, description="Corners taken at low speed (<= corners)")
    track_type: TrackType = Field(description="Layout type; sets the heuristic kerb/bump severity")
    average_speed: FiniteFloat = Field(
        ge=AVERAGE_SPEED_RANGE_KPH[0], le=AVERAGE_SPEED_RANGE_KPH[1], description="Average lap speed in km/h"
    )
    downforce_requirement: FiniteFloat = Field(ge=0.0, le=1.0, description="0 = low-downforce track, 1 = maximum downforce")

    @model_validator(mode="after")
    def _check_sections(self) -> "TrackProfileModel":
        validate_track_profile(self.to_engine())
        return self

    def to_engine(self) -> TrackProfile:
        return TrackProfile(
            track_name=self.track_name,
            track_length=self.track_length,
            corners=self.corners,
            high_speed_sections=self.high_speed_sections,
            low_speed_sections=self.low_speed_sections,
            track_type=self.track_type,
            average_speed=self.average_speed,
            downforce_requirement=self.downforce_requirement,
        )


class WeatherModel(_StrictModel):
    condition: WeatherCondition = Field(description="dry, intermediate or wet")
    temperature: FiniteFloat = Field(
        ge=TEMPERATURE_RANGE_C[0], le=TEMPERATURE_RANGE_C[1], description="Ambient air temperature in °C (sets tyre pressures)"
    )
    humidity: Optional[FiniteFloat] = Field(None, ge=0.0, le=100.0, description="Relative humidity in %; validated, not modelled")
    wind_speed: Optional[FiniteFloat] = Field(
        None, ge=0.0, le=MAX_WIND_SPEED_MS, description="Wind speed in m/s; validated, not modelled"
    )

    def to_engine(self) -> WeatherData:
        return WeatherData(self.condition, self.temperature, self.humidity, self.wind_speed)


class SetupRequest(_StrictModel):
    driver_preferences: DriverPreferencesModel = Field(description="Driver style and optional pinned setup values")
    track_profile: TrackProfileModel = Field(
        description="Circuit characteristics that weight the heuristic objective (layout, speed, downforce need)"
    )
    weather: WeatherModel = Field(
        description="Track condition (sets grip, standing water and ride-height needs) and ambient temperature "
        "(sets cold tyre pressures)"
    )
    n_trials: StrictInt = Field(
        DEFAULT_N_TRIALS,
        ge=MIN_N_TRIALS,
        le=MAX_N_TRIALS,
        description=f"Optuna TPE trial budget; the first {N_STARTUP_TRIALS} trials are random. A deterministic local "
        "refinement follows (~0.6-0.8 s in total at the default, ~2.3 s at 300)",
    )
    seed: StrictInt = Field(
        DEFAULT_SEED,
        ge=0,
        le=MAX_SEED,
        description="TPE sampler seed; the same seed and inputs give an identical response. After refinement the "
        "recommended setup normally does not depend on the seed (confidence < 1 flags when it can)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
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
                    "n_trials": DEFAULT_N_TRIALS,
                    "seed": DEFAULT_SEED,
                }
            ]
        },
    )

    def assumed_defaults(self) -> List[str]:
        """Driver-preference model inputs the request left out, e.g. ``"driver_preferences.risk_tolerance=0.5"``.

        Only fields whose default feeds the model are listed (an absent ``preferred_*`` value just
        leaves that parameter free). Uses pydantic's ``model_fields_set``, so an explicitly sent
        default value counts as supplied.
        """
        preferences = self.driver_preferences
        return [
            f"driver_preferences.{name}={getattr(preferences, name):g}"
            for name, field in DriverPreferencesModel.model_fields.items()
            if field.default is not None and name not in preferences.model_fields_set
        ]

    def to_engine_inputs(self) -> SetupInputs:
        return SetupInputs(
            driver_preferences=self.driver_preferences.to_engine(),
            track_profile=self.track_profile.to_engine(),
            weather=self.weather.to_engine(),
            n_trials=self.n_trials,
            seed=self.seed,
            assumed_defaults=tuple(self.assumed_defaults()),
        )
