"""Pydantic request model and JSON serialiser for the heuristic strategy engine.

``StrategyRequest`` is meant to be used directly as a FastAPI request body and by the
natural-language router (``StrategyRequest.from_context``). Invalid input raises
``pydantic.ValidationError`` (a ``ValueError`` subclass); the engine raises ``ValueError``.
Cross-field rules call the engine's and the calibration's ``check_*`` functions, so each rule
and its message live in one place.

    body = generate_strategy_response(StrategyRequest.model_validate(payload))

(``strategy_result_to_dict`` serialises a ``StrategyResult`` from ``generate_strategy``.)
``TyreCalibrationRequest`` is the body for estimating ``tire_data`` from lap history:

    body = calibrate_tyres_response(TyreCalibrationRequest.model_validate(payload))
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Dict, List, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from .calibration import (
    MAX_CALIBRATION_LAPS,
    MAX_FUEL_CORRECTION_S_PER_LAP,
    CompoundCalibration,
    ExcludedLap,
    LapRecord,
    TyreCalibrationResult,
    check_fuel_race_laps,
    check_override_has_laps,
    estimate_tire_parameters,
)
from .strategy_engine import (
    DAMAGE_PARTS,
    DRIVER_OVERRIDE_KEYS,
    MAX_BASE_PERFORMANCE,
    MAX_BRAKE_TEMP_C,
    MAX_COMPETITORS,
    MAX_DEGRADATION_RATE,
    MAX_FUEL_LOAD_KG,
    MAX_GAP_S,
    MAX_LAP_TIME_S,
    MAX_LAP_TIME_SAMPLES,
    MAX_LIST_ITEMS,
    MAX_PIT_STOP_DELTA_S,
    MAX_PIT_STOPS_COMPLETED,
    MAX_POSITION,
    MAX_SECTORS,
    MAX_TEXT_LENGTH,
    MAX_TIRE_AGE_LAPS,
    MAX_TOTAL_LAPS,
    MAX_TRACK_TEMPERATURE_C,
    MAX_WARM_UP_LAPS,
    MIN_BASE_PERFORMANCE,
    MIN_LAP_TIME_S,
    MIN_TRACK_TEMPERATURE_C,
    CarStatus,
    Competitor,
    CompetitorSignal,
    DriverProfile,
    RaceState,
    StrategyOption,
    StrategyResult,
    TireCompound,
    TireData,
    WeatherCondition,
    check_current_compound_has_tire_data,
    check_driver_value_supplied,
    check_fitted_tyre_pair,
    check_lap_range,
    check_peak_window_order,
    check_tire_key,
    check_unique_driver_ids,
    generate_strategy,
)

STRATEGY_CONTEXT_KEYS: Tuple[str, ...] = (
    "telemetry",
    "car_status",
    "driver_profile",
    "tire_data",
    "race_state",
    "competition",
)

_CONFIG = ConfigDict(extra="forbid", allow_inf_nan=False)

# Numbers are strict: JSON integers are accepted for floats, but booleans, numeric strings
# and fractional lap counts are rejected. NaN/Infinity are rejected by allow_inf_nan=False.
Unit = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
LapTime = Annotated[float, Field(strict=True, ge=MIN_LAP_TIME_S, le=MAX_LAP_TIME_S)]
TyreLap = Annotated[int, Field(strict=True, ge=1, le=MAX_TOTAL_LAPS)]
TyreAge = Annotated[int, Field(strict=True, ge=0, le=MAX_TIRE_AGE_LAPS)]
Gap = Annotated[float, Field(strict=True, ge=0.0, le=MAX_GAP_S)]
ShortText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH)]
SectorTime = Annotated[float, Field(strict=True, gt=0.0, le=MAX_LAP_TIME_S)]
SectorNumber = Annotated[int, Field(ge=1, le=MAX_SECTORS)]  # JSON object keys "1".."50" are parsed to ints

# Free-form entries for inputs that are accepted but not modelled. Values may be text, finite
# numbers, booleans, null, or a flat list of those; nested objects are rejected. All bounded.
_FREE_NUMBER_LIMIT = 1e9
FreeScalar = Union[
    StrictBool,
    Annotated[int, Field(strict=True, ge=-int(_FREE_NUMBER_LIMIT), le=int(_FREE_NUMBER_LIMIT))],
    Annotated[float, Field(strict=True, ge=-_FREE_NUMBER_LIMIT, le=_FREE_NUMBER_LIMIT)],
    ShortText,
    None,
]
FreeValue = Union[FreeScalar, Annotated[List[FreeScalar], Field(max_length=MAX_LIST_ITEMS)]]
FreeEntry = Annotated[Dict[ShortText, FreeValue], Field(max_length=10)]
FREE_ENTRY_DESCRIPTION = (
    "objects of up to 10 keys whose values are text (<= 64 chars), finite numbers, booleans, null or flat "
    "lists of those (<= 50 items); nested objects are rejected"
)


class TelemetryInput(BaseModel):
    """Recent pace data. The mean of lap_times is the base lap time (tyre performance 1.0)."""

    model_config = _CONFIG

    lap_times: List[LapTime] = Field(
        min_length=1,
        max_length=MAX_LAP_TIME_SAMPLES,
        description=f"Recent representative lap times in seconds ({MIN_LAP_TIME_S:g}-{MAX_LAP_TIME_S:g}). Required.",
    )
    braking_consistency: Optional[Unit] = Field(
        None, description="Measured braking consistency 0-1; overrides driver_profile.braking_consistency."
    )
    throttle_aggressiveness: Optional[Unit] = Field(
        None, description="Measured throttle aggressiveness 0-1; overrides driver_profile.throttle_aggressiveness."
    )
    sector_times: Optional[
        Union[
            Annotated[Dict[SectorNumber, SectorTime], Field(min_length=1, max_length=MAX_SECTORS)],
            Annotated[List[SectorTime], Field(min_length=1, max_length=MAX_SECTORS)],
        ]
    ] = Field(
        None,
        description=(
            'Sector times in seconds, {"1": 28.1, ...} or a list from sector 1. Accepted (so one natural-query '
            "context can serve performance and strategy questions) but not modelled."
        ),
    )


class DamageInput(BaseModel):
    """Damage levels 0 (undamaged) to 1 (destroyed). Omitted parts are treated as undamaged."""

    model_config = _CONFIG

    front_wing: Unit = Field(0.0, description="Front-wing damage 0-1 (up to +2.5% lap time).")
    floor: Unit = Field(0.0, description="Floor damage 0-1 (max of floor/diffuser: up to +4% lap time).")
    diffuser: Unit = Field(0.0, description="Diffuser damage 0-1 (max of floor/diffuser: up to +4% lap time).")


class CarStatusInput(BaseModel):
    model_config = _CONFIG

    engine_wear: Unit = Field(description="Engine wear 0-1; above 0.7 adds up to +2% lap time.")
    brake_wear: Unit = Field(description="Brake wear 0-1; above 0.8 adds up to +1.5% lap time.")
    damage: DamageInput = Field(default_factory=DamageInput, description="Aerodynamic damage by part.")
    fuel_load: Optional[Annotated[float, Field(strict=True, ge=0.0, le=MAX_FUEL_LOAD_KG)]] = Field(
        None, description="Fuel load in kg. Accepted but not modelled."
    )
    brake_temp: Optional[Annotated[float, Field(strict=True, ge=0.0, le=MAX_BRAKE_TEMP_C)]] = Field(
        None, description="Brake temperature in degrees C. Accepted but not modelled."
    )
    ers_availability: Optional[Unit] = Field(None, description="ERS availability 0-1. Accepted but not modelled.")


class DriverProfileInput(BaseModel):
    model_config = _CONFIG

    tire_management: Unit = Field(description="0-1; below 0.6 with throttle_aggressiveness above 0.8 adds +10% lap time.")
    risk_tolerance: Unit = Field(description="0-1; above 0.8 adds +3% lap time (heuristic).")
    braking_consistency: Optional[Unit] = Field(
        None,
        description="0-1; below 0.7 adds up to +21% lap time. Required unless telemetry.braking_consistency is given "
        "(which overrides it).",
    )
    throttle_aggressiveness: Optional[Unit] = Field(
        None,
        description="0-1; see tire_management. Required unless telemetry.throttle_aggressiveness is given "
        "(which overrides it).",
    )
    overtaking_style: Optional[ShortText] = Field(None, description="Free text. Accepted but not modelled.")


class TireDataInput(BaseModel):
    """Heuristic tyre model for one compound (performance factors are relative to the base lap time)."""

    model_config = _CONFIG

    compound: Optional[TireCompound] = Field(None, description="Optional; must equal the tire_data key if given.")
    base_performance: float = Field(
        strict=True,
        ge=MIN_BASE_PERFORMANCE,
        le=MAX_BASE_PERFORMANCE,
        description=f"Peak performance factor, {MIN_BASE_PERFORMANCE:g}-{MAX_BASE_PERFORMANCE:g} (1.0 = base lap time).",
    )
    degradation_rate: float = Field(
        strict=True, ge=0.0, le=MAX_DEGRADATION_RATE, description="Performance lost per lap after the peak window."
    )
    warm_up_laps: int = Field(
        strict=True, ge=0, le=MAX_WARM_UP_LAPS, description="Laps to ramp from 90% to 100% performance."
    )
    peak_performance_window: Tuple[TyreLap, TyreLap] = Field(
        description="[start, end] tyre laps; degradation starts after end (start does not change lap times)."
    )
    pit_stop_delta: float = Field(
        strict=True, ge=0.0, le=MAX_PIT_STOP_DELTA_S, description="Seconds lost by a stop that fits this compound."
    )

    @model_validator(mode="after")
    def _window_order(self) -> "TireDataInput":
        check_peak_window_order(*self.peak_performance_window)
        return self


class RaceStateInput(BaseModel):
    model_config = _CONFIG

    current_lap: int = Field(
        strict=True, ge=1, le=MAX_TOTAL_LAPS, description="Next lap to be driven; laps current_lap..total_laps are planned."
    )
    total_laps: int = Field(strict=True, ge=1, le=MAX_TOTAL_LAPS, description="Race distance in laps.")
    weather: WeatherCondition = Field(description="Assumed constant for the remaining laps.")
    track_temperature: float = Field(
        strict=True,
        ge=MIN_TRACK_TEMPERATURE_C,
        le=MAX_TRACK_TEMPERATURE_C,
        description="Degrees C; above 35 soft performance x0.97 and hard x1.02.",
    )
    track_evolution: Optional[Unit] = Field(None, description="Accepted but not modelled.")
    safety_car_probability: Optional[Unit] = Field(None, description="Accepted but not modelled.")
    yellow_flag_risk: Optional[Unit] = Field(None, description="Accepted but not modelled.")
    weather_forecast: Optional[List[FreeEntry]] = Field(
        None,
        max_length=MAX_LIST_ITEMS,
        description=f"Accepted but not modelled (weather is held constant). Entries: {FREE_ENTRY_DESCRIPTION}.",
    )
    current_compound: Optional[TireCompound] = Field(
        None, description="Currently fitted compound; give with current_tire_age. If omitted, plans assume a fresh set."
    )
    current_tire_age: Optional[TyreAge] = Field(None, description="Laps already run on the fitted set.")
    used_compounds: Optional[List[TireCompound]] = Field(
        None,
        max_length=10,
        description="Compounds run earlier in the race (before the fitted set); used for the two-compound rule.",
    )
    own_gap_to_leader: Optional[Gap] = Field(
        None, description="Your gap to the leader in seconds; enables competitor undercut signals."
    )

    @model_validator(mode="after")
    def _consistency(self) -> "RaceStateInput":
        check_lap_range(self.current_lap, self.total_laps)
        check_fitted_tyre_pair(self.current_compound, self.current_tire_age)
        return self


class CompetitorInput(BaseModel):
    model_config = _CONFIG

    driver_id: ShortText = Field(description="Unique competitor identifier.")
    tire_compound: TireCompound
    tire_age: TyreAge = Field(description="Laps on the competitor's current set.")
    gap_to_leader: Gap = Field(description="Seconds; compared with race_state.own_gap_to_leader.")
    current_position: Optional[Annotated[int, Field(strict=True, ge=1, le=MAX_POSITION)]] = Field(
        None, description="Accepted but not modelled."
    )
    gap_ahead: Optional[Gap] = Field(None, description="Accepted but not modelled.")
    gap_behind: Optional[Gap] = Field(None, description="Accepted but not modelled.")
    pit_stops_completed: Optional[Annotated[int, Field(strict=True, ge=0, le=MAX_PIT_STOPS_COMPLETED)]] = Field(
        None, description="Accepted but not modelled."
    )
    estimated_strategy: Optional[List[FreeEntry]] = Field(
        None, max_length=10, description=f"Accepted but not modelled. Entries: {FREE_ENTRY_DESCRIPTION}."
    )


class StrategyRequest(BaseModel):
    """Inputs for the heuristic strategy engine (POST body and natural-query context)."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "telemetry": {"lap_times": [95.6, 95.3, 95.9, 95.4]},
                    "car_status": {"engine_wear": 0.3, "brake_wear": 0.4, "damage": {"front_wing": 0.1}},
                    "driver_profile": {
                        "tire_management": 0.7,
                        "risk_tolerance": 0.6,
                        "braking_consistency": 0.75,
                        "throttle_aggressiveness": 0.7,
                    },
                    "tire_data": {
                        "soft": {"base_performance": 1.0, "degradation_rate": 0.004, "warm_up_laps": 2,
                                 "peak_performance_window": [2, 10], "pit_stop_delta": 22.0},
                        "medium": {"base_performance": 0.992, "degradation_rate": 0.0025, "warm_up_laps": 3,
                                   "peak_performance_window": [3, 18], "pit_stop_delta": 22.0},
                        "hard": {"base_performance": 0.985, "degradation_rate": 0.0015, "warm_up_laps": 4,
                                 "peak_performance_window": [4, 28], "pit_stop_delta": 22.0},
                    },
                    "race_state": {"current_lap": 18, "total_laps": 57, "weather": "dry", "track_temperature": 32.0,
                                   "current_compound": "medium", "current_tire_age": 17, "used_compounds": []},
                    "competition": [],
                }
            ]
        },
    )

    telemetry: TelemetryInput
    car_status: CarStatusInput
    driver_profile: DriverProfileInput
    tire_data: Dict[TireCompound, TireDataInput] = Field(
        min_length=1, max_length=len(TireCompound), description="Tyre model keyed by compound (soft/medium/hard/intermediate/wet)."
    )
    race_state: RaceStateInput
    competition: List[CompetitorInput] = Field(default_factory=list, max_length=MAX_COMPETITORS)

    @model_validator(mode="after")
    def _cross_checks(self) -> "StrategyRequest":
        for key, tire in self.tire_data.items():
            if tire.compound is not None:
                check_tire_key(key, tire.compound)
        check_current_compound_has_tire_data(self.race_state.current_compound, self.tire_data)
        check_unique_driver_ids([rival.driver_id for rival in self.competition])
        for name in DRIVER_OVERRIDE_KEYS:
            check_driver_value_supplied(name, getattr(self.telemetry, name), getattr(self.driver_profile, name))
        return self

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "StrategyRequest":
        """Validate the strategy keys of a natural-query context.

        Unrelated top-level keys (e.g. audio_file, track_profile) are ignored. Inside the strategy keys the
        rules are the same as for the POST body; telemetry.sector_times is accepted but not modelled.
        """

        if not isinstance(context, Mapping):
            raise ValueError("context must be an object")
        return cls.model_validate({key: context[key] for key in STRATEGY_CONTEXT_KEYS if key in context})

    def to_engine_inputs(self) -> Dict[str, Any]:
        """Keyword arguments for ``generate_strategy``."""

        telemetry: Dict[str, Any] = {"lap_times": list(self.telemetry.lap_times)}
        for name in DRIVER_OVERRIDE_KEYS:
            value = getattr(self.telemetry, name)
            if value is not None:
                telemetry[name] = value
        sectors = self.telemetry.sector_times
        if sectors is not None:
            telemetry["sector_times"] = dict(sectors) if isinstance(sectors, dict) else list(sectors)

        car = self.car_status
        damage = {part: getattr(car.damage, part) for part in DAMAGE_PARTS}
        race = self.race_state
        return {
            "telemetry": telemetry,
            "car_status": CarStatus(
                engine_wear=car.engine_wear,
                brake_wear=car.brake_wear,
                damage=damage,
                fuel_load=car.fuel_load,
                brake_temp=car.brake_temp,
                ers_availability=car.ers_availability,
            ),
            "driver_profile": DriverProfile(**self.driver_profile.model_dump()),
            "tire_data": {
                key: TireData(
                    compound=key,
                    base_performance=tire.base_performance,
                    degradation_rate=tire.degradation_rate,
                    warm_up_laps=tire.warm_up_laps,
                    peak_performance_window=tuple(tire.peak_performance_window),
                    pit_stop_delta=tire.pit_stop_delta,
                )
                for key, tire in self.tire_data.items()
            },
            "race_state": RaceState(
                current_lap=race.current_lap,
                total_laps=race.total_laps,
                weather=race.weather,
                track_temperature=race.track_temperature,
                track_evolution=race.track_evolution,
                safety_car_probability=race.safety_car_probability,
                yellow_flag_risk=race.yellow_flag_risk,
                weather_forecast=None if race.weather_forecast is None else [dict(e) for e in race.weather_forecast],
                current_compound=race.current_compound,
                current_tire_age=race.current_tire_age,
                used_compounds=None if race.used_compounds is None else list(race.used_compounds),
                own_gap_to_leader=race.own_gap_to_leader,
            ),
            "competition": [
                Competitor(
                    driver_id=rival.driver_id,
                    tire_compound=rival.tire_compound,
                    tire_age=rival.tire_age,
                    gap_to_leader=rival.gap_to_leader,
                    current_position=rival.current_position,
                    gap_ahead=rival.gap_ahead,
                    gap_behind=rival.gap_behind,
                    pit_stops_completed=rival.pit_stops_completed,
                    estimated_strategy=None
                    if rival.estimated_strategy is None
                    else [dict(e) for e in rival.estimated_strategy],
                )
                for rival in self.competition
            ],
        }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _round(value: float, digits: int = 3) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Refusing to serialise a non-finite value: {value!r}")
    return round(number, digits)


def _stint_to_dict(stint: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "start_lap": int(stint["start_lap"]),
        "end_lap": int(stint["end_lap"]),
        "laps": int(stint["laps"]),
        "tire_compound": stint["tire_compound"].value,
        "tire_age_start": int(stint["tire_age_start"]),
        "tire_age_end": int(stint["tire_age_end"]),
        "fitted_at_stop": bool(stint["fitted_at_stop"]),
        "average_lap_time": _round(stint["average_lap_time"]),
        "best_lap_time": _round(stint["best_lap_time"]),
        "worst_lap_time": _round(stint["worst_lap_time"]),
        "total_time": _round(stint["total_time"]),
        "start_performance": _round(stint["start_performance"], 4),
        "end_performance": _round(stint["end_performance"], 4),
        "laps_beyond_peak_window": int(stint["laps_beyond_peak_window"]),
        "laps_at_performance_floor": int(stint["laps_at_performance_floor"]),
    }


def strategy_option_to_dict(option: StrategyOption, rank: Optional[int] = None) -> Dict[str, Any]:
    """JSON-safe view of one candidate (times in seconds, rounded to milliseconds)."""

    return {
        "strategy_id": option.strategy_id,
        "rank": rank,
        "pit_stops": option.stop_count,
        "tire_compounds": [compound.value for compound in option.tire_compounds],
        "pit_laps": [int(lap) for lap in option.pit_laps],
        "projected_race_time": _round(option.projected_race_time),
        "driving_time_s": _round(option.driving_time_s),
        "pit_time_loss_s": _round(option.pit_time_loss_s),
        "delta_to_best_s": None if option.delta_to_best_s is None else _round(option.delta_to_best_s),
        "risk_level": option.risk_level,
        "two_compound_rule": option.two_compound_rule,
        "stint_breakdown": [_stint_to_dict(stint) for stint in option.stint_breakdown],
        "notes": list(option.notes),
    }


def _signal_to_dict(signal: CompetitorSignal) -> Dict[str, Any]:
    return {
        "driver_id": signal.driver_id,
        "signal": signal.signal,
        "gap_s": _round(signal.gap_s),
        "tire_compound": signal.tire_compound.value,
        "tire_age": int(signal.tire_age),
        "explanation": signal.explanation,
    }


def strategy_result_to_dict(result: StrategyResult) -> Dict[str, Any]:
    """JSON-safe response body for ``generate_strategy`` output (enums as strings, rounded floats)."""

    return {
        "heuristic": True,
        "method": (
            "Heuristic lap-time model; pit laps chosen by exact dynamic programming over stint lengths "
            "for every compound sequence with up to 3 stops."
        ),
        "current_lap": result.current_lap,
        "total_laps": result.total_laps,
        "remaining_laps": result.remaining_laps,
        "tire_state": result.tire_state,
        "model": {
            "base_lap_time_s": _round(result.base_lap_time),
            "driver_multiplier": _round(result.driver_multiplier, 4),
            "damage_multiplier": _round(result.damage_multiplier, 4),
        },
        "best_strategy_id": result.best.strategy_id,
        "strategies": [
            strategy_option_to_dict(option, rank) for rank, option in enumerate(result.candidates, start=1)
        ],
        "assumptions": list(result.assumptions),
        "not_modelled_inputs": list(result.not_modelled_inputs),
        "competitor_signals": [_signal_to_dict(signal) for signal in result.competitor_signals],
        "competitor_signals_note": result.competitor_signals_note,
        "search": {
            "sequences_evaluated": result.sequences_evaluated,
            "sequences_skipped_too_few_laps": result.sequences_skipped_too_few_laps,
            "sequences_excluded_two_compound_rule": result.sequences_excluded_two_compound_rule,
            "candidates_returned": len(result.candidates),
        },
    }


def generate_strategy_response(request: StrategyRequest) -> Dict[str, Any]:
    """Run the engine for an already validated request and serialise the result (raises ValueError)."""

    return strategy_result_to_dict(generate_strategy(**request.to_engine_inputs()))


# ---------------------------------------------------------------------------
# Tyre-parameter calibration (estimate tire_data from lap history)
# ---------------------------------------------------------------------------

WarmUpLaps = Annotated[int, Field(strict=True, ge=0, le=MAX_WARM_UP_LAPS)]


def _example_calibration_laps() -> List[Dict[str, Any]]:
    """A small documented example: one soft and one medium stint, out-laps and in-laps flagged."""

    laps: List[Dict[str, Any]] = []
    race_lap = 1
    for compound, stint, peak_s, loss_s, peak_end in (("soft", 12, 80.0, 0.3, 5), ("medium", 16, 80.6, 0.2, 8)):
        for age in range(1, stint + 1):
            lap: Dict[str, Any] = {
                "compound": compound,
                "tire_age": age,
                "lap_time": round(peak_s + loss_s * max(0, age - peak_end), 3),
                "race_lap": race_lap,
            }
            if age == 1:
                lap["pit_out"] = True
                lap["lap_time"] += 19.0
            if age == stint:
                lap["pit_in"] = True
                lap["lap_time"] += 4.0
            laps.append(lap)
            race_lap += 1
    return laps


class CalibrationLapInput(BaseModel):
    """One timed lap. Flagged laps (pit_out, pit_in, safety_car) are excluded from the fit."""

    model_config = _CONFIG

    compound: TireCompound
    tire_age: Annotated[int, Field(strict=True, ge=1, le=MAX_TIRE_AGE_LAPS)] = Field(
        description="Lap number on this tyre set (1 = first lap on it)."
    )
    lap_time: LapTime = Field(description=f"Lap time in seconds ({MIN_LAP_TIME_S:g}-{MAX_LAP_TIME_S:g}).")
    race_lap: Optional[TyreLap] = Field(
        None, description="Race lap number; required on unflagged laps when fuel_correction_s_per_lap > 0."
    )
    pit_out: StrictBool = Field(False, description="Out-lap from the pit lane (excluded).")
    pit_in: StrictBool = Field(False, description="In-lap to the pit lane (excluded).")
    safety_car: StrictBool = Field(False, description="Safety-car, VSC or red-flag lap (excluded).")


class TyreCalibrationRequest(BaseModel):
    """Lap history for ``estimate_tire_parameters`` (POST body for the tyre-calibration endpoint)."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "laps": _example_calibration_laps(),
                    "weather": "dry",
                    "track_temperature": 31.0,
                    "pit_stop_delta": 22.0,
                }
            ]
        },
    )

    laps: List[CalibrationLapInput] = Field(
        min_length=1,
        max_length=MAX_CALIBRATION_LAPS,
        description=f"Timed laps, at most {MAX_CALIBRATION_LAPS}. At least 5 clean laps per compound are needed.",
    )
    weather: WeatherCondition = Field(description="Conditions of the whole history (one session, held constant).")
    track_temperature: float = Field(
        strict=True,
        ge=MIN_TRACK_TEMPERATURE_C,
        le=MAX_TRACK_TEMPERATURE_C,
        description="Degrees C of the history; above 35 the engine's hot-track factor is divided out.",
    )
    pit_stop_delta: float = Field(
        strict=True,
        ge=0.0,
        le=MAX_PIT_STOP_DELTA_S,
        description="Seconds lost per stop, copied into every tire_data entry (not estimated).",
    )
    fuel_correction_s_per_lap: float = Field(
        0.0,
        strict=True,
        ge=0.0,
        le=MAX_FUEL_CORRECTION_S_PER_LAP,
        description="Seconds gained per lap of fuel burnt; 0 = no correction. Needs race_lap on unflagged laps.",
    )
    peak_window_end: Optional[Dict[TireCompound, TyreLap]] = Field(
        None,
        max_length=len(TireCompound),
        description="Fix the peak-window end for these compounds instead of detecting it.",
    )
    warm_up_laps: Optional[Dict[TireCompound, WarmUpLaps]] = Field(
        None, max_length=len(TireCompound), description="Fix warm_up_laps for these compounds instead of detecting it."
    )

    @model_validator(mode="after")
    def _cross_checks(self) -> "TyreCalibrationRequest":
        present = {lap.compound for lap in self.laps}
        for name in ("peak_window_end", "warm_up_laps"):
            for compound in getattr(self, name) or {}:
                check_override_has_laps(name, compound, present)
        check_fuel_race_laps(self.laps, self.fuel_correction_s_per_lap)
        return self

    def to_calibration_inputs(self) -> Dict[str, Any]:
        """Keyword arguments for ``estimate_tire_parameters``."""

        return {
            "laps": [
                LapRecord(
                    compound=lap.compound,
                    tire_age=lap.tire_age,
                    lap_time=lap.lap_time,
                    race_lap=lap.race_lap,
                    pit_out=lap.pit_out,
                    pit_in=lap.pit_in,
                    safety_car=lap.safety_car,
                )
                for lap in self.laps
            ],
            "weather": self.weather,
            "track_temperature": self.track_temperature,
            "pit_stop_delta": self.pit_stop_delta,
            "fuel_correction_s_per_lap": self.fuel_correction_s_per_lap,
            "peak_window_end": None if self.peak_window_end is None else dict(self.peak_window_end),
            "warm_up_laps": None if self.warm_up_laps is None else dict(self.warm_up_laps),
        }


def _signed_round(value: float, digits: int) -> float:
    return _round(value, digits) + 0.0  # + 0.0 turns -0.0 into 0.0


def tire_data_to_dict(tire: TireData) -> Dict[str, Any]:
    """One ``tire_data`` entry in the StrategyRequest format (validated against ``TireDataInput``)."""

    entry = {
        "compound": tire.compound.value,
        "base_performance": _round(tire.base_performance, 6),
        "degradation_rate": _round(tire.degradation_rate, 6),
        "warm_up_laps": int(tire.warm_up_laps),
        "peak_performance_window": [int(lap) for lap in tire.peak_performance_window],
        "pit_stop_delta": _round(tire.pit_stop_delta),
    }
    TireDataInput.model_validate(entry)  # the calibrated entry must be usable as-is in a StrategyRequest
    return entry


def _fit_to_dict(entry: CompoundCalibration) -> Optional[Dict[str, Any]]:
    peak, loss, spread, ages = (
        entry.peak_lap_time,
        entry.initial_degradation_s_per_lap,
        entry.residual_std_s,
        entry.tire_age_range,
    )
    if peak is None or loss is None or spread is None or ages is None:
        return None  # the compound was not fitted (insufficient data)
    return {
        "peak_lap_time_s": _round(peak),
        "initial_degradation_s_per_lap": _round(loss),
        "tire_age_range": [int(ages[0]), int(ages[1])],
        "r_squared": None if entry.r_squared is None else _signed_round(entry.r_squared, 4),
        "residual_std_s": _round(spread),
        "outlier_rounds": entry.outlier_iterations,
    }


def _compound_calibration_to_dict(entry: CompoundCalibration) -> Dict[str, Any]:
    return {
        "status": entry.status,
        "reason": entry.reason,
        "laps_supplied": entry.laps_supplied,
        "laps_flagged": entry.laps_flagged,
        "clean_laps": entry.clean_laps,
        "laps_used": entry.laps_used,
        "outliers_rejected": entry.outliers_rejected,
        "tire_data": None if entry.tire_data is None else tire_data_to_dict(entry.tire_data),
        "warm_up_source": entry.warm_up_source,
        "peak_window_end_source": entry.peak_window_end_source,
        "degradation_observed": entry.degradation_observed,
        "fit": _fit_to_dict(entry),
        "notes": list(entry.notes),
    }


def _excluded_lap_to_dict(lap: ExcludedLap) -> Dict[str, Any]:
    return {
        "index": lap.index,
        "compound": lap.compound.value,
        "tire_age": lap.tire_age,
        "lap_time": _round(lap.lap_time),
        "reason": lap.reason,
        "fitted_lap_time": None if lap.fitted_lap_time is None else _round(lap.fitted_lap_time),
        "residual_s": None if lap.residual_s is None else _signed_round(lap.residual_s, 3),
    }


def tyre_calibration_to_dict(result: TyreCalibrationResult) -> Dict[str, Any]:
    """JSON-safe response body for ``estimate_tire_parameters`` output (enums as strings, rounded floats).

    ``tire_data`` holds only compounds with status "estimated" and can be pasted into a StrategyRequest;
    pass ``estimated_base_lap_time_s`` as ``telemetry.lap_times`` with it (see the README for the caveats).
    """

    base = result.estimated_base_lap_time
    return {
        "method": (
            "Least-squares inversion of the strategy engine's tyre model (warm-up ramp, peak window, linear "
            "degradation) per compound, with robust outlier rejection."
        ),
        "estimated_base_lap_time_s": None if base is None else _round(base),
        "reference_compound": None if result.reference_compound is None else result.reference_compound.value,
        "tire_data": {compound.value: tire_data_to_dict(tire) for compound, tire in result.tire_data.items()},
        "compounds": {
            compound.value: _compound_calibration_to_dict(entry) for compound, entry in result.compounds.items()
        },
        "excluded_laps": [_excluded_lap_to_dict(lap) for lap in result.excluded_laps],
        "laps_supplied": result.laps_supplied,
        "conditions": {
            "weather": result.weather.value,
            "track_temperature": _round(result.track_temperature, 2),
            "pit_stop_delta": _round(result.pit_stop_delta),
            "fuel_correction_s_per_lap": _round(result.fuel_correction_s_per_lap, 4),
            "fuel_reference_lap": result.fuel_reference_lap,
        },
        "assumptions": list(result.assumptions),
    }


def calibrate_tyres_response(request: TyreCalibrationRequest) -> Dict[str, Any]:
    """Estimate tyre parameters for an already validated request and serialise them (raises ValueError)."""

    return tyre_calibration_to_dict(estimate_tire_parameters(**request.to_calibration_inputs()))
