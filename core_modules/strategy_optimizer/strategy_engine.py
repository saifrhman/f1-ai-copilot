"""Transparent, deterministic heuristic race-strategy engine.

The engine answers one question: under the supplied tyre model, race state and
car/driver condition, which tyre-compound sequence (0 to 3 further stops) and which
pit laps give the lowest projected time for the remaining laps?

Everything here is a HEURISTIC, not a calibrated race simulator:

* lap time = mean(telemetry lap_times) x driver multiplier x damage/wear multiplier
  / tyre performance factor (telemetry braking/throttle values override the profile);
* tyre performance per tyre lap: warm-up ramp from 90% to 100% of base_performance
  over warm_up_laps, base_performance until the END of peak_performance_window, then a
  linear loss of degradation_rate per lap, scaled by weather and hot-track factors and
  floored at PERFORMANCE_FLOOR (the window START does not change lap times; laps at the
  floor are counted per stint and flagged, because their times are optimistic);
* each stop costs the pit_stop_delta of the compound fitted at that stop;
* weather stays constant for the remaining laps.

Lap time depends only on the compound and the tyre's own age, so for every compound
sequence the stint lengths are chosen by exact dynamic programming over cumulative
stint-time tables. Pit laps therefore respond to degradation, warm-up and pit loss.
The order of stints after the first one does not change the projected time in this
model, so equivalent orderings are merged.

All functions are pure: no module-level mutable state, safe to call from threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations_with_replacement
from numbers import Integral, Real
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class TireCompound(Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"


class WeatherCondition(Enum):
    DRY = "dry"
    WET = "wet"
    INTERMEDIATE = "intermediate"


DRY_COMPOUNDS: FrozenSet[TireCompound] = frozenset(
    {TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD}
)
WET_WEATHER_COMPOUNDS: FrozenSet[TireCompound] = frozenset(
    {TireCompound.INTERMEDIATE, TireCompound.WET}
)

# Validation limits (physical/plausibility ranges). The pydantic schema reuses them.
MAX_TOTAL_LAPS = 200
MAX_STOPS = 3
MAX_TIRE_AGE_LAPS = 100
MIN_LAP_TIME_S = 20.0
MAX_LAP_TIME_S = 600.0
MAX_LAP_TIME_SAMPLES = 200
MAX_SECTORS = 50
# base_performance below 0.5 would mean laps at least twice the base lap time; values near
# PERFORMANCE_FLOOR would be indistinguishable from each other, so they are rejected.
MIN_BASE_PERFORMANCE = 0.5
MAX_BASE_PERFORMANCE = 2.0
MAX_DEGRADATION_RATE = 0.5
MAX_WARM_UP_LAPS = 10
MAX_PIT_STOP_DELTA_S = 120.0
MIN_TRACK_TEMPERATURE_C = -10.0
MAX_TRACK_TEMPERATURE_C = 80.0
MAX_FUEL_LOAD_KG = 150.0
MAX_BRAKE_TEMP_C = 1500.0
MAX_GAP_S = 7200.0
MAX_COMPETITORS = 30
MAX_POSITION = 30
MAX_PIT_STOPS_COMPLETED = 10
MAX_LIST_ITEMS = 50
MAX_TEXT_LENGTH = 64
DAMAGE_PARTS: Tuple[str, ...] = ("front_wing", "floor", "diffuser")
DRIVER_OVERRIDE_KEYS: Tuple[str, ...] = ("braking_consistency", "throttle_aggressiveness")
# sector_times is accepted (so one natural-query context can serve performance and strategy
# questions) but not modelled; it is reported in not_modelled_inputs.
TELEMETRY_KEYS: Tuple[str, ...] = ("lap_times", *DRIVER_OVERRIDE_KEYS, "sector_times")

# Heuristic model constants.
PERFORMANCE_FLOOR = 0.20
UNDERCUT_WINDOW_S = 3.0
MAX_CANDIDATES = 8

# Simplified two-dry-compound rule status values.
RULE_SATISFIED = "satisfied"
RULE_WAIVED = "waived"
RULE_UNVERIFIED = "unverified"
RULE_VIOLATED = "violated"

_RULE_NOTES = {
    RULE_SATISFIED: "Simplified two-dry-compound rule: satisfied (two different dry compounds are used in the race).",
    RULE_WAIVED: "Simplified two-dry-compound rule: not required, because intermediate/wet tyres are used.",
    RULE_UNVERIFIED: (
        "Simplified two-dry-compound rule: not verified, because the compounds used before this lap "
        "are unknown (supply race_state.used_compounds)."
    ),
    RULE_VIOLATED: (
        "Simplified two-dry-compound rule: VIOLATED. The known tyre history and this plan use only one "
        "dry compound."
    ),
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class TireData:
    """Heuristic tyre model for one compound (performance 1.0 = the base lap time)."""

    compound: TireCompound
    base_performance: float
    degradation_rate: float
    warm_up_laps: int
    peak_performance_window: Tuple[int, int]
    pit_stop_delta: float


@dataclass
class DriverProfile:
    tire_management: float
    risk_tolerance: float
    # Each of these is needed only when telemetry does not supply it (telemetry overrides it).
    braking_consistency: Optional[float] = None
    throttle_aggressiveness: Optional[float] = None
    overtaking_style: Optional[str] = None  # accepted, not modelled


@dataclass
class CarStatus:
    engine_wear: float
    brake_wear: float
    damage: Dict[str, float] = field(default_factory=dict)  # parts in DAMAGE_PARTS; omitted = 0
    fuel_load: Optional[float] = None  # accepted, not modelled
    brake_temp: Optional[float] = None  # accepted, not modelled
    ers_availability: Optional[float] = None  # accepted, not modelled


@dataclass
class RaceState:
    current_lap: int  # the next lap to be driven (not yet completed)
    total_laps: int
    weather: WeatherCondition
    track_temperature: float
    track_evolution: Optional[float] = None  # accepted, not modelled
    safety_car_probability: Optional[float] = None  # accepted, not modelled
    yellow_flag_risk: Optional[float] = None  # accepted, not modelled
    weather_forecast: Optional[List[Dict[str, Any]]] = None  # accepted, not modelled
    current_compound: Optional[TireCompound] = None  # give together with current_tire_age
    current_tire_age: Optional[int] = None  # laps already run on the fitted set
    used_compounds: Optional[List[TireCompound]] = None  # compounds run before the fitted set
    own_gap_to_leader: Optional[float] = None  # seconds; enables competitor signals


@dataclass
class Competitor:
    driver_id: str
    tire_compound: TireCompound
    tire_age: int
    gap_to_leader: float
    current_position: Optional[int] = None  # accepted, not modelled
    gap_ahead: Optional[float] = None  # accepted, not modelled
    gap_behind: Optional[float] = None  # accepted, not modelled
    pit_stops_completed: Optional[int] = None  # accepted, not modelled
    estimated_strategy: Optional[List[Dict[str, Any]]] = None  # accepted, not modelled


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass
class StrategyOption:
    """One candidate plan for the remaining laps. Times are in seconds."""

    strategy_id: str
    tire_compounds: List[TireCompound]
    pit_laps: List[int]
    stint_breakdown: List[Dict[str, Any]]
    projected_race_time: float  # driving_time_s + pit_time_loss_s for the remaining laps
    driving_time_s: float
    pit_time_loss_s: float
    risk_level: str  # heuristic label from the stop count only
    two_compound_rule: str
    notes: List[str]
    delta_to_best_s: Optional[float] = None  # None when evaluated on its own (evaluate_plan)

    @property
    def stop_count(self) -> int:
        return len(self.pit_laps)


@dataclass
class CompetitorSignal:
    """A labelled gap/tyre-age observation from supplied data, not a simulated outcome."""

    driver_id: str
    signal: str  # "undercut_target" or "undercut_threat"
    gap_s: float  # on-track gap between you and the competitor
    tire_compound: TireCompound
    tire_age: int
    explanation: str


@dataclass
class StrategyResult:
    candidates: List[StrategyOption]  # ranked, fastest first
    current_lap: int
    total_laps: int
    remaining_laps: int
    base_lap_time: float
    driver_multiplier: float
    damage_multiplier: float
    tire_state: str  # "supplied" or "assumed_fresh"
    assumptions: List[str]
    not_modelled_inputs: List[str]
    competitor_signals: List[CompetitorSignal]
    competitor_signals_note: str
    sequences_evaluated: int
    sequences_skipped_too_few_laps: int
    sequences_excluded_two_compound_rule: int

    @property
    def best(self) -> StrategyOption:
        return self.candidates[0]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _number(name: str, value: Any, low: float, high: float, *, low_exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:  # e.g. a 400-digit integer
        raise ValueError(f"{name} must be a finite number in range, got a value too large for a float") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if number < low or number > high or (low_exclusive and number == low):
        bracket = "(" if low_exclusive else "["
        raise ValueError(f"{name} must be in {bracket}{low:g}, {high:g}], got {number:g}")
    return number


def _optional_number(name: str, value: Any, low: float, high: float) -> Optional[float]:
    return None if value is None else _number(name, value, low, high)


def _integer(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {number}")
    return number


def _enum(enum_type: type, name: str, value: Any) -> Any:
    if isinstance(value, enum_type):
        return value
    allowed = ", ".join(member.value for member in enum_type)
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    raise ValueError(f"{name} must be one of: {allowed}; got {value!r}")


def _optional_list(name: str, value: Any, max_items: int) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}")
    if len(value) > max_items:
        raise ValueError(f"{name} can contain at most {max_items} items")


def _validate_tire(key: Any, tire: Any) -> Tuple[TireCompound, TireData]:
    compound = _enum(TireCompound, "tire_data key", key)
    if not isinstance(tire, TireData):
        raise ValueError(f"tire_data[{compound.value}] must be TireData, got {type(tire).__name__}")
    own = _enum(TireCompound, f"tire_data[{compound.value}].compound", tire.compound)
    if own != compound:
        raise ValueError(
            f"tire_data key '{compound.value}' does not match its compound '{own.value}'"
        )
    prefix = f"tire_data[{compound.value}]"
    window = tire.peak_performance_window
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError(f"{prefix}.peak_performance_window must be a (start, end) pair of laps")
    start = _integer(f"{prefix}.peak_performance_window start", window[0], 1, MAX_TOTAL_LAPS)
    end = _integer(f"{prefix}.peak_performance_window end", window[1], 1, MAX_TOTAL_LAPS)
    if start > end:
        raise ValueError(f"{prefix}.peak_performance_window start ({start}) must be <= end ({end})")
    clean = TireData(
        compound=compound,
        base_performance=_number(
            f"{prefix}.base_performance", tire.base_performance, MIN_BASE_PERFORMANCE, MAX_BASE_PERFORMANCE
        ),
        degradation_rate=_number(f"{prefix}.degradation_rate", tire.degradation_rate, 0.0, MAX_DEGRADATION_RATE),
        warm_up_laps=_integer(f"{prefix}.warm_up_laps", tire.warm_up_laps, 0, MAX_WARM_UP_LAPS),
        peak_performance_window=(start, end),
        pit_stop_delta=_number(f"{prefix}.pit_stop_delta", tire.pit_stop_delta, 0.0, MAX_PIT_STOP_DELTA_S),
    )
    return compound, clean


def _validate_tire_data(tire_data: Any) -> Dict[TireCompound, TireData]:
    if not isinstance(tire_data, Mapping) or not tire_data:
        raise ValueError("tire_data must be a non-empty mapping of compound -> TireData")
    clean: Dict[TireCompound, TireData] = {}
    for key, tire in tire_data.items():
        compound, tire_clean = _validate_tire(key, tire)
        if compound in clean:
            raise ValueError(f"tire_data contains compound '{compound.value}' more than once")
        clean[compound] = tire_clean
    return clean


def _base_lap_time(telemetry: Mapping[str, Any]) -> float:
    lap_times = telemetry.get("lap_times")
    if lap_times is None:
        raise ValueError("telemetry.lap_times is required (recent lap times in seconds)")
    if not isinstance(lap_times, (list, tuple, np.ndarray)):
        raise ValueError("telemetry.lap_times must be a list of lap times in seconds")
    if not 1 <= len(lap_times) <= MAX_LAP_TIME_SAMPLES:
        raise ValueError(f"telemetry.lap_times must contain 1 to {MAX_LAP_TIME_SAMPLES} values")
    values = [
        _number(f"telemetry.lap_times[{index}]", value, MIN_LAP_TIME_S, MAX_LAP_TIME_S)
        for index, value in enumerate(lap_times)
    ]
    return float(math.fsum(values) / len(values))


def _validate_sector_times(value: Any) -> None:
    """sector_times is accepted but not modelled; it is still checked so garbage is not echoed as accepted."""

    if value is None:
        return
    name = "telemetry.sector_times"
    if isinstance(value, Mapping):
        for key in value:
            number_key = isinstance(key, Integral) and not isinstance(key, bool)
            text_key = isinstance(key, str) and key.isascii() and key.isdigit()
            if not (number_key or text_key) or not 1 <= int(key) <= MAX_SECTORS:
                raise ValueError(f"{name} keys must be sector numbers 1-{MAX_SECTORS}, got {key!r}")
        values = list(value.values())
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"{name} must be a mapping of sector number -> seconds or a list of seconds")
    if not 1 <= len(values) <= MAX_SECTORS:
        raise ValueError(f"{name} must contain 1 to {MAX_SECTORS} sector times")
    for index, seconds in enumerate(values):
        _number(f"{name}[{index}]", seconds, 0.0, MAX_LAP_TIME_S, low_exclusive=True)


def _validate_telemetry(telemetry: Any) -> Tuple[float, Optional[float], Optional[float]]:
    if not isinstance(telemetry, Mapping):
        raise ValueError("telemetry must be a mapping")
    unknown = sorted(str(key) for key in telemetry if key not in TELEMETRY_KEYS)
    if unknown:
        raise ValueError(f"Unknown telemetry keys {unknown}; allowed: {list(TELEMETRY_KEYS)}")
    _validate_sector_times(telemetry.get("sector_times"))
    return (
        _base_lap_time(telemetry),
        _optional_number("telemetry.braking_consistency", telemetry.get("braking_consistency"), 0.0, 1.0),
        _optional_number("telemetry.throttle_aggressiveness", telemetry.get("throttle_aggressiveness"), 0.0, 1.0),
    )


def _validate_car(car: Any) -> Dict[str, float]:
    if not isinstance(car, CarStatus):
        raise ValueError("car_status must be a CarStatus")
    _number("car_status.engine_wear", car.engine_wear, 0.0, 1.0)
    _number("car_status.brake_wear", car.brake_wear, 0.0, 1.0)
    _optional_number("car_status.fuel_load", car.fuel_load, 0.0, MAX_FUEL_LOAD_KG)
    _optional_number("car_status.brake_temp", car.brake_temp, 0.0, MAX_BRAKE_TEMP_C)
    _optional_number("car_status.ers_availability", car.ers_availability, 0.0, 1.0)
    damage = car.damage if car.damage is not None else {}
    if not isinstance(damage, Mapping):
        raise ValueError("car_status.damage must be a mapping of part -> damage level in [0, 1]")
    unknown = sorted(str(key) for key in damage if key not in DAMAGE_PARTS)
    if unknown:
        raise ValueError(f"Unknown car_status.damage parts {unknown}; modelled parts: {list(DAMAGE_PARTS)}")
    return {part: _number(f"car_status.damage.{part}", damage.get(part, 0.0), 0.0, 1.0) for part in DAMAGE_PARTS}


def _validate_driver(
    driver: Any, braking_override: Optional[float], throttle_override: Optional[float]
) -> Tuple[float, float]:
    """Validate the profile; return the braking and throttle values used (telemetry wins over the profile)."""

    if not isinstance(driver, DriverProfile):
        raise ValueError("driver_profile must be a DriverProfile")
    for name in ("tire_management", "risk_tolerance"):
        _number(f"driver_profile.{name}", getattr(driver, name), 0.0, 1.0)
    style = driver.overtaking_style
    if style is not None and (not isinstance(style, str) or len(style) > MAX_TEXT_LENGTH):
        raise ValueError(f"driver_profile.overtaking_style must be text of at most {MAX_TEXT_LENGTH} characters")
    used: List[float] = []
    for name, measured in zip(DRIVER_OVERRIDE_KEYS, (braking_override, throttle_override)):
        profile = _optional_number(f"driver_profile.{name}", getattr(driver, name), 0.0, 1.0)
        if measured is None and profile is None:
            raise ValueError(f"{name} is required: supply telemetry.{name} or driver_profile.{name}")
        used.append(measured if measured is not None else profile)
    return used[0], used[1]


def _validate_competition(competition: Any) -> List[Competitor]:
    if competition is None:
        return []
    if not isinstance(competition, (list, tuple)) or len(competition) > MAX_COMPETITORS:
        raise ValueError(f"competition must be a list of at most {MAX_COMPETITORS} competitors")
    clean: List[Competitor] = []
    seen = set()
    for index, item in enumerate(competition):
        name = f"competition[{index}]"
        if not isinstance(item, Competitor):
            raise ValueError(f"{name} must be a Competitor")
        if not isinstance(item.driver_id, str) or not 1 <= len(item.driver_id.strip()) <= MAX_TEXT_LENGTH:
            raise ValueError(f"{name}.driver_id must be non-empty text of at most {MAX_TEXT_LENGTH} characters")
        if item.driver_id in seen:
            raise ValueError(f"Duplicate competitor driver_id '{item.driver_id}'")
        seen.add(item.driver_id)
        if item.current_position is not None:
            _integer(f"{name}.current_position", item.current_position, 1, MAX_POSITION)
        if item.pit_stops_completed is not None:
            _integer(f"{name}.pit_stops_completed", item.pit_stops_completed, 0, MAX_PIT_STOPS_COMPLETED)
        _optional_number(f"{name}.gap_ahead", item.gap_ahead, 0.0, MAX_GAP_S)
        _optional_number(f"{name}.gap_behind", item.gap_behind, 0.0, MAX_GAP_S)
        _optional_list(f"{name}.estimated_strategy", item.estimated_strategy, MAX_LIST_ITEMS)
        clean.append(
            Competitor(
                driver_id=item.driver_id,
                tire_compound=_enum(TireCompound, f"{name}.tire_compound", item.tire_compound),
                tire_age=_integer(f"{name}.tire_age", item.tire_age, 0, MAX_TIRE_AGE_LAPS),
                gap_to_leader=_number(f"{name}.gap_to_leader", item.gap_to_leader, 0.0, MAX_GAP_S),
            )
        )
    return clean


# ---------------------------------------------------------------------------
# Heuristic lap-time model
# ---------------------------------------------------------------------------


def _driver_multiplier(driver: DriverProfile, braking: float, throttle: float) -> float:
    """Heuristic lap-time multiplier (>= 1); braking/throttle are the values already resolved from telemetry/profile."""

    multiplier = 1.0
    if braking < 0.7:
        multiplier *= 1.0 + (0.7 - braking) * 0.30
    if throttle > 0.8 and driver.tire_management < 0.6:
        multiplier *= 1.10
    if driver.risk_tolerance > 0.8:
        multiplier *= 1.03
    return float(multiplier)


def _damage_multiplier(damage: Mapping[str, float], engine_wear: float, brake_wear: float) -> float:
    """Heuristic lap-time multiplier (>= 1) for damage and wear; not a physics model."""

    multiplier = 1.0 + 0.025 * damage["front_wing"] + 0.040 * max(damage["floor"], damage["diffuser"])
    if engine_wear > 0.7:
        multiplier += 0.020 * (engine_wear - 0.7) / 0.3
    if brake_wear > 0.8:
        multiplier += 0.015 * (brake_wear - 0.8) / 0.2
    return float(multiplier)


def _weather_multiplier(weather: WeatherCondition, compound: TireCompound) -> float:
    if weather == WeatherCondition.WET:
        return {TireCompound.WET: 1.0, TireCompound.INTERMEDIATE: 0.88}.get(compound, 0.55)
    if weather == WeatherCondition.INTERMEDIATE:
        return {TireCompound.INTERMEDIATE: 1.0, TireCompound.WET: 0.92}.get(compound, 0.78)
    return {TireCompound.INTERMEDIATE: 0.78, TireCompound.WET: 0.65}.get(compound, 1.0)


def _temperature_multiplier(track_temperature: float, compound: TireCompound) -> float:
    if track_temperature > 35.0:
        return {TireCompound.SOFT: 0.97, TireCompound.HARD: 1.02}.get(compound, 1.0)
    return 1.0


def _performance_curve(
    tire: TireData, weather: WeatherCondition, track_temperature: float, laps: int
) -> np.ndarray:
    """Performance factor for tyre laps 1..laps (index 0 = the first lap on the set)."""

    tyre_lap = np.arange(1, laps + 1, dtype=float)
    base = tire.base_performance
    performance = base - np.maximum(0.0, tyre_lap - tire.peak_performance_window[1]) * tire.degradation_rate
    if tire.warm_up_laps > 0:
        warming = tyre_lap <= tire.warm_up_laps
        performance[warming] = base * (0.90 + 0.10 * tyre_lap[warming] / tire.warm_up_laps)
    performance = performance * _weather_multiplier(weather, tire.compound)
    performance = performance * _temperature_multiplier(track_temperature, tire.compound)
    return np.maximum(performance, PERFORMANCE_FLOOR)


# ---------------------------------------------------------------------------
# Per-request context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    current_lap: int
    total_laps: int
    weather: WeatherCondition
    tires: Dict[TireCompound, TireData]
    base_lap_time: float
    driver_multiplier: float
    damage_multiplier: float
    current_compound: Optional[TireCompound]
    current_tire_age: Optional[int]
    used_compounds: Optional[FrozenSet[TireCompound]]
    performance: Dict[TireCompound, np.ndarray]
    lap_times: Dict[TireCompound, np.ndarray]

    @property
    def remaining_laps(self) -> int:
        return self.total_laps - self.current_lap + 1

    @property
    def history_known(self) -> bool:
        """True when every compound run before the planned laps is known."""

        return self.used_compounds is not None or self.current_lap == 1


def _prepare(
    telemetry: Any,
    car_status: Any,
    driver_profile: Any,
    tire_data: Any,
    race_state: Any,
) -> _Context:
    base_lap_time, braking_override, throttle_override = _validate_telemetry(telemetry)
    damage = _validate_car(car_status)
    braking, throttle = _validate_driver(driver_profile, braking_override, throttle_override)
    tires = _validate_tire_data(tire_data)
    if not isinstance(race_state, RaceState):
        raise ValueError("race_state must be a RaceState")

    race = race_state
    total_laps = _integer("race_state.total_laps", race.total_laps, 1, MAX_TOTAL_LAPS)
    current_lap = _integer("race_state.current_lap", race.current_lap, 1, MAX_TOTAL_LAPS)
    if current_lap > total_laps:
        raise ValueError(f"race_state.current_lap ({current_lap}) must be <= total_laps ({total_laps})")
    weather = _enum(WeatherCondition, "race_state.weather", race.weather)
    track_temperature = _number(
        "race_state.track_temperature", race.track_temperature, MIN_TRACK_TEMPERATURE_C, MAX_TRACK_TEMPERATURE_C
    )
    for name in ("track_evolution", "safety_car_probability", "yellow_flag_risk"):
        _optional_number(f"race_state.{name}", getattr(race, name), 0.0, 1.0)
    _optional_list("race_state.weather_forecast", race.weather_forecast, MAX_LIST_ITEMS)
    _optional_number("race_state.own_gap_to_leader", race.own_gap_to_leader, 0.0, MAX_GAP_S)

    if (race.current_compound is None) != (race.current_tire_age is None):
        raise ValueError("race_state.current_compound and race_state.current_tire_age must be given together")
    current_compound: Optional[TireCompound] = None
    current_tire_age: Optional[int] = None
    if race.current_compound is not None:
        current_compound = _enum(TireCompound, "race_state.current_compound", race.current_compound)
        current_tire_age = _integer("race_state.current_tire_age", race.current_tire_age, 0, MAX_TIRE_AGE_LAPS)
        if current_compound not in tires:
            raise ValueError(
                f"race_state.current_compound '{current_compound.value}' needs an entry in tire_data"
            )

    used: Optional[FrozenSet[TireCompound]] = None
    if race.used_compounds is not None:
        _optional_list("race_state.used_compounds", race.used_compounds, len(TireCompound) * 2)
        used = frozenset(
            _enum(TireCompound, f"race_state.used_compounds[{index}]", value)
            for index, value in enumerate(race.used_compounds)
        )

    driver_mult = _driver_multiplier(driver_profile, braking, throttle)
    damage_mult = _damage_multiplier(damage, float(car_status.engine_wear), float(car_status.brake_wear))
    curve_laps = (total_laps - current_lap + 1) + (current_tire_age or 0)
    performance = {
        compound: _performance_curve(tire, weather, track_temperature, curve_laps)
        for compound, tire in tires.items()
    }
    lap_times = {
        compound: base_lap_time * driver_mult * damage_mult / curve for compound, curve in performance.items()
    }
    return _Context(
        current_lap=current_lap,
        total_laps=total_laps,
        weather=weather,
        tires=tires,
        base_lap_time=base_lap_time,
        driver_multiplier=driver_mult,
        damage_multiplier=damage_mult,
        current_compound=current_compound,
        current_tire_age=current_tire_age,
        used_compounds=used,
        performance=performance,
        lap_times=lap_times,
    )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _compound_key(compound: TireCompound) -> int:
    return list(TireCompound).index(compound)


def _allowed_compounds(ctx: _Context) -> List[TireCompound]:
    allowed = DRY_COMPOUNDS if ctx.weather == WeatherCondition.DRY else WET_WEATHER_COMPOUNDS
    return sorted((c for c in allowed if c in ctx.tires), key=_compound_key)


def _rule_status(ctx: _Context, compounds: Sequence[TireCompound]) -> str:
    """Simplified dry-race rule: two different dry compounds unless intermediate/wet tyres are used."""

    known = set(compounds) | set(ctx.used_compounds or ())
    if ctx.current_compound is not None:
        known.add(ctx.current_compound)
    if known & WET_WEATHER_COMPOUNDS:
        return RULE_WAIVED
    if len(known & DRY_COMPOUNDS) >= 2:
        return RULE_SATISFIED
    return RULE_VIOLATED if ctx.history_known else RULE_UNVERIFIED


def _risk_level(stop_count: int) -> str:
    if stop_count >= 3:
        return "high"
    return "low" if stop_count <= 1 else "medium"


def _stint(
    ctx: _Context, compound: TireCompound, start_lap: int, laps: int, age_start: int, fitted_at_stop: bool
) -> Dict[str, Any]:
    times = ctx.lap_times[compound][age_start : age_start + laps]
    performance = ctx.performance[compound][age_start : age_start + laps]
    peak_end = ctx.tires[compound].peak_performance_window[1]
    return {
        "start_lap": start_lap,
        "end_lap": start_lap + laps - 1,
        "laps": laps,
        "tire_compound": compound,
        "tire_age_start": age_start,
        "tire_age_end": age_start + laps,
        "fitted_at_stop": fitted_at_stop,
        "average_lap_time": float(np.mean(times)),
        "best_lap_time": float(np.min(times)),
        "worst_lap_time": float(np.max(times)),
        "total_time": float(np.sum(times)),
        "start_performance": float(performance[0]),
        "end_performance": float(performance[-1]),
        "laps_beyond_peak_window": int(max(0, age_start + laps - max(peak_end, age_start))),
        # Laps whose performance was clamped to PERFORMANCE_FLOOR: their times are optimistic.
        "laps_at_performance_floor": int(np.count_nonzero(performance <= PERFORMANCE_FLOOR)),
    }


def _describe(ctx: _Context, stints: Sequence[Dict[str, Any]]) -> str:
    first = stints[0]["tire_compound"].value
    if len(stints) == 1:
        if ctx.current_compound is not None:
            return f"No further stop: run the current {first} tyres to the flag (lap {ctx.total_laps})."
        return f"No further stop: one {first} stint to the flag (lap {ctx.total_laps})."
    stops = len(stints) - 1
    legs = " -> ".join(f"{s['tire_compound'].value} to lap {s['end_lap']}" for s in stints)
    return f"{stops} stop{'s' if stops > 1 else ''}: {legs}."


def _tyre_state_note(ctx: _Context, first: TireCompound) -> str:
    if ctx.current_compound is not None:
        return (
            f"First stint continues on the fitted {first.value} tyres from age {ctx.current_tire_age} "
            "laps with no pit-stop cost."
        )
    return (
        f"Assumes a fresh set of {first.value} tyres at lap {ctx.current_lap} with no pit-stop cost, "
        "because the current tyre compound/age was not supplied."
    )


def _build_option(ctx: _Context, compounds: Sequence[TireCompound], lengths: Sequence[int]) -> StrategyOption:
    stints: List[Dict[str, Any]] = []
    cursor = ctx.current_lap
    for index, (compound, laps) in enumerate(zip(compounds, lengths)):
        age = (ctx.current_tire_age or 0) if index == 0 else 0
        stints.append(_stint(ctx, compound, cursor, int(laps), age, fitted_at_stop=index > 0))
        cursor += int(laps)

    driving = float(math.fsum(stint["total_time"] for stint in stints))
    pit_loss = float(math.fsum(ctx.tires[compound].pit_stop_delta for compound in compounds[1:]))
    pit_laps = [stint["end_lap"] for stint in stints[:-1]]
    rule = _rule_status(ctx, compounds)
    stops = len(pit_laps)
    notes = [
        _describe(ctx, stints),
        _tyre_state_note(ctx, compounds[0]),
        _RULE_NOTES[rule],
        "Projected times are heuristic estimates, not a calibrated race simulation.",
    ]
    floored = sum(stint["laps_at_performance_floor"] for stint in stints)
    if floored:
        notes.append(
            f"{floored} projected lap(s) hit the {PERFORMANCE_FLOOR:.0%} tyre-performance floor; the heuristic "
            "does not model wear beyond it, so those lap times are optimistic."
        )
    return StrategyOption(
        strategy_id=f"{stops}-stop:" + "/".join(compound.value for compound in compounds),
        tire_compounds=list(compounds),
        pit_laps=pit_laps,
        stint_breakdown=stints,
        projected_race_time=driving + pit_loss,
        driving_time_s=driving,
        pit_time_loss_s=pit_loss,
        risk_level=_risk_level(stops),
        two_compound_rule=rule,
        notes=notes,
    )


def _stint_table(ctx: _Context, compound: TireCompound, age_start: int) -> np.ndarray:
    """table[n] = time of an n-lap stint starting at tyre age age_start (table[0] = inf: stints are >= 1 lap)."""

    remaining = ctx.remaining_laps
    cumulative = np.concatenate(([0.0], np.cumsum(ctx.lap_times[compound])))
    table = cumulative[age_start : age_start + remaining + 1] - cumulative[age_start]
    table[0] = np.inf
    return table


def _optimal_lengths(
    first: np.ndarray, fresh: Sequence[np.ndarray], offsets: np.ndarray, valid: np.ndarray
) -> Optional[List[int]]:
    """Exact min-plus dynamic programme: stint lengths (each >= 1) summing to the remaining laps."""

    remaining = len(first) - 1
    if len(fresh) + 1 > remaining:
        return None
    best = first
    rows = np.arange(remaining + 1)
    choices: List[np.ndarray] = []
    for table in fresh:
        totals = np.where(valid, best[offsets] + table[None, :], np.inf)
        choice = np.argmin(totals, axis=1)
        best = totals[rows, choice]
        choices.append(choice)
    if not np.isfinite(best[remaining]):
        return None
    lengths: List[int] = []
    laps_left = remaining
    for choice in reversed(choices):
        laps = int(choice[laps_left])
        lengths.append(laps)
        laps_left -= laps
    lengths.append(laps_left)
    lengths.reverse()
    return lengths


def _sequences(ctx: _Context) -> List[Tuple[TireCompound, ...]]:
    """Distinct compound sequences with 0..MAX_STOPS further stops (equivalent orderings merged)."""

    allowed = _allowed_compounds(ctx)
    sequences: List[Tuple[TireCompound, ...]] = []
    if ctx.current_compound is not None:
        for stops in range(MAX_STOPS + 1):
            for rest in combinations_with_replacement(allowed, stops):
                sequences.append((ctx.current_compound, *rest))
        return sequences
    for size in range(1, MAX_STOPS + 2):
        for group in combinations_with_replacement(allowed, size):
            # The first (assumed fresh) stint carries no pit cost, so start on the compound whose
            # pit_stop_delta is largest; ties keep the canonical soft -> wet order.
            first = max(group, key=lambda c: (ctx.tires[c].pit_stop_delta, -_compound_key(c)))
            rest = list(group)
            rest.remove(first)
            sequences.append((first, *rest))
    return sequences


def _select(ranked: List[StrategyOption]) -> List[StrategyOption]:
    """Fastest plan for each stop count, then the next fastest plans up to MAX_CANDIDATES."""

    chosen: List[StrategyOption] = []
    stop_counts = set()
    for option in ranked:
        if option.stop_count not in stop_counts:
            stop_counts.add(option.stop_count)
            chosen.append(option)
    chosen_ids = {option.strategy_id for option in chosen}
    for option in ranked:
        if len(chosen) >= MAX_CANDIDATES:
            break
        if option.strategy_id not in chosen_ids:
            chosen.append(option)
            chosen_ids.add(option.strategy_id)
    return sorted(chosen, key=_rank_key)


def _rank_key(option: StrategyOption) -> Tuple[float, int, str]:
    return (option.projected_race_time, option.stop_count, option.strategy_id)


# ---------------------------------------------------------------------------
# Response-level annotations
# ---------------------------------------------------------------------------


def _competitor_signals(
    ctx: _Context, competition: Sequence[Competitor], own_gap: Optional[float]
) -> Tuple[List[CompetitorSignal], str]:
    if not competition:
        return [], "No competitors supplied."
    if own_gap is None:
        return [], (
            "Not computed: race_state.own_gap_to_leader was not supplied, so competitor gaps "
            "cannot be related to your car."
        )
    signals: List[CompetitorSignal] = []
    not_assessed: List[str] = []
    for rival in competition:
        relative = rival.gap_to_leader - own_gap  # < 0: rival ahead of you, 0: level on gap_to_leader
        gap = abs(relative)
        if gap > UNDERCUT_WINDOW_S:
            continue
        tyres = f"{rival.tire_compound.value} tyres aged {rival.tire_age} laps"
        if relative < 0:
            tire = ctx.tires.get(rival.tire_compound)
            if tire is None:
                not_assessed.append(f"{rival.driver_id} ({rival.tire_compound.value})")
                continue
            if rival.tire_age <= tire.peak_performance_window[1]:
                continue
            explanation = (
                f"{rival.driver_id} is {gap:.1f}s ahead on {tyres}, past the end of the supplied "
                f"{rival.tire_compound.value} peak window (lap {tire.peak_performance_window[1]}). Pitting "
                "before them may gain the position; the undercut itself is not simulated."
            )
            signals.append(
                CompetitorSignal(rival.driver_id, "undercut_target", gap, rival.tire_compound, rival.tire_age, explanation)
            )
        else:
            where = "level with you on gap_to_leader" if gap == 0.0 else f"{gap:.1f}s behind"
            explanation = (
                f"{rival.driver_id} is {where} on {tyres}; a car this close could pit first "
                "and try to undercut you. The undercut itself is not simulated."
            )
            signals.append(
                CompetitorSignal(rival.driver_id, "undercut_threat", gap, rival.tire_compound, rival.tire_age, explanation)
            )
    signals.sort(key=lambda s: (s.signal, s.gap_s, s.driver_id))
    note = (
        f"Computed from supplied gap_to_leader values relative to race_state.own_gap_to_leader. "
        f"undercut_target: a car up to {UNDERCUT_WINDOW_S:.1f}s ahead whose tyre age is past the end of its "
        f"compound's peak_performance_window in tire_data. undercut_threat: a car level with you or up to "
        f"{UNDERCUT_WINDOW_S:.1f}s behind. These are labels from supplied data, not simulated outcomes, and they "
        "do not change the ranking."
    )
    if not_assessed:
        note += (
            " Not assessed as undercut targets, because their compound has no tire_data entry (so its peak "
            f"window is unknown): {', '.join(not_assessed)}."
        )
    return signals, note


def _not_modelled(
    telemetry: Mapping[str, Any],
    car: CarStatus,
    driver: DriverProfile,
    race: RaceState,
    competition: Sequence[Competitor],
) -> List[str]:
    """Inputs that were supplied but do not affect any number in the response."""

    names = ["telemetry.sector_times"] if telemetry.get("sector_times") is not None else []
    names += [f"car_status.{name}" for name in ("fuel_load", "brake_temp", "ers_availability") if getattr(car, name) is not None]
    for name in DRIVER_OVERRIDE_KEYS:
        if telemetry.get(name) is not None and getattr(driver, name) is not None:
            names.append(f"driver_profile.{name} (overridden by telemetry.{name})")
    if driver.overtaking_style is not None:
        names.append("driver_profile.overtaking_style")
    names += [
        f"race_state.{name}"
        for name in ("track_evolution", "safety_car_probability", "yellow_flag_risk", "weather_forecast")
        if getattr(race, name) is not None
    ]
    names.append("tire_data.*.peak_performance_window start (only the window end changes lap times)")
    for name in ("current_position", "gap_ahead", "gap_behind", "pit_stops_completed", "estimated_strategy"):
        if any(getattr(rival, name) is not None for rival in competition):
            names.append(f"competition[].{name}")
    if competition and race.own_gap_to_leader is None:
        names.append("competition (needs race_state.own_gap_to_leader to be related to your car)")
    return names


def _rule_fallback_reason(ctx: _Context) -> str:
    """Why no candidate satisfies the simplified two-compound rule (called only in that case)."""

    usable = set(_allowed_compounds(ctx))
    known = set(ctx.used_compounds or ()) | ({ctx.current_compound} if ctx.current_compound is not None else set())
    dry_options = (usable | known) & DRY_COMPOUNDS
    if not usable & WET_WEATHER_COMPOUNDS and len(dry_options) < 2:
        names = ", ".join(c.value for c in sorted(dry_options, key=_compound_key)) or "none"
        return (
            f"only one dry compound ({names}) is known or available: tire_data offers no second dry compound "
            f"(and no intermediate/wet tyre) for a stop in {ctx.weather.value} conditions, so no plan can use "
            "two different dry compounds"
        )
    # A second compound is available, so the only obstacle is that no stop fits.
    return (
        f"no stop fits in the remaining {ctx.remaining_laps} lap(s) (a stop needs at least one lap before and "
        "one lap after it)"
    )


def _assumptions(ctx: _Context, fallback_reason: Optional[str]) -> List[str]:
    allowed = ", ".join(c.value for c in _allowed_compounds(ctx)) or "none available in tire_data"
    notes = [
        (
            f"Heuristic model: lap time = mean(telemetry.lap_times) {ctx.base_lap_time:.3f}s x driver multiplier "
            f"{ctx.driver_multiplier:.4f} x damage/wear multiplier {ctx.damage_multiplier:.4f} / tyre performance "
            "factor. It is not a calibrated race simulator."
        ),
        (
            "Tyre performance: warm-up ramp from 90% over warm_up_laps, base_performance until the end of "
            f"peak_performance_window, then a linear loss of degradation_rate per lap, floored at {PERFORMANCE_FLOOR:.0%} "
            "(laps at the floor are counted in laps_at_performance_floor and are optimistic)."
        ),
        (
            f"Stint lengths are optimised exactly (dynamic programming) for every compound sequence with up to "
            f"{MAX_STOPS} stops. Stint order after the first stint does not change the projected time in this "
            "model, so equivalent orderings are merged."
        ),
        (
            f"The earliest possible stop is at the end of lap {ctx.current_lap}; a pit lap is the last lap of the "
            "stint before the stop. Each stop costs the pit_stop_delta of the compound fitted at that stop."
        ),
        (
            f"Weather is assumed to stay {ctx.weather.value} for the remaining laps; stops fit only these "
            f"compounds: {allowed}."
        ),
        (
            "Two-compound rule (simplified): in a race without intermediate/wet tyres at least two different dry "
            "compounds must be used; tyre-set allocation and other sporting-regulation details are not modelled."
        ),
    ]
    if ctx.current_compound is not None:
        notes.append(
            f"Tyre state supplied: the first stint continues on the fitted {ctx.current_compound.value} set from "
            f"age {ctx.current_tire_age} laps; changing tyres, even at the end of this lap, costs a stop."
        )
    else:
        notes.append(
            f"Tyre state not supplied: every candidate starts a fresh stint at lap {ctx.current_lap} with no "
            "pit-stop cost."
        )
    if ctx.damage_multiplier > 1.0:
        notes.append(f"Lap times include a heuristic damage/wear penalty of {ctx.damage_multiplier - 1.0:.2%}.")
    if fallback_reason is not None:
        notes.append(
            f"No plan satisfies the simplified two-dry-compound rule, because {fallback_reason}; the candidates "
            "shown violate it."
        )
    notes.append(
        f"Returned: the fastest plan for each stop count, then the next fastest plans, up to {MAX_CANDIDATES} "
        "candidates. risk_level is a label from the stop count only."
    )
    return notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_strategy(
    telemetry: Mapping[str, Any],
    car_status: CarStatus,
    driver_profile: DriverProfile,
    tire_data: Mapping[TireCompound, TireData],
    race_state: RaceState,
    competition: Sequence[Competitor] = (),
) -> StrategyResult:
    """Rank heuristic strategy candidates for laps current_lap..total_laps.

    Raises ValueError for missing, non-finite or out-of-range inputs, or when no candidate
    can be built (no weather-appropriate compound in tire_data and no current tyre supplied).
    """

    ctx = _prepare(telemetry, car_status, driver_profile, tire_data, race_state)
    rivals = _validate_competition(competition)

    remaining = ctx.remaining_laps
    offsets = np.arange(remaining + 1)[:, None] - np.arange(remaining + 1)[None, :]
    valid = offsets >= 0
    offsets = np.where(valid, offsets, 0)
    fresh_tables = {compound: _stint_table(ctx, compound, 0) for compound in ctx.tires}

    options: List[StrategyOption] = []
    sequences = _sequences(ctx)
    skipped = 0
    for sequence in sequences:
        first_age = ctx.current_tire_age or 0
        first = fresh_tables[sequence[0]] if first_age == 0 else _stint_table(ctx, sequence[0], first_age)
        lengths = _optimal_lengths(first, [fresh_tables[c] for c in sequence[1:]], offsets, valid)
        if lengths is None:
            skipped += 1
            continue
        options.append(_build_option(ctx, sequence, lengths))

    if not options:
        raise ValueError(
            f"No strategy could be generated: tire_data has no compound usable in {ctx.weather.value} "
            "conditions and no current tyre was supplied"
        )

    ranked = sorted(options, key=_rank_key)
    compliant = [option for option in ranked if option.two_compound_rule != RULE_VIOLATED]
    fallback_reason = None if compliant else _rule_fallback_reason(ctx)
    candidates = _select(compliant or ranked)
    best_time = candidates[0].projected_race_time
    for option in candidates:
        option.delta_to_best_s = option.projected_race_time - best_time
        if fallback_reason is not None:
            option.notes.append(f"Shown although it violates the simplified rule, because {fallback_reason}.")

    own_gap = race_state.own_gap_to_leader
    signals, signals_note = _competitor_signals(ctx, rivals, None if own_gap is None else float(own_gap))
    return StrategyResult(
        candidates=candidates,
        current_lap=ctx.current_lap,
        total_laps=ctx.total_laps,
        remaining_laps=remaining,
        base_lap_time=ctx.base_lap_time,
        driver_multiplier=ctx.driver_multiplier,
        damage_multiplier=ctx.damage_multiplier,
        tire_state="supplied" if ctx.current_compound is not None else "assumed_fresh",
        assumptions=_assumptions(ctx, fallback_reason),
        not_modelled_inputs=_not_modelled(telemetry, car_status, driver_profile, race_state, competition or ()),
        competitor_signals=signals,
        competitor_signals_note=signals_note,
        sequences_evaluated=len(sequences),
        sequences_skipped_too_few_laps=skipped,
        sequences_excluded_two_compound_rule=len(ranked) - len(compliant) if compliant else 0,
    )


def evaluate_plan(
    compounds: Sequence[TireCompound],
    stint_lengths: Sequence[int],
    telemetry: Mapping[str, Any],
    car_status: CarStatus,
    driver_profile: DriverProfile,
    tire_data: Mapping[TireCompound, TireData],
    race_state: RaceState,
) -> StrategyOption:
    """Project one user-specified plan with the same heuristic model ("what if I pit on lap X?").

    compounds[0] is the first stint's tyre (it must equal race_state.current_compound when that is
    supplied); stint_lengths must be >= 1 each and sum to the remaining laps. delta_to_best_s is None.
    """

    ctx = _prepare(telemetry, car_status, driver_profile, tire_data, race_state)
    if not isinstance(compounds, (list, tuple)) or not isinstance(stint_lengths, (list, tuple)):
        raise ValueError("compounds and stint_lengths must be lists")
    if not 1 <= len(compounds) <= MAX_STOPS + 1 or len(compounds) != len(stint_lengths):
        raise ValueError(f"A plan needs 1 to {MAX_STOPS + 1} stints with one length per compound")
    plan = [_enum(TireCompound, f"compounds[{i}]", c) for i, c in enumerate(compounds)]
    for compound in plan:
        if compound not in ctx.tires:
            raise ValueError(f"Compound '{compound.value}' has no entry in tire_data")
    if ctx.current_compound is not None and plan[0] != ctx.current_compound:
        raise ValueError(
            f"The first stint must continue on the fitted '{ctx.current_compound.value}' tyres; "
            "changing tyres at the end of this lap is a stop"
        )
    lengths = [
        _integer(f"stint_lengths[{i}]", laps, 1, ctx.remaining_laps) for i, laps in enumerate(stint_lengths)
    ]
    if sum(lengths) != ctx.remaining_laps:
        raise ValueError(f"stint_lengths must sum to the remaining laps ({ctx.remaining_laps}), got {sum(lengths)}")
    return _build_option(ctx, plan, lengths)
