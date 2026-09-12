from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class StrategyOption:
    """A complete race strategy option."""

    strategy_id: str
    estimated_total_time: float
    confidence_score: float
    stint_breakdown: List[Dict[str, Any]]
    tire_compounds: List[TireCompound]
    pit_laps: List[int]
    projected_race_time: float
    undercut_opportunities: List[Dict[str, Any]]
    overcut_opportunities: List[Dict[str, Any]]
    notes: List[str]
    risk_level: str


@dataclass
class DriverProfile:
    tire_management: float
    risk_tolerance: float
    overtaking_style: str
    braking_consistency: float
    throttle_aggressiveness: float


@dataclass
class CarStatus:
    damage: Dict[str, float]
    fuel_load: float
    brake_temp: float
    engine_wear: float
    ers_availability: float
    brake_wear: float


@dataclass
class TireData:
    compound: TireCompound
    base_performance: float
    degradation_rate: float
    warm_up_laps: int
    peak_performance_window: Tuple[int, int]
    pit_stop_delta: float


@dataclass
class RaceState:
    current_lap: int
    total_laps: int
    weather: WeatherCondition
    track_temperature: float
    track_evolution: float
    safety_car_probability: float
    yellow_flag_risk: float
    weather_forecast: List[Dict[str, Any]]


@dataclass
class Competitor:
    driver_id: str
    current_position: int
    tire_compound: TireCompound
    tire_age: int
    gap_to_leader: float
    gap_ahead: float
    gap_behind: float
    pit_stops_completed: int
    estimated_strategy: List[Dict[str, Any]]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def evaluate_driver_penalty(driver_profile: DriverProfile, telemetry: Dict[str, Any]) -> float:
    """Return a small lap-time multiplier derived from driver/telemetry characteristics."""

    penalty = 1.0
    braking_consistency = float(
        telemetry.get("braking_consistency", driver_profile.braking_consistency)
    )
    throttle_aggressiveness = float(
        telemetry.get("throttle_aggressiveness", driver_profile.throttle_aggressiveness)
    )

    if braking_consistency < 0.7:
        penalty *= 1.0 + (0.7 - braking_consistency) * 0.30
    if throttle_aggressiveness > 0.8 and driver_profile.tire_management < 0.6:
        penalty *= 1.10
    if driver_profile.risk_tolerance > 0.8:
        penalty *= 1.03

    return float(max(1.0, penalty))


def _damage_time_multiplier(car_status: CarStatus) -> float:
    """Approximate performance cost of damage/wear without pretending to be a physics model."""

    front_wing = _clip01(car_status.damage.get("front_wing", 0.0))
    floor = _clip01(car_status.damage.get("floor", 0.0))
    diffuser = _clip01(car_status.damage.get("diffuser", 0.0))
    engine_wear = _clip01(car_status.engine_wear)
    brake_wear = _clip01(car_status.brake_wear)

    aero_damage = max(floor, diffuser)
    multiplier = 1.0
    multiplier += 0.025 * front_wing
    multiplier += 0.040 * aero_damage
    if engine_wear > 0.7:
        multiplier += 0.020 * (engine_wear - 0.7) / 0.3
    if brake_wear > 0.8:
        multiplier += 0.015 * (brake_wear - 0.8) / 0.2
    return float(multiplier)


def adjust_strategy_for_damage(car_status: CarStatus, base_strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Return an annotated copy of a strategy with damage-aware notes/settings."""

    adjusted = dict(base_strategy)
    adjusted["damage_time_multiplier"] = _damage_time_multiplier(car_status)

    if car_status.engine_wear > 0.7:
        adjusted["preferred_compounds"] = [TireCompound.HARD, TireCompound.MEDIUM]
    if car_status.brake_wear > 0.8:
        adjusted["braking_penalty"] = 1.1
    return adjusted


def _weather_multiplier(weather: WeatherCondition, compound: TireCompound) -> float:
    if weather == WeatherCondition.WET:
        if compound == TireCompound.WET:
            return 1.0
        if compound == TireCompound.INTERMEDIATE:
            return 0.88
        return 0.55
    if weather == WeatherCondition.INTERMEDIATE:
        if compound == TireCompound.INTERMEDIATE:
            return 1.0
        if compound == TireCompound.WET:
            return 0.92
        return 0.78
    if compound in (TireCompound.WET, TireCompound.INTERMEDIATE):
        return 0.78 if compound == TireCompound.INTERMEDIATE else 0.65
    return 1.0


def simulate_stint(
    start_lap: int,
    laps: int,
    tire_compound: TireCompound,
    tire_data: Dict[TireCompound, TireData],
    driver_profile: DriverProfile,
    race_state: RaceState,
    telemetry: Dict[str, Any],
    car_status: Optional[CarStatus] = None,
) -> Dict[str, Any]:
    """Simulate one stint with a transparent heuristic lap-time model."""

    if laps <= 0:
        return {"error": "Stint length must be positive"}
    if tire_compound not in tire_data:
        return {"error": f"Tire compound {tire_compound.value} not found in tire_data"}

    tire_info = tire_data[tire_compound]
    if tire_info.base_performance <= 0:
        return {"error": "base_performance must be positive"}

    warm_up_laps = max(1, int(tire_info.warm_up_laps))
    peak_start, peak_end = tire_info.peak_performance_window
    peak_start = max(1, int(peak_start))
    peak_end = max(peak_start, int(peak_end))

    base_lap_time = float(np.mean(telemetry.get("lap_times", [80.0])))
    if not np.isfinite(base_lap_time) or base_lap_time <= 0:
        base_lap_time = 80.0

    driver_penalty = evaluate_driver_penalty(driver_profile, telemetry)
    damage_multiplier = _damage_time_multiplier(car_status) if car_status else 1.0
    lap_times: List[float] = []
    tire_performance: List[float] = []

    for stint_lap in range(1, laps + 1):
        if stint_lap <= warm_up_laps:
            # Start close to peak rather than from zero performance.
            warmup_fraction = stint_lap / warm_up_laps
            performance = tire_info.base_performance * (0.90 + 0.10 * warmup_fraction)
        elif peak_start <= stint_lap <= peak_end:
            performance = tire_info.base_performance
        else:
            laps_over_peak = max(0, stint_lap - peak_end)
            performance = tire_info.base_performance - laps_over_peak * tire_info.degradation_rate

        performance *= _weather_multiplier(race_state.weather, tire_compound)

        if race_state.track_temperature > 35:
            if tire_compound == TireCompound.SOFT:
                performance *= 0.97
            elif tire_compound == TireCompound.HARD:
                performance *= 1.02

        performance = float(max(0.20, performance))
        lap_time = base_lap_time * driver_penalty * damage_multiplier / performance
        lap_times.append(float(lap_time))
        tire_performance.append(performance)

    return {
        "start_lap": start_lap,
        "end_lap": start_lap + laps - 1,
        "laps": laps,
        "tire_compound": tire_compound,
        "average_lap_time": float(np.mean(lap_times)),
        "best_lap_time": float(min(lap_times)),
        "worst_lap_time": float(max(lap_times)),
        "tire_performance_trend": tire_performance,
        "total_time": float(sum(lap_times)),
        "tire_wear_at_end": float(max(0.0, 1.0 - min(tire_performance))),
    }


def _allocate_stint_lengths(total_laps: int, compounds: List[TireCompound]) -> List[int]:
    """Allocate all remaining laps across exactly one stint per compound."""

    if total_laps <= 0:
        raise ValueError("No laps remain to simulate")
    if not compounds:
        raise ValueError("At least one tyre compound is required")
    if total_laps < len(compounds):
        raise ValueError("Not enough remaining laps for the requested number of stints")

    durability = {
        TireCompound.SOFT: 0.80,
        TireCompound.MEDIUM: 1.00,
        TireCompound.HARD: 1.20,
        TireCompound.INTERMEDIATE: 1.00,
        TireCompound.WET: 1.00,
    }
    weights = np.array([durability[c] for c in compounds], dtype=float)
    raw = total_laps * weights / weights.sum()
    lengths = np.floor(raw).astype(int)
    lengths = np.maximum(lengths, 1)

    while int(lengths.sum()) < total_laps:
        idx = int(np.argmax(raw - lengths))
        lengths[idx] += 1
    while int(lengths.sum()) > total_laps:
        candidates = [i for i, value in enumerate(lengths) if value > 1]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: lengths[i] - raw[i])
        lengths[idx] -= 1

    return [int(x) for x in lengths]


def _strategy_variations(race_state: RaceState) -> List[Dict[str, Any]]:
    if race_state.weather == WeatherCondition.WET:
        return [
            {"name": "Wet 1-Stop", "compounds": [TireCompound.WET, TireCompound.WET]},
            {"name": "Wet-to-Intermediate", "compounds": [TireCompound.WET, TireCompound.INTERMEDIATE]},
            {"name": "Wet 2-Stop", "compounds": [TireCompound.WET, TireCompound.WET, TireCompound.INTERMEDIATE]},
        ]
    if race_state.weather == WeatherCondition.INTERMEDIATE:
        return [
            {"name": "Intermediate 1-Stop", "compounds": [TireCompound.INTERMEDIATE, TireCompound.INTERMEDIATE]},
            {"name": "Intermediate-to-Dry", "compounds": [TireCompound.INTERMEDIATE, TireCompound.MEDIUM]},
            {"name": "Intermediate 2-Stop", "compounds": [TireCompound.INTERMEDIATE, TireCompound.INTERMEDIATE, TireCompound.MEDIUM]},
        ]
    return [
        {"name": "Aggressive 2-Stop", "compounds": [TireCompound.SOFT, TireCompound.SOFT, TireCompound.MEDIUM]},
        {"name": "Conservative 1-Stop", "compounds": [TireCompound.MEDIUM, TireCompound.HARD]},
        {"name": "Balanced 2-Stop", "compounds": [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.MEDIUM]},
        {"name": "Ultra-Aggressive 3-Stop", "compounds": [TireCompound.SOFT, TireCompound.SOFT, TireCompound.SOFT, TireCompound.MEDIUM]},
        {"name": "Conservative 2-Stop", "compounds": [TireCompound.MEDIUM, TireCompound.HARD, TireCompound.HARD]},
    ]


def _risk_level(stop_count: int) -> str:
    if stop_count >= 3:
        return "high"
    if stop_count <= 1:
        return "low"
    return "medium"


def score_strategy_option(
    strategy: Dict[str, Any],
    race_state: RaceState,
    competition: List[Competitor],
    driver_profile: DriverProfile,
) -> float:
    """Produce a ranking score. This is a heuristic, not a calibrated probability."""

    projected_time = float(strategy.get("projected_race_time", np.inf))
    score = 0.0 if not np.isfinite(projected_time) else 100000.0 / projected_time

    for pit_lap in strategy.get("pit_laps", []):
        if race_state.safety_car_probability > 0.3 and 15 <= pit_lap <= 35:
            score += 3.0

    if competition and float(np.mean([c.gap_to_leader for c in competition])) < 10.0:
        score += 2.0

    risk = strategy.get("risk_level", "medium")
    if risk == "high" and driver_profile.risk_tolerance > 0.7:
        score += 1.5
    if risk == "low" and driver_profile.risk_tolerance < 0.4:
        score += 1.5
    return float(score)


def _opportunities(competition: List[Competitor]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    undercuts: List[Dict[str, Any]] = []
    overcuts: List[Dict[str, Any]] = []
    for competitor in competition:
        if competitor.tire_age > 15 and competitor.gap_ahead <= 3.0:
            undercuts.append(
                {
                    "competitor": competitor.driver_id,
                    "opportunity_type": "undercut",
                    "reason": f"{competitor.driver_id} is on {competitor.tire_age}-lap-old {competitor.tire_compound.value} tyres",
                }
            )
        if competitor.gap_ahead < 2.0 and competitor.tire_age < 10:
            overcuts.append(
                {
                    "competitor": competitor.driver_id,
                    "opportunity_type": "overcut",
                    "reason": f"Track position is tight ({competitor.gap_ahead:.1f}s gap) while {competitor.driver_id} is on relatively fresh tyres",
                }
            )
    return undercuts, overcuts


def generate_strategy(
    telemetry: Dict[str, Any],
    car_status: CarStatus,
    driver_profile: DriverProfile,
    tire_data: Dict[TireCompound, TireData],
    race_state: RaceState,
    competition: List[Competitor],
) -> List[StrategyOption]:
    """Generate and rank candidate strategies for the remaining race distance."""

    if race_state.total_laps < 1:
        raise ValueError("total_laps must be positive")
    if not 1 <= race_state.current_lap <= race_state.total_laps:
        raise ValueError("current_lap must be between 1 and total_laps")
    if not tire_data:
        raise ValueError("tire_data cannot be empty")

    remaining_laps = race_state.total_laps - race_state.current_lap + 1
    candidates: List[Tuple[float, StrategyOption]] = []
    damage = adjust_strategy_for_damage(car_status, {})

    for index, variation in enumerate(_strategy_variations(race_state), start=1):
        compounds = variation["compounds"]
        if any(compound not in tire_data for compound in compounds):
            continue

        stint_lengths = _allocate_stint_lengths(remaining_laps, compounds)
        stints: List[Dict[str, Any]] = []
        lap_cursor = race_state.current_lap
        driving_time = 0.0

        for stint_length, compound in zip(stint_lengths, compounds):
            stint = simulate_stint(
                start_lap=lap_cursor,
                laps=stint_length,
                tire_compound=compound,
                tire_data=tire_data,
                driver_profile=driver_profile,
                race_state=race_state,
                telemetry=telemetry,
                car_status=car_status,
            )
            if "error" in stint:
                raise ValueError(stint["error"])
            stints.append(stint)
            driving_time += stint["total_time"]
            lap_cursor += stint_length

        stop_count = len(compounds) - 1
        pit_delta_values = [tire_data[c].pit_stop_delta for c in compounds[1:]]
        pit_time = float(sum(pit_delta_values)) if pit_delta_values else 0.0
        total_time = float(driving_time + pit_time)

        pit_laps: List[int] = []
        cumulative = race_state.current_lap
        for stint in stints[:-1]:
            cumulative += int(stint["laps"])
            pit_laps.append(cumulative - 1)

        risk = _risk_level(stop_count)
        undercuts, overcuts = _opportunities(competition)
        strategy_dict = {
            "projected_race_time": total_time,
            "pit_laps": pit_laps,
            "risk_level": risk,
            "stints": stints,
        }
        score = score_strategy_option(strategy_dict, race_state, competition, driver_profile)

        data_completeness = 0.70
        if telemetry.get("lap_times"):
            data_completeness += 0.08
        if competition:
            data_completeness += 0.07
        if race_state.weather_forecast:
            data_completeness += 0.05
        confidence = float(min(0.90, data_completeness))

        notes = [
            f"{variation['name']} strategy",
            f"Estimated {stop_count} pit stop{'s' if stop_count != 1 else ''}",
            f"Risk level: {risk}",
            "Projected times are heuristic estimates, not a calibrated race simulator.",
        ]
        if damage.get("damage_time_multiplier", 1.0) > 1.01:
            notes.append("Projected lap times include a damage/wear performance penalty.")

        option = StrategyOption(
            strategy_id=f"strategy_{index}",
            estimated_total_time=total_time,
            confidence_score=confidence,
            stint_breakdown=stints,
            tire_compounds=compounds,
            pit_laps=pit_laps,
            projected_race_time=total_time,
            undercut_opportunities=undercuts,
            overcut_opportunities=overcuts,
            notes=notes,
            risk_level=risk,
        )
        candidates.append((score, option))

    if not candidates:
        raise ValueError("No strategy could be generated with the supplied tyre data")

    # Race time remains the primary ranking criterion; score is a stable tie-breaker.
    candidates.sort(key=lambda pair: (pair[1].projected_race_time, -pair[0]))
    return [option for _, option in candidates]


def dummy_strategy() -> str:
    return "Strategy module ready"
