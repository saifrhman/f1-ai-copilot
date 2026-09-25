#!/usr/bin/env python3
"""Runnable example for the heuristic strategy engine.

Run from the repository root:
    python -m core_modules.strategy_optimizer.example_usage
"""

import json

from core_modules.strategy_optimizer.schemas import StrategyRequest, generate_strategy_response
from core_modules.strategy_optimizer.strategy_engine import (
    CarStatus,
    Competitor,
    DriverProfile,
    RaceState,
    StrategyResult,
    TireCompound,
    TireData,
    WeatherCondition,
    generate_strategy,
)

# Measured braking consistency comes from telemetry, so the profile does not need to repeat it
# (a profile value would be overridden and reported in not_modelled_inputs).
TELEMETRY = {
    "lap_times": [95.6, 95.3, 95.9, 95.4],
    "braking_consistency": 0.75,
    "sector_times": {"1": 30.1, "2": 38.4, "3": 27.0},  # accepted, reported as not modelled
}
CAR = CarStatus(engine_wear=0.3, brake_wear=0.4, damage={"front_wing": 0.1})
DRIVER = DriverProfile(tire_management=0.7, risk_tolerance=0.6, throttle_aggressiveness=0.7)
TYRES = {
    TireCompound.SOFT: TireData(TireCompound.SOFT, 1.000, 0.0040, 2, (2, 10), 22.0),
    TireCompound.MEDIUM: TireData(TireCompound.MEDIUM, 0.992, 0.0025, 3, (3, 18), 22.0),
    TireCompound.HARD: TireData(TireCompound.HARD, 0.985, 0.0015, 4, (4, 28), 22.0),
}
RIVALS = [
    Competitor("HAM", TireCompound.MEDIUM, tire_age=21, gap_to_leader=4.4),
    Competitor("VER", TireCompound.HARD, tire_age=5, gap_to_leader=8.1),
]


def show(title: str, result: StrategyResult, top: int = 3) -> None:
    print(f"\n=== {title} (laps {result.current_lap}-{result.total_laps}, tyre state: {result.tire_state})")
    for option in result.candidates[:top]:
        print(
            f"  {option.strategy_id:28s} {option.projected_race_time:8.1f}s  +{option.delta_to_best_s:5.1f}s  "
            f"pit laps {option.pit_laps}  rule: {option.two_compound_rule}"
        )
    print(f"  best: {result.best.notes[0]}")
    for signal in result.competitor_signals:
        print(f"  {signal.signal}: {signal.explanation}")
    if result.not_modelled_inputs:
        print(f"  accepted but not modelled: {', '.join(result.not_modelled_inputs)}")


def main() -> None:
    mid_race = RaceState(
        current_lap=18,
        total_laps=57,
        weather=WeatherCondition.DRY,
        track_temperature=32.0,
        safety_car_probability=0.2,  # accepted, reported as not modelled
        current_compound=TireCompound.MEDIUM,
        current_tire_age=17,
        used_compounds=[],
        own_gap_to_leader=6.0,
    )
    show("Mid-race, medium tyres 17 laps old", generate_strategy(TELEMETRY, CAR, DRIVER, TYRES, mid_race, RIVALS))

    final_laps = RaceState(
        current_lap=54,
        total_laps=57,
        weather=WeatherCondition.DRY,
        track_temperature=32.0,
        current_compound=TireCompound.HARD,
        current_tire_age=25,
        used_compounds=[TireCompound.MEDIUM],
    )
    show("Final four laps", generate_strategy(TELEMETRY, CAR, DRIVER, TYRES, final_laps))

    # The same engine through the JSON request schema used by the API.
    payload = StrategyRequest.model_json_schema()["examples"][0]
    body = generate_strategy_response(StrategyRequest.model_validate(payload))
    best = body["strategies"][0]
    print("\n=== JSON request example")
    print(f"  best: {best['strategy_id']} pit laps {best['pit_laps']} ({len(json.dumps(body))} bytes of JSON)")


if __name__ == "__main__":
    main()
