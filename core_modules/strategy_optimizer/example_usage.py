#!/usr/bin/env python3
"""Runnable example for the current F1 strategy heuristic."""

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


def main():
    telemetry = {
        "lap_times": [80.5, 80.2, 80.8, 80.1, 80.3, 80.6, 80.0, 80.4],
        "braking_consistency": 0.75,
        "throttle_aggressiveness": 0.70,
    }
    car_status = CarStatus(
        damage={"front_wing": 0.1, "floor": 0.05, "diffuser": 0.0},
        fuel_load=100.0,
        brake_temp=350.0,
        engine_wear=0.3,
        ers_availability=0.8,
        brake_wear=0.4,
    )
    driver_profile = DriverProfile(
        tire_management=0.7,
        risk_tolerance=0.6,
        overtaking_style="calculated",
        braking_consistency=0.75,
        throttle_aggressiveness=0.7,
    )
    tire_data = {
        TireCompound.SOFT: TireData(TireCompound.SOFT, 1.00, 0.020, 2, (2, 8), 25.0),
        TireCompound.MEDIUM: TireData(TireCompound.MEDIUM, 0.95, 0.015, 3, (3, 15), 25.0),
        TireCompound.HARD: TireData(TireCompound.HARD, 0.90, 0.010, 5, (5, 25), 25.0),
    }
    race_state = RaceState(
        current_lap=5,
        total_laps=50,
        weather=WeatherCondition.DRY,
        track_temperature=32.0,
        track_evolution=0.3,
        safety_car_probability=0.2,
        yellow_flag_risk=0.1,
        weather_forecast=[{"lap": 20, "weather": WeatherCondition.DRY}],
    )
    competition = [
        Competitor("HAM", 2, TireCompound.SOFT, 8, 2.5, 0.0, 1.2, 0, []),
        Competitor("VER", 3, TireCompound.MEDIUM, 12, 5.1, 2.6, 0.8, 0, []),
    ]

    strategies = generate_strategy(
        telemetry=telemetry,
        car_status=car_status,
        driver_profile=driver_profile,
        tire_data=tire_data,
        race_state=race_state,
        competition=competition,
    )

    print(f"Generated {len(strategies)} heuristic strategy candidates")
    for strategy in strategies[:3]:
        print(
            f"{strategy.strategy_id}: {strategy.projected_race_time:.1f}s, "
            f"{len(strategy.pit_laps)} stop(s), "
            f"{' -> '.join(c.value for c in strategy.tire_compounds)}, "
            f"pit laps {strategy.pit_laps}"
        )


if __name__ == "__main__":
    main()
