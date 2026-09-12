"""Regression tests for the current strategy engine."""

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


def create_test_data(weather=WeatherCondition.DRY):
    telemetry = {
        "lap_times": [80.5, 80.2, 80.8, 80.1, 80.3],
        "braking_consistency": 0.78,
        "throttle_aggressiveness": 0.72,
    }
    car_status = CarStatus(
        damage={"front_wing": 0.15, "floor": 0.08, "diffuser": 0.02},
        fuel_load=95.5,
        brake_temp=345.0,
        engine_wear=0.28,
        ers_availability=0.82,
        brake_wear=0.35,
    )
    driver = DriverProfile(0.73, 0.64, "calculated", 0.78, 0.72)
    tire_data = {
        TireCompound.SOFT: TireData(TireCompound.SOFT, 1.00, 0.022, 2, (2, 8), 25.0),
        TireCompound.MEDIUM: TireData(TireCompound.MEDIUM, 0.96, 0.016, 3, (3, 16), 25.0),
        TireCompound.HARD: TireData(TireCompound.HARD, 0.92, 0.012, 4, (4, 28), 25.0),
        TireCompound.INTERMEDIATE: TireData(TireCompound.INTERMEDIATE, 0.88, 0.018, 2, (2, 12), 25.0),
        TireCompound.WET: TireData(TireCompound.WET, 0.82, 0.025, 1, (1, 10), 25.0),
    }
    race = RaceState(
        current_lap=8,
        total_laps=52,
        weather=weather,
        track_temperature=34.5 if weather == WeatherCondition.DRY else 22.0,
        track_evolution=0.35,
        safety_car_probability=0.25,
        yellow_flag_risk=0.15,
        weather_forecast=[],
    )
    competitors = [
        Competitor("HAM", 2, TireCompound.SOFT, 16, 1.8, 1.0, 0.9, 0, []),
        Competitor("VER", 3, TireCompound.MEDIUM, 10, 3.2, 1.4, 0.7, 0, []),
    ]
    return telemetry, car_status, driver, tire_data, race, competitors


def _assert_valid_strategies(strategies, remaining_laps):
    assert strategies
    for strategy in strategies:
        assert strategy.projected_race_time > 0
        assert len(strategy.stint_breakdown) == len(strategy.tire_compounds)
        assert len(strategy.pit_laps) == len(strategy.tire_compounds) - 1
        assert sum(stint["laps"] for stint in strategy.stint_breakdown) == remaining_laps
        assert all(stint["total_time"] > 0 for stint in strategy.stint_breakdown)
        assert 0.0 <= strategy.confidence_score <= 1.0


def test_dry_weather_scenario():
    data = create_test_data(WeatherCondition.DRY)
    strategies = generate_strategy(*data)
    remaining = data[4].total_laps - data[4].current_lap + 1
    _assert_valid_strategies(strategies, remaining)
    assert all(
        compound in {TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD}
        for strategy in strategies
        for compound in strategy.tire_compounds
    )


def test_wet_weather_scenario():
    data = create_test_data(WeatherCondition.WET)
    strategies = generate_strategy(*data)
    remaining = data[4].total_laps - data[4].current_lap + 1
    _assert_valid_strategies(strategies, remaining)
    assert all(
        compound in {TireCompound.WET, TireCompound.INTERMEDIATE}
        for strategy in strategies
        for compound in strategy.tire_compounds
    )


def test_damage_scenario_increases_projected_time():
    baseline = create_test_data(WeatherCondition.DRY)
    baseline_strategies = generate_strategy(*baseline)

    damaged = create_test_data(WeatherCondition.DRY)
    damaged[1].damage = {"front_wing": 0.45, "floor": 0.25, "diffuser": 0.15}
    damaged[1].engine_wear = 0.75
    damaged[1].brake_wear = 0.85
    damaged_strategies = generate_strategy(*damaged)

    assert damaged_strategies[0].projected_race_time > baseline_strategies[0].projected_race_time
