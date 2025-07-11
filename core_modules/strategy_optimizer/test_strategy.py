#!/usr/bin/env python3
"""
Test script for the pit stop strategy recommendation system
"""

from strategy_engine import (
    recommend_pit_stop_strategy,
    create_sample_degradation_curves,
    WeatherConditions,
    WeatherCondition,
    TireCompound
)


def test_dry_weather_strategy():
    """Test strategy recommendation for dry weather conditions"""
    print("=== Testing Dry Weather Strategy ===")
    
    # Create sample data
    degradation_curves = create_sample_degradation_curves()
    weather = WeatherConditions(
        condition=WeatherCondition.DRY,
        temperature=25.0,
        humidity=60.0,
        track_temperature=35.0,
        rain_probability=0.0
    )
    available_compounds = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]
    
    # Get strategy recommendation
    strategy = recommend_pit_stop_strategy(
        total_laps=50,
        degradation_curves=degradation_curves,
        weather=weather,
        available_compounds=available_compounds
    )
    
    print(f"Total stops: {strategy.total_stops}")
    print(f"Stint durations: {strategy.stint_durations}")
    print(f"Tire compounds: {[c.value for c in strategy.tire_compounds]}")
    print(f"Pit laps: {strategy.pit_laps}")
    print(f"Estimated total time: {strategy.estimated_total_time:.2f} seconds")
    print(f"Strategy confidence: {strategy.strategy_confidence:.2f}")
    print()


def test_wet_weather_strategy():
    """Test strategy recommendation for wet weather conditions"""
    print("=== Testing Wet Weather Strategy ===")
    
    degradation_curves = create_sample_degradation_curves()
    weather = WeatherConditions(
        condition=WeatherCondition.WET,
        temperature=18.0,
        humidity=85.0,
        track_temperature=20.0,
        rain_probability=0.9
    )
    available_compounds = [TireCompound.INTERMEDIATE, TireCompound.WET]
    
    strategy = recommend_pit_stop_strategy(
        total_laps=30,
        degradation_curves=degradation_curves,
        weather=weather,
        available_compounds=available_compounds
    )
    
    print(f"Total stops: {strategy.total_stops}")
    print(f"Stint durations: {strategy.stint_durations}")
    print(f"Tire compounds: {[c.value for c in strategy.tire_compounds]}")
    print(f"Pit laps: {strategy.pit_laps}")
    print(f"Estimated total time: {strategy.estimated_total_time:.2f} seconds")
    print(f"Strategy confidence: {strategy.strategy_confidence:.2f}")
    print()


def test_short_race_strategy():
    """Test strategy for a shorter race"""
    print("=== Testing Short Race Strategy ===")
    
    degradation_curves = create_sample_degradation_curves()
    weather = WeatherConditions(
        condition=WeatherCondition.DRY,
        temperature=22.0,
        humidity=55.0,
        track_temperature=28.0,
        rain_probability=0.1
    )
    available_compounds = [TireCompound.SOFT, TireCompound.MEDIUM]
    
    strategy = recommend_pit_stop_strategy(
        total_laps=20,
        degradation_curves=degradation_curves,
        weather=weather,
        available_compounds=available_compounds,
        min_stint_length=3,
        max_stint_length=15
    )
    
    print(f"Total stops: {strategy.total_stops}")
    print(f"Stint durations: {strategy.stint_durations}")
    print(f"Tire compounds: {[c.value for c in strategy.tire_compounds]}")
    print(f"Pit laps: {strategy.pit_laps}")
    print(f"Estimated total time: {strategy.estimated_total_time:.2f} seconds")
    print(f"Strategy confidence: {strategy.strategy_confidence:.2f}")
    print()


def test_target_stops_strategy():
    """Test strategy with a specific target number of stops"""
    print("=== Testing Target Stops Strategy ===")
    
    degradation_curves = create_sample_degradation_curves()
    weather = WeatherConditions(
        condition=WeatherCondition.DRY,
        temperature=24.0,
        humidity=65.0,
        track_temperature=32.0,
        rain_probability=0.0
    )
    available_compounds = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]
    
    strategy = recommend_pit_stop_strategy(
        total_laps=60,
        degradation_curves=degradation_curves,
        weather=weather,
        available_compounds=available_compounds,
        target_stops=2  # Force 2-stop strategy
    )
    
    print(f"Total stops: {strategy.total_stops}")
    print(f"Stint durations: {strategy.stint_durations}")
    print(f"Tire compounds: {[c.value for c in strategy.tire_compounds]}")
    print(f"Pit laps: {strategy.pit_laps}")
    print(f"Estimated total time: {strategy.estimated_total_time:.2f} seconds")
    print(f"Strategy confidence: {strategy.strategy_confidence:.2f}")
    print()


if __name__ == "__main__":
    test_dry_weather_strategy()
    test_wet_weather_strategy()
    test_short_race_strategy()
    test_target_stops_strategy() 