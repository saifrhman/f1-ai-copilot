#!/usr/bin/env python3
"""
Test script for the F1 Strategy Engine
Demonstrates the generate_strategy function with realistic test data
"""

from strategy_engine import (
    generate_strategy,
    DriverProfile,
    CarStatus,
    TireData,
    RaceState,
    Competitor,
    TireCompound,
    WeatherCondition
)


def create_test_data():
    """Create comprehensive test data for strategy generation"""
    
    # Telemetry data
    telemetry = {
        'braking_zones': [1, 3, 5, 7, 9, 11, 13, 15],
        'throttle_trace': [0.8, 0.9, 0.7, 0.85, 0.9, 0.75, 0.88, 0.92],
        'lap_times': [80.5, 80.2, 80.8, 80.1, 80.3, 80.6, 80.0, 80.4],
        'braking_consistency': 0.78,
        'throttle_aggressiveness': 0.72,
        'tire_saving_indicators': [0.6, 0.7, 0.65, 0.75, 0.68, 0.72, 0.69, 0.71]
    }
    
    # Car status
    car_status = CarStatus(
        damage={
            'front_wing': 0.15,
            'floor': 0.08,
            'diffuser': 0.02,
            'rear_wing': 0.05,
            'sidepod': 0.03
        },
        fuel_load=95.5,
        brake_temp=345.0,
        engine_wear=0.28,
        ers_availability=0.82,
        brake_wear=0.35
    )
    
    # Driver profile
    driver_profile = DriverProfile(
        tire_management=0.73,
        risk_tolerance=0.64,
        overtaking_style="calculated",
        braking_consistency=0.78,
        throttle_aggressiveness=0.72
    )
    
    # Tire data for different compounds
    tire_data = {
        TireCompound.SOFT: TireData(
            compound=TireCompound.SOFT,
            base_performance=1.0,
            degradation_rate=0.022,
            warm_up_laps=2,
            peak_performance_window=(2, 8),
            pit_stop_delta=25.0
        ),
        TireCompound.MEDIUM: TireData(
            compound=TireCompound.MEDIUM,
            base_performance=0.96,
            degradation_rate=0.016,
            warm_up_laps=3,
            peak_performance_window=(3, 16),
            pit_stop_delta=25.0
        ),
        TireCompound.HARD: TireData(
            compound=TireCompound.HARD,
            base_performance=0.92,
            degradation_rate=0.012,
            warm_up_laps=4,
            peak_performance_window=(4, 28),
            pit_stop_delta=25.0
        ),
        TireCompound.INTERMEDIATE: TireData(
            compound=TireCompound.INTERMEDIATE,
            base_performance=0.88,
            degradation_rate=0.018,
            warm_up_laps=2,
            peak_performance_window=(2, 12),
            pit_stop_delta=25.0
        ),
        TireCompound.WET: TireData(
            compound=TireCompound.WET,
            base_performance=0.82,
            degradation_rate=0.025,
            warm_up_laps=1,
            peak_performance_window=(1, 10),
            pit_stop_delta=25.0
        )
    }
    
    # Race state
    race_state = RaceState(
        current_lap=8,
        total_laps=52,
        weather=WeatherCondition.DRY,
        track_temperature=34.5,
        track_evolution=0.35,
        safety_car_probability=0.25,
        yellow_flag_risk=0.15,
        weather_forecast=[
            {'lap': 20, 'weather': WeatherCondition.DRY, 'temperature': 33.0},
            {'lap': 35, 'weather': WeatherCondition.DRY, 'temperature': 31.0},
            {'lap': 45, 'weather': WeatherCondition.DRY, 'temperature': 29.0}
        ]
    )
    
    # Competition data
    competition = [
        Competitor(
            driver_id="HAM",
            current_position=2,
            tire_compound=TireCompound.SOFT,
            tire_age=6,
            gap_to_leader=1.8,
            gap_ahead=0.0,
            gap_behind=0.9,
            pit_stops_completed=0,
            estimated_strategy=[{'lap': 15, 'compound': TireCompound.MEDIUM}]
        ),
        Competitor(
            driver_id="VER",
            current_position=3,
            tire_compound=TireCompound.MEDIUM,
            tire_age=10,
            gap_to_leader=3.2,
            gap_ahead=1.4,
            gap_behind=0.7,
            pit_stops_completed=0,
            estimated_strategy=[{'lap': 25, 'compound': TireCompound.HARD}]
        ),
        Competitor(
            driver_id="BOT",
            current_position=4,
            tire_compound=TireCompound.SOFT,
            tire_age=4,
            gap_to_leader=5.8,
            gap_ahead=2.6,
            gap_behind=1.1,
            pit_stops_completed=0,
            estimated_strategy=[{'lap': 18, 'compound': TireCompound.MEDIUM}]
        ),
        Competitor(
            driver_id="LEC",
            current_position=5,
            tire_compound=TireCompound.MEDIUM,
            tire_age=12,
            gap_to_leader=8.1,
            gap_ahead=2.3,
            gap_behind=0.5,
            pit_stops_completed=0,
            estimated_strategy=[{'lap': 30, 'compound': TireCompound.HARD}]
        )
    ]
    
    return telemetry, car_status, driver_profile, tire_data, race_state, competition


def print_strategy_details(strategy, index):
    """Print detailed information about a strategy option"""
    print(f"\n{'='*60}")
    print(f"STRATEGY {index + 1}: {strategy.strategy_id}")
    print(f"{'='*60}")
    
    print(f"📊 PERFORMANCE METRICS:")
    print(f"   • Projected Race Time: {strategy.projected_race_time:.1f} seconds")
    print(f"   • Confidence Score: {strategy.confidence_score:.2f}")
    print(f"   • Risk Level: {strategy.risk_level.upper()}")
    print(f"   • Total Pit Stops: {len(strategy.pit_laps)}")
    
    print(f"\n🛞 TIRE STRATEGY:")
    print(f"   • Compounds: {' → '.join([c.value.upper() for c in strategy.tire_compounds])}")
    print(f"   • Pit Laps: {strategy.pit_laps}")
    
    print(f"\n📈 STINT BREAKDOWN:")
    for i, stint in enumerate(strategy.stint_breakdown):
        compound = stint['tire_compound'].value.upper()
        laps = stint['laps']
        avg_time = stint['average_lap_time']
        total_time = stint['total_time']
        print(f"   • Stint {i+1}: {laps} laps on {compound} (Avg: {avg_time:.1f}s, Total: {total_time:.1f}s)")
    
    print(f"\n🎯 OPPORTUNITIES:")
    if strategy.undercut_opportunities:
        print(f"   • Undercut Opportunities: {len(strategy.undercut_opportunities)}")
        for opp in strategy.undercut_opportunities[:2]:  # Show first 2
            print(f"     - {opp['reason']}")
    else:
        print(f"   • Undercut Opportunities: None")
    
    if strategy.overcut_opportunities:
        print(f"   • Overcut Opportunities: {len(strategy.overcut_opportunities)}")
        for opp in strategy.overcut_opportunities[:2]:  # Show first 2
            print(f"     - {opp['reason']}")
    else:
        print(f"   • Overcut Opportunities: None")
    
    print(f"\n📝 NOTES:")
    for note in strategy.notes:
        print(f"   • {note}")


def test_dry_weather_scenario():
    """Test strategy generation for dry weather conditions"""
    print("🏁 F1 STRATEGY ENGINE - DRY WEATHER SCENARIO")
    print("=" * 60)
    
    # Get test data
    telemetry, car_status, driver_profile, tire_data, race_state, competition = create_test_data()
    
    # Generate strategies
    strategies = generate_strategy(
        telemetry, car_status, driver_profile, tire_data, race_state, competition
    )
    
    print(f"\n📋 Generated {len(strategies)} strategy options")
    print(f"🏆 Top 3 strategies:")
    
    # Display top 3 strategies
    for i, strategy in enumerate(strategies[:3]):
        print_strategy_details(strategy, i)
    
    return strategies


def test_wet_weather_scenario():
    """Test strategy generation for wet weather conditions"""
    print("\n🌧️ F1 STRATEGY ENGINE - WET WEATHER SCENARIO")
    print("=" * 60)
    
    # Get test data
    telemetry, car_status, driver_profile, tire_data, race_state, competition = create_test_data()
    
    # Modify for wet weather
    race_state.weather = WeatherCondition.WET
    race_state.track_temperature = 22.0
    race_state.safety_car_probability = 0.4
    race_state.yellow_flag_risk = 0.3
    
    # Update weather forecast
    race_state.weather_forecast = [
        {'lap': 15, 'weather': WeatherCondition.WET, 'temperature': 20.0},
        {'lap': 30, 'weather': WeatherCondition.INTERMEDIATE, 'temperature': 24.0},
        {'lap': 40, 'weather': WeatherCondition.DRY, 'temperature': 28.0}
    ]
    
    # Generate strategies
    strategies = generate_strategy(
        telemetry, car_status, driver_profile, tire_data, race_state, competition
    )
    
    print(f"\n📋 Generated {len(strategies)} strategy options")
    print(f"🏆 Top 2 strategies:")
    
    # Display top 2 strategies
    for i, strategy in enumerate(strategies[:2]):
        print_strategy_details(strategy, i)
    
    return strategies


def test_damage_scenario():
    """Test strategy generation with significant car damage"""
    print("\n🔧 F1 STRATEGY ENGINE - DAMAGE SCENARIO")
    print("=" * 60)
    
    # Get test data
    telemetry, car_status, driver_profile, tire_data, race_state, competition = create_test_data()
    
    # Simulate significant damage
    car_status.damage = {
        'front_wing': 0.45,  # Significant front wing damage
        'floor': 0.25,       # Moderate floor damage
        'diffuser': 0.15,    # Some diffuser damage
        'rear_wing': 0.10,
        'sidepod': 0.08
    }
    car_status.engine_wear = 0.75  # High engine wear
    car_status.brake_wear = 0.85   # High brake wear
    
    # Generate strategies
    strategies = generate_strategy(
        telemetry, car_status, driver_profile, tire_data, race_state, competition
    )
    
    print(f"\n📋 Generated {len(strategies)} strategy options")
    print(f"🏆 Top 2 strategies (damage-adjusted):")
    
    # Display top 2 strategies
    for i, strategy in enumerate(strategies[:2]):
        print_strategy_details(strategy, i)
    
    return strategies


if __name__ == "__main__":
    # Run all test scenarios
    dry_strategies = test_dry_weather_scenario()
    wet_strategies = test_wet_weather_scenario()
    damage_strategies = test_damage_scenario()
    
    print(f"\n{'='*60}")
    print("🎯 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Dry weather: {len(dry_strategies)} strategies generated")
    print(f"✅ Wet weather: {len(wet_strategies)} strategies generated")
    print(f"✅ Damage scenario: {len(damage_strategies)} strategies generated")
    print(f"\n🏁 Strategy engine is ready for F1 race simulation!") 