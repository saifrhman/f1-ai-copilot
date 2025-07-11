#!/usr/bin/env python3
"""
Example usage of the F1 Strategy Engine
Demonstrates the generate_strategy() function with realistic input data
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


def main():
    """Example usage of the generate_strategy function"""
    
    # Input data as specified in requirements
    
    # Telemetry data
    telemetry = {
        'braking_zones': [1, 3, 5, 7, 9, 11, 13, 15],
        'throttle_trace': [0.8, 0.9, 0.7, 0.85, 0.9, 0.75, 0.88, 0.92],
        'lap_times': [80.5, 80.2, 80.8, 80.1, 80.3, 80.6, 80.0, 80.4]
    }
    
    # Car status
    car_status = CarStatus(
        damage={'front_wing': 0.1, 'floor': 0.05, 'diffuser': 0.0},
        fuel_load=100.0,
        brake_temp=350.0,
        engine_wear=0.3,
        ers_availability=0.8,
        brake_wear=0.4
    )
    
    # Driver profile
    driver_profile = DriverProfile(
        tire_management=0.7,
        risk_tolerance=0.6,
        overtaking_style="calculated",
        braking_consistency=0.75,
        throttle_aggressiveness=0.7
    )
    
    # Tire data
    tire_data = {
        TireCompound.SOFT: TireData(
            compound=TireCompound.SOFT,
            base_performance=1.0,
            degradation_rate=0.02,
            warm_up_laps=2,
            peak_performance_window=(2, 8),
            pit_stop_delta=25.0
        ),
        TireCompound.MEDIUM: TireData(
            compound=TireCompound.MEDIUM,
            base_performance=0.95,
            degradation_rate=0.015,
            warm_up_laps=3,
            peak_performance_window=(3, 15),
            pit_stop_delta=25.0
        ),
        TireCompound.HARD: TireData(
            compound=TireCompound.HARD,
            base_performance=0.9,
            degradation_rate=0.01,
            warm_up_laps=5,
            peak_performance_window=(5, 25),
            pit_stop_delta=25.0
        )
    }
    
    # Race state
    race_state = RaceState(
        current_lap=5,
        total_laps=50,
        weather=WeatherCondition.DRY,
        track_temperature=32.0,
        track_evolution=0.3,
        safety_car_probability=0.2,
        yellow_flag_risk=0.1,
        weather_forecast=[{'lap': 20, 'weather': WeatherCondition.DRY}]
    )
    
    # Competition data
    competition = [
        Competitor(
            driver_id="HAM",
            current_position=2,
            tire_compound=TireCompound.SOFT,
            tire_age=8,
            gap_to_leader=2.5,
            gap_ahead=0.0,
            gap_behind=1.2,
            pit_stops_completed=0,
            estimated_strategy=[]
        ),
        Competitor(
            driver_id="VER",
            current_position=3,
            tire_compound=TireCompound.MEDIUM,
            tire_age=12,
            gap_to_leader=5.1,
            gap_ahead=2.6,
            gap_behind=0.8,
            pit_stops_completed=0,
            estimated_strategy=[]
        )
    ]
    
    # Generate strategies
    print("🏁 F1 Strategy Engine - Example Usage")
    print("=" * 50)
    
    strategies = generate_strategy(
        telemetry=telemetry,
        car_status=car_status,
        driver_profile=driver_profile,
        tire_data=tire_data,
        race_state=race_state,
        competition=competition
    )
    
    # Display results
    print(f"\n📋 Generated {len(strategies)} strategy options:")
    print()
    
    for i, strategy in enumerate(strategies[:3]):  # Show top 3
        print(f"🏆 Strategy {i+1}: {strategy.strategy_id}")
        print(f"   • Projected Time: {strategy.projected_race_time:.1f}s")
        print(f"   • Confidence: {strategy.confidence_score:.2f}")
        print(f"   • Risk Level: {strategy.risk_level}")
        print(f"   • Pit Stops: {len(strategy.pit_laps)}")
        print(f"   • Compounds: {' → '.join([c.value.upper() for c in strategy.tire_compounds])}")
        print(f"   • Pit Laps: {strategy.pit_laps}")
        print(f"   • Undercut Opportunities: {len(strategy.undercut_opportunities)}")
        print(f"   • Overcut Opportunities: {len(strategy.overcut_opportunities)}")
        print(f"   • Notes: {', '.join(strategy.notes)}")
        print()


if __name__ == "__main__":
    main() 