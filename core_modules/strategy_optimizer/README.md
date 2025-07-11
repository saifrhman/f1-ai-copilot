# F1 Strategy Engine

A comprehensive Formula 1 race strategy optimization engine that simulates and recommends optimal race strategies based on telemetry, driver behavior, car condition, tire states, and race dynamics.

## 🏁 Features

- **Multi-factor Strategy Analysis**: Considers driver behavior, car condition, tire degradation, weather, and competition
- **Modular Architecture**: Clean separation of concerns with helper functions for different aspects
- **Real-time Adaptation**: Adjusts strategies based on damage, wear, and changing conditions
- **Competition Intelligence**: Analyzes competitor strategies and identifies undercut/overcut opportunities
- **Weather Integration**: Handles dry, wet, and intermediate conditions with appropriate tire compounds

## 📦 Module Structure

```
core_modules/strategy_optimizer/
├── strategy_engine.py      # Main strategy engine implementation
├── test_strategy_engine.py # Comprehensive test scenarios
├── example_usage.py        # Simple usage example
└── README.md              # This documentation
```

## 🚀 Quick Start

### Basic Usage

```python
from strategy_engine import generate_strategy, DriverProfile, CarStatus, TireData, RaceState, Competitor

# Create input data
telemetry = {
    'braking_zones': [1, 3, 5, 7, 9],
    'throttle_trace': [0.8, 0.9, 0.7, 0.85, 0.9],
    'lap_times': [80.5, 80.2, 80.8, 80.1, 80.3]
}

car_status = CarStatus(
    damage={'front_wing': 0.1, 'floor': 0.05},
    fuel_load=100.0,
    brake_temp=350.0,
    engine_wear=0.3,
    ers_availability=0.8,
    brake_wear=0.4
)

driver_profile = DriverProfile(
    tire_management=0.7,
    risk_tolerance=0.6,
    overtaking_style="calculated",
    braking_consistency=0.75,
    throttle_aggressiveness=0.7
)

# Generate strategies
strategies = generate_strategy(
    telemetry=telemetry,
    car_status=car_status,
    driver_profile=driver_profile,
    tire_data=tire_data,
    race_state=race_state,
    competition=competition
)

# Access results
for strategy in strategies:
    print(f"Strategy: {strategy.strategy_id}")
    print(f"Projected Time: {strategy.projected_race_time:.1f}s")
    print(f"Pit Stops: {len(strategy.pit_laps)}")
    print(f"Compounds: {[c.value for c in strategy.tire_compounds]}")
```

## 📊 Input Parameters

### `telemetry: dict`
- `braking_zones`: List of braking zone numbers
- `throttle_trace`: Throttle application percentages
- `lap_times`: Recent lap times in seconds
- `braking_consistency`: Driver braking consistency (0-1)
- `throttle_aggressiveness`: Throttle aggressiveness (0-1)

### `car_status: CarStatus`
- `damage`: Component damage levels (front_wing, floor, diffuser, etc.)
- `fuel_load`: Current fuel load in kg
- `brake_temp`: Brake temperature
- `engine_wear`: Engine wear percentage (0-1)
- `ers_availability`: ERS energy available (0-1)
- `brake_wear`: Brake wear percentage (0-1)

### `driver_profile: DriverProfile`
- `tire_management`: Tire saving ability (0-1)
- `risk_tolerance`: Risk tolerance level (0-1)
- `overtaking_style`: "aggressive", "conservative", or "calculated"
- `braking_consistency`: Consistency in braking zones (0-1)
- `throttle_aggressiveness`: Throttle application aggressiveness (0-1)

### `tire_data: Dict[TireCompound, TireData]`
- `compound`: Tire compound type
- `base_performance`: Base performance multiplier
- `degradation_rate`: Performance loss per lap
- `warm_up_laps`: Laps needed for optimal performance
- `peak_performance_window`: (start_lap, end_lap) for peak performance
- `pit_stop_delta`: Time lost in pit stop

### `race_state: RaceState`
- `current_lap`: Current lap number
- `total_laps`: Total race laps
- `weather`: Weather condition (DRY, WET, INTERMEDIATE)
- `track_temperature`: Track temperature in Celsius
- `track_evolution`: Track evolution factor (0-1)
- `safety_car_probability`: Probability of safety car (0-1)
- `yellow_flag_risk`: Risk of yellow flags (0-1)
- `weather_forecast`: Future weather predictions

### `competition: List[Competitor]`
- `driver_id`: Competitor driver ID
- `current_position`: Current race position
- `tire_compound`: Current tire compound
- `tire_age`: Laps on current tires
- `gap_to_leader`: Gap to race leader in seconds
- `gap_ahead`: Gap to car ahead in seconds
- `gap_behind`: Gap to car behind in seconds
- `pit_stops_completed`: Number of pit stops completed
- `estimated_strategy`: Estimated competitor strategy

## 📈 Output

Returns a list of `StrategyOption` objects with:

- `strategy_id`: Unique strategy identifier
- `projected_race_time`: Estimated total race time
- `confidence_score`: Strategy confidence (0-1)
- `stint_breakdown`: Detailed stint information
- `tire_compounds`: Tire compound sequence
- `pit_laps`: Lap numbers for pit stops
- `undercut_opportunities`: List of undercut opportunities
- `overcut_opportunities`: List of overcut opportunities
- `notes`: Strategy notes and recommendations
- `risk_level`: Strategy risk level ("low", "medium", "high")

## 🔧 Helper Functions

### `evaluate_driver_penalty(driver_profile, telemetry)`
Evaluates driver-specific penalties based on telemetry and profile characteristics.

### `adjust_strategy_for_damage(car_status, base_strategy)`
Adjusts strategy based on car damage and wear levels.

### `simulate_stint(start_lap, laps, tire_compound, tire_data, driver_profile, race_state, telemetry)`
Simulates a single stint and returns performance metrics.

### `score_strategy_option(strategy, race_state, competition, driver_profile)`
Scores a strategy option based on multiple factors.

## 🧪 Testing

Run the comprehensive test suite:

```bash
python core_modules/strategy_optimizer/test_strategy_engine.py
```

This will test:
- Dry weather scenarios
- Wet weather scenarios  
- Damage scenarios
- Different driver profiles
- Various race conditions

## 🎯 Example Scenarios

### Dry Weather Strategy
- Optimizes for 1-2 stop strategies
- Balances soft and medium compounds
- Considers track temperature effects

### Wet Weather Strategy
- Adapts to changing weather conditions
- Prioritizes appropriate tire compounds
- Accounts for safety car probability

### Damage Scenario
- Adjusts stint lengths for increased tire wear
- Modifies pit stop deltas for aero damage
- Considers engine wear for compound selection

## 🏗️ Architecture

The strategy engine follows a modular design:

1. **Input Processing**: Validates and processes input data
2. **Strategy Generation**: Creates multiple strategy variations
3. **Stint Simulation**: Simulates individual stints with realistic degradation
4. **Strategy Evaluation**: Scores strategies based on multiple factors
5. **Damage Adjustment**: Modifies strategies based on car condition
6. **Competition Analysis**: Identifies overtaking opportunities
7. **Output Ranking**: Returns ranked list of optimal strategies

## 🔮 Future Enhancements

- Machine learning integration for strategy optimization
- Real-time telemetry integration
- Advanced weather modeling
- Multi-lap simulation capabilities
- Driver-specific learning algorithms

## 📝 License

This module is part of the F1 AI Copilot system and is designed for educational and research purposes. 