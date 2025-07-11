import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json


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
    """Represents a complete race strategy option"""
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
    risk_level: str  # "low", "medium", "high"


@dataclass
class DriverProfile:
    """Driver characteristics and preferences"""
    tire_management: float  # 0-1, higher = better tire saving
    risk_tolerance: float   # 0-1, higher = more aggressive
    overtaking_style: str   # "aggressive", "conservative", "calculated"
    braking_consistency: float  # 0-1, consistency in braking zones
    throttle_aggressiveness: float  # 0-1, how aggressive on throttle


@dataclass
class CarStatus:
    """Current car condition and status"""
    damage: Dict[str, float]  # Component damage levels
    fuel_load: float  # Current fuel load in kg
    brake_temp: float  # Brake temperature
    engine_wear: float  # Engine wear percentage
    ers_availability: float  # ERS energy available
    brake_wear: float  # Brake wear percentage


@dataclass
class TireData:
    """Tire compound performance and degradation data"""
    compound: TireCompound
    base_performance: float  # Base performance multiplier
    degradation_rate: float  # Performance loss per lap
    warm_up_laps: int  # Laps needed for optimal performance
    peak_performance_window: Tuple[int, int]  # (start_lap, end_lap)
    pit_stop_delta: float  # Time lost in pit stop


@dataclass
class RaceState:
    """Current race conditions and state"""
    current_lap: int
    total_laps: int
    weather: WeatherCondition
    track_temperature: float
    track_evolution: float  # 0-1, how much track has evolved
    safety_car_probability: float  # 0-1
    yellow_flag_risk: float  # 0-1
    weather_forecast: List[Dict[str, Any]]  # Future weather predictions


@dataclass
class Competitor:
    """Competitor information"""
    driver_id: str
    current_position: int
    tire_compound: TireCompound
    tire_age: int  # Laps on current tires
    gap_to_leader: float  # Seconds
    gap_ahead: float  # Seconds to car ahead
    gap_behind: float  # Seconds to car behind
    pit_stops_completed: int
    estimated_strategy: List[Dict[str, Any]]


def evaluate_driver_penalty(driver_profile: DriverProfile, telemetry: Dict[str, Any]) -> float:
    """
    Evaluate driver-specific penalties based on telemetry and profile
    
    Args:
        driver_profile: Driver characteristics
        telemetry: Telemetry data including braking and throttle traces
    
    Returns:
        Penalty multiplier (1.0 = no penalty, >1.0 = penalty)
    """
    penalty = 1.0
    
    # Analyze braking consistency
    if 'braking_zones' in telemetry:
        braking_consistency = telemetry.get('braking_consistency', 0.8)
        if braking_consistency < 0.7:
            penalty *= (1.0 + (0.7 - braking_consistency) * 0.3)
    
    # Analyze throttle aggressiveness vs tire management
    throttle_aggressiveness = telemetry.get('throttle_aggressiveness', 0.5)
    tire_management = driver_profile.tire_management
    
    # High throttle aggressiveness with poor tire management = penalty
    if throttle_aggressiveness > 0.8 and tire_management < 0.6:
        penalty *= 1.1
    
    # Risk tolerance adjustment
    if driver_profile.risk_tolerance > 0.8:
        penalty *= 1.05  # Slight penalty for high risk tolerance
    
    return penalty


def adjust_strategy_for_damage(car_status: CarStatus, base_strategy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjust strategy based on car damage and wear
    
    Args:
        car_status: Current car condition
        base_strategy: Base strategy to adjust
    
    Returns:
        Adjusted strategy
    """
    adjusted_strategy = base_strategy.copy()
    
    # Front wing damage affects downforce and tire wear
    front_wing_damage = car_status.damage.get('front_wing', 0.0)
    if front_wing_damage > 0.3:
        # Reduce stint lengths due to increased tire wear
        for stint in adjusted_strategy.get('stints', []):
            stint['laps'] = max(5, int(stint['laps'] * 0.8))
    
    # Floor/diffuser damage affects aero efficiency
    floor_damage = car_status.damage.get('floor', 0.0)
    diffuser_damage = car_status.damage.get('diffuser', 0.0)
    aero_damage = max(floor_damage, diffuser_damage)
    
    if aero_damage > 0.2:
        # Increase pit stop delta due to reduced efficiency
        adjusted_strategy['pit_stop_delta'] = adjusted_strategy.get('pit_stop_delta', 25.0) + (aero_damage * 5.0)
    
    # Engine wear affects reliability
    if car_status.engine_wear > 0.7:
        # Reduce engine stress by using harder compounds
        adjusted_strategy['preferred_compounds'] = [TireCompound.HARD, TireCompound.MEDIUM]
    
    # Brake wear affects braking performance
    if car_status.brake_wear > 0.8:
        # Reduce aggressive braking zones
        adjusted_strategy['braking_penalty'] = 1.1
    
    return adjusted_strategy


def simulate_stint(
    start_lap: int,
    laps: int,
    tire_compound: TireCompound,
    tire_data: Dict[TireCompound, TireData],
    driver_profile: DriverProfile,
    race_state: RaceState,
    telemetry: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulate a single stint and return performance metrics
    
    Args:
        start_lap: Starting lap number
        laps: Number of laps in stint
        tire_compound: Tire compound to use
        tire_data: Tire performance data
        driver_profile: Driver characteristics
        race_state: Current race conditions
        telemetry: Telemetry data
    
    Returns:
        Stint simulation results
    """
    if tire_compound not in tire_data:
        return {'error': f'Tire compound {tire_compound} not found in tire_data'}
    
    tire_info = tire_data[tire_compound]
    lap_times = []
    tire_performance = []
    
    # Base lap time (could be parameterized)
    base_lap_time = 80.0
    
    for lap in range(1, laps + 1):
        # Calculate tire performance at this lap
        if lap <= tire_info.warm_up_laps:
            # Warm-up phase
            performance = tire_info.base_performance * (lap / tire_info.warm_up_laps)
        elif tire_info.peak_performance_window[0] <= lap <= tire_info.peak_performance_window[1]:
            # Peak performance window
            performance = tire_info.base_performance
        else:
            # Degradation phase
            laps_over_peak = max(0, lap - tire_info.peak_performance_window[1])
            degradation = laps_over_peak * tire_info.degradation_rate
            performance = max(0.1, tire_info.base_performance - degradation)
        
        # Apply driver-specific adjustments
        driver_penalty = evaluate_driver_penalty(driver_profile, telemetry)
        adjusted_performance = performance / driver_penalty
        
        # Weather adjustments
        weather_multiplier = 1.0
        if race_state.weather == WeatherCondition.WET and tire_compound != TireCompound.WET:
            weather_multiplier = 0.6
        elif race_state.weather == WeatherCondition.INTERMEDIATE and tire_compound not in [TireCompound.INTERMEDIATE, TireCompound.WET]:
            weather_multiplier = 0.8
        
        # Track temperature effects
        if race_state.track_temperature > 35:
            if tire_compound == TireCompound.SOFT:
                weather_multiplier *= 0.95
            elif tire_compound == TireCompound.HARD:
                weather_multiplier *= 1.05
        
        final_performance = adjusted_performance * weather_multiplier
        
        # Calculate lap time
        lap_time = base_lap_time / final_performance
        lap_times.append(lap_time)
        tire_performance.append(final_performance)
    
    return {
        'start_lap': start_lap,
        'end_lap': start_lap + laps - 1,
        'laps': laps,
        'tire_compound': tire_compound,
        'average_lap_time': np.mean(lap_times),
        'best_lap_time': min(lap_times),
        'worst_lap_time': max(lap_times),
        'tire_performance_trend': tire_performance,
        'total_time': sum(lap_times),
        'tire_wear_at_end': 1.0 - min(tire_performance)
    }


def score_strategy_option(
    strategy: Dict[str, Any],
    race_state: RaceState,
    competition: List[Competitor],
    driver_profile: DriverProfile
) -> float:
    """
    Score a strategy option based on multiple factors
    
    Args:
        strategy: Strategy to score
        race_state: Current race conditions
        competition: Competitor information
        driver_profile: Driver characteristics
    
    Returns:
        Strategy score (higher = better)
    """
    score = 0.0
    
    # Base score from projected race time
    projected_time = strategy.get('projected_race_time', float('inf'))
    if projected_time < float('inf'):
        score += 1000.0 / projected_time  # Higher score for faster times
    
    # Safety car probability bonus
    if race_state.safety_car_probability > 0.3:
        # Prefer strategies with pit stops around lap 20-30 for safety car windows
        pit_laps = strategy.get('pit_laps', [])
        for pit_lap in pit_laps:
            if 20 <= pit_lap <= 30:
                score += 50.0
    
    # Weather forecast consideration
    if race_state.weather_forecast:
        # Check if strategy adapts to weather changes
        weather_adaptability = 0.0
        for stint in strategy.get('stints', []):
            compound = stint.get('tire_compound')
            if compound == TireCompound.INTERMEDIATE and race_state.weather == WeatherCondition.INTERMEDIATE:
                weather_adaptability += 1.0
            elif compound == TireCompound.WET and race_state.weather == WeatherCondition.WET:
                weather_adaptability += 1.0
        score += weather_adaptability * 30.0
    
    # Competition analysis
    if competition:
        # Score based on competitive position
        avg_gap_to_leader = np.mean([c.gap_to_leader for c in competition])
        if avg_gap_to_leader < 10.0:  # Close competition
            score += 40.0
    
    # Driver profile compatibility
    risk_tolerance = driver_profile.risk_tolerance
    strategy_risk = strategy.get('risk_level', 'medium')
    
    if strategy_risk == 'high' and risk_tolerance > 0.7:
        score += 20.0
    elif strategy_risk == 'low' and risk_tolerance < 0.4:
        score += 20.0
    
    return score


def generate_strategy(
    telemetry: Dict[str, Any],
    car_status: CarStatus,
    driver_profile: DriverProfile,
    tire_data: Dict[TireCompound, TireData],
    race_state: RaceState,
    competition: List[Competitor]
) -> List[StrategyOption]:
    """
    Generate ranked list of strategy options for the race
    
    Args:
        telemetry: Telemetry data with braking_zones, throttle_trace, lap_times
        car_status: Car condition and status
        driver_profile: Driver characteristics
        tire_data: Tire compound performance data
        race_state: Current race conditions
        competition: List of competitor information
    
    Returns:
        Ranked list of strategy options
    """
    strategies = []
    
    # Generate different strategy variations
    strategy_variations = [
        {'name': 'Aggressive 2-Stop', 'stops': 2, 'compounds': [TireCompound.SOFT, TireCompound.SOFT, TireCompound.MEDIUM]},
        {'name': 'Conservative 1-Stop', 'stops': 1, 'compounds': [TireCompound.MEDIUM, TireCompound.HARD]},
        {'name': 'Balanced 2-Stop', 'stops': 2, 'compounds': [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.MEDIUM]},
        {'name': 'Ultra-Aggressive 3-Stop', 'stops': 3, 'compounds': [TireCompound.SOFT, TireCompound.SOFT, TireCompound.SOFT, TireCompound.MEDIUM]},
        {'name': 'Conservative 2-Stop', 'stops': 2, 'compounds': [TireCompound.MEDIUM, TireCompound.HARD, TireCompound.HARD]}
    ]
    
    for i, variation in enumerate(strategy_variations):
        # Calculate stint breakdown
        total_laps = race_state.total_laps - race_state.current_lap
        stops = variation['stops']
        compounds = variation['compounds']
        
        # Distribute laps across stints
        stint_lengths = []
        remaining_laps = total_laps
        
        for j in range(len(compounds) - 1):
            if j == len(compounds) - 2:  # Last stint
                stint_lengths.append(remaining_laps)
            else:
                # Distribute laps based on compound characteristics
                compound = compounds[j]
                if compound == TireCompound.SOFT:
                    stint_length = min(remaining_laps // (len(compounds) - j), 15)
                elif compound == TireCompound.MEDIUM:
                    stint_length = min(remaining_laps // (len(compounds) - j), 25)
                else:  # HARD
                    stint_length = min(remaining_laps // (len(compounds) - j), 35)
                
                stint_length = max(5, stint_length)  # Minimum stint length
                stint_lengths.append(stint_length)
                remaining_laps -= stint_length
        
        # Simulate each stint
        stints = []
        current_lap = race_state.current_lap
        total_time = 0.0
        
        for stint_length, compound in zip(stint_lengths, compounds):
            stint_result = simulate_stint(
                current_lap, stint_length, compound, tire_data,
                driver_profile, race_state, telemetry
            )
            
            if 'error' not in stint_result:
                stints.append(stint_result)
                total_time += stint_result['total_time']
                current_lap += stint_length
        
        # Add pit stop time
        pit_stop_time = stops * 25.0  # 25 seconds per pit stop
        total_time += pit_stop_time
        
        # Calculate pit lap numbers
        pit_laps = []
        cumulative_laps = race_state.current_lap
        for stint in stints[:-1]:  # Exclude last stint
            cumulative_laps += stint['laps']
            pit_laps.append(cumulative_laps)
        
        # Adjust strategy for car damage
        base_strategy = {
            'stints': stints,
            'pit_stop_delta': 25.0,
            'preferred_compounds': compounds
        }
        adjusted_strategy = adjust_strategy_for_damage(car_status, base_strategy)
        
        # Score the strategy
        strategy_score = score_strategy_option(
            {'projected_race_time': total_time, 'pit_laps': pit_laps, 'risk_level': 'medium'},
            race_state, competition, driver_profile
        )
        
        # Generate undercut/overcut opportunities
        undercut_opportunities = []
        overcut_opportunities = []
        
        for competitor in competition:
            if competitor.tire_age > 15:  # Competitor on old tires
                undercut_opportunities.append({
                    'competitor': competitor.driver_id,
                    'opportunity_type': 'undercut',
                    'reason': f'Competitor on {competitor.tire_age} lap old {competitor.tire_compound.value} tires'
                })
            
            if competitor.gap_ahead < 5.0:  # Close to car ahead
                overcut_opportunities.append({
                    'competitor': competitor.driver_id,
                    'opportunity_type': 'overcut',
                    'reason': f'Close to {competitor.driver_id} (gap: {competitor.gap_ahead:.1f}s)'
                })
        
        # Determine risk level
        risk_level = 'medium'
        if variation['stops'] >= 3:
            risk_level = 'high'
        elif variation['stops'] == 1:
            risk_level = 'low'
        
        # Create strategy option
        strategy_option = StrategyOption(
            strategy_id=f"strategy_{i+1}",
            estimated_total_time=total_time,
            confidence_score=min(0.95, strategy_score / 100.0),
            stint_breakdown=stints,
            tire_compounds=compounds,
            pit_laps=pit_laps,
            projected_race_time=total_time,
            undercut_opportunities=undercut_opportunities,
            overcut_opportunities=overcut_opportunities,
            notes=[
                f"{variation['name']} strategy",
                f"Estimated {stops} pit stops",
                f"Risk level: {risk_level}"
            ],
            risk_level=risk_level
        )
        
        strategies.append(strategy_option)
    
    # Sort strategies by projected race time
    strategies.sort(key=lambda x: x.projected_race_time)
    
    return strategies


def dummy_strategy():
    """Dummy function for testing"""
    return 'Strategy module ready'


# Test data and main execution
if __name__ == "__main__":
    # Create sample data for testing
    telemetry = {
        'braking_zones': [1, 3, 5, 7, 9],
        'throttle_trace': [0.8, 0.9, 0.7, 0.85, 0.9],
        'lap_times': [80.5, 80.2, 80.8, 80.1, 80.3],
        'braking_consistency': 0.75,
        'throttle_aggressiveness': 0.7
    }
    
    car_status = CarStatus(
        damage={'front_wing': 0.1, 'floor': 0.05, 'diffuser': 0.0},
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
    strategies = generate_strategy(
        telemetry, car_status, driver_profile, tire_data, race_state, competition
    )
    
    # Print results
    print("=== F1 Strategy Engine Test Results ===\n")
    
    for i, strategy in enumerate(strategies[:3]):  # Show top 3 strategies
        print(f"Strategy {i+1}: {strategy.strategy_id}")
        print(f"Projected Race Time: {strategy.projected_race_time:.1f} seconds")
        print(f"Confidence Score: {strategy.confidence_score:.2f}")
        print(f"Risk Level: {strategy.risk_level}")
        print(f"Pit Stops: {len(strategy.pit_laps)}")
        print(f"Tire Compounds: {[c.value for c in strategy.tire_compounds]}")
        print(f"Pit Laps: {strategy.pit_laps}")
        print(f"Undercut Opportunities: {len(strategy.undercut_opportunities)}")
        print(f"Overcut Opportunities: {len(strategy.overcut_opportunities)}")
        print(f"Notes: {', '.join(strategy.notes)}")
        print("-" * 50)
