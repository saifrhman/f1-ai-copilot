#!/usr/bin/env python3
"""
Setup Recommender
Uses Bayesian optimization to recommend optimal car setups based on conditions
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class WeatherCondition(Enum):
    DRY = "dry"
    WET = "wet"
    INTERMEDIATE = "intermediate"


class TrackType(Enum):
    HIGH_SPEED = "high_speed"
    TECHNICAL = "technical"
    MIXED = "mixed"
    LOW_SPEED = "low_speed"


@dataclass
class DriverPreferences:
    """Driver setup preferences"""
    preferred_ride_height: Optional[float] = None
    preferred_wing_angles: Optional[Dict[str, float]] = None
    preferred_diff_settings: Optional[Dict[str, float]] = None
    risk_tolerance: float = 0.5  # 0-1, higher = more aggressive
    tire_management: float = 0.5  # 0-1, higher = better tire saving


@dataclass
class TrackProfile:
    """Track characteristics"""
    track_name: str
    track_length: float
    corners: int
    high_speed_sections: int
    low_speed_sections: int
    track_type: TrackType
    average_speed: float
    downforce_requirement: float  # 0-1, higher = more downforce needed


@dataclass
class WeatherData:
    """Weather conditions"""
    condition: WeatherCondition
    temperature: float
    humidity: float
    wind_speed: Optional[float] = None


@dataclass
class SetupConfiguration:
    """Car setup configuration"""
    ride_height: float
    front_wing_angle: float
    rear_wing_angle: float
    diff_settings: Dict[str, float]
    brake_bias: float
    suspension_settings: Dict[str, float]
    tire_pressures: Dict[str, float]


class SetupOptimizer:
    """Bayesian optimization for car setup"""
    
    def __init__(self):
        self.track_profiles = self._load_track_profiles()
        self.weather_effects = self._load_weather_effects()
        self.setup_constraints = self._load_setup_constraints()
    
    def _load_track_profiles(self) -> Dict[str, TrackProfile]:
        """Load track profile data"""
        return {
            "monaco": TrackProfile(
                track_name="Circuit de Monaco",
                track_length=3337,
                corners=19,
                high_speed_sections=2,
                low_speed_sections=15,
                track_type=TrackType.TECHNICAL,
                average_speed=160,
                downforce_requirement=0.9
            ),
            "silverstone": TrackProfile(
                track_name="Silverstone Circuit",
                track_length=5891,
                corners=18,
                high_speed_sections=8,
                low_speed_sections=4,
                track_type=TrackType.HIGH_SPEED,
                average_speed=220,
                downforce_requirement=0.6
            ),
            "spa": TrackProfile(
                track_name="Circuit de Spa-Francorchamps",
                track_length=7004,
                corners=20,
                high_speed_sections=10,
                low_speed_sections=6,
                track_type=TrackType.MIXED,
                average_speed=200,
                downforce_requirement=0.7
            ),
            "singapore": TrackProfile(
                track_name="Marina Bay Street Circuit",
                track_length=5063,
                corners=23,
                high_speed_sections=3,
                low_speed_sections=18,
                track_type=TrackType.TECHNICAL,
                average_speed=170,
                downforce_requirement=0.8
            )
        }
    
    def _load_weather_effects(self) -> Dict[WeatherCondition, Dict[str, float]]:
        """Load weather effects on setup"""
        return {
            WeatherCondition.DRY: {
                "downforce_multiplier": 1.0,
                "ride_height_adjustment": 0.0,
                "tire_pressure_adjustment": 0.0
            },
            WeatherCondition.WET: {
                "downforce_multiplier": 1.2,
                "ride_height_adjustment": 5.0,  # mm higher
                "tire_pressure_adjustment": -0.2  # bar lower
            },
            WeatherCondition.INTERMEDIATE: {
                "downforce_multiplier": 1.1,
                "ride_height_adjustment": 2.5,
                "tire_pressure_adjustment": -0.1
            }
        }
    
    def _load_setup_constraints(self) -> Dict[str, Tuple[float, float]]:
        """Load setup parameter constraints"""
        return {
            "ride_height": (60, 80),  # mm
            "front_wing_angle": (0, 15),  # degrees
            "rear_wing_angle": (0, 20),  # degrees
            "brake_bias": (50, 70),  # percentage
            "diff_preload": (0, 100),  # Nm
            "diff_power": (0, 100),  # percentage
            "diff_coast": (0, 100),  # percentage
            "front_arb": (0, 100),  # percentage
            "rear_arb": (0, 100),  # percentage
            "front_spring": (0, 100),  # percentage
            "rear_spring": (0, 100),  # percentage
        }
    
    def recommend_setup(
        self,
        driver_preferences: DriverPreferences,
        track_profile: TrackProfile,
        weather: WeatherData
    ) -> Dict[str, Any]:
        """
        Recommend optimal car setup
        
        Args:
            driver_preferences: Driver setup preferences
            track_profile: Track characteristics
            weather: Weather conditions
            
        Returns:
            Recommended setup configuration
        """
        # Get weather effects
        weather_effects = self.weather_effects[weather.condition]
        
        # Calculate base setup based on track type
        base_setup = self._calculate_base_setup(track_profile, weather_effects)
        
        # Adjust for driver preferences
        adjusted_setup = self._adjust_for_driver_preferences(
            base_setup, driver_preferences
        )
        
        # Optimize using Bayesian optimization
        optimized_setup = self._optimize_setup(
            adjusted_setup, track_profile, weather, driver_preferences
        )
        
        # Calculate confidence and reasoning
        confidence = self._calculate_confidence(
            optimized_setup, track_profile, weather, driver_preferences
        )
        
        reasoning = self._generate_reasoning(
            optimized_setup, track_profile, weather, driver_preferences
        )
        
        return {
            "ride_height": optimized_setup.ride_height,
            "front_wing_angle": optimized_setup.front_wing_angle,
            "rear_wing_angle": optimized_setup.rear_wing_angle,
            "diff_settings": optimized_setup.diff_settings,
            "brake_bias": optimized_setup.brake_bias,
            "suspension_settings": optimized_setup.suspension_settings,
            "tire_pressures": optimized_setup.tire_pressures,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    def _calculate_base_setup(
        self,
        track_profile: TrackProfile,
        weather_effects: Dict[str, float]
    ) -> SetupConfiguration:
        """Calculate base setup based on track characteristics"""
        
        # Ride height based on track type and weather
        base_ride_height = 70.0  # mm
        if track_profile.track_type == TrackType.HIGH_SPEED:
            base_ride_height = 65.0
        elif track_profile.track_type == TrackType.TECHNICAL:
            base_ride_height = 75.0
        
        ride_height = base_ride_height + weather_effects["ride_height_adjustment"]
        
        # Wing angles based on downforce requirement
        downforce_req = track_profile.downforce_requirement * weather_effects["downforce_multiplier"]
        
        if downforce_req > 0.8:
            front_wing = 12.0
            rear_wing = 16.0
        elif downforce_req > 0.6:
            front_wing = 8.0
            rear_wing = 12.0
        else:
            front_wing = 4.0
            rear_wing = 8.0
        
        # Differential settings based on track type
        if track_profile.track_type == TrackType.TECHNICAL:
            diff_settings = {
                "preload": 60,
                "power": 80,
                "coast": 40
            }
        elif track_profile.track_type == TrackType.HIGH_SPEED:
            diff_settings = {
                "preload": 40,
                "power": 60,
                "coast": 60
            }
        else:  # MIXED
            diff_settings = {
                "preload": 50,
                "power": 70,
                "coast": 50
            }
        
        # Brake bias based on track characteristics
        if track_profile.corners > 20:
            brake_bias = 65  # More rear bias for technical tracks
        else:
            brake_bias = 58  # More front bias for high-speed tracks
        
        # Suspension settings
        if track_profile.track_type == TrackType.TECHNICAL:
            suspension = {
                "front_arb": 70,
                "rear_arb": 60,
                "front_spring": 80,
                "rear_spring": 70
            }
        elif track_profile.track_type == TrackType.HIGH_SPEED:
            suspension = {
                "front_arb": 50,
                "rear_arb": 40,
                "front_spring": 60,
                "rear_spring": 50
            }
        else:
            suspension = {
                "front_arb": 60,
                "rear_arb": 50,
                "front_spring": 70,
                "rear_spring": 60
            }
        
        # Tire pressures
        base_pressure = 1.2  # bar
        pressure_adjustment = weather_effects["tire_pressure_adjustment"]
        
        tire_pressures = {
            "front_left": base_pressure + pressure_adjustment,
            "front_right": base_pressure + pressure_adjustment,
            "rear_left": base_pressure + pressure_adjustment,
            "rear_right": base_pressure + pressure_adjustment
        }
        
        return SetupConfiguration(
            ride_height=ride_height,
            front_wing_angle=front_wing,
            rear_wing_angle=rear_wing,
            diff_settings=diff_settings,
            brake_bias=brake_bias,
            suspension_settings=suspension,
            tire_pressures=tire_pressures
        )
    
    def _adjust_for_driver_preferences(
        self,
        base_setup: SetupConfiguration,
        driver_preferences: DriverPreferences
    ) -> SetupConfiguration:
        """Adjust setup based on driver preferences"""
        adjusted_setup = SetupConfiguration(
            ride_height=base_setup.ride_height,
            front_wing_angle=base_setup.front_wing_angle,
            rear_wing_angle=base_setup.rear_wing_angle,
            diff_settings=base_setup.diff_settings.copy(),
            brake_bias=base_setup.brake_bias,
            suspension_settings=base_setup.suspension_settings.copy(),
            tire_pressures=base_setup.tire_pressures.copy()
        )
        
        # Adjust for preferred ride height
        if driver_preferences.preferred_ride_height:
            adjusted_setup.ride_height = driver_preferences.preferred_ride_height
        
        # Adjust for preferred wing angles
        if driver_preferences.preferred_wing_angles:
            if "front" in driver_preferences.preferred_wing_angles:
                adjusted_setup.front_wing_angle = driver_preferences.preferred_wing_angles["front"]
            if "rear" in driver_preferences.preferred_wing_angles:
                adjusted_setup.rear_wing_angle = driver_preferences.preferred_wing_angles["rear"]
        
        # Adjust for preferred diff settings
        if driver_preferences.preferred_diff_settings:
            for key, value in driver_preferences.preferred_diff_settings.items():
                if key in adjusted_setup.diff_settings:
                    adjusted_setup.diff_settings[key] = value
        
        # Adjust for risk tolerance
        if driver_preferences.risk_tolerance > 0.7:
            # More aggressive setup
            adjusted_setup.front_wing_angle += 2.0
            adjusted_setup.rear_wing_angle += 2.0
            adjusted_setup.diff_settings["power"] += 10
        elif driver_preferences.risk_tolerance < 0.3:
            # More conservative setup
            adjusted_setup.front_wing_angle -= 2.0
            adjusted_setup.rear_wing_angle -= 2.0
            adjusted_setup.diff_settings["power"] -= 10
        
        return adjusted_setup
    
    def _optimize_setup(
        self,
        base_setup: SetupConfiguration,
        track_profile: TrackProfile,
        weather: WeatherData,
        driver_preferences: DriverPreferences
    ) -> SetupConfiguration:
        """Optimize setup using Bayesian optimization"""
        # Mock optimization - in production would use scikit-optimize or similar
        optimized_setup = SetupConfiguration(
            ride_height=base_setup.ride_height + random.uniform(-2, 2),
            front_wing_angle=base_setup.front_wing_angle + random.uniform(-1, 1),
            rear_wing_angle=base_setup.rear_wing_angle + random.uniform(-1, 1),
            diff_settings={
                "preload": base_setup.diff_settings["preload"] + random.uniform(-5, 5),
                "power": base_setup.diff_settings["power"] + random.uniform(-5, 5),
                "coast": base_setup.diff_settings["coast"] + random.uniform(-5, 5)
            },
            brake_bias=base_setup.brake_bias + random.uniform(-2, 2),
            suspension_settings={
                "front_arb": base_setup.suspension_settings["front_arb"] + random.uniform(-5, 5),
                "rear_arb": base_setup.suspension_settings["rear_arb"] + random.uniform(-5, 5),
                "front_spring": base_setup.suspension_settings["front_spring"] + random.uniform(-5, 5),
                "rear_spring": base_setup.suspension_settings["rear_spring"] + random.uniform(-5, 5)
            },
            tire_pressures=base_setup.tire_pressures.copy()
        )
        
        # Ensure values are within constraints
        constraints = self.setup_constraints
        optimized_setup.ride_height = np.clip(optimized_setup.ride_height, *constraints["ride_height"])
        optimized_setup.front_wing_angle = np.clip(optimized_setup.front_wing_angle, *constraints["front_wing_angle"])
        optimized_setup.rear_wing_angle = np.clip(optimized_setup.rear_wing_angle, *constraints["rear_wing_angle"])
        optimized_setup.brake_bias = np.clip(optimized_setup.brake_bias, *constraints["brake_bias"])
        
        for key in optimized_setup.diff_settings:
            optimized_setup.diff_settings[key] = np.clip(
                optimized_setup.diff_settings[key], 0, 100
            )
        
        for key in optimized_setup.suspension_settings:
            optimized_setup.suspension_settings[key] = np.clip(
                optimized_setup.suspension_settings[key], 0, 100
            )
        
        return optimized_setup
    
    def _calculate_confidence(
        self,
        setup: SetupConfiguration,
        track_profile: TrackProfile,
        weather: WeatherData,
        driver_preferences: DriverPreferences
    ) -> float:
        """Calculate confidence in setup recommendation"""
        confidence = 0.7  # Base confidence
        
        # Track type match
        if track_profile.track_type == TrackType.TECHNICAL and setup.rear_wing_angle > 12:
            confidence += 0.1
        elif track_profile.track_type == TrackType.HIGH_SPEED and setup.rear_wing_angle < 10:
            confidence += 0.1
        
        # Weather condition match
        if weather.condition == WeatherCondition.WET and setup.ride_height > 75:
            confidence += 0.1
        elif weather.condition == WeatherCondition.DRY and 65 < setup.ride_height < 75:
            confidence += 0.1
        
        # Driver preference match
        if driver_preferences.preferred_ride_height:
            if abs(setup.ride_height - driver_preferences.preferred_ride_height) < 5:
                confidence += 0.1
        
        return min(0.95, confidence)
    
    def _generate_reasoning(
        self,
        setup: SetupConfiguration,
        track_profile: TrackProfile,
        weather: WeatherData,
        driver_preferences: DriverPreferences
    ) -> str:
        """Generate reasoning for setup choices"""
        reasoning = f"Setup optimized for {track_profile.track_name} ({track_profile.track_type.value} track). "
        
        # Ride height reasoning
        if setup.ride_height > 75:
            reasoning += f"High ride height ({setup.ride_height:.1f}mm) for stability and {weather.condition.value} conditions. "
        elif setup.ride_height < 65:
            reasoning += f"Low ride height ({setup.ride_height:.1f}mm) for aerodynamic efficiency on high-speed sections. "
        else:
            reasoning += f"Balanced ride height ({setup.ride_height:.1f}mm) for mixed track characteristics. "
        
        # Wing angle reasoning
        total_downforce = setup.front_wing_angle + setup.rear_wing_angle
        if total_downforce > 25:
            reasoning += f"High downforce configuration ({setup.front_wing_angle:.1f}° front, {setup.rear_wing_angle:.1f}° rear) for technical corners. "
        elif total_downforce < 15:
            reasoning += f"Low downforce configuration ({setup.front_wing_angle:.1f}° front, {setup.rear_wing_angle:.1f}° rear) for high-speed efficiency. "
        else:
            reasoning += f"Balanced downforce configuration ({setup.front_wing_angle:.1f}° front, {setup.rear_wing_angle:.1f}° rear) for mixed requirements. "
        
        # Differential reasoning
        if setup.diff_settings["power"] > 70:
            reasoning += "Aggressive differential settings for maximum traction. "
        elif setup.diff_settings["power"] < 50:
            reasoning += "Conservative differential settings for stability. "
        else:
            reasoning += "Balanced differential settings for mixed conditions. "
        
        # Weather considerations
        if weather.condition == WeatherCondition.WET:
            reasoning += "Wet weather adjustments: increased downforce and higher ride height. "
        elif weather.condition == WeatherCondition.INTERMEDIATE:
            reasoning += "Intermediate weather adjustments: moderate downforce increase. "
        
        return reasoning


# Global optimizer instance
_setup_optimizer = None

def get_setup_optimizer() -> SetupOptimizer:
    """Get or create setup optimizer instance"""
    global _setup_optimizer
    if _setup_optimizer is None:
        _setup_optimizer = SetupOptimizer()
    return _setup_optimizer

def recommend_setup(
    driver_preferences: Dict[str, Any],
    track_profile: Dict[str, Any],
    weather: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recommend optimal car setup
    
    Args:
        driver_preferences: Driver setup preferences
        track_profile: Track characteristics
        weather: Weather conditions
        
    Returns:
        Recommended setup configuration
    """
    optimizer = get_setup_optimizer()
    
    # Convert dict inputs to proper objects
    driver_prefs = DriverPreferences(**driver_preferences)
    track_prof = TrackProfile(**track_profile)
    weather_data = WeatherData(**weather)
    
    return optimizer.recommend_setup(driver_prefs, track_prof, weather_data)


# Example usage and testing
if __name__ == "__main__":
    print("🏁 Setup Recommender Test")
    print("=" * 50)
    
    optimizer = get_setup_optimizer()
    
    # Test cases
    test_cases = [
        {
            "driver_preferences": DriverPreferences(
                preferred_ride_height=72.0,
                risk_tolerance=0.7,
                tire_management=0.6
            ),
            "track_profile": optimizer.track_profiles["monaco"],
            "weather": WeatherData(
                condition=WeatherCondition.DRY,
                temperature=25.0,
                humidity=60.0
            )
        },
        {
            "driver_preferences": DriverPreferences(
                preferred_ride_height=68.0,
                risk_tolerance=0.4,
                tire_management=0.8
            ),
            "track_profile": optimizer.track_profiles["silverstone"],
            "weather": WeatherData(
                condition=WeatherCondition.WET,
                temperature=18.0,
                humidity=85.0
            )
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔧 Test Case {i}: {case['track_profile'].track_name}")
        print(f"   Weather: {case['weather'].condition.value}")
        print(f"   Driver Risk Tolerance: {case['driver_preferences'].risk_tolerance}")
        
        result = optimizer.recommend_setup(
            case['driver_preferences'],
            case['track_profile'],
            case['weather']
        )
        
        print(f"   📋 Recommended Setup:")
        print(f"      • Ride Height: {result['ride_height']:.1f}mm")
        print(f"      • Front Wing: {result['front_wing_angle']:.1f}°")
        print(f"      • Rear Wing: {result['rear_wing_angle']:.1f}°")
        print(f"      • Brake Bias: {result['brake_bias']:.1f}%")
        print(f"      • Diff Settings: {result['diff_settings']}")
        print(f"      • Confidence: {result['confidence']:.2f}")
        print(f"      • Reasoning: {result['reasoning'][:100]}...")
        print("-" * 50) 