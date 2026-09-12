#!/usr/bin/env python3
"""Car setup recommender using an Optuna TPE search over a transparent heuristic objective."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna


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
    preferred_ride_height: Optional[float] = None
    preferred_wing_angles: Optional[Dict[str, float]] = None
    preferred_diff_settings: Optional[Dict[str, float]] = None
    risk_tolerance: float = 0.5
    tire_management: float = 0.5


@dataclass
class TrackProfile:
    track_name: str
    track_length: float
    corners: int
    high_speed_sections: int
    low_speed_sections: int
    track_type: TrackType
    average_speed: float
    downforce_requirement: float


@dataclass
class WeatherData:
    condition: WeatherCondition
    temperature: float
    humidity: float
    wind_speed: Optional[float] = None


@dataclass
class SetupConfiguration:
    ride_height: float
    front_wing_angle: float
    rear_wing_angle: float
    diff_settings: Dict[str, float]
    brake_bias: float
    suspension_settings: Dict[str, float]
    tire_pressures: Dict[str, float]


class SetupOptimizer:
    """Recommend a setup with a reproducible TPE/Bayesian-style parameter search."""

    def __init__(self, n_trials: int = 48, seed: int = 42):
        self.n_trials = max(8, int(n_trials))
        self.seed = int(seed)
        self.track_profiles = self._load_track_profiles()
        self.weather_effects = self._load_weather_effects()
        self.setup_constraints = self._load_setup_constraints()

    def _load_track_profiles(self) -> Dict[str, TrackProfile]:
        return {
            "monaco": TrackProfile("Circuit de Monaco", 3337, 19, 2, 15, TrackType.TECHNICAL, 160, 0.90),
            "silverstone": TrackProfile("Silverstone Circuit", 5891, 18, 8, 4, TrackType.HIGH_SPEED, 220, 0.60),
            "spa": TrackProfile("Circuit de Spa-Francorchamps", 7004, 20, 10, 6, TrackType.MIXED, 200, 0.70),
            "singapore": TrackProfile("Marina Bay Street Circuit", 5063, 23, 3, 18, TrackType.TECHNICAL, 170, 0.85),
        }

    @staticmethod
    def _load_weather_effects() -> Dict[WeatherCondition, Dict[str, float]]:
        return {
            WeatherCondition.DRY: {"downforce_multiplier": 1.0, "ride_height_adjustment": 0.0, "tire_pressure_adjustment": 0.0},
            WeatherCondition.WET: {"downforce_multiplier": 1.15, "ride_height_adjustment": 5.0, "tire_pressure_adjustment": -0.10},
            WeatherCondition.INTERMEDIATE: {"downforce_multiplier": 1.08, "ride_height_adjustment": 2.5, "tire_pressure_adjustment": -0.05},
        }

    @staticmethod
    def _load_setup_constraints() -> Dict[str, Tuple[float, float]]:
        return {
            "ride_height": (60.0, 85.0),
            "front_wing_angle": (0.0, 15.0),
            "rear_wing_angle": (0.0, 20.0),
            "brake_bias": (50.0, 70.0),
            "diff_preload": (0.0, 100.0),
            "diff_power": (0.0, 100.0),
            "diff_coast": (0.0, 100.0),
            "front_arb": (0.0, 100.0),
            "rear_arb": (0.0, 100.0),
            "front_spring": (0.0, 100.0),
            "rear_spring": (0.0, 100.0),
        }

    def recommend_setup(
        self,
        driver_preferences: DriverPreferences,
        track_profile: TrackProfile,
        weather: WeatherData,
    ) -> Dict[str, Any]:
        self._validate_inputs(driver_preferences, track_profile, weather)
        target = self._calculate_target_setup(track_profile, weather, driver_preferences)
        optimized, objective_value = self._optimize_setup(target, track_profile, weather, driver_preferences)
        confidence = self._calculate_confidence(objective_value, driver_preferences)
        return {
            "ride_height": optimized.ride_height,
            "front_wing_angle": optimized.front_wing_angle,
            "rear_wing_angle": optimized.rear_wing_angle,
            "diff_settings": optimized.diff_settings,
            "brake_bias": optimized.brake_bias,
            "suspension_settings": optimized.suspension_settings,
            "tire_pressures": optimized.tire_pressures,
            "confidence": confidence,
            "reasoning": self._generate_reasoning(optimized, track_profile, weather),
            "optimization_method": "Optuna TPESampler",
            "trials": self.n_trials,
            "objective_value": objective_value,
            "model_scope": "heuristic setup search; not a vehicle-dynamics simulator",
        }

    @staticmethod
    def _validate_inputs(
        driver_preferences: DriverPreferences,
        track_profile: TrackProfile,
        weather: WeatherData,
    ) -> None:
        if track_profile.track_length <= 0 or track_profile.corners <= 0:
            raise ValueError("track_length and corners must be positive")
        if not 0.0 <= track_profile.downforce_requirement <= 1.0:
            raise ValueError("downforce_requirement must be between 0 and 1")
        if not 0.0 <= driver_preferences.risk_tolerance <= 1.0:
            raise ValueError("risk_tolerance must be between 0 and 1")
        if not 0.0 <= driver_preferences.tire_management <= 1.0:
            raise ValueError("tire_management must be between 0 and 1")
        if not 0.0 <= weather.humidity <= 100.0:
            raise ValueError("humidity must be between 0 and 100")

    def _calculate_target_setup(
        self,
        track: TrackProfile,
        weather: WeatherData,
        driver: DriverPreferences,
    ) -> SetupConfiguration:
        weather_effects = self.weather_effects[weather.condition]
        downforce = float(np.clip(track.downforce_requirement * weather_effects["downforce_multiplier"], 0.0, 1.0))

        ride_height = 70.0
        if track.track_type == TrackType.HIGH_SPEED:
            ride_height -= 4.0
        elif track.track_type in (TrackType.TECHNICAL, TrackType.LOW_SPEED):
            ride_height += 4.0
        ride_height += weather_effects["ride_height_adjustment"]

        front_wing = 3.0 + 10.0 * downforce
        rear_wing = 5.0 + 13.0 * downforce
        brake_bias = 58.0 + min(5.0, track.low_speed_sections * 0.20)

        diff = {
            "preload": 45.0 + 15.0 * (track.low_speed_sections / max(1, track.corners)),
            "power": 60.0 + 15.0 * driver.risk_tolerance,
            "coast": 55.0 - 10.0 * driver.risk_tolerance,
        }
        suspension = {
            "front_arb": 50.0 + 20.0 * downforce,
            "rear_arb": 45.0 + 15.0 * downforce,
            "front_spring": 55.0 + 20.0 * (track.high_speed_sections / max(1, track.corners)),
            "rear_spring": 50.0 + 15.0 * (track.high_speed_sections / max(1, track.corners)),
        }

        pressure = 1.20 + weather_effects["tire_pressure_adjustment"]
        pressures = {corner: pressure for corner in ("front_left", "front_right", "rear_left", "rear_right")}

        if driver.preferred_ride_height is not None:
            ride_height = driver.preferred_ride_height
        if driver.preferred_wing_angles:
            front_wing = driver.preferred_wing_angles.get("front", front_wing)
            rear_wing = driver.preferred_wing_angles.get("rear", rear_wing)
        if driver.preferred_diff_settings:
            for key in diff:
                if key in driver.preferred_diff_settings:
                    diff[key] = driver.preferred_diff_settings[key]

        return SetupConfiguration(
            ride_height=float(np.clip(ride_height, *self.setup_constraints["ride_height"])),
            front_wing_angle=float(np.clip(front_wing, *self.setup_constraints["front_wing_angle"])),
            rear_wing_angle=float(np.clip(rear_wing, *self.setup_constraints["rear_wing_angle"])),
            diff_settings={key: float(np.clip(value, 0.0, 100.0)) for key, value in diff.items()},
            brake_bias=float(np.clip(brake_bias, *self.setup_constraints["brake_bias"])),
            suspension_settings={key: float(np.clip(value, 0.0, 100.0)) for key, value in suspension.items()},
            tire_pressures=pressures,
        )

    def _objective(
        self,
        candidate: SetupConfiguration,
        target: SetupConfiguration,
        track: TrackProfile,
        weather: WeatherData,
        driver: DriverPreferences,
    ) -> float:
        # Normalised distance from a domain-informed target.
        loss = 0.0
        loss += ((candidate.ride_height - target.ride_height) / 8.0) ** 2
        loss += ((candidate.front_wing_angle - target.front_wing_angle) / 4.0) ** 2
        loss += ((candidate.rear_wing_angle - target.rear_wing_angle) / 5.0) ** 2
        loss += ((candidate.brake_bias - target.brake_bias) / 5.0) ** 2
        for key in target.diff_settings:
            loss += 0.25 * ((candidate.diff_settings[key] - target.diff_settings[key]) / 20.0) ** 2
        for key in target.suspension_settings:
            loss += 0.15 * ((candidate.suspension_settings[key] - target.suspension_settings[key]) / 20.0) ** 2

        # Extra penalties encode broad trade-offs rather than claiming physical fidelity.
        if track.track_type == TrackType.HIGH_SPEED:
            loss += max(0.0, candidate.rear_wing_angle - 14.0) * 0.03
        if weather.condition == WeatherCondition.WET:
            loss += max(0.0, 73.0 - candidate.ride_height) * 0.05
        if driver.tire_management < 0.4:
            loss += max(0.0, candidate.diff_settings["power"] - 75.0) * 0.02
        return float(loss)

    def _optimize_setup(
        self,
        target: SetupConfiguration,
        track: TrackProfile,
        weather: WeatherData,
        driver: DriverPreferences,
    ) -> Tuple[SetupConfiguration, float]:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            c = self.setup_constraints
            candidate = SetupConfiguration(
                ride_height=trial.suggest_float("ride_height", *c["ride_height"]),
                front_wing_angle=trial.suggest_float("front_wing_angle", *c["front_wing_angle"]),
                rear_wing_angle=trial.suggest_float("rear_wing_angle", *c["rear_wing_angle"]),
                diff_settings={
                    "preload": trial.suggest_float("diff_preload", *c["diff_preload"]),
                    "power": trial.suggest_float("diff_power", *c["diff_power"]),
                    "coast": trial.suggest_float("diff_coast", *c["diff_coast"]),
                },
                brake_bias=trial.suggest_float("brake_bias", *c["brake_bias"]),
                suspension_settings={
                    "front_arb": trial.suggest_float("front_arb", *c["front_arb"]),
                    "rear_arb": trial.suggest_float("rear_arb", *c["rear_arb"]),
                    "front_spring": trial.suggest_float("front_spring", *c["front_spring"]),
                    "rear_spring": trial.suggest_float("rear_spring", *c["rear_spring"]),
                },
                tire_pressures=target.tire_pressures.copy(),
            )
            return self._objective(candidate, target, track, weather, driver)

        # Seed the search with the calculated target, then let TPE explore around it.
        study.enqueue_trial(
            {
                "ride_height": target.ride_height,
                "front_wing_angle": target.front_wing_angle,
                "rear_wing_angle": target.rear_wing_angle,
                "diff_preload": target.diff_settings["preload"],
                "diff_power": target.diff_settings["power"],
                "diff_coast": target.diff_settings["coast"],
                "brake_bias": target.brake_bias,
                "front_arb": target.suspension_settings["front_arb"],
                "rear_arb": target.suspension_settings["rear_arb"],
                "front_spring": target.suspension_settings["front_spring"],
                "rear_spring": target.suspension_settings["rear_spring"],
            }
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        p = study.best_params
        optimized = SetupConfiguration(
            ride_height=float(p["ride_height"]),
            front_wing_angle=float(p["front_wing_angle"]),
            rear_wing_angle=float(p["rear_wing_angle"]),
            diff_settings={"preload": float(p["diff_preload"]), "power": float(p["diff_power"]), "coast": float(p["diff_coast"])},
            brake_bias=float(p["brake_bias"]),
            suspension_settings={
                "front_arb": float(p["front_arb"]),
                "rear_arb": float(p["rear_arb"]),
                "front_spring": float(p["front_spring"]),
                "rear_spring": float(p["rear_spring"]),
            },
            tire_pressures=target.tire_pressures.copy(),
        )
        return optimized, float(study.best_value)

    @staticmethod
    def _calculate_confidence(objective_value: float, driver: DriverPreferences) -> float:
        preference_bonus = 0.05 if any(
            x is not None
            for x in (driver.preferred_ride_height, driver.preferred_wing_angles, driver.preferred_diff_settings)
        ) else 0.0
        return float(np.clip(0.82 - min(objective_value, 2.0) * 0.08 + preference_bonus, 0.55, 0.90))

    @staticmethod
    def _generate_reasoning(setup: SetupConfiguration, track: TrackProfile, weather: WeatherData) -> str:
        return (
            f"TPE search targeted a {track.track_type.value} setup for {track.track_name}. "
            f"The selected configuration uses {setup.front_wing_angle:.1f}°/{setup.rear_wing_angle:.1f}° front/rear wing, "
            f"{setup.ride_height:.1f} mm ride height and {setup.brake_bias:.1f}% brake bias. "
            f"Weather mode is {weather.condition.value}. Values are heuristic engineering recommendations and require simulator/track validation."
        )


_setup_optimizer: Optional[SetupOptimizer] = None


def get_setup_optimizer() -> SetupOptimizer:
    global _setup_optimizer
    if _setup_optimizer is None:
        _setup_optimizer = SetupOptimizer()
    return _setup_optimizer


def recommend_setup(
    driver_preferences: Dict[str, Any],
    track_profile: Dict[str, Any],
    weather: Dict[str, Any],
) -> Dict[str, Any]:
    optimizer = get_setup_optimizer()

    driver_data = dict(driver_preferences)
    track_data = dict(track_profile)
    weather_data = dict(weather)

    if isinstance(track_data.get("track_type"), str):
        track_data["track_type"] = TrackType(track_data["track_type"])
    if isinstance(weather_data.get("condition"), str):
        weather_data["condition"] = WeatherCondition(weather_data["condition"])

    return optimizer.recommend_setup(
        DriverPreferences(**driver_data),
        TrackProfile(**track_data),
        WeatherData(**weather_data),
    )
