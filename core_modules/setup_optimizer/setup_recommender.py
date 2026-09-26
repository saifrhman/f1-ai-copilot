#!/usr/bin/env python3
"""Car setup recommender: a seeded Optuna TPE search over a documented HEURISTIC objective.

Model scope
-----------
This is not a vehicle-dynamics simulator. The objective is a dimensionless
lap-time/handling *proxy* (lower is better) built from simple, bounded,
commented trade-off terms:

* cornering grip from downforce vs straight-line drag, weighted by the track's
  downforce requirement and average speed;
* high-speed and low-speed handling balance (aero balance, roll-stiffness
  balance) vs the driver's preferred balance (risk tolerance, tyre management);
* low ride height (more floor downforce, lower centre of gravity) vs bottoming
  risk (kerbs, standing water, speed; stiffer springs allow running lower);
* spring/anti-roll-bar stiffness: mechanical compliance vs aero platform control
  and steering response;
* brake bias vs front lock-up / rear instability;
* differential (clutch-pack LSD): preload is a constant lock present in every
  phase, the power/coast ramps add lock on top of it; lock buys traction and
  entry stability but costs rotation, and preload on its own steadies the car
  through throttle transitions in fast corners but pushes the front at the apex
  of tight corners.

The terms interact (for example the ideal brake bias depends on downforce and
coast lock, the bottoming limit depends on spring stiffness), so the optimum
has no closed form. Every constant below is a heuristic assumption chosen for
plausible *relative* behaviour; absolute values must be validated in a
simulator or on track.

Some parameters legitimately end on a bound at the optimum: minimum wing on
low-downforce tracks and maximum wing in the wet or on high-downforce tracks.
Two corner solutions are artefacts of the heuristic's linear costs rather than
engineering advice: over a 162-scenario grid (6 tracks x 9 driver styles x 3
weathers) the rear anti-roll bar ends at its softest setting in ~70% of
scenarios and the front spring at its stiffest in ~60%, because rear roll
stiffness pays linear traction/wear costs while roll-stiffness balance is
cheaper to set with the front spring. Treat those two values as "softest /
stiffest the heuristic allows", not as tuned numbers.

Search
------
1. A seeded Optuna TPE study (``n_trials``, the first ``N_STARTUP_TRIALS`` random)
   explores the whole search space.
2. A deterministic local refinement (compass search, pure Python, inside
   ``SETUP_BOUNDS``) is started from the best TPE trial, from the rule-of-thumb
   baseline and from the centre of the bounds. TPE alone stops short of the
   optimum at practical budgets, so without this step the answer would mostly
   depend on the seed. The best refined setup is returned (the TPE start wins
   ties); the TPE best trial and the refinement gain are reported separately.
3. ``confidence`` is the share of the refinement starts that reached the
   returned setup (see ``CONFIDENCE_METHOD``); ``parameter_spread`` shows how far
   the starts' optima differ per parameter.

The closed-form rule of thumb is only a reported ``baseline``; it is never
enqueued into the TPE study. Because it is also a refinement start, the returned
setup is never worse than it.

Thread safety: every call builds its own sampler and in-memory study and the
module holds no mutable state, so concurrent calls are independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import optuna

from core_modules.strategy_optimizer.strategy_engine import WeatherCondition

# TPE budget. Measured TPE latency is ~0.6-0.8 s at 128 trials and ~2.3 s at 300; the local refinement
# adds ~15-20 ms per start (three starts). Small budgets are allowed for quick checks: the refinement
# still converges, the TPE stage just contributes less (``best_trial_objective`` shows it).
DEFAULT_N_TRIALS = 128
MIN_N_TRIALS = 16
MAX_N_TRIALS = 300
DEFAULT_SEED = 42
MAX_SEED = 2**32 - 1
N_STARTUP_TRIALS = 10  # random trials TPE runs before it starts modelling the objective

# Local refinement (compass search). Steps are fractions of each parameter's range.
REFINE_INITIAL_STEP = 0.1
REFINE_MIN_STEP = 1e-4  # converged when no +/- move of this size improves the objective
REFINE_MAX_EVALUATIONS = 20_000  # per start; a hard CPU cap (converged runs used ~0.5-1.5k)
AGREEMENT_TOLERANCE = 0.02  # a start "agrees" if every parameter is within 2% of its range
SELECTION_TOLERANCE = 1e-6  # objective difference below which the TPE start is preferred

# Search space (setup-sheet units). The 0-100 scales are % of the adjuster range.
SETUP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ride_height": (60.0, 85.0),  # mm
    "front_wing_angle": (0.0, 15.0),  # deg
    "rear_wing_angle": (0.0, 20.0),  # deg
    "brake_bias": (50.0, 70.0),  # % of braking force on the front axle
    "diff_preload": (0.0, 100.0),  # % lock
    "diff_power": (0.0, 100.0),  # % lock on throttle
    "diff_coast": (0.0, 100.0),  # % lock off throttle
    "front_arb": (0.0, 100.0),  # 0 = softest, 100 = stiffest
    "rear_arb": (0.0, 100.0),
    "front_spring": (0.0, 100.0),
    "rear_spring": (0.0, 100.0),
}

OUTPUT_UNITS: Dict[str, str] = {
    "ride_height": "mm",
    "front_wing_angle": "deg",
    "rear_wing_angle": "deg",
    "brake_bias": "% front",
    "diff_settings": "% lock (0-100 adjuster scale)",
    "suspension_settings": "0-100 stiffness scale (0 = softest)",
    "tire_pressures_psi": "psi (gauge), cold set pressure",
    "objective_value": "dimensionless heuristic penalty (lower is better)",
    "handling_balance": "dimensionless heuristic index (> 0 understeer, < 0 oversteer)",
    "parameter_spread": "setup-sheet units of each parameter (max - min over the refinement starts)",
}

# Input plausibility limits.
TRACK_LENGTH_RANGE_M = (500.0, 25_000.0)
AVERAGE_SPEED_RANGE_KPH = (20.0, 400.0)
MAX_CORNERS = 100
TRACK_NAME_MAX_LENGTH = 100
TEMPERATURE_RANGE_C = (-10.0, 50.0)  # ambient air temperature
MAX_WIND_SPEED_MS = 60.0

PREFERRED_WING_KEYS = {"front": "front_wing_angle", "rear": "rear_wing_angle"}
PREFERRED_DIFF_KEYS = {"preload": "diff_preload", "power": "diff_power", "coast": "diff_coast"}

MODEL_SCOPE = "heuristic setup search over a documented proxy objective; not a vehicle-dynamics simulator"
OBJECTIVE_DESCRIPTION = (
    "Heuristic lap-time/handling proxy (dimensionless, lower is better): downforce vs drag, "
    "high/low-speed balance vs driver preference, ride height vs bottoming, stiffness vs compliance, "
    "brake bias vs stability, diff traction vs rotation and rear-tyre wear, diff preload transition "
    "stability vs apex understeer."
)
REFINEMENT_METHOD = (
    "Deterministic compass search inside the setup bounds (+/- steps per free parameter, step halved "
    f"when no move improves, stopped below {REFINE_MIN_STEP:g} of each range), started from the best TPE "
    "trial, the rule-of-thumb baseline and the centre of the bounds; the best refined setup is returned."
)
CONFIDENCE_METHOD = (
    "Multi-start agreement on the heuristic objective, 0..1: the share of the refinement starts (best TPE "
    "trial, rule-of-thumb baseline, centre of the bounds) whose refined setup lies within "
    f"{AGREEMENT_TOLERANCE:.0%} of every parameter's range of the returned setup. 1.0 means every start "
    "reached the same optimum, so the answer did not depend on the seed or start point; lower values mean "
    "the objective has several local optima or is nearly flat in some parameters (parameter_spread shows "
    "which ones and by how much). It is not a probability that the setup is right and does not measure how "
    "accurate the heuristic model is for a real car."
)


def not_modelled_inputs(weather: WeatherData) -> List[str]:
    """Supplied inputs that change no number: the track name (always a label) and any given humidity/wind."""

    optional = [f"weather.{name}" for name in ("humidity", "wind_speed") if getattr(weather, name) is not None]
    return ["track_profile.track_name (label only)", *optional]


class TrackType(Enum):
    HIGH_SPEED = "high_speed"
    TECHNICAL = "technical"
    MIXED = "mixed"
    LOW_SPEED = "low_speed"


# Heuristic: how hard the car rides kerbs/bumps by layout type (0 = smooth, 1 = severe).
# Slow/technical layouts are more often kerb-heavy street circuits.
KERB_SEVERITY = {
    TrackType.HIGH_SPEED: 0.3,
    TrackType.MIXED: 0.5,
    TrackType.TECHNICAL: 0.7,
    TrackType.LOW_SPEED: 0.8,
}
# Heuristic: available grip as a fraction of dry grip, and how much standing water there is.
GRIP_LEVEL = {WeatherCondition.DRY: 1.0, WeatherCondition.INTERMEDIATE: 0.8, WeatherCondition.WET: 0.65}
STANDING_WATER = {WeatherCondition.DRY: 0.0, WeatherCondition.INTERMEDIATE: 0.5, WeatherCondition.WET: 1.0}

# Tyre pressures (rule-based, not searched). Assumptions: tyres are set cold at ambient
# temperature without blankets, and reach the running gas temperature below; the cold
# pressure follows from the ideal-gas law at constant volume (Gay-Lussac) so the running
# pressure hits the assumed target. Replace the targets with the tyre supplier's prescription.
ATMOSPHERIC_PRESSURE_PSI = 14.696
TARGET_RUNNING_PRESSURE_PSI = {  # (front, rear) gauge
    WeatherCondition.DRY: (27.0, 25.0),
    WeatherCondition.INTERMEDIATE: (25.0, 23.0),
    WeatherCondition.WET: (23.5, 21.5),
}
RUNNING_GAS_TEMPERATURE_C = {
    WeatherCondition.DRY: 100.0,
    WeatherCondition.INTERMEDIATE: 75.0,
    WeatherCondition.WET: 60.0,
}


@dataclass
class DriverPreferences:
    """Driver inputs. Preferred values are hard constraints: they pin that parameter in the search."""

    preferred_ride_height: Optional[float] = None
    preferred_wing_angles: Optional[Dict[str, float]] = None  # keys: front, rear
    preferred_diff_settings: Optional[Dict[str, float]] = None  # keys: preload, power, coast
    risk_tolerance: float = 0.5  # 0 = wants a stable car, 1 = accepts a pointy car
    tire_management: float = 0.5  # 0 = ignore tyre wear, 1 = protect the tyres


@dataclass
class TrackProfile:
    track_name: Optional[str]  # label for the reasoning text only; not modelled
    track_length: float  # m
    corners: int
    high_speed_sections: int  # corners taken at high speed
    low_speed_sections: int  # corners taken at low speed
    track_type: TrackType
    average_speed: float  # km/h
    downforce_requirement: float  # 0..1


@dataclass
class WeatherData:
    condition: WeatherCondition
    temperature: float  # ambient air temperature, deg C
    humidity: Optional[float] = None  # %, validated but not modelled
    wind_speed: Optional[float] = None  # m/s, validated but not modelled


@dataclass(frozen=True)
class SetupInputs:
    """Everything one recommendation needs; built by ``schemas.SetupRequest.to_engine_inputs``.

    ``assumed_defaults`` names the model inputs the request left out, so the engine's default was
    used (e.g. ``"driver_preferences.risk_tolerance=0.5"``); it is reported, never modelled.
    """

    driver_preferences: DriverPreferences
    track_profile: TrackProfile
    weather: WeatherData
    n_trials: int = DEFAULT_N_TRIALS
    seed: int = DEFAULT_SEED
    assumed_defaults: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SetupConfiguration:
    ride_height: float
    front_wing_angle: float
    rear_wing_angle: float
    brake_bias: float
    diff_preload: float
    diff_power: float
    diff_coast: float
    front_arb: float
    rear_arb: float
    front_spring: float
    rear_spring: float

    @classmethod
    def from_params(cls, params: Mapping[str, float]) -> "SetupConfiguration":
        return cls(**{name: float(params[name]) for name in SETUP_BOUNDS})

    def as_params(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in SETUP_BOUNDS}

    def normalised(self) -> Dict[str, float]:
        """Each parameter mapped to 0..1 across its search bounds."""
        return {name: (getattr(self, name) - lo) / (hi - lo) for name, (lo, hi) in SETUP_BOUNDS.items()}

    def to_output(self, exact: Collection[str] = (), decimals: int = 2) -> Dict[str, Any]:
        """Setup-sheet output rounded to ``decimals``; parameters in ``exact`` (driver pins) are not rounded."""
        v = {name: value if name in exact else round(value, decimals) for name, value in self.as_params().items()}
        return {
            "ride_height": v["ride_height"],
            "front_wing_angle": v["front_wing_angle"],
            "rear_wing_angle": v["rear_wing_angle"],
            "brake_bias": v["brake_bias"],
            "diff_settings": {"preload": v["diff_preload"], "power": v["diff_power"], "coast": v["diff_coast"]},
            "suspension_settings": {
                "front_arb": v["front_arb"],
                "rear_arb": v["rear_arb"],
                "front_spring": v["front_spring"],
                "rear_spring": v["rear_spring"],
            },
        }


@dataclass(frozen=True)
class _ModelContext:
    """Per-request factors derived once from the inputs (all roughly 0..1)."""

    speed: float  # average speed mapped 140..260 km/h -> 0..1
    high_speed_share: float
    low_speed_share: float
    kerb_severity: float
    grip: float
    water: float
    risk: float
    tyre_care: float
    w_downforce: float
    w_drag: float
    w_mech: float
    w_high_speed_balance: float
    w_low_speed_balance: float
    traction_need: float
    preferred_balance: float  # >0 = understeer, <0 = oversteer


@dataclass(frozen=True)
class _SearchResult:
    best: SetupConfiguration
    best_value: float
    best_trial_number: int
    values: List[float]


@dataclass(frozen=True)
class _Refinement:
    start: str  # which start point the local search began from
    start_value: float
    setup: SetupConfiguration
    value: float
    evaluations: int
    converged: bool  # False if REFINE_MAX_EVALUATIONS stopped it before the step tolerance


# --------------------------------------------------------------------------- validation


def _require_number(name: str, value: Any, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}; the value is out of range") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if not lo <= number <= hi:
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}, got {value!r}")
    return number


def _require_int(name: str, value: Any, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _validate_preference_map(name: str, value: Any, allowed: Mapping[str, str]) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object with keys {sorted(allowed)}")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{name} has unknown keys {unknown}; allowed keys are {sorted(allowed)}")
    for key, param in allowed.items():
        if key in value:
            _require_number(f"{name}.{key}", value[key], *SETUP_BOUNDS[param])


def validate_driver_preferences(driver: DriverPreferences) -> None:
    _require_number("risk_tolerance", driver.risk_tolerance, 0.0, 1.0)
    _require_number("tire_management", driver.tire_management, 0.0, 1.0)
    if driver.preferred_ride_height is not None:
        _require_number("preferred_ride_height", driver.preferred_ride_height, *SETUP_BOUNDS["ride_height"])
    _validate_preference_map("preferred_wing_angles", driver.preferred_wing_angles, PREFERRED_WING_KEYS)
    _validate_preference_map("preferred_diff_settings", driver.preferred_diff_settings, PREFERRED_DIFF_KEYS)


def validate_track_profile(track: TrackProfile) -> None:
    if track.track_name is not None:
        if not isinstance(track.track_name, str) or not track.track_name.strip():
            raise ValueError("track_name must be a non-empty string or null")
        if len(track.track_name) > TRACK_NAME_MAX_LENGTH:
            raise ValueError(f"track_name must be at most {TRACK_NAME_MAX_LENGTH} characters")
    if not isinstance(track.track_type, TrackType):
        raise ValueError(f"track_type must be one of {[t.value for t in TrackType]}")
    _require_number("track_length", track.track_length, *TRACK_LENGTH_RANGE_M)
    _require_number("average_speed", track.average_speed, *AVERAGE_SPEED_RANGE_KPH)
    _require_number("downforce_requirement", track.downforce_requirement, 0.0, 1.0)
    corners = _require_int("corners", track.corners, 1, MAX_CORNERS)
    high = _require_int("high_speed_sections", track.high_speed_sections, 0, corners)
    low = _require_int("low_speed_sections", track.low_speed_sections, 0, corners)
    if high + low > corners:
        raise ValueError(
            f"high_speed_sections + low_speed_sections ({high + low}) cannot exceed corners ({corners}); "
            "both count corners, the remainder are treated as medium-speed corners"
        )


def validate_weather(weather: WeatherData) -> None:
    if not isinstance(weather.condition, WeatherCondition):
        raise ValueError(f"condition must be one of {[c.value for c in WeatherCondition]}")
    _require_number("temperature", weather.temperature, *TEMPERATURE_RANGE_C)
    if weather.humidity is not None:
        _require_number("humidity", weather.humidity, 0.0, 100.0)
    if weather.wind_speed is not None:
        _require_number("wind_speed", weather.wind_speed, 0.0, MAX_WIND_SPEED_MS)


def validate_search_settings(n_trials: Any, seed: Any) -> Tuple[int, int]:
    return (
        _require_int("n_trials", n_trials, MIN_N_TRIALS, MAX_N_TRIALS),
        _require_int("seed", seed, 0, MAX_SEED),
    )


# --------------------------------------------------------------------------- heuristic model


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _saturate(x: float, rate: float) -> float:
    """Diminishing returns: 0 at x=0, 1 at x=1."""
    return (1.0 - math.exp(-rate * x)) / (1.0 - math.exp(-rate))


def _softplus(x: float) -> float:
    """Smooth max(0, x): ~0 well below zero, grows linearly above it (numerically stable form)."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _derive_context(driver: DriverPreferences, track: TrackProfile, weather: WeatherData) -> _ModelContext:
    dfr = float(track.downforce_requirement)
    speed = _clip01((float(track.average_speed) - 140.0) / 120.0)
    high = track.high_speed_sections / track.corners
    low = track.low_speed_sections / track.corners
    medium = 1.0 - high - low
    corners_per_km = track.corners / (float(track.track_length) / 1000.0)
    corner_density = _clip01((corners_per_km - 1.5) / 4.5)  # ~1.9/km (Monza) .. ~5.7/km (Monaco)
    water = STANDING_WATER[weather.condition]
    risk = float(driver.risk_tolerance)
    tyre_care = float(driver.tire_management)
    w_mech = 0.3 + 0.7 * (0.5 * low + 0.5 * corner_density)
    return _ModelContext(
        speed=speed,
        high_speed_share=high,
        low_speed_share=low,
        kerb_severity=KERB_SEVERITY[track.track_type],
        grip=GRIP_LEVEL[weather.condition],
        water=water,
        risk=risk,
        tyre_care=tyre_care,
        # Downforce is worth more on high-requirement tracks and in the wet.
        w_downforce=2.0 * (0.25 + 0.75 * dfr) * (1.0 + 0.4 * water),
        # Drag costs more on fast, low-downforce tracks; less in the wet (lower top speeds).
        w_drag=(0.25 + 0.75 * (0.5 * speed + 0.5 * (1.0 - dfr))) * (1.0 - 0.3 * water),
        # Mechanical grip matters on slow, corner-dense layouts.
        w_mech=w_mech,
        w_high_speed_balance=0.3 + 0.7 * (high + 0.5 * medium),
        w_low_speed_balance=0.3 + 0.7 * (low + 0.5 * medium),
        traction_need=w_mech * (0.5 + 0.5 * low) * (1.0 + 0.6 * water),
        # Tyre-savers and the wet favour mild understeer; risk-takers favour a pointy car.
        preferred_balance=0.15 * (tyre_care - 0.5) - 0.2 * (risk - 0.5) + 0.1 * water,
    )


def _balance_penalty(balance: float, ctx: _ModelContext) -> float:
    """Squared distance from the preferred balance; unwanted oversteer costs more for cautious drivers."""
    error = balance - ctx.preferred_balance
    weight = 1.0 + (1.0 - ctx.risk) if error < 0.0 else 1.0
    return weight * error * error


def _balance_indices(n: Mapping[str, float]) -> Tuple[float, float]:
    """HEURISTIC (high-speed, low-speed) balance from normalised parameters: > 0 understeer, < 0 oversteer.

    A front aero share of ~0.45 and even roll stiffness are neutral. Aero dominates at high speed,
    mechanical roll-stiffness balance at low speed.
    """
    fw, rw = n["front_wing_angle"], n["rear_wing_angle"]
    wing_load = 0.45 * fw + 0.55 * rw
    front_aero_share = (0.45 * fw + 0.08) / (wing_load + 0.16)  # 0.08/0.16 = body/floor load, split evenly
    aero_understeer = 2.0 * (0.45 - front_aero_share)
    front_roll = n["front_arb"] + 0.6 * n["front_spring"] + 0.1
    rear_roll = n["rear_arb"] + 0.6 * n["rear_spring"] + 0.1
    mech_understeer = 1.5 * (front_roll / (front_roll + rear_roll) - 0.5)
    return 0.7 * aero_understeer + 0.3 * mech_understeer, 0.3 * aero_understeer + 0.7 * mech_understeer


def handling_balance(setup: SetupConfiguration, ctx: _ModelContext) -> Dict[str, float]:
    """HEURISTIC balance indices of a setup and the driver's target balance (> 0 understeer)."""
    high, low = _balance_indices(setup.normalised())
    return {"high_speed": high, "low_speed": low, "driver_target": ctx.preferred_balance}


def objective_terms(setup: SetupConfiguration, ctx: _ModelContext) -> Dict[str, float]:
    """HEURISTIC objective broken into its terms. Each term is >= 0 and bounded by the search bounds."""
    n = setup.normalised()
    fw, rw, h = n["front_wing_angle"], n["rear_wing_angle"], n["ride_height"]
    preload, power, coast = n["diff_preload"], n["diff_power"], n["diff_coast"]
    farb, rarb, fspr, rspr = n["front_arb"], n["rear_arb"], n["front_spring"], n["rear_spring"]

    # Aero: wings plus floor; the floor gains downforce as the car runs lower and adds no drag.
    wing_load = 0.45 * fw + 0.55 * rw
    downforce = 0.55 * wing_load + 0.45 * (0.3 + 0.7 * (1.0 - h))  # 0.135 .. 1
    drag = 0.35 * fw + 0.65 * rw  # the rear wing is the main drag source
    cornering = ctx.w_downforce * (1.0 - _saturate(downforce, 2.0))  # diminishing returns
    straights = ctx.w_drag * drag

    high_speed_balance, low_speed_balance = _balance_indices(n)
    # Weighted x2 so cheap front-wing downforce cannot be bought at the expense of balance.
    balance = 2.0 * (
        ctx.w_high_speed_balance * _balance_penalty(high_speed_balance, ctx)
        + ctx.w_low_speed_balance * _balance_penalty(low_speed_balance, ctx)
    )

    # Ride height vs bottoming: kerbs, standing water (aquaplaning/plank) and aero load raise the
    # safe minimum; stiffer springs reduce suspension travel so the car can run lower.
    springs = 0.5 * (fspr + rspr)
    critical_height_mm = 62.0 + 8.0 * ctx.kerb_severity + 6.0 * ctx.water + 4.0 * ctx.speed - 6.0 * springs
    # Softplus, not a saturating curve: running further below the limit must keep getting worse.
    bottoming = 0.25 * _softplus((critical_height_mm - setup.ride_height) / 1.5)
    # A higher car raises the centre of gravity: more load transfer, less mechanical grip.
    centre_of_gravity = 0.25 * ctx.w_mech * h

    # Stiffness: compliance over kerbs/bumps (worse in the wet) vs aero platform control (springs)
    # and direction-change response (anti-roll bars).
    stiffness = 0.25 * (farb + rarb + fspr + rspr)
    compliance = 0.4 * ctx.w_mech * (0.5 + ctx.kerb_severity) * (1.0 + ctx.water) * stiffness**2
    platform = 0.3 * ctx.w_downforce * (0.5 + 0.5 * ctx.high_speed_share) * (1.0 - springs) ** 2
    response = 0.3 * ctx.w_mech * (1.0 - 0.5 * (farb + rarb)) ** 2

    # Differential (clutch-pack LSD): preload is a constant lock present in every phase; the ramps add
    # lock in proportion to drive (power) or engine-braking (coast) torque on top of it, up to 100%.
    base_lock = 0.5 * preload  # full preload locks half as hard as a fully loaded ramp
    lock_on = base_lock + (1.0 - base_lock) * power  # 0..1
    lock_off = base_lock + (1.0 - base_lock) * coast  # 0..1

    # Brake bias: more grip/downforce -> more forward weight transfer -> more front bias.
    # Off-throttle lock stabilises the rear on entry and risk-takers accept a more rearward bias.
    deceleration = ctx.grip * (0.5 + 0.5 * downforce)
    ideal_bias = 52.0 + 12.0 * deceleration - 2.0 * lock_off - 1.5 * (ctx.risk - 0.5)
    bias_error = (setup.brake_bias - ideal_bias) / 4.0
    if bias_error > 0.0:  # front lock-ups flat-spot tyres
        brake = 0.15 * bias_error**2 * (1.0 + ctx.tyre_care)
    else:  # rear instability on entry
        brake = 0.225 * bias_error**2 * (1.5 - ctx.risk)
    # Little off-throttle lock lets the rear step out on lift-off/entry; worse in the wet and for
    # cautious drivers.
    entry_stability = 0.25 * (1.0 - lock_off) ** 2 * (1.5 - ctx.risk) * (1.0 + ctx.water)

    # On-throttle lock buys traction with diminishing returns but stops the car rotating on exit;
    # off-throttle lock resists turn-in. Risk-takers value rotation more. A stiff rear end also loses
    # rear grip on exit.
    traction = ctx.traction_need * (1.0 - _saturate(lock_on, 3.0) + 0.4 * (0.6 * rarb + 0.4 * rspr))
    rotation = ctx.w_low_speed_balance * (0.5 + ctx.risk) * (0.3 * lock_on**2 + 0.2 * lock_off**2)
    # Mid-corner, near zero torque, neither ramp is loaded and only preload locks the axle. Some preload
    # keeps the rear settled through throttle transitions in fast corners (more so for cautious drivers
    # and in the wet) ...
    transition = 0.15 * ctx.w_high_speed_balance * (1.5 - ctx.risk) * (1.0 + 0.5 * ctx.water) * (1.0 - preload) ** 2
    # ... but it fights the inner/outer wheel-speed difference in tight corners: apex understeer and scrub.
    apex_push = 0.3 * ctx.w_low_speed_balance * (0.5 + ctx.tyre_care) * preload**2
    # Rear-tyre wear/scrub from on-throttle lock and a stiff rear bar, weighted by the driver's tyre focus.
    wear = 0.3 * ctx.tyre_care * (0.7 * lock_on + 0.3 * rarb) * (0.5 + 0.5 * ctx.low_speed_share)

    return {
        "cornering_downforce": cornering,
        "straight_line_drag": straights,
        "handling_balance": balance,
        "bottoming_risk": bottoming,
        "centre_of_gravity": centre_of_gravity,
        "suspension_compliance": compliance,
        "aero_platform": platform,
        "steering_response": response,
        "brake_stability": brake,
        "entry_stability": entry_stability,
        "traction": traction,
        "rotation": rotation,
        "diff_transition_stability": transition,
        "diff_apex_understeer": apex_push,
        "rear_tyre_wear": wear,
    }


def objective_value(setup: SetupConfiguration, ctx: _ModelContext) -> float:
    return float(sum(objective_terms(setup, ctx).values()))


def tyre_pressures_psi(weather: WeatherData) -> Dict[str, float]:
    """Cold set pressures (psi gauge) from the ideal-gas law; see the assumptions at the top of the module."""
    front_hot, rear_hot = TARGET_RUNNING_PRESSURE_PSI[weather.condition]
    ratio = (float(weather.temperature) + 273.15) / (RUNNING_GAS_TEMPERATURE_C[weather.condition] + 273.15)

    def cold(hot_gauge: float) -> float:
        return round((hot_gauge + ATMOSPHERIC_PRESSURE_PSI) * ratio - ATMOSPHERIC_PRESSURE_PSI, 2)

    front, rear = cold(front_hot), cold(rear_hot)
    return {"front_left": front, "front_right": front, "rear_left": rear, "rear_right": rear}


def _tyre_pressure_basis(weather: WeatherData) -> str:
    front_hot, rear_hot = TARGET_RUNNING_PRESSURE_PSI[weather.condition]
    return (
        f"Rule-based heuristic, not optimised: cold pressures set at {weather.temperature:g} °C ambient so the "
        f"tyres reach assumed running targets of {front_hot:g}/{rear_hot:g} psi (front/rear) at an assumed "
        f"{RUNNING_GAS_TEMPERATURE_C[weather.condition]:g} °C running gas temperature (ideal gas, constant volume, "
        "no blankets). Replace the targets with the tyre supplier's prescription."
    )


# --------------------------------------------------------------------------- search


def _pinned_parameters(driver: DriverPreferences) -> Dict[str, float]:
    pinned: Dict[str, float] = {}
    if driver.preferred_ride_height is not None:
        pinned["ride_height"] = float(driver.preferred_ride_height)
    sources = ((driver.preferred_wing_angles, PREFERRED_WING_KEYS), (driver.preferred_diff_settings, PREFERRED_DIFF_KEYS))
    for source, mapping in sources:
        for key, param in mapping.items():
            if source and key in source:
                pinned[param] = float(source[key])
    return pinned


def baseline_setup(track: TrackProfile, weather: WeatherData, driver: DriverPreferences) -> SetupConfiguration:
    """The previous closed-form rule of thumb, kept only as a comparison baseline (not enqueued)."""
    downforce_multiplier = {WeatherCondition.DRY: 1.0, WeatherCondition.WET: 1.15, WeatherCondition.INTERMEDIATE: 1.08}
    ride_adjustment = {WeatherCondition.DRY: 0.0, WeatherCondition.WET: 5.0, WeatherCondition.INTERMEDIATE: 2.5}
    downforce = _clip01(track.downforce_requirement * downforce_multiplier[weather.condition])
    ride_height = 70.0 + ride_adjustment[weather.condition]
    if track.track_type == TrackType.HIGH_SPEED:
        ride_height -= 4.0
    elif track.track_type in (TrackType.TECHNICAL, TrackType.LOW_SPEED):
        ride_height += 4.0
    high = track.high_speed_sections / track.corners
    params = {
        "ride_height": ride_height,
        "front_wing_angle": 3.0 + 10.0 * downforce,
        "rear_wing_angle": 5.0 + 13.0 * downforce,
        "brake_bias": 58.0 + min(5.0, track.low_speed_sections * 0.20),
        "diff_preload": 45.0 + 15.0 * (track.low_speed_sections / track.corners),
        "diff_power": 60.0 + 15.0 * driver.risk_tolerance,
        "diff_coast": 55.0 - 10.0 * driver.risk_tolerance,
        "front_arb": 50.0 + 20.0 * downforce,
        "rear_arb": 45.0 + 15.0 * downforce,
        "front_spring": 55.0 + 20.0 * high,
        "rear_spring": 50.0 + 15.0 * high,
    }
    params.update(_pinned_parameters(driver))
    return SetupConfiguration.from_params(
        {name: min(hi, max(lo, params[name])) for name, (lo, hi) in SETUP_BOUNDS.items()}
    )


def _run_search(ctx: _ModelContext, pinned: Mapping[str, float], n_trials: int, seed: int) -> _SearchResult:
    def objective(trial: optuna.Trial) -> float:
        params = {
            name: pinned[name] if name in pinned else trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in SETUP_BOUNDS.items()
        }
        return objective_value(SetupConfiguration.from_params(params), ctx)

    # Quieten Optuna before the study exists so study creation does not log at INFO level (only ever
    # lowers the verbosity, so a caller's stricter setting is kept).
    if optuna.logging.get_verbosity() < optuna.logging.WARNING:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_trial = study.best_trial
    best = SetupConfiguration.from_params({**best_trial.params, **pinned})
    values = [float(t.value) for t in study.trials if t.value is not None]
    return _SearchResult(best, float(best_trial.value), int(best_trial.number), values)


def refine_locally(
    start: SetupConfiguration, ctx: _ModelContext, free: Sequence[str]
) -> Tuple[SetupConfiguration, float, int, bool]:
    """Deterministic compass search over the ``free`` parameters inside ``SETUP_BOUNDS``.

    Each sweep tries +step then -step on every free parameter (fixed order, clipped to its bounds)
    and keeps the first improving move per parameter; when a sweep finds no improvement the step is
    halved. It stops when the step drops below ``REFINE_MIN_STEP`` of each range (no move of that
    size improves: a local minimum at that resolution) or after ``REFINE_MAX_EVALUATIONS``.
    Returns (setup, objective, evaluations, converged).
    """
    params = start.as_params()
    value = objective_value(start, ctx)
    step, evaluations = REFINE_INITIAL_STEP, 1
    while step >= REFINE_MIN_STEP:
        if evaluations >= REFINE_MAX_EVALUATIONS:
            return SetupConfiguration.from_params(params), value, evaluations, False
        improved = False
        for name in free:
            lo, hi = SETUP_BOUNDS[name]
            for direction in (1.0, -1.0):
                moved = min(hi, max(lo, params[name] + direction * step * (hi - lo)))
                if moved == params[name]:
                    continue
                candidate = {**params, name: moved}
                candidate_value = objective_value(SetupConfiguration.from_params(candidate), ctx)
                evaluations += 1
                if candidate_value < value:
                    params, value, improved = candidate, candidate_value, True
                    break
        if not improved:
            step /= 2.0
    return SetupConfiguration.from_params(params), value, evaluations, True


def _max_deviation(a: SetupConfiguration, b: SetupConfiguration) -> float:
    """Largest parameter difference between two setups as a fraction of that parameter's range."""
    return max(abs(getattr(a, name) - getattr(b, name)) / (hi - lo) for name, (lo, hi) in SETUP_BOUNDS.items())


def _refine_from_starts(
    starts: Sequence[Tuple[str, SetupConfiguration]], ctx: _ModelContext, free: Sequence[str]
) -> List[_Refinement]:
    refinements = []
    for name, start in starts:
        setup, value, evaluations, converged = refine_locally(start, ctx, free)
        refinements.append(_Refinement(name, objective_value(start, ctx), setup, value, evaluations, converged))
    return refinements


def _select(refinements: Sequence[_Refinement]) -> _Refinement:
    """Best refined setup; the earliest start (the TPE best trial) wins near-ties."""
    best_value = min(r.value for r in refinements)
    return next(r for r in refinements if r.value <= best_value + SELECTION_TOLERANCE)


def multi_start_agreement(refinements: Sequence[_Refinement], selected: SetupConfiguration) -> float:
    """Share of refinement starts whose optimum lies within ``AGREEMENT_TOLERANCE`` of ``selected``."""
    if not refinements:
        raise ValueError("multi_start_agreement needs at least one refinement")
    agreeing = sum(_max_deviation(r.setup, selected) <= AGREEMENT_TOLERANCE for r in refinements)
    return round(agreeing / len(refinements), 3)


def _parameter_spread(refinements: Sequence[_Refinement]) -> Dict[str, float]:
    return {
        name: round(max(getattr(r.setup, name) for r in refinements) - min(getattr(r.setup, name) for r in refinements), 2)
        for name in SETUP_BOUNDS
    }


def _generate_reasoning(
    track: TrackProfile,
    weather: WeatherData,
    selected: _Refinement,
    tpe_refined: _Refinement,
    terms: Mapping[str, float],
    baseline_value: float,
    best_trial_value: float,
    n_trials: int,
    confidence: float,
    pinned: Mapping[str, float],
    assumed_defaults: Sequence[str] = (),
) -> str:
    largest = sorted(terms.items(), key=lambda item: item[1], reverse=True)[:2]
    track_label = f" for {track.track_name}" if track.track_name else ""
    setup = selected.setup
    if best_trial_value < baseline_value:
        tpe_text = (
            f"The best of {n_trials} TPE trials scored {best_trial_value:.3f}, "
            f"{(baseline_value - best_trial_value) / baseline_value * 100.0:.1f}% better than the rule-of-thumb "
            f"baseline ({baseline_value:.3f})"
        )
    else:
        tpe_text = (
            f"No TPE trial in {n_trials} beat the rule-of-thumb baseline ({best_trial_value:.3f} vs "
            f"{baseline_value:.3f})"
        )
    source_text = "" if selected is tpe_refined else f" The best optimum came from the {selected.start} start."
    pinned_text = f" Pinned by driver preference: {', '.join(sorted(pinned))}." if pinned else ""
    assumed_text = (
        f" Not supplied, so defaults were assumed: {', '.join(assumed_defaults)}." if assumed_defaults else ""
    )
    return (
        f"Heuristic setup search on a {track.track_type.value} profile{track_label} in {weather.condition.value} conditions: "
        f"{setup.front_wing_angle:.1f}°/{setup.rear_wing_angle:.1f}° front/rear wing, "
        f"{setup.ride_height:.1f} mm ride height and {setup.brake_bias:.1f}% brake bias. {tpe_text}; local "
        f"refinement then reached {selected.value:.3f}, {(baseline_value - selected.value) / baseline_value * 100.0:.1f}% "
        f"better than the baseline.{source_text} Multi-start agreement {confidence:.2f}. "
        f"Largest remaining penalties: {', '.join(f'{name} {value:.3f}' for name, value in largest)}."
        f"{pinned_text}{assumed_text} Values come from a heuristic model and require simulator/track validation."
    )


class SetupOptimizer:
    """Recommend a setup with a seeded Optuna TPE search plus deterministic local refinement.

    Holds only immutable defaults, so one instance can serve concurrent requests.
    """

    def __init__(self, n_trials: int = DEFAULT_N_TRIALS, seed: int = DEFAULT_SEED):
        self.n_trials, self.seed = validate_search_settings(n_trials, seed)

    def recommend_setup(
        self,
        driver_preferences: DriverPreferences,
        track_profile: TrackProfile,
        weather: WeatherData,
        n_trials: Optional[int] = None,
        seed: Optional[int] = None,
        assumed_defaults: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Recommend a setup. ``assumed_defaults`` (see ``SetupInputs``) is reported, not modelled."""
        n_trials, seed = validate_search_settings(
            self.n_trials if n_trials is None else n_trials,
            self.seed if seed is None else seed,
        )
        validate_driver_preferences(driver_preferences)
        validate_track_profile(track_profile)
        validate_weather(weather)

        ctx = _derive_context(driver_preferences, track_profile, weather)
        pinned = _pinned_parameters(driver_preferences)
        free = [name for name in SETUP_BOUNDS if name not in pinned]
        baseline = baseline_setup(track_profile, weather, driver_preferences)
        baseline_value = objective_value(baseline, ctx)
        centre = SetupConfiguration.from_params({**{n: (lo + hi) / 2 for n, (lo, hi) in SETUP_BOUNDS.items()}, **pinned})

        search = _run_search(ctx, pinned, n_trials, seed)
        # The baseline is a start, so the returned setup can never be worse than it.
        refinements = _refine_from_starts(
            [("optuna_best_trial", search.best), ("rule_of_thumb_baseline", baseline), ("bounds_centre", centre)], ctx, free
        )
        tpe_refined, selected = refinements[0], _select(refinements)
        confidence = multi_start_agreement(refinements, selected.setup)
        terms = objective_terms(selected.setup, ctx)

        return {
            **selected.setup.to_output(exact=pinned),
            "tire_pressures_psi": tyre_pressures_psi(weather),
            "tire_pressure_basis": _tyre_pressure_basis(weather),
            "units": dict(OUTPUT_UNITS),
            "selected_source": f"{selected.start}_refined",
            "objective_value": round(selected.value, 6),
            "objective_breakdown": {name: round(value, 6) for name, value in terms.items()},
            "objective_description": OBJECTIVE_DESCRIPTION,
            "handling_balance": {k: round(v, 4) for k, v in handling_balance(selected.setup, ctx).items()},
            "baseline_setup": baseline.to_output(exact=pinned),
            "baseline_objective_value": round(baseline_value, 6),
            "improvement_over_baseline": round(baseline_value - selected.value, 6),
            # TPE stage on its own, before refinement.
            "best_trial_objective": round(search.best_value, 6),
            "best_trial_number": search.best_trial_number,
            "best_trial_improvement_over_baseline": round(baseline_value - search.best_value, 6),
            "random_startup_trials": N_STARTUP_TRIALS,
            "random_startup_best_objective": round(min(search.values[:N_STARTUP_TRIALS]), 6),
            "trials": n_trials,
            "seed": seed,
            "optimization_method": "Optuna TPESampler + deterministic multi-start compass refinement",
            # Local refinement stage.
            "refinement_method": REFINEMENT_METHOD,
            "refinement_improvement_over_best_trial": round(search.best_value - tpe_refined.value, 6),
            "refinement_starts": [
                {
                    "start": r.start,
                    "start_objective": round(r.start_value, 6),
                    "refined_objective": round(r.value, 6),
                    "max_parameter_deviation": round(_max_deviation(r.setup, selected.setup), 6),
                    "evaluations": r.evaluations,
                    "converged": r.converged,
                }
                for r in refinements
            ],
            "parameter_spread": _parameter_spread(refinements),
            "confidence": confidence,
            "confidence_method": CONFIDENCE_METHOD,
            "pinned_by_driver": sorted(pinned),
            # Inputs the request left out, whose engine default was used (e.g. risk_tolerance=0.5).
            "assumed_defaults": list(assumed_defaults),
            "inputs_not_modelled": not_modelled_inputs(weather),
            "reasoning": _generate_reasoning(
                track_profile, weather, selected, tpe_refined, terms, baseline_value, search.best_value, n_trials,
                confidence, pinned, assumed_defaults,
            ),
            "model_scope": MODEL_SCOPE,
        }


def recommend_setup_from_inputs(inputs: SetupInputs) -> Dict[str, Any]:
    """Entry point for already-validated request models (see ``schemas.SetupRequest.to_engine_inputs``).

    The result's ``assumed_defaults`` lists ``inputs.assumed_defaults``.
    """
    return SetupOptimizer().recommend_setup(
        inputs.driver_preferences,
        inputs.track_profile,
        inputs.weather,
        n_trials=inputs.n_trials,
        seed=inputs.seed,
        assumed_defaults=inputs.assumed_defaults,
    )

