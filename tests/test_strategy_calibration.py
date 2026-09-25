"""Behavioural tests for estimating the strategy engine's tyre parameters from lap history.

Lap histories are generated with the strategy engine's own lap-time model (``evaluate_plan`` on
one-stint plans, differenced into per-lap times), so a round trip checks that the estimator
inverts exactly the model the engine projects with.
"""

import json
import math
import random
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from core_modules.strategy_optimizer.calibration import (
    MAX_CALIBRATION_LAPS,
    MIN_CLEAN_LAPS,
    STATUS_ESTIMATED,
    STATUS_INSUFFICIENT_DATA,
    STATUS_OUTSIDE_ENGINE_LIMITS,
    LapRecord,
    estimate_tire_parameters,
)
from core_modules.strategy_optimizer.schemas import (
    StrategyRequest,
    TyreCalibrationRequest,
    calibrate_tyres_response,
    generate_strategy_response,
    tyre_calibration_to_dict,
)
from core_modules.strategy_optimizer.strategy_engine import (
    CarStatus,
    DriverProfile,
    RaceState,
    TireCompound,
    TireData,
    WeatherCondition,
    evaluate_plan,
    generate_strategy,
)

SOFT, MEDIUM, HARD = TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD
INTER, WET = TireCompound.INTERMEDIATE, TireCompound.WET
DRY = WeatherCondition.DRY
BASE_LAP_TIME = 80.0
PIT_DELTA = 22.0

# (base_performance, degradation_rate, warm_up_laps, peak window end) per compound.
TRUE_PARAMS: Dict[TireCompound, Tuple[float, float, int, int]] = {
    SOFT: (1.000, 0.0040, 3, 8),
    MEDIUM: (0.990, 0.0025, 0, 14),
    HARD: (0.982, 0.0015, 0, 20),
}
# (compound, stint length); every stint starts on a fresh set.
STINTS: Tuple[Tuple[TireCompound, int], ...] = (
    (SOFT, 15), (MEDIUM, 25), (HARD, 30), (SOFT, 12), (MEDIUM, 22), (HARD, 26)
)

# A neutral driver and an undamaged car: the engine's driver and damage multipliers are both 1.0,
# so projected lap times are exactly base lap time / tyre performance.
NEUTRAL_TELEMETRY = {"braking_consistency": 0.8, "throttle_aggressiveness": 0.5}
NEUTRAL_CAR = CarStatus(engine_wear=0.2, brake_wear=0.2)
NEUTRAL_DRIVER = DriverProfile(tire_management=0.7, risk_tolerance=0.5)


def _tire(compound: TireCompound, params: Tuple[float, float, int, int]) -> TireData:
    base, degradation, warm_up, end = params
    return TireData(compound, base, degradation, warm_up, (max(1, min(warm_up, end)), end), PIT_DELTA)


@lru_cache(maxsize=None)
def engine_lap_times(
    compound: TireCompound,
    params: Tuple[float, float, int, int],
    laps: int,
    weather: WeatherCondition = DRY,
    track_temperature: float = 30.0,
    base_lap_time: float = BASE_LAP_TIME,
) -> Tuple[float, ...]:
    """Per-lap times of one stint on a fresh set, from the engine (cumulative stint times differenced)."""

    telemetry = {"lap_times": [base_lap_time], **NEUTRAL_TELEMETRY}
    tires = {compound: _tire(compound, params)}
    totals = [0.0]
    for length in range(1, laps + 1):
        race = RaceState(current_lap=1, total_laps=length, weather=weather, track_temperature=track_temperature)
        option = evaluate_plan([compound], [length], telemetry, NEUTRAL_CAR, NEUTRAL_DRIVER, tires, race)
        totals.append(option.projected_race_time)
    return tuple(float(t) for t in np.diff(totals))


def build_history(
    params: Dict[TireCompound, Tuple[float, float, int, int]] = TRUE_PARAMS,
    stints: Sequence[Tuple[TireCompound, int]] = STINTS,
    noise: float = 0.0,
    seed: int = 0,
    weather: WeatherCondition = DRY,
    track_temperature: float = 30.0,
    fuel_effect: float = 0.0,
) -> List[dict]:
    """Engine lap times plus Gaussian noise; out-laps (+19 s) and in-laps (+4 s) flagged.

    fuel_effect (s per lap of fuel) makes earlier race laps slower, relative to the last race lap.
    """

    rng = np.random.default_rng(seed)
    total_race_laps = sum(length for _, length in stints)
    laps: List[dict] = []
    race_lap = 1
    for compound, length in stints:
        times = engine_lap_times(compound, params[compound], length, weather, track_temperature)
        for age in range(1, length + 1):
            lap_time = times[age - 1] + float(rng.normal(0.0, noise)) + fuel_effect * (total_race_laps - race_lap)
            lap = {"compound": compound.value, "tire_age": age, "lap_time": lap_time, "race_lap": race_lap}
            if age == 1:
                lap["pit_out"] = True
                lap["lap_time"] += 19.0
            if age == length:
                lap["pit_in"] = True
                lap["lap_time"] += 4.0
            laps.append(lap)
            race_lap += 1
    return laps


def _index(laps: List[dict], compound: TireCompound, stint_number: int, age: int) -> int:
    """Position of the lap at tyre age ``age`` in the stint_number-th (0-based) stint of ``compound``."""

    seen = -1
    for position, lap in enumerate(laps):
        if lap["compound"] == compound.value and lap["tire_age"] == 1:
            seen += 1
        if seen == stint_number and lap["compound"] == compound.value and lap["tire_age"] == age:
            return position
    raise AssertionError("lap not found")


def _calibrate(laps, **kwargs):
    kwargs.setdefault("weather", "dry")
    kwargs.setdefault("track_temperature", 30.0)
    kwargs.setdefault("pit_stop_delta", PIT_DELTA)
    return estimate_tire_parameters(laps, **kwargs)


def _assert_recovers(tire: TireData, params: Tuple[float, float, int, int], rel: float = 1e-6) -> None:
    base, degradation, warm_up, end = params
    assert tire.base_performance == pytest.approx(base, rel=rel)
    assert tire.degradation_rate == pytest.approx(degradation, rel=rel)
    assert tire.warm_up_laps == warm_up
    assert tire.peak_performance_window == (max(1, warm_up), end)
    assert tire.pit_stop_delta == PIT_DELTA


# ---------------------------------------------------------------------------
# Round trips through the engine's own lap-time model
# ---------------------------------------------------------------------------


def test_noise_free_history_recovers_the_engine_parameters_exactly():
    laps = build_history()
    result = _calibrate(laps)

    assert result.estimated_base_lap_time == pytest.approx(BASE_LAP_TIME, rel=1e-9)
    assert result.reference_compound == SOFT
    assert set(result.tire_data) == {SOFT, MEDIUM, HARD}
    for compound, params in TRUE_PARAMS.items():
        _assert_recovers(result.tire_data[compound], params)
        entry = result.compounds[compound]
        assert entry.status == STATUS_ESTIMATED
        assert entry.outliers_rejected == 0
        assert entry.residual_std_s < 1e-6
        assert entry.r_squared == pytest.approx(1.0, abs=1e-9)
        assert entry.degradation_observed is True
        assert entry.peak_window_end_source == "detected"
    assert result.compounds[SOFT].warm_up_source == "detected"
    assert result.compounds[MEDIUM].warm_up_source == "not_detected"
    # Only the flagged out-laps and in-laps are excluded, with their flag as the reason.
    assert {lap.reason for lap in result.excluded_laps} == {"pit_out", "pit_in"}
    assert len(result.excluded_laps) == 2 * len(STINTS)

    # Fit statistics in seconds (dry, 30 C: the conditions factors are 1.0). The peak lap time is the base lap
    # time / base_performance; one lap into degradation the lap is base lap time / (base - rate).
    body = tyre_calibration_to_dict(result)
    for compound, (base, degradation, _, _) in TRUE_PARAMS.items():
        entry = result.compounds[compound]
        peak = BASE_LAP_TIME / base
        first_worn_lap = BASE_LAP_TIME / (base - degradation)
        assert entry.peak_lap_time == pytest.approx(peak, rel=1e-9)
        assert entry.initial_degradation_s_per_lap == pytest.approx(first_worn_lap - peak, rel=1e-6)
        fit = body["compounds"][compound.value]["fit"]
        assert fit["peak_lap_time_s"] == round(peak, 3)
        assert fit["initial_degradation_s_per_lap"] == round(first_worn_lap - peak, 3)
    assert body["compounds"]["soft"]["fit"]["initial_degradation_s_per_lap"] == 0.321  # 80 / 0.996 - 80
    assert not any("Poor fit" in note for entry in result.compounds.values() for note in entry.notes)


def test_calibrated_parameters_reproduce_the_history_through_the_engine():
    laps = build_history()
    result = _calibrate(laps)
    for compound in (SOFT, MEDIUM, HARD):
        tire = result.tire_data[compound]
        params = (tire.base_performance, tire.degradation_rate, tire.warm_up_laps, tire.peak_performance_window[1])
        projected = engine_lap_times(compound, params, 30, base_lap_time=result.estimated_base_lap_time)
        observed = engine_lap_times(compound, TRUE_PARAMS[compound], 30)
        assert projected == pytest.approx(observed, abs=1e-6)


def noisy_history_with_outliers(seed: int) -> Tuple[List[dict], Dict[int, float]]:
    """The engine history with 0.15 s noise and five injected, unflagged outliers (position -> seconds added)."""

    laps = build_history(noise=0.15, seed=seed)
    injected = {
        _index(laps, SOFT, 0, 6): 7.0,  # traffic
        _index(laps, MEDIUM, 0, 10): 4.0,  # lock-up
        _index(laps, HARD, 0, 15): -2.5,  # mistimed (too fast)
        _index(laps, MEDIUM, 1, 5): 12.0,  # unflagged incident
        _index(laps, HARD, 1, 9): 25.0,  # unflagged virtual safety car
    }
    for position, delta in injected.items():
        laps[position]["lap_time"] += delta
    return laps, injected


@pytest.mark.parametrize("seed", range(8))
def test_noisy_history_with_outliers_recovers_parameters_within_tolerance(seed):
    laps, injected = noisy_history_with_outliers(seed)
    result = _calibrate(laps)

    outliers = {lap.index: lap for lap in result.excluded_laps if lap.reason == "outlier"}
    assert set(injected) <= set(outliers), "every injected outlier is rejected"
    for position, delta in injected.items():
        assert outliers[position].residual_s == pytest.approx(delta, abs=1.0)
    extra = [lap for index, lap in outliers.items() if index not in injected]
    assert len(extra) <= 2, "at most a couple of ordinary laps are rejected as borderline"
    assert all(abs(lap.residual_s) < 1.0 for lap in extra)

    # These tolerances hold for every seed in 0..299 with this generator (worst cases seen when the test
    # was written: base lap time 0.15 s, base_performance 0.003, degradation 32% / 27% / 40% for
    # soft / medium / hard, window end 1 / 1 / 3 laps, residual std 0.08-0.22 s).
    assert result.estimated_base_lap_time == pytest.approx(BASE_LAP_TIME, abs=0.2)
    assert result.reference_compound == SOFT
    tolerances = {SOFT: (0.4, 1), MEDIUM: (0.35, 1), HARD: (0.5, 3)}  # (degradation rel. error, window-end laps)
    for compound, (base, degradation, warm_up, end) in TRUE_PARAMS.items():
        tire = result.tire_data[compound]
        rel_tolerance, end_tolerance = tolerances[compound]
        assert tire.base_performance == pytest.approx(base, abs=0.004)
        assert tire.degradation_rate == pytest.approx(degradation, rel=rel_tolerance)
        assert tire.warm_up_laps == warm_up
        assert abs(tire.peak_performance_window[1] - end) <= end_tolerance
        entry = result.compounds[compound]
        assert 0.07 < entry.residual_std_s < 0.25  # the injected noise is 0.15 s
        assert entry.laps_used + entry.outliers_rejected == entry.clean_laps


def test_calibrated_tire_data_drives_generate_strategy_like_the_true_parameters():
    body = calibrate_tyres_response(TyreCalibrationRequest.model_validate(
        {"laps": build_history(), "weather": "dry", "track_temperature": 30.0, "pit_stop_delta": PIT_DELTA}
    ))

    def payload(tire_data, lap_time):
        return {
            "telemetry": {"lap_times": [lap_time], **NEUTRAL_TELEMETRY},
            "car_status": {"engine_wear": 0.2, "brake_wear": 0.2},
            "driver_profile": {"tire_management": 0.7, "risk_tolerance": 0.5},
            "tire_data": tire_data,
            "race_state": {"current_lap": 1, "total_laps": 57, "weather": "dry", "track_temperature": 30.0},
        }

    calibrated = generate_strategy_response(
        StrategyRequest.model_validate(payload(body["tire_data"], body["estimated_base_lap_time_s"]))
    )
    true_tire_data = {
        compound.value: {
            "base_performance": base,
            "degradation_rate": degradation,
            "warm_up_laps": warm_up,
            "peak_performance_window": [max(1, warm_up), end],
            "pit_stop_delta": PIT_DELTA,
        }
        for compound, (base, degradation, warm_up, end) in TRUE_PARAMS.items()
    }
    truth = generate_strategy_response(StrategyRequest.model_validate(payload(true_tire_data, BASE_LAP_TIME)))

    assert calibrated["best_strategy_id"] == truth["best_strategy_id"]
    assert calibrated["strategies"][0]["pit_laps"] == truth["strategies"][0]["pit_laps"]
    assert calibrated["strategies"][0]["projected_race_time"] == pytest.approx(
        truth["strategies"][0]["projected_race_time"], abs=0.01
    )


def test_noisy_calibration_output_is_accepted_by_the_engine():
    laps = build_history(noise=0.3, seed=11)
    result = _calibrate(laps)
    strategy = generate_strategy(
        {"lap_times": [result.estimated_base_lap_time], **NEUTRAL_TELEMETRY},
        NEUTRAL_CAR,
        NEUTRAL_DRIVER,
        result.tire_data,
        RaceState(current_lap=20, total_laps=57, weather=DRY, track_temperature=30.0),
    )
    assert strategy.candidates and all(math.isfinite(c.projected_race_time) for c in strategy.candidates)


# ---------------------------------------------------------------------------
# Conditions, fuel and supplied structure
# ---------------------------------------------------------------------------


def test_hot_track_factor_is_divided_out():
    laps = build_history(track_temperature=40.0)  # soft x0.97, hard x1.02 in the engine
    result = _calibrate(laps, track_temperature=40.0)
    for compound, params in TRUE_PARAMS.items():
        _assert_recovers(result.tire_data[compound], params)
    assert result.estimated_base_lap_time == pytest.approx(BASE_LAP_TIME, rel=1e-9)

    # The same laps read as a cool-track session give the effective (factor-included) values instead.
    cool = _calibrate(laps, track_temperature=30.0)
    assert cool.reference_compound == HARD  # hard x1.02 is faster than soft x0.97 on a hot track
    assert cool.tire_data[SOFT].base_performance == pytest.approx(0.97 / (0.982 * 1.02), rel=1e-6)


def test_wet_weather_factors_are_divided_out():
    # A 3-lap warm-up slows tyre ages 1-2; age 1 is always a flagged out-lap here, so age 2 identifies it.
    params = {INTER: (1.0, 0.003, 3, 6), WET: (0.93, 0.002, 0, 10)}
    stints = ((INTER, 18), (WET, 20), (INTER, 14))
    laps = build_history(params=params, stints=stints, weather=WeatherCondition.WET)
    result = _calibrate(laps, weather="wet", track_temperature=18.0)
    assert result.reference_compound == INTER
    for compound, compound_params in params.items():
        _assert_recovers(result.tire_data[compound], compound_params)


def test_fuel_correction_separates_fuel_burn_from_degradation():
    laps = build_history(fuel_effect=0.06)  # fuel term relative to the final race lap (a flagged in-lap)
    last_clean = max(lap["race_lap"] for lap in laps if not (lap.get("pit_in") or lap.get("pit_out")))
    corrected = _calibrate(laps, fuel_correction_s_per_lap=0.06)
    # Lap times are corrected to the fuel load of the latest unflagged lap, one lap before the final lap:
    # every corrected lap is 0.06 s slower than the engine's, a constant the tyre model absorbs almost exactly.
    assert corrected.fuel_reference_lap == last_clean == len(laps) - 1
    assert corrected.estimated_base_lap_time == pytest.approx(BASE_LAP_TIME + 0.06, abs=0.005)
    for compound in (MEDIUM, HARD):
        tire, (base, degradation, warm_up, end) = corrected.tire_data[compound], TRUE_PARAMS[compound]
        assert tire.base_performance == pytest.approx(base, rel=1e-4)
        assert tire.degradation_rate == pytest.approx(degradation, rel=2e-3)
        assert (tire.warm_up_laps, tire.peak_performance_window[1]) == (warm_up, end)

    uncorrected = _calibrate(laps)
    for compound in (MEDIUM, HARD):
        # Fuel burn makes later laps faster, which hides part of the degradation.
        assert uncorrected.tire_data[compound].degradation_rate < 0.9 * TRUE_PARAMS[compound][1]
    assert any("Fuel: not corrected" in note for note in uncorrected.assumptions)


def test_outliers_under_fuel_correction_are_reported_in_the_raw_lap_time_frame():
    laps = build_history(fuel_effect=0.06)
    slow = _index(laps, MEDIUM, 0, 10)
    laps[slow]["lap_time"] += 5.0
    result = _calibrate(laps, fuel_correction_s_per_lap=0.06)
    (outlier,) = [lap for lap in result.excluded_laps if lap.reason == "outlier"]
    assert outlier.index == slow and outlier.lap_time == laps[slow]["lap_time"]
    # The model's lap time at this lap's own fuel load, so lap_time - fitted_lap_time is the residual. (The
    # corrected laps are a constant 0.06 s slower than the engine's, which the tyre model absorbs almost exactly.)
    assert outlier.residual_s == pytest.approx(5.0, abs=1e-4)
    assert outlier.lap_time - outlier.fitted_lap_time == pytest.approx(outlier.residual_s, abs=1e-9)
    engine_time = engine_lap_times(MEDIUM, TRUE_PARAMS[MEDIUM], 25)[9]
    fuel_at_this_lap = 0.06 * (sum(length for _, length in STINTS) - laps[slow]["race_lap"])
    assert outlier.fitted_lap_time == pytest.approx(engine_time + fuel_at_this_lap, abs=0.01)

    body = tyre_calibration_to_dict(result)
    (serialised,) = [lap for lap in body["excluded_laps"] if lap["reason"] == "outlier"]
    assert serialised["lap_time"] - serialised["fitted_lap_time"] == pytest.approx(5.0, abs=0.002)
    assert serialised["residual_s"] == 5.0
    assert body["conditions"]["fuel_reference_lap"] == result.fuel_reference_lap == len(laps) - 1
    assert body["conditions"]["fuel_correction_s_per_lap"] == 0.06


def test_supplied_window_end_and_warm_up_are_honoured():
    laps = build_history()
    result = _calibrate(laps, peak_window_end={"soft": 8, MEDIUM: 14}, warm_up_laps={SOFT: 3})
    _assert_recovers(result.tire_data[SOFT], TRUE_PARAMS[SOFT])
    _assert_recovers(result.tire_data[MEDIUM], TRUE_PARAMS[MEDIUM])
    assert result.compounds[SOFT].warm_up_source == "supplied"
    assert result.compounds[SOFT].peak_window_end_source == "supplied"
    assert result.compounds[HARD].peak_window_end_source == "detected"

    # A window end past the data leaves no laps to measure degradation: 0, flagged, not guessed. These hard
    # stints end before the true window end (20), so the laps show no wear either.
    no_wear_yet = build_history(stints=((HARD, 20), (HARD, 18)))
    hard = _calibrate(no_wear_yet, peak_window_end={HARD: 60}).compounds[HARD]
    assert hard.status == STATUS_ESTIMATED
    assert hard.tire_data.degradation_rate == 0.0
    assert hard.tire_data.peak_performance_window[1] == 60
    assert hard.degradation_observed is False
    assert any("no degradation rate could be estimated" in note for note in hard.notes)

    # In the full history the hard laps past age 20 are clearly slower. A flat window to 60 leaves those 14 of
    # the 52 clean laps unexplained, more than a quarter, so the supplied value is reported as not matching.
    contradicted = _calibrate(laps, peak_window_end={HARD: 60}).compounds[HARD]
    assert contradicted.status == STATUS_INSUFFICIENT_DATA and contradicted.tire_data is None
    assert "14 of 52 clean laps (27%) were rejected as outliers (0 faster and 14 slower" in contradicted.reason
    assert "the supplied peak_window_end=60 does not match them" in contradicted.reason


def test_degradation_rate_needs_three_distinct_tyre_ages_past_the_window_end():
    times = engine_lap_times(MEDIUM, (1.0, 0.004, 0, 10), 13)
    laps = [
        {"compound": "medium", "tire_age": age, "lap_time": times[age - 1]} for _ in range(2) for age in range(1, 14)
    ]
    # Ages 11, 12 and 13 lie past a window end of 10: the rate is estimated exactly.
    three = _calibrate(laps, peak_window_end={MEDIUM: 10}).compounds[MEDIUM]
    assert three.tire_data.degradation_rate == pytest.approx(0.004, rel=1e-6)
    assert three.degradation_observed is True
    # Only ages 12 and 13 lie past a window end of 11: no rate is estimated, and the note says why.
    two = _calibrate(laps, peak_window_end={MEDIUM: 11}).compounds[MEDIUM]
    assert two.tire_data.degradation_rate == 0.0
    assert two.degradation_observed is False
    assert any("Fewer than 3 distinct tyre ages beyond the supplied window end (11)" in note for note in two.notes)
    # Detection follows the same rule: with ages up to 12 only (two ages past 10) no rate is reported, and the
    # slower laps at ages 11-12 that no admissible structure explains are listed as outliers.
    short = [lap for lap in laps if lap["tire_age"] <= 12]
    result = _calibrate(short)
    detected = result.compounds[MEDIUM]
    assert detected.tire_data.degradation_rate == 0.0
    assert detected.peak_window_end_source == "no_degradation_observed"
    assert detected.tire_data.peak_performance_window == (1, 10)
    outliers = [lap for lap in result.excluded_laps if lap.reason == "outlier"]
    assert sorted(lap.tire_age for lap in outliers) == [11, 11, 12, 12]
    assert all(lap.tire_age == short[lap.index]["tire_age"] for lap in outliers)
    # The statistics describe the laps used, and the notes say that slower, older laps were set aside.
    assert (detected.clean_laps, detected.laps_used, detected.outliers_rejected) == (24, 20, 4)
    assert detected.tire_age_range == (1, 10)
    assert any(
        "No degradation detected up to tyre age 10" in note and "the oldest tyre age among the laps used" in note
        for note in detected.notes
    )
    assert any(
        note.startswith("4 slower lap(s) at tyre age 11-12, older than every lap used, were rejected as outliers "
                        "(+0.321s to +0.645s against the fit). Degradation may have started")
        for note in detected.notes
    )
    full = _calibrate(laps).compounds[MEDIUM]
    assert full.tire_data.peak_performance_window == (1, 10)
    assert full.tire_data.degradation_rate == pytest.approx(0.004, rel=1e-6)


def test_flat_history_reports_no_degradation_and_falling_history_says_why():
    flat = [
        {"compound": "medium", "tire_age": age, "lap_time": 81.0 + (0.05 if age % 2 else -0.05)} for age in range(1, 21)
    ]
    result = _calibrate(flat)
    medium = result.compounds[MEDIUM]
    assert medium.tire_data.degradation_rate == 0.0
    assert medium.peak_window_end_source == "no_degradation_observed"
    assert medium.tire_data.peak_performance_window == (1, 20)
    assert any("No degradation detected up to tyre age 20" in note for note in medium.notes)

    falling = [{"compound": "medium", "tire_age": age, "lap_time": 82.0 - 0.08 * age} for age in range(1, 21)]
    medium = _calibrate(falling).compounds[MEDIUM]
    assert medium.tire_data.degradation_rate == 0.0
    assert any("Lap times fall with tyre age" in note for note in medium.notes)


# ---------------------------------------------------------------------------
# Data requirements and outliers
# ---------------------------------------------------------------------------


def test_compound_with_too_few_clean_laps_is_insufficient_data_not_guessed():
    laps = build_history(stints=((MEDIUM, 25), (HARD, 6)))  # hard: 6 laps, 2 of them flagged
    result = _calibrate(laps)
    hard = result.compounds[HARD]
    assert hard.status == STATUS_INSUFFICIENT_DATA
    assert (hard.laps_supplied, hard.laps_flagged, hard.clean_laps, hard.laps_used) == (6, 2, 4, 0)
    assert f"at least {MIN_CLEAN_LAPS}" in hard.reason and "4 clean lap(s)" in hard.reason
    assert hard.tire_data is None and HARD not in result.tire_data
    # The estimated compound becomes the reference.
    assert result.reference_compound == MEDIUM
    assert result.tire_data[MEDIUM].base_performance == 1.0
    assert result.estimated_base_lap_time == pytest.approx(BASE_LAP_TIME / 0.990, rel=1e-9)

    body = tyre_calibration_to_dict(result)
    assert body["compounds"]["hard"]["status"] == "insufficient_data"
    assert body["compounds"]["hard"]["fit"] is None
    assert "hard" not in body["tire_data"]

    # Six clean laps, two of them 9-11 s slow: the four left after outlier rejection are not enough.
    few = [{"compound": "hard", "tire_age": age, "lap_time": 81.0 + delta}
           for age, delta in enumerate((0.05, -0.05, 0.0, 9.0, 0.02, 11.0), 1)]
    hard = _calibrate(few).compounds[HARD]
    assert hard.status == STATUS_INSUFFICIENT_DATA and hard.tire_data is None
    assert hard.reason == (
        "Only 4 of 6 clean laps are consistent with the tyre model after outlier rejection; at least 5 are needed."
    )


def test_all_compounds_insufficient_gives_no_estimate():
    laps = [{"compound": "soft", "tire_age": age, "lap_time": 80.0 + 0.1 * age} for age in range(1, 5)]
    laps.append({"compound": "hard", "tire_age": 3, "lap_time": 81.0, "safety_car": True})
    result = _calibrate(laps)
    assert result.estimated_base_lap_time is None and result.reference_compound is None
    assert result.tire_data == {}
    assert {entry.status for entry in result.compounds.values()} == {STATUS_INSUFFICIENT_DATA}
    body = tyre_calibration_to_dict(result)
    assert body["estimated_base_lap_time_s"] is None and body["tire_data"] == {}
    assert body["excluded_laps"] == [
        {"index": 4, "compound": "hard", "tire_age": 3, "lap_time": 81.0, "reason": "safety_car",
         "fitted_lap_time": None, "residual_s": None}
    ]


def test_outliers_and_flagged_laps_are_excluded_with_reasons_and_do_not_bias_the_fit():
    laps = build_history()
    slow = _index(laps, MEDIUM, 0, 9)
    fast = _index(laps, HARD, 1, 12)
    safety_car = _index(laps, SOFT, 1, 5)
    laps[slow]["lap_time"] += 6.0
    laps[fast]["lap_time"] -= 3.0
    laps[safety_car]["lap_time"] += 30.0
    laps[safety_car]["safety_car"] = True
    result = _calibrate(laps)

    for compound, params in TRUE_PARAMS.items():
        _assert_recovers(result.tire_data[compound], params)
    outliers = {lap.index: lap for lap in result.excluded_laps if lap.reason == "outlier"}
    assert set(outliers) == {slow, fast}
    assert outliers[slow].residual_s == pytest.approx(6.0, abs=1e-6)
    assert outliers[fast].residual_s == pytest.approx(-3.0, abs=1e-6)
    assert outliers[slow].fitted_lap_time == pytest.approx(laps[slow]["lap_time"] - 6.0, abs=1e-6)
    assert (outliers[slow].tire_age, outliers[fast].tire_age) == (9, 12)
    assert [lap.reason for lap in result.excluded_laps if lap.index == safety_car] == ["safety_car"]
    assert result.compounds[MEDIUM].outliers_rejected == 1
    assert result.compounds[SOFT].outliers_rejected == 0
    # Flagged laps and outliers are reported together, in the order of the request.
    positions = [lap.index for lap in result.excluded_laps]
    assert positions == sorted(positions) and slow < positions[-1]


def test_combined_flags_are_all_reported():
    laps = build_history()
    laps[5]["pit_in"] = True
    laps[5]["safety_car"] = True
    result = _calibrate(laps)
    assert [lap.reason for lap in result.excluded_laps if lap.index == 5] == ["pit_in, safety_car"]


def test_compound_outside_engine_limits_is_reported_not_returned():
    laps = [{"compound": "soft", "tire_age": age, "lap_time": 80.0 + 0.2 * max(0, age - 4)} for age in range(1, 16)]
    laps += [{"compound": "hard", "tire_age": age, "lap_time": 170.0} for age in range(1, 16)]
    result = _calibrate(laps)
    hard = result.compounds[HARD]
    assert hard.status == STATUS_OUTSIDE_ENGINE_LIMITS
    assert "below the engine minimum 0.5" in hard.reason
    assert HARD not in result.tire_data and hard.tire_data is None
    assert result.compounds[SOFT].status == STATUS_ESTIMATED


def _two_stints(compound: str, times: Sequence[float]) -> List[dict]:
    """Two unflagged stints with these lap times at tyre ages 1, 2, ..."""

    return [{"compound": compound, "tire_age": age, "lap_time": t} for _ in range(2) for age, t in enumerate(times, 1)]


# Soft: 80 s for tyre ages 1-5, then 1 / lap time falls by 0.0034 per lap (109.9, 175.4 and 434.8 s), so at age
# 8 the fitted performance is 80 / 434.8 = 0.184 of its own peak, below the engine's 0.20 floor.
STEEP_SOFT = _two_stints("soft", [80.0] * 5 + [1.0 / (1.0 / 80.0 - 0.0034 * k) for k in (1, 2, 3)])


def test_performance_floor_is_checked_and_the_reference_is_a_compound_within_the_limits():
    medium = [
        {"compound": "medium", "tire_age": age, "lap_time": 81.0 + 0.05 * max(0, age - 6)} for age in range(1, 16)
    ]
    result = _calibrate(STEEP_SOFT + medium)

    soft = result.compounds[SOFT]
    assert soft.status == STATUS_OUTSIDE_ENGINE_LIMITS and soft.tire_data is None
    assert soft.reason == (
        "Fitted, but the fitted performance falls to 0.184 of its own peak pace (conditions factor included) within "
        "the observed tyre ages, at or below the engine's 0.20 floor where it stops modelling wear."
    )
    # Soft is faster at its peak but cannot be estimated, so medium defines base_performance 1.0 and the base lap
    # time: the reference always has tire_data.
    assert result.reference_compound == MEDIUM and set(result.tire_data) == {MEDIUM}
    assert result.tire_data[MEDIUM].base_performance == 1.0
    assert result.estimated_base_lap_time == pytest.approx(81.0, abs=1e-3)
    assert any(note.startswith("base_performance is relative to medium, the fastest compound whose fit is within")
               for note in result.assumptions)
    assert (
        "Faster at peak but outside the engine's limits, so not the reference and not in tire_data: soft (see its "
        "reason)." in result.assumptions
    )
    body = tyre_calibration_to_dict(result)
    assert body["reference_compound"] == "medium" and body["tire_data"]["medium"]["base_performance"] == 1.0

    alone = _calibrate(STEEP_SOFT)
    assert alone.reference_compound is None and alone.estimated_base_lap_time is None and alone.tire_data == {}
    assert alone.compounds[SOFT].status == STATUS_OUTSIDE_ENGINE_LIMITS
    assert any(note.startswith("No fitted compound is within the engine's limits (see each compound's reason: soft)")
               for note in alone.assumptions)


def test_performance_floor_is_also_checked_relative_to_the_reference():
    # Hard: 100 s for tyre ages 1-5, then 1 / lap time falls by 0.00254 per lap to 420.2 s at age 8. That is 0.238
    # of its own peak pace, above the floor, but 80 / 420.2 = 0.190 of soft's, where the engine would floor it.
    hard = _two_stints("hard", [100.0] * 5 + [1.0 / (0.01 - 0.00254 * k) for k in (1, 2, 3)])
    soft = [{"compound": "soft", "tire_age": age, "lap_time": 80.0} for age in range(1, 16)]
    result = _calibrate(soft + hard)
    assert result.reference_compound == SOFT and set(result.tire_data) == {SOFT}
    assert result.compounds[HARD].status == STATUS_OUTSIDE_ENGINE_LIMITS
    assert result.compounds[HARD].reason == (
        "Fitted, but relative to soft's base lap time the fitted performance falls to 0.190 within the observed tyre "
        "ages, at or below the engine's 0.20 floor where it stops modelling wear."
    )


def test_two_pace_levels_are_not_reported_as_one_tyre_model():
    # 10 laps at 80 s and 11 at 86 s (for example two fuel loads): an 86 s fit would reject the 80 s laps as
    # outliers. More than a quarter of the clean laps rejected means the compound is not estimated.
    mixed = [{"compound": "medium", "tire_age": age, "lap_time": 80.0 + 0.01 * (age % 3)} for age in range(1, 11)]
    mixed += [{"compound": "medium", "tire_age": age, "lap_time": 86.0 + 0.01 * (age % 3)} for age in range(1, 12)]
    result = _calibrate(mixed)
    entry = result.compounds[MEDIUM]
    assert entry.status == STATUS_INSUFFICIENT_DATA and entry.tire_data is None and result.tire_data == {}
    assert entry.reason.startswith(
        "10 of 21 clean laps (48%) were rejected as outliers (10 faster and 0 slower than the fit to the rest). "
        "More than 25% means one tyre model does not describe these laps"
    )
    assert (entry.laps_used, entry.outliers_rejected, entry.peak_lap_time) == (0, 0, None)
    assert not [lap for lap in result.excluded_laps if lap.reason == "outlier"]

    # Two levels mixed evenly: the median-based noise scale spans both, nothing is rejected and the fit lands
    # between them. Its residual standard deviation, above 3% of the peak lap time, shows it.
    even = [{"compound": "medium", "tire_age": age, "lap_time": 80.0 if age % 2 else 86.0} for age in range(1, 21)]
    entry = _calibrate(even).compounds[MEDIUM]
    assert entry.status == STATUS_INSUFFICIENT_DATA
    assert entry.reason.startswith(
        "The best fit leaves a residual standard deviation of 3.096s, more than 3% of its peak lap time"
    )
    assert "clean lap times range from 80.000s to 86.000s" in entry.reason

    # A small faster group (4 of 20 laps) is rejected; the compound is estimated from the rest, with a note.
    minority = [
        {"compound": "medium", "tire_age": age, "lap_time": 86.0 + (0.05 if age % 2 else -0.05)} for age in range(1, 17)
    ]
    minority += [{"compound": "medium", "tire_age": age, "lap_time": 80.0} for age in (3, 7, 11, 15)]
    entry = _calibrate(minority).compounds[MEDIUM]
    assert (entry.status, entry.outliers_rejected) == (STATUS_ESTIMATED, 4)
    assert entry.peak_lap_time == pytest.approx(86.0, abs=0.01)
    assert any(note.startswith("Possible mixed pace: 4 rejected laps were faster than the fit (by 6.000s).")
               for note in entry.notes)


def test_poor_fit_is_noted_when_the_laps_scatter_widely():
    rng = np.random.default_rng(4)
    laps = _two_stints("medium", [float(rng.uniform(80.0, 85.0)) for _ in range(15)])
    entry = _calibrate(laps).compounds[MEDIUM]
    # Scatter between 1% and 3% of the peak lap time: estimated, with a warning.
    assert entry.status == STATUS_ESTIMATED and entry.outliers_rejected == 0
    assert 0.01 * entry.peak_lap_time < entry.residual_std_s < 0.03 * entry.peak_lap_time
    assert any(note.startswith(f"Poor fit: the residual standard deviation {entry.residual_std_s:.3f}s is more than 1%")
               for note in entry.notes)


def test_slower_laps_past_the_fitted_line_are_noted_as_a_possible_cliff():
    times = engine_lap_times(MEDIUM, (1.0, 0.004, 0, 10), 20)
    laps = _two_stints("medium", [t + (3.0 if age > 18 else 0.0) for age, t in enumerate(times, 1)])
    entry = _calibrate(laps).compounds[MEDIUM]
    assert entry.tire_data.degradation_rate == pytest.approx(0.004, rel=1e-6)
    assert entry.tire_age_range == (1, 18) and entry.outliers_rejected == 4
    assert any(
        note.startswith("4 slower lap(s) at tyre age 19-20, older than every lap used, were rejected as outliers "
                        "(+3.000s against the fit). Wear may accelerate beyond tyre age 18")
        for note in entry.notes
    )


def test_window_end_inside_the_warm_up_is_detected():
    # The engine accepts a peak window that ends inside the warm-up (W 4, E 2): after the ramp the tyre starts
    # degradation_rate x (4 - 2) below its peak.
    times = engine_lap_times(MEDIUM, (1.0, 0.004, 4, 2), 25)
    result = _calibrate(_two_stints("medium", times))
    tire = result.tire_data[MEDIUM]
    assert (tire.warm_up_laps, tire.peak_performance_window) == (4, (2, 2))
    assert tire.degradation_rate == pytest.approx(0.004, rel=1e-6)
    assert result.estimated_base_lap_time == pytest.approx(BASE_LAP_TIME, rel=1e-9)
    assert result.compounds[MEDIUM].residual_std_s < 1e-6
    projected = engine_lap_times(
        MEDIUM, (tire.base_performance, tire.degradation_rate, 4, 2), 25, base_lap_time=result.estimated_base_lap_time
    )
    assert projected == pytest.approx(times, abs=1e-6)
    # A supplied window end does not cap the warm-up lengths tried either.
    supplied = _calibrate(_two_stints("medium", times), peak_window_end={MEDIUM: 2}).tire_data[MEDIUM]
    assert (supplied.warm_up_laps, supplied.peak_performance_window) == (4, (2, 2))


def test_degradation_from_the_youngest_observed_age_is_flagged():
    # Degradation starts after tyre age 2, but the laps start at age 6 (used sets): every window end up to 6 fits
    # them equally, so the peak pace cannot be measured and the estimate describes the pace at age 6.
    times = engine_lap_times(MEDIUM, (1.0, 0.004, 0, 2), 25)
    used_sets = [
        {"compound": "medium", "tire_age": age, "lap_time": times[age - 1]} for _ in range(2) for age in range(6, 26)
    ]
    result = _calibrate(used_sets)
    entry = result.compounds[MEDIUM]
    assert entry.status == STATUS_ESTIMATED and entry.tire_data.peak_performance_window == (1, 6)
    assert result.estimated_base_lap_time == pytest.approx(times[5], rel=1e-9)  # 81.30 s at age 6, not 80 s
    assert any(note.startswith("Degradation is already under way at the youngest tyre age used (6), so the peak pace "
                               "is not identifiable") for note in entry.notes)
    # Laps from a fresh set identify the window end, and no such note is given.
    entry = _calibrate(_two_stints("medium", times)).compounds[MEDIUM]
    assert entry.tire_data.peak_performance_window == (1, 2)
    assert not any("not identifiable" in note for note in entry.notes)


def test_small_samples_use_the_small_sample_corrected_rejection_threshold():
    # Eight laps with about 0.1 s of scatter plus one slower lap. 1.4826 x MAD alone would put the threshold near
    # 0.42 s here. The corrections for the small sample (n / (n - 0.8)) and the fitted parameter
    # (sqrt(n / (n - p))) raise it to about 0.485 s, so a lap 0.47 s slow is kept and one 0.52 s slow is not.
    def history(slow_by: float) -> List[dict]:
        pattern = (0.10, -0.10, 0.05, slow_by, -0.05, 0.10, -0.10, 0.0)
        return [{"compound": "medium", "tire_age": age, "lap_time": 81.0 + d} for age, d in enumerate(pattern, 1)]

    kept = _calibrate(history(0.47)).compounds[MEDIUM]
    assert (kept.status, kept.laps_used, kept.outliers_rejected) == (STATUS_ESTIMATED, 8, 0)
    rejected = _calibrate(history(0.52)).compounds[MEDIUM]
    assert (rejected.status, rejected.laps_used, rejected.outliers_rejected) == (STATUS_ESTIMATED, 7, 1)


def contaminated_history(seed: int, kind: str, junk_fraction: float) -> Tuple[List[dict], int]:
    """Two stints (tyre ages 2-29) of 80 s + 0.05 s per lap past age 10 with 0.15 s noise; a random fraction of
    the laps is junk: 6 s slower ("slow") or uniform in 20-600 s ("garbage"). Returns the laps and the junk count."""

    rng = np.random.default_rng(seed)
    ages = np.tile(np.arange(2, 30), 2)
    times = 80.0 + 0.05 * np.maximum(0, ages - 10) + rng.normal(0.0, 0.15, size=ages.size)
    junk = rng.random(ages.size) < junk_fraction
    if kind == "slow":
        times[junk] += 6.0
    else:
        times[junk] = rng.uniform(20.0, 600.0, int(junk.sum()))
    laps = [{"compound": "medium", "tire_age": int(a), "lap_time": float(t)} for a, t in zip(ages, times, strict=True)]
    return laps, int(junk.sum())


@pytest.mark.parametrize("seed, kind, junk_fraction", [(0, "slow", 0.2), (1, "garbage", 0.1)])
def test_contaminated_history_keeps_the_degradation_structure(seed, kind, junk_fraction):
    # Every junk lap costs each candidate structure the same capped loss; the structure is still chosen on the
    # remaining laps' fit (known noise scale), and each candidate's own refits never shrink below 5 laps.
    laps, junk = contaminated_history(seed, kind, junk_fraction)
    entry = _calibrate(laps).compounds[MEDIUM]
    assert entry.status == STATUS_ESTIMATED and entry.outliers_rejected == junk
    assert entry.peak_lap_time == pytest.approx(80.0, abs=0.1)
    assert entry.degradation_observed and abs(entry.tire_data.peak_performance_window[1] - 10) <= 2
    assert entry.tire_data.degradation_rate == pytest.approx(0.05 / 80.0, rel=0.3)


def test_alternating_outlier_rejection_keeps_the_largest_set_of_laps():
    # In this seed of the noisy history one soft lap sits on the rejection threshold, and rejection alternates
    # between keeping and dropping it. The larger set is used (only the injected +7 s lap is rejected), with a note.
    laps, injected = noisy_history_with_outliers(13)
    result = _calibrate(laps)
    soft = result.compounds[SOFT]
    assert any(note.startswith("Outlier rejection alternated on borderline laps") for note in soft.notes)
    assert (soft.clean_laps, soft.laps_used, soft.outliers_rejected) == (23, 22, 1)
    soft_outliers = [lap.index for lap in result.excluded_laps if lap.reason == "outlier" and lap.compound == SOFT]
    assert soft_outliers == [_index(laps, SOFT, 0, 6)]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_calibration_is_deterministic_and_independent_of_lap_order():
    laps = build_history(noise=0.2, seed=3)
    first = tyre_calibration_to_dict(_calibrate(laps))
    assert tyre_calibration_to_dict(_calibrate(laps)) == first

    order = list(range(len(laps)))
    random.Random(5).shuffle(order)
    shuffled = _calibrate([laps[i] for i in order])
    reference = _calibrate(laps)
    assert shuffled.reference_compound == reference.reference_compound
    assert shuffled.estimated_base_lap_time == pytest.approx(reference.estimated_base_lap_time, rel=1e-12)
    for compound, tire in reference.tire_data.items():
        other = shuffled.tire_data[compound]
        assert other.warm_up_laps == tire.warm_up_laps
        assert other.peak_performance_window == tire.peak_performance_window
        assert other.base_performance == pytest.approx(tire.base_performance, rel=1e-12)
        assert other.degradation_rate == pytest.approx(tire.degradation_rate, rel=1e-9)
    # Excluded laps are the same laps, reported at their positions in the shuffled input.
    assert sorted((order[lap.index], lap.reason) for lap in shuffled.excluded_laps) == sorted(
        (lap.index, lap.reason) for lap in reference.excluded_laps
    )


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def _lap(**changes):
    lap = {"compound": "soft", "tire_age": 3, "lap_time": 80.0}
    lap.update(changes)
    return lap


@pytest.mark.parametrize(
    "laps, kwargs, message",
    [
        ([], {}, "1 to 2000 laps"),
        ("soft 80.0", {}, "list of laps"),
        ([_lap(lap_time=float("nan"))], {}, "lap_time must be finite"),
        ([_lap(lap_time=float("inf"))], {}, "lap_time must be finite"),
        ([_lap(lap_time=10.0)], {}, "lap_time must be in [20, 600]"),
        ([_lap(lap_time="80.0")], {}, "lap_time must be a number"),
        ([_lap(lap_time=True)], {}, "lap_time must be a number"),
        ([_lap(tire_age=0)], {}, "tire_age must be in [1, 100]"),
        ([_lap(tire_age=2.5)], {}, "tire_age must be an integer"),
        ([_lap(compound="supersoft")], {}, "compound must be one of"),
        ([_lap(pit_in="yes")], {}, "pit_in must be true or false"),
        ([_lap(sector_1=28.0)], {}, "Unknown laps[0] keys"),
        ([{"compound": "soft", "tire_age": 3}], {}, "missing ['lap_time']"),
        ([42], {}, "must be a LapRecord or a mapping"),
        ([_lap()], {"weather": "sunny"}, "weather must be one of"),
        ([_lap()], {"track_temperature": float("nan")}, "track_temperature must be finite"),
        ([_lap()], {"pit_stop_delta": -1.0}, "pit_stop_delta must be in [0, 120]"),
        ([_lap()], {"fuel_correction_s_per_lap": 0.9}, "fuel_correction_s_per_lap must be in [0, 0.5]"),
        ([_lap()], {"fuel_correction_s_per_lap": 0.05}, "needs race_lap on every unflagged lap"),
        ([_lap(race_lap=1), _lap(race_lap=200)], {"fuel_correction_s_per_lap": 0.5}, "fuel-corrected lap time"),
        ([_lap()], {"peak_window_end": {"hard": 10}}, "'hard', which has no laps"),
        ([_lap()], {"peak_window_end": {"soft": 0}}, "peak_window_end[soft] must be in [1, 200]"),
        ([_lap()], {"warm_up_laps": {"soft": 11}}, "warm_up_laps[soft] must be in [0, 10]"),
        ([_lap()], {"warm_up_laps": {"soft": 2, SOFT: 3}}, "more than once"),
        ([_lap()], {"warm_up_laps": [("soft", 2)]}, "must be a mapping"),
    ],
)
def test_invalid_input_raises_value_error(laps, kwargs, message):
    with pytest.raises(ValueError) as error:
        _calibrate(laps, **kwargs)
    assert message in str(error.value)


def test_too_many_laps_are_rejected():
    laps = [_lap(tire_age=1 + i % 30) for i in range(MAX_CALIBRATION_LAPS + 1)]
    with pytest.raises(ValueError, match="1 to 2000 laps"):
        _calibrate(laps)
    with pytest.raises(ValidationError):
        TyreCalibrationRequest.model_validate(
            {"laps": laps, "weather": "dry", "track_temperature": 30.0, "pit_stop_delta": 22.0}
        )


def test_lap_record_objects_are_accepted_and_validated():
    records = [LapRecord(MEDIUM, age, 81.0 + 0.1 * age) for age in range(1, 11)]
    assert _calibrate(records).compounds[MEDIUM].status == STATUS_ESTIMATED
    with pytest.raises(ValueError, match="tire_age must be in"):
        _calibrate([LapRecord(MEDIUM, 101, 81.0)])


# ---------------------------------------------------------------------------
# Request schema and JSON response
# ---------------------------------------------------------------------------


def _request(**changes):
    body = {"laps": build_history(), "weather": "dry", "track_temperature": 30.0, "pit_stop_delta": PIT_DELTA}
    body.update(changes)
    return body


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": 1},
        {"weather": "sunny"},
        {"track_temperature": "30"},
        {"track_temperature": float("nan")},
        {"pit_stop_delta": True},
        {"fuel_correction_s_per_lap": 0.6},
        {"peak_window_end": {"soft": 0}},
        {"warm_up_laps": {"soft": 11}},
        {"peak_window_end": {"intermediate": 8}},  # no intermediate laps
        {"laps": []},
        {"laps": [{"compound": "soft", "tire_age": 1, "lap_time": 80.0, "sector": 1}]},
        {"laps": [{"compound": "soft", "tire_age": 1.0, "lap_time": 80.0}]},
        {"laps": [{"compound": "soft", "tire_age": 1, "lap_time": 80.0, "pit_in": 1}]},
        {"laps": [{"compound": "soft", "tire_age": 1, "lap_time": float("inf")}]},
        {"laps": [{"compound": "soft", "tire_age": 1, "lap_time": 80.0}], "fuel_correction_s_per_lap": 0.05},
    ],
)
def test_request_schema_rejects_invalid_bodies(changes):
    with pytest.raises(ValidationError):
        TyreCalibrationRequest.model_validate(_request(**changes))


def test_request_schema_round_trip_gives_a_json_safe_response_usable_by_strategy_request():
    request = TyreCalibrationRequest.model_validate(
        _request(fuel_correction_s_per_lap=0.0, peak_window_end={"hard": 20}, warm_up_laps={"soft": 3})
    )
    body = calibrate_tyres_response(request)
    text = json.dumps(body, allow_nan=False)
    assert json.loads(text) == body

    assert body["reference_compound"] == "soft"
    assert body["estimated_base_lap_time_s"] == pytest.approx(BASE_LAP_TIME, abs=0.001)
    assert body["tire_data"]["soft"] == {
        "compound": "soft",
        "base_performance": 1.0,
        "degradation_rate": 0.004,
        "warm_up_laps": 3,
        "peak_performance_window": [3, 8],
        "pit_stop_delta": 22.0,
    }
    assert body["compounds"]["hard"]["peak_window_end_source"] == "supplied"
    assert body["compounds"]["soft"]["fit"]["tire_age_range"] == [2, 14]
    assert body["conditions"] == {
        "weather": "dry",
        "track_temperature": 30.0,
        "pit_stop_delta": 22.0,
        "fuel_correction_s_per_lap": 0.0,
        "fuel_reference_lap": None,
    }
    assert len(body["excluded_laps"]) == 2 * len(STINTS)
    # Every tire_data entry validates as StrategyRequest.tire_data.
    StrategyRequest.model_validate(
        {
            "telemetry": {"lap_times": [body["estimated_base_lap_time_s"]], **NEUTRAL_TELEMETRY},
            "car_status": {"engine_wear": 0.2, "brake_wear": 0.2},
            "driver_profile": {"tire_management": 0.7, "risk_tolerance": 0.5},
            "tire_data": body["tire_data"],
            "race_state": {"current_lap": 10, "total_laps": 57, "weather": "dry", "track_temperature": 30.0},
        }
    )


def test_documented_request_example_produces_estimates():
    example = TyreCalibrationRequest.model_json_schema()["examples"][0]
    body = calibrate_tyres_response(TyreCalibrationRequest.model_validate(example))
    assert set(body["tire_data"]) == {"soft", "medium"}
    assert body["compounds"]["soft"]["fit"]["residual_std_s"] < 0.05
    json.dumps(body, allow_nan=False)
