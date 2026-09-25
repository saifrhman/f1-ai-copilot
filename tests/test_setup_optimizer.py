"""Behavioural tests for the setup optimiser (seeded Optuna TPE + deterministic local refinement over a
documented heuristic objective)."""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import optuna
import pytest

from core_modules.setup_optimizer.schemas import SetupRequest
from core_modules.setup_optimizer.setup_recommender import (
    AGREEMENT_TOLERANCE,
    DEFAULT_N_TRIALS,
    DEFAULT_SEED,
    MAX_N_TRIALS,
    MIN_N_TRIALS,
    N_STARTUP_TRIALS,
    SETUP_BOUNDS,
    DriverPreferences,
    SetupConfiguration,
    SetupInputs,
    SetupOptimizer,
    TrackProfile,
    TrackType,
    WeatherCondition,
    WeatherData,
    objective_terms,
    recommend_setup,
    recommend_setup_from_inputs,
    refine_locally,
    _derive_context,
)

MONZA = TrackProfile("Monza-like", 5793.0, 11, 6, 3, TrackType.HIGH_SPEED, 250.0, 0.2)
SILVERSTONE = TrackProfile("Silverstone-like", 5891.0, 18, 8, 4, TrackType.HIGH_SPEED, 220.0, 0.6)
SPA = TrackProfile("Spa-like", 7004.0, 20, 10, 6, TrackType.MIXED, 200.0, 0.7)
MONACO = TrackProfile("Monaco-like", 3337.0, 19, 2, 15, TrackType.TECHNICAL, 160.0, 0.95)
DRY = WeatherData(WeatherCondition.DRY, 24.0, 50.0)
WET = WeatherData(WeatherCondition.WET, 16.0, 95.0)
NEUTRAL = DriverPreferences()
CAUTIOUS = DriverPreferences(risk_tolerance=0.0, tire_management=1.0)
AGGRESSIVE = DriverPreferences(risk_tolerance=1.0, tire_management=0.0)
# A scenario whose heuristic objective has two local optima (a low- and a high-rear-wing trim).
TWO_OPTIMA = (
    DriverPreferences(
        preferred_wing_angles={"front": 0.4}, preferred_diff_settings={"coast": 53.0}, risk_tolerance=0.86, tire_management=0.09
    ),
    TrackProfile(None, 7774.0, 30, 5, 9, TrackType.TECHNICAL, 175.0, 0.5),
    WeatherData(WeatherCondition.WET, 20.0),
)

API_PAYLOAD = {
    "driver_preferences": {"risk_tolerance": 0.5, "tire_management": 0.7},
    "track_profile": {
        "track_name": "Silverstone Circuit",
        "track_length": 5891,
        "corners": 18,
        "high_speed_sections": 8,
        "low_speed_sections": 4,
        "track_type": "high_speed",
        "average_speed": 220,
        "downforce_requirement": 0.6,
    },
    "weather": {"condition": "dry", "temperature": 24, "humidity": 50},
}

OPTIMIZER = SetupOptimizer()
CENTRE = {n: (lo + hi) / 2 for n, (lo, hi) in SETUP_BOUNDS.items()}


def _flat_setup(result: dict) -> dict:
    return {
        "ride_height": result["ride_height"],
        "front_wing_angle": result["front_wing_angle"],
        "rear_wing_angle": result["rear_wing_angle"],
        "brake_bias": result["brake_bias"],
        **{f"diff_{key}": value for key, value in result["diff_settings"].items()},
        **result["suspension_settings"],
    }


def _assert_consistent(result: dict, driver: DriverPreferences, track: TrackProfile, weather: WeatherData) -> None:
    """The reported objective, breakdown, confidence and the returned setup all describe the same setup."""
    assert result["objective_value"] == pytest.approx(sum(result["objective_breakdown"].values()), abs=1e-5)
    ctx = _derive_context(driver, track, weather)
    recomputed = sum(objective_terms(SetupConfiguration.from_params(_flat_setup(result)), ctx).values())
    assert recomputed == pytest.approx(result["objective_value"], rel=1e-3)
    starts = result["refinement_starts"]
    assert [s["start"] for s in starts] == ["optuna_best_trial", "rule_of_thumb_baseline", "bounds_centre"]
    assert starts[0]["start_objective"] == pytest.approx(result["best_trial_objective"], abs=1e-6)
    assert starts[1]["start_objective"] == pytest.approx(result["baseline_objective_value"], abs=1e-6)
    assert result["objective_value"] == pytest.approx(min(s["refined_objective"] for s in starts), abs=1e-5)
    assert all(s["converged"] and s["refined_objective"] <= s["start_objective"] for s in starts)
    agreeing = sum(s["max_parameter_deviation"] <= AGREEMENT_TOLERANCE for s in starts)
    assert result["confidence"] == pytest.approx(agreeing / len(starts), abs=1e-3)
    assert result["improvement_over_baseline"] == pytest.approx(
        result["baseline_objective_value"] - result["objective_value"], abs=1e-5
    )


# --------------------------------------------------------------------------- optimisation matters


@pytest.mark.parametrize(
    "track, weather",
    [(SILVERSTONE, DRY), (MONACO, WET), (MONZA, DRY)],
    ids=["silverstone-dry", "monaco-wet", "monza-dry"],
)
def test_search_beats_rule_of_thumb_baseline_and_random_startup(track, weather):
    result = OPTIMIZER.recommend_setup(NEUTRAL, track, weather)

    assert result["selected_source"] == "optuna_best_trial_refined"
    assert result["trials"] == DEFAULT_N_TRIALS
    assert result["objective_value"] < result["baseline_objective_value"]
    assert result["improvement_over_baseline"] > 0
    # The TPE stage on its own: the model-guided phase beat the random start and the baseline ...
    assert result["best_trial_objective"] < result["random_startup_best_objective"]
    assert result["best_trial_objective"] < result["baseline_objective_value"]
    assert N_STARTUP_TRIALS <= result["best_trial_number"] < DEFAULT_N_TRIALS
    assert result["best_trial_improvement_over_baseline"] == pytest.approx(
        result["baseline_objective_value"] - result["best_trial_objective"], abs=1e-5
    )
    # ... and the refinement of that trial is reported separately (it never makes things worse).
    assert result["refinement_improvement_over_best_trial"] >= 0
    assert result["objective_value"] <= result["best_trial_objective"]
    _assert_consistent(result, NEUTRAL, track, weather)
    # The baseline is evaluated with the same objective, not enqueued as the answer.
    assert _flat_setup(result) != _flat_setup({**result, **result["baseline_setup"]})


def _random_search_best(track, weather, n_trials, seed):
    ctx = _derive_context(NEUTRAL, track, weather)

    def objective(trial):
        params = {name: trial.suggest_float(name, lo, hi) for name, (lo, hi) in SETUP_BOUNDS.items()}
        return sum(objective_terms(SetupConfiguration.from_params(params), ctx).values())

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_value


@pytest.mark.parametrize("track, weather", [(SILVERSTONE, DRY), (MONACO, WET)], ids=["silverstone-dry", "monaco-wet"])
def test_tpe_beats_pure_random_search_with_the_same_budget(track, weather):
    seeds = (0, 1, 2)
    tpe = [OPTIMIZER.recommend_setup(NEUTRAL, track, weather, seed=s)["best_trial_objective"] for s in seeds]
    random = [_random_search_best(track, weather, DEFAULT_N_TRIALS, s) for s in seeds]

    assert sum(tpe) / len(tpe) < sum(random) / len(random)


def test_larger_budget_with_same_seed_never_does_worse():
    small = OPTIMIZER.recommend_setup(NEUTRAL, SPA, DRY, n_trials=MIN_N_TRIALS, seed=3)
    large = OPTIMIZER.recommend_setup(NEUTRAL, SPA, DRY, n_trials=DEFAULT_N_TRIALS, seed=3)

    # Same seed -> the first trials are identical, so extra trials can only improve the best value.
    assert large["best_trial_objective"] <= small["best_trial_objective"]
    assert large["best_trial_objective"] < small["best_trial_objective"]


def test_when_tpe_loses_to_the_baseline_the_returned_setup_is_still_a_refined_search_result():
    # Replaces the old "baseline is returned" test: the raw rule of thumb is no longer returned as the
    # answer, because it is only one refinement start. With the minimum budget on Monaco this seed's
    # TPE stage loses to the baseline, yet refinement reaches a much better setup.
    result = OPTIMIZER.recommend_setup(NEUTRAL, MONACO, DRY, n_trials=MIN_N_TRIALS, seed=0)

    assert result["best_trial_objective"] >= result["baseline_objective_value"]  # TPE lost ...
    assert result["best_trial_improvement_over_baseline"] <= 0
    assert "No TPE trial in 16 beat the rule-of-thumb baseline" in result["reasoning"]
    assert result["objective_value"] < result["baseline_objective_value"]  # ... the returned setup did not
    assert result["improvement_over_baseline"] > 0
    assert _flat_setup(result) != _flat_setup({**result, **result["baseline_setup"]})
    # Confidence describes the returned (refined) setup, not the TPE stage.
    _assert_consistent(result, NEUTRAL, MONACO, DRY)


# --------------------------------------------------------------------------- reproducibility / thread safety


def test_same_seed_reproduces_the_full_result():
    first = OPTIMIZER.recommend_setup(NEUTRAL, SILVERSTONE, DRY, seed=11)
    second = SetupOptimizer().recommend_setup(NEUTRAL, SILVERSTONE, DRY, seed=11)

    assert first == second
    assert first["seed"] == 11


def test_recommendation_does_not_depend_on_the_seed_at_the_default_budget():
    # Regression: TPE alone stopped short of the optimum, so at 128 trials the front wing ranged
    # 3.1-13.4 deg and diff preload 8-96 % across seeds. The seed must still drive the TPE stage.
    results = [OPTIMIZER.recommend_setup(NEUTRAL, SILVERSTONE, DRY, seed=s) for s in (0, 6, 7)]

    assert len({r["best_trial_objective"] for r in results}) == 3
    reference = _flat_setup(results[0])
    for result in results[1:]:
        for name, value in _flat_setup(result).items():
            lo, hi = SETUP_BOUNDS[name]
            assert abs(value - reference[name]) <= AGREEMENT_TOLERANCE * (hi - lo), name
        assert result["ride_height"] == pytest.approx(reference["ride_height"], abs=0.1)
        assert result["front_wing_angle"] == pytest.approx(reference["front_wing_angle"], abs=0.1)
        assert result["rear_wing_angle"] == pytest.approx(reference["rear_wing_angle"], abs=0.1)
        assert result["objective_value"] == pytest.approx(results[0]["objective_value"], abs=1e-4)
        assert result["confidence"] == 1.0


def test_concurrent_requests_match_sequential_results():
    cases = [(MONZA, DRY, 1), (MONACO, WET, 2), (SPA, DRY, 3), (SILVERSTONE, WET, 4)]
    sequential = [OPTIMIZER.recommend_setup(NEUTRAL, t, w, n_trials=MIN_N_TRIALS, seed=s) for t, w, s in cases]

    def run(case):
        track, weather, seed = case
        return OPTIMIZER.recommend_setup(NEUTRAL, track, weather, n_trials=MIN_N_TRIALS, seed=seed)

    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(run, cases))

    assert concurrent == sequential


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_study_creation_and_trials_do_not_log_at_info_level():
    optuna_logger = logging.getLogger("optuna")
    handler = _CollectingHandler()
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO)  # e.g. another module turned INFO on
    optuna_logger.addHandler(handler)
    try:
        OPTIMIZER.recommend_setup(NEUTRAL, MONZA, DRY, n_trials=MIN_N_TRIALS)
    finally:
        optuna_logger.removeHandler(handler)
        optuna.logging.set_verbosity(previous)
    assert [r.getMessage() for r in handler.records if r.levelno < logging.WARNING] == []


def test_a_stricter_optuna_verbosity_is_left_alone():
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    try:
        OPTIMIZER.recommend_setup(NEUTRAL, MONZA, DRY, n_trials=MIN_N_TRIALS)
        assert optuna.logging.get_verbosity() == optuna.logging.ERROR
    finally:
        optuna.logging.set_verbosity(previous)


# --------------------------------------------------------------------------- confidence


@pytest.mark.parametrize("seed", [0, 1, DEFAULT_SEED])
def test_two_local_optima_lower_confidence_and_the_better_optimum_is_returned_for_every_seed(seed):
    driver, track, weather = TWO_OPTIMA
    result = OPTIMIZER.recommend_setup(driver, track, weather, n_trials=MIN_N_TRIALS, seed=seed)
    refined = [s["refined_objective"] for s in result["refinement_starts"]]

    # The starts end in different optima, so confidence drops and the spread shows the alternative trim.
    assert max(refined) - min(refined) > 0.01
    assert result["confidence"] < 1.0
    assert result["parameter_spread"]["rear_wing_angle"] > 10.0
    assert result["objective_value"] == pytest.approx(min(refined), abs=1e-6)
    _assert_consistent(result, driver, track, weather)
    # Whatever start found it, the same (better) optimum comes back for every seed.
    reference = OPTIMIZER.recommend_setup(driver, track, weather, n_trials=MIN_N_TRIALS, seed=1)
    assert result["objective_value"] == pytest.approx(reference["objective_value"], abs=1e-5)
    assert result["rear_wing_angle"] == pytest.approx(reference["rear_wing_angle"], abs=0.2)


def test_single_optimum_gives_full_agreement_and_zero_spread():
    result = OPTIMIZER.recommend_setup(NEUTRAL, SPA, DRY)

    assert result["confidence"] == 1.0
    for name, (lo, hi) in SETUP_BOUNDS.items():
        assert result["parameter_spread"][name] <= AGREEMENT_TOLERANCE * (hi - lo), name
    _assert_consistent(result, NEUTRAL, SPA, DRY)


# --------------------------------------------------------------------------- local refinement


def test_refinement_is_deterministic_bounded_and_only_moves_free_parameters():
    ctx = _derive_context(NEUTRAL, SILVERSTONE, DRY)
    corner = SetupConfiguration.from_params({n: lo for n, (lo, _) in SETUP_BOUNDS.items()})
    free = [n for n in SETUP_BOUNDS if n != "rear_wing_angle"]

    first = refine_locally(corner, ctx, free)
    assert first == refine_locally(corner, ctx, free)
    setup, value, evaluations, converged = first
    assert converged and 0 < evaluations
    assert value < sum(objective_terms(corner, ctx).values())
    assert setup.rear_wing_angle == corner.rear_wing_angle
    for name, (lo, hi) in SETUP_BOUNDS.items():
        assert lo <= getattr(setup, name) <= hi


# --------------------------------------------------------------------------- bounds and engineering behaviour


@pytest.mark.parametrize(
    "driver, track, weather",
    [
        (CAUTIOUS, MONACO, WET),
        (AGGRESSIVE, MONZA, DRY),
        (
            NEUTRAL,
            TrackProfile(None, 500.0, 1, 0, 0, TrackType.LOW_SPEED, 20.0, 0.0),
            WeatherData(WeatherCondition.INTERMEDIATE, -10.0),
        ),
        (
            NEUTRAL,
            TrackProfile("fast", 25_000.0, 100, 100, 0, TrackType.HIGH_SPEED, 400.0, 1.0),
            WeatherData(WeatherCondition.DRY, 50.0),
        ),
    ],
    ids=["cautious-wet", "aggressive-dry", "min-extremes", "max-extremes"],
)
def test_outputs_respect_bounds_and_are_finite(driver, track, weather):
    result = OPTIMIZER.recommend_setup(driver, track, weather, n_trials=MIN_N_TRIALS)

    for name, value in _flat_setup(result).items():
        lo, hi = SETUP_BOUNDS[name]
        assert lo <= value <= hi, name
    for value in result["tire_pressures_psi"].values():
        assert math.isfinite(value) and 5.0 < value < 40.0
    assert 0.0 < result["confidence"] <= 1.0
    assert all(value >= 0.0 and math.isfinite(value) for value in result["objective_breakdown"].values())
    _assert_consistent(result, driver, track, weather)


@pytest.mark.parametrize("driver", [NEUTRAL, CAUTIOUS, AGGRESSIVE], ids=["neutral", "cautious", "aggressive"])
@pytest.mark.parametrize("weather", [DRY, WET], ids=["dry", "wet"])
@pytest.mark.parametrize("seed", [0, 3])
def test_high_speed_track_gets_less_rear_wing_than_technical_high_downforce_track(driver, weather, seed):
    # Regression: Monza cautious/wet/seed 3 used to get 17.7 deg of rear wing vs 11.75 at Monaco.
    fast = OPTIMIZER.recommend_setup(driver, MONZA, weather, n_trials=MIN_N_TRIALS, seed=seed)
    technical = OPTIMIZER.recommend_setup(driver, MONACO, weather, n_trials=MIN_N_TRIALS, seed=seed)

    assert fast["rear_wing_angle"] + 5.0 < technical["rear_wing_angle"]
    assert fast["objective_breakdown"]["straight_line_drag"] < technical["objective_breakdown"]["straight_line_drag"]


@pytest.mark.parametrize("track", [SPA, MONACO], ids=["spa", "monaco"])
@pytest.mark.parametrize("seed", [0, 1, DEFAULT_SEED])
def test_wet_conditions_raise_recommended_ride_height(track, seed):
    dry = OPTIMIZER.recommend_setup(NEUTRAL, track, DRY, seed=seed)
    wet = OPTIMIZER.recommend_setup(NEUTRAL, track, WET, seed=seed)

    assert wet["ride_height"] > dry["ride_height"]


def test_wet_model_penalises_the_same_low_ride_height_more():
    low = SetupConfiguration.from_params({**CENTRE, "ride_height": 66.0})
    dry_terms = objective_terms(low, _derive_context(NEUTRAL, SPA, DRY))
    wet_terms = objective_terms(low, _derive_context(NEUTRAL, SPA, WET))

    assert wet_terms["bottoming_risk"] > 2 * dry_terms["bottoming_risk"]


def test_bottoming_penalty_keeps_growing_below_the_limit():
    ctx = _derive_context(NEUTRAL, MONACO, WET)
    penalties = [
        objective_terms(SetupConfiguration.from_params({**CENTRE, "ride_height": h}), ctx)["bottoming_risk"]
        for h in (75.0, 70.0, 65.0, 60.0)
    ]
    assert penalties == sorted(penalties)
    assert penalties[-1] - penalties[-2] > 0.5  # no saturation plateau at the lowest ride heights


def _model_optimum(driver, track, weather) -> SetupConfiguration:
    ctx = _derive_context(driver, track, weather)
    setup, _, _, converged = refine_locally(SetupConfiguration.from_params(CENTRE), ctx, list(SETUP_BOUNDS))
    assert converged
    return setup


@pytest.mark.parametrize("track", [MONZA, SILVERSTONE, SPA, MONACO], ids=["monza", "silverstone", "spa", "monaco"])
@pytest.mark.parametrize("weather", [DRY, WET], ids=["dry", "wet"])
def test_diff_optimum_lies_strictly_inside_its_bounds(track, weather):
    # Regression: preload used to buy entry stability for free, so its optimum was always 100 %
    # (a locked diff) combined with a meaningless 10-50 % power ramp.
    optimum = _model_optimum(NEUTRAL, track, weather)

    for name in ("diff_preload", "diff_power", "diff_coast"):
        assert 5.0 < getattr(optimum, name) < 95.0, name


def test_preload_trade_off_follows_the_track_layout():
    # Documented heuristic: preload steadies fast corners but pushes the front at tight apexes.
    fast = _model_optimum(NEUTRAL, MONZA, DRY)
    tight = _model_optimum(NEUTRAL, MONACO, DRY)

    assert tight.diff_preload + 10.0 < fast.diff_preload
    ctx = _derive_context(NEUTRAL, MONACO, DRY)
    more_preload = objective_terms(replace(tight, diff_preload=90.0), ctx)
    assert more_preload["diff_apex_understeer"] > objective_terms(tight, ctx)["diff_apex_understeer"]


@pytest.mark.parametrize("track, weather", [(SPA, DRY), (MONACO, WET)], ids=["spa-dry", "monaco-wet"])
def test_driver_style_shapes_balance_and_off_throttle_lock(track, weather):
    cautious = OPTIMIZER.recommend_setup(CAUTIOUS, track, weather, n_trials=MIN_N_TRIALS)
    aggressive = OPTIMIZER.recommend_setup(AGGRESSIVE, track, weather, n_trials=MIN_N_TRIALS)

    # Documented heuristic: tyre-savers / cautious drivers want understeer and a stable entry,
    # risk-takers a pointy car that rotates.
    assert cautious["handling_balance"]["driver_target"] > aggressive["handling_balance"]["driver_target"]
    assert cautious["handling_balance"]["high_speed"] > aggressive["handling_balance"]["high_speed"] + 0.1
    assert cautious["handling_balance"]["low_speed"] > aggressive["handling_balance"]["low_speed"] + 0.1
    assert cautious["diff_settings"]["coast"] > aggressive["diff_settings"]["coast"] + 20.0
    assert cautious["brake_bias"] > aggressive["brake_bias"]


def test_driver_preferences_pin_parameters_exactly_in_search_refinement_and_baseline():
    driver = DriverPreferences(
        preferred_ride_height=72.345,
        preferred_wing_angles={"rear": 7.125},
        preferred_diff_settings={"power": 40.0, "coast": 55.0},
    )
    result = OPTIMIZER.recommend_setup(driver, SILVERSTONE, DRY)

    assert result["selected_source"] == "optuna_best_trial_refined"
    assert result["ride_height"] == 72.345  # not rounded to 72.34
    assert result["rear_wing_angle"] == 7.125
    assert result["diff_settings"]["power"] == 40.0
    assert result["diff_settings"]["coast"] == 55.0
    assert result["baseline_setup"]["ride_height"] == 72.345
    assert result["baseline_setup"]["rear_wing_angle"] == 7.125
    assert result["pinned_by_driver"] == ["diff_coast", "diff_power", "rear_wing_angle", "ride_height"]
    assert all(result["parameter_spread"][name] == 0.0 for name in result["pinned_by_driver"])
    # The TPE stage honoured the pins: its reported best equals the objective of its (pinned) best setup,
    # and it cannot beat the pinned optimum that refinement reached from it.
    assert result["best_trial_objective"] >= result["objective_value"]
    _assert_consistent(result, driver, SILVERSTONE, DRY)


def test_unmodelled_inputs_are_reported_and_do_not_change_the_setup():
    base = OPTIMIZER.recommend_setup(NEUTRAL, SPA, DRY, n_trials=MIN_N_TRIALS)
    other = OPTIMIZER.recommend_setup(
        NEUTRAL, replace(SPA, track_name="renamed"), replace(DRY, humidity=5.0, wind_speed=20.0), n_trials=MIN_N_TRIALS
    )

    assert set(base["inputs_not_modelled"]) >= {"weather.humidity", "weather.wind_speed"}
    assert _flat_setup(other) == _flat_setup(base)
    assert other["objective_value"] == base["objective_value"]


def test_tyre_pressures_have_units_and_follow_the_ideal_gas_heuristic():
    cool = OPTIMIZER.recommend_setup(NEUTRAL, SPA, replace(DRY, temperature=10.0), n_trials=MIN_N_TRIALS)
    warm = OPTIMIZER.recommend_setup(NEUTRAL, SPA, DRY, n_trials=MIN_N_TRIALS)  # 24 degC
    wet = OPTIMIZER.recommend_setup(NEUTRAL, SPA, replace(WET, temperature=24.0), n_trials=MIN_N_TRIALS)

    assert warm["units"]["tire_pressures_psi"].startswith("psi")
    # (27 + 14.696) psia * (24 + 273.15) / (100 + 273.15) - 14.696 = 18.51 psi gauge
    assert warm["tire_pressures_psi"]["front_left"] == pytest.approx(18.51, abs=0.01)
    assert warm["tire_pressures_psi"]["front_left"] > warm["tire_pressures_psi"]["rear_left"]
    # Colder set temperature -> more pressure rise to the fixed running temperature -> set lower.
    assert cool["tire_pressures_psi"]["front_left"] < warm["tire_pressures_psi"]["front_left"]
    assert wet["tire_pressures_psi"] != warm["tire_pressures_psi"]
    # Temperature only feeds the tyre-pressure rule, never the searched setup.
    assert _flat_setup(cool) == _flat_setup(warm)


# --------------------------------------------------------------------------- validation


def _payload(**overrides):
    payload = {key: dict(value) for key, value in API_PAYLOAD.items()}
    for section, changes in overrides.items():
        payload[section].update(changes)
    return payload


@pytest.mark.parametrize(
    "overrides",
    [
        {"driver_preferences": {"risk_tolerance": 1.5}},
        {"driver_preferences": {"tire_management": float("nan")}},
        {"driver_preferences": {"risk_tolerance": "0.5"}},
        {"driver_preferences": {"risk_tolerance": 10**400}},
        {"driver_preferences": {"preferred_ride_height": 40}},
        {"driver_preferences": {"preferred_ride_height": 200}},
        {"driver_preferences": {"preferred_wing_angles": {"front": 40, "rear": -10}}},
        {"driver_preferences": {"preferred_wing_angles": {"fw": 2}}},
        {"driver_preferences": {"preferred_diff_settings": {"entry": 50}}},
        {"driver_preferences": {"preferred_diff_settings": {"power": 500}}},
        {"driver_preferences": {"preferred_diff_settings": "high"}},
        {"driver_preferences": {"unknown_preference": 1}},
        {"track_profile": {"corners": 2.5}},
        {"track_profile": {"corners": 0}},
        {"track_profile": {"low_speed_sections": -100}},
        {"track_profile": {"high_speed_sections": 19}},
        {"track_profile": {"high_speed_sections": 10, "low_speed_sections": 9}},
        {"track_profile": {"track_length": float("nan")}},
        {"track_profile": {"track_length": 1e9}},
        {"track_profile": {"track_length": 10**400}},
        {"track_profile": {"average_speed": -50}},
        {"track_profile": {"downforce_requirement": 1.2}},
        {"track_profile": {"track_type": None}},
        {"track_profile": {"track_type": "street"}},
        {"track_profile": {"track_name": "   "}},
        {"track_profile": {"surface": "asphalt"}},
        {"weather": {"temperature": -273}},
        {"weather": {"temperature": 60}},
        {"weather": {"temperature": float("nan")}},
        {"weather": {"temperature": 10**400}},
        {"weather": {"humidity": 101}},
        {"weather": {"wind_speed": -1}},
        {"weather": {"wind_speed": 300}},
        {"weather": {"condition": "snow"}},
    ],
)
def test_invalid_inputs_are_rejected_with_value_error(overrides):
    with pytest.raises(ValueError):
        recommend_setup(**_payload(**overrides))


@pytest.mark.parametrize(
    "section, field",
    [("weather", "temperature"), ("weather", "condition"), ("track_profile", "corners"), ("track_profile", "track_type")],
)
def test_missing_required_fields_are_rejected_with_value_error(section, field):
    payload = _payload()
    del payload[section][field]
    with pytest.raises(ValueError):
        recommend_setup(**payload)


@pytest.mark.parametrize("n_trials", [MIN_N_TRIALS - 1, MAX_N_TRIALS + 1, 8, 64.0, True])
def test_trial_budget_outside_bounds_is_rejected(n_trials):
    with pytest.raises(ValueError):
        recommend_setup(**API_PAYLOAD, n_trials=n_trials)
    with pytest.raises(ValueError):
        OPTIMIZER.recommend_setup(NEUTRAL, SPA, DRY, n_trials=n_trials)


@pytest.mark.parametrize(
    "driver, track, weather",
    [
        (DriverPreferences(risk_tolerance=-0.1), SPA, DRY),
        (DriverPreferences(risk_tolerance=10**400), SPA, DRY),
        (DriverPreferences(preferred_wing_angles={"front": 5.0, "flap": 1.0}), SPA, DRY),
        (DriverPreferences(preferred_diff_settings={"power": True}), SPA, DRY),
        (DriverPreferences(preferred_diff_settings={"power": 10**400}), SPA, DRY),
        (NEUTRAL, replace(SPA, corners=20.0), DRY),
        (NEUTRAL, replace(SPA, track_type="mixed"), DRY),
        (NEUTRAL, replace(SPA, average_speed=float("inf")), DRY),
        (NEUTRAL, replace(SPA, track_length=10**400), DRY),
        (NEUTRAL, SPA, replace(DRY, condition="dry")),
        (NEUTRAL, SPA, replace(DRY, humidity=-1.0)),
        (NEUTRAL, SPA, replace(DRY, temperature=10**400)),
    ],
)
def test_engine_validates_dataclass_inputs_directly(driver, track, weather):
    # Huge ints used to escape as OverflowError from float(); every invalid input must be a ValueError.
    with pytest.raises(ValueError):
        OPTIMIZER.recommend_setup(driver, track, weather, n_trials=MIN_N_TRIALS)


def test_optimizer_defaults_are_validated():
    with pytest.raises(ValueError):
        SetupOptimizer(n_trials=8)
    with pytest.raises(ValueError):
        SetupOptimizer(seed=-1)


# --------------------------------------------------------------------------- schema


def test_schema_round_trip_and_engine_conversion():
    request = SetupRequest.model_validate({**API_PAYLOAD, "n_trials": MIN_N_TRIALS, "seed": 5})
    dumped = request.model_dump(mode="json")
    assert SetupRequest.model_validate(dumped) == request
    assert dumped["track_profile"]["track_type"] == "high_speed"
    assert dumped["weather"]["wind_speed"] is None

    inputs = request.to_engine_inputs()
    assert inputs.track_profile.track_type is TrackType.HIGH_SPEED
    assert inputs.weather.condition is WeatherCondition.DRY
    assert (inputs.n_trials, inputs.seed) == (MIN_N_TRIALS, 5)

    via_schema = recommend_setup_from_inputs(inputs)
    via_dicts = recommend_setup(**API_PAYLOAD, n_trials=MIN_N_TRIALS, seed=5)
    assert via_schema == via_dicts


def test_schema_defaults_and_preference_conversion():
    request = SetupRequest.model_validate(
        {**API_PAYLOAD, "driver_preferences": {"preferred_wing_angles": {"rear": 6}, "preferred_diff_settings": {}}}
    )
    driver = request.to_engine_inputs().driver_preferences

    assert request.n_trials == DEFAULT_N_TRIALS and request.seed == DEFAULT_SEED
    assert driver.preferred_wing_angles == {"rear": 6.0}
    assert driver.preferred_diff_settings is None
    assert (driver.risk_tolerance, driver.tire_management) == (0.5, 0.5)


@pytest.mark.parametrize(
    "preferences, assumed",
    [
        ({}, ["driver_preferences.risk_tolerance=0.5", "driver_preferences.tire_management=0.5"]),
        ({"preferred_ride_height": 70.0}, ["driver_preferences.risk_tolerance=0.5", "driver_preferences.tire_management=0.5"]),
        ({"risk_tolerance": 0.5}, ["driver_preferences.tire_management=0.5"]),  # an explicit default is supplied
        ({"tire_management": 0.9}, ["driver_preferences.risk_tolerance=0.5"]),
        ({"risk_tolerance": 0.2, "tire_management": 0.5}, []),
    ],
)
def test_request_lists_the_driver_preference_defaults_it_assumed(preferences, assumed):
    request = SetupRequest.model_validate({**API_PAYLOAD, "driver_preferences": preferences})

    assert request.assumed_defaults() == assumed
    assert request.to_engine_inputs().assumed_defaults == tuple(assumed)


def test_recommendation_reports_assumed_defaults_from_every_request_entry_point():
    # Reviewer repro: driver_preferences {} silently used risk_tolerance=0.5 and tire_management=0.5.
    assumed = ["driver_preferences.risk_tolerance=0.5", "driver_preferences.tire_management=0.5"]
    via_schema = recommend_setup_from_inputs(
        SetupRequest.model_validate({**API_PAYLOAD, "driver_preferences": {}, "n_trials": MIN_N_TRIALS}).to_engine_inputs()
    )
    via_dicts = recommend_setup(**{**API_PAYLOAD, "driver_preferences": {}}, n_trials=MIN_N_TRIALS)

    assert via_schema == via_dicts
    assert via_schema["assumed_defaults"] == assumed
    assert f"Not supplied, so defaults were assumed: {', '.join(assumed)}." in via_schema["reasoning"]

    explicit = recommend_setup(**API_PAYLOAD, n_trials=MIN_N_TRIALS)  # both preferences supplied
    assert explicit["assumed_defaults"] == [] and "assumed" not in explicit["reasoning"]
    # Assuming the default changes nothing but the report: the setup is the one an explicit 0.5 gives.
    same_values = recommend_setup(
        **{**API_PAYLOAD, "driver_preferences": {"risk_tolerance": 0.5, "tire_management": 0.5}}, n_trials=MIN_N_TRIALS
    )
    assert _flat_setup(same_values) == _flat_setup(via_schema)


def test_engine_inputs_without_assumed_defaults_still_work():
    # Dataclass callers cannot tell a default from a choice, so nothing is reported as assumed.
    inputs = SetupInputs(NEUTRAL, SPA, DRY, n_trials=MIN_N_TRIALS)
    assert recommend_setup_from_inputs(inputs)["assumed_defaults"] == []
    assert OPTIMIZER.recommend_setup(NEUTRAL, SPA, DRY, n_trials=MIN_N_TRIALS)["assumed_defaults"] == []


def test_pinned_values_survive_the_dict_entry_point_unrounded():
    payload = _payload(driver_preferences={"preferred_ride_height": 72.345, "preferred_wing_angles": {"front": 4.125}})
    result = recommend_setup(**payload, n_trials=MIN_N_TRIALS)

    assert (result["ride_height"], result["front_wing_angle"]) == (72.345, 4.125)


def test_schema_publishes_bounds_and_descriptions_for_openapi():
    schema = SetupRequest.model_json_schema()
    n_trials = schema["properties"]["n_trials"]
    assert (n_trials["minimum"], n_trials["maximum"], n_trials["default"]) == (MIN_N_TRIALS, MAX_N_TRIALS, DEFAULT_N_TRIALS)
    assert schema["additionalProperties"] is False

    models = {"SetupRequest": schema, **schema["$defs"]}
    missing = [
        f"{model}.{field}"
        for model, definition in models.items()
        for field, spec in definition.get("properties", {}).items()
        if not spec.get("description")
    ]
    assert missing == []
