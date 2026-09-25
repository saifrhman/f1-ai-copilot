"""Behavioural tests for the heuristic strategy engine and its request schema."""

import dataclasses
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations, product

import pytest
from pydantic import ValidationError

from core_modules.strategy_optimizer.schemas import (
    StrategyRequest,
    generate_strategy_response,
    strategy_result_to_dict,
)
from core_modules.strategy_optimizer.strategy_engine import (
    DRY_COMPOUNDS,
    MAX_CANDIDATES,
    MAX_STOPS,
    RULE_SATISFIED,
    RULE_UNVERIFIED,
    RULE_VIOLATED,
    RULE_WAIVED,
    WET_WEATHER_COMPOUNDS,
    CarStatus,
    Competitor,
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


def create_test_data(weather=DRY, current_lap=8, total_laps=52, **race_extra):
    telemetry = {
        "lap_times": [80.5, 80.2, 80.8, 80.1, 80.3],
        "braking_consistency": 0.78,
        "throttle_aggressiveness": 0.72,
    }
    car_status = CarStatus(
        engine_wear=0.28,
        brake_wear=0.35,
        damage={"front_wing": 0.15, "floor": 0.08, "diffuser": 0.02},
    )
    driver = DriverProfile(0.73, 0.64, 0.78, 0.72)
    tire_data = {
        SOFT: TireData(SOFT, 1.00, 0.022, 2, (2, 8), 25.0),
        MEDIUM: TireData(MEDIUM, 0.96, 0.016, 3, (3, 16), 25.0),
        HARD: TireData(HARD, 0.92, 0.012, 4, (4, 28), 25.0),
        INTER: TireData(INTER, 0.88, 0.018, 2, (2, 12), 25.0),
        WET: TireData(WET, 0.82, 0.025, 1, (1, 10), 25.0),
    }
    race = RaceState(
        current_lap=current_lap,
        total_laps=total_laps,
        weather=weather,
        track_temperature=34.5 if weather == DRY else 22.0,
        **race_extra,
    )
    competitors = [
        Competitor("HAM", SOFT, 16, 1.8),
        Competitor("VER", MEDIUM, 10, 3.2),
    ]
    return telemetry, car_status, driver, tire_data, race, competitors


def _race(data, **changes):
    telemetry, car, driver, tires, race, competitors = data
    return telemetry, car, driver, tires, dataclasses.replace(race, **changes), competitors


def _plan_kwargs(data):
    telemetry, car, driver, tires, race, _ = data
    return dict(telemetry=telemetry, car_status=car, driver_profile=driver, tire_data=tires, race_state=race)


def _assert_invariants(result, current_lap, total_laps):
    remaining = total_laps - current_lap + 1
    assert result.remaining_laps == remaining
    assert 1 <= len(result.candidates) <= MAX_CANDIDATES
    times = [c.projected_race_time for c in result.candidates]
    assert times == sorted(times)
    assert result.candidates[0].delta_to_best_s == 0.0
    assert len({c.strategy_id for c in result.candidates}) == len(result.candidates)
    for option in result.candidates:
        stints = option.stint_breakdown
        assert 1 <= len(stints) <= MAX_STOPS + 1
        assert len(stints) == len(option.tire_compounds) == len(option.pit_laps) + 1
        assert [s["tire_compound"] for s in stints] == option.tire_compounds
        assert sum(s["laps"] for s in stints) == remaining
        assert stints[0]["start_lap"] == current_lap
        assert stints[-1]["end_lap"] == total_laps
        for previous, following in zip(stints, stints[1:]):
            assert following["start_lap"] == previous["end_lap"] + 1
        assert option.pit_laps == [s["end_lap"] for s in stints[:-1]]
        for index, stint in enumerate(stints):
            assert stint["laps"] >= 1
            assert stint["end_lap"] - stint["start_lap"] + 1 == stint["laps"]
            assert stint["fitted_at_stop"] is (index > 0)
            assert stint["tire_age_end"] == stint["tire_age_start"] + stint["laps"]
            assert 0 <= stint["laps_beyond_peak_window"] <= stint["laps"]
            assert 0 <= stint["laps_at_performance_floor"] <= stint["laps"]
            assert math.isfinite(stint["total_time"]) and stint["total_time"] > 0
        assert math.isfinite(option.projected_race_time)
        assert option.driving_time_s == pytest.approx(sum(s["total_time"] for s in stints))
        assert option.projected_race_time == pytest.approx(option.driving_time_s + option.pit_time_loss_s)
        assert option.delta_to_best_s == pytest.approx(option.projected_race_time - times[0])
        assert option.two_compound_rule in {RULE_SATISFIED, RULE_WAIVED, RULE_UNVERIFIED, RULE_VIOLATED}


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _old_fixed_ratio_split(total_laps, compounds):
    """The pre-hardening allocation (fixed durability weights), kept here as a baseline."""

    weights = {SOFT: 0.8, MEDIUM: 1.0, HARD: 1.2, INTER: 1.0, WET: 1.0}
    raw = [total_laps * weights[c] / sum(weights[x] for x in compounds) for c in compounds]
    lengths = [max(1, math.floor(value)) for value in raw]
    while sum(lengths) < total_laps:
        index = max(range(len(lengths)), key=lambda i: raw[i] - lengths[i])
        lengths[index] += 1
    while sum(lengths) > total_laps:
        index = max((i for i in range(len(lengths)) if lengths[i] > 1), key=lambda i: lengths[i] - raw[i])
        lengths[index] -= 1
    return lengths


# ---------------------------------------------------------------------------
# Kept scenarios (updated to the new result type)
# ---------------------------------------------------------------------------


def test_dry_weather_scenario():
    data = create_test_data(DRY)
    result = generate_strategy(*data)
    _assert_invariants(result, 8, 52)
    assert all(c in DRY_COMPOUNDS for option in result.candidates for c in option.tire_compounds)
    # The fastest plan of every stop count is always returned, including "no further stop".
    assert {option.stop_count for option in result.candidates} == {0, 1, 2, 3}


def test_wet_weather_scenario():
    data = create_test_data(WeatherCondition.WET)
    result = generate_strategy(*data)
    _assert_invariants(result, 8, 52)
    assert all(c in WET_WEATHER_COMPOUNDS for option in result.candidates for c in option.tire_compounds)
    assert all(option.two_compound_rule == RULE_WAIVED for option in result.candidates)


def test_damage_scenario_increases_projected_time():
    baseline = create_test_data(DRY)
    damaged_car = dataclasses.replace(
        baseline[1],
        damage={"front_wing": 0.45, "floor": 0.25, "diffuser": 0.15},
        engine_wear=0.75,
        brake_wear=0.85,
    )
    damaged = (baseline[0], damaged_car, *baseline[2:])
    base_result = generate_strategy(*baseline)
    damaged_result = generate_strategy(*damaged)
    assert damaged_result.damage_multiplier > base_result.damage_multiplier > 1.0
    assert damaged_result.best.projected_race_time > base_result.best.projected_race_time


# ---------------------------------------------------------------------------
# Late race and zero-stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weather", list(WeatherCondition))
@pytest.mark.parametrize("laps_left", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("tyre_state", ["unknown", "supplied"])
def test_last_laps_never_abort_and_offer_no_stop(weather, laps_left, tyre_state):
    total = 50
    current = total - laps_left + 1
    extra = {}
    if tyre_state == "supplied":
        fitted = SOFT if weather == DRY else INTER
        extra = dict(current_compound=fitted, current_tire_age=6, used_compounds=[MEDIUM])
    data = create_test_data(weather, current_lap=current, total_laps=total, **extra)

    result = generate_strategy(*data)

    _assert_invariants(result, current, total)
    assert any(option.stop_count == 0 for option in result.candidates)
    assert all(option.stop_count <= laps_left - 1 for option in result.candidates)


def test_final_lap_returns_only_no_stop_plans():
    result = generate_strategy(*create_test_data(DRY, current_lap=50, total_laps=50))
    assert result.candidates
    assert all(option.stop_count == 0 for option in result.candidates)
    assert result.sequences_skipped_too_few_laps > 0


def test_staying_out_wins_with_few_laps_left_on_healthy_tyres():
    data = create_test_data(
        DRY, current_lap=46, total_laps=50, current_compound=MEDIUM, current_tire_age=10, used_compounds=[SOFT]
    )
    result = generate_strategy(*data)
    assert result.best.stop_count == 0
    assert result.best.strategy_id == "0-stop:medium"
    assert result.best.pit_time_loss_s == 0.0
    assert result.best.two_compound_rule == RULE_SATISFIED


def test_single_compound_plan_excluded_when_history_is_known():
    # At the start of a dry race the history is known to be empty, so one-compound plans break the rule.
    result = generate_strategy(*create_test_data(DRY, current_lap=1, total_laps=52))
    assert result.candidates
    assert all(option.two_compound_rule == RULE_SATISFIED for option in result.candidates)
    assert all(len(set(option.tire_compounds)) >= 2 for option in result.candidates)
    assert result.sequences_excluded_two_compound_rule > 0


def test_unverifiable_rule_is_flagged_not_hidden():
    result = generate_strategy(*create_test_data(DRY, current_lap=40, total_laps=52))
    single = [o for o in result.candidates if len(set(o.tire_compounds)) == 1]
    assert single, "single-compound plans stay available when the earlier tyres are unknown"
    assert all(o.two_compound_rule == RULE_UNVERIFIED for o in single)
    assert all(any("not verified" in note for note in o.notes) for o in single)


def test_rule_that_cannot_be_met_still_returns_flagged_candidate():
    data = create_test_data(DRY, current_lap=50, total_laps=50, current_compound=SOFT, current_tire_age=30, used_compounds=[])
    result = generate_strategy(*data)
    assert [o.strategy_id for o in result.candidates] == ["0-stop:soft"]
    assert result.best.two_compound_rule == RULE_VIOLATED
    assert any("VIOLATED" in note for note in result.best.notes)
    assert any("violate" in text for text in result.assumptions)
    # Medium/hard exist in tire_data, so the stated cause is the lap count, not the tyre data.
    assert any("no stop fits in the remaining 1 lap(s)" in text for text in result.assumptions)
    assert any("no stop fits in the remaining 1 lap(s)" in note for note in result.best.notes)
    assert not any("only one dry compound" in text for text in result.assumptions)


def test_rule_fallback_names_single_dry_compound_in_tire_data_as_the_cause():
    telemetry, car, driver, tires, _, _ = create_test_data(DRY)
    race = RaceState(1, 50, DRY, 30.0)
    result = generate_strategy(telemetry, car, driver, {SOFT: tires[SOFT]}, race)
    assert result.candidates and all(o.two_compound_rule == RULE_VIOLATED for o in result.candidates)
    assert any("only one dry compound (soft)" in text for text in result.assumptions)
    assert not any("no stop fits" in text for text in result.assumptions)
    for option in result.candidates:
        assert any(note.startswith("Shown although it violates") and "only one dry compound (soft)" in note
                   for note in option.notes)

    # A known earlier medium stint supplies the second compound: every soft plan is then compliant.
    with_history = generate_strategy(telemetry, car, driver, {SOFT: tires[SOFT]}, RaceState(20, 50, DRY, 30.0, used_compounds=[MEDIUM]))
    assert all(o.two_compound_rule == RULE_SATISFIED for o in with_history.candidates)


def test_violated_note_of_a_user_plan_does_not_claim_a_fallback():
    data = create_test_data(DRY, current_lap=40, total_laps=52, current_compound=SOFT, current_tire_age=3, used_compounds=[])
    option = evaluate_plan([SOFT, SOFT], [6, 7], **_plan_kwargs(data))
    assert option.two_compound_rule == RULE_VIOLATED
    assert not any("Shown although" in note or "no compliant plan" in note for note in option.notes)


def test_rule_is_waived_by_intermediate_or_wet_tyres_in_the_history():
    data = create_test_data(DRY, current_lap=30, total_laps=52, current_compound=SOFT, current_tire_age=5,
                            used_compounds=[INTER])
    assert evaluate_plan([SOFT], [23], **_plan_kwargs(data)).two_compound_rule == RULE_WAIVED
    result = generate_strategy(*data)
    assert all(option.two_compound_rule == RULE_WAIVED for option in result.candidates)
    assert result.sequences_excluded_two_compound_rule == 0
    assert "0-stop:soft" in {option.strategy_id for option in result.candidates}


# ---------------------------------------------------------------------------
# Current tyre state
# ---------------------------------------------------------------------------


def test_current_tyre_continues_from_its_age_without_pit_cost():
    data = create_test_data(DRY, current_lap=30, total_laps=52, current_compound=SOFT, current_tire_age=12, used_compounds=[])
    result = generate_strategy(*data)
    tires = data[3]
    _assert_invariants(result, 30, 52)
    assert result.tire_state == "supplied"
    for option in result.candidates:
        first = option.stint_breakdown[0]
        assert option.tire_compounds[0] == SOFT
        assert first["tire_age_start"] == 12 and first["fitted_at_stop"] is False
        assert option.pit_time_loss_s == pytest.approx(sum(tires[c].pit_stop_delta for c in option.tire_compounds[1:]))
        assert any("continues on the fitted soft tyres from age 12" in note for note in option.notes)


def test_older_current_tyre_is_slower_and_switching_now_costs_a_stop():
    young = create_test_data(DRY, current_lap=40, total_laps=52, current_compound=SOFT, current_tire_age=4, used_compounds=[MEDIUM])
    old = _race(young, current_tire_age=20)
    remaining = 13
    young_run = evaluate_plan([SOFT], [remaining], **_plan_kwargs(young))
    old_run = evaluate_plan([SOFT], [remaining], **_plan_kwargs(old))
    assert old_run.projected_race_time > young_run.projected_race_time
    assert old_run.stint_breakdown[0]["start_performance"] < young_run.stint_breakdown[0]["start_performance"]

    switch_now = evaluate_plan([SOFT, HARD], [1, remaining - 1], **_plan_kwargs(old))
    assert switch_now.pit_laps == [40]
    assert switch_now.pit_time_loss_s == pytest.approx(young[3][HARD].pit_stop_delta)
    with pytest.raises(ValueError, match="fitted 'soft'"):
        evaluate_plan([HARD], [remaining], **_plan_kwargs(old))


def test_without_tyre_state_every_candidate_states_the_fresh_set_assumption():
    result = generate_strategy(*create_test_data(DRY, current_lap=20, total_laps=52))
    assert result.tire_state == "assumed_fresh"
    for option in result.candidates:
        assert option.stint_breakdown[0]["tire_age_start"] == 0
        assert any(note.startswith("Assumes a fresh set of") and "lap 20" in note for note in option.notes)


# ---------------------------------------------------------------------------
# Pit-lap optimisation
# ---------------------------------------------------------------------------


def _one_stop_pit_lap(medium_degradation, hard_degradation=0.012):
    telemetry, car, driver, tires, race, rivals = create_test_data(
        DRY, current_lap=1, total_laps=50, current_compound=MEDIUM, current_tire_age=0, used_compounds=[]
    )
    tires = {
        MEDIUM: dataclasses.replace(tires[MEDIUM], degradation_rate=medium_degradation),
        HARD: dataclasses.replace(tires[HARD], degradation_rate=hard_degradation),
    }
    result = generate_strategy(telemetry, car, driver, tires, race, rivals)
    (option,) = [o for o in result.candidates if o.strategy_id == "1-stop:medium/hard"]
    return option.pit_laps[0]


def test_pit_lap_moves_when_degradation_changes():
    baseline = _one_stop_pit_lap(0.010)
    assert _one_stop_pit_lap(0.030) < baseline  # medium wears faster -> pit earlier
    assert _one_stop_pit_lap(0.010, hard_degradation=0.030) > baseline  # hard wears faster -> shorter hard stint


@pytest.mark.parametrize(
    "current_lap,total_laps,extra",
    [
        (1, 11, {}),
        (3, 12, dict(current_compound=SOFT, current_tire_age=9, used_compounds=None)),
        (5, 14, dict(current_compound=MEDIUM, current_tire_age=2, used_compounds=[SOFT])),
    ],
)
def test_dynamic_programme_matches_brute_force(current_lap, total_laps, extra):
    data = create_test_data(DRY, current_lap=current_lap, total_laps=total_laps, **extra)
    result = generate_strategy(*data)
    remaining = total_laps - current_lap + 1
    for option in result.candidates:
        brute = min(
            evaluate_plan(option.tire_compounds, list(lengths), **_plan_kwargs(data)).projected_race_time
            for lengths in _compositions(remaining, len(option.tire_compounds))
        )
        assert option.projected_race_time <= brute + 1e-6


def test_optimiser_is_never_worse_than_old_fixed_ratio_split():
    improvements = []
    scenarios = [
        create_test_data(DRY),
        create_test_data(DRY, current_lap=1, total_laps=70),
        create_test_data(WeatherCondition.WET, current_lap=5, total_laps=60),
        create_test_data(DRY, current_lap=15, total_laps=57, current_compound=MEDIUM, current_tire_age=14, used_compounds=[]),
    ]
    for data in scenarios:
        race = data[4]
        remaining = race.total_laps - race.current_lap + 1
        for option in generate_strategy(*data).candidates:
            fixed = evaluate_plan(
                option.tire_compounds, _old_fixed_ratio_split(remaining, option.tire_compounds), **_plan_kwargs(data)
            )
            assert option.projected_race_time <= fixed.projected_race_time + 1e-6
            improvements.append(fixed.projected_race_time - option.projected_race_time)
    assert max(improvements) > 1.0  # optimisation actually changes pit laps


def _distinct_delta_tyres():
    """Every compound has its own pit_stop_delta, so charging the wrong compound changes the answer."""

    return {
        SOFT: TireData(SOFT, 1.00, 0.020, 2, (2, 8), 20.0),
        MEDIUM: TireData(MEDIUM, 0.97, 0.012, 3, (3, 16), 25.0),
        HARD: TireData(HARD, 0.95, 0.008, 4, (4, 28), 30.0),
        INTER: TireData(INTER, 0.88, 0.018, 2, (2, 12), 27.0),
        WET: TireData(WET, 0.82, 0.025, 1, (1, 10), 33.0),
    }


@pytest.mark.parametrize(
    "weather,extra",
    [
        (DRY, {}),
        (DRY, dict(current_compound=HARD, current_tire_age=12, used_compounds=[])),
        (WeatherCondition.WET, {}),
        (WeatherCondition.INTERMEDIATE, dict(current_compound=INTER, current_tire_age=4)),
    ],
)
def test_each_stop_costs_the_delta_of_the_compound_fitted_there(weather, extra):
    telemetry, car, driver, _, race, _ = create_test_data(weather, current_lap=20, total_laps=52, **extra)
    tires = _distinct_delta_tyres()
    result = generate_strategy(telemetry, car, driver, tires, race)
    assert any(option.stop_count >= 1 for option in result.candidates)
    for option in result.candidates:
        expected = sum(tires[c].pit_stop_delta for c in option.tire_compounds[1:])
        assert option.pit_time_loss_s == pytest.approx(expected)

    kwargs = dict(telemetry=telemetry, car_status=car, driver_profile=driver, tire_data=tires,
                  race_state=RaceState(20, 52, DRY, 30.0))
    assert evaluate_plan([SOFT, HARD], [10, 23], **kwargs).pit_time_loss_s == pytest.approx(30.0)
    assert evaluate_plan([HARD, SOFT], [10, 23], **kwargs).pit_time_loss_s == pytest.approx(20.0)
    assert evaluate_plan([MEDIUM, SOFT, HARD], [5, 5, 23], **kwargs).pit_time_loss_s == pytest.approx(50.0)


def test_fresh_start_puts_the_costliest_stop_compound_first():
    # Without tyre state the first stint is free, so {medium, soft, soft} should start on the medium
    # (25 s) and pay two 20 s soft stops, not start on a soft and pay the 25 s medium stop.
    tires = _distinct_delta_tyres()
    telemetry, car, driver = {"lap_times": [80.0]}, CarStatus(0.2, 0.2), DriverProfile(0.7, 0.5, 0.8, 0.6)
    race = RaceState(20, 52, DRY, 30.0)
    result = generate_strategy(telemetry, car, driver, {c: tires[c] for c in (SOFT, MEDIUM, HARD)}, race)
    by_id = {option.strategy_id: option for option in result.candidates}
    assert "2-stop:medium/soft/soft" in by_id
    assert by_id["2-stop:medium/soft/soft"].pit_time_loss_s == pytest.approx(40.0)
    assert not any(sid.startswith("2-stop:soft/") and "medium" in sid for sid in by_id)


def _brute_force_by_sequence(data):
    """{ordered compound sequence: (best time over every stint split, rule status)} for EVERY ordering."""

    telemetry, car, driver, tires, race, _ = data
    remaining = race.total_laps - race.current_lap + 1
    usable = DRY_COMPOUNDS if race.weather == DRY else WET_WEATHER_COMPOUNDS
    allowed = [c for c in TireCompound if c in usable and c in tires]
    firsts = [race.current_compound] if race.current_compound is not None else allowed
    table = {}
    for stops in range(min(MAX_STOPS, remaining - 1) + 1):
        for first in firsts:
            for rest in product(allowed, repeat=stops):
                sequence = (first, *rest)
                options = [evaluate_plan(list(sequence), list(lengths), **_plan_kwargs(data))
                           for lengths in _compositions(remaining, stops + 1)]
                table[sequence] = (min(o.projected_race_time for o in options), options[0].two_compound_rule)
    return table


def test_plans_match_global_brute_force_over_all_orderings():
    rng = random.Random(3)
    checked = 0
    while checked < 12:
        data = _random_state(rng)
        telemetry, car, driver, tires, race, _ = data
        tires = {c: dataclasses.replace(t, pit_stop_delta=round(rng.uniform(5.0, 40.0), 2)) for c, t in tires.items()}
        race = dataclasses.replace(race, total_laps=race.current_lap + rng.randint(1, 5))
        data = (telemetry, car, driver, tires, race, [])
        try:
            result = generate_strategy(*data)
        except ValueError:
            continue
        table = _brute_force_by_sequence(data)
        compliant = [time for time, rule in table.values() if rule != RULE_VIOLATED]
        target = min(compliant or [time for time, _ in table.values()])
        assert result.best.projected_race_time == pytest.approx(target, abs=1e-6)
        # Every candidate is the best ordering of its compounds (only the first stint is fixed by tyre state).
        for option in result.candidates:
            first, rest = option.tire_compounds[0], option.tire_compounds[1:]
            if race.current_compound is None:
                orderings = set(permutations(option.tire_compounds))
            else:
                orderings = {(first, *perm) for perm in permutations(rest)}
            best_ordering = min(table[sequence][0] for sequence in orderings)
            assert option.projected_race_time == pytest.approx(best_ordering, abs=1e-6), option.strategy_id
        checked += 1


def test_hot_track_factor_changes_soft_and_hard_only():
    data = create_test_data(DRY, current_lap=40, total_laps=52)
    cool = _plan_kwargs(_race(data, track_temperature=35.0))
    hot = _plan_kwargs(_race(data, track_temperature=35.5))
    ratios = {}
    for compound in (SOFT, MEDIUM, HARD):
        cool_run = evaluate_plan([compound], [13], **cool)
        hot_run = evaluate_plan([compound], [13], **hot)
        ratios[compound] = hot_run.stint_breakdown[0]["start_performance"] / cool_run.stint_breakdown[0]["start_performance"]
    assert ratios[SOFT] == pytest.approx(0.97)
    assert ratios[MEDIUM] == pytest.approx(1.0)
    assert ratios[HARD] == pytest.approx(1.02)
    assert evaluate_plan([SOFT], [13], **hot).projected_race_time > evaluate_plan([SOFT], [13], **cool).projected_race_time
    assert evaluate_plan([HARD], [13], **hot).projected_race_time < evaluate_plan([HARD], [13], **cool).projected_race_time


def test_laps_beyond_peak_window_counts_tyre_laps_after_the_window_end():
    # Soft peak window ends at tyre lap 8.
    fresh = create_test_data(DRY, current_lap=33, total_laps=52)
    assert evaluate_plan([SOFT], [20], **_plan_kwargs(fresh)).stint_breakdown[0]["laps_beyond_peak_window"] == 12
    for age, laps, expected in [(10, 5, 5), (3, 3, 0), (5, 6, 3)]:
        data = create_test_data(DRY, current_lap=53 - laps, total_laps=52, current_compound=SOFT, current_tire_age=age)
        stint = evaluate_plan([SOFT], [laps], **_plan_kwargs(data)).stint_breakdown[0]
        assert stint["laps_beyond_peak_window"] == expected, (age, laps)


def test_laps_at_performance_floor_are_counted_and_flagged():
    telemetry, car, driver, tires, race, _ = create_test_data(DRY, current_lap=41, total_laps=52)
    cliff = {**tires, SOFT: TireData(SOFT, 1.0, 0.5, 0, (1, 1), 25.0)}  # 1.0, 0.5, then clamped at 0.2
    kwargs = dict(telemetry=telemetry, car_status=car, driver_profile=driver, tire_data=cliff, race_state=race)
    option = evaluate_plan([SOFT], [12], **kwargs)
    assert option.stint_breakdown[0]["laps_at_performance_floor"] == 10
    assert option.stint_breakdown[0]["end_performance"] == pytest.approx(0.2)
    assert any("performance floor" in note and note.startswith("10 projected lap(s)") for note in option.notes)
    healthy = evaluate_plan([MEDIUM], [12], **kwargs)
    assert healthy.stint_breakdown[0]["laps_at_performance_floor"] == 0
    assert not any("performance floor" in note for note in healthy.notes)


def test_typical_seventy_lap_request_is_fast():
    data = create_test_data(DRY, current_lap=1, total_laps=70)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        generate_strategy(*data)
        timings.append(time.perf_counter() - start)
    assert min(timings) < 0.3


# ---------------------------------------------------------------------------
# Random valid states, determinism, thread safety
# ---------------------------------------------------------------------------


def _random_state(rng):
    total = rng.randint(1, 90)
    current = rng.randint(1, total)
    weather = rng.choice(list(WeatherCondition))
    compounds = [c for c in TireCompound if rng.random() < 0.75] or [rng.choice(list(TireCompound))]
    tires = {}
    for compound in compounds:
        start = rng.randint(1, 15)
        tires[compound] = TireData(
            compound,
            round(rng.uniform(0.8, 1.05), 3),
            round(rng.uniform(0.0, 0.05), 4),
            rng.randint(0, 5),
            (start, start + rng.randint(0, 25)),
            round(rng.uniform(15.0, 35.0), 2),
        )
    extra = {}
    if rng.random() < 0.5:
        extra["current_compound"] = rng.choice(compounds)
        extra["current_tire_age"] = rng.randint(0, 40)
    if rng.random() < 0.5:
        extra["used_compounds"] = rng.sample(list(TireCompound), rng.randint(0, 2))
    telemetry = {"lap_times": [round(rng.uniform(70, 110), 3) for _ in range(rng.randint(1, 6))]}
    car = CarStatus(rng.random(), rng.random(), {"front_wing": rng.random()})
    driver = DriverProfile(rng.random(), rng.random(), rng.random(), rng.random())
    race = RaceState(current, total, weather, rng.uniform(10, 50), **extra)
    return telemetry, car, driver, tires, race, []


def test_invariants_hold_over_many_random_valid_states():
    rng = random.Random(20260924)
    usable_weather = {DRY: DRY_COMPOUNDS, WeatherCondition.WET: WET_WEATHER_COMPOUNDS,
                      WeatherCondition.INTERMEDIATE: WET_WEATHER_COMPOUNDS}
    generated = 0
    for _ in range(250):
        data = _random_state(rng)
        tires, race = data[3], data[4]
        if race.current_compound is None and not usable_weather[race.weather] & set(tires):
            with pytest.raises(ValueError, match="No strategy could be generated"):
                generate_strategy(*data)
            continue
        result = generate_strategy(*data)
        generated += 1
        _assert_invariants(result, race.current_lap, race.total_laps)
        violated = [o for o in result.candidates if o.two_compound_rule == RULE_VIOLATED]
        assert not violated or len(violated) == len(result.candidates)
        has_no_stop = any(o.stop_count == 0 for o in result.candidates)
        assert has_no_stop or result.sequences_excluded_two_compound_rule > 0
        if race.current_compound is not None:
            assert all(o.tire_compounds[0] == race.current_compound for o in result.candidates)
    assert generated > 150


def test_random_small_states_are_optimal_against_brute_force():
    rng = random.Random(7)
    checked = 0
    while checked < 25:
        data = _random_state(rng)
        race = data[4]
        race = dataclasses.replace(race, total_laps=race.current_lap + rng.randint(0, 7))
        data = (*data[:4], race, [])
        try:
            result = generate_strategy(*data)
        except ValueError:
            continue
        remaining = race.total_laps - race.current_lap + 1
        for option in result.candidates:
            brute = min(
                evaluate_plan(option.tire_compounds, list(lengths), **_plan_kwargs(data)).projected_race_time
                for lengths in _compositions(remaining, len(option.tire_compounds))
            )
            assert option.projected_race_time <= brute + 1e-6
        checked += 1


def test_generation_is_deterministic_and_thread_safe():
    states = [create_test_data(w, current_lap=lap) for w, lap in product(WeatherCondition, (1, 20, 45))]
    sequential = [strategy_result_to_dict(generate_strategy(*s)) for s in states]
    assert sequential == [strategy_result_to_dict(generate_strategy(*s)) for s in states]
    with ThreadPoolExecutor(max_workers=6) as pool:
        concurrent = list(pool.map(lambda s: strategy_result_to_dict(generate_strategy(*s)), states * 3))
    assert concurrent == sequential * 3


# ---------------------------------------------------------------------------
# Honesty: unused inputs, competitor signals
# ---------------------------------------------------------------------------


def test_unused_inputs_are_optional_reported_and_do_not_change_numbers():
    plain = create_test_data(DRY, current_lap=20)
    telemetry, car, driver, tires, race, _ = plain
    decorated = (
        telemetry,
        dataclasses.replace(car, fuel_load=5.0, brake_temp=900.0, ers_availability=0.1),
        dataclasses.replace(driver, overtaking_style="kamikaze"),
        tires,
        dataclasses.replace(race, safety_car_probability=0.95, yellow_flag_risk=1.0, track_evolution=1.0,
                            weather_forecast=[{"lap": 30, "weather": "wet"}]),
        [Competitor("HAM", SOFT, 16, 1.8, current_position=2, gap_ahead=1.0, pit_stops_completed=1)],
    )
    plain_result = generate_strategy(*plain[:5])
    decorated_result = generate_strategy(*decorated)
    assert [(o.strategy_id, o.pit_laps, o.projected_race_time) for o in decorated_result.candidates] == [
        (o.strategy_id, o.pit_laps, o.projected_race_time) for o in plain_result.candidates
    ]
    for name in (
        "car_status.fuel_load",
        "car_status.brake_temp",
        "car_status.ers_availability",
        "driver_profile.overtaking_style",
        "race_state.safety_car_probability",
        "race_state.yellow_flag_risk",
        "race_state.track_evolution",
        "race_state.weather_forecast",
        "competition[].current_position",
        "competition[].gap_ahead",
        "competition[].pit_stops_completed",
    ):
        assert name in decorated_result.not_modelled_inputs
    assert "car_status.fuel_load" not in plain_result.not_modelled_inputs


def test_competitor_signals_are_response_level_and_gap_based():
    data = create_test_data(DRY, current_lap=25, own_gap_to_leader=20.0)
    rivals = [
        Competitor("AHEAD_OLD", MEDIUM, 25, 18.5),  # 1.5 s ahead, past medium peak end (16)
        Competitor("AHEAD_FRESH", MEDIUM, 5, 18.0),  # 2.0 s ahead, fresh tyres
        Competitor("BEHIND", HARD, 3, 22.5),  # 2.5 s behind
        Competitor("EDGE_AHEAD", SOFT, 20, 17.0),  # exactly 3.0 s ahead, past soft peak end (8): inside
        Competitor("EDGE_BEHIND", HARD, 3, 23.0),  # exactly 3.0 s behind: inside the window
        Competitor("OUTSIDE", HARD, 3, 23.1),  # 3.1 s behind: outside the window
        Competitor("BACKMARKER", HARD, 30, 95.0, gap_ahead=1.0),  # far behind; gap_ahead is not our gap
    ]
    result = generate_strategy(*data[:5], rivals)
    signals = {(s.driver_id, s.signal) for s in result.competitor_signals}
    assert signals == {
        ("AHEAD_OLD", "undercut_target"),
        ("EDGE_AHEAD", "undercut_target"),
        ("BEHIND", "undercut_threat"),
        ("EDGE_BEHIND", "undercut_threat"),
    }
    assert {s.driver_id: s.gap_s for s in result.competitor_signals}["EDGE_BEHIND"] == pytest.approx(3.0)
    assert all(not hasattr(o, "undercut_opportunities") for o in result.candidates)

    no_gap = generate_strategy(*create_test_data(DRY, current_lap=25)[:5], rivals)
    assert no_gap.competitor_signals == []
    assert "own_gap_to_leader" in no_gap.competitor_signals_note


def test_competitor_level_on_gap_is_a_threat_and_unassessable_rivals_are_named():
    telemetry, car, driver, tires, race, _ = create_test_data(DRY, current_lap=25, own_gap_to_leader=20.0)
    dry_tires = {c: t for c, t in tires.items() if c in DRY_COMPOUNDS}
    rivals = [
        Competitor("LEVEL", MEDIUM, 30, 20.0),
        Competitor("AHEAD_INTER", INTER, 40, 19.0),  # no intermediate entry in tire_data
    ]
    result = generate_strategy(telemetry, car, driver, dry_tires, race, rivals)
    assert [(s.driver_id, s.signal, s.gap_s) for s in result.competitor_signals] == [("LEVEL", "undercut_threat", 0.0)]
    assert "level with you" in result.competitor_signals[0].explanation
    assert "AHEAD_INTER (intermediate)" in result.competitor_signals_note
    assert "Not assessed" in result.competitor_signals_note

    complete = generate_strategy(telemetry, car, driver, tires, race, rivals)
    assert ("AHEAD_INTER", "undercut_target") in {(s.driver_id, s.signal) for s in complete.competitor_signals}
    assert "Not assessed" not in complete.competitor_signals_note


def test_telemetry_overrides_driver_profile_and_the_ignored_value_is_reported():
    telemetry, car, _, tires, race, _ = create_test_data(DRY, current_lap=20)
    lap_times = {"lap_times": telemetry["lap_times"]}

    def run(measured, **profile):
        driver = DriverProfile(tire_management=0.5, risk_tolerance=0.5, **profile)
        return generate_strategy({**lap_times, **measured}, car, driver, tires, race)

    # braking below 0.7 adds (0.7 - braking) x 30 %; telemetry wins over the profile.
    assert run({"braking_consistency": 0.5}, braking_consistency=0.9, throttle_aggressiveness=0.5).driver_multiplier == pytest.approx(1.06)
    assert run({"braking_consistency": 0.9}, braking_consistency=0.5, throttle_aggressiveness=0.5).driver_multiplier == pytest.approx(1.0)
    # throttle above 0.8 with tire_management below 0.6 adds 10 %.
    assert run({"throttle_aggressiveness": 0.9}, braking_consistency=0.9, throttle_aggressiveness=0.1).driver_multiplier == pytest.approx(1.10)
    assert run({"throttle_aggressiveness": 0.1}, braking_consistency=0.9, throttle_aggressiveness=0.9).driver_multiplier == pytest.approx(1.0)

    both = run({"braking_consistency": 0.9, "throttle_aggressiveness": 0.5}, braking_consistency=0.1, throttle_aggressiveness=0.95)
    assert "driver_profile.braking_consistency (overridden by telemetry.braking_consistency)" in both.not_modelled_inputs
    assert "driver_profile.throttle_aggressiveness (overridden by telemetry.throttle_aggressiveness)" in both.not_modelled_inputs

    # The profile values are optional when telemetry supplies them, and then nothing is reported.
    measured_only = run({"braking_consistency": 0.9, "throttle_aggressiveness": 0.5})
    assert measured_only.driver_multiplier == both.driver_multiplier
    assert not any(name.startswith("driver_profile.") for name in measured_only.not_modelled_inputs)
    profile_only = run({}, braking_consistency=0.9, throttle_aggressiveness=0.5)
    assert profile_only.driver_multiplier == both.driver_multiplier
    assert not any("overridden" in name for name in profile_only.not_modelled_inputs)
    with pytest.raises(ValueError, match="braking_consistency is required"):
        run({}, throttle_aggressiveness=0.5)
    with pytest.raises(ValueError, match="throttle_aggressiveness is required"):
        run({"braking_consistency": 0.9})


def test_base_lap_time_is_the_mean_of_supplied_lap_times():
    result = generate_strategy(*create_test_data(DRY))
    assert result.base_lap_time == pytest.approx(sum([80.5, 80.2, 80.8, 80.1, 80.3]) / 5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _mutate(part, **changes):
    index = {"car": 1, "driver": 2, "race": 4}[part]

    def apply(data):
        data = list(data)
        data[index] = dataclasses.replace(data[index], **changes)
        return data

    return apply


def _telemetry(**changes):
    def apply(data):
        data = list(data)
        data[0] = {**data[0], **changes}
        return data

    return apply


def _without_lap_times(data):
    data = list(data)
    data[0] = {k: v for k, v in data[0].items() if k != "lap_times"}
    return data


def _tyre(compound=SOFT, key=None, **changes):
    def apply(data):
        data = list(data)
        tires = dict(data[3])
        tires[key or compound] = dataclasses.replace(tires[compound], **changes)
        data[3] = tires
        return data

    return apply


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(_without_lap_times, id="lap-times-missing"),
        pytest.param(_telemetry(lap_times=[]), id="lap-times-empty"),
        pytest.param(_telemetry(lap_times=[80.0, NAN]), id="lap-times-nan"),
        pytest.param(_telemetry(lap_times=[80.0, INF]), id="lap-times-inf"),
        pytest.param(_telemetry(lap_times=[-80.0]), id="lap-times-negative"),
        pytest.param(_telemetry(lap_times=[5.0]), id="lap-times-implausibly-fast"),
        pytest.param(_telemetry(lap_times=80.0), id="lap-times-scalar"),
        pytest.param(_telemetry(lap_times=[True]), id="lap-times-bool"),
        pytest.param(_telemetry(braking_consistency=-10), id="telemetry-braking-negative"),
        pytest.param(_telemetry(speed_trace=[1, 2, 3]), id="telemetry-unknown-key"),
        pytest.param(_telemetry(lap_times=[10**400]), id="lap-times-huge-int-overflows-float"),
        pytest.param(_telemetry(sector_times=[28.0, 0.0]), id="sector-times-zero"),
        pytest.param(_telemetry(sector_times={"S1": 28.0}), id="sector-times-bad-key"),
        pytest.param(_telemetry(sector_times={"51": 28.0}), id="sector-times-key-out-of-range"),
        pytest.param(_telemetry(sector_times=[]), id="sector-times-empty"),
        pytest.param(_telemetry(sector_times="fast"), id="sector-times-not-a-collection"),
        pytest.param(_tyre(degradation_rate=-0.1), id="degradation-negative"),
        pytest.param(_tyre(degradation_rate=NAN), id="degradation-nan"),
        pytest.param(_tyre(pit_stop_delta=-1.0), id="pit-delta-negative"),
        pytest.param(_tyre(pit_stop_delta=NAN), id="pit-delta-nan"),
        pytest.param(_tyre(pit_stop_delta=INF), id="pit-delta-inf"),
        pytest.param(_tyre(pit_stop_delta=10**400), id="pit-delta-huge-int"),
        pytest.param(_tyre(base_performance=0.0), id="base-performance-zero"),
        pytest.param(_tyre(base_performance=0.19), id="base-performance-below-floor"),
        pytest.param(_tyre(base_performance=0.49), id="base-performance-below-0.5"),
        pytest.param(_tyre(base_performance=2.5), id="base-performance-above-2"),
        pytest.param(_tyre(base_performance=NAN), id="base-performance-nan"),
        pytest.param(_tyre(peak_performance_window=(8, 2)), id="peak-window-reversed"),
        pytest.param(_tyre(peak_performance_window=(2,)), id="peak-window-not-a-pair"),
        pytest.param(_tyre(warm_up_laps=-1), id="warm-up-negative"),
        pytest.param(_tyre(warm_up_laps=1.5), id="warm-up-fractional"),
        pytest.param(_tyre(compound=SOFT, key=HARD), id="tyre-key-compound-mismatch"),
        pytest.param(_mutate("race", total_laps=201, current_lap=1), id="total-laps-above-200"),
        pytest.param(_mutate("race", current_lap=0), id="current-lap-zero"),
        pytest.param(_mutate("race", current_lap=53), id="current-lap-after-finish"),
        pytest.param(_mutate("race", current_lap=5.5), id="current-lap-fractional"),
        pytest.param(_mutate("race", weather="sunny"), id="weather-unknown"),
        pytest.param(_mutate("race", weather=None), id="weather-none"),
        pytest.param(_mutate("race", track_temperature=NAN), id="track-temp-nan"),
        pytest.param(_mutate("race", track_temperature=-300.0), id="track-temp-unphysical"),
        pytest.param(_mutate("race", safety_car_probability=7.0), id="sc-probability-above-1"),
        pytest.param(_mutate("race", yellow_flag_risk=NAN), id="yellow-risk-nan"),
        pytest.param(_mutate("race", current_compound=SOFT), id="current-compound-without-age"),
        pytest.param(_mutate("race", current_compound=SOFT, current_tire_age=-1), id="current-age-negative"),
        pytest.param(_mutate("race", own_gap_to_leader=NAN), id="own-gap-nan"),
        pytest.param(_mutate("car", engine_wear=1.5), id="engine-wear-above-1"),
        pytest.param(_mutate("car", fuel_load=-999.0), id="fuel-negative"),
        pytest.param(_mutate("car", damage={"rear_wing": 1.0}), id="damage-unknown-part"),
        pytest.param(_mutate("car", damage=[1, 2]), id="damage-not-a-mapping"),
        pytest.param(_mutate("driver", braking_consistency=-10.0), id="driver-braking-negative"),
        pytest.param(_mutate("driver", risk_tolerance=NAN), id="driver-risk-nan"),
    ],
)
def test_invalid_inputs_raise_value_error(mutation):
    data = mutation(create_test_data(DRY))
    with pytest.raises(ValueError):
        generate_strategy(*data)


def test_current_compound_must_have_tyre_data_and_competitors_must_be_valid():
    telemetry, car, driver, tires, race, rivals = create_test_data(DRY, current_compound=WET, current_tire_age=3)
    dry_only = {k: v for k, v in tires.items() if k in DRY_COMPOUNDS}
    with pytest.raises(ValueError, match="needs an entry in tire_data"):
        generate_strategy(telemetry, car, driver, dry_only, race, rivals)
    base = create_test_data(DRY)
    with pytest.raises(ValueError, match="Duplicate"):
        generate_strategy(*base[:5], [Competitor("HAM", SOFT, 1, 1.0), Competitor("HAM", SOFT, 2, 2.0)])
    with pytest.raises(ValueError):
        generate_strategy(*base[:5], [Competitor("HAM", SOFT, 1, NAN)])


def test_no_usable_compound_is_a_clear_error():
    telemetry, car, driver, tires, race, rivals = create_test_data(WeatherCondition.WET)
    dry_only = {k: v for k, v in tires.items() if k in DRY_COMPOUNDS}
    with pytest.raises(ValueError, match="No strategy could be generated"):
        generate_strategy(telemetry, car, driver, dry_only, race, rivals)


def test_evaluate_plan_rejects_bad_lengths():
    data = create_test_data(DRY, current_lap=40, total_laps=52)
    with pytest.raises(ValueError, match="sum to the remaining laps"):
        evaluate_plan([SOFT, HARD], [5, 5], **_plan_kwargs(data))
    with pytest.raises(ValueError):
        evaluate_plan([SOFT, HARD], [0, 13], **_plan_kwargs(data))
    with pytest.raises(ValueError):
        evaluate_plan([SOFT] * 5, [3, 3, 3, 2, 2], **_plan_kwargs(data))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _payload():
    return {
        "telemetry": {"lap_times": [80.5, 80.2, 80.8], "braking_consistency": 0.78},
        "car_status": {"engine_wear": 0.28, "brake_wear": 0.35, "damage": {"front_wing": 0.15}, "fuel_load": 95.5},
        "driver_profile": {
            "tire_management": 0.73,
            "risk_tolerance": 0.64,
            "braking_consistency": 0.78,
            "throttle_aggressiveness": 0.72,
            "overtaking_style": "calculated",
        },
        "tire_data": {
            "soft": {"compound": "soft", "base_performance": 1.0, "degradation_rate": 0.022, "warm_up_laps": 2,
                     "peak_performance_window": [2, 8], "pit_stop_delta": 25},
            "medium": {"base_performance": 0.96, "degradation_rate": 0.016, "warm_up_laps": 3,
                       "peak_performance_window": [3, 16], "pit_stop_delta": 25.0},
            "hard": {"base_performance": 0.92, "degradation_rate": 0.012, "warm_up_laps": 4,
                     "peak_performance_window": [4, 28], "pit_stop_delta": 25.0},
        },
        "race_state": {
            "current_lap": 20,
            "total_laps": 52,
            "weather": "dry",
            "track_temperature": 34.5,
            "safety_car_probability": 0.25,
            "current_compound": "medium",
            "current_tire_age": 12,
            "used_compounds": ["soft"],
            "own_gap_to_leader": 4.0,
        },
        "competition": [{"driver_id": "HAM", "tire_compound": "soft", "tire_age": 16, "gap_to_leader": 2.2}],
    }


def test_schema_round_trip_and_json_safe_response():
    request = StrategyRequest.model_validate(json.loads(json.dumps(_payload())))
    again = StrategyRequest.model_validate(json.loads(json.dumps(request.model_dump(mode="json"))))
    assert again == request

    inputs = request.to_engine_inputs()
    assert inputs["tire_data"][SOFT].compound is SOFT
    assert inputs["race_state"].current_compound is MEDIUM
    body = generate_strategy_response(request)
    assert body == strategy_result_to_dict(generate_strategy(**inputs))
    text = json.dumps(body, allow_nan=False)
    assert json.loads(text) == body

    assert body["heuristic"] is True and body["tire_state"] == "supplied"
    assert body["best_strategy_id"] == body["strategies"][0]["strategy_id"]
    assert [s["rank"] for s in body["strategies"]] == list(range(1, len(body["strategies"]) + 1))
    assert body["strategies"][0]["delta_to_best_s"] == 0.0
    assert "confidence_score" not in body["strategies"][0]
    assert "car_status.fuel_load" in body["not_modelled_inputs"]
    assert {s["driver_id"] for s in body["competitor_signals"]} == {"HAM"}
    for strategy in body["strategies"]:
        assert all(isinstance(c, str) for c in strategy["tire_compounds"])
        for stint in strategy["stint_breakdown"]:
            assert isinstance(stint["tire_compound"], str)
            assert not any(isinstance(value, list) for value in stint.values())


def test_serialised_response_stays_small_for_long_races():
    payload = _payload()
    payload["race_state"].update(current_lap=1, total_laps=200, current_tire_age=0, used_compounds=[])
    text = json.dumps(generate_strategy_response(StrategyRequest.model_validate(payload)), allow_nan=False)
    assert len(text) < 40_000


def test_from_context_ignores_unrelated_keys_and_rejects_missing_ones():
    context = {**_payload(), "question_history": ["unrelated"], "audio_file": "x.wav"}
    assert StrategyRequest.from_context(context) == StrategyRequest.model_validate(_payload())
    incomplete = {k: v for k, v in _payload().items() if k != "tire_data"}
    with pytest.raises(ValidationError):
        StrategyRequest.from_context(incomplete)
    with pytest.raises(ValueError):
        StrategyRequest.from_context(["not", "a", "mapping"])


def test_from_context_accepts_a_shared_context_with_sector_times():
    # The natural-query router reads telemetry.sector_times for performance questions; the same
    # context must also serve strategy questions. sector_times is accepted but not modelled.
    context = _payload()
    context["telemetry"]["sector_times"] = {"1": 28.1, "2": 30.2, "3": 22.0}
    context["question_history"] = ["How are my sector times?"]
    request = StrategyRequest.from_context(context)
    assert request.telemetry.sector_times == {1: 28.1, 2: 30.2, 3: 22.0}
    assert StrategyRequest.model_validate(json.loads(request.model_dump_json())) == request

    with_sectors = generate_strategy_response(request)
    without = generate_strategy_response(StrategyRequest.model_validate(_payload()))
    assert "telemetry.sector_times" in with_sectors["not_modelled_inputs"]
    assert "telemetry.sector_times" not in without["not_modelled_inputs"]
    assert with_sectors["strategies"] == without["strategies"]

    as_list = _payload()
    as_list["telemetry"]["sector_times"] = [28.1, 30.2]
    assert "telemetry.sector_times" in generate_strategy_response(StrategyRequest.from_context(as_list))["not_modelled_inputs"]
    unknown = _payload()
    unknown["telemetry"]["speed_trace"] = [301.0]
    with pytest.raises(ValidationError):
        StrategyRequest.from_context(unknown)


def test_schema_driver_values_may_come_from_telemetry_only():
    payload = _payload()
    payload["telemetry"]["throttle_aggressiveness"] = 0.72
    del payload["driver_profile"]["braking_consistency"]
    del payload["driver_profile"]["throttle_aggressiveness"]
    body = generate_strategy_response(StrategyRequest.model_validate(payload))
    assert body["model"]["driver_multiplier"] == 1.0
    assert not any(name.startswith("driver_profile.braking") for name in body["not_modelled_inputs"])

    both = _payload()  # braking in telemetry and in the profile: the profile value is reported as unused
    body = generate_strategy_response(StrategyRequest.model_validate(both))
    assert "driver_profile.braking_consistency (overridden by telemetry.braking_consistency)" in body["not_modelled_inputs"]


def test_schema_free_form_fields_accept_flat_json_values_and_do_not_change_numbers():
    payload = _payload()
    payload["race_state"]["weather_forecast"] = [
        {"laps": [10, 20], "rain": True, "lap": None, "weather": "wet", "chance": 0.4},
    ]
    payload["competition"][0]["estimated_strategy"] = [{"pit_laps": [20, 40], "compounds": ["soft", "hard"]}]
    request = StrategyRequest.model_validate_json(json.dumps(payload))
    assert request.race_state.weather_forecast[0]["rain"] is True
    assert request.race_state.weather_forecast[0]["laps"] == [10, 20]
    assert StrategyRequest.model_validate(json.loads(request.model_dump_json())) == request
    body = generate_strategy_response(request)
    assert body["strategies"] == generate_strategy_response(StrategyRequest.model_validate(_payload()))["strategies"]
    assert {"race_state.weather_forecast", "competition[].estimated_strategy"} <= set(body["not_modelled_inputs"])


def test_schema_base_performance_minimum_is_inclusive():
    payload = _payload()
    payload["tire_data"]["soft"]["base_performance"] = 0.5
    assert StrategyRequest.model_validate(payload).tire_data[SOFT].base_performance == 0.5


def _schema_case(path, value):
    def apply(payload):
        target = payload
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value
        return payload

    return apply


_DELETE = object()


@pytest.mark.parametrize(
    "mutation",
    [
        _schema_case(("telemetry", "lap_times"), json.loads("[80.0, NaN]")),
        _schema_case(("telemetry", "lap_times"), []),
        _schema_case(("telemetry", "lap_times"), [80.0] * 201),
        _schema_case(("telemetry", "lap_times"), _DELETE),
        _schema_case(("telemetry", "lap_times"), ["80.5"]),
        _schema_case(("telemetry", "speed_trace"), [1.0]),
        _schema_case(("telemetry", "sector_times"), [28.0, 0.0]),
        _schema_case(("telemetry", "sector_times"), {"S1": 28.0}),
        _schema_case(("telemetry", "sector_times"), {}),
        _schema_case(("telemetry", "sector_times"), json.loads("[NaN]")),
        _schema_case(("tire_data", "soft", "pit_stop_delta"), json.loads("Infinity")),
        _schema_case(("tire_data", "soft", "pit_stop_delta"), -1.0),
        _schema_case(("tire_data", "soft", "degradation_rate"), -0.01),
        _schema_case(("tire_data", "soft", "base_performance"), 0),
        _schema_case(("tire_data", "soft", "base_performance"), 0.3),
        _schema_case(("tire_data", "soft", "base_performance"), True),
        _schema_case(("tire_data", "soft", "compound"), "hard"),
        _schema_case(("tire_data", "soft", "peak_performance_window"), [8, 2]),
        _schema_case(("tire_data", "soft", "warm_up_laps"), -1),
        _schema_case(("tire_data", "supersoft"), {"base_performance": 1.0, "degradation_rate": 0.01,
                                                  "warm_up_laps": 1, "peak_performance_window": [1, 5],
                                                  "pit_stop_delta": 20.0}),
        _schema_case(("race_state", "current_lap"), 5.5),
        _schema_case(("race_state", "current_lap"), 60),
        _schema_case(("race_state", "total_laps"), 201),
        _schema_case(("race_state", "weather"), "sunny"),
        _schema_case(("race_state", "weather"), None),
        _schema_case(("race_state", "safety_car_probability"), 7.0),
        _schema_case(("race_state", "track_temperature"), -300.0),
        _schema_case(("race_state", "current_tire_age"), _DELETE),
        _schema_case(("race_state", "current_compound"), "wet"),
        _schema_case(("car_status", "damage", "rear_wing"), 1.0),
        _schema_case(("car_status", "engine_wear"), _DELETE),
        _schema_case(("driver_profile", "braking_consistency"), -10),
        # Not supplied in telemetry either (the payload's telemetry has no throttle value).
        _schema_case(("driver_profile", "throttle_aggressiveness"), _DELETE),
        _schema_case(("race_state", "weather_forecast"), [{"window": {"from": 10, "to": 20}}]),
        _schema_case(("race_state", "weather_forecast"), [{"laps": [[10, 20]]}]),
        _schema_case(("race_state", "weather_forecast"), [{"lap": 10**12}]),
        _schema_case(("competition",), [{"driver_id": "HAM", "tire_compound": "soft", "tire_age": 1, "gap_to_leader": 1.0}] * 2),
        _schema_case(("unexpected",), 1),
    ],
)
def test_schema_rejects_invalid_payloads(mutation):
    with pytest.raises(ValidationError):
        StrategyRequest.model_validate(mutation(_payload()))
