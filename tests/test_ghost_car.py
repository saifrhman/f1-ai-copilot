"""Ghost-car comparison: distance alignment, validation and artifact writing.

The synthetic laps are generated from a known speed-vs-distance profile on a
circular 3 km track, so the true time delta at every position is known and can
be asserted numerically.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from core_modules.ghost_car import ghost_car_visualizer as ghost
from core_modules.ghost_car.ghost_car_visualizer import (
    artifact_filename,
    compare_laps,
    generate_ghost_comparison,
)
from core_modules.ghost_car.schemas import GhostCarRequest, LapTelemetry

TRACK_LENGTH = 3000.0
RADIUS = TRACK_LENGTH / (2 * math.pi)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_speed(s):
    return 60.0 + 20.0 * np.sin(2 * np.pi * 3 * s / TRACK_LENGTH)  # m/s


def _time_along(s, factor):
    """Exact (fine-grid) elapsed time to reach each distance in ``s``."""

    pace = 1.0 / (_base_speed(s) * factor(s))
    return np.concatenate(([0.0], np.cumsum(0.5 * (pace[1:] + pace[:-1]) * np.diff(s))))


def make_lap(
    rate_hz,
    factor=lambda s: np.ones_like(s),
    xy=True,
    t0=0.0,
    length=TRACK_LENGTH,
    xy_scale=1.0,
    speed_scale=1.0,
    end_on_line=True,
    **channels,
):
    """Sample a lap driven at ``_base_speed * factor`` at ``rate_hz``.

    The first sample is on the timing line. With ``end_on_line`` a final sample
    is placed exactly on the line; without it sampling simply stops at the last
    tick before the line, as real telemetry does. Extra keyword arguments map a
    channel name to a function of distance.
    """

    s_fine = np.linspace(0.0, length, 300_001)
    t_fine = _time_along(s_fine, factor)
    total = t_fine[-1]
    stamps = np.arange(0.0, total, 1.0 / rate_hz)
    if end_on_line and total - stamps[-1] > 1e-6:
        stamps = np.append(stamps, total)
    s = np.interp(stamps, t_fine, s_fine)
    lap = {
        "timestamps": (stamps + t0).tolist(),
        "speed": (_base_speed(s) * factor(s) * 3.6 * speed_scale).tolist(),
    }
    if xy:
        lap["x"] = (xy_scale * RADIUS * np.cos(s / RADIUS)).tolist()
        lap["y"] = (xy_scale * RADIUS * np.sin(s / RADIUS)).tolist()
    for name, fn in channels.items():
        lap[name] = [fn(value) for value in s]
    return lap


def lap_duration(factor=lambda s: np.ones_like(s), length=TRACK_LENGTH):
    return float(_time_along(np.linspace(0.0, length, 300_001), factor)[-1])


def official(lap, factor=lambda s: np.ones_like(s), length=TRACK_LENGTH):
    """The lap with its true timing-line lap time, rounded to the millisecond."""

    return dict(lap, lap_time=round(lap_duration(factor, length), 3))


def _slowdown(start, end, factor):
    return lambda s: np.where((s >= start) & (s <= end), factor, 1.0)


def _request(lap1, lap2, **extra):
    return {"lap1_telemetry": lap1, "lap2_telemetry": lap2, **extra}


def _delta_at(result, distance):
    traces = result["traces"]
    return float(np.interp(distance, traces["distance_m"], traces["delta_time_s"]))


# --------------------------------------------------------------------------- #
# Alignment correctness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("with_lap_times", [False, True])
@pytest.mark.parametrize("xy", [True, False])
def test_three_second_slower_lap_ends_three_seconds_behind(xy, with_lap_times):
    reference = lap_duration()
    scale = reference / (reference + 3.0)
    slow = lambda s: np.full_like(s, scale)  # noqa: E731
    lap1 = make_lap(4, xy=xy)
    lap2 = make_lap(10, factor=slow, xy=xy, t0=5000.0)
    if with_lap_times:  # both traces then provably run line to line
        lap1, lap2 = official(lap1), official(lap2, factor=slow)

    result = compare_laps(_request(lap1, lap2))
    summary = result["summary"]

    assert result["alignment"]["distance_source"] == ("xy_path" if xy else "speed_integration")
    assert result["alignment"]["method"] == ("lap_fraction" if with_lap_times else "distance")
    assert result["lap_time_delta_s"] == (pytest.approx(3.0, abs=1e-3) if with_lap_times else None)
    assert summary["final_delta_s"] == pytest.approx(3.0, abs=0.01)
    assert summary["max_delta_s"] == pytest.approx(3.0, abs=0.01)
    assert summary["max_delta_at_m"] == pytest.approx(TRACK_LENGTH, abs=5.0)
    assert summary["largest_time_loss"]["time_s"] == pytest.approx(3.0, abs=0.01)
    # The loss is spread over the whole lap; only the first/last 2 % of it is trimmed.
    assert summary["largest_time_loss"]["from_m"] < 150.0
    assert summary["largest_time_loss"]["to_m"] > TRACK_LENGTH - 150.0
    assert summary["largest_time_gain"] is None
    # Uniformly slower: the gap grows with elapsed time, so half-way it is ~1.5 s.
    halfway = lap_duration(length=TRACK_LENGTH / 2) / reference * 3.0
    assert _delta_at(result, TRACK_LENGTH / 2) == pytest.approx(halfway, abs=0.02)
    # Speeds on the grid are the lap's own speeds, lap 2 uniformly lower.
    speed_delta = np.array(result["traces"]["speed_delta_kmh"])
    assert np.all(speed_delta < 0)


def test_faster_comparison_lap_shows_negative_delta_and_gain():
    reference = lap_duration()
    scale = reference / (reference - 2.0)
    result = compare_laps(_request(make_lap(5), make_lap(5, factor=lambda s: np.full_like(s, scale))))

    assert result["summary"]["final_delta_s"] == pytest.approx(-2.0, abs=0.01)
    assert result["summary"]["largest_time_gain"]["time_s"] == pytest.approx(2.0, abs=0.01)
    assert result["summary"]["largest_time_loss"] is None


@pytest.mark.parametrize("xy", [True, False])
@pytest.mark.parametrize("alignment", ["auto", "distance", "lap_fraction"])
def test_identical_driving_at_1hz_and_2hz_shows_no_delta(xy, alignment):
    result = compare_laps(_request(make_lap(1, xy=xy), make_lap(2, xy=xy), alignment=alignment))

    delta = np.array(result["traces"]["delta_time_s"])
    assert result["laps"]["lap1"]["samples"] != result["laps"]["lap2"]["samples"]
    assert np.max(np.abs(delta)) < 0.05
    # 0 by construction for lap_fraction; for forced "distance" with x/y the 1 Hz
    # chords are ~1.5 m shorter over the lap (~0.03 s at 60 m/s).
    assert abs(result["summary"]["final_delta_s"]) < 0.05
    speed_delta = np.array(result["traces"]["speed_delta_kmh"])
    assert np.max(np.abs(speed_delta)) < 3.0  # km/h, linear-interpolation error at 1 Hz only


def test_local_time_loss_is_located_by_distance():
    loss_factor = _slowdown(1000.0, 1300.0, 0.7)
    expected_loss = lap_duration(loss_factor) - lap_duration()
    result = compare_laps(_request(make_lap(10), make_lap(20, factor=loss_factor)))

    loss = result["summary"]["largest_time_loss"]
    assert result["summary"]["final_delta_s"] == pytest.approx(expected_loss, abs=0.01)
    assert loss["time_s"] == pytest.approx(expected_loss, abs=0.02)
    assert loss["from_m"] == pytest.approx(1000.0, abs=15.0)
    assert loss["to_m"] == pytest.approx(1300.0, abs=15.0)
    assert abs(_delta_at(result, 900.0)) < 0.01
    assert _delta_at(result, 2000.0) == pytest.approx(expected_loss, abs=0.01)
    speed_delta_in_zone = np.interp(1150.0, result["traces"]["distance_m"], result["traces"]["speed_delta_kmh"])
    assert speed_delta_in_zone < -40.0


def test_different_sample_counts_share_one_grid():
    result = compare_laps(_request(make_lap(3), make_lap(17)))
    traces = result["traces"]
    grid_points = result["alignment"]["grid_points"]

    assert result["laps"]["lap1"]["samples"] < result["laps"]["lap2"]["samples"]
    assert grid_points == min(result["laps"]["lap2"]["samples"], ghost.MAX_GRID_POINTS)
    assert {len(values) for values in traces.values()} == {grid_points}
    assert traces["distance_m"][0] == 0.0
    assert np.all(np.diff(traces["distance_m"]) > 0)


def test_grid_is_capped_for_large_inputs():
    result = compare_laps(_request(make_lap(50), make_lap(60)))
    assert result["laps"]["lap2"]["samples"] > ghost.MAX_GRID_POINTS
    assert result["alignment"]["grid_points"] == ghost.MAX_GRID_POINTS
    assert abs(result["summary"]["final_delta_s"]) < 0.01


def test_speed_calibration_offset_is_normalised_for_line_to_line_laps():
    # Same driving, but lap 2's speed sensor reads 1 % high -> 1 % longer distance.
    lap1, lap2 = official(make_lap(5, xy=False)), official(make_lap(5, xy=False, speed_scale=1.01))

    auto = compare_laps(_request(lap1, lap2))
    assert auto["alignment"]["method"] == "lap_fraction"
    assert auto["alignment"]["line_to_line"] == {"lap1": True, "lap2": True}
    assert auto["alignment"]["distance_mismatch_fraction"] == pytest.approx(0.01 / 1.01, abs=1e-4)
    assert np.max(np.abs(auto["traces"]["delta_time_s"])) < 0.01
    assert auto["warnings"] == []

    # Forcing metre-by-metre comparison shows the calibration error as a fake gain.
    forced = compare_laps(_request(lap1, lap2, alignment="distance"))
    assert forced["alignment"]["method"] == "distance"
    assert forced["alignment"]["compared_distance_m"] == pytest.approx(TRACK_LENGTH, abs=1.0)
    assert forced["summary"]["final_delta_s"] < -0.3
    assert any("differ by 1.0%" in warning for warning in forced["warnings"])


def test_without_lap_times_auto_compares_metres_and_warns_about_the_drift():
    # Without an official lap time nothing shows that the traces end on the line, so auto
    # alignment compares metres and warns about the drift, even when the totals agree within
    # 3 %: normalising them would fake a gap (see the realistic tests).
    lap1, lap2 = make_lap(5, xy=False), make_lap(5, xy=False, speed_scale=1.01)
    result = compare_laps(_request(lap1, lap2))

    assert result["alignment"]["method"] == "distance"
    assert result["alignment"]["line_to_line"] == {"lap1": False, "lap2": False}
    assert result["summary"]["final_delta_s"] < -0.3
    assert any("differ by 1.0%" in warning for warning in result["warnings"])
    assert result["lap_time_delta_s"] is None


@pytest.mark.parametrize("mismatch, method", [(0.029, "lap_fraction"), (0.031, "distance")])
def test_auto_lap_fraction_distance_tolerance(mismatch, method):
    scale = 1.0 / (1.0 - mismatch)  # lap 2 total distance = scale * lap 1 total
    lap1, lap2 = official(make_lap(5, xy=False)), official(make_lap(5, xy=False, speed_scale=scale))
    result = compare_laps(_request(lap1, lap2))

    assert result["alignment"]["distance_mismatch_fraction"] == pytest.approx(mismatch, abs=1e-6)
    assert result["alignment"]["method"] == method


@pytest.mark.parametrize(
    "lap2_change, line_to_line",
    [
        ({}, True),
        ({"lap_time": "+0.005"}, True),
        ({"lap_time": "+0.02"}, False),
        ({"lap_time": None}, False),
        ({"lap_time": None, "sector_times": "split"}, True),
    ],
)
def test_auto_lap_fraction_needs_timestamps_spanning_the_official_lap_time(lap2_change, line_to_line):
    lap1, lap2 = official(make_lap(5)), official(make_lap(8))
    lap_time = lap2["lap_time"]
    for key, value in lap2_change.items():
        if value is None:
            lap2.pop(key)
        elif value == "split":
            thirds = round(lap_time / 3, 3)
            lap2[key] = [thirds, thirds, round(lap_time - 2 * thirds, 3)]
        else:
            lap2[key] = lap_time + float(value)
    result = compare_laps(_request(lap1, lap2))

    assert result["alignment"]["line_to_line"] == {"lap1": True, "lap2": line_to_line}
    assert result["alignment"]["method"] == ("lap_fraction" if line_to_line else "distance")
    assert abs(result["summary"]["final_delta_s"]) < 0.02


@pytest.mark.parametrize("with_lap_times", [False, True])
@pytest.mark.parametrize("xy", [True, False])
def test_identical_driving_with_realistic_sampling_shows_no_delta(xy, with_lap_times):
    # Sampling stops at the last tick before the line: 1 Hz ends ~1 s short,
    # 2 Hz ~0.5 s short. Stretching each trace to its own end used to show +0.5 s.
    lap1, lap2 = make_lap(1, xy=xy, end_on_line=False), make_lap(2, xy=xy, end_on_line=False)
    if with_lap_times:
        lap1, lap2 = official(lap1), official(lap2)
    result = compare_laps(_request(lap1, lap2))
    delta = np.array(result["traces"]["delta_time_s"])

    assert result["alignment"]["method"] == "distance"
    assert result["alignment"]["line_to_line"] == {"lap1": False, "lap2": False}
    assert np.max(np.abs(delta)) < 0.05
    assert abs(result["summary"]["final_delta_s"]) < 0.05
    assert result["summary"]["largest_time_loss"] is None
    assert result["summary"]["largest_time_gain"] is None
    if with_lap_times:
        assert result["lap_time_delta_s"] == 0.0
        assert sum("less than its lap time" in w for w in result["warnings"]) == 2


def test_slower_lap_with_realistic_sampling_reports_the_gap_where_it_is_measured():
    reference = lap_duration()
    scale = reference / (reference + 3.4)
    slow = lambda s: np.full_like(s, scale)  # noqa: E731
    lap1 = official(make_lap(1, xy=False, end_on_line=False))
    lap2 = official(make_lap(2, factor=slow, xy=False, end_on_line=False), factor=slow)
    result = compare_laps(_request(lap1, lap2))
    summary = result["summary"]

    assert result["alignment"]["method"] == "distance"
    assert result["lap_time_delta_s"] == pytest.approx(3.4, abs=1e-3)
    compared = result["alignment"]["compared_distance_m"]
    assert TRACK_LENGTH - 100.0 < compared < TRACK_LENGTH
    assert summary["final_delta_at_m"] == compared
    # Uniformly slower: the gap at position c is 3.4 s * t(c) / T.
    expected = 3.4 * lap_duration(length=compared) / reference
    assert summary["final_delta_s"] == pytest.approx(expected, abs=0.02)
    assert summary["largest_time_loss"]["time_s"] == pytest.approx(expected, abs=0.02)
    assert any("final_delta_s is the gap at compared_distance_m" in w for w in result["warnings"])


def test_tolerated_pre_line_sample_does_not_shift_the_comparison():
    lap1, lap2 = official(make_lap(5)), official(make_lap(5))
    speed = float(_base_speed(np.array([0.0]))[0])
    before = -0.4 * speed  # one sample 0.4 s before the timing line
    lap1["timestamps"].insert(0, -0.4)
    lap1["speed"].insert(0, speed * 3.6)
    lap1["x"].insert(0, float(RADIUS * np.cos(before / RADIUS)))
    lap1["y"].insert(0, float(RADIUS * np.sin(before / RADIUS)))

    auto = compare_laps(_request(lap1, lap2))
    assert auto["alignment"]["method"] == "distance"
    assert auto["alignment"]["line_to_line"] == {"lap1": False, "lap2": True}
    assert auto["lap_time_delta_s"] == 0.0
    assert abs(auto["summary"]["final_delta_s"]) < 0.05  # lap_fraction gave -0.4
    assert any(w.startswith("lap1:") and "more than its lap time" in w for w in auto["warnings"])

    forced = compare_laps(_request(lap1, lap2, alignment="lap_fraction"))
    assert any("lap_fraction alignment was requested" in w for w in forced["warnings"])


def test_partial_trace_uses_shared_distance_range():
    lap1 = make_lap(5)
    lap2 = make_lap(5, length=0.6 * TRACK_LENGTH)
    result = compare_laps(_request(lap1, lap2))

    assert result["alignment"]["method"] == "distance"
    assert result["alignment"]["compared_distance_m"] == pytest.approx(0.6 * TRACK_LENGTH, abs=2.0)
    assert result["traces"]["distance_m"][-1] == pytest.approx(0.6 * TRACK_LENGTH, abs=2.0)
    assert abs(result["summary"]["final_delta_s"]) < 0.02
    assert any("differ by 40.0%" in warning for warning in result["warnings"])


def _stop_laps():
    """5 Hz laps at 180 km/h (10 m per sample); lap 1 stands still for 2 s at 505 m."""

    lap1_speed = [180.0] * 51 + [0.0] * 11 + [180.0] * 89
    lap2_speed = [180.0] * 51 + [0.0] + [180.0] * 89
    lap1 = {"timestamps": [round(0.2 * i, 1) for i in range(151)], "speed": lap1_speed}
    lap2 = {"timestamps": [round(0.2 * i, 1) for i in range(141)], "speed": lap2_speed}
    return lap1, lap2


def test_stationary_car_loses_time_exactly_at_the_stop():
    result = compare_laps(_request(*_stop_laps()))
    grid = np.array(result["traces"]["distance_m"])
    delta = np.array(result["traces"]["delta_time_s"])

    assert result["laps"]["lap1"]["distance_m"] == result["laps"]["lap2"]["distance_m"] == 1390.0
    # Up to and including the stop both laps arrive at the same time (first
    # arrival); right after it lap 2 is 2 s ahead, also inside the sample
    # interval just before and just after the stop (grid points 500.4, 509.7 m).
    assert np.any((grid > 500.0) & (grid < 505.0)) and np.any((grid > 505.0) & (grid < 510.0))
    assert np.all(delta[grid <= 505.0] == 0.0)
    assert np.all(delta[grid > 505.0] == -2.0)
    assert result["summary"]["largest_time_gain"]["time_s"] == 2.0


def test_interp_uses_first_arrival_at_and_departure_after_a_stop():
    position = np.array([0.0, 10.0, 15.0, 15.0, 20.0, 30.0])
    elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    grid = np.array([0.0, 12.5, 15.0, 17.5, 30.0])
    assert ghost._interp(grid, position, elapsed).tolist() == [0.0, 1.5, 2.0, 3.5, 5.0]
    # Stationary at the very start.
    start = ghost._interp(np.array([0.0, 2.5]), np.array([0.0, 0.0, 0.0, 5.0]), np.array([0.0, 1.0, 2.0, 3.0]))
    assert start.tolist() == [0.0, 2.5]


# --------------------------------------------------------------------------- #
# Optional channels: never fabricated
# --------------------------------------------------------------------------- #


def test_minimal_laps_report_missing_channels_and_null_analyses(tmp_path):
    lap = {"timestamps": [0.0, 1.0, 2.0, 3.0], "speed": [100.0, 120.0, 140.0, 150.0]}
    result = generate_ghost_comparison(_request(lap, lap), tmp_path)

    all_optional = ["x", "y", "brake", "throttle", "steering", "drs", "gear"]
    assert result["missing_channels"] == {"lap1": all_optional, "lap2": all_optional}
    for analysis in ("braking", "drs", "throttle", "gear"):
        assert result[analysis] is None
    assert result["sector_time_deltas_s"] is None
    assert result["laps"]["lap1"]["sector_times_s"] is None
    assert result["laps"]["lap1"]["lap_time_s"] == 3.0
    assert result["laps"]["lap1"]["lap_time_source"] == "telemetry_span"
    assert result["lap_time_delta_s"] is None
    assert any("lap_time_delta_s is null" in warning for warning in result["warnings"])
    assert result["alignment"]["distance_source"] == "speed_integration"
    # trapezoid: (100+120)/2 + (120+140)/2 + (140+150)/2 = 385 km/h*s = 106.94 m
    assert result["laps"]["lap1"]["distance_m"] == pytest.approx(385 / 3.6, abs=0.01)
    assert result["track_section"] is None
    assert (tmp_path / result["artifact_path"]).stat().st_size > 1000


def test_channel_supplied_for_one_lap_only_is_not_compared():
    lap1 = make_lap(5, brake=lambda s: 1.0 if 900 <= s <= 1000 else 0.0, gear=lambda s: 7)
    lap2 = make_lap(5, xy=False, gear=lambda s: 7)
    result = compare_laps(_request(lap1, lap2))

    assert result["braking"] is None
    assert "brake" in result["missing_channels"]["lap2"]
    assert "brake" not in result["missing_channels"]["lap1"]
    assert result["alignment"]["distance_source"] == "speed_integration"  # x/y only on lap 1
    assert result["gear"] == {"lap1_max_gear": 7, "lap1_gear_changes": 0, "lap2_max_gear": 7, "lap2_gear_changes": 0}


def test_braking_and_drs_zones_are_reported_in_metres():
    lap1 = make_lap(
        20,
        brake=lambda s: 1.0 if 900 <= s <= 1000 else 0.0,
        drs=lambda s: bool(2000 <= s <= 2500),
        throttle=lambda s: 1.0 if s < 1500 else 0.5,
    )
    lap2 = make_lap(
        20,
        brake=lambda s: 0.8 if 920 <= s <= 1010 else 0.0,
        drs=lambda s: bool(2100 <= s <= 2500),
        throttle=lambda s: 1.0 if s < 750 else 0.5,
    )
    result = compare_laps(_request(lap1, lap2))

    braking = result["braking"]
    assert "heuristic" in braking and braking["threshold"] == ghost.BRAKE_THRESHOLD
    assert len(braking["lap1_zones"]) == len(braking["lap2_zones"]) == 1
    assert braking["lap1_zones"][0]["start_m"] == pytest.approx(900.0, abs=5.0)
    assert braking["lap1_zones"][0]["end_m"] == pytest.approx(1000.0, abs=5.0)
    (match,) = braking["matched_zones"]
    assert match["brake_point_delta_m"] == pytest.approx(20.0, abs=6.0)

    drs = result["drs"]
    assert drs["lap1_zones"][0]["start_m"] == pytest.approx(2000.0, abs=5.0)
    assert drs["lap2_zones"][0]["start_m"] == pytest.approx(2100.0, abs=5.0)
    assert drs["lap1_open_distance_m"] == pytest.approx(500.0, abs=8.0)
    assert drs["lap2_open_distance_m"] == pytest.approx(400.0, abs=8.0)

    throttle = result["throttle"]
    assert "heuristic" in throttle
    assert throttle["lap1_full_throttle_fraction"] == pytest.approx(0.5, abs=0.01)
    assert throttle["lap2_full_throttle_fraction"] == pytest.approx(0.25, abs=0.01)


def _pairs(zones1, zones2):
    arrays = [np.array([zone[i] for zone in zones], dtype=float) for zones in (zones1, zones2) for i in (0, 1)]
    first, second = ghost._pair_zones(*arrays)
    return list(zip(first.tolist(), second.tolist()))


def test_zone_pairing_uses_overlap():
    zones1 = [(0.0, 10.0), (100.0, 150.0), (400.0, 410.0)]
    zones2 = [(105.0, 160.0), (170.0, 180.0), (395.0, 409.0)]
    assert _pairs(zones1, zones2) == [(1, 0), (2, 2)]
    assert _pairs([], zones2) == [] and _pairs(zones1, []) == []


def test_zone_pairing_is_one_to_one_by_greatest_overlap():
    # One long lap-2 zone overlaps two lap-1 zones (10 m and 30 m): it is paired
    # once, with the zone it overlaps most.
    assert _pairs([(100.0, 150.0), (160.0, 200.0)], [(140.0, 190.0)]) == [(1, 0)]
    # Single-sample zones (start == end) still pair with the zone containing them.
    assert _pairs([(50.0, 50.0)], [(40.0, 60.0)]) == [(0, 0)]


def test_zones_are_placed_on_reference_metres_under_lap_fraction():
    # Same driving, lap 2's speed sensor 2 % high: its own metres run 2 % long,
    # but on the reference axis its brake and DRS zones sit where lap 1's do.
    brake = lambda s: 1.0 if 2000 <= s <= 2100 else 0.0  # noqa: E731
    drs = lambda s: bool(2400 <= s <= 2900)  # noqa: E731
    lap1 = official(make_lap(20, xy=False, brake=brake, drs=drs))
    lap2 = official(make_lap(20, xy=False, speed_scale=1.02, brake=brake, drs=drs))
    result = compare_laps(_request(lap1, lap2))

    assert result["alignment"]["method"] == "lap_fraction"
    (match,) = result["braking"]["matched_zones"]
    assert match["lap1_start_m"] == pytest.approx(2000.0, abs=3.0)
    assert match["lap2_start_m"] == pytest.approx(match["lap1_start_m"], abs=0.02)
    assert match["brake_point_delta_m"] == 0.0
    assert result["drs"]["lap2_zones"] == pytest.approx(result["drs"]["lap1_zones"])
    assert result["drs"]["lap2_open_distance_m"] == pytest.approx(result["drs"]["lap1_open_distance_m"], abs=0.02)


def test_low_rate_zone_and_throttle_resolution_is_exact():
    # 1 Hz, speeds 10/20/20/10/10 m/s -> positions 0, 15, 35, 50, 60 m.
    lap = {
        "timestamps": [0.0, 1.0, 2.0, 3.0, 4.0],
        "speed": [36.0, 72.0, 72.0, 36.0, 36.0],
        "drs": [False, True, True, False, False],
        "brake": [0.0, 0.0, 0.5, 0.5, 0.0],
        "throttle": [1.0, 1.0, 0.0, 0.0, 1.0],
    }
    result = compare_laps(_request(lap, lap))

    # Zones run from the first to the last flagged sample.
    assert result["drs"]["lap1_zones"] == [{"start_m": 15.0, "end_m": 35.0}]
    assert result["drs"]["lap1_open_distance_m"] == 20.0
    assert result["braking"]["lap1_zones"] == [{"start_m": 35.0, "end_m": 50.0}]
    # Throttle holds each sample until the next: segments 0-15 and 15-35 m at full.
    assert result["throttle"]["lap1_full_throttle_fraction"] == pytest.approx(35.0 / 60.0, abs=1e-4)


def test_brake_chatter_at_max_size_is_capped():
    n = 20_000
    stamps = [i * 0.01 for i in range(n)]
    lap = {
        "timestamps": stamps,
        "speed": [200.0] * n,
        "brake": [float(i % 2) for i in range(n)],
        "drs": [bool(i % 2) for i in range(n)],
    }
    result = compare_laps(_request(lap, lap))
    braking, drs = result["braking"], result["drs"]

    assert braking["lap1_zone_count"] == braking["lap2_zone_count"] == braking["matched_zone_count"] == n // 2
    assert braking["zones_truncated"] is True and drs["zones_truncated"] is True
    assert len(braking["lap1_zones"]) == len(braking["matched_zones"]) == ghost.MAX_REPORTED_ZONES
    assert all(m["brake_point_delta_m"] == 0.0 for m in braking["matched_zones"])
    assert drs["lap1_zone_count"] == n // 2 and len(drs["lap1_zones"]) == ghost.MAX_REPORTED_ZONES
    assert len(json.dumps(result)) < 300_000


# --------------------------------------------------------------------------- #
# Lap and sector times
# --------------------------------------------------------------------------- #


def test_lap_and_sector_times_are_reported_with_their_source():
    lap1 = make_lap(5, xy=False)
    lap2 = make_lap(5, xy=False)
    lap1.update(lap_time=80.0, sector_times=[26.5, 26.7, 26.8], lap_number=3)
    lap2.update(sector_times=[26.4, 26.9, 27.2])

    result = compare_laps(_request(lap1, lap2))
    assert result["laps"]["lap1"]["lap_time_source"] == "supplied"
    assert result["laps"]["lap2"]["lap_time_source"] == "sector_times_sum"
    assert result["laps"]["lap2"]["lap_time_s"] == pytest.approx(80.5)
    assert result["lap_time_delta_s"] == pytest.approx(0.5)
    assert result["sector_time_deltas_s"] == pytest.approx([-0.1, 0.2, 0.4])
    assert result["laps"]["lap1"]["lap_number"] == 3


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"sector_times": [26.5, 26.7]}, "at least 3"),
        ({"sector_times": [26.5, 26.7, 26.8, 1.0]}, "at most 3"),
        ({"sector_times": [26.5, -1.0, 26.8]}, "greater than 0"),
        ({"sector_times": [26.5, float("nan"), 26.8]}, "finite"),
        ({"sector_times": [26.5, 26.7, 26.8], "lap_time": 81.0}, "sector_times sum"),
        ({"lap_time": 0.0}, "greater than 0"),
        ({"lap_time": float("inf")}, "finite"),
        ({"lap_time": 1.0}, "longer than the lap time"),
        ({"sector_times": [0.3, 0.3, 0.4]}, "longer than the lap time"),
        ({"sector_times": [1300.0, 1300.0, 1300.0]}, "more than the 3600 s"),
        ({"timestamps": [0.0, 1000.0, 2000.0, 3000.0, 4000.0]}, "more than the 3600 s"),
    ],
)
def test_invalid_timing_is_rejected(changes, message):
    lap = {"timestamps": [0.0, 0.5, 1.0, 1.5, 2.0], "speed": [200.0] * 5, **changes}
    with pytest.raises(ValidationError, match=message):
        LapTelemetry.model_validate(lap)


# --------------------------------------------------------------------------- #
# Strict input validation
# --------------------------------------------------------------------------- #


def _valid_lap(**changes):
    lap = {
        "timestamps": [0.0, 0.5, 1.0, 1.5],
        "speed": [180.0, 200.0, 210.0, 190.0],
        "brake": [0.0, 0.0, 0.0, 0.4],
        "throttle": [0.7, 0.9, 1.0, 0.5],
        "drs": [False, True, True, False],
        "gear": [5, 6, 7, 5],
        "lap_number": 1,
    }
    lap.update(changes)
    return lap


def test_valid_lap_passes_and_integers_are_normalised_to_floats():
    lap = LapTelemetry.model_validate(_valid_lap(timestamps=[0, 1, 2, 3]))
    assert lap.timestamps == [0.0, 1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in lap.timestamps)


@pytest.mark.parametrize(
    "changes",
    [
        {"speed": [180.0, True, 210.0, 190.0]},
        {"speed": [180.0, None, 210.0, 190.0]},
        {"speed": [180.0, "200", 210.0, 190.0]},
        {"speed": [180.0, float("nan"), 210.0, 190.0]},
        {"speed": [180.0, float("inf"), 210.0, 190.0]},
        {"speed": [180.0, -1.0, 210.0, 190.0]},
        {"speed": [180.0, 400.1, 210.0, 190.0]},
        {"speed": [[180.0], 200.0, 210.0, 190.0]},
        {"timestamps": [0.0, 0.5, 0.5, 1.5]},
        {"timestamps": [0.0, 1.0, 0.5, 1.5]},
        {"timestamps": [0.0, None, 1.0, 1.5]},
        {"timestamps": "0,0.5,1,1.5"},
        {"speed": [180.0, 200.0, 210.0]},
        {"brake": [0.0, 0.0, 0.0]},
        {"brake": [0.0, 0.0, 0.0, 1.5]},
        {"throttle": [0.7, 0.9, 1.0, -0.1]},
        {"drs": ["false", "false", "no", "false"]},
        {"drs": [0, 1, 1, 0]},
        {"gear": [5, 6, 7.5, 5]},
        {"gear": [5, 6, 7.0, 5]},
        {"gear": [5, 6, 9, 5]},
        {"gear": [5, True, 7, 5]},
        {"gear": [5, 6, -1, 5]},
        {"x": [0.0, 10.0, 20.0, 30.0]},
        {"y": [0.0, 10.0, 20.0, 30.0]},
        {"x": [0.0, 10.0, 20.0, float("inf")], "y": [0.0, 0.0, 0.0, 0.0]},
        {"lap_number": 0},
        {"lap_number": 1.9},
        {"lap_number": True},
        {"speeds": [180.0, 200.0, 210.0, 190.0]},
        {"timestamps": [0.0], "speed": [180.0], "brake": None, "throttle": None, "drs": None, "gear": None},
    ],
)
def test_invalid_lap_telemetry_is_rejected(changes):
    with pytest.raises(ValidationError):
        LapTelemetry.model_validate(_valid_lap(**changes))


def test_required_channels_cannot_be_omitted():
    for missing in ("timestamps", "speed"):
        lap = _valid_lap()
        del lap[missing]
        with pytest.raises(ValidationError, match=missing):
            LapTelemetry.model_validate(lap)


def test_sample_count_limits():
    n = 20_000
    at_limit = {"timestamps": [0.1 * i for i in range(n)], "speed": [100.0] * n}
    assert len(LapTelemetry.model_validate(at_limit).timestamps) == n
    over = {"timestamps": [0.1 * i for i in range(n + 1)], "speed": [100.0] * (n + 1)}
    with pytest.raises(ValidationError, match="at most 20000"):
        LapTelemetry.model_validate(over)


def test_json_overflow_to_infinity_is_rejected():
    body = '{"lap1_telemetry": {"timestamps": [0, 1], "speed": [100, 1e400]},' \
           ' "lap2_telemetry": {"timestamps": [0, 1], "speed": [100, 100]}}'
    with pytest.raises(ValidationError, match="finite"):
        GhostCarRequest.model_validate_json(body)


@pytest.mark.parametrize(
    "track_section",
    ["../../etc/passwd", "a/b", "$\\notacommand$", "a" * 41, "", "   ", "a#b", "a?b", "..", "mon\naco", "é"],
)
def test_unsafe_track_section_is_rejected(track_section):
    with pytest.raises(ValidationError):
        GhostCarRequest.model_validate(_request(_valid_lap(), _valid_lap(), track_section=track_section))


@pytest.mark.parametrize("track_section", ["monaco", "Silverstone", "Turn 1-3", "sector_2", "a" * 40])
def test_safe_track_section_is_accepted(track_section):
    request = GhostCarRequest.model_validate(_request(_valid_lap(), _valid_lap(), track_section=track_section))
    assert request.track_section == track_section


def test_surrounding_whitespace_is_stripped_from_track_section():
    request = GhostCarRequest.model_validate(_request(_valid_lap(), _valid_lap(), track_section=" monaco\n"))
    assert request.track_section == "monaco"


def test_unknown_alignment_and_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        GhostCarRequest.model_validate(_request(_valid_lap(), _valid_lap(), alignment="index"))
    with pytest.raises(ValidationError):
        GhostCarRequest.model_validate(_request(_valid_lap(), _valid_lap(), extra=1))


def test_stationary_lap_is_rejected():
    stopped = {"timestamps": [0.0, 1.0, 2.0], "speed": [0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="covers no distance"):
        compare_laps(_request(stopped, _valid_lap()))


@pytest.mark.parametrize("xy_scale", [10.0, 2.0, 1.15, 0.85, 0.5])
def test_xy_not_in_metres_is_rejected(xy_scale):
    # x/y in decimetres (as some feeds provide) disagree 10x with speed * time.
    with pytest.raises(ValueError, match="must be in metres"):
        compare_laps(_request(make_lap(5), make_lap(5, xy_scale=xy_scale)))


@pytest.mark.parametrize("xy_scale", [0.95, 1.05])
def test_xy_within_ten_percent_of_speed_distance_is_accepted(xy_scale):
    result = compare_laps(_request(make_lap(5), make_lap(5, xy_scale=xy_scale)))
    assert result["alignment"]["distance_source"] == "xy_path"


@pytest.mark.parametrize("jump_m", [500.0, 30.0])
def test_single_xy_glitch_is_rejected_naming_the_sample(jump_m):
    lap2 = make_lap(5)
    i = len(lap2["x"]) // 2
    lap2["x"][i] += jump_m  # the speed channel says ~10 m per sample
    with pytest.raises(ValueError, match=f"lap2: x/y moves .* between samples {i - 1} and {i} "):
        compare_laps(_request(make_lap(5), lap2))


def test_xy_position_noise_is_tolerated():
    rng = np.random.default_rng(7)
    lap1, lap2 = make_lap(5), make_lap(8)
    for lap in (lap1, lap2):
        for axis in ("x", "y"):
            lap[axis] = (np.array(lap[axis]) + rng.normal(0.0, 0.5, len(lap[axis]))).tolist()
    result = compare_laps(_request(lap1, lap2))
    assert result["alignment"]["distance_source"] == "xy_path"


@pytest.mark.parametrize(
    "lap",
    [
        {"timestamps": [0.0, 0.001], "speed": [200.0, 200.0]},  # 5.6 cm
        {"timestamps": [0.0, 1e-300], "speed": [400.0, 400.0]},
        {"timestamps": [0.0, 3000.0], "speed": [400.0, 400.0]},  # 333 km
    ],
)
def test_implausible_lap_distance_is_rejected_with_a_units_hint(lap):
    with pytest.raises(ValueError, match="check the units"):
        compare_laps(_request(lap, make_lap(5, xy=False)))


def test_loss_and_gain_below_the_sampling_floor_are_not_reported():
    # Identical driving at 1 Hz and 2 Hz: interpolation alone wobbles the
    # delta by ~0.04 s, which is not a gain or loss.
    result = compare_laps(_request(make_lap(1, xy=False), make_lap(2, xy=False)))
    summary = result["summary"]
    assert summary["loss_gain_floor_s"] == pytest.approx(0.1)
    assert summary["largest_time_loss"] is None and summary["largest_time_gain"] is None
    assert "loss_gain_floor_s" in summary["loss_gain_heuristic"]
    # At 20 Hz the floor is the 0.01 s minimum.
    fine = compare_laps(_request(make_lap(20, xy=False), make_lap(20, xy=False)))
    assert fine["summary"]["loss_gain_floor_s"] == pytest.approx(0.01)


def test_errors_are_value_errors_for_the_api_layer():
    with pytest.raises(ValueError):
        compare_laps(_request({"timestamps": []}, _valid_lap()))


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #


def _png_complete(path: Path) -> bool:
    data = path.read_bytes()
    return data.startswith(PNG_MAGIC) and data.endswith(b"IEND\xaeB`\x82") and len(data) > 1000


def test_artifact_is_written_relative_to_output_dir(tmp_path):
    output_dir = tmp_path / "nested" / "ghost"
    result = generate_ghost_comparison(
        _request(make_lap(5), make_lap(5, factor=_slowdown(500, 700, 0.8)), track_section="silverstone"),
        output_dir,
    )
    relative = result["artifact_path"]

    assert relative == Path(relative).name  # no directories, no absolute path
    assert relative.startswith("ghost_") and relative.endswith(".png")
    assert "silverstone" not in relative
    assert _png_complete(output_dir / relative)
    assert sorted(p.name for p in output_dir.iterdir()) == [relative]  # no temp files left
    json.dumps(result, allow_nan=False)  # response is strict JSON


def test_artifact_name_is_idempotent_and_collision_free(tmp_path):
    lap1, lap2 = make_lap(5), make_lap(5, factor=_slowdown(500, 700, 0.8))
    first = generate_ghost_comparison(_request(lap1, lap2, track_section="monza"), tmp_path)
    again = generate_ghost_comparison(_request(lap1, lap2, track_section="monza"), tmp_path)
    other_label = artifact_filename(_request(lap1, lap2, track_section="spa"))
    changed = dict(lap2, speed=[v + 0.001 for v in lap2["speed"]])
    other_data = artifact_filename(_request(lap1, changed, track_section="monza"))

    assert first["artifact_path"] == again["artifact_path"]
    assert len({first["artifact_path"], other_label, other_data}) == 3
    assert [p.name for p in tmp_path.iterdir()] == [first["artifact_path"]]
    # Integer and float spellings of the same number are the same normalised input.
    ints = {"timestamps": [0, 1, 2], "speed": [100, 110, 120]}
    floats = {"timestamps": [0.0, 1.0, 2.0], "speed": [100.0, 110.0, 120.0]}
    assert artifact_filename(_request(ints, ints)) == artifact_filename(_request(floats, floats))


def _digests(directory: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_concurrent_generation_is_safe(tmp_path):
    requests = [
        _request(make_lap(4), make_lap(4, factor=_slowdown(200 * k, 200 * k + 150, 0.8)), track_section=f"t{k}")
        for k in range(4)
    ]
    serial_dir, concurrent_dir = tmp_path / "serial", tmp_path / "concurrent"
    for request in requests:
        generate_ghost_comparison(request, serial_dir)
    jobs = requests * 2
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda req: generate_ghost_comparison(req, concurrent_dir), jobs))

    names = [r["artifact_path"] for r in results]
    for index, request in enumerate(jobs):
        assert names[index] == artifact_filename(request)
    assert len(set(names)) == len(requests)
    files = sorted(concurrent_dir.iterdir())
    assert [p.name for p in files] == sorted(set(names))
    assert all(_png_complete(p) for p in files)
    # Concurrent rendering produces byte-identical images to serial rendering.
    assert _digests(concurrent_dir) == _digests(serial_dir)
    # Every thread got the analysis of its own request.
    for index, request in enumerate(jobs):
        assert results[index]["track_section"] == request["track_section"]
        expected = lap_duration(_slowdown(200 * (index % 4), 200 * (index % 4) + 150, 0.8)) - lap_duration()
        assert results[index]["summary"]["final_delta_s"] == pytest.approx(expected, abs=0.02)


MINIMAL_LAP = {"timestamps": [0.0, 1.0, 2.0, 3.0], "speed": [100.0, 120.0, 140.0, 150.0]}


def _minimal_request(k=0):
    return _request(MINIMAL_LAP, dict(MINIMAL_LAP, speed=[100.0, 120.0, 140.0, 150.0 - 0.001 * k]))


def test_regenerating_replaces_the_file_atomically(tmp_path):
    request = _minimal_request()
    target = tmp_path / artifact_filename(request)
    target.write_bytes(b"stale partial file")
    with open(target, "rb") as reader:
        result = generate_ghost_comparison(request, tmp_path)
        # A reader of the old file is never exposed to a half-written new one:
        # the PNG is written elsewhere and renamed over the old name.
        assert reader.read() == b"stale partial file"
    assert result["artifact_path"] == target.name
    assert _png_complete(target)


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_old_and_excess_artifacts_are_pruned(tmp_path):
    hours = [1, 2, 3]
    older = [tmp_path / f"ghost_{k:032x}.png" for k in hours]
    for path, age in zip(older, hours):
        path.write_bytes(b"old")
        _age(path, age * 3600)
    expired = tmp_path / f"ghost_{'f' * 32}.png"
    expired.write_bytes(b"old")
    _age(expired, 2 * 24 * 3600)
    foreign = tmp_path / "notes.txt"
    foreign.write_text("not ours")
    _age(foreign, 30 * 24 * 3600)
    stale_temp, fresh_temp = tmp_path / ".ghost-crashed.png.part", tmp_path / ".ghost-inflight.png.part"
    stale_temp.write_bytes(b"")
    _age(stale_temp, 2 * 3600)
    fresh_temp.write_bytes(b"")

    by_age = generate_ghost_comparison(_minimal_request(), tmp_path, max_artifacts=10, max_artifact_age_s=2.5 * 3600)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    # Under the count limit, only PNGs older than 2.5 h go (3 h and 2 days); a
    # crashed temp file goes too; an in-flight temp file and foreign files stay.
    kept = [by_age["artifact_path"], older[0].name, older[1].name, foreign.name, fresh_temp.name]
    assert remaining == sorted(kept)

    by_count = generate_ghost_comparison(_minimal_request(1), tmp_path, max_artifacts=2)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    # Newest two ghost PNGs survive: the new one and the previous one.
    assert remaining == sorted([by_count["artifact_path"], by_age["artifact_path"], foreign.name, fresh_temp.name])


def test_distinct_requests_do_not_grow_the_directory_without_bound(tmp_path):
    for k in range(5):
        result = generate_ghost_comparison(_minimal_request(k), tmp_path, max_artifacts=2)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 2 and result["artifact_path"] in names


def test_prune_rejects_nonsense_limits(tmp_path):
    with pytest.raises(ValueError):
        ghost.prune_artifacts(tmp_path, max_files=0)
    with pytest.raises(ValueError):
        ghost.prune_artifacts(tmp_path, max_age_s=0)


def test_large_track_renders_from_a_decimated_path(tmp_path):
    n, radius = 20_000, 800.0
    stamps = np.arange(n) * 0.005
    s = np.cumsum(np.full(n, 200 / 3.6 * 0.005)) - 200 / 3.6 * 0.005
    lap = {
        "timestamps": stamps.tolist(),
        "speed": [200.0] * n,
        "x": (radius * np.cos(s / radius)).tolist(),
        "y": (radius * np.sin(s / radius)).tolist(),
    }
    result = generate_ghost_comparison(_request(lap, lap), tmp_path)
    assert result["summary"]["final_delta_s"] == 0.0
    assert _png_complete(tmp_path / result["artifact_path"])
    decimated = ghost._decimated(lap["x"])
    assert len(decimated) <= ghost.MAX_TRACK_PLOT_POINTS
    assert decimated[0] == lap["x"][0] and decimated[-1] == lap["x"][-1]


def test_rendering_never_imports_pyplot(tmp_path):
    script = (
        "import sys\n"
        "from core_modules.ghost_car.ghost_car_visualizer import generate_ghost_comparison\n"
        "lap = {'timestamps': [0.0, 1.0, 2.0], 'speed': [100.0, 120.0, 130.0],"
        " 'x': [0.0, 30.0, 63.0], 'y': [0.0, 0.0, 0.0]}\n"
        f"result = generate_ghost_comparison({{'lap1_telemetry': lap, 'lap2_telemetry': lap}}, {str(tmp_path)!r})\n"
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot imported'\n"
        "print(result['artifact_path'])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert _png_complete(tmp_path / completed.stdout.strip())


def test_chart_title_describes_the_alignment_in_words(tmp_path, monkeypatch):
    import re

    titles = []
    build = ghost._build_figure

    def capture(request, comparison):
        figure = build(request, comparison)
        titles.append(figure._suptitle.get_text())
        return figure

    monkeypatch.setattr(ghost, "_build_figure", capture)
    generate_ghost_comparison(_request(make_lap(10), make_lap(10), track_section="monza", alignment="lap_fraction"), tmp_path)
    generate_ghost_comparison(_request(make_lap(10, xy=False), make_lap(10, xy=False), alignment="distance"), tmp_path)
    assert titles[0] == "Ghost-car comparison: monza (aligned by lap fraction; distance from x/y path)"
    assert re.fullmatch(r"Ghost-car comparison \(aligned by distance; distance from integrated speed\)", titles[1]), titles[1]
