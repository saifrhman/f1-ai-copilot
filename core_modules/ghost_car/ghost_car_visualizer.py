#!/usr/bin/env python3
"""Ghost-car lap comparison: distance-aligned time delta plus a headless PNG.

Lap 1 is the reference ("ghost"), lap 2 the comparison lap. Both laps are put
on a common distance axis before anything is compared, so different sample
counts and sampling rates do not distort the result:

* Distance is the x/y path length when both laps supply x/y (in metres, checked
  segment by segment against the speed trace), otherwise speed (km/h -> m/s)
  integrated over time with the trapezoid rule.
* ``distance`` alignment compares equal metres from each lap's first sample
  over the shared range ``0..min(total1, total2)``. It assumes both traces
  start at the same track position and assumes nothing about where they end,
  so a trace that stops a sample short of the line is simply compared over a
  slightly shorter range. ``final_delta_s`` is then the gap at
  ``compared_distance_m``.
* ``lap_fraction`` alignment maps each lap onto 0..1 of its own total distance
  and reports positions in lap 1 metres. This removes speed-calibration and
  racing-line differences in total distance, but it is only correct when both
  traces run exactly from timing line to timing line: otherwise each trace is
  stretched to its own last sample and the finish is misplaced by up to one
  sample interval.
* ``auto`` therefore uses ``lap_fraction`` only when each lap's timestamps span
  its official lap time (``lap_time`` or the ``sector_times`` sum) within
  ``LINE_TO_LINE_TOLERANCE_S`` and the total distances agree within
  ``AUTO_LAP_FRACTION_TOLERANCE``; otherwise it uses ``distance``.

``delta_time_s = elapsed_lap2 - elapsed_lap1`` at equal position, so a positive
value means lap 2 is behind. Nothing is invented: optional channels that are
missing are reported and the analyses that need them are ``None``; caveats
about the alignment are listed in ``warnings``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core_modules.ghost_car.schemas import (
    AUTO_LAP_FRACTION_TOLERANCE,
    LINE_TO_LINE_TOLERANCE_S,
    AlignmentMode,
    GhostCarRequest,
    LapTelemetry,
)

MAX_GRID_POINTS = 1000
# A lap or section outside this range means the units are wrong (for example
# millisecond timestamps), not that the car drove 400 km.
MIN_LAP_DISTANCE_M = 1.0
MAX_LAP_DISTANCE_M = 30_000.0
# The x/y path length must agree with the speed-integrated distance within
# these ratios; a factor of 10 means x/y are not in metres.
XY_TO_SPEED_DISTANCE_BOUNDS = (0.9, 1.1)
# No single x/y segment may be longer than this multiple of the distance the
# speed trace covers in the same interval, plus a slack for position noise.
XY_SEGMENT_MAX_RATIO = 1.5
XY_SEGMENT_SLACK_M = 5.0
# With distance alignment, total distances differing by more than this are
# reported in ``warnings`` (calibration or racing-line drift).
DISTANCE_DRIFT_WARNING_FRACTION = 0.005
BRAKE_THRESHOLD = 0.1
FULL_THROTTLE_THRESHOLD = 0.95
RISE_TRIM_FRACTION = 0.02
# Heuristic resolution of largest_time_loss/gain: changes below
# max(MIN_LOSS_GAIN_S, LOSS_GAIN_FLOOR_PER_INTERVAL * longest sample interval)
# are within what sampling and interpolation alone produce and are not reported.
MIN_LOSS_GAIN_S = 0.01
LOSS_GAIN_FLOOR_PER_INTERVAL = 0.1
MAX_REPORTED_ZONES = 100
MAX_TRACK_PLOT_POINTS = 2000
ARTIFACT_VERSION = "ghost-comparison-v3"
DEFAULT_MAX_ARTIFACTS = 200
DEFAULT_MAX_ARTIFACT_AGE_S = 24 * 3600.0
STALE_TEMP_FILE_AGE_S = 3600.0

_ARTIFACT_NAME = re.compile(r"ghost_[0-9a-f]{32}\.png")
_TEMP_PREFIX, _TEMP_SUFFIX = ".ghost-", ".png.part"

# Matplotlib does not guarantee thread safety even for separate figures, so
# rendering is serialised. The analysis itself runs concurrently.
_RENDER_LOCK = threading.Lock()

_REFERENCE_COLOR = "#2a78d6"
_COMPARISON_COLOR = "#eb6834"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e4e3df"

RequestLike = Union[GhostCarRequest, Mapping[str, Any]]


@dataclass(frozen=True)
class _Lap:
    """One lap reduced to arrays on its own sample index."""

    telemetry: LapTelemetry
    elapsed: np.ndarray  # s since the first sample
    distance: np.ndarray  # m since the first sample (own measure)
    reference_distance: np.ndarray  # position on the comparison axis (m)
    lap_time: float
    lap_time_source: str  # "supplied", "sector_times_sum" or "telemetry_span"

    @property
    def total_distance(self) -> float:
        return float(self.distance[-1])

    @property
    def span(self) -> float:
        return float(self.elapsed[-1])

    @property
    def has_official_time(self) -> bool:
        return self.lap_time_source != "telemetry_span"

    @property
    def line_to_line(self) -> bool:
        """True when the timestamps span the official lap time (see module doc)."""

        return self.has_official_time and abs(self.span - self.lap_time) <= LINE_TO_LINE_TOLERANCE_S


@dataclass(frozen=True)
class _Comparison:
    lap1: _Lap
    lap2: _Lap
    method: AlignmentMode
    distance_source: str
    grid: np.ndarray
    delta: np.ndarray
    speed1: np.ndarray
    speed2: np.ndarray

    @property
    def distance_mismatch(self) -> float:
        total1, total2 = self.lap1.total_distance, self.lap2.total_distance
        return abs(total1 - total2) / max(total1, total2)


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #


def compare_laps(request: RequestLike) -> Dict[str, Any]:
    """Return the JSON-serialisable comparison without writing any file.

    Raises ``ValueError`` (including pydantic ``ValidationError``) for invalid
    or physically inconsistent telemetry.
    """

    validated = _validated(request)
    return _response(validated, _analyse(validated))


def generate_ghost_comparison(
    request: RequestLike,
    output_dir: Union[str, os.PathLike],
    *,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    max_artifact_age_s: float = DEFAULT_MAX_ARTIFACT_AGE_S,
) -> Dict[str, Any]:
    """Compare the laps, write the PNG atomically under ``output_dir`` and prune old PNGs.

    The file name is a hash of the validated request, so identical requests
    reuse one name and different requests never share one. The response field
    ``artifact_path`` is relative to ``output_dir``. After writing, ghost PNGs
    beyond the newest ``max_artifacts`` or older than ``max_artifact_age_s``
    are deleted (see ``prune_artifacts``), so a returned artifact stays
    available until that many newer ones have been written or it expires.
    """

    validated = _validated(request)
    comparison = _analyse(validated)
    result = _response(validated, comparison)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = artifact_filename(validated)
    _render_png(validated, comparison, directory, filename)
    prune_artifacts(directory, keep=filename, max_files=max_artifacts, max_age_s=max_artifact_age_s)
    result["artifact_path"] = filename
    return result


def artifact_filename(request: RequestLike) -> str:
    """Deterministic, user-text-free file name for a request's PNG."""

    payload = json.dumps(
        {"version": ARTIFACT_VERSION, "request": _validated(request).model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"ghost_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}.png"


def prune_artifacts(
    output_dir: Union[str, os.PathLike],
    *,
    keep: Optional[str] = None,
    max_files: int = DEFAULT_MAX_ARTIFACTS,
    max_age_s: float = DEFAULT_MAX_ARTIFACT_AGE_S,
) -> int:
    """Delete ghost PNGs beyond the newest ``max_files`` or older than ``max_age_s``.

    Only files named like ``artifact_filename`` output are considered, plus
    temporary ``.ghost-*.png.part`` files older than ``STALE_TEMP_FILE_AGE_S``
    (left by a killed process). ``keep`` is never deleted. Safe to run
    concurrently: files another thread already removed are skipped. Returns
    the number of files deleted.
    """

    if max_files < 1 or not max_age_s > 0:
        raise ValueError("max_files must be >= 1 and max_age_s must be > 0")
    now = time.time()
    artifacts: List[Tuple[float, str]] = []
    stale: List[str] = []
    with os.scandir(output_dir) as entries:
        for entry in entries:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                mtime = entry.stat(follow_symlinks=False).st_mtime
            except FileNotFoundError:
                continue
            if _ARTIFACT_NAME.fullmatch(entry.name):
                artifacts.append((mtime, entry.name))
            elif entry.name.startswith(_TEMP_PREFIX) and entry.name.endswith(_TEMP_SUFFIX):
                if now - mtime > STALE_TEMP_FILE_AGE_S:
                    stale.append(entry.name)
    artifacts.sort(reverse=True)
    doomed = stale + [
        name
        for rank, (mtime, name) in enumerate(artifacts)
        if name != keep and (rank >= max_files or now - mtime > max_age_s)
    ]
    removed = 0
    for name in doomed:
        try:
            os.unlink(os.path.join(output_dir, name))
            removed += 1
        except FileNotFoundError:
            pass
    return removed


class GhostCarVisualizer:
    """Comparison bound to one artifact directory (immutable, thread-safe)."""

    def __init__(
        self,
        output_dir: Union[str, os.PathLike],
        max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
        max_artifact_age_s: float = DEFAULT_MAX_ARTIFACT_AGE_S,
    ):
        self.output_dir = Path(output_dir)
        self.max_artifacts = max_artifacts
        self.max_artifact_age_s = max_artifact_age_s

    def generate_ghost_comparison(
        self,
        lap1_telemetry: Union[LapTelemetry, Mapping[str, Any]],
        lap2_telemetry: Union[LapTelemetry, Mapping[str, Any]],
        track_section: Optional[str] = None,
        alignment: str = AlignmentMode.AUTO.value,
    ) -> Dict[str, Any]:
        request = GhostCarRequest.model_validate(
            {
                "lap1_telemetry": lap1_telemetry,
                "lap2_telemetry": lap2_telemetry,
                "track_section": track_section,
                "alignment": alignment,
            }
        )
        return generate_ghost_comparison(
            request,
            self.output_dir,
            max_artifacts=self.max_artifacts,
            max_artifact_age_s=self.max_artifact_age_s,
        )


# --------------------------------------------------------------------------- #
# Distance and alignment
# --------------------------------------------------------------------------- #


def _validated(request: RequestLike) -> GhostCarRequest:
    if isinstance(request, GhostCarRequest):
        return request
    return GhostCarRequest.model_validate(request)


def _speed_distance(elapsed: np.ndarray, speed_kmh: np.ndarray) -> np.ndarray:
    speed_ms = speed_kmh / 3.6
    steps = 0.5 * (speed_ms[1:] + speed_ms[:-1]) * np.diff(elapsed)
    return np.concatenate(([0.0], np.cumsum(steps)))


def _xy_distance(telemetry: LapTelemetry) -> np.ndarray:
    steps = np.hypot(np.diff(np.asarray(telemetry.x)), np.diff(np.asarray(telemetry.y)))
    return np.concatenate(([0.0], np.cumsum(steps)))


def _check_total_distance(total: float, name: str) -> None:
    if total <= 0.0:
        raise ValueError(f"{name}: speed is 0 throughout, so the telemetry covers no distance")
    if not MIN_LAP_DISTANCE_M <= total <= MAX_LAP_DISTANCE_M:
        raise ValueError(
            f"{name}: speed and timestamps imply {total:.4g} m, outside the plausible "
            f"{MIN_LAP_DISTANCE_M:g} m to {MAX_LAP_DISTANCE_M / 1000:g} km for a lap or section; "
            "check the units (timestamps in seconds, not milliseconds; speed in km/h)"
        )


def _check_xy_segments(by_xy: np.ndarray, by_speed: np.ndarray, elapsed: np.ndarray, name: str) -> None:
    """Reject any x/y step the speed trace cannot explain (glitch or wrong units)."""

    xy_steps, speed_steps = np.diff(by_xy), np.diff(by_speed)
    bad = np.flatnonzero(xy_steps > XY_SEGMENT_MAX_RATIO * speed_steps + XY_SEGMENT_SLACK_M)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"{name}: x/y moves {xy_steps[i]:.1f} m between samples {i} and {i + 1} "
            f"({elapsed[i + 1] - elapsed[i]:.3f} s apart) but speed implies {speed_steps[i]:.1f} m; "
            "x/y must be in metres, time-aligned with speed and free of position glitches"
        )


def _lap_distance(telemetry: LapTelemetry, elapsed: np.ndarray, use_xy: bool, name: str) -> np.ndarray:
    by_speed = _speed_distance(elapsed, np.asarray(telemetry.speed))
    _check_total_distance(float(by_speed[-1]), name)
    if not use_xy:
        return by_speed
    by_xy = _xy_distance(telemetry)
    _check_xy_segments(by_xy, by_speed, elapsed, name)
    ratio = by_xy[-1] / by_speed[-1]
    low, high = XY_TO_SPEED_DISTANCE_BOUNDS
    if not low <= ratio <= high:
        raise ValueError(
            f"{name}: x/y path length ({by_xy[-1]:.1f}) is {ratio:.2f}x the distance implied by "
            f"speed and timestamps ({by_speed[-1]:.1f} m); x/y must be in metres and match the speed trace"
        )
    return by_xy


def _lap_time(telemetry: LapTelemetry, span: float) -> Tuple[float, str]:
    if telemetry.lap_time is not None:
        return telemetry.lap_time, "supplied"
    if telemetry.sector_times is not None:
        return sum(telemetry.sector_times), "sector_times_sum"
    return span, "telemetry_span"


def _choose_method(requested: AlignmentMode, mismatch: float, line_to_line: bool) -> AlignmentMode:
    if requested != AlignmentMode.AUTO:
        return requested
    if line_to_line and mismatch <= AUTO_LAP_FRACTION_TOLERANCE:
        return AlignmentMode.LAP_FRACTION
    return AlignmentMode.DISTANCE


def _interp(grid: np.ndarray, position: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear interpolation on a non-decreasing position axis.

    Where the car is stationary (repeated positions) a grid point at that
    position takes the first sample (first arrival), and points after it are
    interpolated from the last sample (departure), so a stop neither leaks
    into the approach nor gets smeared over the following segment.
    """

    right = np.clip(np.searchsorted(position, grid, side="left"), 1, len(position) - 1)
    left = right - 1
    start, width = position[left], position[right] - position[left]
    weight = np.divide(grid - start, width, out=np.zeros_like(grid), where=width > 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    return values[left] + weight * (values[right] - values[left])


def _analyse(request: GhostCarRequest) -> _Comparison:
    t1, t2 = request.lap1_telemetry, request.lap2_telemetry
    use_xy = t1.x is not None and t2.x is not None
    e1 = np.asarray(t1.timestamps) - t1.timestamps[0]
    e2 = np.asarray(t2.timestamps) - t2.timestamps[0]
    d1 = _lap_distance(t1, e1, use_xy, "lap1")
    d2 = _lap_distance(t2, e2, use_xy, "lap2")
    lap1 = _Lap(t1, e1, d1, d1, *_lap_time(t1, float(e1[-1])))
    lap2 = _Lap(t2, e2, d2, d2, *_lap_time(t2, float(e2[-1])))
    total1, total2 = lap1.total_distance, lap2.total_distance
    mismatch = abs(total1 - total2) / max(total1, total2)
    method = _choose_method(request.alignment, mismatch, lap1.line_to_line and lap2.line_to_line)

    if method == AlignmentMode.LAP_FRACTION:
        ref2, end = d2 * (total1 / total2), total1
        lap2 = replace(lap2, reference_distance=ref2)
    else:
        ref2, end = d2, min(total1, total2)

    grid = np.linspace(0.0, end, min(max(len(e1), len(e2)), MAX_GRID_POINTS))
    delta = _interp(grid, ref2, e2) - _interp(grid, d1, e1)
    speed1 = _interp(grid, d1, np.asarray(t1.speed))
    speed2 = _interp(grid, ref2, np.asarray(t2.speed))
    return _Comparison(lap1, lap2, method, "xy_path" if use_xy else "speed_integration", grid, delta, speed1, speed2)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #


def _rounded(values: Any, digits: int) -> Any:
    rounded = np.round(np.asarray(values, dtype=float), digits) + 0.0  # + 0.0 drops "-0.0"
    return rounded.tolist()


def _loss_gain_floor(comparison: _Comparison) -> float:
    longest = max(float(np.max(np.diff(lap.elapsed))) for lap in (comparison.lap1, comparison.lap2))
    return max(MIN_LOSS_GAIN_S, LOSS_GAIN_FLOOR_PER_INTERVAL * longest)


def _largest_rise(grid: np.ndarray, values: np.ndarray, floor: float) -> Optional[Dict[str, float]]:
    """Largest increase of ``values`` over any interval (heuristic location).

    ``time_s`` is the full increase; increases below ``floor`` return None.
    ``from_m``/``to_m`` are trimmed to the stretch in which all but
    ``2 * RISE_TRIM_FRACTION`` of it happens, so tiny wobbles far before or
    after the real loss do not stretch the interval.
    """

    rises = values - np.minimum.accumulate(values)
    end = int(np.argmax(rises))
    rise = float(rises[end])
    if rise < floor:
        return None
    start = int(np.argmin(values[: end + 1]))
    tolerance = RISE_TRIM_FRACTION * rise
    core_start = start + int(np.flatnonzero(values[start : end + 1] <= values[start] + tolerance)[-1])
    core_end = core_start + int(np.flatnonzero(values[core_start : end + 1] >= values[end] - tolerance)[0])
    return {
        "time_s": _rounded(rise, 4),
        "from_m": _rounded(grid[core_start], 2),
        "to_m": _rounded(grid[core_end], 2),
    }


def _summary(comparison: _Comparison) -> Dict[str, Any]:
    grid, delta = comparison.grid, comparison.delta
    i_max, i_min = int(np.argmax(delta)), int(np.argmin(delta))
    floor = _loss_gain_floor(comparison)
    return {
        "final_delta_s": _rounded(delta[-1], 4),
        "final_delta_at_m": _rounded(grid[-1], 2),
        "max_delta_s": _rounded(delta[i_max], 4),
        "max_delta_at_m": _rounded(grid[i_max], 2),
        "min_delta_s": _rounded(delta[i_min], 4),
        "min_delta_at_m": _rounded(grid[i_min], 2),
        "largest_time_loss": _largest_rise(grid, delta, floor),
        "largest_time_gain": _largest_rise(grid, -delta, floor),
        "loss_gain_floor_s": _rounded(floor, 4),
        "loss_gain_heuristic": (
            "largest_time_loss/gain.time_s is the largest continuous change of delta_time_s for lap 2; "
            f"from_m/to_m bound the stretch holding all but {2 * RISE_TRIM_FRACTION:.0%} of that change. "
            f"Changes below loss_gain_floor_s ({LOSS_GAIN_FLOOR_PER_INTERVAL:g} x the longest sample "
            f"interval, at least {MIN_LOSS_GAIN_S:g} s) are within sampling and interpolation error "
            "and are reported as null."
        ),
    }


def _zone_bounds(mask: np.ndarray, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Runs of consecutive samples where ``mask`` is true, as (start_m, end_m) arrays.

    A zone runs from its first to its last true sample, so it is resolved to
    the sample spacing.
    """

    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return position[np.flatnonzero(edges == 1)], position[np.flatnonzero(edges == -1) - 1]


def _zone_list(starts: np.ndarray, ends: np.ndarray) -> List[Dict[str, float]]:
    starts, ends = _rounded(starts[:MAX_REPORTED_ZONES], 2), _rounded(ends[:MAX_REPORTED_ZONES], 2)
    return [{"start_m": a, "end_m": b} for a, b in zip(starts, ends)]


def _pair_zones(
    starts1: np.ndarray, ends1: np.ndarray, starts2: np.ndarray, ends2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """One-to-one pairing of overlapping zones, greatest overlap first (heuristic).

    Zones of one lap are ordered and disjoint, so the lap-2 zones overlapping
    lap-1 zone ``i`` are one contiguous run ``lo[i]..hi[i]`` found by binary
    search; there are at most ``n1 + n2 - 1`` such pairs. Ties prefer the
    closer start. Returns index arrays into both laps, ordered along lap 1.
    """

    lo = np.searchsorted(ends2, starts1, side="left")
    hi = np.searchsorted(starts2, ends1, side="right")
    counts = np.maximum(hi - lo, 0)
    first = np.repeat(np.arange(len(starts1)), counts)
    offsets = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    second = np.repeat(lo, counts) + offsets
    overlap = np.minimum(ends1[first], ends2[second]) - np.maximum(starts1[first], starts2[second])
    order = np.lexsort((first, np.abs(starts2[second] - starts1[first]), -overlap))
    used1, used2, chosen = set(), set(), []
    first_list, second_list = first.tolist(), second.tolist()
    for k in order.tolist():
        a, b = first_list[k], second_list[k]
        if a not in used1 and b not in used2:
            used1.add(a)
            used2.add(b)
            chosen.append(k)
    chosen_index = np.asarray(sorted(chosen, key=first_list.__getitem__), dtype=int)
    return first[chosen_index], second[chosen_index]


def _braking(comparison: _Comparison) -> Optional[Dict[str, Any]]:
    lap1, lap2 = comparison.lap1, comparison.lap2
    if lap1.telemetry.brake is None or lap2.telemetry.brake is None:
        return None
    starts1, ends1 = _zone_bounds(np.asarray(lap1.telemetry.brake) >= BRAKE_THRESHOLD, lap1.reference_distance)
    starts2, ends2 = _zone_bounds(np.asarray(lap2.telemetry.brake) >= BRAKE_THRESHOLD, lap2.reference_distance)
    i1, i2 = _pair_zones(starts1, ends1, starts2, ends2)
    shown1, shown2 = i1[:MAX_REPORTED_ZONES], i2[:MAX_REPORTED_ZONES]
    columns = {
        "lap1_start_m": starts1[shown1],
        "lap2_start_m": starts2[shown2],
        "brake_point_delta_m": starts2[shown2] - starts1[shown1],
        "lap1_end_m": ends1[shown1],
        "lap2_end_m": ends2[shown2],
    }
    rounded = {key: _rounded(values, 2) for key, values in columns.items()}
    matched = [dict(zip(rounded, row)) for row in zip(*rounded.values())]
    counts = (len(starts1), len(starts2), len(i1))
    return {
        "heuristic": (
            f"A braking zone is a run of samples with brake >= {BRAKE_THRESHOLD}; overlapping zones are "
            "paired one to one, greatest overlap first. brake_point_delta_m > 0 means lap 2 starts "
            "braking later. Zone edges are resolved to the sample spacing."
        ),
        "threshold": BRAKE_THRESHOLD,
        "lap1_zone_count": counts[0],
        "lap2_zone_count": counts[1],
        "matched_zone_count": counts[2],
        "zones_truncated": max(counts) > MAX_REPORTED_ZONES,
        "lap1_zones": _zone_list(starts1, ends1),
        "lap2_zones": _zone_list(starts2, ends2),
        "matched_zones": matched,
    }


def _drs(comparison: _Comparison) -> Optional[Dict[str, Any]]:
    lap1, lap2 = comparison.lap1, comparison.lap2
    if lap1.telemetry.drs is None or lap2.telemetry.drs is None:
        return None
    result: Dict[str, Any] = {
        "resolution": "zones run from the first to the last sample with DRS open",
    }
    truncated = False
    for key, lap in (("lap1", lap1), ("lap2", lap2)):
        starts, ends = _zone_bounds(np.asarray(lap.telemetry.drs, dtype=bool), lap.reference_distance)
        result[f"{key}_zone_count"] = len(starts)
        result[f"{key}_zones"] = _zone_list(starts, ends)
        result[f"{key}_open_distance_m"] = _rounded(float(np.sum(ends - starts)), 2)
        truncated = truncated or len(starts) > MAX_REPORTED_ZONES
    result["zones_truncated"] = truncated
    return result


def _full_throttle_fraction(lap: _Lap) -> float:
    segments = np.diff(lap.distance)
    at_full = np.asarray(lap.telemetry.throttle[:-1]) >= FULL_THROTTLE_THRESHOLD
    return float(segments[at_full].sum() / lap.total_distance)


def _throttle(comparison: _Comparison) -> Optional[Dict[str, Any]]:
    lap1, lap2 = comparison.lap1, comparison.lap2
    if lap1.telemetry.throttle is None or lap2.telemetry.throttle is None:
        return None
    return {
        "heuristic": (
            f"Share of each lap's distance driven with throttle >= {FULL_THROTTLE_THRESHOLD}, "
            "holding each sample until the next."
        ),
        "threshold": FULL_THROTTLE_THRESHOLD,
        "lap1_full_throttle_fraction": _rounded(_full_throttle_fraction(lap1), 4),
        "lap2_full_throttle_fraction": _rounded(_full_throttle_fraction(lap2), 4),
    }


def _gear(comparison: _Comparison) -> Optional[Dict[str, Any]]:
    gears1, gears2 = comparison.lap1.telemetry.gear, comparison.lap2.telemetry.gear
    if gears1 is None or gears2 is None:
        return None
    result: Dict[str, Any] = {}
    for key, gears in (("lap1", gears1), ("lap2", gears2)):
        result[f"{key}_max_gear"] = max(gears)
        result[f"{key}_gear_changes"] = int(np.count_nonzero(np.diff(gears)))
    return result


def _lap_info(lap: _Lap) -> Dict[str, Any]:
    return {
        "lap_number": lap.telemetry.lap_number,
        "samples": len(lap.elapsed),
        "telemetry_duration_s": _rounded(lap.span, 4),
        "distance_m": _rounded(lap.total_distance, 2),
        "lap_time_s": _rounded(lap.lap_time, 4),
        "lap_time_source": lap.lap_time_source,
        "sector_times_s": lap.telemetry.sector_times,
    }


def _alignment_info(request: GhostCarRequest, comparison: _Comparison) -> Dict[str, Any]:
    end = float(comparison.grid[-1])
    if comparison.method == AlignmentMode.LAP_FRACTION:
        description = (
            "Each lap is mapped onto 0..1 of its own distance; positions are lap 1 metres. "
            "Assumes both traces run from timing line to timing line."
        )
    else:
        description = (
            f"Laps are compared at equal metres from their first sample over the shared range "
            f"0..{end:.1f} m. Assumes both traces start at the same track position."
        )
    return {
        "method": comparison.method.value,
        "requested": request.alignment.value,
        "distance_source": comparison.distance_source,
        "lap1_distance_m": _rounded(comparison.lap1.total_distance, 2),
        "lap2_distance_m": _rounded(comparison.lap2.total_distance, 2),
        "distance_mismatch_fraction": _rounded(comparison.distance_mismatch, 5),
        "line_to_line": {"lap1": comparison.lap1.line_to_line, "lap2": comparison.lap2.line_to_line},
        "compared_distance_m": _rounded(end, 2),
        "grid_points": len(comparison.grid),
        "description": description,
    }


def _span_warning(key: str, lap: _Lap) -> Optional[str]:
    gap = lap.span - lap.lap_time
    if not lap.has_official_time or abs(gap) <= LINE_TO_LINE_TOLERANCE_S:
        return None
    if gap > 0:
        return (
            f"{key}: telemetry spans {lap.span:.3f} s, {gap:.3f} s more than its lap time of "
            f"{lap.lap_time:.3f} s, so it includes samples outside the lap; if they precede the start "
            "line, the traces do not start at the same track position and delta_time_s is offset"
        )
    return (
        f"{key}: telemetry spans {lap.span:.3f} s, {-gap:.3f} s less than its lap time of "
        f"{lap.lap_time:.3f} s, so it does not run from timing line to timing line; final_delta_s "
        "is the gap at compared_distance_m, not at the finish line"
    )


def _warnings(comparison: _Comparison) -> List[str]:
    laps = (("lap1", comparison.lap1), ("lap2", comparison.lap2))
    notes = [note for note in (_span_warning(key, lap) for key, lap in laps) if note]
    without_time = [key for key, lap in laps if not lap.has_official_time]
    if without_time:
        notes.append(
            f"lap_time_delta_s is null: {' and '.join(without_time)} supplied no lap_time or "
            "sector_times, and a telemetry span misses up to one sample interval at each end"
        )
    if comparison.method == AlignmentMode.LAP_FRACTION and not (
        comparison.lap1.line_to_line and comparison.lap2.line_to_line
    ):
        notes.append(
            "lap_fraction alignment was requested, but not both traces are shown to run from timing "
            f"line to timing line (timestamps spanning lap_time or sector_times within "
            f"{LINE_TO_LINE_TOLERANCE_S} s); each trace is stretched to its own last sample, which "
            "can misplace the finish by up to one sample interval"
        )
    mismatch = comparison.distance_mismatch
    if comparison.method == AlignmentMode.DISTANCE and mismatch > DISTANCE_DRIFT_WARNING_FRACTION:
        notes.append(
            f"total distances differ by {mismatch:.1%}; only the shared "
            f"0..{float(comparison.grid[-1]):.1f} m are compared. If both traces cover the same "
            "stretch of track, the difference comes from speed calibration or the racing line and "
            "shows up as a steady drift in delta_time_s"
        )
    return notes


def _response(request: GhostCarRequest, comparison: _Comparison) -> Dict[str, Any]:
    t1, t2 = request.lap1_telemetry, request.lap2_telemetry
    lap1, lap2 = comparison.lap1, comparison.lap2
    sector_deltas = None
    if t1.sector_times is not None and t2.sector_times is not None:
        sector_deltas = _rounded(np.subtract(t2.sector_times, t1.sector_times), 4)
    lap_time_delta = None
    if lap1.has_official_time and lap2.has_official_time:
        lap_time_delta = _rounded(lap2.lap_time - lap1.lap_time, 4)
    return {
        "track_section": request.track_section,
        "delta_convention": "delta_time_s = lap 2 elapsed - lap 1 elapsed at equal position; > 0 means lap 2 is behind",
        "laps": {"lap1": _lap_info(lap1), "lap2": _lap_info(lap2)},
        "lap_time_delta_s": lap_time_delta,
        "sector_time_deltas_s": sector_deltas,
        "missing_channels": {"lap1": t1.missing_channels(), "lap2": t2.missing_channels()},
        "alignment": _alignment_info(request, comparison),
        "warnings": _warnings(comparison),
        "traces": {
            "distance_m": _rounded(comparison.grid, 2),
            "delta_time_s": _rounded(comparison.delta, 4),
            "lap1_speed_kmh": _rounded(comparison.speed1, 2),
            "lap2_speed_kmh": _rounded(comparison.speed2, 2),
            "speed_delta_kmh": _rounded(comparison.speed2 - comparison.speed1, 2),
        },
        "summary": _summary(comparison),
        "braking": _braking(comparison),
        "drs": _drs(comparison),
        "throttle": _throttle(comparison),
        "gear": _gear(comparison),
    }


# --------------------------------------------------------------------------- #
# Rendering (object-oriented Matplotlib + Agg; no pyplot global state)
# --------------------------------------------------------------------------- #


def _label(base: str, telemetry: LapTelemetry) -> str:
    return base if telemetry.lap_number is None else f"{base} (lap {telemetry.lap_number})"


def _decimated(values: List[float], limit: int = MAX_TRACK_PLOT_POINTS) -> np.ndarray:
    """At most ``limit`` evenly spaced points, always keeping the first and last."""

    array = np.asarray(values)
    if len(array) <= limit:
        return array
    return array[np.unique(np.linspace(0, len(array) - 1, limit).round().astype(int))]


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(_SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=_INK)
    ax.set_xlabel(xlabel, color=_INK_SECONDARY)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)


def _plot_track(ax: Axes, request: GhostCarRequest, labels: Tuple[str, str]) -> None:
    for telemetry, color, label in (
        (request.lap1_telemetry, _REFERENCE_COLOR, labels[0]),
        (request.lap2_telemetry, _COMPARISON_COLOR, labels[1]),
    ):
        ax.plot(_decimated(telemetry.x), _decimated(telemetry.y), color=color, linewidth=1.5, label=label)
    start = request.lap1_telemetry
    ax.plot([start.x[0]], [start.y[0]], "o", color=_INK, markersize=5, label="Start (lap 1)")
    ax.set_aspect("equal", adjustable="datalim")
    _style(ax, "Driven path", "x (m)", "y (m)")
    ax.legend(frameon=False, fontsize=9)


def _build_figure(request: GhostCarRequest, comparison: _Comparison) -> Figure:
    labels = (_label("Reference", request.lap1_telemetry), _label("Comparison", request.lap2_telemetry))
    with_track = comparison.distance_source == "xy_path"
    rows = 3 if with_track else 2
    fig = Figure(figsize=(10, 3.4 * rows), facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, 1)
    if with_track:
        _plot_track(axes[0], request, labels)
    xlabel = (
        "Distance along reference lap (m)"
        if comparison.method == AlignmentMode.LAP_FRACTION
        else "Distance from first sample (m)"
    )
    speed_ax, delta_ax = axes[-2], axes[-1]
    speed_ax.plot(comparison.grid, comparison.speed1, color=_REFERENCE_COLOR, linewidth=1.5, label=labels[0])
    speed_ax.plot(comparison.grid, comparison.speed2, color=_COMPARISON_COLOR, linewidth=1.5, label=labels[1])
    _style(speed_ax, "Speed", xlabel, "km/h")
    speed_ax.legend(frameon=False, fontsize=9)

    delta_ax.axhline(0.0, color=_INK_SECONDARY, linewidth=0.8)
    delta_ax.plot(comparison.grid, comparison.delta, color=_INK, linewidth=1.5)
    _style(delta_ax, "Gap of comparison lap to reference (s); above 0 = behind", xlabel, "s")

    heading = "Ghost-car comparison"
    if request.track_section:
        heading += f": {request.track_section}"
    aligned = "by lap fraction" if comparison.method == AlignmentMode.LAP_FRACTION else "by distance"
    distance = "x/y path" if comparison.distance_source == "xy_path" else "integrated speed"
    fig.suptitle(f"{heading} (aligned {aligned}; distance from {distance})", color=_INK)
    return fig


def _render_png(request: GhostCarRequest, comparison: _Comparison, directory: Path, filename: str) -> None:
    """Render to a temporary file in ``directory`` and atomically rename it."""

    handle, temp_name = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=directory)
    try:
        with os.fdopen(handle, "wb") as stream:
            with _RENDER_LOCK:
                _build_figure(request, comparison).savefig(stream, format="png", dpi=110)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, directory / filename)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
