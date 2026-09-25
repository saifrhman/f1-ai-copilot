"""Estimate the strategy engine's tyre parameters from real lap history.

The strategy engine (``strategy_engine.py``) projects every lap as

    lap time = base lap time / (performance(k) x weather factor x hot-track factor)

where k is the tyre's own lap (1 = first lap on the set), the driver and damage multipliers
are folded into the base lap time (they scale every lap equally), and

    performance(k) = base_performance x (0.90 + 0.10 k / warm_up_laps)            for k <= warm_up_laps
    performance(k) = base_performance - degradation_rate x max(0, k - window_end) otherwise

(``window_end`` is the end of ``peak_performance_window``; its start does not change lap times).
Taking the reciprocal gives

    1 / lap time = a x g_W(k) - b x h_WE(k)

with ``a = base_performance x factor / base lap time`` and ``b = degradation_rate x factor /
base lap time``. For a fixed warm-up length W and window end E this is LINEAR in (a, b), so
``estimate_tire_parameters`` fits each compound as follows:

* laps flagged ``pit_in``, ``pit_out`` or ``safety_car`` are excluded, and an optional fuel
  correction (seconds per lap of fuel burnt, supplied by the caller) is applied;
* for every admissible (W, E) the two coefficients are solved by weighted least squares on
  1 / lap time; the weights (lap time squared) make each residual approximately a lap-time
  residual in seconds. A flat model (b = 0, no degradation) is always a candidate, and b is
  constrained to be >= 0 (the engine cannot represent tyres that get faster with age). Window
  ends run from the youngest observed tyre age, plus ends inside the warm-up (E < W, which the
  engine accepts) when laps inside the ramp pin the peak pace;
* the structure with the lowest Bayesian information criterion is kept, counting a detected
  window end and a detected warm-up as one parameter each, so a breakpoint or warm-up is only
  reported when it improves the fit by more than chance would;
* outliers are rejected iteratively: a lap whose residual is more than 3.5 robust standard
  deviations from the median residual is dropped (never a lap within 0.1 s). The robust standard
  deviation is 1.4826 x the median absolute deviation of all clean laps' residuals, with
  small-sample and fitted-parameter corrections. Each round drops the largest deviations first,
  re-admits laps the new fit explains and re-selects the structure, scoring every candidate on
  all clean laps with residuals capped at the threshold, so a few gross outliers cannot hide a
  real warm-up or degradation phase. It stops when neither the lap set nor the structure changes;
* a compound needs at least 5 clean laps, a degradation rate needs at least 3 distinct tyre ages
  past the window end, and a warm-up of W laps needs at least 2 clean laps younger than W and
  2 at least W laps old; otherwise the compound is ``insufficient_data`` or the feature is
  reported as not detected. Nothing is guessed;
* a fit that one tyre model evidently does not describe is not reported as an estimate: the
  compound is ``insufficient_data`` when fewer than 5 laps survive outlier rejection, when more
  than 25% of its clean laps are rejected (for example two pace levels mixed), or when the
  residual standard deviation exceeds 3% of the peak lap time (the outlier scale itself broke
  down). Notes flag the softer signs: several rejected laps faster than the fit (a possible
  faster population), slower laps rejected beyond the oldest tyre age used (degradation or a
  cliff the model could not fit), a window end at the youngest observed age (peak pace not
  identifiable) and a residual standard deviation above 1% of the peak lap time.

``base_performance`` is expressed relative to the reference compound after the engine's weather
and hot-track factors are divided out: the fastest compound whose own fitted performance stays
above the engine's 0.20 floor (it gets exactly 1.0, the engine convention, and always has
``tire_data``). ``estimated_base_lap_time`` is its peak lap time with the same factors removed.
Everything is deterministic: no randomness, compounds are processed in a fixed order and ties are
broken explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .strategy_engine import (
    MAX_DEGRADATION_RATE,
    MAX_LAP_TIME_S,
    MAX_PIT_STOP_DELTA_S,
    MAX_TIRE_AGE_LAPS,
    MAX_TOTAL_LAPS,
    MAX_TRACK_TEMPERATURE_C,
    MAX_WARM_UP_LAPS,
    MIN_BASE_PERFORMANCE,
    MIN_LAP_TIME_S,
    MIN_TRACK_TEMPERATURE_C,
    PERFORMANCE_FLOOR,
    TireCompound,
    TireData,
    WeatherCondition,
    _enum,
    _integer,
    _number,
    _temperature_multiplier,
    _validate_tire,
    _weather_multiplier,
)

# Input limits.
MAX_CALIBRATION_LAPS = 2000
MAX_FUEL_CORRECTION_S_PER_LAP = 0.5

# Minimum data requirements.
MIN_CLEAN_LAPS = 5  # per compound, after flagged laps and outliers are removed
MIN_DEGRADATION_AGES = 3  # distinct tyre ages past the window end needed to fit a degradation rate
MIN_WARM_UP_SUPPORT_LAPS = 2  # laps younger than W, and laps at least W old, needed to test warm-up W
# More than this fraction of a compound's clean laps rejected as outliers: one tyre model does not
# describe the laps (for example two pace levels mixed), so the compound is insufficient_data. With
# normal noise the 3.5-sigma rule rejects about 0.05% of laps, so a quarter is far beyond chance.
MAX_OUTLIER_FRACTION = 0.25
# A residual standard deviation above this fraction of the peak lap time (2.4 s at 80 s, three times
# the "Poor fit" note level) means no single tyre model describes the laps, so the compound is
# insufficient_data. This catches what the outlier rules cannot: when half or more of the laps are
# junk, or two pace levels are mixed about evenly, the median-based noise scale is itself inflated,
# so nothing is rejected and the fit lands between the pace levels. Many grossly slow laps (tens to
# hundreds of seconds off) can have the same effect even as a minority: the lap-time-squared weights
# give them extra pull on the first fit, which can derail the rejection.
MAX_RESIDUAL_STD_FRACTION = 0.03
# This many rejected laps FASTER than the fit earn a "mixed pace" note: genuine outliers (traffic,
# mistakes, incidents) are slow, so several fast ones suggest a faster population the fit left out.
MIXED_PACE_FAST_LAPS = 3

# Robust fitting constants.
OUTLIER_THRESHOLD = 3.5  # robust z-score (Iglewicz-Hoaglin modified z-score cut-off)
MAD_TO_SIGMA = 1.4826  # median absolute deviation -> standard deviation for normal noise
MIN_OUTLIER_THRESHOLD_S = 0.1  # residuals this small are never called outliers (timing noise)
TIMING_RESOLUTION_S = 0.001  # lower bound on the residual scale used by the selection criterion
MAX_OUTLIER_ITERATIONS = 50
CONCENTRATION_STEPS = 2  # per-candidate refits on the laps within the rejection threshold
POOR_FIT_RESIDUAL_FRACTION = 0.01  # residual std above 1% of the peak lap time earns a warning note

STATUS_ESTIMATED = "estimated"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_OUTSIDE_ENGINE_LIMITS = "outside_engine_limits"

WARM_UP_DETECTED = "detected"
WARM_UP_NOT_DETECTED = "not_detected"
WINDOW_END_DETECTED = "detected"
WINDOW_END_NO_DEGRADATION = "no_degradation_observed"
SOURCE_SUPPLIED = "supplied"

OUTLIER_REASON = "outlier"
_COMPOUND_ORDER: Tuple[TireCompound, ...] = tuple(TireCompound)  # tie-break order (soft -> wet)
LAP_FLAGS: Tuple[str, ...] = ("pit_out", "pit_in", "safety_car")
LAP_KEYS: Tuple[str, ...] = ("compound", "tire_age", "lap_time", "race_lap", *LAP_FLAGS)
_REQUIRED_LAP_KEYS: Tuple[str, ...] = ("compound", "tire_age", "lap_time")


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LapRecord:
    """One timed lap. tire_age is the lap number on the set (1 = first lap on it)."""

    compound: TireCompound
    tire_age: int
    lap_time: float  # seconds
    race_lap: Optional[int] = None  # race lap number; required only for a fuel correction
    pit_out: bool = False  # excluded: out-lap from the pit lane
    pit_in: bool = False  # excluded: in-lap to the pit lane
    safety_car: bool = False  # excluded: any safety-car / VSC / red-flag lap


@dataclass(frozen=True)
class ExcludedLap:
    """A lap that did not contribute to any fit, with the reason."""

    index: int  # position in the supplied laps
    compound: TireCompound
    tire_age: int
    lap_time: float
    reason: str  # flag names joined by ", ", or "outlier"
    # Outliers only, in the same frame as lap_time (raw: any fuel correction is added back), so
    # residual_s == lap_time - fitted_lap_time.
    fitted_lap_time: Optional[float] = None
    residual_s: Optional[float] = None


@dataclass
class CompoundCalibration:
    """Per-compound estimate and fit statistics (seconds; statistics over the laps used)."""

    compound: TireCompound
    status: str  # STATUS_ESTIMATED, STATUS_INSUFFICIENT_DATA or STATUS_OUTSIDE_ENGINE_LIMITS
    reason: Optional[str]
    laps_supplied: int
    laps_flagged: int  # excluded by pit_in / pit_out / safety_car
    clean_laps: int  # laps_supplied - laps_flagged
    laps_used: int  # clean laps kept after outlier rejection (0 when not fitted)
    outliers_rejected: int
    tire_data: Optional[TireData] = None  # only when status == "estimated"
    warm_up_source: Optional[str] = None
    peak_window_end_source: Optional[str] = None
    degradation_observed: bool = False
    # Fitted lap time at peak performance under the history's conditions (at the fuel load of
    # fuel_reference_lap when a fuel correction is applied).
    peak_lap_time: Optional[float] = None
    initial_degradation_s_per_lap: Optional[float] = None  # lap time one lap into degradation minus peak
    tire_age_range: Optional[Tuple[int, int]] = None  # youngest/oldest tyre age among the laps used
    r_squared: Optional[float] = None  # None when every lap used has the same time
    residual_std_s: Optional[float] = None
    outlier_iterations: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class TyreCalibrationResult:
    compounds: Dict[TireCompound, CompoundCalibration]  # every compound present in the laps
    tire_data: Dict[TireCompound, TireData]  # compounds with status "estimated", ready for the engine
    estimated_base_lap_time: Optional[float]  # lap time at performance 1.0 (factors removed)
    # The fastest compound whose fit is within the engine's limits; it has base_performance 1.0 and
    # always has tire_data. None when no compound could be estimated.
    reference_compound: Optional[TireCompound]
    excluded_laps: List[ExcludedLap]
    laps_supplied: int
    weather: WeatherCondition
    track_temperature: float
    pit_stop_delta: float
    fuel_correction_s_per_lap: float
    fuel_reference_lap: Optional[int]  # lap times are corrected to this race lap's fuel load
    assumptions: List[str]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _flag(name: str, value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _validate_lap(index: int, lap: Any) -> LapRecord:
    name = f"laps[{index}]"
    if isinstance(lap, LapRecord):
        values = {item.name: getattr(lap, item.name) for item in fields(LapRecord)}
    elif isinstance(lap, Mapping):
        unknown = sorted(str(key) for key in lap if key not in LAP_KEYS)
        if unknown:
            raise ValueError(f"Unknown {name} keys {unknown}; allowed: {list(LAP_KEYS)}")
        missing = [key for key in _REQUIRED_LAP_KEYS if lap.get(key) is None]
        if missing:
            raise ValueError(f"{name} is missing {missing}")
        values = dict(lap)
    else:
        raise ValueError(f"{name} must be a LapRecord or a mapping, got {type(lap).__name__}")
    race_lap = values.get("race_lap")
    return LapRecord(
        compound=_enum(TireCompound, f"{name}.compound", values["compound"]),
        tire_age=_integer(f"{name}.tire_age", values["tire_age"], 1, MAX_TIRE_AGE_LAPS),
        lap_time=_number(f"{name}.lap_time", values["lap_time"], MIN_LAP_TIME_S, MAX_LAP_TIME_S),
        race_lap=None if race_lap is None else _integer(f"{name}.race_lap", race_lap, 1, MAX_TOTAL_LAPS),
        **{flag: _flag(f"{name}.{flag}", values.get(flag, False)) for flag in LAP_FLAGS},
    )


def _validate_laps(laps: Any) -> List[LapRecord]:
    if isinstance(laps, (str, bytes)) or not isinstance(laps, Sequence):
        raise ValueError("laps must be a list of laps")
    if not 1 <= len(laps) <= MAX_CALIBRATION_LAPS:
        raise ValueError(f"laps must contain 1 to {MAX_CALIBRATION_LAPS} laps, got {len(laps)}")
    return [_validate_lap(index, lap) for index, lap in enumerate(laps)]


def _validate_overrides(
    name: str, overrides: Any, low: int, high: int, present: Sequence[TireCompound]
) -> Dict[TireCompound, int]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{name} must be a mapping of compound -> laps")
    clean: Dict[TireCompound, int] = {}
    for key, value in overrides.items():
        compound = _enum(TireCompound, f"{name} key", key)
        if compound in clean:
            raise ValueError(f"{name} contains compound '{compound.value}' more than once")
        if compound not in present:
            raise ValueError(f"{name} is given for '{compound.value}', which has no laps")
        clean[compound] = _integer(f"{name}[{compound.value}]", value, low, high)
    return clean


# ---------------------------------------------------------------------------
# Model (mirrors strategy_engine._performance_curve; checked by the round-trip tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Fit:
    warm_up_laps: int
    peak_end: int
    a: float  # peak performance x factor / base lap time (1 / peak lap time)
    b: float  # degradation per lap x factor / base lap time; 0 for a flat fit
    criterion: float  # information criterion used for the structure choice
    free_parameters: int
    linear_parameters: int


@dataclass(frozen=True, eq=False)  # holds numpy arrays: identity comparison only
class _CompoundFit:
    """One compound's final fit and the clean laps it was made on."""

    fit: _Fit
    indices: np.ndarray  # positions of the compound's clean laps in the supplied laps
    ages: np.ndarray
    times: np.ndarray  # lap times after any fuel correction
    inliers: np.ndarray  # the laps used (not rejected as outliers)
    fitted: np.ndarray  # fitted lap time of every clean lap, same frame as times; +inf where performance <= 0
    factor: float  # the engine's weather x hot-track factor for the history's conditions

    @property
    def base_time(self) -> float:
        """Peak lap time with the conditions factor removed (the lap time at performance 1.0)."""

        return self.factor / self.fit.a

    def lowest_performance(self, base_lap_time: float) -> float:
        """Lowest engine performance (conditions factor included) over the laps used, for this base lap time."""

        return float(np.min(base_lap_time / self.fitted[self.inliers]))


def _warm_up_factor(ages: np.ndarray, warm_up: int) -> np.ndarray:
    factor = np.ones(len(ages))
    if warm_up > 0:
        warming = ages <= warm_up
        factor[warming] = 0.90 + 0.10 * ages[warming] / warm_up
    return factor


def _laps_past_window(ages: np.ndarray, warm_up: int, peak_end: Any) -> np.ndarray:
    """max(0, age - window end) per lap (one column per window end when peak_end is an array)."""

    ends = np.asarray(peak_end, dtype=float)
    laps = np.maximum(0.0, ages[:, None] - ends[None, :]) if ends.ndim else np.maximum(0.0, ages - ends)
    if warm_up > 0:
        laps[ages <= warm_up] = 0.0  # the engine applies the warm-up ramp instead on these laps
    return laps


def _predict(fit: _Fit, ages: np.ndarray) -> np.ndarray:
    """Fitted lap times; +inf where the fitted performance is not positive."""

    inverse = fit.a * _warm_up_factor(ages, fit.warm_up_laps) - fit.b * _laps_past_window(
        ages, fit.warm_up_laps, fit.peak_end
    )
    positive = inverse > 0.0
    return np.where(positive, 1.0 / np.where(positive, inverse, 1.0), np.inf)


def _criterion(loss: float, laps: int, parameters: int, cap: float) -> float:
    """Bayesian information criterion for Gaussian lap-time noise.

    First round (no cap yet): noise variance unknown, n ln(SSE / n) + k ln n. Later rounds: the noise
    scale is known from the previous round (sigma = cap / OUTLIER_THRESHOLD), so loss / sigma^2 + k ln n.
    The known-scale form matters: every candidate pays the same capped cost for a gross outlier, and
    in the log form that shared constant would dilute real differences between structures.
    """

    if math.isinf(cap):
        return laps * math.log(max(loss / laps, TIMING_RESOLUTION_S**2)) + parameters * math.log(laps)
    sigma = cap / OUTLIER_THRESHOLD
    return loss / (sigma * sigma) + parameters * math.log(laps)


def _distinct_ages_past(ages: np.ndarray, warm_up: int, peak_end: int) -> int:
    return int(np.unique(ages[(ages > peak_end) & (ages > warm_up)]).size)


def _warm_up_options(ages: np.ndarray) -> List[int]:
    # W = 1 gives exactly the same lap times as W = 0 in the engine, so it is not a separate option.
    # W may exceed the window end: the engine accepts that (see _window_ends).
    options = [0]
    for warm_up in range(2, MAX_WARM_UP_LAPS + 1):
        young = int(np.count_nonzero(ages < warm_up))
        if young >= MIN_WARM_UP_SUPPORT_LAPS and len(ages) - young >= MIN_WARM_UP_SUPPORT_LAPS:
            options.append(warm_up)
    return options


def _window_ends(warm_up: int, min_age: int, max_age: int) -> List[int]:
    """Window ends E tried for warm-up length W, given the youngest and oldest inlier tyre ages.

    E >= W: every E from max(W, youngest age) to one below the oldest age. Any E at or below the
    youngest age gives the same fitted lap times once no warm-up lap is left to pin the peak pace, so
    only the youngest age is tried (_describe_fit adds a note when that makes the peak pace
    unidentifiable). E < W: the engine accepts a window that ends inside the warm-up; the laps after the
    ramp then start degradation_rate x (W - E) lower than with E = W. Those ends are distinct only when
    inlier laps inside the ramp pin the peak pace, so they are tried only then.
    """

    inside_ramp = list(range(1, warm_up)) if warm_up >= 2 and min_age < warm_up else []
    return inside_ramp + list(range(max(warm_up, min_age, 1), max_age))


def _solve_columns(
    times: np.ndarray, g: np.ndarray, past: Optional[np.ndarray], use: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least squares of 1/t = a g - b past, one column per window end, each on its own laps.

    Rows are scaled by t^2 so each weighted residual is approximately the lap-time residual in
    seconds (d(1/t) x t^2 = -dt); the scaled target t^2 x (1/t) is t. past=None fits the flat model
    (b = 0). ``use`` (laps x columns, 0/1) selects each column's laps. Returns (a, b, well_conditioned).
    """

    x1 = times**2 * g
    s11 = use.T @ (x1 * x1)
    t1 = use.T @ (x1 * times)
    if past is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = t1 / s11
        return a, np.zeros_like(a), s11 > 0.0
    used_x2 = use * (-(times**2)[:, None] * past)
    s12 = used_x2.T @ x1
    s22 = np.einsum("ij,ij->j", used_x2, used_x2)  # use is 0/1, so use * x2^2 == (use * x2)^2
    t2 = used_x2.T @ times
    det = s11 * s22 - s12**2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a = (s22 * t1 - s12 * t2) / det
        b = (s11 * t2 - s12 * t1) / det
    return a, b, det > 1e-12 * s11 * s22


def _column_fit(
    times: np.ndarray, g: np.ndarray, past: Optional[np.ndarray], use: np.ndarray, cap: float, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve every column, then ``steps`` concentration steps (refit on the laps within cap).

    Returns (a, b, well_conditioned, use, fitted 1 / lap time, residuals in seconds) of the last solve;
    residuals are -inf where the fitted performance is not positive. A step never shrinks a column
    below MIN_CLEAN_LAPS laps.
    """

    for step in range(steps + 1):
        a, b, conditioned = _solve_columns(times, g, past, use)
        with np.errstate(invalid="ignore", over="ignore"):
            inverse = a[None, :] * g[:, None]
            if past is not None:
                inverse = inverse - b[None, :] * past
        positive = inverse > 0.0
        residuals = times[:, None] - np.where(positive, 1.0 / np.where(positive, inverse, 1.0), np.inf)
        if step < steps:
            within = (np.abs(residuals) <= cap).astype(float)
            enough = within.sum(axis=0) >= MIN_CLEAN_LAPS
            use = np.where(enough[None, :], within, use)
    return a, b, conditioned, use, inverse, residuals


def _best_fit(
    ages: np.ndarray,
    times: np.ndarray,
    inliers: np.ndarray,
    cap: float,
    warm_up: Optional[int],
    peak_end: Optional[int],
) -> _Fit:
    """Best structure (warm-up length, window end) and its coefficients.

    Every candidate structure starts from a fit on the current inlier laps. When ``cap`` (the
    rejection threshold, seconds) is finite it then takes ``CONCENTRATION_STEPS`` steps of its own
    (refit on the laps whose residual under this candidate is within cap), so one outlier kept in the
    shared inlier set cannot spoil the fair candidate. Candidates are compared on EVERY clean lap with
    the truncated squared loss min(residual^2, cap^2) plus the information-criterion penalty (see
    ``_criterion``), so a rejected lap still counts (at most cap^2) and a structure that explains it
    can win it back. A degradation candidate needs its own laps at MIN_DEGRADATION_AGES or more
    distinct tyre ages past the window end, b > 0 and positive performance on its laps. warm_up /
    peak_end are fixed when supplied.
    """

    laps = len(times)
    fit_ages = ages[inliers]
    fit_distinct = np.unique(fit_ages)
    min_age, max_age = int(fit_distinct[0]), int(fit_distinct[-1])
    steps = CONCENTRATION_STEPS if math.isfinite(cap) else 0
    start_use = inliers.astype(float)[:, None]
    # (criterion, free parameters, warm-up, window end) -> fit, plus the laps a degrading fit used.
    ranked: List[Tuple[Tuple[float, int, int, int], _Fit, Optional[np.ndarray]]] = []
    for w in [warm_up] if warm_up is not None else _warm_up_options(ages):
        warm_free = 1 if warm_up is None and w >= 2 else 0
        g = _warm_up_factor(ages, w)
        ends = np.asarray([peak_end] if peak_end is not None else _window_ends(w, min_age, max_age), dtype=float)
        # Distinct inlier tyre ages past each window end (and past the warm-up).
        support = len(fit_distinct) - np.searchsorted(fit_distinct, np.maximum(ends, w), side="right")
        ends = ends[support >= MIN_DEGRADATION_AGES]
        degrading: List[Tuple[Tuple[float, int, int, int], _Fit, Optional[np.ndarray]]] = []
        if len(ends):
            past = _laps_past_window(ages, w, ends)
            use = np.repeat(start_use, len(ends), axis=1)
            a, b, conditioned, use, inverse, residuals = _column_fit(times, g, past, use, cap, steps)
            valid = (
                conditioned
                & np.isfinite(a)
                & np.isfinite(b)
                & (b > 0.0)
                & np.all((inverse > 0.0) | (use == 0.0), axis=0)
            )
            loss = np.sum(np.minimum(residuals**2, cap * cap), axis=0)
            parameters = 2 + (1 if peak_end is None else 0) + warm_free
            for column in np.flatnonzero(valid & np.isfinite(loss)):
                fit = _Fit(
                    warm_up_laps=w,
                    peak_end=int(ends[column]),
                    a=float(a[column]),
                    b=float(b[column]),
                    criterion=_criterion(float(loss[column]), laps, parameters, cap),
                    free_parameters=parameters,
                    linear_parameters=2,
                )
                degrading.append((_rank(fit), fit, use[:, column] > 0.0))
            if peak_end is not None:  # one column: check its support now (it decides the flat fallback)
                degrading = [item for item in degrading if _has_degradation_support(ages, item[1], item[2])]
            ranked.extend(degrading)
        # With a supplied window end the degrading model is used whenever it can be fitted; the flat
        # model (the constrained least-squares optimum when b would be <= 0) replaces it otherwise.
        if peak_end is None or not degrading:
            a, _, _, _, _, residuals = _column_fit(times, g, None, start_use, cap, steps)
            loss = float(np.sum(np.minimum(residuals**2, cap * cap)))
            fit = _Fit(
                warm_up_laps=w,
                peak_end=peak_end if peak_end is not None else max(max_age, w),
                a=float(a[0]),
                b=0.0,
                criterion=_criterion(loss, laps, 1 + warm_free, cap),
                free_parameters=1 + warm_free,
                linear_parameters=1,
            )
            ranked.append((_rank(fit), fit, None))
    # The support check needs a pass over the laps per candidate, so it is done lazily in rank order.
    # A flat candidate is always present (auto structure) or replaces an unsupported degrading one.
    for _, fit, used in sorted(ranked, key=lambda item: item[0]):
        if used is None or _has_degradation_support(ages, fit, used):
            return fit
    raise RuntimeError("no admissible tyre-model structure")  # unreachable: a flat candidate is always ranked


def _rank(fit: _Fit) -> Tuple[float, int, int, int]:
    return (fit.criterion, fit.free_parameters, fit.warm_up_laps, fit.peak_end)


def _has_degradation_support(ages: np.ndarray, fit: _Fit, used: np.ndarray) -> bool:
    """A degradation rate needs its own laps at MIN_DEGRADATION_AGES+ distinct tyre ages past the window."""

    return _distinct_ages_past(ages[used], fit.warm_up_laps, fit.peak_end) >= MIN_DEGRADATION_AGES


def _robust_sigma(deviation: np.ndarray, parameters: int) -> float:
    """Noise standard deviation from the median absolute deviation of all clean laps' residuals.

    1.4826 x MAD is consistent for normal noise; n / (n - 0.8) corrects the small-sample bias of the
    MAD and sqrt(n / (n - p)) the residual shrinkage from fitting p parameters (structure included).
    Without them a 20-lap compound rejects ordinary 2-3 sigma laps far too often.
    """

    laps = len(deviation)
    return (
        MAD_TO_SIGMA
        * float(np.median(deviation))
        * laps
        / (laps - 0.8)
        * math.sqrt(laps / max(laps - parameters, 1))
    )


def _refit(
    fit: _Fit, ages: np.ndarray, times: np.ndarray, inliers: np.ndarray, peak_end: Optional[int]
) -> _Fit:
    """Plain least squares of the chosen structure on exactly the laps kept (what is reported).

    If a detected degradation phase does not survive the refit (b <= 0 on these laps), the flat fit
    reports the oldest kept tyre age as the window end, as a detected flat structure would.
    """

    refitted = _best_fit(ages, times, inliers, math.inf, fit.warm_up_laps, fit.peak_end)
    if refitted.b == 0.0 and peak_end is None:
        refitted = replace(refitted, peak_end=max(int(ages[inliers].max()), refitted.warm_up_laps))
    return replace(refitted, criterion=fit.criterion, free_parameters=fit.free_parameters)


def _robust_fit(
    ages: np.ndarray, times: np.ndarray, warm_up: Optional[int], peak_end: Optional[int]
) -> Tuple[_Fit, np.ndarray, int, Optional[str]]:
    """Fit, reject the largest outliers, refit (structure included), re-test every lap; repeat until stable.

    The rejection threshold is OUTLIER_THRESHOLD robust standard deviations (``_robust_sigma``: the
    median absolute deviation of ALL clean laps' residuals, robust to up to half of them being
    outliers, so it does not shrink as laps are rejected). Round 1 compares structures on plain
    squared error; later rounds cap each lap's squared residual at the previous round's threshold
    (see ``_best_fit``). Each round removes only the laps beyond the threshold whose deviation is at
    least half of the largest one, and re-admits rejected laps that the new fit explains, so a few
    gross outliers cannot drag genuine warm-up or degradation laps out with them. The loop stops when
    neither the lap set nor the chosen structure changes; if the state starts to cycle (borderline
    laps), the largest lap set of the cycle is kept.

    Returns the final fit, the inlier mask, the rounds used and a note when it did not settle
    normally. The caller checks that enough laps remain.
    """

    inliers = np.ones(len(times), dtype=bool)
    history: List[np.ndarray] = []
    seen: Dict[Tuple[bytes, Tuple[int, int, bool]], int] = {}
    previous: Optional[Tuple[int, int, bool]] = None
    cap = math.inf
    for iteration in range(1, MAX_OUTLIER_ITERATIONS + 1):
        fit = _best_fit(ages, times, inliers, cap, warm_up, peak_end)
        structure = (fit.warm_up_laps, fit.peak_end, fit.b > 0.0)
        residuals = times - _predict(fit, ages)  # -inf where the fit has no positive performance
        centre = float(np.median(residuals))
        deviation = np.abs(residuals - centre)
        cap = max(OUTLIER_THRESHOLD * _robust_sigma(deviation, fit.free_parameters), MIN_OUTLIER_THRESHOLD_S)
        within = deviation <= cap
        returning = within & ~inliers
        beyond = inliers & ~within
        if not returning.any() and not beyond.any() and structure == previous:
            return _refit(fit, ages, times, inliers, peak_end), inliers, iteration, None
        previous = structure
        updated = inliers | returning
        if beyond.any():
            worst = float(np.max(deviation[beyond]))
            updated &= ~(beyond & (deviation >= 0.5 * worst))
        if np.count_nonzero(updated) < MIN_CLEAN_LAPS:
            return fit, updated, iteration, None
        key = (updated.tobytes(), structure)
        if key in seen:
            cycle = history[seen[key]:]
            inliers = max(cycle, key=lambda mask: int(np.count_nonzero(mask)))  # first largest set
            fit = _refit(_best_fit(ages, times, inliers, cap, warm_up, peak_end), ages, times, inliers, peak_end)
            return fit, inliers, iteration, (
                "Outlier rejection alternated on borderline laps; the largest set of laps in the cycle is used."
            )
        seen[key] = len(history)
        history.append(updated)
        inliers = updated
    fit = _refit(_best_fit(ages, times, inliers, cap, warm_up, peak_end), ages, times, inliers, peak_end)
    return fit, inliers, MAX_OUTLIER_ITERATIONS, (
        f"Outlier rejection did not settle within {MAX_OUTLIER_ITERATIONS} rounds; the fit on the last set of laps "
        "is reported."
    )


def _falling_trend_note(ages: np.ndarray, times: np.ndarray, warm_up: int) -> Optional[str]:
    """For flat fits: say so when lap times clearly fall with tyre age (e.g. uncorrected fuel burn)."""

    keep = ages > warm_up
    x, y = ages[keep].astype(float), times[keep]
    if np.unique(x).size < 3:
        return None
    sxx = float(np.sum((x - x.mean()) ** 2))
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    residuals = y - (y.mean() + slope * (x - x.mean()))
    std_error = math.sqrt(float(np.sum(residuals**2)) / (len(y) - 2) / sxx)
    if slope < 0.0 and slope < -2.0 * std_error:
        return (
            f"Lap times fall with tyre age in this data (least-squares slope {slope:.3f} s/lap, standard error "
            f"{std_error:.3f}). The engine cannot represent tyres that get faster, so degradation_rate is 0; "
            "fuel burn or track evolution is the usual cause (see fuel_correction_s_per_lap)."
        )
    return None


def _residual_std(residuals: np.ndarray, fit: _Fit) -> float:
    """Residual standard deviation (seconds) over the laps used, corrected for the linear coefficients fitted."""

    return math.sqrt(float(np.sum(residuals**2)) / (len(residuals) - fit.linear_parameters))


def _age_span(ages: np.ndarray) -> str:
    low, high = int(ages.min()), int(ages.max())
    return str(low) if low == high else f"{low}-{high}"


def _seconds_span(values: np.ndarray, sign: str = "") -> str:
    """'+0.321s to +0.645s', or one value when both ends print the same."""

    low, high = f"{float(values.min()):{sign}.3f}s", f"{float(values.max()):{sign}.3f}s"
    return low if low == high else f"{low} to {high}"


def _describe_fit(entry: CompoundCalibration, item: _CompoundFit, warm_up_supplied: bool, end_supplied: bool) -> None:
    """Fill a fitted compound's statistics and notes (none of them depend on the reference compound)."""

    fit = item.fit
    used_ages, used_times, fitted = item.ages[item.inliers], item.times[item.inliers], item.fitted[item.inliers]
    sse = float(np.sum((used_times - fitted) ** 2))
    sst = float(np.sum((used_times - used_times.mean()) ** 2))
    entry.laps_used = len(used_times)
    entry.outliers_rejected = len(item.times) - len(used_times)
    entry.r_squared = 1.0 - sse / sst if sst > 0.0 else None
    entry.residual_std_s = _residual_std(used_times - fitted, fit)
    entry.peak_lap_time = 1.0 / fit.a
    entry.initial_degradation_s_per_lap = 1.0 / (fit.a - fit.b) - 1.0 / fit.a if 0.0 < fit.b < fit.a else 0.0
    entry.tire_age_range = (int(used_ages.min()), int(used_ages.max()))
    entry.degradation_observed = fit.b > 0.0
    youngest, oldest = entry.tire_age_range
    notes = entry.notes

    if entry.residual_std_s > POOR_FIT_RESIDUAL_FRACTION * entry.peak_lap_time:
        notes.append(
            f"Poor fit: the residual standard deviation {entry.residual_std_s:.3f}s is more than "
            f"{POOR_FIT_RESIDUAL_FRACTION:.0%} of the peak lap time. One tyre model does not describe these laps "
            "well (mixed conditions, drivers, fuel loads or unflagged incidents?); treat the estimate with caution."
        )
    if warm_up_supplied:
        entry.warm_up_source = SOURCE_SUPPLIED
    elif fit.warm_up_laps >= 2:
        entry.warm_up_source = WARM_UP_DETECTED
        notes.append(
            f"Warm-up of {fit.warm_up_laps} laps detected: tyre laps 1-{fit.warm_up_laps - 1} are slower, using "
            "the engine's fixed ramp from 90% to 100% of base performance."
        )
    else:
        entry.warm_up_source = WARM_UP_NOT_DETECTED
        notes.append(
            f"No warm-up detected (warm_up_laps 0). A warm-up of W laps is tested only when at least "
            f"{MIN_WARM_UP_SUPPORT_LAPS} clean laps are younger than W and {MIN_WARM_UP_SUPPORT_LAPS} are at "
            f"least W laps old; the youngest clean tyre age here is {int(item.ages.min())}."
        )
    if end_supplied:
        entry.peak_window_end_source = SOURCE_SUPPLIED
        if fit.b == 0.0:
            if _distinct_ages_past(used_ages, fit.warm_up_laps, fit.peak_end) < MIN_DEGRADATION_AGES:
                notes.append(
                    f"Fewer than {MIN_DEGRADATION_AGES} distinct tyre ages beyond the supplied window end "
                    f"({fit.peak_end}), so no degradation rate could be estimated; degradation_rate is 0 and the "
                    "engine will project no wear, which is optimistic."
                )
            else:
                notes.append(
                    "The least-squares degradation beyond the supplied window end was not positive (laps did not "
                    "get slower), so degradation_rate is 0."
                )
    elif fit.b > 0.0:
        entry.peak_window_end_source = WINDOW_END_DETECTED
    else:
        entry.peak_window_end_source = WINDOW_END_NO_DEGRADATION
        notes.append(
            f"No degradation detected up to tyre age {oldest}: degradation_rate is 0 and the peak window ends at "
            "the oldest tyre age among the laps used. The engine will project no wear beyond it, which is "
            "optimistic for longer stints."
        )
    if fit.b == 0.0:
        trend = _falling_trend_note(used_ages, used_times, fit.warm_up_laps)
        if trend is not None:
            notes.append(trend)
    else:
        notes.append(
            f"Degradation is fitted on tyre ages up to {oldest}; the engine extrapolates it linearly beyond that."
        )
        # Any window end at or below the youngest age gives the same lap times when no warm-up lap pins
        # the peak, so the search stops at the youngest age (see _window_ends).
        if not end_supplied and fit.peak_end <= youngest and (fit.warm_up_laps < 2 or youngest > fit.warm_up_laps):
            notes.append(
                f"Degradation is already under way at the youngest tyre age used ({youngest}), so the peak pace is "
                f"not identifiable from these laps: the window end is placed at tyre age {youngest}, and "
                "base_performance and peak_lap_time describe the pace at that age. The real peak may be faster, with "
                "degradation starting earlier; laps on fresher tyres would show it (or supply peak_window_end if it "
                "is known)."
            )

    rejected = ~item.inliers
    residuals = item.times - item.fitted  # -inf where the fitted performance is not positive
    beyond = rejected & (item.ages > oldest)
    if beyond.any() and np.all(residuals[beyond] > 0.0):
        count = int(np.count_nonzero(beyond))
        observed = (
            f"{count} slower lap(s) at tyre age {_age_span(item.ages[beyond])}, older than every lap used, were "
            f"rejected as outliers ({_seconds_span(residuals[beyond], '+')} against the fit). "
        )
        if fit.b == 0.0:
            notes.append(
                observed + "Degradation may have started, but a degradation rate needs clean laps at "
                f"{MIN_DEGRADATION_AGES} or more distinct tyre ages past the window end, so it could not be estimated "
                "from them; longer stints would show it."
            )
        else:
            notes.append(
                observed + f"Wear may accelerate beyond tyre age {oldest} (for example a cliff), which the linear "
                "model does not represent."
            )
    faster = rejected & (residuals < 0.0)
    if np.count_nonzero(faster) >= MIXED_PACE_FAST_LAPS:
        gaps = -residuals[faster & np.isfinite(residuals)]
        by = f" (by {_seconds_span(gaps)})" if gaps.size else ""
        notes.append(
            f"Possible mixed pace: {int(np.count_nonzero(faster))} rejected laps were faster than the fit{by}. Genuine "
            "outliers are usually slow (traffic, mistakes, incidents), so these may be a faster population (lighter "
            "fuel, a newer set, another driver or session) mixed with the laps used; the estimate describes only "
            "the laps used. Check or split the laps."
        )


def _relative_limit_problems(item: _CompoundFit, reference: TireCompound, base_lap_time: float) -> List[str]:
    """Engine limits for a compound whose own performance stays above the floor, relative to the reference."""

    base_performance = base_lap_time / item.base_time  # exactly 1.0 for the reference itself
    degradation = item.fit.b * base_lap_time / item.factor
    lowest = item.lowest_performance(base_lap_time)
    problems: List[str] = []
    if base_performance < MIN_BASE_PERFORMANCE:
        problems.append(
            f"base_performance {base_performance:.4f} is below the engine minimum {MIN_BASE_PERFORMANCE:g} "
            f"(peak pace more than twice as slow as {reference.value})"
        )
    if lowest <= PERFORMANCE_FLOOR:
        problems.append(
            f"relative to {reference.value}'s base lap time the fitted performance falls to {lowest:.3f} within the "
            f"observed tyre ages, at or below the engine's {PERFORMANCE_FLOOR:.2f} floor where it stops modelling wear"
        )
    # Defensive: a positive rate is only fitted on laps at MIN_DEGRADATION_AGES or more distinct ages past
    # the window end, so passing the floor check above implies degradation_rate < base_performance / 3
    # <= 1/3, below MAX_DEGRADATION_RATE. Kept so an unexpected fit is reported, not raised by _validate_tire.
    if degradation > MAX_DEGRADATION_RATE:
        problems.append(f"degradation_rate {degradation:.4f} is above the engine maximum {MAX_DEGRADATION_RATE:g}")
    return problems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_tire_parameters(
    laps: Sequence[Union[LapRecord, Mapping[str, Any]]],
    *,
    weather: Union[WeatherCondition, str],
    track_temperature: float,
    pit_stop_delta: float,
    fuel_correction_s_per_lap: float = 0.0,
    peak_window_end: Optional[Mapping[Any, int]] = None,
    warm_up_laps: Optional[Mapping[Any, int]] = None,
) -> TyreCalibrationResult:
    """Estimate per-compound ``TireData`` for the strategy engine from timed laps.

    ``laps``: ``LapRecord`` objects or mappings with the same keys (compound, tire_age, lap_time and
    optional race_lap, pit_out, pit_in, safety_car). ``weather`` and ``track_temperature`` are the
    session's conditions; the engine's weather and hot-track factors for them are divided out.
    ``pit_stop_delta`` (seconds) is copied into every ``TireData``: it cannot be estimated from lap
    history. ``fuel_correction_s_per_lap`` (seconds gained per lap of fuel burnt, default 0 = no
    correction) needs ``race_lap`` on every unflagged lap; lap times are corrected to the fuel load
    of the latest unflagged race lap. ``peak_window_end`` / ``warm_up_laps`` fix those values for
    the named compounds instead of detecting them.

    Raises ValueError for invalid input. A compound without enough clean laps is reported as
    ``insufficient_data`` (never guessed); the function does not raise for that.
    """

    records = _validate_laps(laps)
    weather_condition = _enum(WeatherCondition, "weather", weather)
    temperature = _number("track_temperature", track_temperature, MIN_TRACK_TEMPERATURE_C, MAX_TRACK_TEMPERATURE_C)
    pit_delta = _number("pit_stop_delta", pit_stop_delta, 0.0, MAX_PIT_STOP_DELTA_S)
    fuel = _number("fuel_correction_s_per_lap", fuel_correction_s_per_lap, 0.0, MAX_FUEL_CORRECTION_S_PER_LAP)
    present = sorted({lap.compound for lap in records}, key=_COMPOUND_ORDER.index)
    supplied_ends = _validate_overrides("peak_window_end", peak_window_end, 1, MAX_TOTAL_LAPS, present)
    supplied_warm_up = _validate_overrides("warm_up_laps", warm_up_laps, 0, MAX_WARM_UP_LAPS, present)

    excluded: List[ExcludedLap] = []
    clean: List[int] = []
    for index, lap in enumerate(records):
        flags = [flag for flag in LAP_FLAGS if getattr(lap, flag)]
        if flags:
            excluded.append(ExcludedLap(index, lap.compound, lap.tire_age, lap.lap_time, ", ".join(flags)))
        else:
            clean.append(index)

    corrected = {index: records[index].lap_time for index in clean}
    fuel_reference: Optional[int] = None
    if fuel > 0.0 and clean:
        race_laps: Dict[int, int] = {}
        missing: List[int] = []
        for index in clean:
            race_lap = records[index].race_lap
            if race_lap is None:
                missing.append(index)
            else:
                race_laps[index] = race_lap
        if missing:
            raise ValueError(
                f"fuel_correction_s_per_lap needs race_lap on every unflagged lap; missing on laps{missing[:10]}"
            )
        fuel_reference = max(race_laps.values())
        for index in clean:
            value = records[index].lap_time - fuel * (fuel_reference - race_laps[index])
            if value < MIN_LAP_TIME_S:
                raise ValueError(
                    f"laps[{index}]: the fuel-corrected lap time {value:.3f}s is below {MIN_LAP_TIME_S:g}s; "
                    "fuel_correction_s_per_lap is too large for this data"
                )
            corrected[index] = value

    fits: Dict[TireCompound, _CompoundFit] = {}
    compounds: Dict[TireCompound, CompoundCalibration] = {}
    for compound in present:
        supplied = sum(1 for lap in records if lap.compound == compound)
        indices = np.asarray([index for index in clean if records[index].compound == compound], dtype=int)
        entry = CompoundCalibration(
            compound=compound,
            status=STATUS_INSUFFICIENT_DATA,
            reason=None,
            laps_supplied=supplied,
            laps_flagged=supplied - len(indices),
            clean_laps=len(indices),
            laps_used=0,
            outliers_rejected=0,
        )
        compounds[compound] = entry
        if len(indices) < MIN_CLEAN_LAPS:
            entry.reason = (
                f"{len(indices)} clean lap(s) after excluding flagged laps; at least {MIN_CLEAN_LAPS} are needed."
            )
            continue
        ages = np.asarray([records[index].tire_age for index in indices], dtype=float)
        times = np.asarray([corrected[index] for index in indices], dtype=float)
        fit, inliers, iterations, loop_note = _robust_fit(
            ages, times, supplied_warm_up.get(compound), supplied_ends.get(compound)
        )
        kept = int(np.count_nonzero(inliers))
        entry.outlier_iterations = iterations
        if kept < MIN_CLEAN_LAPS:
            entry.reason = (
                f"Only {kept} of {len(indices)} clean laps are consistent with the tyre model after outlier "
                f"rejection; at least {MIN_CLEAN_LAPS} are needed."
            )
            continue
        fitted = _predict(fit, ages)
        rejected = len(indices) - kept
        if rejected > MAX_OUTLIER_FRACTION * len(indices):
            faster = int(np.count_nonzero(times[~inliers] < fitted[~inliers]))
            supplied_structure = " and ".join(
                f"{name}={values[compound]}"
                for name, values in (("peak_window_end", supplied_ends), ("warm_up_laps", supplied_warm_up))
                if compound in values
            )
            entry.reason = (
                f"{rejected} of {len(indices)} clean laps ({rejected / len(indices):.0%}) were rejected as outliers "
                f"({faster} faster and {rejected - faster} slower than the fit to the rest). More than "
                f"{MAX_OUTLIER_FRACTION:.0%} means one tyre model does not describe these laps: they probably mix "
                "different pace (fuel loads, tyre sets, drivers, sessions or unflagged incidents)"
                + (f", or the supplied {supplied_structure} does not match them" if supplied_structure else "")
                + ", and a fit would describe only part of them. Flag or split the laps."
            )
            continue
        scatter = _residual_std(times[inliers] - fitted[inliers], fit)
        if scatter > MAX_RESIDUAL_STD_FRACTION / fit.a:  # 1 / a is the fitted peak lap time
            entry.reason = (
                f"The best fit leaves a residual standard deviation of {scatter:.3f}s, more than "
                f"{MAX_RESIDUAL_STD_FRACTION:.0%} of its peak lap time {1.0 / fit.a:.3f}s, so no single tyre model "
                f"describes these laps (clean lap times range from {times.min():.3f}s to {times.max():.3f}s). "
                "Typical causes: many grossly slow unflagged laps (safety-car, red-flag or timing-error laps), which "
                "can defeat the outlier rejection, or two pace levels mixed about evenly. Flag, remove or split those "
                "laps."
            )
            continue
        if loop_note is not None:
            entry.notes.append(loop_note)
        factor = _weather_multiplier(weather_condition, compound) * _temperature_multiplier(temperature, compound)
        fits[compound] = _CompoundFit(fit, indices, ages, times, inliers, fitted, factor)

    for compound, item in fits.items():  # statistics, excluded laps and notes (independent of the reference)
        entry = compounds[compound]
        for position in np.flatnonzero(~item.inliers):
            index = int(item.indices[position])
            lap = records[index]
            model = float(item.fitted[position])
            finite = math.isfinite(model)
            excluded.append(
                ExcludedLap(
                    index,
                    compound,
                    lap.tire_age,
                    lap.lap_time,
                    OUTLIER_REASON,
                    # Back in lap_time's raw frame: add the fuel correction that was taken off this lap.
                    fitted_lap_time=model + (lap.lap_time - corrected[index]) if finite else None,
                    residual_s=float(item.times[position]) - model if finite else None,
                )
            )
        _describe_fit(entry, item, compound in supplied_warm_up, compound in supplied_ends)

    # Reference compound: the fastest (peak lap time with the conditions factor removed; ties keep the
    # soft -> wet order) among compounds whose OWN fitted performance stays above the engine's floor. A
    # compound failing that check is never estimated, so it cannot define base_performance 1.0; every
    # other compound is then checked relative to the reference, which therefore always gets tire_data.
    eligible = [
        compound for compound, item in fits.items() if item.lowest_performance(item.base_time) > PERFORMANCE_FLOOR
    ]
    reference: Optional[TireCompound] = None
    base_lap_time: Optional[float] = None
    if eligible:
        reference = min(eligible, key=lambda c: (fits[c].base_time, _COMPOUND_ORDER.index(c)))
        base_lap_time = fits[reference].base_time
    faster_excluded = [
        compound
        for compound in fits
        if compound not in eligible and (base_lap_time is None or fits[compound].base_time < base_lap_time)
    ]

    tire_data: Dict[TireCompound, TireData] = {}
    for compound, item in fits.items():
        entry = compounds[compound]
        if reference is None or base_lap_time is None or compound not in eligible:
            entry.status = STATUS_OUTSIDE_ENGINE_LIMITS
            entry.reason = (
                f"Fitted, but the fitted performance falls to {item.lowest_performance(item.base_time):.3f} of its "
                "own peak pace (conditions factor included) within the observed tyre ages, at or below the engine's "
                f"{PERFORMANCE_FLOOR:.2f} floor where it stops modelling wear."
            )
            continue
        problems = _relative_limit_problems(item, reference, base_lap_time)
        if problems:
            entry.status = STATUS_OUTSIDE_ENGINE_LIMITS
            entry.reason = "Fitted, but " + "; ".join(problems) + "."
            continue
        fit = item.fit
        candidate = TireData(
            compound=compound,
            base_performance=1.0 if compound == reference else base_lap_time / item.base_time,
            degradation_rate=fit.b * base_lap_time / item.factor,
            warm_up_laps=fit.warm_up_laps,
            peak_performance_window=(min(max(1, fit.warm_up_laps), fit.peak_end), fit.peak_end),
            pit_stop_delta=pit_delta,
        )
        _, entry.tire_data = _validate_tire(compound, candidate)  # the engine's own checks
        entry.status = STATUS_ESTIMATED
        tire_data[compound] = entry.tire_data

    excluded.sort(key=lambda lap: lap.index)
    return TyreCalibrationResult(
        compounds=compounds,
        tire_data=tire_data,
        estimated_base_lap_time=base_lap_time,
        reference_compound=reference,
        excluded_laps=excluded,
        laps_supplied=len(records),
        weather=weather_condition,
        track_temperature=temperature,
        pit_stop_delta=pit_delta,
        fuel_correction_s_per_lap=fuel,
        fuel_reference_lap=fuel_reference,
        assumptions=_assumptions(
            weather_condition, temperature, fuel, fuel_reference, reference, base_lap_time, bool(fits), faster_excluded
        ),
    )


def _assumptions(
    weather: WeatherCondition,
    temperature: float,
    fuel: float,
    fuel_reference: Optional[int],
    reference: Optional[TireCompound],
    base_lap_time: Optional[float],
    fitted_any: bool,
    faster_excluded: Sequence[TireCompound],
) -> List[str]:
    notes = [
        (
            "Model: the strategy engine's tyre model, inverted. Performance ramps from 90% over warm_up_laps, holds "
            "base_performance to the end of peak_performance_window, then falls LINEARLY by degradation_rate per lap. "
            "Real degradation that is not linear (for example a late cliff) is approximated by this line."
        ),
        (
            "Fit: per compound, weighted least squares on 1/lap time for every admissible warm-up length and window "
            "end; the lowest Bayesian information criterion wins. A degradation rate needs clean laps at "
            f"{MIN_DEGRADATION_AGES} or more distinct tyre ages past the window end and is constrained to be >= 0."
        ),
        (
            f"Outliers: laps more than {OUTLIER_THRESHOLD:g} robust standard deviations (1.4826 x MAD of the "
            f"residuals, small-sample corrected) from the median residual, and more than {MIN_OUTLIER_THRESHOLD_S:g}s, "
            "are rejected iteratively; borderline laps near that threshold can be rejected too. Laps flagged "
            "pit_in, pit_out or safety_car are always excluded; unflagged safety-car, traffic or yellow-flag laps are "
            "only caught when they stand out from the fit."
        ),
        (
            f"Conditions: all laps are treated as one session in {weather.value} conditions at {temperature:g} C. "
            "The engine's weather and hot-track factors for these conditions are divided out, so base_performance "
            "is in the engine's convention (before those factors)."
        ),
        (
            "Pace: every stint of a compound shares one base pace. Track evolution, traffic and set-to-set "
            "differences are not modelled and end up in the residuals."
        ),
    ]
    if fuel > 0.0:
        notes.append(
            f"Fuel: lap times are corrected by {fuel:g}s per lap of fuel burnt to the fuel load of race lap "
            f"{fuel_reference} (the latest unflagged lap). The correction is the value you supplied, not an estimate."
        )
    else:
        notes.append(
            "Fuel: not corrected. Fuel burn makes later laps faster, which hides part of the tyre degradation "
            "(underestimated degradation_rate) unless fuel_correction_s_per_lap and race_lap are supplied."
        )
    skipped = ", ".join(compound.value for compound in faster_excluded)
    if reference is not None and base_lap_time is not None:
        notes.append(
            f"base_performance is relative to {reference.value}, the fastest compound whose fit is within the engine's "
            f"limits (1.0). The estimated base lap time {base_lap_time:.3f}s includes the driver's and car's actual "
            "pace: the engine multiplies telemetry lap times by its driver and damage multipliers, so pass it as "
            "telemetry.lap_times with a neutral driver profile and an undamaged car to reproduce the observed pace."
        )
        if skipped:
            notes.append(
                f"Faster at peak but outside the engine's limits, so not the reference and not in tire_data: {skipped} "
                "(see its reason)."
            )
    elif fitted_any:
        notes.append(
            f"No fitted compound is within the engine's limits (see each compound's reason: {skipped}), so no base "
            "lap time or tire_data could be estimated."
        )
    else:
        notes.append(
            "No compound had enough clean laps consistent with one tyre model, so no base lap time or tire_data could "
            "be estimated."
        )
    notes.append("pit_stop_delta is copied from the request; it is not estimated from lap history.")
    return notes
