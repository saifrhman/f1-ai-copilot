# F1 Strategy Engine

This module ranks race-strategy candidates for the remaining laps of a race. It uses the supplied telemetry, tyre model, race state, car condition, driver parameters and (optionally) competitor gaps.

It is a **transparent, deterministic heuristic**. It is not a calibrated race simulator, and its output is not authoritative strategy advice. Every assumption is listed in the response.

The tyre parameters in `tire_data` are inputs. You can supply them by hand, or estimate them from real lap history with `calibration.estimate_tire_parameters` (see [Calibrating the tyre model from lap history](#calibrating-the-tyre-model-from-lap-history)).

## Model

For each lap:

```text
lap time = mean(telemetry.lap_times) x driver multiplier x damage/wear multiplier / tyre performance
```

- **Tyre performance** depends on the compound and on the tyre's own age. It ramps from 90% to 100% of `base_performance` over `warm_up_laps`, holds `base_performance` until the end of `peak_performance_window`, then loses `degradation_rate` per lap. It is scaled by a weather factor (for example dry tyres on a wet track) and a hot-track factor (above 35 °C: soft x0.97, hard x1.02), and floored at 0.20. The start of the peak window does not change lap times. Laps clamped to the floor are counted per stint (`laps_at_performance_floor`) and flagged in the candidate's `notes`, because the model does not represent wear beyond the floor and those lap times are optimistic.
- **Driver multiplier** (heuristic, applied the same to every lap):
  - `braking_consistency` below 0.7 adds up to +21%.
  - `throttle_aggressiveness` above 0.8 with `tire_management` below 0.6 adds +10%.
  - `risk_tolerance` above 0.8 adds +3%.
  - `telemetry.braking_consistency` / `telemetry.throttle_aggressiveness` override the driver profile. Each value is required in telemetry or in `driver_profile`. When both are given, the unused profile value is listed in `not_modelled_inputs`.
- **Damage/wear multiplier** (heuristic, applied the same to every lap):
  - front wing: up to +2.5%;
  - max(floor, diffuser): up to +4%;
  - `engine_wear` above 0.7: up to +2%;
  - `brake_wear` above 0.8: up to +1.5%.
- **Pit stops**: each stop costs the `pit_stop_delta` of the compound fitted at that stop. The earliest stop is at the end of `current_lap`, and a pit lap is the last lap of the stint before the stop.
- **Weather** is assumed constant for the remaining laps. Stops fit soft/medium/hard tyres in the dry and intermediate/wet tyres otherwise. No drying-track (intermediate-to-dry) candidate is generated, because the model cannot represent a drying track.

### Current tyre state

- If `race_state.current_compound` and `current_tire_age` are given (always together), the first stint continues on that set from its age, with no pit cost. Changing tyres, even at the end of the current lap, costs a stop.
- Otherwise every candidate assumes a fresh set at `current_lap` with no pit cost. That assumption appears in every candidate's `notes`, and the response has `tire_state: "assumed_fresh"`.

### Pit-lap optimisation

The engine enumerates every compound sequence with 0 to 3 further stops, then chooses the stint lengths that minimise projected time. This uses exact dynamic programming (min-plus convolution) over cumulative stint-time tables. As a result, pit laps move when degradation, warm-up, peak window or pit loss change.

In this model lap time depends only on the compound and the tyre's age. The order of stints after the first one therefore does not change the time, and equivalent orderings are merged. A 70-lap request takes about 15-60 ms; the 200-lap maximum takes about 0.2 s.

Sequences that need more stints than there are laps left are skipped. The request never fails for that reason, and a zero-stop ("no further stop") candidate is always available when it is allowed by the rule below.

### Simplified two-dry-compound rule

In a race without intermediate/wet tyres, at least two different dry compounds must be used. Tyre-set allocation and all other sporting-regulation details are not modelled. The rule is checked against the plan, the fitted set and `race_state.used_compounds` (compounds run before the fitted set). Each candidate reports `two_compound_rule` as one of:

| Value | Meaning |
| --- | --- |
| `satisfied` | Two different dry compounds are used. |
| `waived` | An intermediate or wet tyre is used. |
| `unverified` | Only one dry compound is known, and the tyres used before `current_lap` are unknown (`current_lap > 1` and no `used_compounds`). The candidate is kept and flagged. |
| `violated` | The history is known and only one dry compound is used. Such candidates are removed unless no plan can comply; they are then returned, and the assumptions and each candidate's notes state the cause: either only one dry compound is known or available (for example `tire_data` has a single dry compound), or no stop fits in the remaining laps. |

## Response

`generate_strategy(...)` returns a `StrategyResult`. `schemas.strategy_result_to_dict` turns it into a JSON-safe body with enums as strings, times rounded to milliseconds and no per-lap arrays.

- `strategies`: up to 8 candidates, fastest first. The fastest plan of each stop count is always included. Each candidate has:
  - `strategy_id` (for example `1-stop:medium/hard`), `rank`, `pit_stops`, `tire_compounds` and `pit_laps`;
  - `projected_race_time`, `driving_time_s` and `pit_time_loss_s` (seconds, remaining laps only);
  - `delta_to_best_s` (seconds behind the fastest plan; there is no confidence score);
  - `risk_level`, a label from the stop count only;
  - `two_compound_rule`, `stint_breakdown` and `notes`.
- `stint_breakdown`: each entry has `start_lap`, `end_lap`, `laps`, `tire_compound`, `tire_age_start`, `tire_age_end`, `fitted_at_stop`, `average_lap_time`, `best_lap_time`, `worst_lap_time`, `total_time`, `start_performance`, `end_performance`, `laps_beyond_peak_window` (tyre laps after the end of the peak window) and `laps_at_performance_floor`.
- `model`: `base_lap_time_s`, `driver_multiplier` and `damage_multiplier`.
- `assumptions`: every modelling assumption that applies to this response.
- `not_modelled_inputs`: inputs that were supplied but do not affect any number, in this order:
  - `telemetry.sector_times` (accepted so that one natural-query context can serve performance and strategy questions);
  - `car_status.fuel_load`, `car_status.brake_temp`, `car_status.ers_availability`;
  - `driver_profile.braking_consistency (overridden by telemetry.braking_consistency)` and the same entry for `throttle_aggressiveness`, when telemetry and the driver profile both give the value;
  - `driver_profile.overtaking_style`;
  - `race_state.track_evolution`, `race_state.safety_car_probability`, `race_state.yellow_flag_risk`, `race_state.weather_forecast`;
  - `tire_data.*.peak_performance_window start (only the window end changes lap times)`;
  - `competition[].current_position`, `competition[].gap_ahead`, `competition[].gap_behind`, `competition[].pit_stops_completed`, `competition[].estimated_strategy`, each listed when any competitor gives it;
  - `competition (needs race_state.own_gap_to_leader to be related to your car)`, when competitors are given without `race_state.own_gap_to_leader`. Competitor signals are then not computed, so the competitor data changes nothing.

  Every entry is listed only when its input was supplied, except the peak-window start. That entry is always present: the window start is a required input (validated as 1 <= start <= end), but only the window end changes lap times. All the other entries refer to optional inputs.
- `competitor_signals`: response-level labels. They are computed only when `race_state.own_gap_to_leader` is given, from each competitor's `gap_to_leader`:
  - `undercut_target`: a car up to 3.0 s ahead whose tyre age is past the end of its compound's peak window in `tire_data`;
  - `undercut_threat`: a car level with you (same `gap_to_leader`) or up to 3.0 s behind.

  They do not change the ranking, and the undercut itself is not simulated. `competitor_signals_note` gives this definition, or explains why the signals were not computed. It also names cars up to 3.0 s ahead that could not be assessed because their compound has no `tire_data` entry.
- `search`: how many sequences were evaluated, skipped for too few laps, or excluded by the rule.

## Validation

All inputs are validated, and invalid input raises `ValueError`. Through the pydantic `StrategyRequest` it raises `ValidationError`, which is a `ValueError` subclass. Checks and limits:

- Numbers must be finite (no NaN or infinity), and booleans are not accepted as numbers.
- `telemetry.lap_times` is required: a non-empty list of 20-600 s values (at most 200). No default lap time is invented.
- Optional `telemetry.sector_times`: `{"1": 28.1, ...}` (sector numbers 1-50) or a list from sector 1, with values in (0, 600] s.
- `braking_consistency` and `throttle_aggressiveness` must each come from telemetry or from `driver_profile`.
- `base_performance` in [0.5, 2] (lower values would make laps at least twice the base lap time, and values near the 0.20 floor would all give the same result); `degradation_rate` in [0, 0.5]; `warm_up_laps` in 0-10.
- `peak_performance_window` is 1 <= start <= end <= 200.
- `pit_stop_delta` in [0, 120] s.
- The `tire_data` key must match its `compound`.
- `total_laps` <= 200 and 1 <= `current_lap` <= `total_laps`, both integers.
- Probabilities and wear/damage/driver values are in [0, 1]; `track_temperature` in [-10, 80] °C.
- Damage parts are limited to `front_wing`, `floor` and `diffuser`; unknown telemetry keys are rejected.
- Competitor `driver_id` values must be unique.
- Integers too large for a float (for example a 400-digit number) raise `ValueError`, not `OverflowError`.
- `race_state.weather_forecast` and `competition[].estimated_strategy` are accepted but not modelled. Their entries are objects with at most 10 keys. Values may be text (at most 64 characters), finite numbers (absolute value at most 1e9), booleans, null, or flat lists of those (at most 50 items). Nested objects are rejected.

## Usage

```python
from core_modules.strategy_optimizer.schemas import StrategyRequest, generate_strategy_response

request = StrategyRequest.model_validate(payload)   # or StrategyRequest.from_context(context)
body = generate_strategy_response(request)          # the JSON body of POST /api/strategy/generate
```

`generate_strategy_response` raises `ValueError` on invalid or impossible input. Callers that need the `StrategyResult` itself run `generate_strategy(**request.to_engine_inputs())` and serialise it with `strategy_result_to_dict`; `result.best` is the fastest candidate.

`StrategyRequest.from_context(context)` ignores unrelated top-level keys of a natural-query context (for example `audio_file` or `track_profile`). The strategy keys are validated exactly like the POST body, so a context whose `telemetry` also has `sector_times` is accepted. There is no confidence value: report `delta_to_best_s` or `tire_state` instead.

`evaluate_plan(compounds, stint_lengths, telemetry, car_status, driver_profile, tire_data, race_state)` projects one user-specified plan with the same model (a "what if I pit on lap X" query).

The engine keeps no shared mutable state, so it is safe to call from FastAPI's threadpool. It does not use matplotlib.

## Calibrating the tyre model from lap history

Hand-supplied `base_performance`, `degradation_rate`, `warm_up_laps` and `peak_performance_window` make the projections uncalibrated. `calibration.estimate_tire_parameters` estimates them from timed laps. It inverts the same lap-time model the engine projects with, so the estimates can be used as `tire_data` directly.

### Workflow

1. Collect laps. Each lap has `compound`, `tire_age` (the lap number on that set; 1 is the first lap on it) and `lap_time` in seconds. Flag `pit_out`, `pit_in` and `safety_car` laps (safety-car, VSC or red-flag laps); flagged laps are always excluded. `race_lap` is optional and only needed for a fuel correction.
2. Estimate:

   ```python
   from core_modules.strategy_optimizer.calibration import estimate_tire_parameters

   result = estimate_tire_parameters(
       laps,                       # LapRecord objects or dicts with the keys above (at most 2000 laps)
       weather="dry",              # conditions of the whole history
       track_temperature=31.0,     # degrees C of the history
       pit_stop_delta=22.0,        # copied into every tire_data entry; not estimated
       fuel_correction_s_per_lap=0.0,  # optional, see Limits
   )
   ```

   For an HTTP body, use `schemas.TyreCalibrationRequest` (same fields, `extra="forbid"`, finite numbers only, at most 2000 laps). `schemas.calibrate_tyres_response(request)` returns the JSON-safe response, and `schemas.tyre_calibration_to_dict(result)` serialises a result you already have.
3. Check every compound's `status`, fit statistics and `notes` before using the numbers.
4. Use `result.tire_data` (JSON: `tire_data`) as the strategy request's `tire_data`, and pass `estimated_base_lap_time` (JSON: `estimated_base_lap_time_s`) as `telemetry.lap_times`. The engine multiplies telemetry lap times by its driver and damage multipliers, and the estimated base lap time already contains the driver's and car's actual pace. Use a neutral driver profile and an undamaged car (both multipliers 1.0) to reproduce the observed pace; otherwise those penalties are counted twice.

`peak_window_end={"soft": 8}` and `warm_up_laps={"soft": 3}` fix those values for the named compounds instead of detecting them.

### How the fit works

The engine's lap time is `base lap time / (performance(k) x weather factor x hot-track factor)`, where `performance(k)` is the warm-up ramp, then `base_performance`, then a linear loss of `degradation_rate` per lap after the window end. Taking the reciprocal, `1 / lap time = a x g(k) - b x h(k)`. For a fixed warm-up length W and window end E this is linear in `a` and `b`. For each compound:

- Flagged laps are excluded and the optional fuel correction is applied.
- For every admissible W (0 or 2-10; W = 1 gives the same lap times as W = 0) and every window end E, `a` and `b` are solved by weighted least squares on 1 / lap time. The weights make each residual approximately a lap-time residual in seconds. `b` is constrained to be >= 0, because the engine cannot represent tyres that get faster with age. A flat model (no degradation) is always a candidate.
- Window ends run from the youngest observed tyre age to one below the oldest. Any earlier end fits the laps equally well once no warm-up lap pins the peak, so it cannot be told apart (a note says so; see [Calibration response](#calibration-response)). Ends inside the warm-up (E < W, which the engine accepts: the tyre then leaves the ramp already degraded) are also tried when laps inside the ramp pin the peak pace. A supplied window end does not limit the warm-up lengths tried.
- The structure with the lowest Bayesian information criterion wins. A detected window end and a detected warm-up count as one parameter each, so they are reported only when they improve the fit by more than noise would.
- Outliers are rejected iteratively. A lap is an outlier when its residual is more than 3.5 robust standard deviations from the median residual, and never within 0.1 s. The robust standard deviation is 1.4826 x the median absolute deviation of all clean laps' residuals, corrected for small samples (n / (n - 0.8)) and for the fitted parameters (sqrt(n / (n - p))). Each round removes the largest deviations first and re-admits laps that the new fit explains. Candidate structures are scored on all clean laps, with each squared residual capped at the threshold; after the first round the noise scale is taken as known (threshold / 3.5), so the capped cost of a gross outlier, which every candidate pays, does not dilute real differences between structures. This way a few gross outliers cannot hide a real warm-up or degradation phase. The loop stops when neither the lap set nor the chosen structure changes. If it starts to alternate on a borderline lap, the largest set of laps in the cycle is kept and a note says so.
- The engine's weather and hot-track factors for the given conditions are divided out. `base_performance` is then relative to the reference compound, which gets exactly 1.0 (the engine convention). The reference is the fastest compound whose own fitted performance stays above the engine's 0.20 floor. A faster compound that fails this check is reported as `outside_engine_limits` and named in the assumptions; it never becomes the reference, so the reference always has `tire_data`. `estimated_base_lap_time` is the reference's peak lap time with the same factors removed.

The result is deterministic and, up to floating-point rounding, does not depend on the order of the laps.

### Minimum data

Missing data is reported, never guessed:

| Requirement | If it is not met |
| --- | --- |
| At least 5 clean (unflagged) laps per compound | The compound is `insufficient_data` and has no `tire_data`. |
| At least 5 laps left after outlier rejection | `insufficient_data`. |
| At most 25% of the clean laps rejected as outliers | `insufficient_data`: one tyre model does not describe the laps. The reason gives how many rejected laps were faster and slower than the fit to the rest; the usual causes are two pace levels mixed (fuel loads, tyre sets, drivers, sessions), many unflagged incidents, or a supplied `peak_window_end` / `warm_up_laps` that does not match the laps (named in the reason). With normal noise the 3.5-sigma rule rejects about 0.05% of laps. |
| Residual standard deviation at most 3% of the peak lap time (about 2.4 s at 80 s) | `insufficient_data`. This catches what the outlier rule cannot: when about half or more of the laps are junk, or two pace levels are mixed about evenly, the median-based noise scale itself is inflated, nothing is rejected and the fit lands between the levels. Many grossly slow unflagged laps can have the same effect (see [Calibration limits](#calibration-limits)). The reason gives the range of clean lap times. Between 1% and 3% the compound is estimated with a "Poor fit" note. |
| Clean laps at 3 or more distinct tyre ages after the window end | `degradation_rate` is 0 and `degradation_observed` is false. A note explains why; the engine will then project no wear, which is optimistic. Slower laps at the one or two oldest ages that no admissible structure explains are rejected as outliers, listed in `excluded_laps`, and a note says that degradation may have started there. |
| A warm-up of W laps: at least 2 clean laps younger than W and 2 at least W laps old | That W is not tested. If no warm-up is found, `warm_up_source` is `not_detected`. |
| Fitted values inside the engine's limits: performance above the 0.20 floor at every tyre age used (against the compound's own peak, then against the reference) and `base_performance` >= 0.5 | `outside_engine_limits`, with the reason. The compound has no `tire_data`. `degradation_rate` <= 0.5 is also checked, but only defensively: a rate is fitted only from 3 or more ages past the window end, so a fit above the floor always has `degradation_rate` below `base_performance` / 3. |

If no compound can be estimated, `estimated_base_lap_time` is `None` and `tire_data` is empty. The call does not raise for that. Invalid input (non-finite or out-of-range numbers, booleans as numbers, unknown keys or compounds, a fuel correction without `race_lap`, overrides for compounds without laps) raises `ValueError`, or `ValidationError` through the schema.

### Calibration response

- `tire_data`: only compounds with status `estimated`, in the `StrategyRequest.tire_data` format (each entry is validated against it).
- `estimated_base_lap_time_s` and `reference_compound`: the fastest compound within the engine's limits. It has `base_performance` 1.0 and always has a `tire_data` entry (both are `null` when no compound is estimated).
- `compounds.<compound>` has these fields:
  - `status`, `reason`, `laps_supplied`, `laps_flagged`, `clean_laps`, `laps_used`, `outliers_rejected` and `tire_data`. `laps_used` and `outliers_rejected` are 0 for an `insufficient_data` compound; its `reason` gives the counts.
  - `warm_up_source` (`detected`, `not_detected` or `supplied`), `peak_window_end_source` (`detected`, `no_degradation_observed` or `supplied`) and `degradation_observed`;
  - `fit`: `peak_lap_time_s`, `initial_degradation_s_per_lap` (the lap one lap into degradation minus the peak lap), `tire_age_range` of the laps used, `r_squared`, `residual_std_s` and `outlier_rounds`. Lap times here are under the history's conditions and, with a fuel correction, at the fuel load of `fuel_reference_lap`. R² is near 0 when no warm-up or degradation is present, because the model then explains nothing beyond the mean.
  - `notes`, including these warnings:
    - "Poor fit": the residual standard deviation is above 1% of the peak lap time.
    - "Possible mixed pace": 3 or more rejected laps were faster than the fit. Genuine outliers are usually slow, so these may be a faster population the estimate leaves out.
    - Slower laps older than every lap used were rejected: degradation may have started there (flat fit), or wear may accelerate beyond the oldest age used (for example a cliff).
    - Degradation is already under way at the youngest tyre age used (for example stints on used sets): the peak pace is not identifiable, so `base_performance` and the base lap time describe the pace at that age.
    - Outlier rejection alternated on borderline laps (the largest set of laps was kept).
- `excluded_laps`: `index` (position in the request), `compound`, `tire_age`, `lap_time`, `reason` (the flags, or `outlier`), and for outliers `fitted_lap_time` and `residual_s`. They are sorted by `index`. `fitted_lap_time` is in the same raw frame as `lap_time`: with a fuel correction, the model's lap time at that lap's own fuel load. So `residual_s` = `lap_time` - `fitted_lap_time`.
- `conditions` (including `fuel_reference_lap`) and `assumptions`.

### Accuracy in tests

The test histories are generated by the engine itself: about 120 clean laps over six stints, Gaussian noise of 0.15 s, and five injected outliers from -2.5 s to +25 s. Across 300 random seeds the worst cases were:

- `base_performance` within 0.003 and the base lap time within 0.16 s;
- `degradation_rate` within 32% (soft), 27% (medium) and 40% (hard);
- window end within 1, 1 and 3 laps; the 3-lap warm-up was always found.

Every injected outlier was rejected. At most 2 ordinary laps per run were rejected as borderline (all within 0.7 s of the fit), and no run gave a "Poor fit" or "Possible mixed pace" note. Noise-free histories are recovered exactly, including a window end inside the warm-up. These figures describe the engine's own model with synthetic noise, not real tyres.

Contaminated histories (development sweeps, not part of the test suite; one compound, 80 s plus 0.05 s per lap after tyre age 10):

- Two stints (56 laps) with runs of 3-5 consecutive unflagged safety-car laps (35-50 s slower): with one run, 100 of 100 seeds were estimated within 0.15 s of the true peak. With two or three runs, 193 of 200 were; 6 were `insufficient_data` and 1 was estimated 0.39 s off. Without the 25% and 3% guards all 300 would be reported as `estimated`, five of them 2.6-9.9 s off (peaks from 70.1 to 87.9 s) with only a "Poor fit" note.
- 1500 histories of 10-79 laps with 30-70% junk laps (a second level 6 s slower, Cauchy errors, or uniform 20-600 s): 1001 were `insufficient_data`. 77 of the 499 estimates were more than 0.5 s from the 80 s level. 63 of those carried a "Poor fit" or "Possible mixed pace" note; the other 14 were cases where the junk laps were the majority (the fit then follows the majority), or estimates 0.5-0.9 s off under Cauchy errors.

On a development machine (8 cores, load about 2.5) a calibration of 2000 laps took about 0.1 s, and about 0.3 s with 30% junk laps (all compounds are then `insufficient_data`).

### Calibration limits

- **Linear degradation.** Wear is one straight line after one breakpoint. A cliff or curved wear is approximated by that line. Beyond the oldest tyre age in the data the engine extrapolates it linearly, or holds performance flat when no degradation was observed.
- **Fixed warm-up shape.** The ramp is always from 90% to 100% over `warm_up_laps`. A milder real warm-up is either not detected or fitted with the nearest length. Because out-laps are excluded, a 2-lap warm-up (which only slows tyre age 1) is usually not identifiable.
- **Fuel is not separated unless you model it.** Fuel burn makes later laps faster and hides degradation, which biases `degradation_rate` low. `fuel_correction_s_per_lap` (for example about 0.03-0.06 s per lap) corrects every lap to the fuel load of the latest unflagged `race_lap`. The value is yours; it is not estimated.
- **Clean green-flag laps are required.** Unflagged safety-car, traffic or yellow-flag laps are removed only when they stand out from the fit. A run of unflagged slow laps at the end of a stint can look like degradation. Flag them. Grossly slow unflagged laps (tens to hundreds of seconds off) are usually rejected. Several of them can still defeat the rejection even as a minority, because the lap-time-squared weights give very slow laps extra pull on the first fit. The compound is then reported as `insufficient_data` (residual scatter above 3%), not estimated.
- **One pace per compound.** Track evolution, set-to-set differences, different drivers or cars, and mixed sessions are not modelled. They end up in the residuals or bias the estimates. The 25% and 3% guards and the "Poor fit" and "Possible mixed pace" notes catch clear cases only: a faster group of 1-2 laps, or a group that is the majority, is not flagged. When about 20-25% of the laps are a second population, the reported estimate describes the rest. Calibrate one car in one session's conditions.
- **The peak needs fresh-tyre laps.** If degradation is already under way at the youngest tyre age in the data (for example stints on used sets), the window end is placed at that age, and `base_performance` and the base lap time describe the pace there, not the peak. A note says so.
- **Outlier guards are thresholds.** 25% rejected laps and 3% residual scatter are fixed limits. A history just inside them is estimated (with notes where they apply), and one just outside is not.
- **Conditions are held constant.** One `weather` and one `track_temperature` apply to the whole history.
- **No uncertainty intervals.** Fit statistics are conditional on the chosen structure. Borderline laps near the outlier threshold can be rejected.

## Run the example and tests

From the repository root:

```bash
python -m core_modules.strategy_optimizer.example_usage
python -m pytest -q tests/test_strategy_engine.py tests/test_strategy_calibration.py
```

The tests cover:

- the last 1-5 laps in every weather, with and without tyre state;
- zero-stop availability;
- continuing on the current tyre;
- pit laps moving when degradation changes;
- optimality against brute force and against the old fixed-ratio split, including a global brute force over every ordered compound sequence with a distinct `pit_stop_delta` for each compound;
- the pit cost of each stop (the fitted compound's delta), the hot-track factor, the telemetry override, `laps_beyond_peak_window`, the performance-floor count, and the rule being waived by intermediate/wet tyres in the history;
- the reason given when no plan can satisfy the two-compound rule, and competitor signals at the window edges;
- invariants over random valid states;
- determinism and thread safety;
- NaN, negative, out-of-range and mismatched inputs;
- the schema round trip, a shared natural-query context with `sector_times`, the free-form fields, and the JSON-safe response;
- that the request schema and the engine reject each cross-field rule with one shared message (the engine adds the field's location to it).

The calibration tests (`tests/test_strategy_calibration.py`) generate lap histories with the engine's own model (`evaluate_plan`) and cover:

- exact recovery from noise-free data, including through the engine (the calibrated `tire_data` reproduces the history), a window end inside the warm-up, and the exact `peak_lap_time_s` and `initial_degradation_s_per_lap`;
- recovery within stated tolerances with noise, injected outliers and flagged laps, and of the degradation structure with 10-20% junk laps;
- the same best strategy and pit laps from calibrated and true parameters;
- hot-track and wet-weather factors, the fuel correction (including outliers reported in the raw lap-time frame), and supplied window ends and warm-ups;
- flat and falling histories, insufficient data, and every guard: fewer than 5 laps kept, more than 25% rejected, two pace levels, residual scatter above 3%, the performance floor (against the compound's own peak and against the reference), and the reference falling back to a compound within the limits;
- the notes: "Poor fit", "Possible mixed pace", slower laps beyond the oldest age used (flat fit and cliff), an unidentifiable peak, and alternating outlier rejection keeping the largest set;
- the small-sample rejection threshold, outlier and flag reasons, `tire_age_range` of the laps used, and excluded laps sorted by position;
- determinism and lap-order independence;
- invalid input through the function and the schema, with one shared message for each cross-field rule.

## Not modelled

The engine does not model:

- fuel burn, traffic or overtaking;
- safety-car timing and probability;
- track evolution and weather changes;
- tyre-temperature physics and ERS deployment;
- tyre-set allocation;
- team-specific models.

Use it to compare explicitly supplied assumptions, not as a validated simulator.
