# F1 Strategy Engine

This module generates and ranks Formula 1 race-strategy candidates from supplied telemetry, tyre characteristics, race state, car condition, driver parameters and competitor context.

It is a **transparent heuristic simulator**, not a validated vehicle-dynamics model and not a claim of optimal real-world race strategy. The purpose is to make the assumptions inspectable and to provide a usable strategy component for the F1 AI Copilot demo.

## What it models

- complete multi-stint allocation over the remaining race distance
- soft, medium and hard candidates in dry conditions
- intermediate and wet candidates when the race state is non-dry
- configurable per-compound base performance, degradation, warm-up and pit-stop time loss
- simple track-temperature effects
- driver braking/throttle penalties from supplied telemetry/profile values
- simple front-wing/floor/diffuser, engine-wear and brake-wear time penalties
- undercut/overcut opportunity labels from supplied competitor gaps and tyre age
- deterministic candidate ranking

## What it does not model

It does not currently simulate fuel burn, traffic at every lap, stochastic safety-car timing, tyre-temperature physics, aerodynamic maps, ERS deployment, detailed weather transitions or team-specific race models. Confidence values are data-completeness/heuristic scores, not calibrated probabilities.

## Run the example

From the repository root:

```bash
python -m core_modules.strategy_optimizer.example_usage
```

## Run the tests

From the repository root:

```bash
python -m pytest -q core_modules/strategy_optimizer/test_strategy_engine.py
```

The regression tests cover dry, wet and damaged-car scenarios and verify that every candidate accounts for every remaining lap and contains one stint per requested compound.

## Main API

```python
from core_modules.strategy_optimizer.strategy_engine import generate_strategy

strategies = generate_strategy(
    telemetry=telemetry,
    car_status=car_status,
    driver_profile=driver_profile,
    tire_data=tire_data,
    race_state=race_state,
    competition=competition,
)
```

`generate_strategy` returns a ranked list of `StrategyOption` values containing:

- `strategy_id`
- `projected_race_time`
- `confidence_score`
- `stint_breakdown`
- `tire_compounds`
- `pit_laps`
- `undercut_opportunities`
- `overcut_opportunities`
- `notes`
- `risk_level`

Each stint includes its start/end lap, lap count, compound, average/best/worst estimated lap time, performance trend, total estimated time and end-of-stint wear proxy.

## Scope

Use this module for experimentation, API demonstrations and comparisons between explicitly supplied assumptions. Do not treat its projected time or strategy ranking as authoritative motorsport engineering advice without validation against a higher-fidelity simulator and real data.
