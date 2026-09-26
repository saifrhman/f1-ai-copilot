"""Web UI form pages (race strategy, car setup, incident triage), submitted through the real API.

Each page runs in ``AppTest`` with its client bound to the FastAPI app, so the forms are
checked against the real request models, and every value asserted on the page is compared
with what the API itself returns for the request the page sent.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.ui_support import (  # noqa: F401 (pytest fixtures)
    ENTRY_POINT,
    assert_no_exception,
    column_formats,
    expander_labels,
    metrics,
    run_page,
    table_rows,
    ui_api,
    ui_api_down,
)
from ui.api_client import CALIBRATE_TYRES_PATH
from ui.components import documented_example, md_text

CALIBRATION_LABEL = "Calibrate tyre parameters from lap history"
HEURISTIC_BADGE = ":orange-badge[:material/science: Heuristic · not validated]"
EXAMPLE_BADGE = ":gray-badge[:material/lab_profile: Example inputs · not your data]"
# What the default strategy form sends: the mid-race example of the engine's documentation.
DEFAULT_STRATEGY_REQUEST = {
    "telemetry": {"lap_times": [95.6, 95.3, 95.9, 95.4]},
    "car_status": {"engine_wear": 0.3, "brake_wear": 0.4, "damage": {"front_wing": 0.1, "floor": 0.0, "diffuser": 0.0}},
    "driver_profile": {
        "tire_management": 0.7, "risk_tolerance": 0.6, "braking_consistency": 0.75, "throttle_aggressiveness": 0.7,
    },
    "tire_data": {
        compound: {"base_performance": base, "degradation_rate": rate, "warm_up_laps": warm_up,
                   "peak_performance_window": [start, end], "pit_stop_delta": 22.0}
        for compound, base, rate, warm_up, start, end in (
            ("soft", 1.0, 0.004, 2, 2, 10), ("medium", 0.992, 0.0025, 3, 3, 18), ("hard", 0.985, 0.0015, 4, 4, 28),
        )
    },
    "race_state": {"current_lap": 18, "total_laps": 57, "weather": "dry", "track_temperature": 32.0,
                   "current_compound": "medium", "current_tire_age": 17, "used_compounds": [], "own_gap_to_leader": 6.0},
    "competition": [
        {"driver_id": "HAM", "tire_compound": "medium", "tire_age": 21, "gap_to_leader": 4.4},
        {"driver_id": "VER", "tire_compound": "hard", "tire_age": 5, "gap_to_leader": 8.1},
    ],
}  # fmt: skip
# The strategy form's tyre table as the page lays it out (the peak window in two columns).
DEFAULT_TYRE_ROWS = [
    {"compound": compound, "window_start": entry["peak_performance_window"][0], "window_end": entry["peak_performance_window"][1],
     **{key: value for key, value in entry.items() if key != "peak_performance_window"}}
    for compound, entry in DEFAULT_STRATEGY_REQUEST["tire_data"].items()
]  # fmt: skip
WET_TYRE_ROW = {"compound": "wet", "base_performance": 0.9, "degradation_rate": 0.002, "warm_up_laps": 2,
                "window_start": 1, "window_end": 25, "pit_stop_delta": 22.0}  # fmt: skip
DEFAULT_TRIAGE_REQUEST = {
    "incident_type": "collision",
    "track_condition": "wet",
    "intent": "racing_incident",
    "driver_history": {
        "recent_penalties": ["5-second time penalty for track limits", "Grid drop for an unsafe release"],
        "total_penalties": 4,
    },
}


@pytest.fixture
def table_edits(monkeypatch) -> Dict[str, pd.DataFrame]:
    """Edited tables by key prefix: AppTest cannot type into a data editor, so its return value is replaced."""

    edits: Dict[str, pd.DataFrame] = {}
    real = st.data_editor

    def data_editor(data, **options):
        shown = real(data, **options)  # still rendered, with its own key
        key = str(options.get("key", ""))
        return next((frame for prefix, frame in edits.items() if key.startswith(prefix)), shown)

    monkeypatch.setattr(st, "data_editor", data_editor)
    return edits


def submit(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click().run()
    assert_no_exception(at)
    return at


def shown_text(at: AppTest) -> str:
    """The page's visible text, without the raw JSON expanders (an assertion must not pass on the API's JSON)."""

    kinds = ("title", "subheader", "markdown", "caption", "error", "warning", "info", "success")
    return "\n".join(str(element.value) for kind in kinds for element in getattr(at, kind))


def sent_json(at: AppTest, marker: str) -> Dict[str, Any]:
    """The JSON shown in the page's expanders whose top level has ``marker`` (request or response)."""

    bodies = [json.loads(element.value) for element in at.json]
    return next(body for body in bodies if marker in body)


def position(at: AppTest, kind: str, text: str) -> int:
    """Where the first ``kind`` element containing ``text`` is in the page (document order)."""

    return next(index for index, node in enumerate(at.main) if node.type == kind and text in str(node.value))


def plan_details(at: AppTest) -> Any:
    """The strategy result's only 'Plan details' selectbox (its key names the result it belongs to)."""

    (selectbox,) = [element for element in at.selectbox if element.label == "Plan details"]
    return selectbox


def chart_rows(at: AppTest) -> List[Dict[str, Any]]:
    """The data rows of the page's only chart (Streamlit sends them as an Arrow stream)."""

    (chart,) = at.get("vega_lite_chart")
    (dataset,) = chart.proto.datasets
    return pa.ipc.open_stream(dataset.data.data).read_all().to_pandas().to_dict("records")


# ------------------------------------------------------------------ race strategy


def test_strategy_form_sends_the_example_and_shows_the_api_plans(ui_api):
    at = run_page("strategy")
    assert_no_exception(at)
    assert at.title[0].value == "Race strategy"
    assert HEURISTIC_BADGE in shown_text(at)
    submit(at, "strategy_submit")

    request = sent_json(at, "telemetry")
    assert request == DEFAULT_STRATEGY_REQUEST
    expected = ui_api.generate_strategy(request)
    best, runner_up = expected["strategies"][0], expected["strategies"][1]
    assert best["strategy_id"] == expected["best_strategy_id"]
    assert f"**{md_text(best['strategy_id'])}** · {' → '.join(best['tire_compounds'])}" in shown_text(at)
    shown = metrics(at)
    assert shown["Pit stops"] == str(best["pit_stops"])
    assert shown["Pit laps"] == ", ".join(str(lap) for lap in best["pit_laps"])
    assert shown["Margin to plan 2"] == f"{runner_up['delta_to_best_s']:.3f} s"
    assert shown["Base lap time"] == f"{expected['model']['base_lap_time_s']:.3f} s"
    time_metric = next(metric for metric in at.metric if metric.label == "Projected time, laps 18-57")
    assert f"{best['projected_race_time']:.3f} s" in time_metric.proto.help

    plans = table_rows(at, "Projected (s)")
    assert list(plans[0])[:3] == ["Rank", "Plan", "Two-compound rule"]  # the rule is visible without scrolling
    for row, option in zip(plans, expected["strategies"], strict=True):
        assert row["Plan"] == option["strategy_id"]
        assert row["Two-compound rule"] == option["two_compound_rule"]
        assert row["Projected (s)"] == option["projected_race_time"]
        assert row["Gap to best (s)"] == option["delta_to_best_s"]
        assert row["Pit loss (s)"] == option["pit_time_loss_s"]
        assert option["projected_race_time"] == pytest.approx(option["driving_time_s"] + option["pit_time_loss_s"], abs=2e-3)
        assert row["Plan"].endswith("/".join(option["tire_compounds"]))  # the plan id names the compounds
    # Narrow enough for a laptop screen: no copy of the compounds in the plan id, the risk label is in the plan
    # details, and the driving time is the projected time minus the pit loss.
    assert list(plans[0]) == ["Rank", "Plan", "Two-compound rule", "Pit laps", "Projected (s)", "Gap to best (s)",
                              "Pit loss (s)"]  # fmt: skip
    plans_frame = next(frame for frame in at.dataframe if "Projected (s)" in frame.value.columns)
    assert column_formats(plans_frame) == dict.fromkeys(("Projected (s)", "Gap to best (s)", "Pit loss (s)"), "%.3f")
    # The Plan column is as wide as its longest id, so no id is cut off.
    longest = max(len(option["strategy_id"]) for option in expected["strategies"])
    assert json.loads(plans_frame.proto.columns)["Plan"]["width"] == round(20 + 7.0 * longest)
    # The tyre model editor's seven columns have short headers, so they fit a laptop screen.
    tyre_editor = next(frame for frame in at.dataframe if "base_performance" in frame.value.columns)
    headers = {name: config["label"] for name, config in json.loads(tyre_editor.proto.columns).items() if "label" in config}
    assert headers == {
        "compound": "Compound", "base_performance": "Base perf.", "degradation_rate": "Deg./lap", "warm_up_laps": "Warm-up",
        "window_start": "Peak from", "window_end": "Peak to", "pit_stop_delta": "Pit loss (s)",
    }  # fmt: skip
    assert not at.warning  # every plan satisfies the two-compound rule

    # The stint timeline: one bar per stint of every plan, from the lap before its first lap to its last lap.
    bars = chart_rows(at)
    stints = [(option, stint) for option in expected["strategies"] for stint in option["stint_breakdown"]]
    assert len(bars) == len(stints)
    for bar, (option, stint) in zip(bars, stints, strict=True):
        assert bar["Row"] == f"{option['rank']}. {option['pit_stops']}-stop"
        assert (bar["Plan"], bar["Compound"]) == (option["strategy_id"], stint["tire_compound"])
        assert bar["Laps"] == f"{stint['start_lap']}-{stint['end_lap']}"
        assert stint["start_lap"] - 1 < bar["From"] < stint["start_lap"] - 0.5
        assert stint["end_lap"] - 0.5 < bar["To"] < stint["end_lap"]
        assert bar["Laps on the set"] == f"{stint['tire_age_start']} → {stint['tire_age_end']}"

    # Laps on the set before the stint and after its last lap: the fitted set's 17 laps plus 8, a new set's 0 plus 16.
    stints = table_rows(at, "Stint time (s)")
    assert [row["Laps on the set"] for row in stints] == ["17 → 25", "0 → 16", "0 → 16"]
    stint_frame = next(frame for frame in at.dataframe if "Stint time (s)" in frame.value.columns)
    assert column_formats(stint_frame) == dict.fromkeys(
        ("Average lap (s)", "Best lap (s)", "Worst lap (s)", "Stint time (s)"), "%.3f"
    )
    wear = table_rows(at, "Laps at performance floor")  # the second stint table: tyre performance
    assert [row["Laps past peak window"] for row in wear] == [s["laps_beyond_peak_window"] for s in best["stint_breakdown"]]
    assert wear[0]["Performance start → end"] == (
        f"{best['stint_breakdown'][0]['start_performance']:.4f} → {best['stint_breakdown'][0]['end_performance']:.4f}"
    )
    assert f"Risk label (the API's, set from the number of stops only): {best['risk_level']}." in shown_text(at)

    signals = {row["Driver"]: row["Signal"] for row in table_rows(at, "Signal")}
    assert signals == {s["driver_id"]: s["signal"].replace("_", " ") for s in expected["competitor_signals"]}
    assert signals == {"HAM": "undercut target", "VER": "undercut threat"}
    assert "Explanation" not in table_rows(at, "Signal")[0]  # sentences are listed below the table, not cut off in it
    for signal in expected["competitor_signals"]:
        assert f"- **{signal['driver_id']}**: {md_text(signal['explanation'])}" in shown_text(at)
    text = shown_text(at)
    for assumption in expected["assumptions"]:
        assert md_text(assumption) in text
    assert expected["not_modelled_inputs"]
    assert "Accepted but not modelled: " + "; ".join(md_text(item) for item in expected["not_modelled_inputs"]) in text
    assert sent_json(at, "strategies") == expected  # the raw response in its expander


def test_strategy_plan_details_follow_the_selected_plan(ui_api):
    at = submit(run_page("strategy"), "strategy_submit")
    expected = ui_api.generate_strategy(sent_json(at, "telemetry"))
    second = expected["strategies"][1]
    plan_details(at).set_value(second["strategy_id"]).run()
    assert_no_exception(at)
    stints = table_rows(at, "Stint time (s)")
    assert [row["Laps"] for row in stints] == [f"{s['start_lap']}-{s['end_lap']}" for s in second["stint_breakdown"]]
    assert [row["Average lap (s)"] for row in stints] == [s["average_lap_time"] for s in second["stint_breakdown"]]
    assert md_text(second["notes"][0]) in shown_text(at)


def test_a_new_strategy_result_opens_its_own_recommended_plan(ui_api):
    at = submit(run_page("strategy"), "strategy_submit")
    first = sent_json(at, "strategies")
    first_widget = plan_details(at)
    assert first_widget.value == first["best_strategy_id"]
    at.number_input(key="strategy_current_lap").set_value(40)
    at.slider(key="strategy_brake_wear").set_value(0.0)
    submit(at, "strategy_submit")
    second = sent_json(at, "strategies")
    best = second["strategies"][0]
    assert best["strategy_id"] == second["best_strategy_id"] != first["best_strategy_id"]
    # The first result's plan is also in the new list, so a kept choice would still be shown.
    assert first["best_strategy_id"] in [option["strategy_id"] for option in second["strategies"]]
    # A new widget that starts at the new plan: a browser keeps showing a kept widget's old value, whatever its default.
    widget = plan_details(at)
    assert widget.id != first_widget.id
    assert widget.value == widget.options[widget.proto.default] == best["strategy_id"]
    stints = table_rows(at, "Stint time (s)")
    assert [row["Laps"] for row in stints] == [f"{s['start_lap']}-{s['end_lap']}" for s in best["stint_breakdown"]]
    assert f"Risk label (the API's, set from the number of stops only): {best['risk_level']}." in shown_text(at)


@pytest.mark.parametrize(
    ("page", "submit_key", "prefilled", "change"),
    [
        ("strategy", "strategy_submit", "Pre-filled with the engine's documented mid-race example (lap 18 of 57",
         lambda at: at.number_input(key="strategy_current_lap").set_value(19)),
        ("setup", "setup_submit", "Pre-filled with the API's documented example (Silverstone in the dry)",
         lambda at: at.number_input(key="setup_n_trials").set_value(24)),
        ("triage", "triage_submit", "Pre-filled with an example incident (a collision in the wet",
         lambda at: at.selectbox(key="triage_intent").set_value("unknown")),
    ],
)  # fmt: skip
def test_example_inputs_are_announced_and_their_results_labelled(ui_api, page, submit_key, prefilled, change):
    at = run_page(page)
    assert prefilled in shown_text(at) and EXAMPLE_BADGE not in shown_text(at)
    submit(at, submit_key)
    assert shown_text(at).count(EXAMPLE_BADGE) == 1  # the result of the unchanged example
    change(at)
    submit(at, submit_key)
    assert EXAMPLE_BADGE not in shown_text(at) and HEURISTIC_BADGE in shown_text(at)


def test_strategy_leaves_out_optional_inputs_and_reads_lap_clock_times(ui_api):
    at = run_page("strategy")
    at.selectbox(key="strategy_current_compound").set_value("Fresh set (not specified)")
    at.checkbox(key="strategy_history_known").uncheck()
    at.number_input(key="strategy_own_gap").set_value(None)
    at.number_input(key="strategy_measured_braking").set_value(0.9)
    at.text_input(key="strategy_lap_times").set_value("1:35.6; 95.3 1:35.9")
    submit(at, "strategy_submit")

    request = sent_json(at, "telemetry")
    assert request["telemetry"] == {"lap_times": [95.6, 95.3, 95.9], "braking_consistency": 0.9}
    assert set(request["race_state"]) == {"current_lap", "total_laps", "weather", "track_temperature"}
    response = sent_json(at, "strategies")
    assert response["tire_state"] == "assumed_fresh" and metrics(at)["Tyre state"] == "assumed fresh"
    assert md_text(response["competitor_signals_note"]) in shown_text(at)
    assert response["competitor_signals_note"].startswith("Not computed")


def test_strategy_validation_errors_name_the_form_fields(ui_api):
    at = run_page("strategy")
    at.number_input(key="strategy_current_lap").set_value(70)
    at.number_input(key="strategy_total_laps").set_value(57)
    submit(at, "strategy_submit")
    assert "rejected by the API (HTTP 422)" in at.error[0].value
    assert "- **Race state**: Current lap (70) must be \\<= Total laps (57)" in shown_text(at)
    assert not at.metric  # nothing is shown in place of the missing result

    at.number_input(key="strategy_current_lap").set_value(18)
    at.text_input(key="strategy_lap_times").set_value("95.6, 5.0")
    submit(at, "strategy_submit")
    assert "**Telemetry › Recent lap times › entry 2**: Input should be greater than or equal to 20" in shown_text(at)


def test_edited_tables_are_sent_and_kept_after_a_page_switch(ui_api, table_edits):
    tyres = pd.DataFrame([{**row, "degradation_rate": 0.006} if row["compound"] == "soft" else row for row in DEFAULT_TYRE_ROWS])
    table_edits["strategy_tyres"] = tyres
    table_edits["strategy_rivals"] = pd.DataFrame(
        [{"driver_id": "LEC", "tire_compound": "soft", "tire_age": None, "gap_to_leader": 5.0}]
    )
    at = run_page(ENTRY_POINT)
    at.switch_page("views/strategy.py").run()
    submit(at, "strategy_submit")
    assert "rejected by the API (HTTP 422)" in at.error[0].value  # the rival row lacks its tyre age
    assert "**Competitors › row 1 › Tyre age**: Field required" in shown_text(at)

    table_edits["strategy_rivals"] = pd.DataFrame(
        [{"driver_id": None, "tire_compound": None, "tire_age": None, "gap_to_leader": None}]
    )
    submit(at, "strategy_submit")
    first = sent_json(at, "telemetry")
    assert first["tire_data"]["soft"]["degradation_rate"] == 0.006 and first["competition"] == []  # empty rows are dropped

    table_edits.clear()  # the tables now start from the rows kept at the last submit
    at.switch_page("views/setup.py").run()
    at.switch_page("views/strategy.py").run()
    submit(at, "strategy_submit")
    assert sent_json(at, "telemetry") == first


def test_tyre_rows_that_cannot_be_keyed_are_not_sent(ui_api, table_edits):
    table_edits["strategy_tyres"] = pd.DataFrame(
        [DEFAULT_TYRE_ROWS[0], DEFAULT_TYRE_ROWS[0], {**DEFAULT_TYRE_ROWS[1], "compound": None}]
    )
    at = submit(run_page("strategy"), "strategy_submit")
    assert at.error[0].value == "Nothing was sent to the API. Please fix:"
    text = shown_text(at)
    assert f"- {md_text('Tyre model: soft has more than one row; keep one.')}" in text
    assert f"- {md_text('Tyre model, row 3: choose a compound.')}" in text
    assert not at.json


def test_unreadable_lap_times_are_not_sent(ui_api):
    at = run_page("strategy")
    at.text_input(key="strategy_lap_times").set_value("95.6, fast, nan")
    submit(at, "strategy_submit")
    assert at.error[0].value == "Nothing was sent to the API. Please fix:"
    assert "'fast', 'nan' is not a lap time" in shown_text(at)
    assert not at.json and not at.metric


def test_peak_window_errors_name_the_window_columns(ui_api, table_edits):
    table_edits["strategy_tyres"] = pd.DataFrame(
        [{**row, "window_start": 0} if row["compound"] == "hard" else row for row in DEFAULT_TYRE_ROWS]
    )
    at = submit(run_page("strategy"), "strategy_submit")
    assert "rejected by the API (HTTP 422)" in at.error[0].value
    text = shown_text(at)
    assert "- **Tyre model › hard › Peak from**: Input should be greater than or equal to 1 (got 0)" in text
    assert "row 1" not in text  # the index is the window start, not a table row

    table_edits["strategy_tyres"] = pd.DataFrame(
        [{**row, "window_start": 12} if row["compound"] == "soft" else row for row in DEFAULT_TYRE_ROWS]
    )
    submit(at, "strategy_submit")
    assert "- **Tyre model › soft**: Peak window start (12) must be \\<= end (10)" in shown_text(at)


def test_recommended_plan_warns_when_it_breaks_the_two_compound_rule(ui_api, table_edits):
    at = run_page("strategy")
    at.selectbox(key="strategy_weather").set_value("wet")  # only dry compounds in the tyre model
    submit(at, "strategy_submit")
    best = sent_json(at, "strategies")["strategies"][0]
    assert best["two_compound_rule"] == "violated"
    (warning,) = at.warning
    assert warning.value.startswith("**The recommended plan breaks the simplified two-compound rule.**")
    assert [note for note in best["notes"] if "rule" in note.lower()] == [
        "Simplified two-dry-compound rule: VIOLATED. The known tyre history and this plan use only one dry compound.",
        "Shown although it violates the simplified rule, because only one dry compound (medium) is known or available: "
        "tire_data offers no second dry compound (and no intermediate/wet tyre) for a stop in wet conditions, so no plan "
        "can use two different dry compounds.",
    ]
    # Both quoted, with the form's name for tire_data.
    assert md_text("rule: VIOLATED. The known tyre history and this plan use only one dry compound.") in warning.value
    assert md_text("available: Tyre model offers no second dry compound") in warning.value
    assert "Add an intermediate or wet row to the tyre model table" in warning.value
    # Next to the recommended plan, not only in the notes further down.
    assert (
        position(at, "subheader", "Recommended plan")
        < position(at, "warning", "breaks the simplified")
        < position(at, "markdown", "Plans compared")
    )

    at.selectbox(key="strategy_current_compound").set_value("Fresh set (not specified)")
    submit(at, "strategy_submit")  # the engine refuses: no fitted tyre and no tyre for the conditions
    assert "rejected by the API (HTTP 422)" in at.error[0].value
    text = shown_text(at)
    assert "- No strategy could be generated\\: Tyre model has no compound usable in wet conditions" in text
    assert "tire\\_data" not in text
    assert "Add an intermediate or wet row to the tyre model table" in at.info[0].value
    assert not at.metric

    table_edits["strategy_tyres"] = pd.DataFrame([*DEFAULT_TYRE_ROWS, WET_TYRE_ROW])
    submit(at, "strategy_submit")
    response = sent_json(at, "strategies")
    assert response["strategies"][0]["tire_compounds"][0] == "wet"
    assert response["strategies"][0]["two_compound_rule"] == "waived"
    assert not at.warning and not at.info


def test_recommended_plan_says_when_the_rule_cannot_be_checked(ui_api):
    at = run_page("strategy")
    at.number_input(key="strategy_current_lap").set_value(55)
    at.checkbox(key="strategy_history_known").uncheck()
    submit(at, "strategy_submit")
    best = sent_json(at, "strategies")["strategies"][0]
    assert best["two_compound_rule"] == "unverified"
    (info,) = at.info
    assert "could not be checked for the recommended plan" in info.value and "Tyre history known" in info.value
    assert "(supply race_state.used_compounds)" in next(note for note in best["notes"] if "rule" in note.lower())
    assert md_text("(supply Race state › Compounds used before the fitted set)") in info.value
    assert not at.warning


# ------------------------------------------------------------------ tyre calibration (optional endpoint)


def test_calibration_is_offered_when_the_api_documents_it(ui_api):
    assert "post" in ui_api.get_openapi()["paths"][CALIBRATE_TYRES_PATH]
    at = run_page("strategy")
    assert_no_exception(at)
    assert CALIBRATION_LABEL in expander_labels(at)
    assert at.button(key="calibration_submit").label == "Estimate tyre parameters"


def test_calibration_is_hidden_for_an_api_without_the_endpoint(ui_api, monkeypatch):
    older = copy.deepcopy(ui_api.get_openapi())
    del older["paths"][CALIBRATE_TYRES_PATH]
    monkeypatch.setattr(ui_api, "get_openapi", lambda: older)
    at = run_page("strategy")
    assert_no_exception(at)
    assert CALIBRATION_LABEL not in expander_labels(at)
    assert "calibration_submit" not in [button.key for button in at.button]
    submit(at, "strategy_submit")  # the strategy form works as before
    assert md_text(sent_json(at, "strategies")["best_strategy_id"]) in shown_text(at)


def test_calibration_estimates_the_documented_example_and_fills_the_strategy_form(ui_api):
    at = submit(run_page("strategy"), "calibration_submit")
    example = documented_example(ui_api.get_openapi(), "TyreCalibrationRequest")
    request = sent_json(at, "laps")
    assert request["laps"] == [{key: value for key, value in lap.items() if value is not False} for lap in example["laps"]]
    assert {key: request[key] for key in ("weather", "track_temperature", "pit_stop_delta")} == {
        key: example[key] for key in ("weather", "track_temperature", "pit_stop_delta")
    }
    expected = ui_api.calibrate_tyres(request)
    assert sent_json(at, "compounds") == expected
    shown = metrics(at)
    assert shown["Estimated base lap time"] == f"{expected['estimated_base_lap_time_s']:.3f} s"
    assert shown["Reference compound"] == expected["reference_compound"]
    assert shown["Laps excluded"] == str(len(expected["excluded_laps"]))
    assert "synthetic laps generated from the engine's own lap-time model" in shown_text(at)
    # Two narrow tables (fit quality first, then the estimated model) instead of one that is cut off on a desktop.
    fits = {row["Compound"]: row for row in table_rows(at, "R²")}
    models = {row["Compound"]: row for row in table_rows(at, "Degradation per lap")}
    assert list(next(iter(fits.values()))) == [
        "Compound",
        "Status",
        "R²",
        "Residual SD (s)",
        "Laps used / clean / supplied",
        "Outliers",
    ]
    assert shown_text(at).index("**Fit per compound**") < shown_text(at).index("**Estimated tyre model**")
    assert len(next(iter(models.values()))) == 7
    for compound, entry in expected["compounds"].items():
        fit, tire = entry["fit"], entry["tire_data"]
        assert fits[compound]["Status"] == entry["status"]
        assert (fits[compound]["R²"], fits[compound]["Residual SD (s)"]) == (fit["r_squared"], fit["residual_std_s"])
        assert fits[compound]["Outliers"] == entry["outliers_rejected"]
        assert models[compound]["Degradation per lap"] == tire["degradation_rate"]
        assert models[compound]["Peak window"] == "-".join(map(str, tire["peak_performance_window"])) + " (detected)"
        assert (models[compound]["Peak lap (s)"], models[compound]["Initial loss (s/lap)"]) == (
            fit["peak_lap_time_s"],
            fit["initial_degradation_s_per_lap"],
        )
    assert [row["Reason"] for row in table_rows(at, "Residual (s)")] == [
        lap["reason"].replace("_", " ") for lap in expected["excluded_laps"]
    ]

    submit(at, "calibration_apply")
    assert at.text_input(key="strategy_lap_times").value == str(expected["estimated_base_lap_time_s"])
    assert "now uses the calibrated tyre model (soft, medium)" in at.success[0].value
    # The example history has no hard laps: the removed row is named, not dropped silently.
    assert set(expected["tire_data"]) == {"soft", "medium"}
    assert [warning.value for warning in at.warning] == [
        "**hard was removed from the tyre model:** the calibration estimated no parameters for it, so the plans "
        "cannot use it. To keep a compound available, add its row back to the tyre model table with values relative "
        "to the new base lap time."
    ]
    submit(at, "strategy_submit")
    strategy_request = sent_json(at, "telemetry")
    assert strategy_request["telemetry"]["lap_times"] == [expected["estimated_base_lap_time_s"]]
    assert strategy_request["tire_data"] == {
        compound: {key: value for key, value in entry.items() if key != "compound"}
        for compound, entry in expected["tire_data"].items()
    }
    response = sent_json(at, "strategies")
    assert response == ui_api.generate_strategy(strategy_request)
    # The default car has front-wing damage: its penalty is counted on top of the calibrated pace.
    assert response["model"]["damage_multiplier"] == 1.0025 and response["model"]["driver_multiplier"] == 1.0
    (warning,) = at.warning
    assert "counted twice" in warning.value and "damage/wear multiplier 1.0025" in warning.value
    assert "driver multiplier" not in warning.value

    at.slider(key="strategy_front_wing").set_value(0.0)
    submit(at, "strategy_submit")
    assert sent_json(at, "strategies")["model"]["damage_multiplier"] == 1.0
    assert not at.warning  # penalty-free: nothing is counted twice


def test_inputs_tables_and_results_survive_a_page_switch(ui_api):
    at = run_page(ENTRY_POINT)
    at.switch_page("views/strategy.py").run()
    at.number_input(key="strategy_current_lap").set_value(30)
    at.number_input(key="strategy_own_gap").set_value(None)
    submit(at, "calibration_submit")
    submit(at, "calibration_apply")  # the tyre table now holds the calibrated soft and medium rows
    submit(at, "strategy_submit")
    first = sent_json(at, "telemetry")
    assert set(first["tire_data"]) == {"soft", "medium"} and "own_gap_to_leader" not in first["race_state"]

    at.switch_page("views/triage.py").run()
    at.switch_page("views/strategy.py").run()
    assert_no_exception(at)
    assert at.number_input(key="strategy_current_lap").value == 30
    assert at.number_input(key="strategy_own_gap").value is None
    assert sent_json(at, "telemetry") == first  # the last result is still shown
    assert "Estimated base lap time" in metrics(at)  # and the last calibration
    submit(at, "strategy_submit")
    assert sent_json(at, "telemetry") == first  # the same inputs, tyre table included


def test_calibration_sends_the_fixed_values_table(ui_api, table_edits):
    table_edits["calibration_overrides"] = pd.DataFrame(
        [
            {"compound": "soft", "peak_window_end": 5, "warm_up_laps": None},
            {"compound": "medium", "peak_window_end": None, "warm_up_laps": 0},
        ]
    )
    at = submit(run_page("strategy"), "calibration_submit")
    request = sent_json(at, "laps")
    assert request["peak_window_end"] == {"soft": 5} and request["warm_up_laps"] == {"medium": 0}
    expected = ui_api.calibrate_tyres(request)
    rows = {row["Compound"]: row for row in table_rows(at, "Degradation per lap")}
    assert expected["compounds"]["soft"]["peak_window_end_source"] == "supplied"
    assert rows["soft"]["Peak window"] == "1-5 (supplied)"
    assert rows["medium"]["Warm-up laps"] == "0 (supplied)"


def test_calibration_checks_race_laps_and_fixed_values_before_sending(ui_api, table_edits):
    table_edits["calibration_laps"] = pd.DataFrame(
        [
            {"compound": "soft", "tire_age": age, "lap_time": 80.0 + 0.1 * age, "race_lap": None if age in (1, 3, 4) else age,
             "pit_out": age == 1, "pit_in": False, "safety_car": False}
            for age in range(1, 9)
        ]
    )  # fmt: skip
    table_edits["calibration_overrides"] = pd.DataFrame(
        [
            {"compound": "soft", "peak_window_end": 5, "warm_up_laps": None},
            {"compound": "soft", "peak_window_end": 9, "warm_up_laps": None},
            {"compound": None, "peak_window_end": 6, "warm_up_laps": None},
        ]
    )
    at = run_page("strategy")
    at.number_input(key="calibration_fuel_correction").set_value(0.05)
    submit(at, "calibration_submit")
    assert at.error[0].value == "Nothing was sent to the API. Please fix:"
    text = shown_text(at)
    # 1-based table rows; the flagged out-lap (row 1) needs no race lap.
    assert md_text("Lap history, rows 3, 4: enter the race lap.") in text
    assert md_text("Fixed values: soft has more than one row; keep one.") in text  # not silently the last one
    assert md_text("Fixed values, row 3: choose a compound.") in text
    assert "null" not in text
    assert not at.json and not at.metric


def test_calibration_validation_errors_name_the_form_fields(ui_api):
    at = run_page("strategy")
    at.number_input(key="calibration_track_temperature").set_value(None)
    at.number_input(key="calibration_fuel_correction").set_value(0.9)
    submit(at, "calibration_submit")
    assert "The calibration request was rejected by the API (HTTP 422)" in at.error[0].value
    text = shown_text(at)
    assert "**Track temperature**: Field required" in text
    assert "**Fuel correction**: Input should be less than or equal to 0.5 (got 0.9)" in text


# ------------------------------------------------------------------ car setup


def test_setup_form_sends_the_documented_example_and_shows_the_api_result(ui_api):
    at = run_page("setup")
    assert_no_exception(at)
    assert at.title[0].value == "Car setup"
    submit(at, "setup_submit")

    request = sent_json(at, "track_profile")
    assert request == documented_example(ui_api.get_openapi(), "SetupRequest")
    expected = ui_api.recommend_setup(request)  # deterministic for the same inputs and seed
    assert sent_json(at, "objective_value") == expected
    text = shown_text(at)
    assert text.count(HEURISTIC_BADGE) == 2  # the page and its result
    shown = metrics(at)
    assert shown["Objective (lower is better)"] == f"{expected['objective_value']:.3f}"
    assert shown["Rule-of-thumb baseline"] == f"{expected['baseline_objective_value']:.3f}"
    improvement = expected["improvement_over_baseline"]
    reduction = next(metric for metric in at.metric if metric.label == "Reduction vs baseline")
    assert reduction.value == f"{improvement:.3f}"
    assert reduction.delta == f"{improvement / expected['baseline_objective_value']:.1%} lower objective"
    assert "not a predicted lap-time gain" in text  # visible, not only in a tooltip
    assert shown["Multi-start agreement"] == f"{expected['confidence']:.2f}"
    assert shown["Front left"] == f"{expected['tire_pressures_psi']['front_left']:.2f}"
    for name, value in expected["handling_balance"].items():  # sign kept: > 0 understeer, < 0 oversteer
        assert shown[name.replace("_", " ").capitalize()] == f"{value:+.4f}"

    rows = {row["Parameter"]: row for row in table_rows(at, "Baseline")}
    assert rows["Ride height"]["Recommended"] == expected["ride_height"]
    assert rows["Ride height"]["Baseline"] == expected["baseline_setup"]["ride_height"]
    assert rows["Differential off throttle"]["Recommended"] == expected["diff_settings"]["coast"]
    assert rows["Rear springs"]["Unit"] == expected["units"]["suspension_settings"]
    for row in rows.values():
        assert row["Change"] == round(row["Recommended"] - row["Baseline"], 2)
    assert rows["Rear anti-roll bar"]["Change"] == round(
        expected["suspension_settings"]["rear_arb"] - expected["baseline_setup"]["suspension_settings"]["rear_arb"], 2
    )
    assert not any(row["Pinned"] for row in rows.values())
    assert 'The API calls the agreement value "confidence"' in text and "not a probability that the setup is right" in text
    assert md_text(expected["reasoning"]) in text
    assert "**Assumed defaults:** none" in text
    # The default form sends humidity but no wind speed, so wind speed is not listed as accepted.
    assert expected["inputs_not_modelled"] == ["track_profile.track_name (label only)", "weather.humidity"]
    assert "Accepted but not modelled: " + "; ".join(md_text(item) for item in expected["inputs_not_modelled"]) in text
    assert "wind_speed" not in text


def test_setup_reports_assumed_defaults_and_pinned_values(ui_api):
    at = run_page("setup")
    at.number_input(key="setup_risk_tolerance").set_value(None)
    at.number_input(key="setup_pin_front_wing").set_value(5.0)
    at.number_input(key="setup_n_trials").set_value(24)
    submit(at, "setup_submit")
    request = sent_json(at, "track_profile")
    assert request["driver_preferences"] == {"tire_management": 0.7, "preferred_wing_angles": {"front": 5.0}}
    expected = ui_api.recommend_setup(request)
    assert expected["assumed_defaults"] == ["driver_preferences.risk_tolerance=0.5"]
    assert "**Assumed defaults:** `driver_preferences.risk_tolerance=0.5`" in shown_text(at)
    row = next(row for row in table_rows(at, "Baseline") if row["Parameter"] == "Front wing angle")
    assert row["Pinned"] and row["Recommended"] == 5.0 == expected["front_wing_angle"]


@pytest.mark.parametrize(
    ("key", "value", "line"),
    [
        ("setup_high_speed_sections", 30, "- **Track**: High-speed corners must be between 0 and 18, got 30"),
        ("setup_track_length", 100.0, "- **Track › Lap length**: Input should be greater than or equal to 500 (got 100.0)"),
    ],
)
def test_setup_validation_errors_name_the_form_fields(ui_api, key, value, line):
    at = run_page("setup")
    at.number_input(key=key).set_value(value)
    submit(at, "setup_submit")
    assert "The setup request was rejected by the API (HTTP 422)" in at.error[0].value
    assert line in shown_text(at)
    assert not at.metric


def test_setup_shows_each_new_result_and_never_a_stale_one(ui_api):
    at = submit(run_page("setup"), "setup_submit")
    first = sent_json(at, "objective_value")
    at.selectbox(key="setup_condition").set_value("wet")
    at.number_input(key="setup_n_trials").set_value(24)
    submit(at, "setup_submit")
    request = sent_json(at, "track_profile")
    assert request["weather"]["condition"] == "wet" and request["n_trials"] == 24
    second = ui_api.recommend_setup(request)
    assert second["objective_value"] != first["objective_value"]
    assert sent_json(at, "objective_value") == second
    assert metrics(at)["Objective (lower is better)"] == f"{second['objective_value']:.3f}"
    rows = {row["Parameter"]: row for row in table_rows(at, "Baseline")}
    assert rows["Rear wing angle"]["Recommended"] == second["rear_wing_angle"]

    at.number_input(key="setup_track_length").set_value(100.0)
    submit(at, "setup_submit")
    assert "rejected by the API (HTTP 422)" in at.error[0].value
    assert not at.metric and not at.dataframe and len(at.json) == 1  # only the rejected request


def test_setup_marks_values_at_the_end_of_their_scale(ui_api):
    at = run_page("setup")
    at.number_input(key="setup_pin_diff_preload").set_value(0.0)
    submit(at, "setup_submit")
    expected = ui_api.recommend_setup(sent_json(at, "track_profile"))
    rows = {row["Parameter"]: row for row in table_rows(at, "Baseline")}
    # The engine's documented corner solutions for this track: softest rear bar, stiffest rear springs.
    assert (expected["suspension_settings"]["rear_arb"], expected["suspension_settings"]["rear_spring"]) == (0.0, 100.0)
    assert rows["Rear anti-roll bar"]["Search limit"] == "lowest (0)"
    assert rows["Rear springs"]["Search limit"] == "highest (100)"
    preload = rows["Differential preload"]
    assert preload["Pinned"] and preload["Recommended"] == 0.0 and preload["Search limit"] == ""  # the driver's choice
    for row in rows.values():
        if row["Recommended"] not in (0.0, 100.0):
            assert row["Search limit"] == ""
    assert "read such a value as a direction, not as a tuned setting" in shown_text(at)


# ------------------------------------------------------------------ incident triage


def test_triage_is_labelled_and_shows_the_api_result(ui_api):
    at = run_page("triage")
    assert_no_exception(at)
    assert at.title[0].value == "Incident triage"
    assert "not an FIA steward-decision predictor" in at.warning[0].value
    assert "never cites FIA articles" in at.warning[0].value
    assert [(link.proto.label, link.proto.page) for link in at.get("page_link")] == [("FIA regulations", "regulations")]
    submit(at, "triage_submit")

    request = sent_json(at, "incident_type")
    assert request == DEFAULT_TRIAGE_REQUEST
    expected = ui_api.predict_penalty(request)
    assert expected["triage_category"] == "driving_incident_review"
    assert "#### Review category: driving incident review" in shown_text(at)  # wrapping text, not a metric
    shown = metrics(at)
    assert "Review category" not in shown
    assert shown["Severity band"] == expected["severity_band"]
    assert shown["Severity score"] == f"{expected['severity_score']:.2f}"
    assert shown["Confidence (input completeness)"] == f"{expected['confidence']:.2f}"
    assert table_rows(at, "Change") == [
        {"Input": item["source"].replace("_", " "), "Value": str(item["value"]).replace("_", " "), "Change": item["delta"]}
        for item in expected["severity_adjustments"]
    ]
    text = shown_text(at)
    assert md_text(expected["reasoning"]) in text
    assert expected["referenced_rule"] is None and "never cites FIA articles" in text
    assert "not a probability of any steward decision" in text


def test_triage_without_history_and_with_history_adjustments(ui_api):
    at = run_page("triage")
    at.text_area(key="triage_recent_penalties").set_value("")
    at.number_input(key="triage_total_penalties").set_value(None)
    submit(at, "triage_submit")
    assert "driver_history" not in sent_json(at, "incident_type")
    assert "driver history not supplied" in shown_text(at)

    at.selectbox(key="triage_incident_type").set_value("dangerous_driving")
    at.selectbox(key="triage_intent").set_value("intentional")
    at.text_area(key="triage_recent_penalties").set_value("Reprimand\n\nTime penalty\nGrid drop")
    at.number_input(key="triage_total_penalties").set_value(12)
    submit(at, "triage_submit")
    request = sent_json(at, "incident_type")
    assert request["driver_history"] == {"recent_penalties": ["Reprimand", "Time penalty", "Grid drop"], "total_penalties": 12}
    expected = ui_api.predict_penalty(request)
    assert metrics(at)["Severity band"] == expected["severity_band"] == "high"
    assert [row["Input"] for row in table_rows(at, "Change")] == [
        "intent",
        "driver history.recent penalties",
        "driver history.total penalties",
    ]


def test_triage_sends_a_partial_history_and_never_shows_a_stale_result(ui_api):
    at = run_page("triage")
    at.text_area(key="triage_recent_penalties").set_value("")
    at.number_input(key="triage_total_penalties").set_value(11)
    submit(at, "triage_submit")
    request = sent_json(at, "incident_type")
    assert request["driver_history"] == {"total_penalties": 11}
    expected = ui_api.predict_penalty(request)
    assert metrics(at)["Severity score"] == f"{expected['severity_score']:.2f}"
    assert "driver history.total penalties" in [row["Input"] for row in table_rows(at, "Change")]

    at.text_area(key="triage_recent_penalties").set_value("a\nb\nc")
    at.number_input(key="triage_total_penalties").set_value(1)
    submit(at, "triage_submit")
    assert "rejected by the API (HTTP 422)" in at.error[0].value
    assert not at.metric and not at.dataframe and "Review category" not in shown_text(at)


def test_triage_validation_errors_name_the_form_fields(ui_api):
    at = run_page("triage")
    at.text_area(key="triage_recent_penalties").set_value("a\nb\nc")
    at.number_input(key="triage_total_penalties").set_value(1)
    submit(at, "triage_submit")
    assert "The triage request was rejected by the API (HTTP 422)" in at.error[0].value
    assert (
        "- **Driver history › Total penalties on record** (1) cannot be smaller than the number of Recent penalties (3)"
    ) in shown_text(at)
    assert not at.metric


# ------------------------------------------------------------------ API not running


@pytest.mark.parametrize(
    ("page", "submit_key", "action"),
    [
        ("strategy", "strategy_submit", "The strategy request"),
        ("setup", "setup_submit", "The setup request"),
        ("triage", "triage_submit", "The triage request"),
    ],
)
def test_forms_explain_an_unreachable_api(ui_api_down, page, submit_key, action):
    at = run_page(page)
    assert_no_exception(at)
    assert CALIBRATION_LABEL not in expander_labels(at)
    submit(at, submit_key)
    assert at.error[0].value.startswith(f"{action} could not reach the API at `{ui_api_down.base_url}`: connection failed")
    assert "uvicorn app.main:app" in at.info[0].value
    assert not at.metric
