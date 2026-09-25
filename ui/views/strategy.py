"""Race strategy: ranked pit-stop plans from the API's heuristic strategy engine.

Every number shown comes from ``POST /api/strategy/generate``. When the API offers
``POST /api/strategy/calibrate-tyres`` (checked in its OpenAPI schema), the tyre model can
first be estimated from lap history and copied into the form.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from ui.api_client import CALIBRATE_TYRES_PATH, ApiError, ApiUnavailable, get_client
from ui.components import (
    FieldLabels,
    api_schema,
    documented_example,
    example_inputs_badge,
    heuristic_badge,
    json_expander,
    md_text,
    offers_endpoint,
    show_request_error,
    unchanged_example,
)

COMPOUNDS = ("soft", "medium", "hard", "intermediate", "wet")
WEATHER = ("dry", "intermediate", "wet")
FRESH_SET = "Fresh set (not specified)"
# Pirelli's compound colours (the white hard tyre drawn in grey); each stint bar also carries its letter.
COMPOUND_COLORS = {"soft": "#e34948", "medium": "#eda100", "hard": "#8a8f98", "intermediate": "#008300", "wet": "#2a78d6"}
LETTER_INK = {"soft": "#ffffff", "medium": "#0b0b0b", "hard": "#0b0b0b", "intermediate": "#ffffff", "wet": "#ffffff"}
STINT_GAP_LAPS = 0.15  # visible gap between stints (a pit stop) in the timeline
# Approximate width of one character and a cell's padding in a table (px): a plan id column sized from its
# longest id is never cut off (the table otherwise sizes a column from a sample of its rows).
TABLE_CHAR_PX, TABLE_CELL_PADDING_PX = 7.0, 20

# The mid-race example of the engine's documentation (core_modules/strategy_optimizer/example_usage.py).
FORM_DEFAULTS: Dict[str, Any] = {
    "strategy_current_lap": 18,
    "strategy_total_laps": 57,
    "strategy_weather": "dry",
    "strategy_track_temperature": 32.0,
    "strategy_current_compound": "medium",
    "strategy_tyre_age": 17,
    "strategy_used_compounds": [],
    "strategy_history_known": True,
    "strategy_own_gap": 6.0,
    "strategy_lap_times": "95.6, 95.3, 95.9, 95.4",
    "strategy_tire_management": 0.7,
    "strategy_risk_tolerance": 0.6,
    "strategy_braking_consistency": 0.75,
    "strategy_throttle_aggressiveness": 0.7,
    "strategy_measured_braking": None,
    "strategy_measured_throttle": None,
    "strategy_engine_wear": 0.3,
    "strategy_brake_wear": 0.4,
    "strategy_front_wing": 0.1,
    "strategy_floor": 0.0,
    "strategy_diffuser": 0.0,
}
TYRE_COLUMNS = (
    "compound",
    "base_performance",
    "degradation_rate",
    "warm_up_laps",
    "window_start",
    "window_end",
    "pit_stop_delta",
)
DEFAULT_TYRES = [
    dict(zip(TYRE_COLUMNS, values, strict=True))
    for values in (
        ("soft", 1.0, 0.004, 2, 2, 10, 22.0),
        ("medium", 0.992, 0.0025, 3, 3, 18, 22.0),
        ("hard", 0.985, 0.0015, 4, 4, 28, 22.0),
    )
]
RIVAL_COLUMNS = ("driver_id", "tire_compound", "tire_age", "gap_to_leader")
DEFAULT_RIVALS = [
    {"driver_id": "HAM", "tire_compound": "medium", "tire_age": 21, "gap_to_leader": 4.4},
    {"driver_id": "VER", "tire_compound": "hard", "tire_age": 5, "gap_to_leader": 8.1},
]
LAP_COLUMNS = ("compound", "tire_age", "lap_time", "race_lap", "pit_out", "pit_in", "safety_car")
LAP_FLAGS = ("pit_out", "pit_in", "safety_car")
OVERRIDE_COLUMNS = ("compound", "peak_window_end", "warm_up_laps")

WET_COMPOUNDS = ("intermediate", "wet")  # the only tyres a stop can fit when the weather is not dry
RULE_VIOLATED, RULE_UNVERIFIED = "violated", "unverified"  # two_compound_rule values that need a warning

KEEP = "session"  # form values survive page switches (persist_state)
NOTICE_KEY = "strategy_notice"
REMOVED_NOTICE_KEY = "strategy_removed_compounds_notice"
OUTCOME_KEY = "strategy_outcome"
CALIBRATION_KEY = "strategy_calibration_outcome"
CALIBRATED_LAP_KEY = "strategy_calibrated_lap_time"  # base lap time copied from the last calibration
PLAN_DETAIL_KEY = "strategy_plan_detail"

# API field names in the words of the forms, for validation errors (HTTP 422).
FIELD_LABELS = {
    "telemetry": "Telemetry", "lap_times": "Recent lap times", "car_status": "Car condition",
    "engine_wear": "Engine wear", "brake_wear": "Brake wear", "damage": "Damage", "front_wing": "Front wing",
    "floor": "Floor", "diffuser": "Diffuser", "driver_profile": "Driver profile", "tire_management": "Tyre management",
    "risk_tolerance": "Risk tolerance", "braking_consistency": "Braking consistency",
    "throttle_aggressiveness": "Throttle aggressiveness", "tire_data": "Tyre model", "compound": "Compound",
    "base_performance": "Base performance", "degradation_rate": "Degradation per lap", "warm_up_laps": "Warm-up laps",
    "peak_performance_window": "Peak window", "pit_stop_delta": "Pit-stop loss", "race_state": "Race state",
    "current_lap": "Current lap", "total_laps": "Total laps", "weather": "Weather", "track_temperature": "Track temperature",
    "current_compound": "Fitted tyres", "current_tire_age": "Laps on the fitted set",
    "used_compounds": "Compounds used before the fitted set", "own_gap_to_leader": "Your gap to the leader",
    "competition": "Competitors", "driver_id": "Driver", "tire_compound": "Compound", "tire_age": "Tyre age",
    "gap_to_leader": "Gap to leader", "laps": "Lap history", "lap_time": "Lap time", "race_lap": "Race lap",
    "pit_out": "Out-lap", "pit_in": "In-lap", "safety_car": "Safety car", "fuel_correction_s_per_lap": "Fuel correction",
    "peak_window_end": "Fixed peak-window end",
}  # fmt: skip
# competition and laps are edited as tables (an index is a table row); a peak window's two items are the
# tyre table's "Peak from" and "Peak to" columns.
FIELDS = FieldLabels(
    FIELD_LABELS, table_lists=("competition", "laps"), index_names={"peak_performance_window": ("Peak from", "Peak to")}
)


# ------------------------------------------------------------------ input helpers


def show_input_problems(problems: Sequence[str]) -> None:
    st.error("Some inputs could not be read, so nothing was sent to the API:", icon=":material/edit_note:")
    st.markdown("\n".join(f"- {md_text(problem)}" for problem in problems))


def add_problem(problems: List[str], problem: str) -> None:
    if problem not in problems:
        problems.append(problem)


def row_list(numbers: Sequence[int], shown: int = 10) -> str:
    """Table row numbers as text: ``row 3``, ``rows 3, 4`` or ``rows 1, ..., 10 and 4 more``."""

    text = ", ".join(str(number) for number in numbers[:shown])
    more = f" and {len(numbers) - shown} more" if len(numbers) > shown else ""
    return f"{'row' if len(numbers) == 1 else 'rows'} {text}{more}"


def cell(value: Any) -> Any:
    """A table cell as a plain Python value; None when it is empty."""

    if isinstance(value, str):
        return value.strip() or None
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value  # numpy scalars are not JSON serialisable


def whole(value: Any) -> Any:
    """Whole numbers as int (the API's integer fields are strict); anything else unchanged, for the API to judge."""

    return int(value) if isinstance(value, float) and value.is_integer() else value


def table_rows(frame: pd.DataFrame, columns: Sequence[str]) -> List[Tuple[int, Dict[str, Any]]]:
    """Non-empty rows of an edited table as ``(row number, {column: value})``."""

    rows = []
    for number, record in enumerate(frame.to_dict("records"), start=1):
        values = {name: cell(record.get(name)) for name in columns}
        if any(value is not None and value is not False for value in values.values()):
            rows.append((number, values))
    return rows


def edited_table(name: str, default_rows: Sequence[Mapping[str, Any]], columns: Sequence[str], **options: Any) -> pd.DataFrame:
    """A data editor that starts from the rows kept for ``name`` (the defaults at first).

    Tables have no ``persist_state``: ``keep_table`` stores the submitted rows, and a new widget key
    shows them without the old key's edits being applied twice.
    """

    rows = st.session_state.setdefault(f"{name}_rows", [dict(row) for row in default_rows])
    version = st.session_state.setdefault(f"{name}_version", 0)
    return st.data_editor(pd.DataFrame(rows, columns=columns), key=f"{name}_{version}", hide_index=True, **options)


def keep_table(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Show ``rows`` in the table from the next run on (after a submit, a page switch or a calibration)."""

    st.session_state[f"{name}_rows"] = [dict(row) for row in rows]
    st.session_state[f"{name}_version"] = st.session_state.get(f"{name}_version", 0) + 1


def parse_lap_times(text: str) -> Tuple[List[float], List[str]]:
    """Lap times in seconds (``95.6``) or minutes:seconds (``1:35.6``); returns ``(values, unreadable entries)``."""

    values, unreadable = [], []
    for token in re.split(r"[\s,;]+", text.strip()):
        if not token:
            continue
        minutes, _, seconds = token.rpartition(":")
        try:
            value = float(seconds) + 60 * int(minutes or 0)
        except ValueError:
            unreadable.append(token)
            continue
        if math.isfinite(value):
            values.append(round(value, 3))
        else:
            unreadable.append(token)
    return values, unreadable


def optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def text_width(values: Sequence[str]) -> int:
    """A table column width (px) that shows the longest of ``values`` in full."""

    return round(TABLE_CELL_PADDING_PX + TABLE_CHAR_PX * max((len(value) for value in values), default=0))


def clock(seconds: float) -> str:
    """Seconds as ``h:mm:ss.sss`` (or ``m:ss.sss``)."""

    millis = round(float(seconds) * 1000)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours}:{minutes:02d}:{rest / 1000:06.3f}" if hours else f"{minutes}:{rest / 1000:06.3f}"


# ------------------------------------------------------------------ strategy request


def tyre_table_config() -> Dict[str, Any]:
    """Short headers (seven columns fit a laptop screen); the help of each names it in full."""

    number = st.column_config.NumberColumn
    return {
        "compound": st.column_config.SelectboxColumn("Compound", options=COMPOUNDS, required=True),
        "base_performance": number("Base perf.", format="%.6f", step=0.000001,
                                   help="Base performance: peak performance factor, 1.0 = the base lap time, lower = "
                                        "slower (0.5-2)."),
        "degradation_rate": number("Deg./lap", format="%.6f", step=0.000001,
                                   help="Degradation per lap: performance lost per tyre lap after the peak window (0-0.5)."),
        "warm_up_laps": number("Warm-up", format="%d", step=1,
                               help="Warm-up laps: laps to ramp from 90% to 100% performance (0-10)."),
        "window_start": number("Peak from", format="%d", step=1,
                               help="Start of the peak performance window (tyre lap); validated but does not change lap times."),
        "window_end": number("Peak to", format="%d", step=1,
                             help="End of the peak performance window: the tyre lap after which degradation starts."),
        "pit_stop_delta": number("Pit loss (s)", format="%.1f", step=0.1, help="Seconds lost by a stop that fits this compound."),
    }  # fmt: skip


def tyre_model(frame: pd.DataFrame, problems: List[str]) -> Dict[str, Dict[str, Any]]:
    tyres: Dict[str, Dict[str, Any]] = {}
    for number, row in table_rows(frame, TYRE_COLUMNS):
        compound = row["compound"]
        if compound is None:
            add_problem(problems, f"Tyre model, row {number}: choose a compound.")
            continue
        if compound in tyres:
            add_problem(problems, f"Tyre model: {compound} has more than one row; keep one.")
            continue
        entry = {name: row[name] for name in ("base_performance", "degradation_rate", "pit_stop_delta")}
        entry["warm_up_laps"] = whole(row["warm_up_laps"])
        if row["window_start"] is not None and row["window_end"] is not None:
            entry["peak_performance_window"] = [whole(row["window_start"]), whole(row["window_end"])]
        tyres[compound] = {name: value for name, value in entry.items() if value is not None}
    return tyres


def competitors(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {name: whole(value) for name, value in row.items() if value is not None} for _, row in table_rows(frame, RIVAL_COLUMNS)
    ]


def build_request(tyres: pd.DataFrame, rivals: pd.DataFrame) -> Tuple[Dict[str, Any], List[str]]:
    """The StrategyRequest body from the submitted form, and problems found while reading it."""

    state = st.session_state
    problems: List[str] = []
    lap_times, unreadable = parse_lap_times(state["strategy_lap_times"])
    if unreadable:
        problems.append(
            f"Recent lap times: {', '.join(repr(token) for token in unreadable[:5])} is not a lap time. Use seconds "
            "(95.6) or minutes:seconds (1:35.6), separated by commas or spaces."
        )
    telemetry: Dict[str, Any] = {"lap_times": lap_times}
    for field, key in (
        ("braking_consistency", "strategy_measured_braking"),
        ("throttle_aggressiveness", "strategy_measured_throttle"),
    ):
        if state[key] is not None:
            telemetry[field] = optional(state[key])
    race_state: Dict[str, Any] = {
        "current_lap": state["strategy_current_lap"],
        "total_laps": state["strategy_total_laps"],
        "weather": state["strategy_weather"],
        "track_temperature": state["strategy_track_temperature"],
    }
    if state["strategy_current_compound"] != FRESH_SET:
        race_state["current_compound"] = state["strategy_current_compound"]
        race_state["current_tire_age"] = state["strategy_tyre_age"]
    if state["strategy_history_known"]:
        race_state["used_compounds"] = list(state["strategy_used_compounds"])
    if state["strategy_own_gap"] is not None:
        race_state["own_gap_to_leader"] = optional(state["strategy_own_gap"])
    request = {
        "telemetry": telemetry,
        "car_status": {
            "engine_wear": optional(state["strategy_engine_wear"]),
            "brake_wear": optional(state["strategy_brake_wear"]),
            "damage": {part: optional(state[f"strategy_{part}"]) for part in ("front_wing", "floor", "diffuser")},
        },
        "driver_profile": {
            name: optional(state[f"strategy_{name}"])
            for name in ("tire_management", "risk_tolerance", "braking_consistency", "throttle_aggressiveness")
        },
        "tire_data": tyre_model(tyres, problems),
        "race_state": race_state,
        "competition": competitors(rivals),
    }
    return request, problems


def example_tables() -> bool:
    """Whether the tyre model and competitor tables still hold the example rows."""

    state = st.session_state
    return state.get("strategy_tyres_rows") == DEFAULT_TYRES and state.get("strategy_rivals_rows") == DEFAULT_RIVALS


def render_form() -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """The strategy form; on submit, the request body and any input problems."""

    unit = {"min_value": 0.0, "max_value": 1.0, "step": 0.05, "persist_state": KEEP}
    st.caption(
        "Pre-filled with the engine's documented mid-race example (lap 18 of 57, its lap times, tyre model and the "
        "example competitors HAM and VER): replace these values with your own race's."
    )
    with st.form("strategy_form"):
        st.markdown("**Race state**")
        columns = st.columns(4)
        columns[0].number_input(
            "Current lap",
            min_value=1,
            step=1,
            key="strategy_current_lap",
            persist_state=KEEP,
            help="The next lap to be driven; laps from here to the finish are planned.",
        )
        columns[1].number_input(
            "Total laps", min_value=1, step=1, key="strategy_total_laps", persist_state=KEEP, help="Race distance in laps."
        )
        columns[2].selectbox(
            "Weather",
            WEATHER,
            key="strategy_weather",
            persist_state=KEEP,
            help="Held constant for the remaining laps; decides which compounds a stop can fit.",
        )
        columns[3].number_input(
            "Track temperature (°C)",
            step=0.5,
            format="%.1f",
            key="strategy_track_temperature",
            persist_state=KEEP,
            help="Above 35 °C the model slows soft tyres and helps hard tyres.",
        )
        columns = st.columns(4)
        columns[0].selectbox(
            "Fitted tyres",
            (FRESH_SET, *COMPOUNDS),
            key="strategy_current_compound",
            persist_state=KEEP,
            help="The set on the car now. With 'Fresh set' every plan starts on a new set without a pit cost.",
        )
        columns[1].number_input(
            "Laps on the fitted set",
            min_value=0,
            step=1,
            key="strategy_tyre_age",
            persist_state=KEEP,
            help="Ignored for a fresh set.",
        )
        columns[2].multiselect(
            "Compounds used before the fitted set",
            COMPOUNDS,
            key="strategy_used_compounds",
            persist_state=KEEP,
            help="Checked by the simplified two-compound rule.",
        )
        columns[2].checkbox(
            "Tyre history known",
            key="strategy_history_known",
            persist_state=KEEP,
            help=(
                "Untick when the compounds run earlier are unknown: plans that may break the two-compound rule are "
                "then kept and flagged 'unverified'."
            ),
        )
        columns[3].number_input(
            "Your gap to the leader (s)",
            value=None,
            min_value=0.0,
            step=0.1,
            format="%.1f",
            key="strategy_own_gap",
            persist_state=KEEP,
            placeholder="not given",
            help="Relates the competitors below to your car (undercut signals). Leave empty to skip the signals.",
        )

        st.markdown("**Pace and driver**")
        st.text_input(
            "Recent lap times",
            key="strategy_lap_times",
            persist_state=KEEP,
            help=(
                "Representative laps in seconds (95.6) or minutes:seconds (1:35.6), separated by commas or spaces. "
                "Their mean is the base lap time."
            ),
        )
        columns = st.columns(4)
        columns[0].slider(
            "Tyre management",
            **unit,
            key="strategy_tire_management",
            help="0-1. Below 0.6 together with throttle aggressiveness above 0.8 adds 10% lap time.",
        )
        columns[1].slider("Risk tolerance", **unit, key="strategy_risk_tolerance", help="0-1. Above 0.8 adds 3% lap time.")
        columns[2].slider(
            "Braking consistency", **unit, key="strategy_braking_consistency", help="0-1. Below 0.7 adds up to 21% lap time."
        )
        columns[3].slider(
            "Throttle aggressiveness", **unit, key="strategy_throttle_aggressiveness", help="0-1. See tyre management."
        )
        columns = st.columns(4)
        columns[0].number_input(
            "Measured braking consistency",
            value=None,
            **unit,
            key="strategy_measured_braking",
            placeholder="from profile",
            help="Optional value from telemetry; replaces the driver-profile value.",
        )
        columns[1].number_input(
            "Measured throttle aggressiveness",
            value=None,
            **unit,
            key="strategy_measured_throttle",
            placeholder="from profile",
            help="Optional value from telemetry; replaces the driver-profile value.",
        )

        st.markdown("**Car condition** (0 = as new, 1 = worn out or destroyed)")
        columns = st.columns(5)
        columns[0].slider("Engine wear", **unit, key="strategy_engine_wear", help="Above 0.7 adds up to 2% lap time.")
        columns[1].slider("Brake wear", **unit, key="strategy_brake_wear", help="Above 0.8 adds up to 1.5% lap time.")
        columns[2].slider("Front wing damage", **unit, key="strategy_front_wing", help="Adds up to 2.5% lap time.")
        columns[3].slider(
            "Floor damage", **unit, key="strategy_floor", help="The larger of floor and diffuser damage adds up to 4% lap time."
        )
        columns[4].slider(
            "Diffuser damage",
            **unit,
            key="strategy_diffuser",
            help="The larger of floor and diffuser damage adds up to 4% lap time.",
        )

        st.markdown("**Tyre model** (one row per available compound; add intermediate or wet rows for a wet race)")
        tyres = edited_table("strategy_tyres", DEFAULT_TYRES, TYRE_COLUMNS, num_rows="dynamic", column_config=tyre_table_config())
        st.markdown("**Competitors** (optional)")
        rivals = edited_table(
            "strategy_rivals",
            DEFAULT_RIVALS,
            RIVAL_COLUMNS,
            num_rows="dynamic",
            column_config={
                "driver_id": st.column_config.TextColumn("Driver", help="Unique identifier, e.g. HAM."),
                "tire_compound": st.column_config.SelectboxColumn("Compound", options=COMPOUNDS),
                "tire_age": st.column_config.NumberColumn("Tyre age (laps)", format="%d", step=1),
                "gap_to_leader": st.column_config.NumberColumn("Gap to leader (s)", format="%.1f", step=0.1),
            },
        )
        submitted = st.form_submit_button("Generate strategy", type="primary", icon=":material/timeline:", key="strategy_submit")
    if not submitted:
        return None
    keep_table("strategy_tyres", [row for _, row in table_rows(tyres, TYRE_COLUMNS)])
    keep_table("strategy_rivals", [row for _, row in table_rows(rivals, RIVAL_COLUMNS)])
    return build_request(tyres, rivals)


# ------------------------------------------------------------------ strategy result


def stint_chart(strategies: Sequence[Mapping[str, Any]], first_lap: int, last_lap: int) -> alt.LayerChart:
    """Timeline of every plan's stints (x = lap, one row per plan, colour and letter = compound).

    Rows are labelled by rank and stop count only (the full plan is in the tooltip and the table), so
    the lap axis keeps most of a phone-width screen and the legend fits under it.
    """

    rows = []
    for option in strategies:
        for stint in option.get("stint_breakdown") or []:
            start, end, compound = stint["start_lap"], stint["end_lap"], stint["tire_compound"]
            rows.append({
                "Row": f"{option['rank']}. {option['pit_stops']}-stop",
                "Plan": option["strategy_id"],
                "Compound": compound,
                "Letter": compound[:1].upper(),
                "Ink": LETTER_INK.get(compound, "#0b0b0b"),
                "From": start - 1 + STINT_GAP_LAPS,
                "To": end - STINT_GAP_LAPS,
                "Middle": (start - 1 + end) / 2,
                "Stint laps": stint["laps"],
                "Laps": f"{start}-{end}",
                "Laps on the set": f"{stint['tire_age_start']} → {stint['tire_age_end']}",
                "Average lap (s)": stint["average_lap_time"],
                "Best lap (s)": stint["best_lap_time"],
                "Worst lap (s)": stint["worst_lap_time"],
            })  # fmt: skip
    frame = pd.DataFrame(rows)
    present = [compound for compound in COMPOUNDS if compound in set(frame["Compound"])]
    plans = list(dict.fromkeys(frame["Row"]))
    y = alt.Y("Row:N", sort=plans, title=None, scale=alt.Scale(paddingInner=0.3))
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("From:Q", title="Race lap", scale=alt.Scale(domain=[first_lap - 1, last_lap], nice=False)),
            x2="To:Q",
            y=y,
            color=alt.Color(
                "Compound:N",
                scale=alt.Scale(domain=present, range=[COMPOUND_COLORS.get(c, "#8a8f98") for c in present]),
                legend=alt.Legend(orient="bottom", title=None, columns=3, offset=12),  # wraps instead of clipping
            ),
            tooltip=[
                "Plan",
                "Compound",
                "Laps",
                "Laps on the set",
                *(alt.Tooltip(name, format=".3f") for name in ("Average lap (s)", "Best lap (s)", "Worst lap (s)")),
            ],
        )
    )
    letters = (
        alt.Chart(frame)
        .transform_filter(alt.datum["Stint laps"] >= 2)
        .mark_text(fontWeight="bold", fontSize=11)
        .encode(x="Middle:Q", y=y, text="Letter:N", color=alt.Color("Ink:N", scale=None))
    )
    # fit-x: the height is the plot's own, so a second legend row adds height instead of squeezing the bars
    return (bars + letters).properties(height=36 * len(plans), autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))


def wet_tyre_hint(request: Mapping[str, Any]) -> Optional[str]:
    """Why no stop can fit tyres: a wet or intermediate race with only dry compounds in the tyre model."""

    weather = (request.get("race_state") or {}).get("weather")
    if weather == "dry" or set(request.get("tire_data") or {}) & set(WET_COMPOUNDS):
        return None
    return (
        f"The weather is {weather}, but the tyre model has no intermediate or wet row, so a pit stop cannot fit "
        "tyres for these conditions. Add an intermediate or wet row to the tyre model table and generate again."
    )


def show_plan_warnings(outcome: Mapping[str, Any], best: Mapping[str, Any]) -> None:
    """Caveats of the recommended plan that belong next to it, not only in the notes further down."""

    rule = best.get("two_compound_rule")
    # The plan's own notes on the rule, with the form's labels for the API field names they mention.
    rule_notes = " ".join(md_text(FIELDS.message(note)) for note in best.get("notes") or [] if "rule" in str(note).lower())
    if rule == RULE_VIOLATED:
        hint = wet_tyre_hint(outcome["request"])
        st.warning(
            f"**The recommended plan breaks the simplified two-compound rule.** {rule_notes}"
            + (f"\n\n{md_text(hint)}" if hint else ""),
            icon=":material/gpp_bad:",
        )
    elif rule == RULE_UNVERIFIED:
        st.info(
            f"**The two-compound rule could not be checked for the recommended plan.** {rule_notes} Tick 'Tyre "
            "history known' and list the compounds used before the fitted set to check it.",
            icon=":material/help:",
        )
    model = outcome["response"].get("model") or {}
    penalties = [
        f"{name} {model[key]:.4f}"
        for name, key in (("driver multiplier", "driver_multiplier"), ("damage/wear multiplier", "damage_multiplier"))
        if isinstance(model.get(key), (int, float)) and not math.isclose(model[key], 1.0)
    ]
    if outcome.get("calibrated_lap_time") and penalties:
        st.warning(
            "**Driver or car penalties are counted twice.** The recent lap time is the calibrated base lap time, "
            f"which already contains the driver's and car's pace, but this result also applies a "
            f"{' and a '.join(penalties)}. Set the driver profile and car condition penalty-free (multipliers 1.0 "
            "under Model factors used) and generate again.",
            icon=":material/functions:",
        )


def render_strategy(outcome: Mapping[str, Any]) -> None:
    body = outcome["response"]
    strategies = [option for option in body.get("strategies") or [] if isinstance(option, dict)]
    best = next((option for option in strategies if option.get("strategy_id") == body.get("best_strategy_id")), None)
    st.subheader("Recommended plan")
    heuristic_badge()
    if outcome.get("example"):
        example_inputs_badge()
    st.caption(
        f"{md_text(body.get('method', ''))} Result for the request sent at "
        f"{time.strftime('%H:%M:%S', time.localtime(outcome['at']))}."
    )
    if best is None:
        st.warning("The API response contains no ranked plan.", icon=":material/help:")
        json_expander(body)
        return
    first_lap, last_lap = body.get("current_lap"), body.get("total_laps")
    show_plan_warnings(outcome, best)
    st.markdown(f"**{md_text(best['strategy_id'])}** · {md_text(' → '.join(best['tire_compounds']))}")
    columns = st.columns(4)
    columns[0].metric("Pit stops", best["pit_stops"], help="Further stops from now to the finish.")
    columns[1].metric(
        "Pit laps", ", ".join(str(lap) for lap in best["pit_laps"]) or "none", help="A pit lap is the last lap before the stop."
    )
    columns[2].metric(
        f"Projected time, laps {first_lap}-{last_lap}",
        clock(best["projected_race_time"]),
        help=(
            f"{best['projected_race_time']:.3f} s: driving {best['driving_time_s']:.3f} s + pit loss "
            f"{best['pit_time_loss_s']:.3f} s. A heuristic estimate for the remaining laps, not a calibrated simulation."
        ),
    )
    if len(strategies) > 1 and strategies[1].get("delta_to_best_s") is not None:
        columns[3].metric(
            "Margin to plan 2",
            f"{strategies[1]['delta_to_best_s']:.3f} s",
            help=(
                f"Plan 2 is {strategies[1]['strategy_id']}. The model gives no uncertainty range, so a small margin "
                "does not separate the plans."
            ),
        )

    st.markdown(
        "**Plans compared** (fastest first; all times in seconds for the remaining laps; projected = driving time + "
        "pit loss)"
    )
    seconds = st.column_config.NumberColumn(format="%.3f")
    # The plan id names the compounds and is never cut off; the columns fit a laptop screen without scrolling.
    st.dataframe(
        [
            {
                "Rank": option["rank"],
                "Plan": option["strategy_id"],
                "Two-compound rule": option["two_compound_rule"],
                "Pit laps": ", ".join(str(lap) for lap in option["pit_laps"]) or "none",
                "Projected (s)": option["projected_race_time"],
                "Gap to best (s)": option["delta_to_best_s"],
                "Pit loss (s)": option["pit_time_loss_s"],
            }
            for option in strategies
        ],
        hide_index=True,
        column_config={name: seconds for name in ("Projected (s)", "Gap to best (s)", "Pit loss (s)")}
        | {"Plan": st.column_config.TextColumn(width=text_width([option["strategy_id"] for option in strategies]))},
    )
    if all(option.get("stint_breakdown") for option in strategies):
        st.altair_chart(stint_chart(strategies, first_lap, last_lap), width="stretch")
        st.caption("Stints per plan; the gaps are pit stops. Hover a stint for its lap range, laps on the set and lap times.")

    plans = [option["strategy_id"] for option in strategies]
    chosen = st.selectbox(  # a new result starts at its recommended plan (the page drops the old choice)
        "Plan details", plans, index=plans.index(best["strategy_id"]), key=PLAN_DETAIL_KEY, help="Stints and notes of one plan."
    )
    option = strategies[plans.index(chosen)] if chosen in plans else best
    stints = option.get("stint_breakdown") or []
    st.caption(
        f"Risk label (the API's, set from the number of stops only): {md_text(option.get('risk_level', '–'))}. "
        "Lap times per stint, then the tyre performance behind them."
    )
    st.dataframe(
        [
            {
                "Laps": f"{stint['start_lap']}-{stint['end_lap']}",
                "Compound": stint["tire_compound"],
                "Laps on the set": f"{stint['tire_age_start']} → {stint['tire_age_end']}",
                "New set": stint["fitted_at_stop"],
                "Average lap (s)": stint["average_lap_time"],
                "Best lap (s)": stint["best_lap_time"],
                "Worst lap (s)": stint["worst_lap_time"],
                "Stint time (s)": stint["total_time"],
            }
            for stint in stints
        ],
        hide_index=True,
        column_config={name: seconds for name in ("Average lap (s)", "Best lap (s)", "Worst lap (s)", "Stint time (s)")}
        | {
            "Laps on the set": st.column_config.TextColumn(
                help="Laps already run on this set when the stint starts → after its last lap."
            )
        },
    )
    st.dataframe(
        [
            {
                "Laps": f"{stint['start_lap']}-{stint['end_lap']}",
                "Compound": stint["tire_compound"],
                "Performance start → end": f"{stint['start_performance']:.4f} → {stint['end_performance']:.4f}",
                "Laps past peak window": stint["laps_beyond_peak_window"],
                "Laps at performance floor": stint["laps_at_performance_floor"],
            }
            for stint in stints
        ],
        hide_index=True,
    )
    st.markdown("\n".join(f"- {md_text(note)}" for note in option.get("notes") or []))

    model = body.get("model") or {}
    st.markdown("**Model factors used**")
    columns = st.columns(4)
    columns[0].metric("Base lap time", f"{model.get('base_lap_time_s', 0):.3f} s", help="Mean of the recent lap times.")
    columns[1].metric("Driver multiplier", f"{model.get('driver_multiplier', 0):.4f}", help="1.0 = no driver penalty.")
    columns[2].metric(
        "Damage/wear multiplier", f"{model.get('damage_multiplier', 0):.4f}", help="1.0 = no damage or wear penalty."
    )
    columns[3].metric(
        "Tyre state",
        str(body.get("tire_state", "")).replace("_", " "),
        help="supplied: the fitted set continues; assumed fresh: no fitted set was given.",
    )

    st.markdown("**Competitor signals**")
    signals = body.get("competitor_signals") or []
    if signals:
        st.dataframe(
            [
                {"Driver": s["driver_id"], "Signal": s["signal"].replace("_", " "), "Gap (s)": s["gap_s"],
                 "Compound": s["tire_compound"], "Tyre age": s["tire_age"]}
                for s in signals
            ],
            hide_index=True,
            column_config={"Gap (s)": st.column_config.NumberColumn(format="%.1f")},
        )  # fmt: skip
        # The explanations are sentences: as a table column they would be cut off.
        st.markdown("\n".join(f"- **{md_text(s['driver_id'])}**: {md_text(s['explanation'])}" for s in signals))
    st.caption(md_text(body.get("competitor_signals_note") or "No competitor signals."))

    st.markdown("**Assumptions and limitations** (from the API)")
    st.markdown("\n".join(f"- {md_text(item)}" for item in body.get("assumptions") or []))
    not_modelled = body.get("not_modelled_inputs") or []
    if not_modelled:
        st.caption("Accepted but not used in any number: " + "; ".join(md_text(item) for item in not_modelled))
    search = body.get("search") or {}
    if search:
        st.caption(
            f"Search: {search.get('sequences_evaluated')} compound sequences evaluated, "
            f"{search.get('sequences_skipped_too_few_laps')} skipped (too few laps), "
            f"{search.get('sequences_excluded_two_compound_rule')} excluded by the two-compound rule."
        )


def render_outcome(outcome: Mapping[str, Any]) -> None:
    if outcome.get("problems"):
        show_input_problems(outcome["problems"])
        return
    error = outcome.get("error")
    if error is not None:
        show_request_error(error, "The strategy request", FIELDS)
        engine_refusal = isinstance(error, ApiError) and error.status_code == 422 and not error.errors
        hint = wet_tyre_hint(outcome["request"]) if engine_refusal else None
        if hint:
            st.info(hint, icon=":material/water_drop:")
    else:
        render_strategy(outcome)
    json_expander(outcome["request"], "Request sent to the API")
    if outcome.get("response") is not None:
        json_expander(outcome["response"])


# ------------------------------------------------------------------ tyre calibration (optional endpoint)


def number_or_none(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def fixed_values(frame: pd.DataFrame, problems: List[str]) -> Dict[str, Dict[str, Any]]:
    """The fixed-values table as ``{field: {compound: value}}``; one row per compound."""

    fixed: Dict[str, Dict[str, Any]] = {field: {} for field in OVERRIDE_COLUMNS[1:]}
    seen = set()
    for number, row in table_rows(frame, OVERRIDE_COLUMNS):
        compound = row["compound"]
        if compound is None:
            add_problem(problems, f"Fixed values, row {number}: choose a compound.")
            continue
        if compound in seen:
            add_problem(problems, f"Fixed values: {compound} has more than one row; keep one.")
            continue
        seen.add(compound)
        for field, values in fixed.items():
            if row[field] is not None:
                values[compound] = whole(row[field])
    return {field: values for field, values in fixed.items() if values}


def calibration_request(laps: pd.DataFrame, overrides: pd.DataFrame) -> Tuple[Dict[str, Any], List[str]]:
    """The TyreCalibrationRequest body from the submitted form, and problems found while reading it."""

    state = st.session_state
    problems: List[str] = []
    rows = table_rows(laps, LAP_COLUMNS)
    request: Dict[str, Any] = {
        "laps": [
            {
                name: (True if name in LAP_FLAGS else whole(value) if name in ("tire_age", "race_lap") else value)
                for name, value in row.items()
                if value is not None and value is not False
            }
            for _, row in rows
        ],
        "weather": state["calibration_weather"],
    }
    for field, key in (
        ("track_temperature", "calibration_track_temperature"),
        ("pit_stop_delta", "calibration_pit_stop_delta"),
        ("fuel_correction_s_per_lap", "calibration_fuel_correction"),
    ):
        if state[key] is not None:
            request[field] = optional(state[key])
    if request.get("fuel_correction_s_per_lap"):
        missing = [number for number, row in rows if row["race_lap"] is None and not any(row[flag] for flag in LAP_FLAGS)]
        if missing:
            problems.append(
                f"Lap history, {row_list(missing)}: enter the race lap. A fuel correction needs the race lap of every "
                "lap that is not flagged as an out-lap, in-lap or safety-car lap (or set the fuel correction to 0)."
            )
    request.update(fixed_values(overrides, problems))
    return request, problems


def use_calibration(body: Mapping[str, Any]) -> None:
    """Copy the calibrated tyre model and base lap time into the strategy form (button callback)."""

    tire_data = body.get("tire_data") or {}
    kept = [row.get("compound") for row in st.session_state.get("strategy_tyres_rows", [])]
    removed = [compound for compound in dict.fromkeys(kept) if compound and compound not in tire_data]
    keep_table("strategy_tyres", [
        {
            "compound": compound,
            "base_performance": entry["base_performance"],
            "degradation_rate": entry["degradation_rate"],
            "warm_up_laps": entry["warm_up_laps"],
            "window_start": entry["peak_performance_window"][0],
            "window_end": entry["peak_performance_window"][1],
            "pit_stop_delta": entry["pit_stop_delta"],
        }
        for compound, entry in tire_data.items()
    ])  # fmt: skip
    st.session_state["strategy_lap_times"] = str(body["estimated_base_lap_time_s"])
    st.session_state[CALIBRATED_LAP_KEY] = body["estimated_base_lap_time_s"]
    st.session_state[NOTICE_KEY] = (
        f"The strategy form now uses the calibrated tyre model ({', '.join(tire_data)}) and the estimated base lap "
        f"time {body['estimated_base_lap_time_s']} s. That lap time already contains the driver's and car's pace: "
        "to reproduce it, keep the driver profile and car condition penalty-free (driver and damage multipliers "
        "1.0 in the result). The fitted tyres must be one of the calibrated compounds."
    )
    if removed:
        one = len(removed) == 1
        st.session_state[REMOVED_NOTICE_KEY] = (
            f"**{' and '.join(removed)} {'was' if one else 'were'} removed from the tyre model:** the calibration "
            f"estimated no parameters for {'it' if one else 'them'}, so the plans cannot use {'it' if one else 'them'}. "
            "To keep a compound available, add its row back to the tyre model table with values relative to the new "
            "base lap time."
        )


def render_calibration_result(outcome: Mapping[str, Any]) -> None:
    if outcome.get("problems"):
        show_input_problems(outcome["problems"])
        return
    if outcome.get("error") is not None:
        show_request_error(outcome["error"], "The calibration request", FIELDS)
        st.json(outcome["request"], expanded=False)
        return
    body = outcome["response"]
    compounds = body.get("compounds") or {}
    base = body.get("estimated_base_lap_time_s")
    tabs = st.tabs(["Estimated parameters", "Excluded laps", "Assumptions", "Raw JSON"])
    with tabs[0]:
        st.caption(md_text(body.get("method", "")))
        columns = st.columns(4)
        columns[0].metric(
            "Estimated base lap time",
            "–" if base is None else f"{base:.3f} s",
            help=(
                "Peak pace of the reference compound under the history's conditions, with the weather and hot-track "
                "factors removed."
            ),
        )
        columns[1].metric("Reference compound", body.get("reference_compound") or "–", help="Base performance 1.0 by definition.")
        columns[2].metric("Laps supplied", body.get("laps_supplied", "–"))
        columns[3].metric("Laps excluded", len(body.get("excluded_laps") or []), help="Flagged laps and rejected outliers.")
        if base is None:
            st.warning("No compound could be estimated from these laps; the reasons are below.", icon=":material/report:")
        # Two narrow tables rather than one wide one: the fit first (it decides whether the parameters can be
        # used), then the estimated tyre model.
        fits, models = [], []
        for compound, entry in compounds.items():
            tire = entry.get("tire_data") or {}
            fit = entry.get("fit") or {}
            window = tire.get("peak_performance_window")
            warm_up_source = str(entry.get("warm_up_source", "")).replace("_", " ")
            window_source = str(entry.get("peak_window_end_source", "")).replace("_", " ")
            laps = " / ".join(str(entry.get(name)) for name in ("laps_used", "clean_laps", "laps_supplied"))
            fits.append({
                "Compound": compound,
                "Status": str(entry.get("status", "")).replace("_", " "),
                "R²": fit.get("r_squared"),
                "Residual SD (s)": fit.get("residual_std_s"),
                "Laps used / clean / supplied": laps,
                "Outliers": entry.get("outliers_rejected"),
            })  # fmt: skip
            models.append({
                "Compound": compound,
                "Base performance": tire.get("base_performance"),
                "Degradation per lap": tire.get("degradation_rate"),
                "Warm-up laps": f"{tire['warm_up_laps']} ({warm_up_source})" if tire else None,
                "Peak window": f"{window[0]}-{window[1]} ({window_source})" if window else None,
                "Peak lap (s)": fit.get("peak_lap_time_s"),
                "Initial loss (s/lap)": fit.get("initial_degradation_s_per_lap"),
            })  # fmt: skip
        number = st.column_config.NumberColumn
        st.markdown("**Fit per compound**")
        st.dataframe(fits, hide_index=True, column_config={"R²": number(format="%.3f"), "Residual SD (s)": number(format="%.3f")})
        st.markdown("**Estimated tyre model**")
        st.dataframe(
            models,
            hide_index=True,
            column_config={
                "Base performance": number(format="%.6f"),
                "Degradation per lap": number(format="%.6f"),
                "Peak lap (s)": number(format="%.3f"),
                "Initial loss (s/lap)": number(format="%.4f"),
            },
        )
        for compound, entry in compounds.items():
            if entry.get("reason"):
                st.warning(f"{compound}: {md_text(entry['reason'])}", icon=":material/report:")
            notes = entry.get("notes") or []
            if notes:
                st.markdown(f"**{compound} notes**\n" + "\n".join(f"- {md_text(note)}" for note in notes))
        if body.get("tire_data"):
            st.button(
                "Use in the strategy form",
                icon=":material/input:",
                key="calibration_apply",
                on_click=use_calibration,
                args=(body,),
                help="Replaces the tyre model table and the recent lap times of the strategy form.",
            )
    with tabs[1]:
        excluded = body.get("excluded_laps") or []
        if excluded:
            st.dataframe(
                [
                    {"Row": lap["index"] + 1, "Compound": lap["compound"], "Tyre age": lap["tire_age"],
                     "Lap time (s)": lap["lap_time"], "Reason": lap["reason"].replace("_", " "),
                     "Fitted lap time (s)": lap.get("fitted_lap_time"), "Residual (s)": lap.get("residual_s")}
                    for lap in excluded
                ],
                hide_index=True,
                column_config={
                    name: st.column_config.NumberColumn(format="%.3f")
                    for name in ("Lap time (s)", "Fitted lap time (s)", "Residual (s)")
                },
            )  # fmt: skip
        else:
            st.caption("No lap was excluded.")
    with tabs[2]:
        st.markdown("\n".join(f"- {md_text(item)}" for item in body.get("assumptions") or []))
    with tabs[3]:
        st.caption("Request sent to the API")
        st.json(outcome["request"], expanded=False)
        st.caption("Raw API response")
        st.json(body, expanded=1)


def render_calibration() -> None:
    """Estimate ``tire_data`` from lap history, when this API version offers the endpoint."""

    try:
        schema = api_schema()
    except (ApiUnavailable, ApiError):  # the page's other requests report the failure
        return
    if not offers_endpoint(schema, CALIBRATE_TYRES_PATH):
        return
    example = documented_example(schema, "TyreCalibrationRequest") or {}
    example_laps = example.get("laps") if isinstance(example.get("laps"), list) else []
    lap_rows = [
        {name: lap.get(name, False if name in LAP_FLAGS else None) for name in LAP_COLUMNS}
        for lap in example_laps
        if isinstance(lap, dict)
    ]
    weather = example.get("weather")
    for key, value in (
        ("calibration_weather", weather if weather in WEATHER else WEATHER[0]),
        ("calibration_track_temperature", number_or_none(example.get("track_temperature"))),
        ("calibration_pit_stop_delta", number_or_none(example.get("pit_stop_delta"))),
        ("calibration_fuel_correction", number_or_none(example.get("fuel_correction_s_per_lap", 0.0))),
    ):
        st.session_state.setdefault(key, value)
    outcome = st.session_state.get(CALIBRATION_KEY)
    with st.expander("Calibrate tyre parameters from lap history", icon=":material/tune:", expanded=outcome is not None):
        st.markdown(
            "Estimates the strategy form's tyre model from timed laps by inverting the engine's own lap-time model. "
            "At least 5 clean laps per compound are needed; flag out-laps, in-laps and safety-car laps (they are "
            "excluded). Missing data is reported per compound, never guessed."
        )
        if lap_rows:
            st.caption(
                "Pre-filled with the example documented by the API: synthetic laps generated from the engine's own "
                "lap-time model, so they fit almost exactly. Replace them with real laps (edit the table or paste "
                "from a spreadsheet); only a fit to real laps says anything about your tyres."
            )
        with st.form("calibration_form"):
            laps = edited_table(
                "calibration_laps",
                lap_rows,
                LAP_COLUMNS,
                num_rows="dynamic",
                height=280,
                column_config={
                    "compound": st.column_config.SelectboxColumn("Compound", options=COMPOUNDS, required=True),
                    "tire_age": st.column_config.NumberColumn(
                        "Tyre age", format="%d", step=1, help="Lap number on this set (1 = first lap on it)."
                    ),
                    "lap_time": st.column_config.NumberColumn("Lap time (s)", format="%.3f", step=0.001),
                    "race_lap": st.column_config.NumberColumn(
                        "Race lap", format="%d", step=1, help="Needed on unflagged laps only with a fuel correction."
                    ),
                    "pit_out": st.column_config.CheckboxColumn("Out-lap", default=False),
                    "pit_in": st.column_config.CheckboxColumn("In-lap", default=False),
                    "safety_car": st.column_config.CheckboxColumn(
                        "Safety car", default=False, help="Safety-car, VSC or red-flag lap."
                    ),
                },
            )
            columns = st.columns(4)
            columns[0].selectbox(
                "Weather", WEATHER, key="calibration_weather", persist_state=KEEP, help="Conditions of the whole history."
            )
            columns[1].number_input(
                "Track temperature (°C)",
                value=None,
                step=0.5,
                format="%.1f",
                key="calibration_track_temperature",
                persist_state=KEEP,
                placeholder="required",
            )
            columns[2].number_input(
                "Pit-stop loss (s)",
                value=None,
                min_value=0.0,
                step=0.1,
                format="%.1f",
                key="calibration_pit_stop_delta",
                persist_state=KEEP,
                placeholder="required",
                help="Copied into every estimated compound; not estimated.",
            )
            columns[3].number_input(
                "Fuel correction (s/lap)",
                value=None,
                min_value=0.0,
                step=0.01,
                format="%.3f",
                key="calibration_fuel_correction",
                persist_state=KEEP,
                placeholder="none",
                help="Seconds gained per lap of fuel burnt (e.g. 0.03-0.06); 0 = no correction. Needs race laps.",
            )
            st.caption("Optional: fix the peak-window end or warm-up length of a compound instead of detecting it.")
            overrides = edited_table(
                "calibration_overrides",
                [],
                OVERRIDE_COLUMNS,
                num_rows="dynamic",
                column_config={
                    "compound": st.column_config.SelectboxColumn("Compound", options=COMPOUNDS),
                    "peak_window_end": st.column_config.NumberColumn("Fixed peak-window end", format="%d", step=1),
                    "warm_up_laps": st.column_config.NumberColumn("Fixed warm-up laps", format="%d", step=1),
                },
            )
            submitted = st.form_submit_button("Estimate tyre parameters", icon=":material/query_stats:", key="calibration_submit")
        if submitted:
            keep_table("calibration_laps", [row for _, row in table_rows(laps, LAP_COLUMNS)])
            keep_table("calibration_overrides", [row for _, row in table_rows(overrides, OVERRIDE_COLUMNS)])
            request, problems = calibration_request(laps, overrides)
            outcome = {"request": request, "problems": problems, "response": None, "error": None}
            if not problems:
                try:
                    outcome["response"] = get_client().calibrate_tyres(request)
                except (ApiUnavailable, ApiError) as exc:
                    outcome["error"] = exc
            st.session_state[CALIBRATION_KEY] = outcome
        if outcome is not None:
            render_calibration_result(outcome)


# ------------------------------------------------------------------ page

st.title("Race strategy")
heuristic_badge()
st.caption(
    "Ranks pit-stop plans for the rest of a race: a transparent lap-time heuristic (tyre warm-up, peak window and "
    "linear degradation, driver and damage penalties) with an exact search over pit laps for every compound "
    "sequence with up to 3 stops. It is not a calibrated race simulator; fuel, traffic, safety cars and weather "
    "changes are not modelled."
)
for widget_key, default in FORM_DEFAULTS.items():
    st.session_state.setdefault(widget_key, default)

render_calibration()
notice, removed_notice = st.session_state.pop(NOTICE_KEY, None), st.session_state.pop(REMOVED_NOTICE_KEY, None)
if notice:
    st.success(notice, icon=":material/check_circle:")
if removed_notice:
    st.warning(removed_notice, icon=":material/playlist_remove:")
submission = render_form()
if submission is not None:
    strategy_request, input_problems = submission
    calibrated_lap = st.session_state.get(CALIBRATED_LAP_KEY)
    result: Dict[str, Any] = {
        "request": strategy_request,
        "problems": input_problems,
        "response": None,
        "error": None,
        "at": time.time(),
        "example": unchanged_example(FORM_DEFAULTS) and example_tables(),
        # The base lap time is still the calibrated one, which already contains the driver's and car's pace.
        "calibrated_lap_time": calibrated_lap is not None and strategy_request["telemetry"]["lap_times"] == [calibrated_lap],
    }
    if not input_problems:
        try:
            result["response"] = get_client().generate_strategy(strategy_request)
        except (ApiUnavailable, ApiError) as exc:
            result["error"] = exc
    st.session_state[OUTCOME_KEY] = result
    st.session_state.pop(PLAN_DETAIL_KEY, None)  # the plan chosen for the previous result
if st.session_state.get(OUTCOME_KEY):
    render_outcome(st.session_state[OUTCOME_KEY])
