"""Car setup: the API's heuristic setup search (Optuna TPE plus local refinement over a proxy objective).

Every number shown comes from ``POST /api/setup/recommend``. The objective is a documented,
dimensionless heuristic penalty, not a lap time, and "confidence" is multi-start agreement.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, Mapping, Optional

import altair as alt
import pandas as pd
import streamlit as st

from ui.api_client import ApiError, ApiUnavailable, get_client
from ui.components import (
    FieldLabels,
    example_inputs_badge,
    heuristic_badge,
    json_expander,
    md_text,
    show_request_error,
    unchanged_example,
)

TRACK_TYPES = ("high_speed", "technical", "mixed", "low_speed")
CONDITIONS = ("dry", "intermediate", "wet")
BAR_COLOR = "#2a78d6"

# Silverstone, the documented example of the setup request (SetupRequest in the API schema).
FORM_DEFAULTS: Dict[str, Any] = {
    "setup_track_name": "Silverstone Circuit",
    "setup_track_type": "high_speed",
    "setup_track_length": 5891.0,
    "setup_average_speed": 220.0,
    "setup_corners": 18,
    "setup_high_speed_sections": 8,
    "setup_low_speed_sections": 4,
    "setup_downforce_requirement": 0.6,
    "setup_condition": "dry",
    "setup_temperature": 24.0,
    "setup_humidity": 50.0,
    "setup_wind_speed": None,
    "setup_risk_tolerance": 0.5,
    "setup_tire_management": 0.7,
    "setup_pin_ride_height": None,
    "setup_pin_front_wing": None,
    "setup_pin_rear_wing": None,
    "setup_pin_diff_preload": None,
    "setup_pin_diff_power": None,
    "setup_pin_diff_coast": None,
    "setup_n_trials": 128,
    "setup_seed": 42,
}
# (label, key in the response, key in the baseline, unit key in "units", pin name in "pinned_by_driver")
PARAMETERS = (
    ("Ride height", ("ride_height",), "ride_height", "ride_height"),
    ("Front wing angle", ("front_wing_angle",), "front_wing_angle", "front_wing_angle"),
    ("Rear wing angle", ("rear_wing_angle",), "rear_wing_angle", "rear_wing_angle"),
    ("Brake bias", ("brake_bias",), "brake_bias", "brake_bias"),
    ("Differential preload", ("diff_settings", "preload"), "diff_settings", "diff_preload"),
    ("Differential on throttle", ("diff_settings", "power"), "diff_settings", "diff_power"),
    ("Differential off throttle", ("diff_settings", "coast"), "diff_settings", "diff_coast"),
    ("Front anti-roll bar", ("suspension_settings", "front_arb"), "suspension_settings", "front_arb"),
    ("Rear anti-roll bar", ("suspension_settings", "rear_arb"), "suspension_settings", "rear_arb"),
    ("Front springs", ("suspension_settings", "front_spring"), "suspension_settings", "front_spring"),
    ("Rear springs", ("suspension_settings", "rear_spring"), "suspension_settings", "rear_spring"),
)
OUTCOME_KEY = "setup_outcome"
KEEP = "session"  # form values survive page switches (persist_state)
API_DEFAULT_HELP = "Leave empty to let the API assume its default (listed under assumed defaults)."
# A unit that names a 0-100 adjuster scale (differential and suspension): 0 and 100 are the ends of the search.
ZERO_TO_HUNDRED = re.compile(r"\b0-100\b")

# API field names in the words of the form, for validation errors (HTTP 422).
FIELD_LABELS = {
    "driver_preferences": "Driver", "risk_tolerance": "Risk tolerance", "tire_management": "Tyre management",
    "preferred_ride_height": "Pinned ride height", "preferred_wing_angles": "Pinned wing angles", "front": "Front",
    "rear": "Rear", "preferred_diff_settings": "Pinned differential", "preload": "Preload", "power": "On throttle",
    "coast": "Off throttle", "track_profile": "Track", "track_name": "Track name", "track_length": "Lap length",
    "corners": "Corners", "high_speed_sections": "High-speed corners", "low_speed_sections": "Low-speed corners",
    "track_type": "Track type", "average_speed": "Average speed", "downforce_requirement": "Downforce requirement",
    "weather": "Weather", "condition": "Track condition", "temperature": "Air temperature", "humidity": "Humidity",
    "wind_speed": "Wind speed", "n_trials": "Search trials", "seed": "Search seed",
}  # fmt: skip
FIELDS = FieldLabels(FIELD_LABELS)


def rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def build_request() -> Dict[str, Any]:
    """The SetupRequest body from the submitted form; empty optional fields are left out."""

    state = st.session_state
    preferences: Dict[str, Any] = {
        name: rounded(state[f"setup_{name}"])
        for name in ("risk_tolerance", "tire_management")
        if state[f"setup_{name}"] is not None  # left out: the API assumes its default and reports it
    }
    pins = {
        "preferred_ride_height": state["setup_pin_ride_height"],
        "preferred_wing_angles": {"front": state["setup_pin_front_wing"], "rear": state["setup_pin_rear_wing"]},
        "preferred_diff_settings": {
            "preload": state["setup_pin_diff_preload"],
            "power": state["setup_pin_diff_power"],
            "coast": state["setup_pin_diff_coast"],
        },
    }
    for name, value in pins.items():
        if isinstance(value, dict):
            value = {key: item for key, item in value.items() if item is not None} or None
        if value is not None:
            preferences[name] = value
    track: Dict[str, Any] = {
        "track_length": state["setup_track_length"],
        "corners": state["setup_corners"],
        "high_speed_sections": state["setup_high_speed_sections"],
        "low_speed_sections": state["setup_low_speed_sections"],
        "track_type": state["setup_track_type"],
        "average_speed": state["setup_average_speed"],
        "downforce_requirement": rounded(state["setup_downforce_requirement"]),
    }
    if state["setup_track_name"].strip():
        track["track_name"] = state["setup_track_name"].strip()
    weather: Dict[str, Any] = {"condition": state["setup_condition"], "temperature": state["setup_temperature"]}
    for name in ("humidity", "wind_speed"):
        if state[f"setup_{name}"] is not None:
            weather[name] = state[f"setup_{name}"]
    return {
        "driver_preferences": preferences,
        "track_profile": track,
        "weather": weather,
        "n_trials": state["setup_n_trials"],
        "seed": state["setup_seed"],
    }


def render_form() -> Optional[Dict[str, Any]]:
    """The setup form; on submit, the request body."""

    unit = {"min_value": 0.0, "max_value": 1.0, "step": 0.05, "persist_state": KEEP}
    st.caption(
        "Pre-filled with the API's documented example (Silverstone in the dry): replace it with your track, weather "
        "and driver."
    )
    with st.form("setup_form"):
        st.markdown("**Track**")
        columns = st.columns(4)
        columns[0].text_input(
            "Track name",
            key="setup_track_name",
            persist_state=KEEP,
            help="A label for the reasoning text only; it is not modelled.",
        )
        columns[1].selectbox(
            "Track type",
            TRACK_TYPES,
            format_func=lambda value: value.replace("_", " "),
            key="setup_track_type",
            persist_state=KEEP,
            help="Sets the heuristic kerb and bump severity.",
        )
        columns[2].number_input(
            "Lap length (m)",
            min_value=0.0,
            step=1.0,
            format="%.0f",
            key="setup_track_length",
            persist_state=KEEP,
            help="500-25,000 m.",
        )
        columns[3].number_input(
            "Average speed (km/h)",
            min_value=0.0,
            step=1.0,
            format="%.0f",
            key="setup_average_speed",
            persist_state=KEEP,
            help="Average lap speed, 20-400 km/h.",
        )
        columns = st.columns(4)
        columns[0].number_input("Corners", min_value=1, step=1, key="setup_corners", persist_state=KEEP)
        columns[1].number_input(
            "High-speed corners",
            min_value=0,
            step=1,
            key="setup_high_speed_sections",
            persist_state=KEEP,
            help="Corners taken at high speed; high- plus low-speed corners cannot exceed the corners.",
        )
        columns[2].number_input("Low-speed corners", min_value=0, step=1, key="setup_low_speed_sections", persist_state=KEEP)
        columns[3].slider(
            "Downforce requirement",
            **unit,
            key="setup_downforce_requirement",
            help="0 = low-downforce track, 1 = maximum downforce.",
        )

        st.markdown("**Weather**")
        columns = st.columns(4)
        columns[0].selectbox(
            "Track condition",
            CONDITIONS,
            key="setup_condition",
            persist_state=KEEP,
            help="Sets grip, standing water and the ride-height need.",
        )
        columns[1].number_input(
            "Air temperature (°C)",
            step=0.5,
            format="%.1f",
            key="setup_temperature",
            persist_state=KEEP,
            help="Ambient temperature, -10 to 50 °C; sets the cold tyre pressures.",
        )
        columns[2].number_input(
            "Humidity (%)",
            value=None,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%.0f",
            key="setup_humidity",
            persist_state=KEEP,
            placeholder="not given",
            help="Validated but not modelled.",
        )
        columns[3].number_input(
            "Wind speed (m/s)",
            value=None,
            min_value=0.0,
            step=0.5,
            format="%.1f",
            key="setup_wind_speed",
            persist_state=KEEP,
            placeholder="not given",
            help="Validated but not modelled.",
        )

        st.markdown("**Driver**")
        columns = st.columns(4)
        columns[0].number_input(
            "Risk tolerance",
            value=None,
            **unit,
            key="setup_risk_tolerance",
            placeholder="API default",
            help=f"0 = wants a stable car, 1 = accepts a pointy car. {API_DEFAULT_HELP}",
        )
        columns[1].number_input(
            "Tyre management",
            value=None,
            **unit,
            key="setup_tire_management",
            placeholder="API default",
            help=f"0 = ignore tyre wear, 1 = protect the tyres. {API_DEFAULT_HELP}",
        )
        with st.expander("Pin setup values (optional)", icon=":material/push_pin:"):
            st.caption(
                "A pinned value is used exactly and left out of the search. Leave a field empty to let the search choose it."
            )
            columns = st.columns(3)
            columns[0].number_input(
                "Ride height (mm)",
                value=None,
                step=0.5,
                format="%.1f",
                key="setup_pin_ride_height",
                persist_state=KEEP,
                placeholder="free",
                help="60-85 mm.",
            )
            columns[1].number_input(
                "Front wing angle (°)",
                value=None,
                step=0.5,
                format="%.1f",
                key="setup_pin_front_wing",
                persist_state=KEEP,
                placeholder="free",
                help="0-15°.",
            )
            columns[2].number_input(
                "Rear wing angle (°)",
                value=None,
                step=0.5,
                format="%.1f",
                key="setup_pin_rear_wing",
                persist_state=KEEP,
                placeholder="free",
                help="0-20°.",
            )
            columns = st.columns(3)
            columns[0].number_input(
                "Differential preload (%)",
                value=None,
                step=1.0,
                format="%.0f",
                key="setup_pin_diff_preload",
                persist_state=KEEP,
                placeholder="free",
                help="0-100 % lock.",
            )
            columns[1].number_input(
                "Differential on throttle (%)",
                value=None,
                step=1.0,
                format="%.0f",
                key="setup_pin_diff_power",
                persist_state=KEEP,
                placeholder="free",
                help="0-100 % lock.",
            )
            columns[2].number_input(
                "Differential off throttle (%)",
                value=None,
                step=1.0,
                format="%.0f",
                key="setup_pin_diff_coast",
                persist_state=KEEP,
                placeholder="free",
                help="0-100 % lock.",
            )
        with st.expander("Search settings", icon=":material/settings:"):
            columns = st.columns(2)
            columns[0].number_input(
                "Search trials",
                min_value=1,
                step=1,
                key="setup_n_trials",
                persist_state=KEEP,
                help="Optuna TPE trial budget (16-300); about 1 s at 128. The first 10 trials are random.",
            )
            columns[1].number_input(
                "Search seed",
                min_value=0,
                step=1,
                key="setup_seed",
                persist_state=KEEP,
                help="The same seed and inputs give an identical response.",
            )
        submitted = st.form_submit_button("Recommend setup", type="primary", icon=":material/tune:", key="setup_submit")
    return build_request() if submitted else None


def value_at(data: Mapping[str, Any], path: Iterable[str]) -> Any:
    for key in path:
        data = data.get(key) if isinstance(data, Mapping) else None
    return data


def search_limit(value: Any, unit: str) -> str:
    """``lowest (0)`` or ``highest (100)`` for a value at an end of its 0-100 scale; empty otherwise."""

    if not isinstance(value, (int, float)) or not ZERO_TO_HUNDRED.search(unit):
        return ""
    return "lowest (0)" if value <= 0 else "highest (100)" if value >= 100 else ""


def render_setup(outcome: Mapping[str, Any]) -> None:
    body = outcome["response"]
    units = body.get("units") or {}
    st.subheader("Recommended setup")
    heuristic_badge()
    if outcome.get("example"):
        example_inputs_badge()
    st.caption(
        f"{md_text(str(body.get('model_scope', '')).capitalize())}. Result for the request sent at "
        f"{time.strftime('%H:%M:%S', time.localtime(outcome['at']))}."
    )
    baseline_value, improvement = body.get("baseline_objective_value"), body.get("improvement_over_baseline")
    columns = st.columns(4)
    columns[0].metric(
        "Objective (lower is better)",
        f"{body.get('objective_value', 0):.3f}",
        help=f"{md_text(units.get('objective_value', ''))}. {md_text(body.get('objective_description', ''))}",
    )
    columns[1].metric("Rule-of-thumb baseline", f"{baseline_value:.3f}" if baseline_value is not None else "–")
    if improvement is not None and baseline_value:
        columns[2].metric(
            "Reduction vs baseline",
            f"{improvement:.3f}",
            delta=f"{improvement / baseline_value:.1%} lower objective",
            delta_color="off",
            delta_arrow="off",
            help="Baseline objective minus the recommended setup's objective.",
        )
    columns[3].metric(
        "Multi-start agreement",
        f"{body.get('confidence', 0):.2f}",
        help=md_text(body.get("confidence_method", "")),
    )
    st.caption(
        f"The objective is a {md_text(units.get('objective_value', 'dimensionless heuristic penalty'))}, not a lap "
        "time: the reduction against the rule-of-thumb baseline is a model score, not a predicted lap-time gain.  \n"
        'The API calls the agreement value "confidence": it is the share of the search\'s start points that '
        "reached the same setup (1.0 = all of them), not a probability that the setup is right or a measure of "
        "the model's accuracy."
    )

    pinned = set(body.get("pinned_by_driver") or [])
    baseline = body.get("baseline_setup") or {}
    rows = []
    for label, path, unit_key, pin in PARAMETERS:
        recommended, base = value_at(body, path), value_at(baseline, path)
        unit = str(units.get(unit_key, ""))
        rows.append({
            "Parameter": label,
            "Recommended": recommended,
            "Search limit": "" if pin in pinned else search_limit(recommended, unit),
            "Baseline": base,
            "Change": None if recommended is None or base is None else round(recommended - base, 2),
            "Unit": unit,
            "Pinned": pin in pinned,
        })  # fmt: skip
    st.markdown("**Setup parameters** (recommended against the rule-of-thumb baseline)")
    st.dataframe(
        rows,
        hide_index=True,
        height=35 * (len(rows) + 1) + 3,  # every parameter without scrolling
        column_config={
            "Recommended": st.column_config.NumberColumn(format="%.2f"),
            "Baseline": st.column_config.NumberColumn(format="%.2f"),
            "Change": st.column_config.NumberColumn(format="%+.2f"),
            "Search limit": st.column_config.TextColumn(help="The search ended at an end of this 0-100 scale."),
            "Pinned": st.column_config.CheckboxColumn(help="Pinned by the driver preferences; not searched."),
        },
    )
    if any(row["Search limit"] for row in rows):
        st.caption(
            "**Search limit**: the value sits at an end of its 0-100 scale, i.e. as soft or stiff (or with as little "
            "or as much lock) as the heuristic allows. The heuristic's linear costs push some parameters there, so "
            "read such a value as a direction, not as a tuned setting. Only the 0-100 scales are checked: the API "
            "does not report the search limits of the other parameters."
        )

    pressures = body.get("tire_pressures_psi") or {}
    if pressures:
        st.markdown(f"**Tyre pressures** ({md_text(units.get('tire_pressures_psi', 'psi'))})")
        columns = st.columns(len(pressures))
        for column, (corner, value) in zip(columns, pressures.items(), strict=True):
            column.metric(corner.replace("_", " ").capitalize(), f"{value:.2f}")
        st.caption(md_text(body.get("tire_pressure_basis", "")))

    balance = body.get("handling_balance") or {}
    if balance:
        st.markdown(f"**Handling balance** ({md_text(units.get('handling_balance', ''))})")
        columns = st.columns(len(balance))
        for column, (name, value) in zip(columns, balance.items(), strict=True):
            column.metric(name.replace("_", " ").capitalize(), f"{value:+.4f}")

    breakdown = body.get("objective_breakdown") or {}
    if breakdown:
        st.markdown("**Remaining objective penalties** (what the recommended setup still trades off)")
        frame = pd.DataFrame(
            [{"Term": name.replace("_", " "), "Penalty": value} for name, value in breakdown.items()]
        ).sort_values("Penalty", ascending=False)
        chart = (
            alt.Chart(frame)
            .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, height=14)
            .encode(
                x=alt.X("Penalty:Q", title="Penalty (dimensionless, lower is better)"),
                y=alt.Y("Term:N", sort=list(frame["Term"]), title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                tooltip=["Term", alt.Tooltip("Penalty:Q", format=".4f")],
            )
            .properties(height=26 * len(frame) + 40)
        )
        st.altair_chart(chart, width="stretch")

    st.markdown("**Reasoning** (from the API)")
    st.markdown(md_text(body.get("reasoning", "")))
    assumed = body.get("assumed_defaults") or []
    st.markdown(
        "**Assumed defaults:** "
        + (
            "; ".join(md_text(f"`{item}`") for item in assumed) + " (not in the request, so the API used its default)"
            if assumed
            else "none; every driver input was supplied."
        )
    )
    st.markdown("**Limitations** (from the API)")
    limitations = [
        f"Scope: {body.get('model_scope', '')}.",
        f"Objective: {body.get('objective_description', '')}",
        f"Tyre pressures: {body.get('tire_pressure_basis', '')}",
    ]
    not_modelled = body.get("inputs_not_modelled") or []
    if not_modelled:
        limitations.append(f"Accepted but not modelled: {', '.join(not_modelled)}.")
    st.markdown("\n".join(f"- {md_text(item)}" for item in limitations))

    with st.expander("Search details", icon=":material/query_stats:"):
        st.markdown(
            f"{md_text(body.get('optimization_method', ''))}: {body.get('trials')} trials (seed {body.get('seed')}); "
            f"best trial #{body.get('best_trial_number')} scored {body.get('best_trial_objective', 0):.3f}; "
            f"selected: {md_text(str(body.get('selected_source', '')).replace('_', ' '))}."
        )
        st.caption(md_text(body.get("refinement_method", "")))
        starts = body.get("refinement_starts") or []
        if starts:
            st.dataframe(
                [
                    {"Start": s["start"].replace("_", " "), "Start objective": s["start_objective"],
                     "Refined objective": s["refined_objective"], "Max parameter deviation": s["max_parameter_deviation"],
                     "Evaluations": s["evaluations"], "Converged": s["converged"]}
                    for s in starts
                ],
                hide_index=True,
            )  # fmt: skip
        spread = body.get("parameter_spread") or {}
        if spread:
            st.caption(
                f"Parameter spread over the starts ({md_text(units.get('parameter_spread', ''))}): "
                + ", ".join(f"{md_text(name)} {value:g}" for name, value in spread.items())
            )


def render_outcome(outcome: Mapping[str, Any]) -> None:
    if outcome.get("error") is not None:
        show_request_error(outcome["error"], "The setup request", FIELDS)
    else:
        render_setup(outcome)
    json_expander(outcome["request"], "Request sent to the API")
    if outcome.get("response") is not None:
        json_expander(outcome["response"])


st.title("Car setup")
heuristic_badge()
st.caption(
    "Searches eleven setup parameters (ride height, wings, brake bias, differential, anti-roll bars, springs) "
    "for the lowest score of a documented heuristic objective that trades downforce against drag, balance, "
    "bottoming, compliance and tyre wear for the track and weather below. It is not a vehicle-dynamics "
    "simulator: validate any setup in a simulator or on track."
)
for widget_key, default in FORM_DEFAULTS.items():
    st.session_state.setdefault(widget_key, default)

setup_request = render_form()
if setup_request is not None:
    result: Dict[str, Any] = {
        "request": setup_request,
        "response": None,
        "error": None,
        "at": time.time(),
        "example": unchanged_example(FORM_DEFAULTS),
    }
    with st.spinner("Searching setups..."):
        try:
            result["response"] = get_client().recommend_setup(setup_request)
        except (ApiUnavailable, ApiError) as exc:
            result["error"] = exc
    st.session_state[OUTCOME_KEY] = result
if st.session_state.get(OUTCOME_KEY):
    render_outcome(st.session_state[OUTCOME_KEY])
