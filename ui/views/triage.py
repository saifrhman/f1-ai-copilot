"""Incident triage: the API's transparent severity heuristic (not a steward-decision predictor).

Every value shown comes from ``POST /api/penalty/predict``. Rule questions belong to the
regulations page, which answers from the official FIA documents with citations.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import streamlit as st

from ui.api_client import get_client
from ui.components import (
    KEEP,
    FieldLabels,
    clock_time,
    example_inputs_badge,
    heuristic_badge,
    humanise,
    md_text,
    page_links,
    render_form_outcome,
    seed_form,
    submit_form,
    unchanged_example,
)

INCIDENT_TYPES = (
    "track_limits", "unsafe_release", "collision", "blocking", "dangerous_driving",
    "technical_infringement", "speeding_in_pit", "illegal_overtaking",
)  # fmt: skip
TRACK_CONDITIONS = ("dry", "wet", "intermediate", "mixed", "unknown")
INTENTS = ("accidental", "intentional", "racing_incident", "unknown")

FORM_DEFAULTS: Dict[str, Any] = {
    "triage_incident_type": "collision",
    "triage_track_condition": "wet",
    "triage_intent": "racing_incident",
    "triage_recent_penalties": "5-second time penalty for track limits\nGrid drop for an unsafe release",
    "triage_total_penalties": 4,
}
OUTCOME_KEY = "triage_outcome"

# API field names in the words of the form, for validation errors (HTTP 422).
FIELD_LABELS = {
    "incident_type": "Incident type", "track_condition": "Track condition", "intent": "Intent",
    "driver_history": "Driver history", "recent_penalties": "Recent penalties",
    "total_penalties": "Total penalties on record",
}  # fmt: skip
FIELDS = FieldLabels(FIELD_LABELS)


def build_request() -> Dict[str, Any]:
    """The PenaltyRequest body; driver history is sent only when some of it is known."""

    state = st.session_state
    request: Dict[str, Any] = {
        "incident_type": state["triage_incident_type"],
        "track_condition": state["triage_track_condition"],
        "intent": state["triage_intent"],
    }
    history: Dict[str, Any] = {}
    recent = [line.strip() for line in state["triage_recent_penalties"].splitlines() if line.strip()]
    if recent:
        history["recent_penalties"] = recent
    if state["triage_total_penalties"] is not None:
        history["total_penalties"] = state["triage_total_penalties"]
    if history:
        request["driver_history"] = history
    return request


def render_form() -> Optional[Dict[str, Any]]:
    """The triage form; on submit, the request body."""

    st.caption(
        "Pre-filled with an example incident (a collision in the wet, with two recent penalties): replace it with "
        "the incident you want to triage."
    )
    with st.form("triage_form"):
        columns = st.columns(3)
        columns[0].selectbox(
            "Incident type",
            INCIDENT_TYPES,
            format_func=humanise,
            key="triage_incident_type",
            persist_state=KEEP,
            help="Sets the review category and the base severity.",
        )
        columns[1].selectbox(
            "Track condition",
            TRACK_CONDITIONS,
            format_func=humanise,
            key="triage_track_condition",
            persist_state=KEEP,
            help="Context only: it never changes the severity. Use 'unknown' when it is not known.",
        )
        columns[2].selectbox(
            "Intent",
            INTENTS,
            format_func=humanise,
            key="triage_intent",
            persist_state=KEEP,
            help=(
                "As described: 'intentional' raises on-track driving cases, 'racing incident' lowers car-to-car "
                "cases. Use 'unknown' when it is not known."
            ),
        )
        st.markdown("**Driver history** (optional; leave both empty when it is unknown)")
        columns = st.columns([3, 1])
        columns[0].text_area(
            "Recent penalties, one per line",
            key="triage_recent_penalties",
            persist_state=KEEP,
            height=100,
            help="Only the count is used: three or more raise the severity.",
        )
        columns[1].number_input(
            "Total penalties on record",
            value=None,
            min_value=0,
            step=1,
            key="triage_total_penalties",
            persist_state=KEEP,
            placeholder="unknown",
            help=(
                "Ten or more raise the severity slightly; 0 = a clean record. Cannot be smaller than the number of "
                "recent penalties."
            ),
        )
        submitted = st.form_submit_button("Triage incident", type="primary", icon=":material/flag:", key="triage_submit")
    return build_request() if submitted else None


def render_triage(outcome: Mapping[str, Any]) -> None:
    body = outcome["response"]
    st.subheader("Triage result")
    heuristic_badge()
    if outcome.get("example"):
        example_inputs_badge()
    st.caption(
        f"Method: {md_text(body.get('method', ''))}. Result for the request sent at "
        f"{clock_time(outcome['at'])}."
    )
    # Text, not a metric: a metric value never wraps, and the category is the main result.
    st.markdown(
        f"#### Review category: {md_text(humanise(body.get('triage_category', '')))}",
        anchors=False,
        help="Which kind of steward review the case needs; not a predicted sanction.",
    )
    columns = st.columns(3)
    columns[0].metric(
        "Severity band", str(body.get("severity_band", "")), help="low below 0.50, medium from 0.50, high from 0.80."
    )
    columns[1].metric(
        "Severity score",
        f"{body.get('severity_score', 0):.2f}",
        help="0-1: the incident type's base severity plus the adjustments below.",
    )
    columns[2].metric(
        "Confidence (input completeness)", f"{body.get('confidence', 0):.2f}", help=md_text(body.get("confidence_basis", ""))
    )
    st.caption(
        "The API's \"confidence\" measures how many of the inputs that can change this incident type's severity "
        "were supplied. It is not a probability of any steward decision."
    )

    adjustments = body.get("severity_adjustments") or []
    st.markdown("**Severity adjustments**")
    if adjustments:
        st.dataframe(
            [
                {"Input": humanise(item["source"]), "Value": humanise(item["value"]), "Change": item["delta"]}
                for item in adjustments
            ],
            hide_index=True,
            column_config={"Change": st.column_config.NumberColumn(format="%+.2f")},
        )
    else:
        st.caption("None: the base severity of this incident type applies unchanged.")
    counted = body.get("confidence_inputs") or {}
    if counted:
        st.caption(
            "Inputs counted for completeness: "
            + ", ".join(f"{humanise(name)} {'supplied' if given else 'not supplied'}" for name, given in counted.items())
            + "."
        )

    st.markdown("**Reasoning** (from the API)")
    st.markdown(md_text(body.get("reasoning", "")))
    rule = body.get("referenced_rule")
    st.markdown(
        f"**Referenced rule:** {md_text(rule)}"
        if rule
        else "**Referenced rule:** none. The triage never cites FIA articles; ask the regulations page what the rules say."
    )
    if body.get("disclaimer"):
        st.caption(md_text(body["disclaimer"]))


st.title("Incident triage")
heuristic_badge()
st.warning(
    "**Heuristic triage, not an FIA steward-decision predictor.** It sorts an incident into a review category "
    "with a preliminary severity from a transparent rule table. It does not predict penalties, is not a legal "
    "or regulatory determination and never cites FIA articles. For what the regulations say, ask the FIA "
    "regulations page, which answers only from the official documents with citations.",
    icon=":material/gavel:",
)
page_links(["regulations"])
seed_form(FORM_DEFAULTS)

triage_request = render_form()
if triage_request is not None:
    st.session_state[OUTCOME_KEY] = submit_form(
        triage_request, get_client().predict_penalty, example=unchanged_example(FORM_DEFAULTS)
    )
if st.session_state.get(OUTCOME_KEY):
    render_form_outcome(st.session_state[OUTCOME_KEY], "The triage request", FIELDS, render_triage)
