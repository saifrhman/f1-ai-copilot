"""Ask the copilot: a natural-language question routed to one module by keywords (POST /api/query/natural).

Context examples come from the API's own OpenAPI document (the documented StrategyRequest and
SetupRequest examples), so this page never keeps a second copy of the request formats.
"""

from __future__ import annotations

import base64
import json
import math
from typing import Any, Dict, List, Optional

import streamlit as st

from ui.api_client import ApiError, ApiUnavailable, RequestNotSent, encode_json, get_client
from ui.components import (
    CLIP_SOURCES,
    DEFINITIONS_ONLY_NOTE,
    MAX_AUDIO_TEXT,
    UPLOAD,
    api_schema,
    cached_health,
    cites_regulation_passage,
    clip_input,
    decline_explanation,
    documented_example,
    evidence_label,
    heuristic_badge,
    humanise,
    json_expander,
    md_text,
    not_modelled_text,
    oversize_problem,
    page_links,
    profile_tie_note,
    render_passage,
    show_api_error,
    show_input_problems,
    state_badge,
    transcription_state,
)

QUERY_KEY = "assistant_query"
CONTEXT_KEY = "assistant_context"
NOTICE_KEY = "assistant_notice"
RESULT_KEY = "assistant_result"
EXAMPLE_KEY = "assistant_example"
EXAMPLE_VALUES_KEY = "assistant_example_values"  # context key -> value last inserted by a helper
MAX_QUERY_CHARS = 2000  # the API's limit (MAX_QUERY_CHARS in core_modules/llm_query)

EXAMPLE_QUESTIONS = (
    "What is the pit lane speed limit rule?",
    "What pit stop strategy should I use?",
    "What setup should I run for this track?",
    "How does my latest lap compare with my best?",
    "How does the driver sound on the team radio?",
)
# Helper button -> (OpenAPI schema whose documented example is used, keys copied; None = all).
CONTEXT_EXAMPLES = {
    "strategy": ("StrategyRequest", None),
    "setup": ("SetupRequest", None),
    "telemetry": ("StrategyRequest", ("telemetry",)),
}
# (question about, answered by, context it needs)
ROUTING_GUIDE = (
    (
        'FIA rules: penalties, flags, "allowed", "Article 33"',
        "FIA regulation QA: passages from the official PDFs, citation-checked answer, declined without evidence",
        "none",
    ),
    (
        "Race strategy: pit stops, tyres, undercut",
        "Strategy engine (heuristic)",
        "`telemetry`, `car_status`, `driver_profile`, `tire_data`, `race_state`; optional `competition`",
    ),
    (
        "Car setup: wings, ride height, balance",
        "Setup search (heuristic)",
        "`driver_preferences`, `track_profile`, `weather`; optional `n_trials`, `seed`",
    ),
    (
        "Lap performance: lap times, sectors, braking",
        "Restates the supplied telemetry",
        "`telemetry` (`lap_times`, `sector_times`, `braking_consistency`, `throttle_aggressiveness`)",
    ),
    ("Driver radio: mood, tone, emotion", "Radio emotion heuristic", "an attached clip (`audio_file`)"),
)
# query_type -> (label, module, heuristic module)
ROUTES = {
    "regulatory": ("FIA regulations", "the FIA regulation QA", False),
    "strategy": ("Race strategy", "the strategy engine", True),
    "technical": ("Car setup", "the setup search", True),
    "performance": ("Lap performance", "the telemetry summary", False),
    "emotion": ("Driver radio", "the radio emotion heuristic", True),
    "general": ("No module", "no module", False),
}
# query_type -> (name of the API's "confidence" value for that module, what it means)
CONFIDENCE_MEANING = {
    "regulatory": (
        "Evidence strength",
        "Similarity of the best cited regulation passage (0-1); 0 when declined, – when only definitions were cited. "
        "Not a probability.",
    ),
    "strategy": ("Score", "The strategy engine reports time margins between plans, not a confidence."),
    "technical": (
        "Multi-start agreement",
        "Share of the setup search's start points that reached the same setup (0-1). Not a probability.",
    ),
    "performance": (
        "Coverage",
        "Heuristic: 0.8 x the share of the asked-about aspects that the telemetry covers. Not a probability.",
    ),
    "emotion": (
        "Heuristic score",
        "Not a probability: the acoustic confidence, combined with the transcript keyword score when a transcript "
        "is used. The rules are under the answer.",
    ),
}
# data_sources ids of the API -> what the answer used
DATA_SOURCES = {
    "fia_regulations": "FIA regulation passages",
    "strategy_engine": "Strategy engine",
    "setup_optimizer": "Setup search",
    "telemetry": "Supplied telemetry",
    "driver_emotion": "Radio clip analysis",
}
NOT_RUN_MEANING = "The module did not run, so there is no score (the raw response reports 0)."
# query_type -> what to do when the module had no context to work with
MISSING_CONTEXT_STEPS = {
    "strategy": "Add it with **Strategy example** above (example values) or your own JSON, and ask again.",
    "technical": "Add it with **Setup example** above (example values) or your own JSON, and ask again.",
    "performance": "Add it with **Telemetry example** above (example values) or your own JSON, and ask again.",
    "emotion": "Attach a clip under **Attach a driver-radio clip** above and ask again. The Driver radio page shows "
    "the full analysis of a clip.",
}
DECISION_RULES = {
    "no_vocabulary_match": "No known topic word was found, so no module was chosen.",
    "regulatory_cue": "A regulatory cue (rule, penalty, allowed, FIA, ...) sends a question to the regulations, whatever else "
    "it mentions.",
    "highest_score": "The module with the highest keyword score was chosen.",
    "tie_regulatory_priority": "A tie that includes the regulations goes to the regulations.",
    "tie_context_supplied": "A tie went to the only tied module whose required context was supplied.",
    "tie_priority_order": "A tie was broken by the fixed order regulatory > strategy > technical > emotion > performance.",
}


# ------------------------------------------------------------------ context


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not a valid JSON number")


def _finite_number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # e.g. 1e400, which Python reads as infinity
        raise ValueError(f"{text} is beyond the range of a 64-bit number")
    return value


def parse_context(text: str) -> Optional[Dict[str, Any]]:
    """The context editor's JSON object (None when empty); raises ValueError with the reason.

    Only values the client can send are accepted: finite numbers and valid text.
    """

    if not text.strip():
        return None
    try:
        value = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_number)
        encode_json(value)  # as the request body is encoded
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from None
    except RequestNotSent as exc:
        raise ValueError(exc.reason) from None
    except RecursionError:
        raise ValueError("it is nested too deeply") from None
    if not isinstance(value, dict):
        raise ValueError('it must be a JSON object, e.g. {"telemetry": {"lap_times": [95.6, 95.3]}}')
    return value


def insert_example(kind: str) -> None:
    """Button callback: merge a documented example into the context editor."""

    schema, keys = CONTEXT_EXAMPLES[kind]
    try:
        current = parse_context(st.session_state.get(CONTEXT_KEY, "")) or {}
        example = documented_example(api_schema(), schema)
    except ValueError as exc:
        st.session_state[NOTICE_KEY] = f"Fix the context JSON before adding an example: {exc}."
        return
    except (ApiUnavailable, ApiError) as exc:
        st.session_state[NOTICE_KEY] = exc
        return
    if example is None:
        st.session_state[NOTICE_KEY] = f"This API version documents no {schema} example."
        return
    fragment = example if keys is None else {key: example[key] for key in keys if key in example}
    st.session_state[CONTEXT_KEY] = json.dumps({**current, **fragment}, indent=2)
    st.session_state[EXAMPLE_VALUES_KEY] = {**st.session_state.get(EXAMPLE_VALUES_KEY, {}), **fragment}
    st.session_state[NOTICE_KEY] = (
        f"Added {', '.join(f'`{key}`' for key in fragment)} from the API's documented {schema} example: "
        "example values, not your data. Edit them below."
    )


def clear_context() -> None:
    st.session_state[CONTEXT_KEY] = ""
    st.session_state.pop(EXAMPLE_VALUES_KEY, None)


def use_example_question() -> None:
    choice = st.session_state.get(EXAMPLE_KEY)
    if choice:
        st.session_state[QUERY_KEY] = choice
        st.session_state[EXAMPLE_KEY] = None


def build_context(clip: Any, transcribe: bool) -> Optional[Dict[str, Any]]:
    """The context to send: the editor's JSON plus the attached clip; raises ValueError."""

    try:
        context = parse_context(st.session_state.get(CONTEXT_KEY, ""))
    except ValueError as exc:
        raise ValueError(f"Context: {exc}.") from None
    if clip is None:
        return context
    audio = clip.getvalue()
    too_large = oversize_problem(audio)
    if too_large:
        raise ValueError(too_large)
    if context and "audio_file" in context:
        raise ValueError("Context: remove `audio_file` from the JSON, or detach the clip.")
    context = {**(context or {}), "audio_file": base64.b64encode(audio).decode("ascii")}
    if transcribe:
        context["transcribe"] = True
    return context


def describe_context(context: Optional[Dict[str, Any]]) -> str:
    """The context keys sent, marking those that still hold a helper's documented example values."""

    if not context:
        return "none"
    examples = st.session_state.get(EXAMPLE_VALUES_KEY) or {}
    text = ", ".join("audio_file (attached clip)" if key == "audio_file" else key for key in context)
    unchanged = [key for key in context if key in examples and context[key] == examples[key]]
    if len(unchanged) == len(context):
        return f"{text} (all the API's documented example values, not your data)"
    if unchanged:
        return f"{text} ({', '.join(unchanged)}: the API's documented example values, not your data)"
    return text


# ------------------------------------------------------------------ result


def render_regulatory(extra: Dict[str, Any], confidence: Optional[float]) -> None:
    """A grounded answer with its cited passages, or a decline; a rejected answer's citations are never shown as evidence."""

    passages = extra.get("retrieved_passages") or []
    best_retrieved = float(extra.get("top_retrieval_score") or 0.0)
    if extra.get("grounded"):
        st.success(
            "Grounded answer: it cites the passages below, and its citations, rule numbers and numbers were checked "
            "against them.",
            icon=":material/verified:",
        )
        if cites_regulation_passage(passages):
            st.caption(
                f"Evidence strength {confidence or 0.0:.3f} (similarity of the best cited regulation passage; not a "
                f"probability); best retrieval similarity {best_retrieved:.3f}."
            )
        else:
            st.caption(f"{DEFINITIONS_ONLY_NOTE} Best retrieval similarity {best_retrieved:.3f}.")
        if extra.get("referenced_rules"):
            st.caption(f"Rules referenced, each found in the cited passages: {md_text(', '.join(extra['referenced_rules']))}")
        for passage in (p for p in passages if p.get("cited")):
            with st.expander(evidence_label(passage, grounded=True), expanded=True, icon=":material/format_quote:"):
                render_passage(passage)
        others = [p for p in passages if not p.get("cited")]
        if others:
            with st.expander(f"Other evidence given to the model, not cited ({len(others)})", icon=":material/description:"):
                for passage in others:
                    render_passage(passage)
    else:
        reason = extra.get("decline_reason")
        headline, happened, next_step = decline_explanation(reason)
        st.warning(
            f"Declined ({md_text(reason or 'no reason given')}): {headline}. {happened} No unverified answer is shown.",
            icon=":material/block:",
        )
        if next_step:
            st.markdown(f"**Next step:** {next_step}")
        st.caption(
            f"Evidence strength {confidence or 0.0:.3f}: a declined answer always scores 0 (not a probability); "
            f"best retrieval similarity {best_retrieved:.3f}."
        )
        if passages:
            with st.expander(f"Evidence given to the model ({len(passages)})", icon=":material/description:"):
                if any(p.get("cited") for p in passages):
                    st.caption("Passages marked as cited were cited by the rejected output: they are not verified evidence.")
                for passage in passages:
                    render_passage(passage, grounded=False)
    if not passages:
        st.caption("No passage passed the similarity threshold.")
    st.caption("The FIA regulations page shows retrieval settings and passages below the threshold.")


def render_strategy(extra: Dict[str, Any]) -> None:
    st.dataframe(
        [
            {
                "Rank": plan.get("rank"),
                "Plan": plan.get("strategy_id"),
                "Stops": plan.get("pit_stops"),
                "Compounds": " → ".join(plan.get("tire_compounds") or []),
                "Pit laps": ", ".join(str(lap) for lap in plan.get("pit_laps") or []) or "none",
                "Projected time (s)": plan.get("projected_race_time"),
                "Behind best (s)": plan.get("delta_to_best_s"),
            }
            for plan in (extra.get("strategies") or [])[:5]
        ],
        hide_index=True,
        column_config={name: st.column_config.NumberColumn(format="%.3f") for name in ("Projected time (s)", "Behind best (s)")},
    )
    if extra.get("not_modelled_inputs"):
        st.caption(not_modelled_text(extra["not_modelled_inputs"]))


def render_setup(extra: Dict[str, Any]) -> None:
    units = extra.get("units") or {}
    rows = [
        {"Parameter": humanise(name), "Value": extra[name], "Unit": units.get(name, "")}
        for name in ("ride_height", "front_wing_angle", "rear_wing_angle", "brake_bias")
        if name in extra
    ]
    rows += [
        {"Parameter": f"diff {name}", "Value": value, "Unit": units.get("diff_settings", "")}
        for name, value in (extra.get("diff_settings") or {}).items()
    ]
    st.dataframe(rows, hide_index=True)
    if extra.get("confidence_method"):
        st.caption(f"Multi-start agreement (the API's confidence): {md_text(extra['confidence_method'])}")


def render_performance(extra: Dict[str, Any]) -> None:
    st.markdown(f"Asked about: {md_text(', '.join(extra.get('requested') or []) or 'nothing specific')}")
    if extra.get("not_supplied"):
        st.markdown(f"Not in the supplied telemetry: {md_text(', '.join(extra['not_supplied']))}")
    evidence = extra.get("evidence") or {}
    if evidence:
        st.dataframe(
            [{"Telemetry": key, "Supplied value": json.dumps(value)} for key, value in evidence.items()], hide_index=True
        )
    for key, label in (("method", "Method"), ("confidence_basis", "Coverage"), ("lap_numbering", "Lap numbers")):
        if extra.get(key):
            st.caption(f"{label}: {md_text(extra[key])}")


def render_emotion(extra: Dict[str, Any]) -> None:
    status = extra.get("transcription_status")
    note = {
        "not_requested": "transcription not requested",
        "unavailable": f"transcription unavailable: {extra.get('transcription_unavailable_reason')}",
        "empty": "Whisper returned no text",
        "completed": f"transcript: {extra.get('transcription')}",
    }.get(status, str(status))
    st.caption(
        f"Acoustic label {md_text(extra.get('acoustic_emotion'))} ({extra.get('acoustic_confidence')}); {md_text(note)}; "
        f"combination rule: {md_text(humanise(extra.get('evidence_combination')))}. The Driver radio page shows the "
        "features and the profile scores."
    )
    for label, key in (("Acoustic confidence", "acoustic_confidence_rule"), ("Combination rule", "evidence_combination_rule")):
        if extra.get(key):
            st.caption(f"**{label}:** {md_text(extra[key])}")
    tie = profile_tie_note(extra.get("acoustic_profile_scores") or {}, extra.get("acoustic_emotion"))
    if tie:
        st.warning(tie, icon=":material/balance:")
    if extra.get("disclaimer"):
        st.caption(md_text(extra["disclaimer"]))


def render_routing(routing: Dict[str, Any]) -> None:
    with st.expander("How the question was routed", icon=":material/alt_route:"):
        rule = routing.get("decision_rule")
        st.markdown(f"{md_text(routing.get('method', ''))}: {DECISION_RULES.get(rule, md_text(rule))}")
        matched = routing.get("matched_terms") or {}
        st.dataframe(
            [
                {"Module": f"{route_label(kind)} ({kind})", "Score": score, "Matched terms": ", ".join(matched.get(kind, []))}
                for kind, score in (routing.get("scores") or {}).items()
            ],
            hide_index=True,
        )


def route_label(kind: Any) -> str:
    return ROUTES[kind][0] if kind in ROUTES else humanise(kind)


def render_result(result: Dict[str, Any]) -> None:
    body = result["response"]
    kind = body.get("query_type")
    _, module, heuristic = ROUTES.get(kind, (None, humanise(kind), False))
    extra = body.get("additional_context") or {}
    confidence = body.get("confidence")
    sources = body.get("data_sources") or []
    ran = kind == "regulatory" or bool(sources)  # the other modules report 0 when they had nothing to work with

    st.subheader("Answer")
    st.caption(f"Question: {md_text(result['query'])} · context sent: {md_text(result.get('context', 'none'))}")
    if heuristic:
        heuristic_badge()
    with st.container(border=True):
        st.markdown(md_text(body.get("answer", "")))
    columns = st.columns(3)
    columns[0].metric("Routed to", route_label(kind), help=f"Answered by {module} (query_type {kind})")
    score_name, meaning = CONFIDENCE_MEANING.get(kind, ("Score", NOT_RUN_MEANING))
    if confidence is None:
        score = "no score"
    elif not ran:
        score = "not run"
    elif kind == "regulatory" and extra.get("grounded") and not cites_regulation_passage(extra.get("retrieved_passages") or []):
        score = "–"  # only definitions were cited, and they have no similarity
    else:
        score = f"{confidence:.3f}"
    columns[1].metric(score_name, score, help=meaning if ran else NOT_RUN_MEANING)
    columns[2].metric(
        "Data sources",
        ", ".join(DATA_SOURCES.get(source, humanise(source)) for source in sources) or "none",
        help=f"Evidence the answer is based on (data_sources: {', '.join(sources) or 'none'})",
    )
    if kind == "regulatory":
        render_regulatory(extra, confidence)
    elif kind == "general":
        st.info("No module matched: use a topic word from the table above (e.g. rule, strategy, setup, lap, radio).")
    elif not sources:
        st.info(
            f"No evidence was used: this module answers only from context you supply. {MISSING_CONTEXT_STEPS.get(kind, '')}",
            icon=":material/data_info_alert:",
        )
        if kind == "emotion":
            page_links(["radio"])
    elif kind == "strategy":
        render_strategy(extra)
    elif kind == "technical":
        render_setup(extra)
    elif kind == "performance":
        render_performance(extra)
    elif kind == "emotion":
        render_emotion(extra)
    render_routing(body.get("routing") or {})
    json_expander(body)


# ------------------------------------------------------------------ page


st.title("Ask the copilot")
heuristic_badge("Keyword routing · heuristic")
st.markdown(
    "Ask in plain English. The question is sent to **one** module, chosen by a transparent keyword heuristic "
    "(not a language model). **Regulatory questions go through the FIA regulation QA**: it answers only from "
    "passages of the official regulation PDFs, cites them, and **declines** when they do not contain enough "
    "evidence. The other modules answer only from the context you supply; without it they say what they need."
)
try:
    health = cached_health()
except (ApiUnavailable, ApiError):
    health = {}  # the error is shown when a question is sent
rag_state = (health.get("modules") or {}).get("fia_rag")
if rag_state not in (None, "ready"):
    st.markdown(
        f"{state_badge(rag_state)} Regulation QA is not ready on this API: regulatory questions get "
        '"service unavailable" (HTTP 503) until it is. The Overview page shows the fix.'
    )
    page_links(["overview"])
with st.expander("What can I ask?", icon=":material/help:"):
    st.markdown(
        "| Question about | Answered by | Context it needs |\n| --- | --- | --- |\n"
        + "\n".join(f"| {topic} | {module} | {needs} |" for topic, module, needs in ROUTING_GUIDE)
    )

st.pills("Example questions", EXAMPLE_QUESTIONS, key=EXAMPLE_KEY, on_change=use_example_question)
st.text_area(
    "Question",
    key=QUERY_KEY,
    max_chars=MAX_QUERY_CHARS,
    height=90,
    placeholder="e.g. Is DRS allowed under a yellow flag?",
)

st.markdown("**Context** (optional)")
st.caption('A JSON object with the evidence the routed module needs (see "What can I ask?"). The helpers add example values.')
helpers = st.columns(4)
helpers[0].button(
    "Strategy example",
    on_click=insert_example,
    args=("strategy",),
    icon=":material/timeline:",
    key="assistant_add_strategy",
    help="telemetry, car_status, driver_profile, tire_data, race_state and competition from the API's documented example",
    width="stretch",
)
helpers[1].button(
    "Setup example",
    on_click=insert_example,
    args=("setup",),
    icon=":material/tune:",
    key="assistant_add_setup",
    help="driver_preferences, track_profile, weather, n_trials and seed from the API's documented example",
    width="stretch",
)
helpers[2].button(
    "Telemetry example",
    on_click=insert_example,
    args=("telemetry",),
    icon=":material/speed:",
    key="assistant_add_telemetry",
    help="Lap times for performance questions (the telemetry of the API's documented strategy example)",
    width="stretch",
)
helpers[3].button("Clear context", on_click=clear_context, icon=":material/clear:", key="assistant_clear", width="stretch")
notice = st.session_state.pop(NOTICE_KEY, None)
if isinstance(notice, (ApiUnavailable, ApiError)):
    show_api_error(notice, "Loading the example")
elif notice:
    st.info(notice, icon=":material/info:")
st.text_area(
    "Context JSON",
    key=CONTEXT_KEY,
    height=220,
    placeholder='{"telemetry": {"lap_times": [95.6, 95.3, 95.9, 95.4]}}',
    label_visibility="collapsed",
)
with st.expander("Attach a driver-radio clip (for radio questions)", icon=":material/graphic_eq:"):
    clip_source = st.segmented_control("Clip", CLIP_SOURCES, default=UPLOAD, required=True, key="assistant_clip_source")
    clip = clip_input(clip_source, "assistant")
    transcription = transcription_state()
    whisper_missing = transcription["available"] is False
    transcribe = st.toggle(
        "Transcribe with Whisper",
        disabled=whisper_missing,
        key="assistant_transcribe",
        help=f"Unavailable on this API: {transcription['reason']}"
        if whisper_missing
        else "Adds a keyword heuristic on the transcript (slower).",
    )
    st.caption(f"Sent as `audio_file` (base64) with the question; at most {MAX_AUDIO_TEXT}.")

if st.button("Ask", type="primary", icon=":material/send:", key="assistant_ask"):
    st.session_state.pop(RESULT_KEY, None)
    query = st.session_state.get(QUERY_KEY, "").strip()
    problems: List[str] = [] if query else ["Type a question."]
    context: Optional[Dict[str, Any]] = None
    try:
        context = build_context(clip, transcribe and not whisper_missing)
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        show_input_problems(problems)
    else:
        try:
            with st.spinner("Routing the question..."):
                body = get_client().natural_query(query, context)
        except (ApiUnavailable, ApiError) as exc:
            show_api_error(exc, "The question")
        else:
            st.session_state[RESULT_KEY] = {"query": query, "context": describe_context(context), "response": body}

if RESULT_KEY in st.session_state:
    render_result(st.session_state[RESULT_KEY])
