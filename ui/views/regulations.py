"""FIA regulations: questions answered only from the official regulation PDFs, with checked citations.

"Ask" calls POST /api/fia/query (retrieval, one answer-model call, validation); "Search passages"
calls POST /api/fia/retrieve (retrieval only, no answer model). Both show what the API returned and
nothing else: declines stay declines, and a rejected model output is only shown as rejected.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import streamlit as st

from ui.api_client import ApiError, ApiUnavailable, get_client
from ui.components import (
    DEFINITIONS_ONLY_NOTE,
    cached_health,
    cites_regulation_passage,
    clear_health_cache,
    code_span,
    decline_explanation,
    evidence_label,
    humanise,
    index_metrics,
    is_definition,
    json_expander,
    location,
    md_text,
    pdf_link,
    provider_metric,
    refresh_health_once,
    render_passage,
    render_rag_problems,
    score_text,
    show_api_error,
    state_badge,
    uses_compose,
)

# The API's request limits (FIAQueryRequest.question and RetrievalConfig.MAX_TOP_K).
MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 50
# Every request fails in these states until the setup is fixed (the next steps say how). In the others
# (provider failing, storage unavailable) a new attempt can succeed, so the forms stay usable.
BLOCKING_STATES = frozenset(
    {"not_configured", "misconfigured", "index_missing", "index_empty", "index_stale", "index_incomplete"}
)
# Answerable questions, verbatim from scripts/fia_rag_eval_questions.json (short label, question).
EXAMPLE_QUESTIONS: Tuple[Tuple[str, str], ...] = (
    (
        "Pit lane speed limit",
        "What is the pit lane speed limit, and what is the fine for exceeding it during a free practice session?",
    ),
    (
        "Unsafe release",
        "What happens if a car is released from its pit stop position in an unsafe condition during the race?",
    ),
    ("Minimum mass", "What is the minimum mass of the car during the qualifying sessions?"),
    ("Factory shutdown", "What factory shutdown periods must F1 teams observe?"),
)
# A truncated_model_output decline names what stopped the reply (validation.finish_reason) ->
# (what happened, what the user can do). Providers call the output limit "length" or "max_tokens".
_OUTPUT_LIMIT = (
    "The model's reply reached its output limit and could end mid-sentence; incomplete answers are never shown.",
    "Ask a narrower question, or raise `FIA_RAG_MAX_OUTPUT_TOKENS` for the API.",
)
FINISH_REASONS: Dict[str, Tuple[str, str]] = {
    "length": _OUTPUT_LIMIT,
    "max_tokens": _OUTPUT_LIMIT,
    "content_filter": (
        "The provider's content filter stopped the model's reply before it finished; incomplete answers are never shown.",
        "Rephrasing the question can help; a higher output limit does not.",
    ),
}
# validation field -> label, shown for rejected answers and in the validation details
VALIDATION_LISTS: Tuple[Tuple[str, str], ...] = (
    ("invalid_citations", "Cited labels that were not supplied"),
    ("unsupported_rules", "Rule references not found in the cited passages"),
    ("unsupported_numbers", "Numbers not found in the cited passages"),
    ("uncited_claims", "Statements without a citation"),
    ("unverified_claims", "Sentences the claim verifier flagged"),
)
EVIDENCE_COLOR = "#2a78d6"  # passages above the threshold
BELOW_THRESHOLD_COLOR = "#898781"  # greyed: retrieved, but not evidence
_RESYNCED_KEY = "_f1_regulations_resynced"
ASK_OUTCOME_KEY = "_f1_fia_ask_outcome"  # last answer or API error, kept for the session
SEARCH_OUTCOME_KEY = "_f1_fia_search_outcome"
INPUT_DEFAULTS_KEY = "_f1_fia_input_defaults"  # input key -> the API default it last followed

Outcome = Union[Dict[str, Any], ApiUnavailable, ApiError]


# ------------------------------------------------------------------ formatting


def reference(passage: Dict[str, Any]) -> str:
    """Short identity of a passage: nearest rule (or defined term) and printed page (plain text, not Markdown)."""

    subject = passage.get("defined_term") if is_definition(passage) else passage.get("nearest_rule")
    page = passage.get("page_label") or (f"PDF p. {passage['page']}" if passage.get("page") is not None else None)
    return " · ".join(str(part) for part in (subject or "no rule heading", page) if part)


def with_citation_chips(answer: str, spans: Sequence[Dict[str, Any]]) -> str:
    """Markdown for an answer: the API text as written, each citation the API located as a badge of its labels.

    ``spans`` is the answer's ``citation_spans``: where each citation is written in it and the labels it names
    ("[source: 2 & 3]" -> S2, S3; "[S1-S3]" -> S1, S2, S3; in "source S2" only the label is replaced).
    """

    text = ""
    last = 0
    for span in spans:
        start, end = span["start"], span["end"]
        if not last <= start < end <= len(answer):  # malformed: show the text as written
            continue
        text += md_text(answer[last:start], keep_emphasis=True)
        separator = "" if not text or text[-1].isspace() else " "  # "[S1][S3]": a badge needs a space before it
        text += separator + f":blue-badge[{md_text(', '.join(span['labels']))}]"
        last = end
    return text + md_text(answer[last:], keep_emphasis=True)


# ------------------------------------------------------------------ passages


def evidence_table(passages: Sequence[Dict[str, Any]], grounded: bool) -> None:
    cited_column = "Cited" if grounded else "Cited (rejected)"
    rows = [
        {
            "Label": p.get("label"),
            cited_column: bool(p.get("cited")),
            "Kind": "definition" if is_definition(p) else "regulation",
            "Rule / defined term": p.get("defined_term") if is_definition(p) else p.get("nearest_rule"),
            "Section": p.get("section"),
            "Printed page": p.get("page_label"),
            "PDF page": p.get("page"),
            "Similarity": None if is_definition(p) else p.get("score"),
            "Official PDF": pdf_link(p),
        }
        for p in passages
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            cited_column: st.column_config.CheckboxColumn(),
            "Similarity": st.column_config.ProgressColumn(
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
                color=EVIDENCE_COLOR,
                help="Cosine similarity to the question (none for definitions)",
            ),
            "Official PDF": st.column_config.LinkColumn(display_text="open at page"),
        },
    )


def render_below_threshold(
    passages: Sequence[Dict[str, Any]], min_score: Any, note: str, first_rank: int = 1, expanded: bool = False
) -> None:
    """Retrieved passages under the threshold, greyed: they are not evidence."""

    if not passages:
        return
    with st.expander(f"Below the threshold ({len(passages)}): {note}", expanded=expanded):
        st.caption(f"Similarity under {score_text(min_score)}. Listed for inspection only; they are not evidence.")
        for rank, passage in enumerate(passages, start=first_rank):
            rule = f" · {code_span(passage['nearest_rule'])}" if passage.get("nearest_rule") else ""
            st.caption(
                f"**#{rank} · similarity {score_text(passage.get('score'))}** · {md_text(location(passage))}{rule} · "
                f"{md_text(passage.get('source') or '')}"
            )
            st.caption(md_text(passage.get("text", "")))


# ------------------------------------------------------------------ Ask


def render_citations(citations: Sequence[str], by_label: Dict[str, Dict[str, Any]]) -> None:
    """One chip per cited label; clicking it shows the passage."""

    with st.container(horizontal=True, gap="small"):
        for label in citations:
            if label in by_label:
                with st.popover(f"{label} · {md_text(reference(by_label[label]))}", icon=":material/format_quote:"):
                    render_passage(by_label[label])


def render_decline(body: Dict[str, Any]) -> None:
    reason = body.get("decline_reason")
    validation = body.get("validation") or {}
    retrieval = body.get("retrieval") or {}
    rejected = validation.get("rejected_model_output")
    headline, happened, next_step = decline_explanation(reason)
    if reason == "truncated_model_output" and validation.get("finish_reason") in FINISH_REASONS:
        happened, next_step = FINISH_REASONS[validation["finish_reason"]]
    with st.container(border=True, key="fia_decline_card"):
        st.badge("Declined: no answer is shown", icon=":material/block:", color="orange")
        st.markdown(f"**{headline}**")
        st.markdown(happened)
        if reason == "no_evidence_above_threshold":
            st.markdown(
                f"Best similarity {score_text(body.get('top_retrieval_score'))}, threshold "
                f"{score_text(retrieval.get('min_score'))}."
            )
        for field, label in VALIDATION_LISTS:
            values = validation.get(field) or []
            if values:
                st.markdown(f"{label}:\n" + "\n".join(f"- {md_text(value)}" for value in values))
        if next_step:
            st.markdown(f"**What you can do:** {next_step}")
        st.caption(f"Reason code {code_span(reason)}. API message: {md_text(body.get('answer', ''))}")
        if rejected:
            with st.expander("Rejected model output: failed validation, not an answer", key="fia_rejected_output"):
                st.markdown("Shown for inspection only. It is not a statement of the regulations.")
                st.code(rejected, language=None, wrap_lines=True)


def render_validation(body: Dict[str, Any]) -> None:
    validation = body.get("validation") or {}
    retrieval = body.get("retrieval") or {}
    models = body.get("models") or {}
    with st.expander("Validation details", icon=":material/fact_check:"):
        rules = body.get("referenced_rules") or []
        lines = [
            "Referenced rules (each found in the cited passages): "
            + (", ".join(code_span(rule) for rule in rules) if rules else "none"),
            "Citations: " + (", ".join(code_span(label) for label in body.get("citations") or []) or "none"),
        ]
        for field, label in VALIDATION_LISTS:
            values = validation.get(field) or []
            lines.append(f"{label}: " + ("; ".join(md_text(value) for value in values) if values else "none"))
        st.markdown("\n".join(f"- {line}" for line in lines))
        report = validation.get("claim_verification")
        if report:
            st.markdown(
                f"**Claim verification:** {md_text(humanise(report.get('status', 'unknown')))} · "
                f"{report.get('sentences', '?')} sentences checked · unsupported sentence numbers: "
                f"{md_text(report.get('unsupported'))}"
            )
            if report.get("verifier_output"):
                st.code(report["verifier_output"], language=None, wrap_lines=True)
        else:
            st.caption("Claim verification did not run (`FIA_RAG_VERIFY_CLAIMS` is off, or the answer was declined first).")
        st.caption(
            f"Retrieval: top_k {retrieval.get('top_k')} · threshold {score_text(retrieval.get('min_score'))} · "
            f"passages above it: {retrieval.get('passages_above_threshold')} · definitions added: "
            f"{retrieval.get('definitions_added')} · duplicates removed: {retrieval.get('duplicates_removed')}. "
            f"Embedding model {code_span(models.get('embedding'))}, answer model "
            f"{code_span(models.get('generation') or 'not called')}."
        )


def checks_note(report: Optional[Dict[str, Any]]) -> str:
    """What the validation of a grounded answer established, and what it did not."""

    checked = "Checked: every citation is a supplied passage, and every rule number and number occurs in the passages it cites."
    if report and report.get("status") == "verified":
        return (
            f"{checked} The claim verifier (a second model call) also judged every sentence supported by its cited "
            "passages: a model's judgement, not proof. Read the cited passages."
        )
    return (
        f"{checked} Not checked: whether each sentence says what its passage says (the claim verifier is off). "
        "Read the cited passages."
    )


def render_answer(body: Dict[str, Any]) -> None:
    passages = body.get("retrieved_passages") or []
    retrieval = body.get("retrieval") or {}
    grounded = bool(body.get("grounded"))
    st.markdown(f"**Question:** {md_text(body.get('question', ''))}")
    if grounded:
        citations = body.get("citations") or []
        with st.container(border=True, key="fia_answer_card"):
            st.badge("Grounded answer", icon=":material/verified:", color="green")
            st.markdown(with_citation_chips(body.get("answer", ""), body.get("citation_spans") or []))
            render_citations(citations, {p.get("label"): p for p in passages})
            st.markdown(checks_note((body.get("validation") or {}).get("claim_verification")))
        regulation_cited = cites_regulation_passage(passages)
        columns = st.columns(3)
        columns[0].metric(
            "Evidence strength",
            score_text(body.get("confidence")) if regulation_cited else "–",
            help="Best similarity among the cited regulation passages (0 to 1): not a probability of correctness.",
        )
        columns[1].metric("Passages cited", f"{len(citations)} of {len(passages)}")
        columns[2].metric("Rules referenced", len(body.get("referenced_rules") or []))
        if regulation_cited:
            st.markdown(
                "Evidence strength is the best similarity of the cited regulation passages to the question, not a "
                "probability: it says how close they are to the question, not whether the answer is right."
            )
        else:
            st.markdown(DEFINITIONS_ONLY_NOTE)
    else:
        render_decline(body)
    st.subheader("Evidence given to the model")
    if passages:
        st.caption(
            "Passages above the similarity threshold in rank order, then official definitions of terms they use "
            "or the question names. Their labels are the only valid citations. Expand a passage to read it."
        )
        evidence_table(passages, grounded)
        for passage in passages:
            icon = ":material/menu_book:" if is_definition(passage) else ":material/article:"
            with st.expander(evidence_label(passage, grounded), icon=":material/format_quote:" if passage.get("cited") else icon):
                render_passage(passage, grounded)
    else:
        st.caption("None: no passage reached the similarity threshold, so nothing was given to the model.")
    above = retrieval.get("passages_above_threshold") or 0
    render_below_threshold(
        retrieval.get("below_threshold") or [], retrieval.get("min_score"), "retrieved, not given to the model", above + 1
    )
    render_validation(body)
    json_expander(body)


# ------------------------------------------------------------------ Search passages


def score_chart(passages: Sequence[Dict[str, Any]], below: Sequence[Dict[str, Any]], min_score: float) -> None:
    """Horizontal bars of the similarity of every retrieved passage in rank order, the threshold as a dashed rule."""

    above, rejected = "Above the threshold", "Below the threshold"
    ranked = [(p, above) for p in passages] + [(p, rejected) for p in below]  # the API's split, not recomputed
    rows = [
        {
            "passage": f"#{rank} {reference(p)}",
            "similarity": p.get("score"),
            "group": group,
            "location": location(p),
            "source": p.get("source"),
        }
        for rank, (p, group) in enumerate(ranked, start=1)
    ]
    lowest = min([0.0, *(row["similarity"] or 0.0 for row in rows)])
    x_scale = {"domain": [lowest, 1.0]}
    # The legend lists only the groups this result has (no "Below the threshold" entry when none is below).
    present = {group for _, group in ranked}
    groups = [(group, color) for group, color in ((above, EVIDENCE_COLOR), (rejected, BELOW_THRESHOLD_COLOR)) if group in present]
    spec = {
        "layer": [
            {
                "mark": {"type": "bar", "cornerRadiusEnd": 4, "size": 16},
                "encoding": {
                    "y": {"field": "passage", "type": "nominal", "sort": None, "title": None, "axis": {"labelLimit": 260}},
                    "x": {
                        "field": "similarity",
                        "type": "quantitative",
                        "scale": x_scale,
                        "axis": {"tickCount": 10},
                        "title": "Similarity",  # short: long titles are cut off on a phone
                    },
                    "color": {
                        "field": "group",
                        "type": "nominal",
                        "scale": {"domain": [group for group, _ in groups], "range": [color for _, color in groups]},
                        # one entry per row: side by side they are cut off on a phone
                        "legend": {"orient": "bottom", "direction": "vertical", "title": None, "labelLimit": 320},
                    },
                    "tooltip": [
                        {"field": "passage", "title": "Passage"},
                        {"field": "similarity", "type": "quantitative", "format": ".3f", "title": "Similarity"},
                        {"field": "group", "title": "Status"},
                        {"field": "location", "title": "Location"},
                        {"field": "source", "title": "File"},
                    ],
                },
            },
            {
                "mark": {"type": "rule", "strokeDash": [4, 3], "strokeWidth": 2, "color": "#52514e"},
                "encoding": {"x": {"datum": min_score, "scale": x_scale}},
            },
        ],
    }
    st.vega_lite_chart(rows, spec, width="stretch", height=28 * len(rows) + 110)  # the spec's own height is ignored
    st.caption(f"Cosine similarity of each retrieved passage to the question; dashed line: threshold {min_score:.3f}.")


def render_retrieval(body: Dict[str, Any], default_min_score: Any) -> None:
    """A search result. Ask always uses the API default threshold, so only a search at it says what Ask would do."""

    passages = body.get("passages") or []
    below = body.get("below_threshold") or []
    definitions = body.get("definitions") or []
    min_score = body.get("min_score")
    as_ask = isinstance(min_score, (int, float)) and isinstance(default_min_score, (int, float))
    as_ask = as_ask and abs(min_score - default_min_score) < 1e-9
    st.markdown(f"**Question:** {md_text(body.get('question', ''))}")
    columns = st.columns(4)
    columns[0].metric("Best similarity", score_text(body.get("top_score")))
    columns[1].metric(
        "Above the threshold",
        len(passages),
        help="At the API default threshold, the passages Ask gives the answer model (for the same top_k)"
        if as_ask
        else "Passages reaching this threshold (not the one Ask uses)",
    )
    columns[2].metric("Below the threshold", len(below), help="Retrieved, but not evidence")
    columns[3].metric("Definitions added", len(definitions))
    st.caption(
        f"top_k {body.get('top_k')}, threshold {score_text(min_score)}, duplicates removed: {body.get('duplicates_removed', 0)}."
    )
    st.markdown("No answer model was called: these are search results, not an answer.")
    if not as_ask:
        default = f" ({score_text(default_min_score)})" if isinstance(default_min_score, (int, float)) else ""
        st.markdown(
            f"Ask always uses the API default threshold{default}; this threshold is for exploring the scores "
            "only, so this split is not what Ask would use."
        )
    if passages or below:
        score_chart(passages, below, float(min_score or 0.0))
    st.subheader(f"Above the threshold ({len(passages)})")
    if not passages:
        st.caption(
            "None: at the API default threshold, Ask would decline this question without calling the answer model."
            if as_ask
            else "None at this threshold."
        )
    for rank, passage in enumerate(passages, start=1):
        with st.expander(f"#{rank} · {md_text(reference(passage))} · similarity {score_text(passage.get('score'))}"):
            render_passage(passage)
    note = "retrieved, would not be given to the model" if as_ask else "retrieved, not evidence at this threshold"
    render_below_threshold(below, min_score, note, len(passages) + 1, expanded=True)
    if definitions:
        st.subheader(
            f"Definitions Ask would add ({len(definitions)})" if as_ask else f"Definitions at this threshold ({len(definitions)})"
        )
        st.caption(
            "Official definitions of terms that the passages above use or the question names; they supplement "
            "the evidence and are never evidence alone."
        )
        for passage in definitions:
            with st.expander(md_text(f"{passage.get('defined_term') or 'Definition'} · {location(passage)}")):
                render_passage(passage)
    json_expander(body)


# ------------------------------------------------------------------ status and forms


def is_blocked(fia: Dict[str, Any]) -> bool:
    """Whether every request fails until the setup is fixed (the status card shows the next steps)."""

    glossary = (fia.get("index") or {}).get("glossary") or {}
    # A missing definitions glossary fails every request too (the API reports it as "unavailable").
    glossary_missing = glossary.get("status") == "missing" and bool((fia.get("settings") or {}).get("max_definitions"))
    return fia.get("state") in BLOCKING_STATES or glossary_missing


def render_index_status(fia: Dict[str, Any], base_url: str) -> None:
    """State, size and models of the regulation index; problems and fix steps when it is not ready."""

    state = fia.get("state")
    settings = fia.get("settings") or {}
    with st.container(border=True):
        head, refresh = st.columns([5, 1], vertical_alignment="center")
        head.markdown(f"{state_badge(state)} **Regulation index**")
        if refresh.button("Refresh", icon=":material/refresh:", key="fia_refresh_status", width="stretch"):
            clear_health_cache()
            st.rerun()
        if settings:  # absent when the settings themselves are invalid
            with st.container(horizontal=True, gap="medium"):  # wraps on a phone instead of stacking five rows
                st.metric("Documents", len(fia.get("documents") or []), width="content")
                index_metrics(fia.get("index") or {})
                st.metric(
                    "Claim verifier",
                    "on" if settings.get("verify_claims") else "off",
                    help="FIA_RAG_VERIFY_CLAIMS: a second model call checks every sentence of an accepted answer",
                    width="content",
                )
                provider_metric(fia)
            st.caption(
                f"Embedding model {code_span(settings.get('embedding_model'))} · answer model "
                f"{code_span(settings.get('generation_model'))} · default top_k {settings.get('top_k')} · similarity "
                f"threshold {score_text(settings.get('min_score'))}"
            )
        render_rag_problems(fia, uses_compose(base_url, settings))
        if is_blocked(fia):
            st.info(
                "Asking and searching are disabled until the index is ready. Results from earlier in this session stay visible.",
                icon=":material/lock:",
            )
        elif state == "provider_failing":
            st.markdown(
                "Submitting again calls the model provider again (with its automatic retries); the state clears "
                "after a successful request."
            )


def sidebar_state() -> Optional[str]:
    """The regulation QA state in the cached /health snapshot, which the status sidebar shows."""

    try:
        return (cached_health().get("modules") or {}).get("fia_rag")
    except (ApiUnavailable, ApiError):
        return None


def follow_api_defaults(settings: Dict[str, Any]) -> None:
    """Start the retrieval inputs from the API defaults, and follow a default that changes (e.g. after a fix).

    The inputs are keyed widgets, which keep their first value whatever ``value`` later says; so their
    values are set here instead. A value the user changed is kept.
    """

    top_k, min_score = settings.get("top_k"), settings.get("min_score")
    if top_k is None or min_score is None:
        return  # unknown (API unreachable or settings invalid): the forms are disabled until they are known
    followed = st.session_state.setdefault(INPUT_DEFAULTS_KEY, {})
    for key, default in (("fia_ask_top_k", top_k), ("fia_search_top_k", top_k), ("fia_search_min_score", float(min_score))):
        previous = followed.get(key)
        if key not in st.session_state or (default != previous and (previous is None or st.session_state[key] == previous)):
            st.session_state[key] = default
        followed[key] = default


def fill_question(target: str, question: str) -> None:
    st.session_state[target] = question


def render_examples(target: str, disabled: bool) -> None:
    st.caption("Examples from the project's evaluation questions (click to fill in the question):")
    with st.container(horizontal=True, gap="small"):
        for index, (label, question) in enumerate(EXAMPLE_QUESTIONS):
            st.button(
                label,
                key=f"{target}_example_{index}",
                help=question,
                on_click=fill_question,
                args=(target, question),
                disabled=disabled,
                icon=":material/lightbulb:",
            )


def top_k_input(key: str, settings: Dict[str, Any], disabled: bool) -> Optional[int]:
    return st.number_input(
        "Passages to retrieve (top_k)",
        min_value=1,
        max_value=MAX_TOP_K,
        # The value comes from follow_api_defaults; None (an empty box) only while the default is unknown,
        # since it also makes the box clearable.
        value=None if settings.get("top_k") is None else "min",
        step=1,
        key=key,
        disabled=disabled,
        help=f"API default: {settings.get('top_k', 'unknown')} (FIA_RAG_TOP_K). More passages can bring in other "
        "documents; every passage above the threshold is given to the model.",
    )


def question_input(key: str, disabled: bool) -> None:
    st.text_area(
        "Question",
        key=key,
        max_chars=MAX_QUESTION_CHARS,
        height=90,
        disabled=disabled,
        placeholder="e.g. What is the pit lane speed limit?",
    )


def submit(question_key: str, outcome_key: str, action: Callable[[str], Dict[str, Any]], spinner: str) -> bool:
    """Run ``action(question)`` for the submitted question; the outcome (body or API error) is kept for the session."""

    question = (st.session_state.get(question_key) or "").strip()
    if not question:
        st.warning("Enter a question first.", icon=":material/edit:")
        return False
    with st.spinner(spinner):
        try:
            outcome: Outcome = action(question)
        except (ApiUnavailable, ApiError) as exc:
            outcome = exc
    st.session_state[outcome_key] = outcome
    return True


def render_outcome(outcome_key: str, action: str, render_body: Callable[[Dict[str, Any]], None]) -> None:
    outcome = st.session_state.get(outcome_key)
    if isinstance(outcome, (ApiUnavailable, ApiError)):
        show_api_error(outcome, action)
    elif outcome is not None:
        render_body(outcome)


def render_ask_tab(settings: Dict[str, Any], usable: bool) -> bool:
    render_examples("fia_ask_question", not usable)
    with st.form("fia_ask_form"):
        question_input("fia_ask_question", not usable)
        top_k = top_k_input("fia_ask_top_k", settings, not usable)
        submitted = st.form_submit_button(
            "Ask", type="primary", icon=":material/send:", key="fia_ask_submit", disabled=not usable
        )
    st.caption(
        "Model-provider requests per question: one query embedding (none if this exact question was embedded "
        "before) and, if a passage reaches the threshold, one answer-model call (two with the claim verifier). "
        "A failed call is retried automatically, up to `FIA_RAG_MAX_RETRIES` more times (default 2)."
    )
    requested = submitted and submit(
        "fia_ask_question",
        ASK_OUTCOME_KEY,
        lambda question: get_client().fia_query(question, top_k),
        "Retrieving passages, then generating and checking the answer (model calls can take a while)...",
    )
    render_outcome(ASK_OUTCOME_KEY, "The question", render_answer)
    return requested


def render_search_tab(settings: Dict[str, Any], usable: bool) -> bool:
    st.caption(
        "Retrieval only: the question is embedded and matched against the index, and no answer model is called. "
        "Use it to see which passages a question gets and how a threshold splits them."
    )
    render_examples("fia_search_question", not usable)
    with st.form("fia_search_form"):
        question_input("fia_search_question", not usable)
        left, right = st.columns(2)
        with left:
            top_k = top_k_input("fia_search_top_k", settings, not usable)
        with right:
            min_score = st.slider(  # its value comes from follow_api_defaults
                "Similarity threshold (min_score)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key="fia_search_min_score",
                disabled=not usable,
                help=f"For this search only: Ask always uses the API default, {score_text(settings.get('min_score'))} "
                "(FIA_RAG_MIN_SCORE). Passages below the threshold are listed, but they are not evidence.",
            )
        submitted = st.form_submit_button(
            "Search", type="primary", icon=":material/search:", key="fia_search_submit", disabled=not usable
        )
    requested = submitted and submit(
        "fia_search_question",
        SEARCH_OUTCOME_KEY,
        lambda question: get_client().fia_retrieve(question, top_k, min_score),
        "Retrieving passages...",
    )
    render_outcome(SEARCH_OUTCOME_KEY, "The search", lambda body: render_retrieval(body, settings.get("min_score")))
    return requested


def fetch_status() -> Union[Dict[str, Any], ApiUnavailable, ApiError]:
    try:
        return get_client().fia_status()
    except (ApiUnavailable, ApiError) as exc:
        return exc


st.title("FIA regulations")
st.caption(
    "Questions answered only from the official FIA Formula 1 Regulations PDFs indexed by the API. Answers cite "
    "the retrieved passages ([S1], [S2], ...), and their citations, rule numbers and numbers are checked against "
    "them; otherwise the question is declined. The checks do not prove that each sentence says what its passage "
    "says: read the cited passage in the official PDF before relying on an answer."
)
status_slot = st.container()
shown_state = sidebar_state()  # before any request of this run changes the index state
status = fetch_status()
fia = status if isinstance(status, dict) else None
usable = fia is not None and not is_blocked(fia)
settings = (fia or {}).get("settings") or {}
follow_api_defaults(settings)

ask_tab, search_tab = st.tabs(["Ask", "Search passages"])
with ask_tab:
    asked = render_ask_tab(settings, usable)
with search_tab:
    searched = render_search_tab(settings, usable)

if asked or searched:  # a request can change the index state (e.g. a provider failure): show the current one
    status = fetch_status()
with status_slot:
    if isinstance(status, dict):
        render_index_status(status, get_client().base_url)
    else:
        show_api_error(status, "The regulation index status check")
if isinstance(status, dict):
    refresh_health_once(_RESYNCED_KEY, shown_state != status.get("state"))
