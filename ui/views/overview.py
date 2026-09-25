"""Overview: what F1 AI Copilot is, and the live readiness of every API component (never assumed)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

import streamlit as st

from ui.api_client import ApiError, ApiUnavailable, get_client
from ui.components import (
    MODULES,
    PAGES,
    PROVIDER_STATUS_HELP,
    app_page,
    cached_health,
    clear_health_cache,
    docs_reference,
    health_checked_at,
    humanise,
    json_expander,
    local_time,
    md_text,
    rag_fix_steps,
    show_api_error,
    state_badge,
    uses_compose,
)

_RESYNCED_KEY = "_f1_overview_resynced"


def render_intro() -> None:
    st.title("F1 AI Copilot")
    st.markdown(
        "Formula 1 analysis built around a **question-answering system over the official FIA Formula 1 "
        "Regulations**: the PDFs are parsed, chunked and indexed, and questions are answered only from "
        "retrieved passages, with `[S#]` citations that are checked against the evidence before an answer "
        "is shown. When the evidence is insufficient, the question is declined instead of guessed.\n\n"
        "Around it sit explicitly heuristic modules for race strategy, car setup search, lap telemetry "
        "comparison, driver-radio audio analysis, question routing and incident triage."
    )
    st.caption(
        "A research and demo project: not an FIA steward tool, a validated vehicle-dynamics simulator or a "
        "production race-engineering system. Heuristic results are labelled as such throughout."
    )


def render_status_summary(health: Dict[str, Any], base_url: str) -> None:
    checked = health_checked_at()
    when = f" · checked at {time.strftime('%H:%M:%S', time.localtime(checked))}" if checked else ""
    st.markdown(f"{state_badge(health.get('status'))} API {md_text(health.get('version', ''))} at `{base_url}`{when}")
    if health.get("status") == "degraded":
        st.caption("Degraded: at least one component below is not ready. The other components keep working.")


def render_cards(modules: Optional[Dict[str, Any]]) -> None:
    """One card per page with the live state of the modules behind it (``None``: not checked)."""

    names = [name for name in PAGES if name != "overview"]
    for start in range(0, len(names), 4):
        for column, name in zip(st.columns(4), names[start : start + 4], strict=False):  # last row may be short
            with column.container(border=True, height="stretch"):
                st.page_link(app_page(name), help=PAGES[name].summary)
                for key, spec in MODULES.items():
                    if spec.page != name:
                        continue
                    badge = state_badge(modules.get(key)) if modules is not None else ":gray-badge[not checked]"
                    heuristic = " :orange-badge[heuristic]" if spec.heuristic else ""
                    st.markdown(f"{badge} {spec.label}{heuristic}")
                    st.caption(spec.method)
    if modules:
        unknown = [key for key in modules if key not in MODULES]
        if unknown:
            st.markdown("Other components: " + ", ".join(f"{state_badge(modules[key])} {humanise(key)}" for key in unknown))


def render_component_notes(health: Dict[str, Any]) -> None:
    modules = health.get("modules", {})
    details = health.get("details", {})
    reason = details.get("emotion_transcription")
    if modules.get("emotion_transcription") not in (None, "ready") and reason:
        st.caption(
            f"Radio transcription is optional and unavailable on this API: {md_text(reason)}. Emotion analysis works without it."
        )
    if modules.get("ghost") == "artifacts_not_writable":
        st.warning(
            "The API cannot write ghost-comparison images: make `F1_ARTIFACTS_DIR` (default `outputs`) "
            "writable by the API process.",
            icon=":material/folder_off:",
        )


def render_rag(fia: Dict[str, Any], compose: bool) -> None:
    state = fia.get("state")
    settings = fia.get("settings") or {}
    index = fia.get("index") or {}
    documents = fia.get("documents") or []
    glossary = index.get("glossary") or {}
    st.markdown(f"{state_badge(state)} Regulation QA")
    if state == "ready":
        st.success(
            f"Ready: {index.get('points')} indexed passages from {len(documents)} documents. Answers cite the "
            "retrieved passages and are declined when the evidence is insufficient.",
            icon=":material/verified:",
        )
    if settings:  # absent when the settings themselves are invalid
        points, expected = index.get("points"), index.get("expected_points")
        entries = glossary.get("entries")
        with st.container(horizontal=True, gap="medium"):  # wraps on a phone instead of stacking five rows
            st.metric("Documents", len(documents), width="content")
            st.metric("Index", humanise(index.get("status", "unknown")), width="content")
            st.metric(
                "Indexed passages",
                "–" if points is None else points,
                help=None if expected is None else f"{expected} expected for the current documents and settings",
                width="content",
            )
            st.metric(
                "Definitions",
                humanise(glossary.get("status", "unknown")) if entries is None else entries,
                help="Official definitions of defined terms, added to the evidence when passages use them",
                width="content",
            )
            st.metric(
                "Model provider", humanise(fia.get("provider_status", "unknown")), help=PROVIDER_STATUS_HELP, width="content"
            )
    for problem in fia.get("problems") or []:
        st.warning(md_text(problem), icon=":material/report:")
    if fia.get("provider_status") == "failing" and fia.get("last_error_at"):
        st.caption(f"Last provider error at {local_time(fia['last_error_at'])}.")
    if fia.get("embedding_cache_problem"):
        st.caption(f"Embedding cache: {md_text(fia['embedding_cache_problem'])}")
    steps = rag_fix_steps(fia, compose)
    if steps:
        st.markdown("**Next steps** (from the project folder):\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)))
    if documents:
        with st.expander(f"Documents ({len(documents)})", icon=":material/description:"):
            st.dataframe(
                [
                    {"File": d.get("filename"), "Section": d.get("section"), "Size (MB)": round((d.get("bytes") or 0) / 1e6, 2)}
                    for d in documents
                ],
                hide_index=True,
            )
    if settings:
        with st.expander("Retrieval and model settings", icon=":material/settings:"):
            st.dataframe([{"Setting": key, "Value": str(value)} for key, value in settings.items()], hide_index=True)


render_intro()
header, refresh = st.columns([5, 1], vertical_alignment="bottom")
header.subheader("System status")
if refresh.button("Refresh", icon=":material/refresh:", key="overview_refresh", width="stretch"):
    clear_health_cache()
    st.rerun()

client = get_client()
health: Optional[Dict[str, Any]] = None
fia: Optional[Dict[str, Any]] = None
failure: Optional[Union[ApiUnavailable, ApiError]] = None
fia_failure: Optional[Union[ApiUnavailable, ApiError]] = None
try:
    health = cached_health()
except (ApiUnavailable, ApiError) as exc:
    failure = exc
else:
    try:
        fia = client.fia_status()
    except (ApiUnavailable, ApiError) as exc:
        fia_failure = exc

resynced = st.session_state.pop(_RESYNCED_KEY, False)
if health and fia and (health.get("modules") or {}).get("fia_rag") != fia.get("state") and not resynced:
    # The cached /health snapshot (sidebar, cards) predates the live index state: refresh both once.
    clear_health_cache()
    st.session_state[_RESYNCED_KEY] = True
    st.rerun()

if failure is not None:
    show_api_error(failure, "The status check")
elif health is not None:
    render_status_summary(health, client.base_url)

render_cards(health.get("modules", {}) if health else None)

st.subheader("FIA regulation index")
if health is None:
    st.caption("Shown when the API is reachable.")
else:
    if fia_failure is not None:
        show_api_error(fia_failure, "The FIA status check")
    elif fia is not None:
        render_rag(fia, uses_compose(client.base_url, fia.get("settings") or {}))
    render_component_notes(health)
    json_expander({"health": health, "fia_status": fia}, "Raw status responses")
    st.caption(f"Interactive API reference: {docs_reference(client.base_url, st.context.url)}")
