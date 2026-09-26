"""Shared Streamlit building blocks: the page registry, text and error display, form submission, radio
clips, regulation passages and declines, the regulation index status, and the status sidebar.

Everything that more than one page shows lives here: a page script cannot import another page (importing
it would run it).
"""

from __future__ import annotations

import copy
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

import streamlit as st

from ui.api_client import ApiError, ApiUnavailable, RequestNotSent, get_client, is_loopback

# Page scripts live in ui/views/: a directory named "pages" next to the entry point would switch
# Streamlit into its legacy multipage mode, where page URLs bypass ui/streamlit_app.py.
VIEWS_DIR = Path(__file__).resolve().parent / "views"
HEALTH_TTL_S = 30.0
# A failed check is kept briefly too, so an unreachable API does not delay every rerun by the connect timeout.
FAILED_HEALTH_TTL_S = 10.0
_HEALTH_KEY = "_f1_health_snapshot"
_SCHEMA_KEY = "_f1_openapi"


class PageSpec(NamedTuple):
    title: str
    icon: str
    summary: str


PAGES: Dict[str, PageSpec] = {
    "overview": PageSpec("Overview", ":material/dashboard:", "What the project does and whether each component is ready."),
    "regulations": PageSpec(
        "FIA regulations", ":material/gavel:", "Questions answered only from the official FIA regulation PDFs, with citations."
    ),
    "strategy": PageSpec("Race strategy", ":material/timeline:", "Ranked pit-stop plans for the rest of a race (heuristic)."),
    "setup": PageSpec("Car setup", ":material/tune:", "Setup search over a documented heuristic objective."),
    "ghost": PageSpec("Ghost car", ":material/compare_arrows:", "Distance-aligned comparison of two laps of telemetry."),
    "radio": PageSpec("Driver radio", ":material/graphic_eq:", "Coarse acoustic emotion label for a radio clip (heuristic)."),
    "assistant": PageSpec(
        "Ask the copilot", ":material/chat:", "Natural-language questions routed to the right module by keywords (heuristic)."
    ),
    "triage": PageSpec("Incident triage", ":material/flag:", "Preliminary severity category (not a steward-decision predictor)."),
}
NAV_SECTIONS: Dict[str, List[str]] = {
    "Overview": ["overview"],
    "Regulations": ["regulations"],
    "Race engineering": ["strategy", "setup", "ghost", "radio"],
    "Assistant": ["assistant", "triage"],
}


class ModuleSpec(NamedTuple):
    label: str
    page: str
    method: str
    heuristic: bool


# Keys of /health "modules"; unknown keys (newer API versions) are shown under their own name.
MODULES: Dict[str, ModuleSpec] = {
    "fia_rag": ModuleSpec(
        "FIA regulation QA", "regulations", "Retrieval over the official PDFs, citation-checked answers", False
    ),
    "strategy": ModuleSpec("Strategy engine", "strategy", "Hand-set lap-time model, exhaustive plan search", True),
    "setup": ModuleSpec("Setup recommender", "setup", "Optuna search over a hand-set objective", True),
    "ghost": ModuleSpec("Ghost comparison", "ghost", "Distance-aligned time delta; zone detection is heuristic", True),
    "emotion": ModuleSpec("Driver-radio emotion", "radio", "Acoustic thresholds, not a validated emotion model", True),
    "emotion_transcription": ModuleSpec("Radio transcription", "radio", "Optional local Whisper", False),
    "natural_query": ModuleSpec("Natural-language router", "assistant", "Whole-word keyword routing to the modules", True),
    "penalty_triage": ModuleSpec("Incident triage", "triage", "Transparent severity heuristic", True),
}


# Why the regulation QA answered 503, by the API's error_type (the RAG exception class).
RAG_UNAVAILABLE_CAUSES: Dict[str, str] = {
    "IndexNotReadyError": "The regulation index is missing, incomplete or out of date.",
    "ProviderError": (
        "The model provider call failed (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, model names, quota or rate limit); "
        "the index itself is fine. Every retry calls the provider again."
    ),
    "RAGConfigurationError": (
        "A required setting (such as `OPENAI_API_KEY`) or the regulation PDFs are missing or invalid; the message "
        "above names which."
    ),
    "VectorStoreError": (
        "The Qdrant index storage or server is unavailable (embedded storage may be held by another process, "
        "such as an index build)."
    ),
    "DocumentError": "A regulation PDF is missing, unreadable or changed since the index was built.",
}


# /api/fia/status "provider_status" records the outcome of the latest regulation request, not a live check:
# a search answered from cached embeddings makes no provider call but still counts as a success.
PROVIDER_STATUS_HELP = (
    "From the API's latest regulation request, not a live check: ok when it succeeded (a search answered from "
    "cached embeddings counts too, although it makes no provider call), failing when a model-provider call "
    "failed, unknown until a request has succeeded"
)
# The API's limit on decoded radio audio (MAX_AUDIO_BYTES in core_modules/driver_emotion), checked before
# sending, and the clip formats it decodes (M4A and WebM need ffmpeg where the API runs).
MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AUDIO_TEXT = f"{MAX_AUDIO_BYTES // 2**20} MiB"
AUDIO_TYPES = ["wav", "flac", "ogg", "mp3", "m4a", "webm"]
UPLOAD, RECORD = "Upload a file", "Record"  # where a radio clip comes from
CLIP_SOURCES = [UPLOAD, RECORD]
DEFAULT_DOCS_PATH = "data/fia_docs"  # FIA_DOCS_PATH default of the API and of scripts/fetch_fia_regulations.py
# A lead over the runner-up profile below this is too small for the audio to separate the two.
CLOSE_PROFILE_LEAD = 0.05
EXAMPLE_INPUTS_BADGE = "Example inputs · not your data"
KEEP = "session"  # form values survive page switches (persist_state)

REPHRASE = "Rephrasing the question can help."
# decline_reason of the regulation QA -> (headline, what happened, what the user can do), on the FIA
# regulations and Ask the copilot pages.
DECLINE_REASONS: Dict[str, Tuple[str, str, str]] = {
    "no_evidence_above_threshold": (
        "No passage was similar enough to the question",
        "No regulation passage reached the similarity threshold, so the answer model was not called. The "
        "question may be about something the regulations do not cover (race results, tickets, ...).",
        "Use the regulations' own terms, or run **Search passages** on the FIA regulations page to see the closest "
        "passages and their scores.",
    ),
    "model_declined": (
        "The answer model found no answer in the evidence",
        "The model read the retrieved passages and replied that they do not contain enough evidence.",
        "Check the evidence below: if it does not cover the question, the regulations probably do not either. "
        "Rephrasing, or a larger top_k on the FIA regulations page, can bring in other passages.",
    ),
    "empty_model_output": (
        "The answer model returned nothing",
        "The model's reply was empty, so there was nothing to check.",
        "Ask again; if it repeats, check the answer model and provider settings of the API.",
    ),
    "invalid_citation": (
        "The answer cited a source that was not supplied",
        "The model cited a source label that is not one of the retrieved passages, so its statements cannot "
        "be traced to the regulations.",
        REPHRASE,
    ),
    "missing_citation": (
        "The answer cited no source",
        "Every statement must cite the passages it relies on; this answer cited none of them.",
        REPHRASE,
    ),
    "unsupported_rule_reference": (
        "The answer named a rule that is not in its sources",
        "The answer mentions an article or rule number that does not occur in the passages it cites, so it "
        "may be invented or misattributed.",
        REPHRASE,
    ),
    "uncited_claim": (
        "Part of the answer had no citation",
        "The answer made at least one statement without citing a passage for it.",
        REPHRASE,
    ),
    "unsupported_number": (
        "The answer stated a number that is not in its sources",
        "The answer contains a number (a limit, penalty, amount or time) that does not occur in the passages "
        "it cites. A number repeated only from the question counts too.",
        REPHRASE,
    ),
    "truncated_model_output": (
        "The answer was cut off",
        "The model's reply stopped before it finished, at its output limit or by the provider's content filter; "
        "incomplete answers are never shown.",
        "Ask a narrower question. If the output limit stopped it (the FIA regulations page says which one did), "
        "raise `FIA_RAG_MAX_OUTPUT_TOKENS` for the API.",
    ),
    "unverified_claim": (
        "The claim check failed",
        "The claim verifier (`FIA_RAG_VERIFY_CLAIMS`) found sentences the cited passages do not support, or "
        "its reply could not be read.",
        REPHRASE,
    ),
}
# The official PDF link of a passage (source_url, from the user-editable manifest): a plain http(s) URL that
# cannot break out of a Markdown link.
SAFE_URL = re.compile(r"https?://[^\s()<>\[\]\"'`]+")
DEFINITIONS_ONLY_NOTE = (
    "Only official definitions were cited. Definitions are not retrieved by similarity, so there is no evidence "
    "strength to report."
)

# Where docker-compose.yml publishes the API service ("api") on the computer running it.
COMPOSE_PUBLISHED_API_URL = "http://127.0.0.1:8000"

STOP_EMBEDDED_API_STEP = (
    "Stop the API first (Ctrl+C in its terminal; with `scripts/run_app.py` this UI stops too): embedded Qdrant "
    "storage (the default) can be opened by only one process. To build while the API runs, set `QDRANT_URL` "
    "to a Qdrant server."
)


def app_page(name: str) -> Any:
    """The ``st.Page`` for a UI page; the same object shape is used by the navigation and by links."""

    spec = PAGES[name]
    return st.Page(VIEWS_DIR / f"{name}.py", title=spec.title, icon=spec.icon, url_path=name, default=name == "overview")


def humanise(state: Any) -> str:
    return str(state).replace("_", " ")


def state_badge(state: Optional[str]) -> str:
    """Markdown badge for a module or service state."""

    if state is None:
        return ":gray-badge[unknown]"
    color = {"ready": "green", "healthy": "green", "degraded": "orange", "unavailable": "gray"}.get(state, "red")
    return f":{color}-badge[{md_text(humanise(state))}]"


def md_text(text: Any, keep_emphasis: bool = False) -> str:
    """API text for Markdown: shown literally (no emphasis, links, HTML or directives), `code spans` kept.

    With ``keep_emphasis`` (a model answer), bold and italics still show.
    """

    special = "\\[]<>#:$~|`" if keep_emphasis else "\\*_[]<>#:$~|`"
    parts = str(text).split("`")
    if len(parts) % 2 == 0:  # unbalanced backticks: escape everything
        parts = ["`".join(parts)]
    for index in range(0, len(parts), 2):
        for char in special:
            parts[index] = parts[index].replace(char, "\\" + char)
    return "`".join(parts)


def code_span(value: Any) -> str:
    """A value (file, column or rule name) as a Markdown code span."""

    return "`" + str(value).replace("`", "'") + "`"


def not_modelled_text(items: Sequence[Any]) -> str:
    """Inputs a response lists as accepted but not modelled, as one Markdown line ("; " since an item can hold commas)."""

    return "Accepted but not modelled: " + "; ".join(md_text(item) for item in items)


def score_text(score: Any) -> str:
    """A similarity score with three decimals; "–" when there is none."""

    return f"{score:.3f}" if isinstance(score, (int, float)) else "–"


def round4(value: Optional[float]) -> Optional[float]:
    """A form number rounded to 4 decimals (slider and step arithmetic leaves float noise); None stays None."""

    return None if value is None else round(float(value), 4)


def clock_time(epoch: float) -> str:
    """A ``time.time()`` value as this computer's clock time."""

    return time.strftime("%H:%M:%S", time.localtime(epoch))


def local_time(timestamp: Any) -> str:
    """An API timestamp (ISO 8601) in this computer's local time; the date only when it is not today."""

    try:
        moment = datetime.fromisoformat(str(timestamp)).astimezone()
    except ValueError:
        return md_text(timestamp)
    today = datetime.now().astimezone().date()
    return moment.strftime("%H:%M:%S" if moment.date() == today else "%Y-%m-%d %H:%M:%S")


def heuristic_badge(label: str = "Heuristic · not validated") -> None:
    """Label for results of the explicitly heuristic modules."""

    st.badge(label, icon=":material/science:", color="orange")


def json_expander(data: Any, label: str = "Raw API response") -> None:
    with st.expander(label, icon=":material/data_object:"):
        st.json(data, expanded=2)


def page_links(names: Sequence[str]) -> None:
    for name in names:
        st.page_link(app_page(name), help=PAGES[name].summary)


class FieldLabels:
    """A form's words for the API field paths and names in validation errors (HTTP 422).

    A list index becomes ``row N`` for lists edited as tables, a named item for lists given in
    ``index_names`` (e.g. a window's start and end), and ``entry N`` for any other list.
    """

    _IDENTIFIER = re.compile(r"\b[a-z][a-z0-9]*(?:[_.][a-z0-9]+)+\b")

    def __init__(
        self,
        labels: Mapping[str, str],
        table_lists: Sequence[str] = (),
        index_names: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.labels = labels
        self.table_lists = tuple(table_lists)
        self.index_names = dict(index_names or {})

    def label(self, path: str) -> str:
        """An API field path (``race_state.current_lap``, ``competition[0].tire_age``) in the form's words."""

        words: List[str] = []
        parent = ""
        for part in re.findall(r"[^.\[\]]+", path):
            names = self.index_names.get(parent, ())
            if not part.isdigit():
                words.append(self.labels.get(part, part))
            elif int(part) < len(names):
                words[-1] = names[int(part)]
            else:
                words.append(f"{'row' if parent in self.table_lists else 'entry'} {int(part) + 1}")
            parent = part
        return " › ".join(words)

    def message(self, text: str) -> str:
        """API text with the form's labels for the field names it mentions (plain text, not Markdown)."""

        def label(match: "re.Match[str]") -> str:
            token = match.group(0)
            return self.label(token) if all(part in self.labels for part in token.split(".")) else token

        return self._IDENTIFIER.sub(label, text)

    def errors(self, errors: Sequence[str]) -> List[str]:
        """The API's validation lines (``path: message``) as Markdown, with the form's labels for field names."""

        lines = []
        for error in errors:
            path, separator, message = error.partition(": ")
            if not separator:
                lines.append(md_text(error))
                continue
            lead = self._IDENTIFIER.match(message)
            if path == "request body":
                lines.append(md_text(self.message(message)))
            elif lead and f"{lead.group(0)}.".startswith(f"{path}."):  # the message names the field itself
                _, field, rest = message.partition(lead.group(0))
                lines.append(f"**{md_text(self.message(field))}**{md_text(self.message(rest))}")
            else:
                lines.append(f"**{md_text(self.label(path))}**: {md_text(self.message(message))}")
        return lines


def show_input_problems(problems: Sequence[str]) -> None:
    """Problems a page found in its inputs before sending anything (plain text, `code spans` kept)."""

    st.error("Nothing was sent to the API. Please fix:", icon=":material/edit_note:")
    st.markdown("\n".join(f"- {md_text(problem)}" for problem in problems))


def show_request_error(exc: Union[ApiUnavailable, ApiError], action: str, fields: FieldLabels) -> None:
    """An API failure of a form; validation errors name the form fields they refer to."""

    if isinstance(exc, ApiError) and exc.status_code == 422:
        lines = fields.errors(exc.errors) if exc.errors else [md_text(fields.message(exc.detail))]
        st.error(f"{action} was rejected by the API (HTTP 422). Correct these inputs and submit again:", icon=":material/rule:")
        st.markdown("\n".join(f"- {line}" for line in lines))
    else:
        show_api_error(exc, action)


def show_api_error(exc: Union[ApiUnavailable, ApiError], action: str = "The request") -> None:
    """Explain an API failure with next steps; nothing is substituted for the missing result."""

    if isinstance(exc, ApiUnavailable):
        st.error(f"{action} could not reach the API at `{exc.base_url}`: {md_text(exc.reason)}.", icon=":material/cloud_off:")
        st.info(exc.hint, icon=":material/lightbulb:")
        return
    detail = md_text(exc.detail)
    if isinstance(exc, RequestNotSent):
        st.error(f"{action} was not sent: {detail}.", icon=":material/data_alert:")
        st.caption("Correct that value and submit again.")
        return
    if exc.status_code == 422:
        st.error(f"{action} was rejected by the API (HTTP 422):", icon=":material/rule:")
        st.markdown("\n".join(f"- {md_text(line)}" for line in exc.errors) if exc.errors else detail)
        st.caption("Correct the input and submit again.")
    elif exc.status_code == 503:
        clear_health_cache()  # a component changed state: the next run shows fresh readiness
        st.warning(f"Service unavailable (HTTP 503): {detail}", icon=":material/construction:")
        if exc.detail.startswith("FIA RAG is unavailable"):
            cause = RAG_UNAVAILABLE_CAUSES.get(exc.error_type or "", "The regulation QA is not ready.")
            st.caption(f"{cause} The Overview page shows its state and the exact fix.")
            page_links(["overview"])
        elif exc.detail.startswith("Transcription failed"):
            st.caption("Whisper transcription is optional: submit again without it, or check the API log.")
    elif exc.status_code == 413:
        st.error(f"The request is too large for the API (HTTP 413): {detail}", icon=":material/data_alert:")
    elif exc.status_code >= 500:
        st.error(f"The API failed with HTTP {exc.status_code}: {detail}", icon=":material/error:")
        st.caption("This is a server-side error; the API log has the details.")
    else:
        st.error(f"{action} failed with HTTP {exc.status_code}: {detail}", icon=":material/error:")


# ------------------------------------------------------------------ form pages


def seed_form(defaults: Mapping[str, Any]) -> None:
    """Start every form value at its pre-filled example the first time a page runs in a session."""

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def submit_form(
    request: Dict[str, Any],
    send: Callable[[Dict[str, Any]], Dict[str, Any]],
    example: bool,
    problems: Sequence[str] = (),
    **extra: Any,
) -> Dict[str, Any]:
    """Send a form's request unless reading the form found ``problems``; the outcome keeps the request, the
    response or API error, when it was sent and whether it came from the unchanged ``example``."""

    outcome: Dict[str, Any] = {
        "request": request,
        "problems": list(problems),
        "response": None,
        "error": None,
        "at": time.time(),
        "example": example,
        **extra,
    }
    if not problems:
        try:
            outcome["response"] = send(request)
        except (ApiUnavailable, ApiError) as exc:
            outcome["error"] = exc
    return outcome


def render_form_outcome(
    outcome: Mapping[str, Any],
    action: str,
    fields: FieldLabels,
    render_result: Callable[[Mapping[str, Any]], None],
    after_error: Optional[Callable[[Union[ApiUnavailable, ApiError]], None]] = None,
) -> None:
    """A form's result, or the API's error in the form's words; then the request sent and the raw response."""

    error = outcome.get("error")
    if error is not None:
        show_request_error(error, action, fields)
        if after_error is not None:
            after_error(error)
    else:
        render_result(outcome)
    json_expander(outcome["request"], "Request sent to the API")
    if outcome.get("response") is not None:
        json_expander(outcome["response"])


# ------------------------------------------------------------------ radio clips


def clip_input(source: str, key_prefix: str) -> Any:
    """A radio clip from the file uploader (``<key_prefix>_file``) or the browser recorder (``<key_prefix>_recording``)."""

    if source == UPLOAD:
        return st.file_uploader("Radio clip", type=AUDIO_TYPES, key=f"{key_prefix}_file")
    return st.audio_input(
        "Record a clip", key=f"{key_prefix}_recording", help="Browsers allow the microphone only on localhost or HTTPS pages."
    )


def oversize_problem(audio: bytes) -> Optional[str]:
    """Why a clip is not sent when it is larger than the API accepts; None when it fits."""

    if len(audio) <= MAX_AUDIO_BYTES:
        return None
    return (
        f"The clip is {len(audio) / 2**20:.1f} MiB; the API accepts at most {MAX_AUDIO_TEXT}. Trim it, or save it in "
        "a compressed format (FLAC, OGG or MP3)."
    )


def transcription_state() -> Dict[str, Any]:
    """Whisper on the API: ``{"available": True/False/None, "reason": str}`` from /health (None: the check failed)."""

    try:
        health = cached_health()
    except (ApiUnavailable, ApiError) as exc:
        reason = exc.reason if isinstance(exc, ApiUnavailable) else exc.detail
        return {"available": None, "reason": f"the API status check failed ({reason})"}
    state = (health.get("modules") or {}).get("emotion_transcription")
    return {
        "available": state == "ready",
        "reason": (health.get("details") or {}).get("emotion_transcription") or "no reason given",
    }


def tied_profiles(scores: Mapping[str, Any]) -> List[str]:
    """The emotion profiles sharing the best acoustic similarity when more than one does, in the API's order."""

    if not scores:
        return []
    best = max(float(value) for value in scores.values())
    tied = [str(name) for name, value in scores.items() if float(value) == best]
    return tied if len(tied) > 1 else []


def profile_tie_note(scores: Mapping[str, Any], acoustic_label: Any) -> Optional[str]:
    """Markdown warning when profiles tie for the best similarity (the API then reports neutral), or when the
    acoustic label leads the runner-up by very little: either way the audio barely separates them."""

    tied = tied_profiles(scores)
    label = str(acoustic_label)
    if tied:
        names = f"{', '.join(tied[:-1])} and {tied[-1]}"
        return (
            f"**Tied profiles:** {md_text(names)} {'both' if len(tied) == 2 else 'all'} score "
            f"{float(scores[tied[0]]):.3f}, so the audio does not separate them: the acoustic label is **neutral** "
            "with confidence 0."
        )
    others = [(float(value), str(name)) for name, value in scores.items() if str(name) != label]
    if label not in scores or not others:  # e.g. a neutral label (below the threshold): no profile won
        return None
    best = float(scores[label])
    second, runner_up = max(others, key=lambda item: item[0])
    if not 0 <= best - second < CLOSE_PROFILE_LEAD:
        return None
    return (
        f"**Close profiles:** {md_text(label)} {best:.3f} leads {md_text(runner_up)} {second:.3f} by only "
        f"{best - second:.3f} (under {CLOSE_PROFILE_LEAD}), so the audio barely separates them: a slightly different "
        "recording could swap the label. The small lead also keeps the acoustic confidence low."
    )


# ------------------------------------------------------------------ regulation passages and declines


def decline_explanation(reason: Any) -> Tuple[str, str, str]:
    """(headline, what happened, what the user can do) for a regulation QA decline_reason."""

    return DECLINE_REASONS.get(
        reason,
        ("The API declined to answer", "This UI does not know the reason code; the raw API response has the details.", ""),
    )


def is_definition(passage: Mapping[str, Any]) -> bool:
    return passage.get("kind") == "definition"


def cites_regulation_passage(passages: Sequence[Mapping[str, Any]]) -> bool:
    """Whether an answer cites a regulation passage: its evidence strength counts only those (definitions have no similarity)."""

    return any(passage.get("cited") and not is_definition(passage) for passage in passages)


def pdf_link(passage: Mapping[str, Any]) -> Optional[str]:
    """The official PDF URL, opened at the passage's PDF page; None unless it is a plain http(s) URL."""

    url = passage.get("source_url")
    if not isinstance(url, str) or not SAFE_URL.fullmatch(url):
        return None
    page = passage.get("page")
    return f"{url}#page={page}" if isinstance(page, int) else url


def location(passage: Mapping[str, Any]) -> str:
    parts = []
    if passage.get("section"):
        parts.append(f"Section {passage['section']}")
    if passage.get("page_label"):
        parts.append(f"printed page {passage['page_label']}")
    if passage.get("page") is not None:
        parts.append(f"PDF page {passage['page']}")
    return " · ".join(parts) or "location unknown"


def cited_note(grounded: bool) -> str:
    return "cited" if grounded else "cited by the rejected output"


def evidence_label(passage: Mapping[str, Any], grounded: bool) -> str:
    """A passage's one-line title (Markdown): label, citation, rule or defined term, location and similarity."""

    parts = [str(passage.get("label", "?")), cited_note(grounded) if passage.get("cited") else ""]
    if is_definition(passage):
        parts += [f"definition: {passage.get('defined_term') or 'unknown'}", location(passage)]
    else:
        parts += [passage.get("nearest_rule") or "", location(passage), f"similarity {score_text(passage.get('score'))}"]
    return md_text(" · ".join(part for part in parts if part))


def render_passage(passage: Mapping[str, Any], grounded: bool = True) -> None:
    """Badges, location, verbatim text and the official PDF link of one passage given to the answer model.

    A citation of an answer that failed validation (``grounded`` False) is marked as such, not as evidence.
    """

    badges = []
    if passage.get("label"):
        badges.append(f":blue-badge[{md_text(passage['label'])}]")
    if passage.get("cited"):
        badges.append(f":{'green' if grounded else 'orange'}-badge[{cited_note(grounded)}]")
    if is_definition(passage):
        badges.append(":violet-badge[definition]")
        details = [f"defined term **{md_text(passage.get('defined_term') or 'unknown')}**", md_text(location(passage))]
    else:
        badges.append(":gray-badge[regulation passage]")
        details = [md_text(location(passage)), f"similarity {score_text(passage.get('score'))}"]
        if passage.get("nearest_rule"):
            details.insert(1, f"nearest rule {code_span(passage['nearest_rule'])}")
    st.markdown(" ".join(badges) + " " + " · ".join(details))
    if is_definition(passage):
        st.markdown(
            "An official definition of a term the passages use or the question names. It is added to the "
            "evidence, not retrieved by similarity, so its score is not a similarity."
        )
    st.markdown("> " + md_text(passage.get("text", "")).replace("\n", "\n> "))
    link = pdf_link(passage)
    source = md_text(passage.get("source") or "unknown file")
    st.caption(f"Source: [{source}, official FIA PDF]({link})" if link else f"Source: {source} (no official URL recorded)")


# ------------------------------------------------------------------ regulation index status


def uses_compose(base_url: str, settings: Dict[str, Any]) -> bool:
    """Whether the API runs in this project's Docker Compose stack (service names ``api`` and ``qdrant``)."""

    location = str(settings.get("qdrant_location") or "")
    return urlsplit(base_url).hostname == "api" or urlsplit(location).hostname == "qdrant"


def rag_fix_steps(fia: Dict[str, Any], compose: bool = False) -> List[str]:
    """Next steps for the RAG state reported by /api/fia/status, as host or Docker Compose commands."""

    state = fia.get("state")
    settings = fia.get("settings") or {}
    index = fia.get("index") or {}
    embedded = settings.get("qdrant_mode") == "local" and not compose

    def script(command: str) -> str:
        return f"`docker compose run --rm api python scripts/{command}`" if compose else f"`python scripts/{command}`"

    def build(lead: str) -> List[str]:
        step = (
            f"{lead}: {script('build_fia_index.py --dry-run')} (chunk and request estimate, no API calls), "
            f"then {script('build_fia_index.py')}."
        )
        return [STOP_EMBEDDED_API_STEP, step, "Start the API again."] if embedded else [step]

    if compose:
        apply_env = "apply `.env` changes with `docker compose up -d api` (`docker compose restart` keeps the old settings)"
    else:
        apply_env = "restart the API so it reads the new settings"
    steps: List[str] = []
    if state == "not_configured":
        if not fia.get("documents"):
            docs = str(settings.get("docs_path") or DEFAULT_DOCS_PATH).replace("`", "'")
            fetch = f"Download the official PDFs: {script('fetch_fia_regulations.py')}"
            if compose or docs == DEFAULT_DOCS_PATH:  # in Compose the script runs with the API's own settings
                steps.append(f"{fetch} (saved to `{docs}`).")
            else:
                steps.append(
                    f"{fetch}. It saves to `FIA_DOCS_PATH` (from the environment or `.env`, default "
                    f"`{DEFAULT_DOCS_PATH}`) and this API reads `{docs}`, so run it with the API's `FIA_DOCS_PATH`."
                )
        key_missing = not settings.get("api_key_configured")
        if key_missing:
            steps.append(
                "Set `OPENAI_API_KEY` in `.env` (start from `.env.example`; OpenAI-compatible providers such as "
                "OpenRouter also need `OPENAI_BASE_URL`, `FIA_RAG_EMBEDDING_MODEL` and `FIA_RAG_MODEL`)."
            )
        rebuild = index.get("status") != "current"
        if rebuild:
            steps.extend(build("Build the index"))
        if key_missing and not (rebuild and embedded):  # otherwise "Start the API again" covers it
            steps.append(f"{apply_env[0].upper()}{apply_env[1:]}.")
    elif state == "misconfigured":
        steps.append(f"Correct the setting named above in `.env` or the environment, then {apply_env}.")
    elif state in ("index_missing", "index_empty", "index_incomplete", "index_stale"):
        if state == "index_stale":
            steps.append("The documents or the chunking/embedding settings changed since the index was built.")
        steps.extend(build("Build the index"))
    elif state == "provider_failing":
        steps.append(
            f"Check `OPENAI_API_KEY`, `OPENAI_BASE_URL`, the model names and your provider quota; after changing "
            f"`.env`, {apply_env}. The state clears after the next successful regulation request."
        )
    elif state != "ready" and (index.get("glossary") or {}).get("status") == "missing":
        steps.extend(build("Rebuild the index to add the definitions glossary"))
    return steps


def render_rag_problems(fia: Dict[str, Any], compose: bool) -> None:
    """The API's problem messages for the regulation QA, then the numbered steps that fix its state."""

    for problem in fia.get("problems") or []:
        st.warning(md_text(problem), icon=":material/report:")
    steps = rag_fix_steps(fia, compose)
    if steps:
        st.markdown("**Next steps** (from the project folder):\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)))


def index_metrics(index: Mapping[str, Any]) -> None:
    """The regulation index's passage and definition counts (for a row of metrics)."""

    points, expected = index.get("points"), index.get("expected_points")
    glossary = index.get("glossary") or {}
    entries = glossary.get("entries")
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


def provider_metric(fia: Mapping[str, Any]) -> None:
    st.metric("Model provider", humanise(fia.get("provider_status", "unknown")), help=PROVIDER_STATUS_HELP, width="content")


def refresh_health_once(key: str, stale: bool) -> None:
    """Rerun once with a fresh /health snapshot when it is ``stale``.

    The sidebar is drawn before the page, from the cached snapshot; when the page's live /api/fia/status shows
    another regulation QA state, both are brought in step. ``key`` (one per page) stops a rerun loop when the
    two keep disagreeing.
    """

    if st.session_state.pop(key, False) or not stale:
        return
    clear_health_cache()
    st.session_state[key] = True
    st.rerun()


def docs_reference(base_url: str, browser_url: Optional[str]) -> str:
    """The API reference URL: a link only when the API and the browser are both on this computer."""

    docs_url = f"{base_url}/docs"
    local_browser = bool(browser_url) and is_loopback(str(browser_url))
    if local_browser and is_loopback(base_url):
        return docs_url
    if local_browser and urlsplit(base_url).hostname == "api":  # the Compose API service, published on this computer
        return f"{COMPOSE_PUBLISHED_API_URL}/docs (the API container of Docker Compose)"
    # e.g. a browser on another device in LAN mode
    return f"`{docs_url}` (the address the UI server uses)"


def cached_health() -> Dict[str, Any]:
    """/health for this browser session, refetched when stale; a failed check is re-raised until it is retried."""

    snapshot = st.session_state.get(_HEALTH_KEY)
    if snapshot is not None:
        max_age = HEALTH_TTL_S if isinstance(snapshot[2], dict) else FAILED_HEALTH_TTL_S
        if time.monotonic() - snapshot[0] > max_age:
            snapshot = None
    if snapshot is None:
        try:
            result: Union[Dict[str, Any], ApiUnavailable, ApiError] = get_client().health()
        except (ApiUnavailable, ApiError) as exc:
            result = exc
        snapshot = (time.monotonic(), time.time(), result)
        st.session_state[_HEALTH_KEY] = snapshot
    if isinstance(snapshot[2], Exception):
        raise snapshot[2]
    return snapshot[2]


def api_schema() -> Dict[str, Any]:
    """The API's OpenAPI document, fetched once per session and API address; raises ApiUnavailable or ApiError."""

    client = get_client()
    cached = st.session_state.get(_SCHEMA_KEY)
    if cached is None or cached[0] != client.base_url:
        cached_health()  # a known-unreachable API is not asked again on every rerun
        cached = (client.base_url, client.get_openapi())
        st.session_state[_SCHEMA_KEY] = cached
    return cached[1]


def offers_endpoint(schema: Mapping[str, Any], path: str, method: str = "post") -> bool:
    """Whether the API's OpenAPI document lists ``method path`` (for optional endpoints of newer API versions)."""

    return method.lower() in ((schema.get("paths") or {}).get(path) or {})


def documented_example(schema: Mapping[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """A copy of the first documented example of the API request model ``model``; None when it documents none."""

    examples = ((schema.get("components") or {}).get("schemas") or {}).get(model, {}).get("examples") or [None]
    return copy.deepcopy(examples[0]) if isinstance(examples[0], dict) else None


def unchanged_example(defaults: Mapping[str, Any]) -> bool:
    """Whether every form value still holds its pre-filled example value."""

    return all(st.session_state.get(key) == value for key, value in defaults.items())


def example_inputs_badge() -> None:
    """Label for a result computed from a form's unchanged example inputs."""

    st.badge(EXAMPLE_INPUTS_BADGE, icon=":material/lab_profile:", color="gray")


def health_checked_at() -> Optional[float]:
    snapshot = st.session_state.get(_HEALTH_KEY)
    return snapshot[1] if snapshot else None


def clear_health_cache() -> None:
    st.session_state.pop(_HEALTH_KEY, None)


def module_rows(modules: Mapping[str, Any]) -> List[str]:
    """One Markdown line per /health module, known modules first."""

    keys = [key for key in MODULES if key in modules] + [key for key in modules if key not in MODULES]
    return [f"{state_badge(modules[key])} {MODULES[key].label if key in MODULES else humanise(key)}" for key in keys]


def render_api_sidebar() -> None:
    """API address and live per-module readiness, with a refresh button."""

    client = get_client()
    with st.sidebar:
        st.caption("API")
        st.code(client.base_url, language=None)
        try:
            health = cached_health()
        except (ApiUnavailable, ApiError) as exc:
            st.markdown(f"{state_badge('unreachable')} API status")
            reason = exc.reason if isinstance(exc, ApiUnavailable) else exc.detail
            st.caption(f"Status check failed: {md_text(reason)}")
        else:
            st.markdown(f"{state_badge(health.get('status'))} API {health.get('version', '')}".strip())
            st.markdown("  \n".join(module_rows(health.get("modules", {}))))
            checked = health_checked_at()
            if checked is not None:
                st.caption(f"Checked at {clock_time(checked)}")
        if st.button("Refresh status", icon=":material/refresh:", key="sidebar_refresh_health"):
            clear_health_cache()
            st.rerun()
