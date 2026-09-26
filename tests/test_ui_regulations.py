"""Web UI: the FIA regulations page (ui/views/regulations.py) against the real API in-process.

Only the embedding and chat services are replaced (tests.helpers stand-ins); PDF parsing,
indexing, retrieval, answer validation and the HTTP layer are real.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
from streamlit.testing.v1 import AppTest

import core_modules.rule_checker.fia_rag.pipeline as rag_pipeline
import ui.api_client as api_client
from app.main import MAX_QUESTION_CHARS, app
from core_modules.rule_checker.fia_rag import FIARegulationRAG, QdrantConfig, RAGSettings, VectorStoreError
from core_modules.rule_checker.fia_rag.config import ChunkingConfig, GenerationConfig, RetrievalConfig
from core_modules.rule_checker.fia_rag.generation import VERIFIER_PROMPT
from core_modules.rule_checker.fia_rag.retrieval import RetrievedPassage
from tests.helpers import DEFINITIONS_QUESTION, ScriptedChatModel, make_definitions_rag, write_definitions_corpus
from tests.ui_support import (  # noqa: F401 (pytest fixtures)
    ENTRY_POINT,
    REPO_ROOT,
    assert_no_exception,
    expander_labels,
    expanders,
    install_client,
    markdown_text,
    page_text,
    run_page,
    sidebar_text,
    ui_api,
    ui_api_down,
)
from ui.api_client import ApiClient
from ui.components import PROVIDER_STATUS_HELP

PIT_QUESTION = "What is the speed limit in the pit lane?"
UNANSWERABLE = "Who won the 2021 Abu Dhabi Grand Prix?"  # verbatim from the evaluation set
FABRICATED = "The limit is 80km/h according to Article B9.9 [S9]."
OFFICIAL_URL = "https://www.fia.com/example-section-b.pdf"
GOOD_ANSWER = "A speed limit of 80km/h is imposed in the pit lane [S1]."


def verifier_rejects_the_first_sentence(messages) -> str:
    return '{"unsupported": [1]}' if messages[0].content == VERIFIER_PROMPT else GOOD_ANSWER


# decline_reason -> (scripted model reply, headline the page shows); the real validator decides the reason
DECLINE_CASES = {
    "model_declined": ("INSUFFICIENT_EVIDENCE", "The answer model found no answer in the evidence"),
    "empty_model_output": ("", "The answer model returned nothing"),
    "missing_citation": ("A speed limit of 80km/h is imposed in the pit lane.", "The answer cited no source"),
    "unsupported_rule_reference": (
        "Article B9.9 imposes a speed limit of 80km/h in the pit lane [S1].",
        "The answer named a rule that is not in its sources",
    ),
    "uncited_claim": (GOOD_ANSWER + "\n\nTeams may refuel their cars during the race.", "Part of the answer had no citation"),
    "unsupported_number": (
        "A speed limit of 90km/h is imposed in the pit lane [S1].",
        "The answer stated a number that is not in its sources",
    ),
    "truncated_model_output": (GOOD_ANSWER, "The answer was cut off"),  # with finish_reason "length"
    "unverified_claim": (verifier_rejects_the_first_sentence, "The claim check failed"),
}


@pytest.fixture
def built_rag(installed_rag):
    rag, llm = installed_rag
    rag.build_index()
    return rag, llm


def ask(at: AppTest, question: str, top_k: int | None = None) -> AppTest:
    at.text_area(key="fia_ask_question").input(question)
    if top_k is not None:
        at.number_input(key="fia_ask_top_k").set_value(top_k)
    at.button(key="fia_ask_submit").click()
    return at.run()


def search(at: AppTest, question: str, top_k: int, min_score: float) -> AppTest:
    at.text_area(key="fia_search_question").input(question)
    at.number_input(key="fia_search_top_k").set_value(top_k)
    at.slider(key="fia_search_min_score").set_value(min_score)
    at.button(key="fia_search_submit").click()
    return at.run()


def expander(block, prefix: str):
    matches = [node for node in expanders(block) if str(node.label).startswith(prefix)]
    assert len(matches) == 1, (prefix, expander_labels(block))
    return matches[0]


def texts(block) -> str:
    return "\n".join(str(node.value) for kind in ("markdown", "caption", "code") for node in getattr(block, kind))


def answer_line(at: AppTest) -> str:
    return texts(at.get_by_key("fia_answer_card")).split("\n")[1]


def retrieval_inputs(at: AppTest):
    return (
        at.number_input(key="fia_ask_top_k").value,
        at.number_input(key="fia_search_top_k").value,
        at.slider(key="fia_search_min_score").value,
    )


def with_retrieval(rag, monkeypatch, **config) -> None:
    """Change the API's retrieval settings (as after an edit of .env and an API restart)."""

    monkeypatch.setattr(rag, "settings", replace(rag.settings, retrieval=RetrievalConfig(**config)))


def forms_locked(at: AppTest) -> bool:
    return at.button(key="fia_ask_submit").disabled and at.button(key="fia_search_submit").disabled


def evidence_frame(at: AppTest):
    return at.dataframe[0].value


def service_unavailable(at: AppTest) -> str:
    """The 503 warning of a request (the index status above it can have warnings of its own)."""

    return next(node.value for node in at.warning if node.value.startswith("Service unavailable (HTTP 503)"))


# ------------------------------------------------------------------ Ask


def test_grounded_answer_links_each_citation_to_its_passage(ui_api, built_rag):
    rag, llm = built_rag
    at = run_page("regulations")
    assert_no_exception(at)
    status = page_text(at)
    assert ":green-badge[ready] **Regulation index**" in status and "Documents: 2" in status
    assert f"Indexed passages: {rag.status()['index']['points']}" in status and "Claim verifier: off" in status
    assert "Embedding model `hashing-test-512` · answer model `gpt-4o-mini` · default top_k 4" in status
    assert "Model provider: ok" in status and "Definitions: 0" in status  # the build embedded; no defined terms
    # "ok" also follows a search answered from cached embeddings (no provider call): the help does not claim a call.
    provider = next(metric for metric in at.metric if metric.label == "Model provider")
    assert provider.help == PROVIDER_STATUS_HELP and "not a live check" in provider.help
    assert retrieval_inputs(at) == (4, 4, pytest.approx(0.2))  # the API defaults from /api/fia/status

    body = api_client.get_client().fia_query(PIT_QUESTION)
    llm.calls.clear()
    ask(at, PIT_QUESTION, top_k=3)
    assert_no_exception(at)
    assert len(llm.calls) == 1

    card = at.get_by_key("fia_answer_card")
    assert card.markdown[0].value == ":green-badge[:material/verified: Grounded answer]"  # short: details follow
    assert "The regulations state\\: 80km/h :blue-badge[S1]." in texts(card)
    assert "Not checked: whether each sentence says what its passage says (the claim verifier is off)." in markdown_text(card)
    assert body["citations"] == ["S1"] and body["retrieved_passages"][0]["cited"]
    chips = at.get("popover")
    assert [chip.proto.popover.label for chip in chips] == ["S1 · B1.6 · B1"]
    cited = body["retrieved_passages"][0]
    assert "80km/h will be imposed in the pit lane" in texts(chips[0])
    assert ":blue-badge[S1] :green-badge[cited] :gray-badge[regulation passage]" in texts(chips[0])

    passage = expander(at, "S1 · cited")
    assert passage.label == "S1 · cited · B1.6 · printed page B1 · PDF page 1 · similarity 0.644"
    assert "> B1.6 Pit Lane Speed" in texts(passage) and "Source: section\\_b\\_sporting.pdf" in texts(passage)
    frame = evidence_frame(at)
    assert frame["Label"].tolist() == ["S1"] and frame["Cited"].tolist() == [True]
    assert frame["Similarity"].tolist() == [cited["score"]] and frame["Rule / defined term"].tolist() == ["B1.6"]
    assert frame["Printed page"].tolist() == ["B1"] and frame["PDF page"].tolist() == [1]
    assert frame["Section"].isna().all()  # no manifest: the section is unknown, not guessed

    text = page_text(at)
    assert "Evidence strength: 0.644" in text and "Passages cited: 1 of 1" in text
    assert "Evidence strength is the best similarity of the cited regulation passages to the question, not a " in (
        markdown_text(at)
    )
    assert "Retrieval: top_k 3 · threshold 0.200 · passages above it: 1" in text  # the requested depth was sent
    below = expander(at, "Below the threshold (2)")
    assert below.label == "Below the threshold (2): retrieved, not given to the model"
    assert below.caption[1].value.startswith("**#2 · similarity")  # ranks continue after the passage above
    assert "Validation details" in expander_labels(at) and "Raw API response" in expander_labels(at)
    assert not at.get("vega_lite_chart")

    at.button(key="fia_search_question_example_0").click().run()  # the answer stays while the other tab is used
    assert_no_exception(at)
    assert at.get_by_key("fia_answer_card") and len(llm.calls) == 1


def test_definition_passage_is_shown_as_a_definition(ui_api, tmp_path, monkeypatch, qdrant):
    folder = write_definitions_corpus(tmp_path / "fia_docs")
    manifest = {"documents": [{"filename": "section_b_sporting.pdf", "section": "B", "source_url": OFFICIAL_URL}]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    llm = ScriptedChatModel(
        "Speeding in the pit lane during a TTCS gives a drive through penalty [S1][S2]. TTCS include the Race session [S2]."
    )
    rag = make_definitions_rag(folder, tmp_path, qdrant, llm=llm)
    rag.build_index()
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    at = ask(run_page("regulations"), DEFINITIONS_QUESTION)
    assert_no_exception(at)
    glossary = rag.status()["index"]["glossary"]
    assert glossary["entries"] >= 3 and f"Definitions: {glossary['entries']}" in page_text(at)
    assert [chip.proto.popover.label for chip in at.get("popover")] == [
        "S1 · B1.6 · B1",
        "S2 · Total Time Classified Session (TTCS) · B85",
    ]
    answer = texts(at.get_by_key("fia_answer_card")).split("\n")[1]
    assert answer == (  # adjacent citations stay two badges
        "Speeding in the pit lane during a TTCS gives a drive through penalty :blue-badge[S1] :blue-badge[S2]. "
        "TTCS include the Race session :blue-badge[S2]."
    )
    definition = expander(at, "S2 · cited")
    assert (
        definition.label
        == "S2 · cited · definition\\: Total Time Classified Session (TTCS) · Section B · printed page B85 · PDF page 2"
    )
    body = texts(definition)
    assert ":blue-badge[S2] :green-badge[cited] :violet-badge[definition]" in body
    assert "defined term **Total Time Classified Session (TTCS)**" in body and "similarity" not in body.split("\n")[0]
    assert "its score is not a similarity" in body
    assert f"[section\\_b\\_sporting.pdf, official FIA PDF]({OFFICIAL_URL}#page=2)" in body
    regulation = texts(expander(at, "S1 · cited"))
    assert (
        ":gray-badge[regulation passage] Section B · printed page B1 · PDF page 1 · nearest rule `B1.6` · similarity"
        in regulation
    )
    frame = evidence_frame(at)
    assert frame["Kind"].tolist() == ["regulation", "definition", "definition"]
    assert frame["Similarity"].isna().tolist() == [False, True, True]  # a definition score is not a similarity
    assert frame["Cited"].tolist() == [True, True, False]
    assert frame["Official PDF"].tolist() == [f"{OFFICIAL_URL}#page=1"] + [f"{OFFICIAL_URL}#page=2"] * 2
    assert frame["Section"].tolist() == ["B"] * 3 and frame["Printed page"].tolist() == ["B1", "B85", "B85"]
    assert frame["PDF page"].tolist() == [1, 2, 2]

    at = search(at, DEFINITIONS_QUESTION, top_k=1, min_score=0.1)  # the API default threshold of this pipeline
    assert_no_exception(at)
    assert "Definitions added: 2" in page_text(at) and "Definitions Ask would add (2)" in page_text(at)
    assert expander(at, "Total Time Classified Session (TTCS) · Section B")


def test_an_answer_citing_only_definitions_has_no_evidence_strength(ui_api, tmp_path, monkeypatch, qdrant):
    folder = write_definitions_corpus(tmp_path / "fia_docs")
    rag = make_definitions_rag(folder, tmp_path, qdrant, llm=ScriptedChatModel("TTCS include the Race session [S2]."))
    rag.build_index()
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    body = api_client.get_client().fia_query(DEFINITIONS_QUESTION)
    assert body["grounded"] and body["confidence"] == 0.0 and body["top_retrieval_score"] > 0.5
    at = ask(run_page("regulations"), DEFINITIONS_QUESTION)
    assert_no_exception(at)
    text = page_text(at)
    assert "Evidence strength: –" in text and "Evidence strength: 0.000" not in text
    assert "Only official definitions were cited. Definitions are not retrieved by similarity" in markdown_text(at)


def test_evidence_strength_is_the_cited_passage_score_not_the_top_retrieval_score(ui_api, built_rag, monkeypatch):
    rag, llm = built_rag
    with_retrieval(rag, monkeypatch, top_k=4, min_score=0.05)  # two passages clear it; the model cites the second
    llm.reply = "A car must not be released from its pit stop position in an unsafe condition [S2]."
    body = api_client.get_client().fia_query(PIT_QUESTION)
    assert body["grounded"] and body["citations"] == ["S2"]
    assert 0 < body["confidence"] < body["top_retrieval_score"]
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    assert f"Evidence strength: {body['confidence']:.3f}" in page_text(at)


def test_a_no_evidence_decline_reports_the_best_similarity_against_the_threshold(ui_api, built_rag, monkeypatch):
    rag, llm = built_rag
    with_retrieval(rag, monkeypatch, top_k=4, min_score=0.9)
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    assert llm.calls == []
    card = texts(at.get_by_key("fia_decline_card"))
    assert "Reason code `no_evidence_above_threshold`" in card and "Best similarity 0.644, threshold 0.900." in card


def test_a_verified_answer_says_what_the_claim_verifier_judged(ui_api, built_rag, monkeypatch):
    rag, llm = built_rag
    monkeypatch.setattr(rag, "settings", replace(rag.settings, generation=GenerationConfig(verify_claims=True)))
    llm.reply = lambda messages: '{"unsupported": []}' if messages[0].content == VERIFIER_PROMPT else GOOD_ANSWER
    at = run_page("regulations")
    assert "Claim verifier: on" in page_text(at)
    at = ask(at, PIT_QUESTION)
    assert_no_exception(at)
    assert len(llm.calls) == 2
    note = markdown_text(at.get_by_key("fia_answer_card"))
    assert "The claim verifier (a second model call) also judged every sentence supported" in note
    assert "a model's judgement, not proof" in note and "Not checked" not in note
    assert "**Claim verification:** verified · 1 sentences checked" in texts(expander(at, "Validation details"))


@pytest.mark.parametrize(
    "reply, shown",
    [
        ("The pit lane limit is 80km/h [1].", "The pit lane limit is 80km/h :blue-badge[S1]."),
        ("The pit lane limit is 80km/h (S1).", "The pit lane limit is 80km/h :blue-badge[S1]."),
        ("The pit lane limit is 80km/h [source: S1].", "The pit lane limit is 80km/h :blue-badge[S1]."),
        ("The pit lane limit is 80km/h【S1】.", "The pit lane limit is 80km/h :blue-badge[S1]."),
        ("The pit lane limit is 80km/h [Source 1].", "The pit lane limit is 80km/h :blue-badge[S1]."),
        ("The pit lane limit is 80km/h, as source S1 states.", "The pit lane limit is 80km/h, as source :blue-badge[S1] states."),
        # full-width and superscript forms the API folds to [S1] / (source S1)
        ("The pit lane limit is 80km/h［Ｓ１］.", "The pit lane limit is 80km/h :blue-badge[S1]."),
        ("The pit lane limit is 80km/h [S¹].", "The pit lane limit is 80km/h :blue-badge[S1]."),
        ("The pit lane limit is 80km/h (ｓｏｕｒｃｅ S1).", "The pit lane limit is 80km/h :blue-badge[S1]."),
    ],
)
def test_every_citation_style_the_api_accepts_is_shown_as_its_label(ui_api, built_rag, reply, shown):
    _, llm = built_rag
    llm.reply = reply
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    assert answer_line(at) == shown
    assert [chip.proto.popover.label for chip in at.get("popover")] == ["S1 · B1.6 · B1"]


def test_a_range_citation_is_shown_as_the_labels_it_names(ui_api, built_rag, monkeypatch):
    rag, llm = built_rag
    with_retrieval(rag, monkeypatch, top_k=4, min_score=0.05)  # two passages clear it
    llm.reply = "The pit lane and pit stop releases are both regulated [S1-S2]."
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    assert answer_line(at) == "The pit lane and pit stop releases are both regulated :blue-badge[S1, S2]."


def test_bold_and_italics_of_an_answer_render_but_links_and_directives_do_not(ui_api, built_rag):
    _, llm = built_rag
    llm.reply = "The pit lane limit is **80km/h** in _all_ sessions, see [the PDF](https://example.com) [S1]."
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    assert answer_line(at) == (
        "The pit lane limit is **80km/h** in _all_ sessions, see \\[the PDF\\](https\\://example.com) :blue-badge[S1]."
    )


def test_unanswerable_question_is_declined_without_citations(ui_api, built_rag):
    _, llm = built_rag
    llm.calls.clear()
    at = ask(run_page("regulations"), UNANSWERABLE)
    assert_no_exception(at)
    assert llm.calls == []  # nothing reached the threshold: no model call
    card = texts(at.get_by_key("fia_decline_card"))
    assert "**No passage was similar enough to the question**" in card
    assert "the answer model was not called" in card and "Best similarity 0.000, threshold 0.200." in card
    assert "Reason code `no_evidence_above_threshold`" in card
    text = page_text(at)
    assert "Declined: no answer is shown" in text and "Grounded answer" not in text
    assert "None: no passage reached the similarity threshold" in text
    with pytest.raises(KeyError):
        at.get_by_key("fia_answer_card")
    assert not at.get("popover") and not at.dataframe and ":blue-badge[S" not in text
    assert "Evidence strength" not in text
    assert expander(at, "Below the threshold (4)")
    with pytest.raises(KeyError):
        at.get_by_key("fia_rejected_output")  # the model was not called, so there is no rejected output


def test_fabricated_citation_is_declined_and_its_text_kept_aside(ui_api, built_rag):
    _, llm = built_rag
    llm.reply = FABRICATED
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    card = at.get_by_key("fia_decline_card")
    assert "**The answer cited a source that was not supplied**" in texts(card)
    assert "Cited labels that were not supplied:\n- S9" in texts(card) and "Reason code `invalid_citation`" in texts(card)
    rejected = at.get_by_key("fia_rejected_output")
    assert rejected.label == "Rejected model output: failed validation, not an answer"
    assert rejected.proto.expanded is False
    assert [node.value for node in rejected.code] == [FABRICATED]
    rendered = [node.value for kind in ("markdown", "caption") for node in getattr(at, kind)]
    assert not any("according to Article B9.9" in value for value in rendered)  # only as rejected code, and in the raw JSON
    assert "Rule references not found in the cited passages:\n- B9.9" in texts(card)  # named as unsupported
    assert not at.get("popover")
    with pytest.raises(KeyError):
        at.get_by_key("fia_answer_card")
    assert "Cited labels that were not supplied: S9" in texts(expander(at, "Validation details"))


@pytest.mark.parametrize("reason", sorted(DECLINE_CASES))
def test_every_decline_reason_is_explained(ui_api, built_rag, monkeypatch, reason):
    rag, llm = built_rag
    reply, headline = DECLINE_CASES[reason]
    llm.reply = reply
    if reason == "truncated_model_output":
        llm.finish_reason = "length"
    if reason == "unverified_claim":
        monkeypatch.setattr(rag, "settings", replace(rag.settings, generation=GenerationConfig(verify_claims=True)))
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    card = texts(at.get_by_key("fia_decline_card"))
    assert f"**{headline}**" in card and f"Reason code `{reason}`" in card
    assert "**What you can do:** " in markdown_text(at.get_by_key("fia_decline_card"))  # not in a faint caption
    assert ("FIA_RAG_MAX_OUTPUT_TOKENS" in card) == (reason == "truncated_model_output")
    if reason == "empty_model_output":
        with pytest.raises(KeyError):
            at.get_by_key("fia_rejected_output")  # nothing to inspect
    else:
        rejected = at.get_by_key("fia_rejected_output")
        assert rejected.proto.expanded is False and rejected.code[0].value
        assert "It is not a statement of the regulations." in markdown_text(rejected)
    details = texts(expander(at, "Validation details"))
    if reason == "unverified_claim":
        assert "**Claim verification:** unverified · 1 sentences checked · unsupported sentence numbers: \\[1\\]" in details
        assert '{"unsupported": [1]}' in details and "Sentences the claim verifier flagged:\n- " in card
    else:
        assert "Claim verification did not run" in details


def test_a_reply_stopped_by_the_content_filter_is_not_blamed_on_the_output_limit(ui_api, built_rag):
    _, llm = built_rag
    llm.reply, llm.finish_reason = GOOD_ANSWER, "content_filter"
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    card = texts(at.get_by_key("fia_decline_card"))
    assert "**The answer was cut off**" in card and "Reason code `truncated_model_output`" in card
    assert "The provider's content filter stopped the model's reply" in card
    assert "a higher output limit does not" in card and "FIA_RAG_MAX_OUTPUT_TOKENS" not in card and "output limit and" not in card
    assert at.get_by_key("fia_rejected_output").code[0].value == GOOD_ANSWER  # the model's text as it was


def test_a_reply_stopped_at_max_tokens_is_explained_as_the_output_limit(ui_api, built_rag):
    _, llm = built_rag
    llm.reply, llm.finish_reason = GOOD_ANSWER, "max_tokens"  # some providers' name for "length"
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    card = texts(at.get_by_key("fia_decline_card"))
    assert "The model's reply reached its output limit" in card and "raise `FIA_RAG_MAX_OUTPUT_TOKENS`" in card


def test_the_covered_decline_reasons_are_the_documented_ones(ui_api):
    documented = ui_api.get_openapi()["components"]["schemas"]["FIAAnswerResponse"]["properties"]["decline_reason"]
    reasons = {reason.strip() for reason in documented["description"].split("|")}
    # the other two are covered by the no-evidence and fabricated-citation tests above
    assert reasons == set(DECLINE_CASES) | {"no_evidence_above_threshold", "invalid_citation"}


# ------------------------------------------------------------------ Search passages


def test_search_shows_scores_and_the_threshold_split_without_the_answer_model(ui_api, built_rag):
    _, llm = built_rag
    llm.calls.clear()
    at = run_page("regulations")
    assert at.slider(key="fia_search_min_score").value == pytest.approx(0.2)  # the API default
    at = search(at, PIT_QUESTION, top_k=3, min_score=0.3)  # both differ from the API defaults (4, 0.2)
    assert_no_exception(at)
    assert llm.calls == []
    text = page_text(at)
    assert "Best similarity: 0.644" in text and "Above the threshold: 1" in text and "Below the threshold: 2" in text
    assert "top_k 3, threshold 0.300" in text
    assert "No answer model was called: these are search results, not an answer." in markdown_text(at)

    chart = at.get("vega_lite_chart")[0]
    spec = json.loads(chart.proto.spec)
    rule = spec["layer"][1]
    assert rule["mark"]["type"] == "rule" and rule["encoding"]["x"]["datum"] == pytest.approx(0.3)
    assert spec["layer"][0]["encoding"]["x"]["title"] == "Similarity"  # short enough for a phone
    assert spec["layer"][0]["encoding"]["color"]["legend"]["direction"] == "vertical"  # not cut off on a phone
    assert "dashed line: threshold 0.300." in text
    rows = convert_arrow_bytes_to_pandas_df(chart.proto.data.data)
    assert rows["similarity"].tolist() == sorted(rows["similarity"].tolist(), reverse=True)
    assert rows["group"].tolist() == ["Above the threshold"] + ["Below the threshold"] * 2
    assert rows["passage"].tolist()[0] == "#1 B1.6 · B1"

    assert expander(at, "#1 · B1.6 · B1 · similarity 0.644")
    below = expander(at, "Below the threshold (2)")
    assert below.proto.expanded is True
    greyed = [node.value for node in below.caption]
    assert greyed[0] == "Similarity under 0.300. Listed for inspection only; they are not evidence."
    assert greyed[1].startswith("**#2 · similarity 0.080** · printed page B3 · PDF page 3 · `B4.2`")
    assert not below.markdown  # below-threshold passages are only shown greyed
    assert spec["layer"][0]["encoding"]["color"]["scale"]["domain"] == ["Above the threshold", "Below the threshold"]

    at = search(at, PIT_QUESTION, top_k=1, min_score=0.3)  # one passage, above the threshold
    assert_no_exception(at)
    assert "Below the threshold: 0" in page_text(at)
    scale = json.loads(at.get("vega_lite_chart")[0].proto.spec)["layer"][0]["encoding"]["color"]["scale"]
    assert scale["domain"] == ["Above the threshold"] and len(scale["range"]) == 1  # no legend entry for an empty group


def test_passage_titles_from_the_pdfs_are_shown_literally_on_both_tabs(ui_api, built_rag, monkeypatch):
    rag, llm = built_rag
    real = rag.retrieve
    definition = RetrievedPassage(
        "definition:B:x", "A marked-up term.", 0.0, "b.pdf", 2, kind="definition", defined_term="*Pit* [lane]"
    )

    def marked_up(*args, **kwargs):  # rule headings and defined terms are PDF text, never Markdown
        result = real(*args, **kwargs)
        return replace(
            result, passages=[replace(p, nearest_rule="B1.6 *x* [y]") for p in result.passages], definitions=[definition]
        )

    monkeypatch.setattr(rag, "retrieve", marked_up)
    at = search(run_page("regulations"), PIT_QUESTION, top_k=1, min_score=0.3)
    assert_no_exception(at)
    assert expander(at, "#1 · B1.6 \\*x\\* \\[y\\] · B1 · similarity")
    assert expander(at, "\\*Pit\\* \\[lane\\] · PDF page 2")
    llm.reply = GOOD_ANSWER
    at = ask(at, PIT_QUESTION)
    assert_no_exception(at)
    assert [chip.proto.popover.label for chip in at.get("popover")] == ["S1 · B1.6 \\*x\\* \\[y\\] · B1"]


def test_search_says_what_ask_would_do_only_at_the_api_default_threshold(ui_api, built_rag):
    at = run_page("regulations")
    assert at.slider(key="fia_search_min_score").help.startswith("For this search only: Ask always uses the API default, 0.200")

    at = search(at, PIT_QUESTION, top_k=4, min_score=0.9)
    assert_no_exception(at)
    text = page_text(at)
    assert "Above the threshold: 0" in text and "None at this threshold." in text and "Ask would decline" not in text
    assert "Ask always uses the API default threshold (0.200); this threshold is for exploring" in markdown_text(at)
    assert expander(at, "Below the threshold (4)").label.endswith("retrieved, not evidence at this threshold")
    help_text = {metric.label: metric.proto.help for metric in at.metric}["Above the threshold"]
    assert "answer model" not in help_text
    at = ask(at, PIT_QUESTION)  # Ask used the API default (0.2), where the pit lane passage is evidence
    assert_no_exception(at)
    assert at.get_by_key("fia_answer_card")

    at = search(at, UNANSWERABLE, top_k=4, min_score=0.2)
    assert_no_exception(at)
    text = page_text(at)
    assert "None: at the API default threshold, Ask would decline this question without calling the answer model." in text
    assert "Ask always uses the API default threshold" not in text
    help_text = {metric.label: metric.proto.help for metric in at.metric}["Above the threshold"]
    assert help_text.startswith("At the API default threshold, the passages Ask gives the answer model")
    assert expander(at, "Below the threshold (4)").label.endswith("retrieved, would not be given to the model")


# ------------------------------------------------------------------ status, errors and inputs


def test_unconfigured_rag_shows_the_reason_and_the_fix_and_locks_the_forms(ui_api):
    at = run_page("regulations")
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[not configured] **Regulation index**" in text
    assert "OPENAI\\_API\\_KEY is not set" in [node.value for node in at.warning]
    assert "Download the official PDFs: `python scripts/fetch_fia_regulations.py`" in text
    assert "Asking and searching are disabled until the index is ready" in at.info[0].value
    assert all(node.disabled for node in at.text_area) and at.slider(key="fia_search_min_score").disabled
    assert at.button(key="fia_ask_submit").disabled and at.button(key="fia_search_submit").disabled
    assert not at.button(key="fia_refresh_status").disabled


def test_an_index_that_is_not_built_locks_the_forms(ui_api, installed_rag):
    at = run_page("regulations")  # the documents and models are there, the index is not
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[index missing] **Regulation index**" in text and "`python scripts/build_fia_index.py`" in text
    assert forms_locked(at) and "Asking and searching are disabled until the index is ready" in at.info[0].value
    assert "Model provider: unknown" in text and "Indexed passages: 0" in text and "Definitions: unknown" in text


def test_an_out_of_date_index_locks_the_forms(ui_api, built_rag, monkeypatch):
    rag, _ = built_rag
    monkeypatch.setattr(rag, "settings", replace(rag.settings, chunking=ChunkingConfig(chunk_size=300, chunk_overlap=30)))
    at = run_page("regulations")
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[index stale] **Regulation index**" in text and "chunking/embedding settings changed" in text
    assert forms_locked(at) and "Asking and searching are disabled until the index is ready" in at.info[0].value


def test_a_missing_glossary_locks_the_forms_but_unavailable_storage_does_not(ui_api, built_rag, monkeypatch):
    rag, _ = built_rag
    with monkeypatch.context() as patch:
        patch.setattr(rag, "glossary", lambda fingerprint: None)  # e.g. an index built by an older glossary version
        at = run_page("regulations")
        assert_no_exception(at)
        text = page_text(at)
        assert "**Regulation index**" in text and rag.status()["index"]["glossary"]["status"] == "missing"
        assert "the definitions glossary is missing" in text and "Rebuild the index to add the definitions glossary" in text
        assert forms_locked(at) and at.text_area(key="fia_ask_question").disabled  # every request would be a 503
        assert "Asking and searching are disabled until the index is ready" in at.info[0].value

    def storage_held_elsewhere():
        raise VectorStoreError("Storage folder is already accessed by another instance of Qdrant client")

    monkeypatch.setattr(rag, "index", storage_held_elsewhere)
    at = run_page("regulations")
    assert_no_exception(at)
    assert "unavailable] **Regulation index**" in page_text(at) and "already accessed" in page_text(at)
    assert not forms_locked(at) and not at.info  # a later attempt can succeed


def test_retrieval_inputs_start_from_the_api_defaults_once_the_api_is_reachable(ui_api_down, built_rag):
    at = run_page("regulations")
    assert forms_locked(at) and retrieval_inputs(at)[:2] == (None, None)  # unknown: empty, not a made-up value
    install_client(ApiClient(http=TestClient(app)))  # the API was started
    at.run()
    assert_no_exception(at)
    assert ":green-badge[ready] **Regulation index**" in page_text(at) and not forms_locked(at)
    assert retrieval_inputs(at) == (4, 4, pytest.approx(0.2))


def test_retrieval_inputs_are_back_at_the_api_defaults_after_visiting_another_page(ui_api, built_rag):
    at = run_page(ENTRY_POINT)
    at.switch_page("views/regulations.py").run()
    assert retrieval_inputs(at) == (4, 4, pytest.approx(0.2))
    at.switch_page("views/overview.py").run()  # the regulation inputs are not drawn: Streamlit drops their state
    at.switch_page("views/regulations.py").run()
    assert_no_exception(at)
    assert retrieval_inputs(at) == (4, 4, pytest.approx(0.2))  # not the widgets' minimums (1 and 0.00)


def test_retrieval_inputs_follow_a_fixed_or_changed_api_default_but_keep_a_chosen_value(ui_api, built_rag, monkeypatch):
    rag, _ = built_rag
    monkeypatch.setattr(rag_pipeline, "_instance", None)
    monkeypatch.setenv("FIA_RAG_TOP_K", "0")  # invalid: /api/fia/status has no settings
    at = run_page("regulations")
    assert ":red-badge[misconfigured] **Regulation index**" in page_text(at) and forms_locked(at)

    monkeypatch.setattr(rag_pipeline, "_instance", rag)  # fixed, and the API restarted
    at.button(key="fia_refresh_status").click().run()
    assert ":green-badge[ready] **Regulation index**" in page_text(at)
    assert retrieval_inputs(at) == (4, 4, pytest.approx(0.2))

    at = search(at, PIT_QUESTION, top_k=3, min_score=0.2)  # the user chose top_k 3 and kept the threshold
    with_retrieval(rag, monkeypatch, top_k=2, min_score=0.3)
    at.button(key="fia_refresh_status").click().run()
    assert_no_exception(at)
    assert "default top_k 2 · similarity threshold 0.300" in page_text(at)
    assert retrieval_inputs(at) == (2, 3, pytest.approx(0.3))  # untouched inputs follow; the chosen top_k stays
    at = ask(at, PIT_QUESTION)
    assert_no_exception(at)
    assert "Retrieval: top_k 2 · threshold 0.300" in page_text(at)
    at.button(key="fia_search_submit").click().run()
    assert "top_k 3, threshold 0.300" in page_text(at)


def test_a_503_from_an_unconfigured_rag_shows_the_api_reason(ui_api, built_rag, tmp_path, monkeypatch):
    rag, llm = built_rag
    # No key and no PDFs; built from explicit settings, never from the environment.
    unconfigured = FIARegulationRAG(
        RAGSettings(docs_path=tmp_path / "no_documents", qdrant=QdrantConfig(path=tmp_path / "qdrant"))
    )

    def configuration_lost(question, top_k=None):  # while the question is in flight
        monkeypatch.setattr(rag_pipeline, "_instance", unconfigured)
        return unconfigured.answer(question, top_k)

    monkeypatch.setattr(rag, "answer", configuration_lost)
    at = run_page("regulations")
    assert ":green-badge[ready]" in page_text(at)
    at = ask(at, PIT_QUESTION)
    assert llm.calls == []
    assert_no_exception(at)
    warning = service_unavailable(at)
    assert warning.startswith("Service unavailable (HTTP 503): FIA RAG is unavailable")
    assert "no\\_documents does not exist. Run `python scripts/fetch_fia_regulations.py`" in warning
    # The cause names both possibilities: here the PDFs are missing, not a setting.
    assert "A required setting (such as `OPENAI_API_KEY`) or the regulation PDFs are missing or invalid" in page_text(at)
    text = page_text(at)  # the status fetched after the request explains the fix
    assert ":red-badge[not configured] **Regulation index**" in text and "Set `OPENAI_API_KEY` in `.env`" in text
    assert at.button(key="fia_ask_submit").disabled


def test_a_failing_provider_keeps_the_forms_usable_and_the_sidebar_in_step(ui_api, built_rag, monkeypatch):
    rag, _ = built_rag
    broken = [True]
    embeddings = rag.embedder()._embeddings
    original = embeddings.embed_query

    def flaky(text):
        if broken[0]:
            raise ConnectionError("provider unreachable")
        return original(text)

    monkeypatch.setattr(embeddings, "embed_query", flaky)
    at = run_page(ENTRY_POINT)
    at.switch_page("views/regulations.py").run()
    at = ask(at, PIT_QUESTION)
    assert_no_exception(at)
    assert "provider unreachable" in service_unavailable(at)
    assert "The model provider call failed" in page_text(at)
    sidebar = sidebar_text(at)
    assert ":red-badge[provider failing] FIA regulation QA" in sidebar
    assert ":red-badge[provider failing] **Regulation index**" in page_text(at) and "Model provider: failing" in page_text(at)
    assert not at.button(key="fia_ask_submit").disabled  # a retry can succeed
    assert (
        "Submitting again calls the model provider again (with its automatic retries); the state clears after a "
        "successful request."
    ) in markdown_text(at)

    broken[0] = False
    at.button(key="fia_ask_submit").click().run()
    assert_no_exception(at)
    assert at.get_by_key("fia_answer_card")
    sidebar = sidebar_text(at)
    assert ":green-badge[ready] FIA regulation QA" in sidebar and "provider failing" not in sidebar


def test_api_down_shows_the_connection_hint(ui_api_down):
    at = run_page("regulations")
    assert_no_exception(at)
    error = at.error[0].value
    assert error.startswith(f"The regulation index status check could not reach the API at `{ui_api_down.base_url}`")
    port = ui_api_down.base_url.rsplit(":", 1)[1]
    assert f"uvicorn app.main:app --port {port}" in at.info[0].value
    assert at.button(key="fia_ask_submit").disabled and at.button(key="fia_search_submit").disabled
    assert not at.metric and "Regulation index" not in page_text(at)


def test_inputs_follow_the_api_limits_and_empty_questions_are_not_sent(ui_api, built_rag, monkeypatch):
    at = run_page("regulations")
    assert at.text_area(key="fia_ask_question").max_chars == MAX_QUESTION_CHARS
    assert at.number_input(key="fia_ask_top_k").max == RetrievalConfig.MAX_TOP_K
    assert at.number_input(key="fia_search_top_k").min == 1
    assert "if a passage reaches the threshold, one answer-model call" in page_text(at)  # a no-evidence decline makes none
    assert "retried automatically, up to `FIA_RAG_MAX_RETRIES` more times" in page_text(at)
    calls = []
    monkeypatch.setattr(ui_api, "fia_query", lambda *args: calls.append(args))
    at = ask(at, "   ")
    assert_no_exception(at)
    assert calls == [] and at.warning[0].value == "Enter a question first."


def test_a_422_is_shown_with_the_validation_message(ui_api, built_rag, monkeypatch):
    rag, _ = built_rag

    def rejected(question, top_k=None):
        raise ValueError("question must be at most 12 characters")  # a stricter API: its ValueError becomes a 422

    monkeypatch.setattr(rag, "answer", rejected)
    at = ask(run_page("regulations"), PIT_QUESTION)
    assert_no_exception(at)
    assert at.error[0].value == "The question was rejected by the API (HTTP 422):"
    assert "question must be at most 12 characters" in page_text(at) and "Correct the input and submit again." in page_text(at)
    assert not at.get_by_key("fia_ask_question").disabled  # the form stays usable to correct the question


def test_examples_are_verbatim_answerable_evaluation_questions(ui_api, built_rag):
    evaluation = json.loads((REPO_ROOT / "scripts" / "fia_rag_eval_questions.json").read_text(encoding="utf-8"))
    answerable = {item["question"] for item in evaluation["questions"] if item["category"] == "answerable"}
    at = run_page("regulations")
    examples = [button for button in at.button if button.key and button.key.startswith("fia_ask_question_example_")]
    assert len(examples) == 4 and {button.help for button in examples} == answerable
    for button in examples:
        button.click().run()
        assert_no_exception(at)
        assert at.text_area(key="fia_ask_question").value == button.help
    at.button(key="fia_search_question_example_2").click().run()
    assert at.text_area(key="fia_search_question").value == examples[2].help
