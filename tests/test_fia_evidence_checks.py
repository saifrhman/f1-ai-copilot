"""Defined-term definitions, number integrity and the optional claim verifier.

PDF parsing, the glossary extraction, Qdrant storage and validation are real;
only the embedding and chat APIs are replaced (HashingEmbeddings, ScriptedChatModel).
"""

import json

import pytest
from langchain_core.documents import Document

import core_modules.rule_checker.fia_rag.pipeline as rag_pipeline
from core_modules.rule_checker.fia_rag import (
    IndexNotReadyError,
    RAGConfigurationError,
    RAGSettings,
    RetrievedPassage,
)
from core_modules.rule_checker.fia_rag.config import GenerationConfig
from core_modules.rule_checker.fia_rag.errors import ProviderError
from core_modules.rule_checker.fia_rag.generation import (
    VERIFIER_PROMPT,
    GroundedAnswerGenerator,
    answer_sentences,
    parse_verifier_reply,
)
from core_modules.rule_checker.fia_rag.glossary import (
    MAX_TERM_FREQUENCY,
    MIN_CHUNKS_FOR_FREQUENCY,
    GlossaryEntry,
    definitions_for,
    extract_glossary,
    with_frequencies,
)
from core_modules.rule_checker.fia_rag.grounding import DeclineReason, label_passages, validate_answer
from tests.helpers import (
    DEFINITIONS,
    DEFINITIONS_QUESTION,
    HashingEmbeddings,
    ScriptedChatModel,
    make_definitions_rag,
    write_definitions_corpus,
)


def page(text, section="B", number=1, source="b.pdf"):
    return Document(page_content=text, metadata={"source": source, "page": number, "page_label": f"{section}{number}", "section": section})


def entry(term, abbreviation=None, definition=None, section="B", frequency=0.0):
    return GlossaryEntry(
        term=term,
        abbreviation=abbreviation,
        definition=definition or f"“{term}” is a defined term of section {section} with a long enough text.",
        source=f"{section.lower()}.pdf",
        page=1,
        page_label=f"{section}1",
        section=section,
        source_url=None,
        frequency=frequency,
    )


# ------------------------------------------------------------------ glossary extraction


def test_definitions_are_extracted_verbatim_with_abbreviations_and_metadata():
    entries = {e.term: e for e in extract_glossary([page(DEFINITIONS, number=85)])}
    ttcs = entries["Total Time Classified Session"]
    assert ttcs.abbreviation == "TTCS" and ttcs.page == 85 and ttcs.page_label == "B85" and ttcs.section == "B"
    # The definition runs up to the next defined term and is copied verbatim.
    assert ttcs.definition.startswith("“Total Time Classified Session” (or “TTCS”) is any")
    assert ttcs.definition.endswith("include the Sprint session and the Race session.")
    assert "Lap Time" not in ttcs.definition
    assert entries["Lap Time Classified Session"].abbreviation == "LTCS"
    assert entries["Official"].abbreviation is None


def test_inline_and_acronym_only_definitions():
    text = (
        '"Accepted Breach Agreement (ABA)" means an agreement between the Cost Cap Administration and a team. '
        '"ASN" means a national sporting authority recognised by the FIA as the sole holder of sporting power.'
    )
    entries = {e.term: e for e in extract_glossary([page(text)])}
    assert entries["Accepted Breach Agreement"].abbreviation == "ABA"
    assert entries["ASN"].abbreviation == "ASN"


def test_frequencies_are_the_share_of_chunks_using_the_abbreviation():
    chunks = ["FIA rule", "FIA and TTCS", "FIAT is not FIA", "nothing"] * (MIN_CHUNKS_FOR_FREQUENCY // 4 + 1)
    entries = with_frequencies([entry("Fédération", "FIA"), entry("Total Time", "TTCS"), entry("Official")], chunks)
    assert [e.frequency for e in entries] == [0.75, 0.25, 0.0]
    # A handful of chunks says nothing about how common a term is.
    assert all(e.frequency == 0.0 for e in with_frequencies([entry("Fédération", "FIA")], ["FIA"] * 3))


# ------------------------------------------------------------------ selection


def test_abbreviations_used_in_passages_select_their_definitions_in_rank_order():
    glossary = [entry("Lap Time Classified Session", "LTCS"), entry("Total Time Classified Session", "TTCS")]
    selected = definitions_for(glossary, ["During a TTCS ...", "During an LTCS ..."], ["B", "B"], "penalty?", 3)
    assert [e.abbreviation for e in selected] == ["TTCS", "LTCS"]


def test_abbreviations_match_case_sensitively_as_whole_words():
    glossary = [entry("Total Time Classified Session", "TTCS")]
    assert definitions_for(glossary, ["ttcs and TTCSX and XTTCS"], ["B"], "q", 3) == []


def test_ubiquitous_abbreviations_are_skipped():
    glossary = [entry("Federation", "FIA", frequency=MAX_TERM_FREQUENCY + 0.01), entry("Total", "TTCS", frequency=0.05)]
    selected = definitions_for(glossary, ["The FIA may impose a penalty during a TTCS."], ["B"], "q", 3)
    assert [e.abbreviation for e in selected] == ["TTCS"]


def test_referential_definitions_add_nothing_and_are_skipped():
    glossary = [entry("Cost Cap", definition="“Cost Cap” has the meaning set out in Article D4.1.2.")]
    assert definitions_for(glossary, ["passage"], ["D"], "What is the cost cap?", 3) == []


def test_question_terms_multi_word_any_case_single_word_exact_case():
    glossary = [entry("Official"), entry("Inaugural Season")]
    assert definitions_for(glossary, ["p"], ["E"], "Is an official allowed?", 3) == []
    assert [e.term for e in definitions_for(glossary, ["p"], ["E"], "Is an Official allowed?", 3)] == ["Official"]
    assert [e.term for e in definitions_for(glossary, ["p"], ["E"], "from the inaugural season on?", 3)] == ["Inaugural Season"]


def test_the_passage_section_definition_is_preferred():
    glossary = [entry("Power Unit", "PU", section="D"), entry("Power Unit", "PU", section="E")]
    selected = definitions_for(glossary, ["other", "the PU cost"], ["D", "E"], "q", 3)
    assert [e.section for e in selected] == ["E"]
    # For a term named in the question, the best-ranked passage's section wins.
    selected = definitions_for(glossary, ["x", "y"], ["E", "D"], "What counts as a Power Unit?", 3)
    assert [e.section for e in selected] == ["E"]


def test_question_terms_come_first_limit_and_known_definitions():
    glossary = [entry("Total Time Classified Session", "TTCS"), entry("Lap Time Classified Session", "LTCS"), entry("Inaugural Season")]
    selected = definitions_for(glossary, ["TTCS and LTCS"], ["B"], "inaugural season?", 2)
    assert [e.term for e in selected] == ["Inaugural Season", "Total Time Classified Session"]
    assert definitions_for(glossary, ["TTCS"], ["B"], "q", 0) == []
    # A definition the passages already contain is not repeated.
    ttcs = glossary[0]
    assert definitions_for([ttcs], [f"TTCS ... {ttcs.definition}"], ["B"], "q", 3) == []


# ------------------------------------------------------------------ pipeline integration


@pytest.fixture
def docs(tmp_path):
    return write_definitions_corpus(tmp_path / "fia_docs")


def test_index_build_stores_the_glossary_and_retrieval_adds_definitions(docs, tmp_path, qdrant):
    rag = make_definitions_rag(docs, tmp_path, qdrant)
    report = rag.build_index()
    assert report.status == "rebuilt" and report.definitions >= 3
    result = rag.retrieve(DEFINITIONS_QUESTION)
    assert len(result.passages) == 1 and "B1.6.4" in result.passages[0].rule_ids
    assert [(d.kind, d.defined_term, d.page_label, d.score) for d in result.definitions] == [
        ("definition", "Total Time Classified Session (TTCS)", "B85", 0.0),
        ("definition", "Lap Time Classified Session (LTCS)", "B85", 0.0),
    ]
    assert result.to_dict()["definitions"][0]["text"].endswith("the Race session.")
    status = rag.status()
    assert status["index"]["glossary"] == {"status": "current", "entries": report.definitions}
    assert not any("glossary" in problem for problem in status["problems"])


def test_glossary_is_reloaded_from_qdrant_by_a_new_process(docs, tmp_path, qdrant):
    make_definitions_rag(docs, tmp_path, qdrant).build_index()
    fresh = make_definitions_rag(docs, tmp_path, qdrant)
    assert fresh.retrieve(DEFINITIONS_QUESTION).definitions[0].defined_term == "Total Time Classified Session (TTCS)"


def test_missing_glossary_blocks_queries_until_rebuilt_without_embedding_calls(docs, tmp_path, qdrant):
    embeddings = HashingEmbeddings()
    rag = make_definitions_rag(docs, tmp_path, qdrant, embeddings=embeddings)
    rag.build_index()
    qdrant.delete_collection(rag.index().alias_target("fia_test_glossary"))
    fresh = make_definitions_rag(docs, tmp_path, qdrant, embeddings=embeddings)
    status = fresh.status()
    assert not status["ready"] and status["index"]["glossary"]["status"] == "missing"
    assert any("glossary" in problem for problem in status["problems"])
    with pytest.raises(IndexNotReadyError, match="glossary"):
        fresh.retrieve(DEFINITIONS_QUESTION)
    # With definitions disabled the index is usable without a glossary.
    assert make_definitions_rag(docs, tmp_path, qdrant, embeddings=embeddings, max_definitions=0).retrieve(DEFINITIONS_QUESTION).definitions == []
    embedded_before = embeddings.documents_embedded
    report = fresh.build_index()
    assert report.status == "glossary_rebuilt" and embeddings.documents_embedded == embedded_before
    assert fresh.retrieve(DEFINITIONS_QUESTION).definitions
    assert fresh.build_index().status == "up_to_date"


def test_glossary_from_an_older_extractor_version_is_not_used(docs, tmp_path, qdrant, monkeypatch):
    make_definitions_rag(docs, tmp_path, qdrant).build_index()
    monkeypatch.setattr(rag_pipeline, "GLOSSARY_VERSION", "999")
    with pytest.raises(IndexNotReadyError, match="glossary"):
        make_definitions_rag(docs, tmp_path, qdrant).retrieve(DEFINITIONS_QUESTION)


def test_definitions_are_labelled_after_passages_and_do_not_raise_confidence(docs, tmp_path, qdrant):
    llm = ScriptedChatModel(
        "Speeding in the pit lane during a TTCS is penalised with a drive through penalty [S1], "
        "and TTCS include the Race session [S2]."
    )
    rag = make_definitions_rag(docs, tmp_path, qdrant, llm=llm)
    rag.build_index()
    result = rag.answer(DEFINITIONS_QUESTION)
    assert result["grounded"], result["validation"]
    prompt = llm.last_prompt
    assert 'label="S2"' in prompt and 'kind="definition"' in prompt and 'defined_term="Total Time Classified Session (TTCS)"' in prompt
    kinds = {p["label"]: (p["kind"], p["cited"]) for p in result["retrieved_passages"]}
    assert kinds["S1"] == ("regulation", True) and kinds["S2"] == ("definition", True)
    assert result["confidence"] == round(result["retrieved_passages"][0]["score"], 4) > 0
    assert result["retrieval"]["definitions_added"] == 2


def test_answer_uses_the_retrieved_definitions_and_a_re_split_retrieval_derives_them_again(docs, tmp_path, qdrant, monkeypatch):
    rag = make_definitions_rag(docs, tmp_path, qdrant, llm=ScriptedChatModel("TTCS include the Race session [S2]."))
    rag.build_index()
    derived = []
    original = rag_pipeline.definitions_for
    monkeypatch.setattr(rag_pipeline, "definitions_for", lambda *args: derived.append(args) or original(*args))
    assert rag.answer(DEFINITIONS_QUESTION)["retrieval"]["definitions_added"] == 2
    assert len(derived) == 1  # by retrieve; the answer uses them as they are
    # with_threshold drops the definitions (the evaluation script re-splits retrievals), so they are derived again.
    resplit = rag.retrieve(DEFINITIONS_QUESTION).with_threshold(0.1)
    assert resplit.definitions == []
    result = rag.answer_from_retrieval(resplit)
    assert result["grounded"] and result["retrieval"]["definitions_added"] == 2


def test_answer_citing_only_definitions_has_zero_evidence_score(docs, tmp_path, qdrant):
    rag = make_definitions_rag(docs, tmp_path, qdrant, llm=ScriptedChatModel("TTCS include the Race session [S2]."))
    rag.build_index()
    result = rag.answer(DEFINITIONS_QUESTION)
    assert result["grounded"] and result["confidence"] == 0.0


def test_definitions_alone_are_never_evidence():
    llm = ScriptedChatModel("should not be called")
    definition = RetrievedPassage.from_glossary(entry("Total Time Classified Session", "TTCS"))
    result = GroundedAnswerGenerator(llm, GenerationConfig()).generate("What is a TTCS?", [], [definition])
    assert result.validation.reason == DeclineReason.NO_EVIDENCE and llm.calls == []


# ------------------------------------------------------------------ number integrity

PIT = RetrievedPassage("c1", "a. A speed limit of 80km/h will be imposed. The fine is 100 EUR.", 0.8, "b.pdf", 10, nearest_rule="12.4")
FUEL = RetrievedPassage("c2", "The fuel mass flow must not exceed one hundred kilograms per hour. The cap is USD 135,000,000 and the ratio 0.30.", 0.7, "c.pdf", 5)
# On PDF page 9, printed page B9, of a real file name (2026, 08, 05, 7); the text itself states no number.
SPORTING = RetrievedPassage(
    "c3", "B1.6.4 Speeding in the pit lane is penalised with a drive through penalty.", 0.7,
    "fia_2026_f1_regulations_-_section_b_sporting_-_iss_08_-_2026-08-05_7.pdf", 9, page_label="B9", nearest_rule="B1.6",
)
EVIDENCE = label_passages([PIT, FUEL, SPORTING])


@pytest.mark.parametrize(
    "answer",
    [
        "The pit lane speed limit is 80 km/h [S1].",
        "The fuel flow limit is 100 kg/h [S2].",  # number word in the evidence
        "The cap is US$135 million [S2].",  # scaled amount
        "The ratio is 0.3 [S2].",  # same value, different notation
        "1. The pit lane limit is 80km/h [S1].\n2. The fine is 100 EUR [S1].",  # list markers
        "Under Article 12.4 the limit is 80km/h [S1].",  # numeric rule id supported by metadata
        "The limit is 80km/h [S1].\n\nTherefore a car at 95km/h is 15km/h too fast.",  # marked inference
        # references to where the cited passage is
        "Speeding is penalised with a drive through penalty (page 9) [S3].",
        "Speeding is penalised with a drive through penalty (PDF page 9, printed page B9, Issue 08) [S3].",
        f"Speeding is penalised with a drive through penalty ({SPORTING.source}) [S3].",
        "The 2026 Sporting Regulations penalise speeding with a drive through penalty [S3].",
    ],
)
def test_numbers_stated_by_the_evidence_are_accepted(answer):
    result = validate_answer(answer, EVIDENCE)
    assert result.grounded, (result.reason, result.unsupported_numbers, result.uncited_claims)


@pytest.mark.parametrize(
    "answer, number",
    [
        ("The pit lane speed limit is 60 km/h [S1].", "60"),
        ("The pit lane limit is 80km/h and the fine is 250 EUR [S1].", "250"),
        ("The cap is US$145 million [S2].", "145"),
        ("The pit lane limit is 80km/h [S2].", "80"),  # stated by S1, but only S2 is cited
        ("The limit is 80km/h for 3 laps [S1].", "3"),
        ("The limit is 80km/h for 10 laps [S1].", "10"),  # S1 is on page 10, but its text does not say 10
        # only the identifier itself is exempt, not every other 12 in the answer
        ("Article 12 sets the limit of 80km/h and adds a 12 place grid penalty [S1].", "12"),
        # a passage's metadata (page, issue, file-name date and suffix) is not evidence
        ("Speeding is penalised with a drive through penalty (page 99) [S3].", "99"),
        ("The 2027 Sporting Regulations penalise speeding with a drive through penalty [S3].", "2027"),
        ("Speeding is penalised with a drive through penalty plus a 5-place grid drop [S3].", "5"),
        ("Speeding is penalised with a 7 second Stop-and-Go Penalty [S3].", "7"),
        ("Speeding costs 8 championship points [S3].", "8"),
        # each reference is compared with its own kind of metadata: 8 is the issue, not the page; 9 the page
        ("Speeding is penalised with a drive through penalty (page 8) [S3].", "8"),
        ("Speeding is penalised with a drive through penalty (Issue 9) [S3].", "9"),
        # the verb "issue" is no document reference, whatever number follows it
        ("The stewards may issue 5 penalty points for speeding in the pit lane [S3].", "5"),
        ("The stewards may issue 8 penalty points for speeding in the pit lane [S3].", "8"),
        # a rule identifier split after a dot does not swallow the number of the next sentence
        ("Speeding is penalised with a drive through penalty under Article B1.6. 4 penalty points are also given [S3].", "4"),
    ],
)
def test_numbers_missing_from_the_cited_evidence_are_declined(answer, number):
    result = validate_answer(answer, EVIDENCE)
    assert result.reason == DeclineReason.UNSUPPORTED_NUMBER and number in result.unsupported_numbers


# ------------------------------------------------------------------ claim verifier

ANSWER = "The pit lane speed limit is 80km/h [S1]. The fine is 100 EUR [S1]."


def scripted(verifier_reply, answer=ANSWER, verifier_finish="stop"):
    class Model(ScriptedChatModel):
        def invoke(self, messages):
            self.calls.append(list(messages))
            from langchain_core.messages import AIMessage

            if messages[0].content == VERIFIER_PROMPT:
                if isinstance(verifier_reply, Exception):
                    raise verifier_reply
                return AIMessage(content=verifier_reply, response_metadata={"finish_reason": verifier_finish})
            return AIMessage(content=answer, response_metadata={"finish_reason": "stop"})

    return Model()


def generate(llm, verify=True):
    return GroundedAnswerGenerator(llm, GenerationConfig(verify_claims=verify)).generate("pit lane limit?", [PIT, FUEL])


def test_verifier_is_off_by_default():
    llm = scripted('{"unsupported": [1]}')
    result = generate(llm, verify=False)
    assert result.validation.grounded and len(llm.calls) == 1 and result.verification is None
    assert GenerationConfig().verify_claims is False


def test_verified_answer_is_accepted_and_verifier_sees_only_cited_excerpts():
    llm = scripted('{"unsupported": []}')
    result = generate(llm)
    assert result.validation.grounded and result.verification["status"] == "verified"
    verifier_prompt = llm.calls[1][1].content
    assert 'label="S1"' in verifier_prompt and 'label="S2"' not in verifier_prompt
    assert "1. The pit lane speed limit is 80km/h [S1]." in verifier_prompt and "2. The fine is 100 EUR [S1]." in verifier_prompt


def test_unsupported_sentence_declines_the_answer():
    result = generate(scripted('```json\n{"unsupported": [2]}\n```'))
    assert result.validation.reason == DeclineReason.UNVERIFIED_CLAIM and not result.validation.grounded
    assert result.validation.unverified_claims == ["The fine is 100 EUR [S1]."]
    assert result.validation.model_output == ANSWER and result.verification["status"] == "unverified"


@pytest.mark.parametrize("reply", ["Looks fine to me.", '{"unsupported": [7]}', '{"unsupported": "none"}', '{"unsupported": [true]}'])
def test_unparseable_verifier_reply_is_not_treated_as_verification(reply):
    result = generate(scripted(reply))
    assert result.validation.reason == DeclineReason.UNVERIFIED_CLAIM
    assert result.verification["status"] == "unparseable_verifier_output"


def test_truncated_verifier_reply_is_not_treated_as_verification():
    result = generate(scripted('{"unsupported": []}', verifier_finish="length"))
    assert result.validation.reason == DeclineReason.UNVERIFIED_CLAIM


def test_verifier_only_runs_on_answers_that_passed_the_deterministic_checks():
    llm = scripted('{"unsupported": []}', answer="The limit is 60km/h [S1].")
    result = generate(llm)
    assert result.validation.reason == DeclineReason.UNSUPPORTED_NUMBER and len(llm.calls) == 1


def test_verifier_provider_failure_is_a_provider_error():
    with pytest.raises(ProviderError, match="Claim verification"):
        generate(scripted(RuntimeError("upstream 502")))


def test_answer_sentences_keep_their_citations():
    assert answer_sentences("A is 1. [S1]\nB is 2 [S2]. Thus C.\n\n- D [S1]") == ["A is 1. [S1]", "B is 2 [S2].", "Thus C.", "- D [S1]"]
    assert parse_verifier_reply('{"unsupported": [2, 1, 2]}', 2) == [1, 2]
    assert parse_verifier_reply("", 2) is None


def test_pipeline_reports_claim_verification(docs, tmp_path, qdrant):
    answer = "Speeding in the pit lane during a TTCS is penalised with a drive through penalty [S1]."
    rag = make_definitions_rag(docs, tmp_path, qdrant, llm=scripted('{"unsupported": []}', answer=answer), verify=True)
    rag.build_index()
    result = rag.answer(DEFINITIONS_QUESTION)
    assert result["grounded"] and result["validation"]["claim_verification"]["status"] == "verified"


# ------------------------------------------------------------------ settings


def test_definition_and_verifier_settings_from_env():
    settings = RAGSettings.from_env({"FIA_RAG_MAX_DEFINITIONS": "0", "FIA_RAG_VERIFY_CLAIMS": "true"})
    assert settings.retrieval.max_definitions == 0 and settings.generation.verify_claims is True
    assert settings.summary()["max_definitions"] == 0 and settings.summary()["verify_claims"] is True
    defaults = RAGSettings.from_env({})
    assert defaults.retrieval.max_definitions == 3 and defaults.generation.verify_claims is False


@pytest.mark.parametrize(
    "env",
    [{"FIA_RAG_MAX_DEFINITIONS": "11"}, {"FIA_RAG_MAX_DEFINITIONS": "-1"}, {"FIA_RAG_MAX_DEFINITIONS": "two"},
     {"FIA_RAG_VERIFY_CLAIMS": "maybe"}],
)
def test_invalid_definition_and_verifier_settings_are_rejected(env):
    with pytest.raises(RAGConfigurationError):
        RAGSettings.from_env(env)


def test_verifier_output_is_json_serialisable(docs, tmp_path, qdrant):
    rag = make_definitions_rag(docs, tmp_path, qdrant, llm=scripted('{"unsupported": [1]}', answer="Speeding in the pit lane during a TTCS is penalised [S1]."), verify=True)
    rag.build_index()
    result = rag.answer(DEFINITIONS_QUESTION)
    assert result["decline_reason"] == DeclineReason.UNVERIFIED_CLAIM
    json.dumps(result)
