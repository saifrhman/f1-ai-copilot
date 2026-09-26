"""The end-to-end evaluation's checks must fail when the RAG misbehaves (no trivially passing harness)."""

import json
import sys
from dataclasses import replace

import pytest

import scripts.check_fia_rag as check_fia_rag
from core_modules.rule_checker.fia_rag import GenerationConfig
from core_modules.rule_checker.fia_rag.generation import VERIFIER_PROMPT
from core_modules.rule_checker.fia_rag.retrieval import RetrievalResult, RetrievedPassage
from scripts.check_fia_rag import (
    ANSWERABLE_CATEGORIES,
    QUESTIONS_FILE,
    _contains,
    calibrate,
    evaluate,
    min_score_argument,
    required_score,
)
from tests.helpers import build_test_rag, cite_passage_containing


def result(answer="The limit is 80km/h [S1].", grounded=True, sections=("B",), cited=(True,), texts=("A speed limit of 80km/h",), citations=("S1",), unsupported=()):
    passages = [
        {"label": f"S{i}", "cited": c, "section": s, "text": t}
        for i, (s, c, t) in enumerate(zip(sections, cited, texts), start=1)
    ]
    return {
        "answer": answer,
        "grounded": grounded,
        "decline_reason": None if grounded else "model_declined",
        "citations": list(citations),
        "retrieved_passages": passages,
        "validation": {"unsupported_rules": list(unsupported)},
    }


ANSWERABLE = {"category": "answerable", "expected_sections": ["B"], "expected_facts_any": [["80km/h", "80 km/h"]]}


def test_correct_grounded_answer_passes():
    assert evaluate(ANSWERABLE, result()) == []
    assert evaluate(ANSWERABLE, result(answer="The limit is 80 km/h [S1].")) == []


def test_decline_of_answerable_question_fails():
    assert "expected a grounded answer" in evaluate(ANSWERABLE, result(grounded=False))[0]


def test_wrong_section_or_missing_fact_fails():
    assert any("Section B" in f for f in evaluate(ANSWERABLE, result(sections=("C",))))
    assert any("lacks expected fact" in f for f in evaluate(ANSWERABLE, result(answer="The limit is 60km/h [S1].")))


def test_fact_must_be_in_cited_text_not_just_the_answer():
    failures = evaluate(ANSWERABLE, result(texts=("Unrelated passage about tyres",)))
    assert any("not in the cited passages" in f for f in failures)


def test_cross_document_answer_must_cite_every_expected_section():
    spec = {"category": "cross_document", "expected_sections": ["C", "B"], "expected_facts_any": [["3000"], ["80"]]}
    both = result(answer="3000 MJ/h [S1] and 80 km/h [S2].", sections=("C", "B"), cited=(True, True), texts=("3000MJ/h", "80km/h"), citations=("S1", "S2"))
    assert evaluate(spec, both) == []
    one = result(answer="3000 MJ/h [S1] and 80 km/h [S1].", sections=("C", "B"), cited=(True, False), texts=("3000MJ/h 80km/h", "80km/h"))
    assert any("Section B" in f for f in evaluate(spec, one))


def test_unanswerable_question_must_be_declined():
    spec = {"category": "unanswerable"}
    assert evaluate(spec, result(grounded=False)) == []
    assert evaluate(spec, result()) == ["expected a decline, got a grounded answer"]


def test_adversarial_answer_must_not_contain_invented_content():
    spec = {"category": "adversarial", "forbidden_facts": ["B42.1"]}
    assert evaluate(spec, result(grounded=False, answer="I cannot answer")) == []
    assert evaluate(spec, result(answer="Article B42.1 requires green helmets [S1].")) != []


def test_citation_to_missing_passage_fails_any_category():
    assert any("does not map" in f for f in evaluate(ANSWERABLE, result(citations=("S1", "S7"))))


def test_question_file_is_well_formed():
    questions = json.loads(QUESTIONS_FILE.read_text())["questions"]
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    categories = {q["category"] for q in questions}
    assert {"answerable", "paraphrased", "cross_document", "unanswerable", "adversarial"} <= categories
    for q in questions:
        if q["category"] in ANSWERABLE_CATEGORIES:
            assert q["expected_sections"] and q["expected_facts_any"]


def test_fact_checks_use_whole_numbers_not_substrings():
    assert _contains("The limit is 80 km/h [S1].", "80") and _contains("fined €100 per km/h", "100")
    assert not _contains("up to 800 km/h", "80") and not _contains("from 29 December", "9")
    assert _contains("US $215,000,000", "215000000") and _contains("US Dollars 215,000,000", "215,000,000")
    assert _contains("a limit of 80km/h", "80 km/h") and not _contains("a limit of 180km/h", "80km/h")
    shutdown = {"category": "answerable", "expected_sections": ["F"], "expected_facts_any": [["14", "fourteen"], ["9", "nine"]]}
    wrong = result(answer="Two periods: 14 days, then seven (7) days from 29 December [S1].", sections=("F",),
                   texts=("fourteen (14) consecutive days ... nine (9) consecutive calendar days starting on 24 December",))
    assert any("lacks expected fact" in failure for failure in evaluate(shutdown, wrong))


# ------------------------------------------------------------------ threshold calibration


def passage(text, score, section="B"):
    return RetrievedPassage(chunk_id=text, text=text, score=score, source="x.pdf", page=1, section=section)


def retrieval(question, *passages):
    return RetrievalResult(question=question, top_k=8, min_score=0.0, passages=list(passages))


def test_required_score_is_the_weakest_best_evidence():
    spec = {"category": "cross_document", "expected_sections": ["C", "B"], "expected_facts_any": [["3000"], ["80"]]}
    passages = [passage("fuel flow 3000MJ/h", 0.62, "C"), passage("pit 80km/h", 0.41, "B"), passage("pit 80 km/h again", 0.35, "B")]
    assert required_score(spec, passages) == 0.41
    # A fact that was not retrieved at all is a retrieval miss, not a threshold question.
    assert required_score(spec, passages[:1]) is None
    # Whole-number matching: "800" does not carry the fact "80".
    assert required_score({"expected_facts_any": [["80"]]}, [passage("limit 800", 0.9)]) is None


def test_calibration_recommends_the_highest_threshold_keeping_all_answerable_evidence():
    specs = [
        {"id": "a1", "category": "answerable", "expected_sections": ["B"], "expected_facts_any": [["80"]]},
        {"id": "a2", "category": "paraphrased", "expected_sections": ["C"], "expected_facts_any": [["726"]]},
        {"id": "u1", "category": "unanswerable"},
        {"id": "u2", "category": "unanswerable"},
        {"id": "r1", "category": "review"},
    ]
    retrievals = {
        "a1": retrieval("q", passage("limit 80km/h", 0.52)),
        "a2": retrieval("q", passage("mass 726kg", 0.47, "C")),
        "u1": retrieval("q", passage("unrelated", 0.31)),
        "u2": retrieval("q", passage("unrelated", 0.55)),
        "r1": retrieval("q", passage("x", 0.9)),
    }
    report = calibrate(specs, retrievals)
    assert report["recommended_min_score"] == 0.47
    assert report["unanswerable_rejected_at_recommended"] == 1  # u1 (0.31) yes, u2 (0.55) left to the generator
    row = next(r for r in report["sweep"] if r["threshold"] == 0.5)
    assert row == {"threshold": 0.5, "answerable_kept": 1, "unanswerable_rejected": 1}
    assert report["questions"] == {"answerable": 2, "unanswerable": 2}


def test_calibration_makes_no_recommendation_when_evidence_is_missed():
    specs = [{"id": "a1", "category": "answerable", "expected_sections": ["B"], "expected_facts_any": [["80"]]}]
    report = calibrate(specs, {"a1": retrieval("q", passage("no fact here", 0.9))})
    assert report["recommended_min_score"] is None and report["retrieval_misses"] == ["a1"]


def test_facts_split_by_a_pdf_line_break_inside_a_hyphenated_word_are_found():
    assert _contains("during a TTCS, a Stop-\nand-Go Penalty will be imposed", "stop-and-go")
    assert not _contains("a stop and a go-kart", "stop-and-go")
    # Models often write non-breaking hyphens (U+2011).
    assert _contains("a Stop\u2011and\u2011Go Penalty", "stop-and-go")


# ------------------------------------------------------------------ command line


@pytest.mark.parametrize("value", ["0", "0.3", "1"])
def test_threshold_argument_accepts_a_similarity(value):
    assert min_score_argument(value) == float(value)


@pytest.mark.parametrize("value", ["1.5", "-0.1", "nan", "inf", "high"])
def test_an_invalid_threshold_is_rejected_before_anything_runs(value, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_fia_rag.py", "--threshold", value])
    monkeypatch.setattr(check_fia_rag, "get_fia_rag", lambda: pytest.fail("the retrieval phase started"))
    with pytest.raises(SystemExit) as exited:
        check_fia_rag.main()
    assert exited.value.code == 2 and "argument --threshold" in capsys.readouterr().err


def test_the_summary_counts_the_claim_verifier_calls(tmp_path, monkeypatch, capsys):
    rag, llm, qdrant = build_test_rag(tmp_path / "fia_docs")
    rag.settings = replace(rag.settings, generation=GenerationConfig(verify_claims=True))
    answer = cite_passage_containing("80km/h")
    llm.reply = lambda messages: '{"unsupported": []}' if messages[0].content == VERIFIER_PROMPT else answer(messages)
    rag.build_index()
    question = {
        "id": "pit-limit",
        "category": "answerable",
        "question": "What is the speed limit in the pit lane?",
        "expected_facts_any": [["80km/h"]],
    }
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [question]}))
    monkeypatch.setattr(check_fia_rag, "QUESTIONS_FILE", questions)
    monkeypatch.setattr(check_fia_rag, "get_fia_rag", lambda: rag)
    monkeypatch.setattr(sys, "argv", ["check_fia_rag.py", "--report", str(tmp_path / "report.json")])
    try:
        assert check_fia_rag.main() == 0
    finally:
        qdrant.close()
    assert len(llm.calls) == 2  # the answer and its claim verification
    assert "chat requests=2" in capsys.readouterr().out
