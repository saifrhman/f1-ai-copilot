"""Natural-language router: classification, tie-breaking, diagnostics and handler validation.

Regulatory questions are mostly checked through classification. The one handler test that
reaches the FIA RAG uses a real in-memory pipeline over generated PDFs with a deterministic
embedder and a scripted chat model, so no model provider is ever called.
"""

import base64
import io
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf
from qdrant_client import QdrantClient

import core_modules.rule_checker.fia_rag.pipeline as rag_pipeline
from core_modules.llm_query.natural_query import (
    MAX_LAP_SECONDS,
    MAX_QUERY_CHARS,
    MAX_SECTORS,
    MAX_TELEMETRY_LAPS,
    NaturalQueryProcessor,
    QueryType,
    _compile_term,
    classify_query,
    process_natural_query,
)
from core_modules.rule_checker.fia_rag import (
    DECLINE_ANSWER,
    ChunkingConfig,
    EmbeddingConfig,
    FIARegulationRAG,
    QdrantConfig,
    RAGSettings,
    RetrievalConfig,
)
from tests.helpers import HashingEmbeddings, ScriptedChatModel, fia_page, write_pdf
from tests.test_fia_index_retrieval import FUEL_FLOW, PIT_LANE, REAR_WING, UNSAFE_RELEASE

REGULATORY_QUESTIONS = [
    # Misroutes reported by the audit (performance / general / technical before the fix).
    "What is the pit lane speed limit rule?",
    "What is the penalty for speeding in the pit lane?",
    "Which regulation governs braking during a safety car?",
    "Is DRS allowed?",
    "When can a driver use DRS?",
    "What conditions govern use of DRS?",
    "What does Article 33 say?",
    "What does Article 33.4 say?",
    "How many power units can a driver use?",
    "Explain the cost cap",
    "What is the minimum car weight?",
    "What happens in parc fermé?",
    "Is following another car closely allowed?",
    "Is there enough space for another car at the apex?",
    # Already correct before; must stay correct.
    "What is the penalty for exceeding track limits?",
    "Which FIA regulation applies to an unsafe release?",
    "What does a double yellow flag mean?",
    "Who are the stewards at this event?",
    "What do the sporting regulations say about a red flag?",
    # Penalty names in their penalty sense, and the narrowed "article"/"breach" terms.
    "What is a stop-go penalty?",
    "Is it a stop and go penalty?",
    "When do you get a drive-through?",
    "What is a drive through penalty?",
    "Was Red Bull in breach of the cost cap?",
    "Which article covers unsafe release?",
    "What happens on the formation lap?",
]

STRATEGY_QUESTIONS = [
    "Should I pit following the safety car?",
    "Should we go for the undercut following Hamilton's stop?",
    "Which tyre strategy is fastest given my lap time?",
    "Should I pit now?",
    "When should I box for new tyres?",
    "Is a two-stop faster than a one-stop here?",
    "Is a one stop viable?",
    "How long should the next stint on this compound be?",
    "Can we rule out a two-stop?",
    # Review findings: "stop goes"/"article" used to trigger the regulatory cue.
    "If the stop goes wrong, should we switch to a two-stop?",
    "What if our stop goes long, is the undercut still on?",
    "I read an article on tyre strategy, is a two-stop better?",
    # Review findings: vocabulary gaps (general) and pit loss (regulatory) before the fix.
    "When should I change tyres?",
    "Which tyres should we start on?",
    "Should we start on the medium?",
    "Should we stay out?",
    "How much time do we lose in the pit lane?",
    "What is the pit lane time loss at Monza?",
    # Neutral phrases are claimed without scoring.
    "As a rule, should we pit early?",
    "What's the rule of thumb for the undercut?",
    "It rules out a two-stop",
    "The two-stop was ruled out, what now?",
]

PERFORMANCE_QUESTIONS = [
    "Why was my lap time slower?",
    "Why did I lose time in sector 2?",
    "How is my throttle application?",
    "Is my pace improving?",
    "How do I drive through the chicane faster?",
    # Review findings: general before the fix.
    "Where do I lose the most time?",
    "Why am I losing more than a second?",
    "How was my last lap?",
    "Compare my laps",
    "Where am I slow?",
    "How can I improve my lap?",
    # A bare "hard" is an adjective here, not the hard compound.
    "Why did I brake so hard in sector 2?",
]

TECHNICAL_QUESTIONS = [
    "What ride height and rear wing angle should I run?",
    "Recommend a car set-up for Monza",
    "Should I move the brake bias forward?",
    # Review findings: car-handling uses of "nervous"/"upset" went to emotion.
    "Why does the rear feel nervous and oversteer?",
    "The car gets upset under braking",
    "Why is the car so nervous on corner entry?",
    "The car is upset over the kerbs, what should I change?",
]

EMOTION_QUESTIONS = [
    "How was the driver's radio during the safety car?",
    "How was the driver’s radio during the safety car?",
    "Is the driver frustrated on the radio?",
    "What emotion is in this clip?",
    "Is the driver nervous?",
    "Does the driver sound upset on the radio?",
]

GENERAL_QUESTIONS = [
    "Is the Mafia involved?",
    "Who won the 1988 championship?",
    "Is the space big enough?",
    "Is the gearbox reliable?",
    "Is the track radioactive?",
    "Show me the pitch",
    "Gear box issue",
    "Can I drive through turn 3 flat out?",
]


def _classify(query, context=None):
    return NaturalQueryProcessor()._classify_query(query, context)


def _all_matched_terms(decision):
    return {term for terms in decision.matched_terms.values() for term in terms}


@pytest.mark.parametrize("query", REGULATORY_QUESTIONS)
def test_regulatory_questions_route_to_fia_rag(query):
    assert _classify(query) is QueryType.REGULATORY


@pytest.mark.parametrize("query", STRATEGY_QUESTIONS)
def test_strategy_questions_route_to_strategy(query):
    assert _classify(query) is QueryType.STRATEGY


@pytest.mark.parametrize("query", PERFORMANCE_QUESTIONS)
def test_performance_questions_route_to_performance(query):
    assert _classify(query) is QueryType.PERFORMANCE


@pytest.mark.parametrize("query", TECHNICAL_QUESTIONS)
def test_setup_questions_route_to_technical(query):
    assert _classify(query) is QueryType.TECHNICAL


@pytest.mark.parametrize("query", EMOTION_QUESTIONS)
def test_emotion_questions_route_to_emotion(query):
    assert _classify(query) is QueryType.EMOTION


@pytest.mark.parametrize("query", GENERAL_QUESTIONS)
def test_unrelated_questions_route_to_general(query):
    decision = classify_query(query)
    assert decision.query_type is QueryType.GENERAL
    assert decision.matched_terms == {}
    assert decision.decision_rule == "no_vocabulary_match"


@pytest.mark.parametrize(
    "query, forbidden_term",
    [
        ("Should I pit following the safety car?", "wing"),
        ("Is there enough space for another car at the apex?", "pace"),
        ("Is the Mafia involved?", "fia"),
        ("Is the gearbox reliable?", "box"),
        ("Is the track radioactive?", "radio"),
        ("Show me the pitch", "pit"),
        ("What is the pit lane speed limit rule?", "speed"),
        ("What is the pit lane speed limit rule?", "pit"),
        ("Can we rule out a two-stop?", "rule"),
    ],
)
def test_terms_match_whole_words_and_longest_phrases_only(query, forbidden_term):
    assert forbidden_term not in _all_matched_terms(classify_query(query))


def test_phrase_variants_match_the_same_term():
    for variant in ("one-stop", "one stop", "One-Stop"):
        assert classify_query(f"Is a {variant} possible?").matched_terms == {"strategy": ["one stop"]}
    assert classify_query("parc fermé rules?").matched_terms["regulatory"] == ["parc ferme", "rule"]


def test_regulatory_cue_outranks_performance_words_even_with_telemetry():
    telemetry_context = {"telemetry": {"lap_times": [80.0, 80.4], "braking_consistency": 0.8}}
    decision = classify_query("Which regulation governs braking during a safety car?", telemetry_context)
    assert decision.query_type is QueryType.REGULATORY
    assert decision.decision_rule == "regulatory_cue"
    assert decision.scores["performance"] > 0  # the performance word was seen, and outranked

    speed_rule = classify_query("What is the pit lane speed limit rule?", telemetry_context)
    assert speed_rule.query_type is QueryType.REGULATORY
    assert "performance" not in speed_rule.matched_terms


def test_tie_with_regulatory_goes_to_regulatory_regardless_of_context():
    for context in (None, {"telemetry": {"lap_times": [80.0]}}):
        decision = classify_query("Did DRS help my lap time?", context)
        assert decision.scores["regulatory"] == decision.scores["performance"]
        assert decision.query_type is QueryType.REGULATORY
        assert decision.decision_rule == "tie_regulatory_priority"


def test_supplied_context_breaks_ties_between_non_regulatory_types():
    query = "Which tyre strategy is fastest given my lap time?"
    without = classify_query(query)
    assert without.scores["strategy"] == without.scores["performance"]
    assert without.query_type is QueryType.STRATEGY
    assert without.decision_rule == "tie_priority_order"
    assert without.tied_types == ["strategy", "performance"]

    with_telemetry = classify_query(query, {"telemetry": {"lap_times": [80.0]}})
    assert with_telemetry.query_type is QueryType.PERFORMANCE
    assert with_telemetry.decision_rule == "tie_context_supplied"


def test_highest_score_wins_without_a_tie():
    decision = classify_query("How was the driver's radio during the safety car?")
    assert decision.scores["emotion"] > decision.scores["strategy"] > 0
    assert decision.decision_rule == "highest_score"


@pytest.mark.parametrize(
    "query, winner, tied",
    [
        ("Is downforce important for the stint?", "strategy", ["strategy", "technical"]),
        ("Is the driver calm about the downforce?", "technical", ["technical", "emotion"]),
        ("Was I calm under braking?", "emotion", ["emotion", "performance"]),
    ],
)
def test_ties_without_context_follow_the_documented_priority(query, winner, tied):
    decision = classify_query(query)
    assert decision.query_type.value == winner
    assert decision.decision_rule == "tie_priority_order"
    assert decision.tied_types == tied


def test_emotion_performance_tie_goes_to_performance_when_telemetry_is_supplied():
    decision = classify_query("Was I calm under braking?", {"telemetry": {"lap_times": [80.0]}})
    assert decision.query_type is QueryType.PERFORMANCE
    assert decision.decision_rule == "tie_context_supplied"


def test_possessive_is_removed_before_phrase_matching():
    for query in ("How was the driver's radio?", "How was the driver’s radio?"):
        assert classify_query(query).matched_terms == {"emotion": ["driver radio"]}


@pytest.mark.parametrize(
    "query",
    [
        "If the stop goes wrong, should we switch to a two-stop?",
        "What if our stop goes long, is the undercut still on?",
        "Can I drive through turn 3 flat out?",
        "How do I drive through the chicane faster?",
        "I read an article on tyre strategy, is a two-stop better?",
        "Why does my lap time drop off when the tyres breach the cliff?",
        "The car gets upset under braking",
    ],
)
def test_everyday_words_are_not_regulatory_cues(query):
    decision = classify_query(query)
    assert decision.query_type is not QueryType.REGULATORY
    assert decision.decision_rule != "regulatory_cue"


def test_plural_suffix_never_creates_verb_forms():
    # "stop go" + "es" used to match the verb phrase "stop goes".
    assert classify_query("If the stop goes wrong").matched_terms == {"strategy": ["stop"]}
    assert classify_query("How many boxes are there?").matched_terms == {"strategy": ["box"]}
    assert classify_query("Which penalties apply?").matched_terms == {"regulatory": ["penalties"]}


def test_term_matching_rules():
    # Plural: "es" only after s/x/z/ch/sh, otherwise "s", so no term matches a verb form like "goes".
    assert _compile_term("box").search("two boxes")
    assert _compile_term("stop").search("two stops")
    assert _compile_term("stop go").search("the stop goes wrong") is None
    # A hyphen in a term is required; a space is optional.
    assert _compile_term("drive-through").search("a drive-through")
    assert _compile_term("drive-through").search("a drive/through")
    assert _compile_term("drive-through").search("drive through turn 3") is None
    assert _compile_term("one stop").search("a onestop race")


@pytest.mark.parametrize(
    "query, kind",
    [
        ("Is the driver nervous?", "emotion"),
        ("Is the driver upset?", "emotion"),
        ("Which article is that?", "regulatory"),
        ("Is that a breach of the code?", "regulatory"),
    ],
)
def test_ambiguous_words_carry_only_topic_weight(query, kind):
    decision = classify_query(query)
    assert decision.scores[kind] == 1
    assert decision.query_type.value == kind


def test_article_is_a_cue_only_with_a_number():
    numbered = classify_query("Does Article 33.4 cover my lap time?")
    assert numbered.query_type is QueryType.REGULATORY
    assert numbered.decision_rule == "regulatory_cue"
    assert numbered.matched_terms["regulatory"] == ["article <number>"]
    assert classify_query("I read an article on tyre strategy").scores["regulatory"] == 1


@pytest.mark.parametrize(
    "query",
    [
        "pit " * (MAX_QUERY_CHARS // 4 + 1),
        "ﷺ" * MAX_QUERY_CHARS,  # NFKD expands each character to 18
        None,
        42,
        b"pit stop",
    ],
    ids=["too-long", "nfkd-expansion", "none", "int", "bytes"],
)
def test_classify_query_rejects_invalid_or_oversized_queries(query):
    with pytest.raises(ValueError, match="query"):
        classify_query(query)


def test_classify_query_rejects_non_object_context():
    with pytest.raises(ValueError, match="context"):
        classify_query("Should I pit now?", ["telemetry"])


def test_classify_query_accepts_a_query_at_the_limit():
    decision = classify_query(("pit " * MAX_QUERY_CHARS)[:MAX_QUERY_CHARS])
    assert decision.query_type is QueryType.STRATEGY


def test_classification_is_deterministic_across_threads():
    queries = REGULATORY_QUESTIONS + STRATEGY_QUESTIONS + PERFORMANCE_QUESTIONS + EMOTION_QUESTIONS
    expected = [classify_query(q) for q in queries]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(5):
            assert list(pool.map(classify_query, queries)) == expected


def test_output_keeps_existing_keys_and_adds_json_serialisable_routing():
    out = process_natural_query("Why was my lap time slower?", {"telemetry": {"lap_times": [80.0, 80.4]}})
    assert set(out) == {"answer", "query_type", "confidence", "data_sources", "additional_context", "routing"}
    routing = out["routing"]
    assert routing["query_type"] == out["query_type"] == "performance"
    assert routing["matched_terms"] == {"performance": ["lap time", "slower"]}
    assert routing["scores"] == {"regulatory": 0, "strategy": 0, "technical": 0, "emotion": 0, "performance": 2}
    assert routing["decision_rule"] == "highest_score"
    assert "heuristic" in routing["method"]
    json.dumps(out)


def test_general_query_declines_with_diagnostics():
    out = process_natural_query("Who won the 1988 championship?")
    assert out["query_type"] == "general"
    assert out["confidence"] == 0.0
    assert out["data_sources"] == []
    assert out["routing"]["decision_rule"] == "no_vocabulary_match"


@pytest.mark.parametrize("query", ["", "   ", "a" * (MAX_QUERY_CHARS + 1), None, 42])
def test_invalid_query_is_rejected(query):
    with pytest.raises(ValueError):
        process_natural_query(query)


@pytest.mark.parametrize("context", ["telemetry", [1, 2], 5])
def test_non_object_context_is_rejected(context):
    with pytest.raises(ValueError, match="context"):
        process_natural_query("Why was my lap time slower?", context)


# ---------------------------------------------------------------------------
# Performance handler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lap_times",
    [
        "8080",
        80.0,
        {"1": 80.0},
        [],
        [80.0, None],
        [80.0, float("nan")],
        [80.0, float("inf")],
        [80.0, -5.0],
        [80.0, 0.0],
        [80.0, 1e308],
        [True, 80.0],
        [80.0, "81.0"],
    ],
)
def test_invalid_lap_times_are_rejected(lap_times):
    with pytest.raises(ValueError, match="lap_times"):
        process_natural_query("Why was my lap time slower?", {"telemetry": {"lap_times": lap_times}})


@pytest.mark.parametrize(
    "telemetry, field",
    [
        ([80.0, 80.4], "telemetry"),
        ({"lap_times": [80.0], "braking_consistency": 1.5}, "braking_consistency"),
        ({"lap_times": [80.0], "braking_consistency": float("nan")}, "braking_consistency"),
        ({"lap_times": [80.0], "throttle_aggressiveness": "high"}, "throttle_aggressiveness"),
        ({"lap_times": [80.0], "sector_times": {"2": -1.0}}, "sector_times"),
        ({"lap_times": [80.0], "sector_times": {"x": 27.0}}, "sector_times"),
        ({"lap_times": [80.0], "sector_times": {"\u00b2": 27.0}}, "sector_times"),
        ({"lap_times": [80.0], "sector_times": "27.1"}, "sector_times"),
        ({"lap_times": [80.0], "sector_times": [27.1, float("nan")]}, "sector_times"),
    ],
)
def test_invalid_telemetry_fields_are_rejected(telemetry, field):
    with pytest.raises(ValueError, match=field):
        process_natural_query("Why did I lose time in sector 2 under braking?", {"telemetry": telemetry})


def test_performance_answer_only_restates_supplied_values():
    out = process_natural_query(
        "Why was my lap time slower?",
        {"telemetry": {"lap_times": [80.0, 80.4], "braking_consistency": 0.8}},
    )
    assert out["query_type"] == "performance"
    assert out["data_sources"] == ["telemetry"]
    assert "80.400" in out["answer"]
    assert out["additional_context"]["evidence"] == {"lap_times": [80.0, 80.4], "braking_consistency": 0.8}
    numbers = {float(n) for n in re.findall(r"[-+]?\d+\.\d+", out["answer"])}
    assert all(any(math.isclose(n, v, abs_tol=1e-9) for v in (80.0, 80.4, 0.4, 0.8)) for n in numbers)
    assert out["confidence"] == 0.8


@pytest.mark.parametrize(
    "sector_times",
    [{"1": 25.0, "2": 27.1, "3": 28.3}, [25.0, 27.1, 28.3], {2: 27.1}],
)
def test_missing_sector_is_reported_and_lowers_confidence(sector_times):
    query = "Why did I lose time in sector 2?"
    missing = process_natural_query(query, {"telemetry": {"lap_times": [80.0, 80.4]}})
    supplied = process_natural_query(query, {"telemetry": {"lap_times": [80.0, 80.4], "sector_times": sector_times}})

    assert missing["query_type"] == supplied["query_type"] == "performance"
    assert "Sector 2 time was not supplied" in missing["answer"]
    assert missing["additional_context"]["not_supplied"] == ["sector_2"]
    assert "sector_2" not in missing["additional_context"]["evidence"]
    assert missing["confidence"] == 0.0

    assert "Supplied Sector 2 time: 27.100s." in supplied["answer"]
    assert supplied["additional_context"]["evidence"]["sector_2"] == 27.1
    assert supplied["additional_context"]["not_supplied"] == []
    assert supplied["confidence"] == 0.8 > missing["confidence"]


def test_partially_answered_question_gets_partial_confidence():
    out = process_natural_query(
        "Was my lap time slower because of braking?",
        {"telemetry": {"lap_times": [80.0, 80.4]}},
    )
    assert out["additional_context"]["requested"] == ["lap_times", "braking_consistency"]
    assert out["additional_context"]["not_supplied"] == ["braking_consistency"]
    assert "Braking consistency was not supplied" in out["answer"]
    assert out["confidence"] == pytest.approx(0.4)


def test_speed_questions_are_not_answered_from_other_fields():
    out = process_natural_query("What was my top speed?", {"telemetry": {"lap_times": [80.0, 80.4]}})
    assert out["query_type"] == "performance"
    assert "Speed data is not analysed" in out["answer"]
    assert out["confidence"] == 0.0


def test_performance_without_telemetry_declines():
    out = process_natural_query("Why was my lap time slower?")
    assert out["query_type"] == "performance"
    assert out["confidence"] == 0.0
    assert out["data_sources"] == []


def test_telemetry_without_supported_fields_declines():
    out = process_natural_query("Why was my lap time slower?", {"telemetry": {"rpm": [11000, 11500]}})
    assert "none of the supported fields" in out["answer"]
    assert out["confidence"] == 0.0
    assert out["data_sources"] == []


PARTIAL_TELEMETRY = {"lap_times": [80.0, 80.4], "sector_times": {"1": 25.0}}


@pytest.mark.parametrize(
    "query, extra_requested, not_supplied, phrase",
    [
        (
            "Compare sectors 1 and 2: where was I slower?",
            ["sector_1", "sector_2"],
            ["sector_2"],
            "Sector 2 time was not supplied",
        ),
        ("Why was I slower in sectors 1-3?", ["sector_1", "sector_2", "sector_3"], ["sector_2", "sector_3"], "Sector 3"),
        ("Why was I slower in sectors 1, 2, and 3?", ["sector_1", "sector_2", "sector_3"], ["sector_2", "sector_3"], "Sector 2"),
        ("Why was I slower in the second sector?", ["sector_2"], ["sector_2"], "Sector 2 time was not supplied"),
        ("Why was I slower in the middle sector?", ["sector_2"], ["sector_2"], "Sector 2 time was not supplied"),
        ("Why was I slower in the final sector?", ["sector_3"], ["sector_3"], "Sector 3 time was not supplied"),
        ("Why was I slower in S2?", ["sector_2"], ["sector_2"], "Sector 2 time was not supplied"),
        ("Why was I slower in turn 3?", ["turn_3"], ["turn_3"], "Turn 3 data was not supplied"),
        ("Why was I slower through turns 3 and 4?", ["turn_3", "turn_4"], ["turn_3", "turn_4"], "turn 4"),
        ("Why was I slower through the chicane?", ["corners"], ["corners"], "Corner-level data was not supplied"),
        ("Why am I slow on the straights?", ["straights"], ["straights"], "Straight-line data was not supplied"),
        ("Why was I slower than Verstappen?", ["comparison_target"], ["comparison_target"], "'verstappen'"),
        ("Was I slower than my teammate?", ["comparison_target"], ["comparison_target"], "'my teammate'"),
        ("Why is my lap slower vs the car ahead?", ["comparison_target"], ["comparison_target"], "another driver"),
    ],
)
def test_question_about_data_that_was_not_supplied_says_so_and_lowers_confidence(
    query, extra_requested, not_supplied, phrase
):
    out = process_natural_query(query, {"telemetry": PARTIAL_TELEMETRY})
    context = out["additional_context"]
    assert out["query_type"] == "performance"
    assert context["requested"] == ["lap_times"] + extra_requested
    assert context["not_supplied"] == not_supplied
    assert phrase in out["answer"]
    covered = 1 + len(extra_requested) - len(not_supplied)
    assert out["confidence"] == pytest.approx(round(0.8 * covered / (1 + len(extra_requested)), 3))
    assert out["confidence"] < 0.8
    assert not any(key in context["evidence"] for key in not_supplied)


def test_unsupported_aspects_are_listed_separately_from_unsupplied_fields():
    out = process_natural_query(
        "Why was I slower in sector 2 and turn 3 than Hamilton?", {"telemetry": PARTIAL_TELEMETRY}
    )
    context = out["additional_context"]
    assert context["not_supplied"] == ["sector_2", "turn_3", "comparison_target"]
    assert context["unsupported"] == ["turn_3", "comparison_target"]


@pytest.mark.parametrize(
    "query",
    [
        "Was I slower than my best lap?",
        "Was I slower than usual?",
        "Am I losing more than a second per lap?",
        "Was I slow rather than fast?",
    ],
)
def test_comparisons_within_the_supplied_laps_are_fully_covered(query):
    out = process_natural_query(query, {"telemetry": PARTIAL_TELEMETRY})
    assert out["additional_context"]["requested"] == ["lap_times"]
    assert out["additional_context"]["not_supplied"] == []
    assert out["confidence"] == 0.8


def test_supplied_sectors_named_by_list_or_short_form_are_restated():
    telemetry = {"lap_times": [80.0, 80.4], "sector_times": [25.0, 27.1, 28.3]}
    for query in ("Compare sectors 1 and 2", "S1 vs S2 pace?", "Why was the first sector faster than the second sector?"):
        out = process_natural_query(query, {"telemetry": telemetry})
        assert out["additional_context"]["evidence"]["sector_1"] == 25.0
        assert out["additional_context"]["evidence"]["sector_2"] == 27.1
        assert out["additional_context"]["not_supplied"] == []
        assert out["confidence"] == 0.8


LAP_TELEMETRY = {"lap_times": [91.2, 90.8, 95.4, 90.9, 90.1, 90.6]}


def test_laps_named_in_the_question_are_answered_from_the_supplied_lap_times():
    # Reviewer repro: this used to restate only the latest and best lap, at confidence 0.8.
    out = process_natural_query("Why was lap 3 so much slower than lap 5?", {"telemetry": LAP_TELEMETRY})
    context = out["additional_context"]

    assert out["query_type"] == "performance"
    assert context["requested"] == ["lap_times", "lap_3", "lap_5"]
    assert context["evidence"]["lap_3"] == 95.4 and context["evidence"]["lap_5"] == 90.1
    assert context["not_supplied"] == []
    assert "lap 3 95.400s; lap 5 90.100s" in out["answer"]
    assert "Lap 3 was 5.300s slower than lap 5." in out["answer"]
    assert out["answer"].index("lap 3 95.400s") < out["answer"].index("Latest lap")
    assert "N-th value of telemetry.lap_times" in context["lap_numbering"]
    assert out["confidence"] == 0.8


def test_named_lap_beyond_the_supplied_laps_is_not_supplied_and_lowers_confidence():
    out = process_natural_query("Why was lap 3 slower than lap 8?", {"telemetry": LAP_TELEMETRY})
    context = out["additional_context"]

    assert context["requested"] == ["lap_times", "lap_3", "lap_8"]
    assert context["not_supplied"] == ["lap_8"]
    assert "lap_8" not in context["evidence"] and context["evidence"]["lap_3"] == 95.4
    assert "Lap 8 time was not supplied: telemetry.lap_times holds 6 laps" in out["answer"]
    assert "slower than lap" not in out["answer"]  # nothing to compare lap 3 with
    assert out["confidence"] == pytest.approx(round(0.8 * 2 / 3, 3))


def test_named_lap_without_lap_times_is_not_supplied():
    out = process_natural_query("Why was lap 3 slow?", {"telemetry": {"sector_times": [25.0, 27.1, 28.3]}})
    assert out["additional_context"]["not_supplied"] == ["lap_times", "lap_3"]
    assert "Lap 3 time was not supplied: telemetry.lap_times was not supplied" in out["answer"]
    assert out["confidence"] == 0.0 and out["data_sources"] == []


@pytest.mark.parametrize(
    "query, laps, phrase",
    [
        ("Compare laps 2-4", [2, 3, 4], "Slowest of these: lap 3; fastest: lap 2; lap 3 was 4.600s slower than lap 2."),
        ("Laps 3, 4 and 5 vs lap #6?", [3, 4, 5, 6], "lap 3 was 5.300s slower than lap 5"),
        ("Was lap one faster than lap two?", [1, 2], "Lap 1 was 0.400s slower than lap 2."),
        ("Is my lap slower than lap 3?", [3], "lap 3 95.400s"),
        ("Were laps 4 and 6 equal?", [4, 6], "Lap 4 was 0.300s slower than lap 6."),
    ],
)
def test_lap_number_references_are_parsed_as_positions_in_lap_times(query, laps, phrase):
    out = process_natural_query(query, {"telemetry": LAP_TELEMETRY})
    assert out["additional_context"]["requested"] == ["lap_times"] + [f"lap_{n}" for n in laps]
    assert out["additional_context"]["not_supplied"] == []
    assert phrase in out["answer"]
    assert out["confidence"] == 0.8


@pytest.mark.parametrize(
    "query",
    [
        "Why was my lap time slower?",
        "What were my lap times?",
        "How were the last 3 laps?",
        "Was a lap 1:31.2 good?",
        "Was my lap 91.2s?",
    ],
)
def test_lap_times_and_lap_counts_are_not_lap_numbers(query):
    out = process_natural_query(query, {"telemetry": LAP_TELEMETRY})
    assert out["additional_context"]["requested"] == ["lap_times"]
    assert "lap_numbering" not in out["additional_context"]
    assert out["answer"].startswith("Latest lap: 90.600s")
    assert out["confidence"] == 0.8


def test_lap_count_limit_is_enforced():
    ok = process_natural_query("How consistent is my pace?", {"telemetry": {"lap_times": [80.0] * MAX_TELEMETRY_LAPS}})
    assert ok["additional_context"]["evidence"]["lap_times"] == [80.0] * MAX_TELEMETRY_LAPS
    with pytest.raises(ValueError, match=f"at most {MAX_TELEMETRY_LAPS} laps"):
        process_natural_query("How consistent is my pace?", {"telemetry": {"lap_times": [80.0] * (MAX_TELEMETRY_LAPS + 1)}})


def test_lap_time_upper_bound_is_inclusive():
    out = process_natural_query("Why was my lap time slower?", {"telemetry": {"lap_times": [80.0, MAX_LAP_SECONDS]}})
    assert out["additional_context"]["evidence"]["lap_times"] == [80.0, MAX_LAP_SECONDS]
    with pytest.raises(ValueError, match="lap_times"):
        process_natural_query("Why was my lap time slower?", {"telemetry": {"lap_times": [80.0, MAX_LAP_SECONDS + 0.001]}})


@pytest.mark.parametrize(
    "sector_times, message",
    [
        ([25.0] * (MAX_SECTORS + 1), f"at most {MAX_SECTORS} sectors"),
        ({"0": 25.0}, "between 1 and"),
        ({str(MAX_SECTORS + 1): 25.0}, "between 1 and"),
        ({"1": 25.0, "01": 26.0}, "more than once"),
        ({1: 25.0, "1": 26.0}, "more than once"),
    ],
)
def test_sector_count_range_and_duplicates_are_rejected(sector_times, message):
    with pytest.raises(ValueError, match=message):
        process_natural_query("Why was my lap time slower?", {"telemetry": {"lap_times": [80.0], "sector_times": sector_times}})


def test_maximum_number_of_sectors_is_accepted():
    sectors = [20.0 + index for index in range(MAX_SECTORS)]
    out = process_natural_query(f"Why was sector {MAX_SECTORS} slower?", {"telemetry": {"lap_times": [80.0], "sector_times": sectors}})
    assert out["additional_context"]["evidence"][f"sector_{MAX_SECTORS}"] == sectors[-1]


def test_lap_question_with_only_sector_data_names_the_missing_field():
    out = process_natural_query("Why was my lap time slower?", {"telemetry": {"sector_times": [25.0, 27.1, 28.3]}})
    assert "Lap times were not supplied" in out["answer"]
    assert "none of the supported fields" not in out["answer"]
    assert out["additional_context"]["not_supplied"] == ["lap_times"]
    assert out["confidence"] == 0.0
    assert out["data_sources"] == []


# ---------------------------------------------------------------------------
# Emotion handler
# ---------------------------------------------------------------------------


def _sine_wave(sample_rate=22050, duration=1.0):
    samples = np.arange(int(sample_rate * duration)) / sample_rate
    return 0.15 * np.sin(2 * math.pi * 220 * samples), sample_rate


def _wav_base64():
    waveform, sample_rate = _sine_wave()
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_emotion_query_classifies_real_base64_audio():
    from core_modules.driver_emotion.emotion_classifier import EmotionType

    out = process_natural_query("What emotion is in this driver radio clip?", {"audio_file": _wav_base64()})
    assert out["query_type"] == "emotion"
    assert out["data_sources"] == ["driver_emotion"]
    emotion = out["additional_context"]["emotion"]
    assert emotion in {member.value for member in EmotionType}
    assert f"'{emotion}'" in out["answer"]
    assert "heuristic" in out["answer"]
    assert 0.0 <= out["confidence"] <= 1.0


def test_emotion_query_refuses_server_file_paths(tmp_path):
    waveform, sample_rate = _sine_wave()
    path = tmp_path / "radio.wav"
    sf.write(path, waveform, sample_rate)
    with pytest.raises(ValueError):
        process_natural_query("What emotion is in this driver radio clip?", {"audio_file": str(path)})


@pytest.mark.parametrize(
    "context, field",
    [
        ({"audio_file": 123}, "audio_file"),
        ({"audio_file": ["UklGRg=="]}, "audio_file"),
        ({"audio_file": "UklGRg==", "transcribe": "yes"}, "transcribe"),
    ],
)
def test_emotion_query_rejects_malformed_context(context, field):
    with pytest.raises(ValueError, match=field):
        process_natural_query("What emotion is in this driver radio clip?", context)


@pytest.mark.parametrize("context", [None, {"audio_file": ""}, {"audio_file": "   "}])
def test_emotion_query_without_audio_declines(context):
    out = process_natural_query("What emotion is in this driver radio clip?", context)
    assert out["query_type"] == "emotion"
    assert out["confidence"] == 0.0
    assert out["data_sources"] == []


def test_emotion_query_forwards_the_transcribe_flag(monkeypatch):
    from core_modules.driver_emotion import emotion_classifier

    # Hermetic: "import whisper" fails, so a forwarded request is reported as unavailable.
    monkeypatch.setitem(sys.modules, "whisper", None)
    monkeypatch.setattr(emotion_classifier, "_transcriber", None)
    query, audio = "What emotion is in this driver radio clip?", _wav_base64()

    requested = process_natural_query(query, {"audio_file": audio, "transcribe": True})["additional_context"]
    assert requested["transcription_status"] in {"unavailable", "completed"}
    assert requested["transcription_status"] == "unavailable"
    assert "not installed" in requested["transcription_unavailable_reason"]

    for context in ({"audio_file": audio}, {"audio_file": audio, "transcribe": False}):
        assert process_natural_query(query, context)["additional_context"]["transcription_status"] == "not_requested"


# ---------------------------------------------------------------------------
# Regulatory handler
# ---------------------------------------------------------------------------


@pytest.fixture
def declining_rag(tmp_path, monkeypatch):
    """A real in-memory FIA RAG over generated PDFs whose (scripted) model declines to answer."""

    folder = tmp_path / "fia_docs"
    write_pdf(folder / "section_b_sporting.pdf", [fia_page(1, PIT_LANE), None, fia_page(3, UNSAFE_RELEASE)])
    write_pdf(folder / "section_c_technical.pdf", [FUEL_FLOW, REAR_WING])
    settings = RAGSettings(
        docs_path=folder,
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=40),
        embedding=EmbeddingConfig(model="hashing-test-512", batch_size=8),
        retrieval=RetrievalConfig(top_k=4, min_score=0.2),
        qdrant=QdrantConfig(collection="router_test"),
    )
    llm = ScriptedChatModel("INSUFFICIENT_EVIDENCE")
    qdrant = QdrantClient(":memory:")
    rag = FIARegulationRAG(settings, embeddings=HashingEmbeddings(), llm=llm, qdrant_client=qdrant)
    rag.build_index()
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    yield llm
    qdrant.close()


def test_regulatory_query_declined_by_the_rag_claims_no_source_and_no_confidence(declining_rag):
    out = process_natural_query("What is the pit lane speed limit rule?")
    context = out["additional_context"]

    assert out["query_type"] == "regulatory"
    # Passages were retrieved and the model was asked, but it declined ...
    assert len(declining_rag.calls) == 1
    assert context["retrieved_passages"] and context["top_retrieval_score"] > 0.2
    assert context["grounded"] is False and context["decline_reason"]
    assert context["citations"] == []
    # ... so the router must not present the regulations as a source or claim any confidence.
    assert out["answer"] == DECLINE_ANSWER
    assert out["data_sources"] == []
    assert out["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Technical (setup) handler
# ---------------------------------------------------------------------------

SETUP_CONTEXT = {
    "driver_preferences": {},
    "track_profile": {
        "track_name": "Silverstone Circuit", "track_length": 5891, "corners": 18, "high_speed_sections": 8,
        "low_speed_sections": 4, "track_type": "high_speed", "average_speed": 220, "downforce_requirement": 0.6,
    },
    "weather": {"condition": "dry", "temperature": 24},
    "n_trials": 16,
    "seed": 1,
}


def test_setup_query_names_the_driver_preferences_it_assumed():
    # Reviewer repro: driver_preferences {} silently used risk_tolerance=0.5 and tire_management=0.5.
    out = process_natural_query("What setup should I run?", SETUP_CONTEXT)
    assumed = ["driver_preferences.risk_tolerance=0.5", "driver_preferences.tire_management=0.5"]

    assert out["query_type"] == "technical"
    assert out["additional_context"]["assumed_defaults"] == assumed
    assert "defaults were assumed: " + ", ".join(assumed) in out["answer"]

    partial = {**SETUP_CONTEXT, "driver_preferences": {"risk_tolerance": 0.5}}
    assert process_natural_query("What setup should I run?", partial)["additional_context"]["assumed_defaults"] == assumed[1:]

    explicit = {**SETUP_CONTEXT, "driver_preferences": {"risk_tolerance": 0.5, "tire_management": 0.5}}
    out = process_natural_query("What setup should I run?", explicit)
    assert out["additional_context"]["assumed_defaults"] == []
    assert "assumed" not in out["answer"]
