"""Incident/penalty triage: input validation, normalisation, modifiers and the request model."""

import re

import pytest
from pydantic import ValidationError

from core_modules.rule_checker.penalty_predictor import (
    IncidentType,
    Intent,
    TrackCondition,
    predict_penalty,
)
from core_modules.rule_checker.schemas import PenaltyRequest

# Spelled out here (not imported) so the tests pin the documented behaviour.
DRIVING_CONDUCT = {"track_limits", "collision", "blocking", "dangerous_driving", "illegal_overtaking"}
CAR_TO_CAR = {"collision", "blocking", "illegal_overtaking"}


def _triage(incident="collision", condition="dry", intent="accidental", history=None):
    return predict_penalty(incident, condition, intent, history)


@pytest.mark.parametrize(
    "field, overrides",
    [
        ("track_condition", {"condition": "sunny"}),
        ("track_condition", {"condition": "snow"}),
        ("track_condition", {"condition": ""}),
        ("track_condition", {"condition": None}),
        ("intent", {"intent": "malicious"}),
        ("intent", {"intent": "deliberate"}),
        ("intent", {"intent": "   "}),
        ("intent", {"intent": 3}),
        ("incident_type", {"incident": "crash"}),
        ("incident_type", {"incident": "x" * 10_000}),
    ],
)
def test_invalid_enum_values_raise_instead_of_becoming_unknown(field, overrides):
    with pytest.raises(ValueError, match=field):
        _triage(**overrides)


def test_unknown_is_an_explicit_allowed_value():
    result = _triage(condition="unknown", intent="unknown")
    assert result["inputs"]["track_condition"] == "unknown"
    assert result["inputs"]["intent"] == "unknown"
    assert result["confidence"] < _triage()["confidence"]


def test_all_three_enums_are_normalised_the_same_way_and_echoed():
    messy = _triage("  Collision ", " DRY ", "Racing Incident")
    assert messy == _triage("collision", "dry", "racing_incident")
    assert messy["inputs"] == {
        "incident_type": "collision",
        "track_condition": "dry",
        "intent": "racing_incident",
        "driver_history": None,
    }
    hyphenated = _triage("Unsafe-Release", "Intermediate", "RACING-INCIDENT")
    assert hyphenated["inputs"]["incident_type"] == "unsafe_release"
    assert hyphenated["inputs"]["track_condition"] == "intermediate"
    assert hyphenated["inputs"]["intent"] == "racing_incident"


def test_output_names_a_review_category_not_a_penalty():
    result = _triage("unsafe_release", "dry", "accidental")
    assert result["triage_category"] == "unsafe_release_review"
    assert "predicted_penalty" not in result
    assert result["referenced_rule"] is None
    assert result["method"] == "transparent heuristic triage"
    assert "not an FIA steward-decision predictor" in result["disclaimer"]
    assert "POST /api/fia/query" in result["disclaimer"]  # points to the regulation QA for evidence
    assert not re.search(r"article\s*\d", result["reasoning"], re.IGNORECASE)
    assert "heuristic" in result["confidence_basis"]


@pytest.mark.parametrize(
    "history",
    [
        {"foo": "bar"},
        {"recent_penalties": 5},
        {"recent_penalties": "three"},
        {"recent_penalties": [1, 2]},
        {"recent_penalties": [""]},
        {"recent_penalties": ["x"] * 51},
        {"total_penalties": "lots"},
        {"total_penalties": "10"},
        {"total_penalties": -1},
        {"total_penalties": True},
        {"total_penalties": 2.5},
        {"total_penalties": 10_001},
        {"recent_penalties": ["x" * 201]},
        {"recent_penalties": ["a", "b", "c"], "total_penalties": 0},
        {"recent_penalties": ["a", "b", "c"], "total_penalties": 2},
        ["recent"],
        "history",
    ],
)
def test_malformed_driver_history_is_rejected(history):
    with pytest.raises(ValueError, match="driver_history"):
        _triage(history=history)


def test_confidence_counts_only_validated_relevant_history():
    baseline = _triage()
    assert _triage(history={})["confidence"] == baseline["confidence"]
    assert _triage(history={"recent_penalties": None})["confidence"] == baseline["confidence"]

    with_history = _triage(history={"total_penalties": 0})
    assert with_history["confidence"] > baseline["confidence"]
    assert with_history["severity_score"] == baseline["severity_score"]
    assert with_history["confidence"] == pytest.approx(0.7)
    assert with_history["inputs"]["driver_history"] == {"total_penalties": 0}


def test_history_note_length_limit_is_inclusive():
    note = "x" * 200
    assert _triage(history={"recent_penalties": [note]})["inputs"]["driver_history"] == {"recent_penalties": [note]}


def test_history_thresholds_change_severity():
    base = _triage()["severity_score"]
    assert _triage(history={"recent_penalties": ["a", "b"]})["severity_score"] == base
    assert _triage(history={"recent_penalties": ["a", "b", "c"]})["severity_score"] == pytest.approx(base + 0.08)
    assert _triage(history={"total_penalties": 9})["severity_score"] == base
    assert _triage(history={"total_penalties": 10})["severity_score"] == pytest.approx(base + 0.05)


def test_both_history_thresholds_apply_independently():
    base = _triage()["severity_score"]
    both = _triage(history={"recent_penalties": ["a", "b", "c"], "total_penalties": 10})
    assert both["severity_score"] == pytest.approx(base + 0.08 + 0.05)
    assert [item["source"] for item in both["severity_adjustments"]] == [
        "driver_history.recent_penalties",
        "driver_history.total_penalties",
    ]
    # A total equal to the recent count is consistent.
    assert _triage(history={"recent_penalties": ["a", "b", "c"], "total_penalties": 3})["severity_score"] == pytest.approx(
        base + 0.08
    )


@pytest.mark.parametrize("incident", [member.value for member in IncidentType])
def test_intent_modifiers_apply_only_where_they_make_sense(incident):
    accidental = _triage(incident, intent="accidental")["severity_score"]
    intentional = _triage(incident, intent="intentional")["severity_score"]
    racing = _triage(incident, intent="racing_incident")["severity_score"]

    if incident in DRIVING_CONDUCT:
        assert intentional > accidental
    else:
        assert intentional == accidental
    if incident in CAR_TO_CAR:
        assert racing < accidental
    else:
        assert racing == accidental


def test_track_condition_never_changes_severity_and_is_noted_only_for_driving():
    for incident in IncidentType:
        dry = _triage(incident.value, "dry")
        wet = _triage(incident.value, "wet")
        assert wet["severity_score"] == dry["severity_score"]
        assert ("Reduced-grip" in wet["reasoning"]) == (incident.value in DRIVING_CONDUCT)

    # Audit case: previously 0.72 with a racing-incident discount and a grip note.
    technical = _triage("technical_infringement", "wet", "racing_incident")
    assert technical["severity_score"] == pytest.approx(0.80)
    assert technical["severity_adjustments"] == []


def test_track_condition_never_changes_confidence():
    # Track condition has no effect on severity, so it must not raise "confidence" either.
    for incident in IncidentType:
        results = [_triage(incident.value, condition.value) for condition in TrackCondition]
        assert len({result["confidence"] for result in results}) == 1
        assert all("track_condition" not in result["confidence_inputs"] for result in results)


def test_confidence_inputs_are_the_inputs_that_can_change_severity():
    driving = _triage("collision", "wet", "unknown", {"total_penalties": 1})
    assert driving["confidence_inputs"] == {"driver_history": True, "intent": False}
    assert driving["confidence"] == pytest.approx(0.6)
    technical = _triage("technical_infringement", "unknown", "intentional")
    assert technical["confidence_inputs"] == {"driver_history": False}
    assert technical["confidence"] == pytest.approx(0.5)


def test_confidence_ignores_inputs_irrelevant_to_the_incident():
    assert (
        _triage("technical_infringement", "unknown", "unknown")["confidence"]
        == _triage("technical_infringement", "dry", "accidental")["confidence"]
    )
    assert _triage("collision", "unknown", "unknown")["confidence"] < _triage("collision", "dry", "accidental")["confidence"]
    for incident in IncidentType:
        full = _triage(incident.value, history={"total_penalties": 1})
        assert 0.5 <= full["confidence"] <= 0.7


def test_severity_is_clamped_and_banded():
    worst = _triage("dangerous_driving", "dry", "intentional", {"recent_penalties": ["a", "b", "c"]})
    assert worst["severity_score"] == 1.0
    assert worst["severity_band"] == "high"
    assert _triage("track_limits")["severity_band"] == "low"
    assert _triage("collision")["severity_band"] == "medium"


@pytest.mark.parametrize(
    "incident, intent, score, band",
    [
        ("blocking", "accidental", 0.50, "medium"),  # lower edge of "medium"
        ("speeding_in_pit", "accidental", 0.45, "low"),
        ("blocking", "racing_incident", 0.42, "low"),
        ("technical_infringement", "accidental", 0.80, "high"),  # lower edge of "high"
        ("illegal_overtaking", "intentional", 0.75, "medium"),
    ],
)
def test_severity_band_boundaries(incident, intent, score, band):
    result = _triage(incident, "dry", intent)
    assert result["severity_score"] == pytest.approx(score)
    assert result["severity_band"] == band


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


def test_penalty_request_normalises_like_the_module():
    body = {
        "incident_type": " Unsafe-Release ",
        "track_condition": "WET",
        "intent": "racing incident",
        "driver_history": {"recent_penalties": [" grid drop "], "total_penalties": 2},
    }
    request = PenaltyRequest.model_validate(body)
    kwargs = request.to_predictor_kwargs()
    assert kwargs == {
        "incident_type": "unsafe_release",
        "track_condition": "wet",
        "intent": "racing_incident",
        "driver_history": {"recent_penalties": ["grid drop"], "total_penalties": 2},
    }
    assert predict_penalty(**kwargs) == predict_penalty(
        body["incident_type"], body["track_condition"], body["intent"], body["driver_history"]
    )


def test_penalty_request_without_history():
    request = PenaltyRequest.model_validate({"incident_type": "collision", "track_condition": "dry", "intent": "unknown"})
    assert request.to_predictor_kwargs()["driver_history"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"track_condition": "snow"},
        {"intent": "deliberate"},
        {"incident_type": "crash"},
        {"extra": 1},
        {"driver_history": {"foo": 1}},
        {"driver_history": {"total_penalties": "10"}},
        {"driver_history": {"total_penalties": 2.0}},
        {"driver_history": {"total_penalties": -1}},
        {"driver_history": {"recent_penalties": ["x"] * 51}},
        {"driver_history": {"recent_penalties": ["  "]}},
        {"driver_history": {"recent_penalties": ["x" * 201]}},
        {"driver_history": {"recent_penalties": ["a", "b", "c"], "total_penalties": 2}},
    ],
)
def test_penalty_request_rejects_invalid_bodies(overrides):
    body = {"incident_type": "collision", "track_condition": "dry", "intent": "accidental", **overrides}
    with pytest.raises(ValidationError):
        PenaltyRequest.model_validate(body)


def test_penalty_request_requires_all_enum_fields():
    with pytest.raises(ValidationError):
        PenaltyRequest.model_validate({"incident_type": "collision", "track_condition": "dry"})


def test_penalty_request_schema_exposes_module_enum_values():
    schema = PenaltyRequest.model_json_schema()
    definitions = schema["$defs"]
    assert definitions["IncidentType"]["enum"] == [member.value for member in IncidentType]
    assert definitions["TrackCondition"]["enum"] == [member.value for member in TrackCondition]
    assert definitions["Intent"]["enum"] == [member.value for member in Intent]
    assert "unknown" in definitions["TrackCondition"]["enum"]
    assert set(schema["required"]) == {"incident_type", "track_condition", "intent"}
    assert all(schema["properties"][name].get("description") for name in schema["properties"])
