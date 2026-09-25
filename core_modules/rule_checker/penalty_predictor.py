#!/usr/bin/env python3
"""Transparent incident/penalty triage for the F1 AI Copilot demo.

This module deliberately does not claim to reproduce steward decisions or cite
hard-coded FIA article numbers. Exact regulatory questions belong to the FIA RAG
module, which can ground answers in the current source documents.

Everything here is a documented heuristic:

* Each incident type has a base severity and a review category
  (``triage_category``), not a predicted sanction.
* Intent modifiers apply only where intent plausibly changes how serious a
  case is: "intentional" (+0.15) for on-track driving-conduct incidents, and
  "racing_incident" (-0.08) only for car-to-car interactions. Pit-procedure
  and technical cases are judged on measured compliance, so intent does not
  change their triage.
* Track condition never changes severity or confidence. For on-track driving
  incidents, reduced grip is recorded as context in the reasoning.
* Driver history counts only through two validated fields, and both
  adjustments apply independently (together at most +0.13):
  ``recent_penalties`` (three or more gives +0.08, a recent pattern) and
  ``total_penalties`` (ten or more gives +0.05, the long-term record). A
  ``total_penalties`` smaller than the number of ``recent_penalties`` is
  contradictory and rejected.
* Confidence measures input completeness: 0.5 plus 0.2 times the share of the
  inputs that can change this incident type's severity and were supplied
  (``confidence_inputs``: driver history always, intent for on-track driving
  incidents). It is not a probability of any steward decision.
"""

import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Type, TypeVar


class IncidentType(str, Enum):
    TRACK_LIMITS = "track_limits"
    UNSAFE_RELEASE = "unsafe_release"
    COLLISION = "collision"
    BLOCKING = "blocking"
    DANGEROUS_DRIVING = "dangerous_driving"
    TECHNICAL_INFRINGEMENT = "technical_infringement"
    SPEEDING_IN_PIT = "speeding_in_pit"
    ILLEGAL_OVERTAKING = "illegal_overtaking"


class TrackCondition(str, Enum):
    DRY = "dry"
    WET = "wet"
    INTERMEDIATE = "intermediate"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Intent(str, Enum):
    ACCIDENTAL = "accidental"
    INTENTIONAL = "intentional"
    RACING_INCIDENT = "racing_incident"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IncidentProfile:
    outcome_category: str
    base_severity: float
    rationale: str


PROFILES: Dict[IncidentType, IncidentProfile] = {
    IncidentType.TRACK_LIMITS: IncidentProfile(
        "track_limits_review",
        0.35,
        "Track-limit outcomes depend on the current regulation, number/nature of infringements and session context.",
    ),
    IncidentType.UNSAFE_RELEASE: IncidentProfile(
        "unsafe_release_review",
        0.60,
        "Unsafe-release assessment depends on danger/impeding, responsibility and the applicable current sporting rule.",
    ),
    IncidentType.COLLISION: IncidentProfile(
        "driving_incident_review",
        0.55,
        "Contact incidents require steward assessment of responsibility, consequence and mitigating circumstances.",
    ),
    IncidentType.BLOCKING: IncidentProfile(
        "impeding_review",
        0.50,
        "Impeding/blocking is context-sensitive and should be checked against the session-specific sporting rules and evidence.",
    ),
    IncidentType.DANGEROUS_DRIVING: IncidentProfile(
        "serious_driving_incident_review",
        0.85,
        "Potentially dangerous driving warrants high-priority steward review and may lead to severe sporting consequences.",
    ),
    IncidentType.TECHNICAL_INFRINGEMENT: IncidentProfile(
        "technical_compliance_review",
        0.80,
        "Technical infringements depend on the exact failed requirement and any applicable exception or tolerance.",
    ),
    IncidentType.SPEEDING_IN_PIT: IncidentProfile(
        "pit_lane_speed_review",
        0.45,
        "Pit-lane speed cases depend on the measured excess, session and current event/regulatory provisions.",
    ),
    IncidentType.ILLEGAL_OVERTAKING: IncidentProfile(
        "overtaking_review",
        0.60,
        "Overtaking legality depends on flags, safety-car/VSC state, track position and the current sporting regulations.",
    ),
}

# On-track driving conduct: intent ("intentional") and grip context are relevant.
DRIVING_CONDUCT: FrozenSet[IncidentType] = frozenset({
    IncidentType.TRACK_LIMITS,
    IncidentType.COLLISION,
    IncidentType.BLOCKING,
    IncidentType.DANGEROUS_DRIVING,
    IncidentType.ILLEGAL_OVERTAKING,
})
# Car-to-car interactions: the only cases a "racing incident" description can soften.
CAR_TO_CAR: FrozenSet[IncidentType] = frozenset({
    IncidentType.COLLISION,
    IncidentType.BLOCKING,
    IncidentType.ILLEGAL_OVERTAKING,
})
REDUCED_GRIP: FrozenSet[TrackCondition] = frozenset({TrackCondition.WET, TrackCondition.INTERMEDIATE, TrackCondition.MIXED})

INTENTIONAL_DELTA = 0.15
RACING_INCIDENT_DELTA = -0.08
RECENT_PENALTIES_THRESHOLD = 3
RECENT_PENALTIES_DELTA = 0.08
TOTAL_PENALTIES_THRESHOLD = 10
TOTAL_PENALTIES_DELTA = 0.05

MAX_CHOICE_CHARS = 64
MAX_RECENT_PENALTIES = 50
MAX_PENALTY_NOTE_CHARS = 200
MAX_TOTAL_PENALTIES = 1000
DRIVER_HISTORY_FIELDS = ("recent_penalties", "total_penalties")

METHOD = "transparent heuristic triage"
CONFIDENCE_BASIS = (
    "input-completeness heuristic: 0.5 + 0.2 x share of the inputs that can change this incident type's "
    "severity (confidence_inputs) that were supplied; track condition is context only and not counted; "
    "not a probability of any steward decision"
)
DISCLAIMER = (
    "This is not an FIA steward-decision predictor and is not a legal/regulatory determination. "
    "For evidence from the current regulations, ask the FIA regulation QA (POST /api/fia/query, or the "
    "FIA regulations page of the web UI)."
)

_Choice = TypeVar("_Choice", IncidentType, TrackCondition, Intent)


def normalise_token(value: str) -> str:
    """'  Racing Incident ' / 'racing-incident' -> 'racing_incident'."""

    return re.sub(r"[\s\-]+", "_", value.strip().lower())


def parse_choice(value: Any, enum_cls: Type[_Choice], field: str) -> _Choice:
    """Normalise case/whitespace/hyphens and map to ``enum_cls``; ValueError otherwise."""

    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    allowed = ", ".join(member.value for member in enum_cls)
    token = normalise_token(value) if len(value) <= MAX_CHOICE_CHARS else ""
    try:
        return enum_cls(token)
    except ValueError as exc:
        shown = value if len(value) <= MAX_CHOICE_CHARS else value[:MAX_CHOICE_CHARS] + "..."
        hint = "; use 'unknown' when it is not known" if "unknown" in allowed else ""
        raise ValueError(f"Unknown {field} {shown!r}; expected one of: {allowed}{hint}") from exc


def validate_driver_history(history: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the validated relevant history fields, or None if nothing relevant was supplied."""

    if history is None:
        return None
    if not isinstance(history, dict):
        raise ValueError("driver_history must be an object")
    unknown = sorted(str(key) for key in history if key not in DRIVER_HISTORY_FIELDS)
    if unknown:
        raise ValueError(
            f"Unsupported driver_history field(s): {', '.join(unknown)}; "
            f"supported fields: {', '.join(DRIVER_HISTORY_FIELDS)}"
        )

    validated: Dict[str, Any] = {}
    recent = history.get("recent_penalties")
    if recent is not None:
        if not isinstance(recent, list):
            raise ValueError("driver_history.recent_penalties must be a list of penalty descriptions")
        if len(recent) > MAX_RECENT_PENALTIES:
            raise ValueError(f"driver_history.recent_penalties may contain at most {MAX_RECENT_PENALTIES} entries")
        notes: List[str] = []
        for index, note in enumerate(recent):
            if not isinstance(note, str) or not note.strip():
                raise ValueError(f"driver_history.recent_penalties[{index}] must be a non-empty string")
            if len(note.strip()) > MAX_PENALTY_NOTE_CHARS:
                raise ValueError(
                    f"driver_history.recent_penalties[{index}] must be at most {MAX_PENALTY_NOTE_CHARS} characters"
                )
            notes.append(note.strip())
        validated["recent_penalties"] = notes

    total = history.get("total_penalties")
    if total is not None:
        if isinstance(total, bool) or not isinstance(total, int):
            raise ValueError("driver_history.total_penalties must be an integer")
        if not 0 <= total <= MAX_TOTAL_PENALTIES:
            raise ValueError(f"driver_history.total_penalties must be between 0 and {MAX_TOTAL_PENALTIES}")
        validated["total_penalties"] = total

    check_history_consistency(validated.get("recent_penalties"), validated.get("total_penalties"))
    return validated or None


def check_history_consistency(recent: Optional[List[Any]], total: Optional[int]) -> None:
    """Reject a total that is smaller than the number of recent penalties (both supplied)."""

    if recent is not None and total is not None and total < len(recent):
        raise ValueError(
            f"driver_history.total_penalties ({total}) cannot be smaller than the number of "
            f"recent_penalties ({len(recent)})"
        )


def _severity_band(severity: float) -> str:
    if severity >= 0.80:
        return "high"
    if severity >= 0.50:
        return "medium"
    return "low"


class PenaltyPredictor:
    """Deterministic triage estimator, not a trained steward-decision model. Stateless and thread-safe."""

    VALID_CONDITIONS = frozenset(member.value for member in TrackCondition)
    VALID_INTENTS = frozenset(member.value for member in Intent)

    def predict_penalty(
        self,
        incident_type: str,
        track_condition: str,
        intent: str,
        driver_history: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        incident = parse_choice(incident_type, IncidentType, "incident_type")
        condition = parse_choice(track_condition, TrackCondition, "track_condition")
        intent_value = parse_choice(intent, Intent, "intent")
        history = validate_driver_history(driver_history)

        profile = PROFILES[incident]
        severity = profile.base_severity
        reasoning = [profile.rationale]
        adjustments: List[Dict[str, Any]] = []

        def adjust(source: str, value: Any, delta: float, reason: str) -> None:
            nonlocal severity
            severity += delta
            adjustments.append({"source": source, "value": value, "delta": delta})
            reasoning.append(reason)

        if intent_value is Intent.INTENTIONAL and incident in DRIVING_CONDUCT:
            adjust("intent", intent_value.value, INTENTIONAL_DELTA, "Intentional conduct raises the triage severity.")
        elif intent_value is Intent.RACING_INCIDENT and incident in CAR_TO_CAR:
            adjust(
                "intent",
                intent_value.value,
                RACING_INCIDENT_DELTA,
                "A racing-incident description lowers the preliminary severity, subject to evidence.",
            )
        elif intent_value in (Intent.INTENTIONAL, Intent.RACING_INCIDENT):
            reasoning.append(f"The '{intent_value.value}' intent does not change the triage of a {incident.value} case.")

        if condition in REDUCED_GRIP and incident in DRIVING_CONDUCT:
            reasoning.append("Reduced-grip conditions are recorded as context but do not automatically excuse an infringement.")

        if history:
            recent = history.get("recent_penalties")
            total = history.get("total_penalties")
            if recent is not None and len(recent) >= RECENT_PENALTIES_THRESHOLD:
                adjust(
                    "driver_history.recent_penalties",
                    len(recent),
                    RECENT_PENALTIES_DELTA,
                    "The supplied recent-history count increases the triage severity.",
                )
            if total is not None and total >= TOTAL_PENALTIES_THRESHOLD:
                adjust(
                    "driver_history.total_penalties",
                    total,
                    TOTAL_PENALTIES_DELTA,
                    "The supplied long-term history increases the triage severity slightly.",
                )

        # The band is derived from the same rounded score that is reported.
        severity = round(max(0.0, min(1.0, severity)), 3)

        # Completeness of the inputs that can change this incident type's severity
        # (track condition never does, so it is not counted).
        supplied = {"driver_history": history is not None}
        if incident in DRIVING_CONDUCT:
            supplied["intent"] = intent_value is not Intent.UNKNOWN
        confidence = 0.50 + 0.20 * sum(supplied.values()) / len(supplied)

        return {
            "triage_category": profile.outcome_category,
            "severity_band": _severity_band(severity),
            "severity_score": severity,
            "severity_adjustments": adjustments,
            "confidence": round(confidence, 3),
            "confidence_basis": CONFIDENCE_BASIS,
            "confidence_inputs": supplied,
            "referenced_rule": None,
            "reasoning": " ".join(reasoning),
            "inputs": {
                "incident_type": incident.value,
                "track_condition": condition.value,
                "intent": intent_value.value,
                "driver_history": history,
            },
            "method": METHOD,
            "disclaimer": DISCLAIMER,
        }


_penalty_predictor: Optional[PenaltyPredictor] = None
_penalty_predictor_lock = threading.Lock()


def get_penalty_predictor() -> PenaltyPredictor:
    global _penalty_predictor
    with _penalty_predictor_lock:
        if _penalty_predictor is None:
            _penalty_predictor = PenaltyPredictor()
        return _penalty_predictor


def predict_penalty(
    incident_type: str,
    track_condition: str,
    intent: str,
    driver_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return get_penalty_predictor().predict_penalty(
        incident_type=incident_type,
        track_condition=track_condition,
        intent=intent,
        driver_history=driver_history,
    )
