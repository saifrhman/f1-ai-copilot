#!/usr/bin/env python3
"""Transparent incident/penalty triage for the F1 AI Copilot demo.

This module deliberately does not claim to reproduce steward decisions or cite
hard-coded FIA article numbers. Exact regulatory questions belong to the FIA RAG
module, which can ground answers in the current source documents.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class IncidentType(Enum):
    TRACK_LIMITS = "track_limits"
    UNSAFE_RELEASE = "unsafe_release"
    COLLISION = "collision"
    BLOCKING = "blocking"
    DANGEROUS_DRIVING = "dangerous_driving"
    TECHNICAL_INFRINGEMENT = "technical_infringement"
    SPEEDING_IN_PIT = "speeding_in_pit"
    ILLEGAL_OVERTAKING = "illegal_overtaking"


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


class PenaltyPredictor:
    """Deterministic triage estimator, not a trained steward-decision model."""

    VALID_CONDITIONS = {"dry", "wet", "intermediate", "mixed", "unknown"}
    VALID_INTENTS = {"accidental", "intentional", "racing_incident", "unknown"}

    def predict_penalty(
        self,
        incident_type: str,
        track_condition: str,
        intent: str,
        driver_history: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            incident = IncidentType(incident_type)
        except ValueError as exc:
            raise ValueError(f"Unknown incident_type: {incident_type}") from exc

        condition = track_condition.lower().strip() or "unknown"
        intent_value = intent.lower().strip() or "unknown"
        if condition not in self.VALID_CONDITIONS:
            condition = "unknown"
        if intent_value not in self.VALID_INTENTS:
            intent_value = "unknown"

        profile = PROFILES[incident]
        severity = profile.base_severity
        reasoning = [profile.rationale]

        if intent_value == "intentional":
            severity += 0.15
            reasoning.append("Intentional conduct raises the triage severity.")
        elif intent_value == "racing_incident":
            severity -= 0.08
            reasoning.append("A racing-incident description lowers the preliminary severity, subject to evidence.")

        if condition in {"wet", "intermediate", "mixed"}:
            reasoning.append("Reduced-grip conditions are recorded as context but do not automatically excuse an infringement.")

        history = driver_history or {}
        recent = history.get("recent_penalties", [])
        total = history.get("total_penalties", 0)
        try:
            total_value = int(total)
        except (TypeError, ValueError):
            total_value = 0
        if isinstance(recent, list) and len(recent) >= 3:
            severity += 0.08
            reasoning.append("The supplied recent-history count increases the triage severity.")
        elif total_value >= 10:
            severity += 0.05
            reasoning.append("The supplied long-term history increases the triage severity slightly.")

        severity = max(0.0, min(1.0, severity))
        if severity >= 0.80:
            band = "high"
        elif severity >= 0.50:
            band = "medium"
        else:
            band = "low"

        # Confidence reflects completeness of the supplied triage fields, not the
        # probability that FIA stewards will choose a particular sanction.
        confidence = 0.50
        if condition != "unknown":
            confidence += 0.08
        if intent_value != "unknown":
            confidence += 0.08
        if driver_history:
            confidence += 0.04
        confidence = min(0.70, confidence)

        return {
            "predicted_penalty": profile.outcome_category,
            "severity_band": band,
            "severity_score": round(severity, 3),
            "confidence": round(confidence, 3),
            "referenced_rule": None,
            "reasoning": " ".join(reasoning),
            "method": "transparent heuristic triage",
            "disclaimer": (
                "This is not an FIA steward-decision predictor and is not a legal/regulatory determination. "
                "Use the FIA RAG endpoint for evidence from the current regulations."
            ),
        }


_penalty_predictor: Optional[PenaltyPredictor] = None


def get_penalty_predictor() -> PenaltyPredictor:
    global _penalty_predictor
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
