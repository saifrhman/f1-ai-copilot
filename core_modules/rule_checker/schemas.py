#!/usr/bin/env python3
"""Pydantic request model for the incident/penalty triage endpoint.

The enums and limits come from ``penalty_predictor`` so the API schema and the
module validation cannot drift apart. Enum inputs are normalised the same way
as in the module (case, surrounding whitespace, spaces/hyphens -> "_").
"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from core_modules.rule_checker.penalty_predictor import (
    MAX_CHOICE_CHARS,
    MAX_PENALTY_NOTE_CHARS,
    MAX_RECENT_PENALTIES,
    MAX_TOTAL_PENALTIES,
    IncidentType,
    Intent,
    TrackCondition,
    check_history_consistency,
    normalise_token,
)

PenaltyNote = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PENALTY_NOTE_CHARS),
]


class DriverHistory(BaseModel):
    """Driver history fields that the triage heuristic actually uses."""

    model_config = ConfigDict(extra="forbid")

    recent_penalties: Optional[List[PenaltyNote]] = Field(
        default=None,
        max_length=MAX_RECENT_PENALTIES,
        description=(
            "Short descriptions of the driver's recent penalties. Only the count is used: "
            "three or more raise the triage severity (independently of total_penalties)."
        ),
    )
    total_penalties: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_TOTAL_PENALTIES,
        strict=True,
        description=(
            "Total number of penalties on record (integer). Ten or more raise the triage severity slightly. "
            "Must not be smaller than the number of recent_penalties when both are given."
        ),
    )

    @model_validator(mode="after")
    def _consistent_counts(self) -> "DriverHistory":
        check_history_consistency(self.recent_penalties, self.total_penalties)
        return self


class PenaltyRequest(BaseModel):
    """Request body for transparent heuristic incident triage (not a steward-decision predictor)."""

    model_config = ConfigDict(extra="forbid")

    incident_type: IncidentType = Field(description="Type of incident to triage.")
    track_condition: TrackCondition = Field(
        description="Track condition at the time of the incident. Use 'unknown' if it is not known."
    )
    intent: Intent = Field(description="Described intent of the driver. Use 'unknown' if it is not known.")
    driver_history: Optional[DriverHistory] = Field(
        default=None,
        description="Optional validated driver history; unknown fields are rejected.",
    )

    @field_validator("incident_type", "track_condition", "intent", mode="before")
    @classmethod
    def _normalise_choice(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) <= MAX_CHOICE_CHARS:
            return normalise_token(value)
        return value

    def to_predictor_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``penalty_predictor.predict_penalty``."""

        return {
            "incident_type": self.incident_type.value,
            "track_condition": self.track_condition.value,
            "intent": self.intent.value,
            "driver_history": (
                self.driver_history.model_dump(exclude_none=True) if self.driver_history is not None else None
            ),
        }
