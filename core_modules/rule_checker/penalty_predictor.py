#!/usr/bin/env python3
"""
Penalty Predictor
Estimates penalties based on incident type, FIA precedent, track conditions, and intent
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random


class IncidentType(Enum):
    TRACK_LIMITS = "track_limits"
    UNSAFE_RELEASE = "unsafe_release"
    COLLISION = "collision"
    BLOCKING = "blocking"
    DANGEROUS_DRIVING = "dangerous_driving"
    TECHNICAL_INFRINGEMENT = "technical_infringement"
    SPEEDING_IN_PIT = "speeding_in_pit"
    ILLEGAL_OVERTAKING = "illegal_overtaking"


class PenaltyType(Enum):
    TIME_PENALTY = "time_penalty"
    GRID_DROP = "grid_drop"
    DISQUALIFICATION = "disqualification"
    WARNING = "warning"
    DRIVE_THROUGH = "drive_through"
    STOP_GO = "stop_go"


@dataclass
class PenaltyPrecedent:
    """Historical penalty precedent"""
    incident_type: IncidentType
    track_condition: str
    intent: str
    penalty_type: PenaltyType
    penalty_value: str  # e.g., "5s", "3_grid", "disqualification"
    fia_rule: str  # e.g., "Article 38.3"
    severity: float  # 0-1, how severe the incident was
    description: str


class PenaltyPredictor:
    """Predicts penalties based on incident analysis and FIA precedent"""
    
    def __init__(self):
        self.precedents = self._load_precedents()
        self.rule_mappings = self._load_rule_mappings()
    
    def _load_precedents(self) -> List[PenaltyPrecedent]:
        """Load historical penalty precedents"""
        return [
            # Track limits precedents
            PenaltyPrecedent(
                incident_type=IncidentType.TRACK_LIMITS,
                track_condition="dry",
                intent="accidental",
                penalty_type=PenaltyType.TIME_PENALTY,
                penalty_value="5s",
                fia_rule="Article 38.3",
                severity=0.3,
                description="Multiple track limits violations in one lap"
            ),
            PenaltyPrecedent(
                incident_type=IncidentType.TRACK_LIMITS,
                track_condition="wet",
                intent="accidental",
                penalty_type=PenaltyType.TIME_PENALTY,
                penalty_value="3s",
                fia_rule="Article 38.3",
                severity=0.2,
                description="Track limits violation in wet conditions"
            ),
            
            # Unsafe release precedents
            PenaltyPrecedent(
                incident_type=IncidentType.UNSAFE_RELEASE,
                track_condition="dry",
                intent="accidental",
                penalty_type=PenaltyType.TIME_PENALTY,
                penalty_value="10s",
                fia_rule="Article 38.4",
                severity=0.7,
                description="Unsafe release causing near-miss"
            ),
            PenaltyPrecedent(
                incident_type=IncidentType.UNSAFE_RELEASE,
                track_condition="dry",
                intent="intentional",
                penalty_type=PenaltyType.GRID_DROP,
                penalty_value="3_grid",
                fia_rule="Article 38.4",
                severity=0.9,
                description="Deliberate unsafe release"
            ),
            
            # Collision precedents
            PenaltyPrecedent(
                incident_type=IncidentType.COLLISION,
                track_condition="dry",
                intent="racing_incident",
                penalty_type=PenaltyType.TIME_PENALTY,
                penalty_value="5s",
                fia_rule="Article 38.1",
                severity=0.6,
                description="Racing incident causing collision"
            ),
            PenaltyPrecedent(
                incident_type=IncidentType.COLLISION,
                track_condition="dry",
                intent="intentional",
                penalty_type=PenaltyType.GRID_DROP,
                penalty_value="5_grid",
                fia_rule="Article 38.1",
                severity=0.9,
                description="Deliberate collision"
            ),
            
            # Blocking precedents
            PenaltyPrecedent(
                incident_type=IncidentType.BLOCKING,
                track_condition="dry",
                intent="intentional",
                penalty_type=PenaltyType.TIME_PENALTY,
                penalty_value="5s",
                fia_rule="Article 38.2",
                severity=0.5,
                description="Deliberate blocking of faster car"
            ),
            
            # Dangerous driving precedents
            PenaltyPrecedent(
                incident_type=IncidentType.DANGEROUS_DRIVING,
                track_condition="dry",
                intent="intentional",
                penalty_type=PenaltyType.DISQUALIFICATION,
                penalty_value="disqualification",
                fia_rule="Article 38.1",
                severity=1.0,
                description="Extremely dangerous driving"
            ),
            
            # Technical infringements
            PenaltyPrecedent(
                incident_type=IncidentType.TECHNICAL_INFRINGEMENT,
                track_condition="dry",
                intent="accidental",
                penalty_type=PenaltyType.DISQUALIFICATION,
                penalty_value="disqualification",
                fia_rule="Article 30.1",
                severity=0.8,
                description="Technical regulation violation"
            ),
            
            # Speeding in pit lane
            PenaltyPrecedent(
                incident_type=IncidentType.SPEEDING_IN_PIT,
                track_condition="dry",
                intent="accidental",
                penalty_type=PenaltyType.TIME_PENALTY,
                penalty_value="5s",
                fia_rule="Article 34.1",
                severity=0.4,
                description="Exceeding pit lane speed limit"
            ),
            
            # Illegal overtaking
            PenaltyPrecedent(
                incident_type=IncidentType.ILLEGAL_OVERTAKING,
                track_condition="dry",
                intent="intentional",
                penalty_type=PenaltyType.TIME_PENALTY,
                penalty_value="5s",
                fia_rule="Article 38.2",
                severity=0.5,
                description="Overtaking under yellow flags"
            )
        ]
    
    def _load_rule_mappings(self) -> Dict[str, str]:
        """Load FIA rule mappings for different incident types"""
        return {
            IncidentType.TRACK_LIMITS.value: "Article 38.3",
            IncidentType.UNSAFE_RELEASE.value: "Article 38.4",
            IncidentType.COLLISION.value: "Article 38.1",
            IncidentType.BLOCKING.value: "Article 38.2",
            IncidentType.DANGEROUS_DRIVING.value: "Article 38.1",
            IncidentType.TECHNICAL_INFRINGEMENT.value: "Article 30.1",
            IncidentType.SPEEDING_IN_PIT.value: "Article 34.1",
            IncidentType.ILLEGAL_OVERTAKING.value: "Article 38.2"
        }
    
    def predict_penalty(
        self,
        incident_type: str,
        track_condition: str,
        intent: str,
        driver_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict penalty for an incident
        
        Args:
            incident_type: Type of incident
            track_condition: Track condition during incident
            intent: Driver intent (accidental/intentional/racing_incident)
            driver_history: Driver's penalty history
            
        Returns:
            Predicted penalty with confidence and reasoning
        """
        try:
            incident_enum = IncidentType(incident_type)
        except ValueError:
            return {
                "predicted_penalty": "unknown",
                "confidence": 0.0,
                "referenced_rule": "Unknown",
                "reasoning": f"Unknown incident type: {incident_type}"
            }
        
        # Find relevant precedents
        relevant_precedents = [
            p for p in self.precedents
            if p.incident_type == incident_enum
            and p.track_condition == track_condition
            and p.intent == intent
        ]
        
        if not relevant_precedents:
            # Find precedents with same incident type and intent
            relevant_precedents = [
                p for p in self.precedents
                if p.incident_type == incident_enum
                and p.intent == intent
            ]
        
        if not relevant_precedents:
            # Find any precedents with same incident type
            relevant_precedents = [
                p for p in self.precedents
                if p.incident_type == incident_enum
            ]
        
        if not relevant_precedents:
            return {
                "predicted_penalty": "warning",
                "confidence": 0.3,
                "referenced_rule": self.rule_mappings.get(incident_type, "Unknown"),
                "reasoning": f"No precedent found for {incident_type} incident"
            }
        
        # Select best precedent based on severity and conditions
        best_precedent = self._select_best_precedent(
            relevant_precedents, track_condition, intent, driver_history
        )
        
        # Adjust penalty based on driver history
        adjusted_penalty = self._adjust_for_driver_history(
            best_precedent, driver_history
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            best_precedent, track_condition, intent, driver_history
        )
        
        return {
            "predicted_penalty": adjusted_penalty.penalty_value,
            "confidence": confidence,
            "referenced_rule": adjusted_penalty.fia_rule,
            "reasoning": self._generate_reasoning(adjusted_penalty, track_condition, intent)
        }
    
    def _select_best_precedent(
        self,
        precedents: List[PenaltyPrecedent],
        track_condition: str,
        intent: str,
        driver_history: Optional[Dict[str, Any]]
    ) -> PenaltyPrecedent:
        """Select the best precedent based on conditions"""
        # Sort by relevance (exact match first, then by severity)
        scored_precedents = []
        
        for precedent in precedents:
            score = 0.0
            
            # Exact condition match
            if precedent.track_condition == track_condition:
                score += 2.0
            
            # Exact intent match
            if precedent.intent == intent:
                score += 2.0
            
            # Severity consideration
            score += precedent.severity
            
            scored_precedents.append((score, precedent))
        
        # Return highest scored precedent
        scored_precedents.sort(key=lambda x: x[0], reverse=True)
        return scored_precedents[0][1]
    
    def _adjust_for_driver_history(
        self,
        precedent: PenaltyPrecedent,
        driver_history: Optional[Dict[str, Any]]
    ) -> PenaltyPrecedent:
        """Adjust penalty based on driver's penalty history"""
        if not driver_history:
            return precedent
        
        # Check for repeat offenses
        recent_penalties = driver_history.get("recent_penalties", [])
        total_penalties = driver_history.get("total_penalties", 0)
        
        # Escalate penalty for repeat offenders
        if len(recent_penalties) >= 3 or total_penalties >= 10:
            # Upgrade penalty
            if precedent.penalty_type == PenaltyType.TIME_PENALTY:
                # Convert time penalty to grid drop
                return PenaltyPrecedent(
                    incident_type=precedent.incident_type,
                    track_condition=precedent.track_condition,
                    intent=precedent.intent,
                    penalty_type=PenaltyType.GRID_DROP,
                    penalty_value="3_grid",
                    fia_rule=precedent.fia_rule,
                    severity=precedent.severity + 0.2,
                    description=f"{precedent.description} (escalated due to repeat offenses)"
                )
        
        return precedent
    
    def _calculate_confidence(
        self,
        precedent: PenaltyPrecedent,
        track_condition: str,
        intent: str,
        driver_history: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in penalty prediction"""
        confidence = 0.7  # Base confidence
        
        # Adjust based on condition match
        if precedent.track_condition == track_condition:
            confidence += 0.1
        
        # Adjust based on intent match
        if precedent.intent == intent:
            confidence += 0.1
        
        # Adjust based on driver history availability
        if driver_history:
            confidence += 0.05
        
        # Adjust based on precedent severity
        confidence += precedent.severity * 0.1
        
        return min(0.95, confidence)
    
    def _generate_reasoning(
        self,
        precedent: PenaltyPrecedent,
        track_condition: str,
        intent: str
    ) -> str:
        """Generate reasoning for penalty prediction"""
        reasoning = f"Based on FIA precedent: {precedent.description}. "
        reasoning += f"Incident occurred in {track_condition} conditions with {intent} intent. "
        reasoning += f"Referenced rule: {precedent.fia_rule}."
        
        if precedent.penalty_type == PenaltyType.TIME_PENALTY:
            reasoning += f" Standard penalty for this type of incident is {precedent.penalty_value}."
        elif precedent.penalty_type == PenaltyType.GRID_DROP:
            reasoning += f" Grid drop penalty reflects the severity of the incident."
        elif precedent.penalty_type == PenaltyType.DISQUALIFICATION:
            reasoning += f" Disqualification is reserved for the most serious violations."
        
        return reasoning


# Global penalty predictor instance
_penalty_predictor = None

def get_penalty_predictor() -> PenaltyPredictor:
    """Get or create penalty predictor instance"""
    global _penalty_predictor
    if _penalty_predictor is None:
        _penalty_predictor = PenaltyPredictor()
    return _penalty_predictor

def predict_penalty(
    incident_type: str,
    track_condition: str,
    intent: str,
    driver_history: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predict penalty for an incident
    
    Args:
        incident_type: Type of incident
        track_condition: Track condition during incident
        intent: Driver intent
        driver_history: Driver's penalty history
        
    Returns:
        Predicted penalty with confidence and reasoning
    """
    predictor = get_penalty_predictor()
    return predictor.predict_penalty(incident_type, track_condition, intent, driver_history)


# Example usage and testing
if __name__ == "__main__":
    print("🏁 Penalty Predictor Test")
    print("=" * 50)
    
    predictor = PenaltyPredictor()
    
    # Test cases
    test_cases = [
        {
            "incident_type": "track_limits",
            "track_condition": "dry",
            "intent": "accidental",
            "driver_history": {"recent_penalties": [], "total_penalties": 2}
        },
        {
            "incident_type": "unsafe_release",
            "track_condition": "dry",
            "intent": "intentional",
            "driver_history": {"recent_penalties": ["5s", "3s"], "total_penalties": 8}
        },
        {
            "incident_type": "collision",
            "track_condition": "wet",
            "intent": "racing_incident",
            "driver_history": None
        },
        {
            "incident_type": "dangerous_driving",
            "track_condition": "dry",
            "intent": "intentional",
            "driver_history": {"recent_penalties": ["10s", "grid_drop"], "total_penalties": 15}
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"   Incident: {case['incident_type']}")
        print(f"   Condition: {case['track_condition']}")
        print(f"   Intent: {case['intent']}")
        
        result = predictor.predict_penalty(
            case['incident_type'],
            case['track_condition'],
            case['intent'],
            case['driver_history']
        )
        
        print(f"   📋 Predicted Penalty: {result['predicted_penalty']}")
        print(f"   🎯 Confidence: {result['confidence']:.2f}")
        print(f"   📜 Referenced Rule: {result['referenced_rule']}")
        print(f"   💭 Reasoning: {result['reasoning']}")
        print("-" * 50) 