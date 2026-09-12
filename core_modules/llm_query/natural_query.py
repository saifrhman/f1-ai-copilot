#!/usr/bin/env python3
"""Natural-language query router for F1 AI Copilot modules."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryType(Enum):
    PERFORMANCE = "performance"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"
    STRATEGY = "strategy"
    EMOTION = "emotion"
    GENERAL = "general"


@dataclass
class QueryResult:
    answer: str
    query_type: QueryType
    confidence: float
    data_sources: List[str]
    additional_context: Optional[Dict[str, Any]] = None


class NaturalQueryProcessor:
    """Classify a query and route it to a module that has the required evidence."""

    def __init__(self):
        self.keywords = {
            QueryType.PERFORMANCE: ["lost time", "sector", "lap time", "pace", "performance", "braking", "throttle", "speed", "acceleration", "cornering"],
            QueryType.REGULATORY: ["rule", "regulation", "penalty", "fia", "violation", "track limits", "unsafe release", "collision", "blocking"],
            QueryType.TECHNICAL: ["setup", "ride height", "wing", "differential", "brake bias", "suspension"],
            QueryType.STRATEGY: ["strategy", "pit stop", "tire compound", "tyre compound", "undercut", "overcut", "stint", "safety car"],
            QueryType.EMOTION: ["driver emotion", "radio", "emotion", "frustrated", "angry", "calm", "panicked", "excited"],
        }

    def process_natural_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryResult:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        context = context or {}
        query_type = self._classify_query(query)
        handlers = {
            QueryType.PERFORMANCE: self._handle_performance_query,
            QueryType.REGULATORY: self._handle_regulatory_query,
            QueryType.TECHNICAL: self._handle_technical_query,
            QueryType.STRATEGY: self._handle_strategy_query,
            QueryType.EMOTION: self._handle_emotion_query,
            QueryType.GENERAL: self._handle_general_query,
        }
        return handlers[query_type](query, context)

    def _classify_query(self, query: str) -> QueryType:
        text = query.lower()
        scores = {kind: sum(1 for keyword in words if keyword in text) for kind, words in self.keywords.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else QueryType.GENERAL

    def _handle_performance_query(self, query: str, context: Dict[str, Any]) -> QueryResult:
        telemetry = context.get("telemetry")
        if not isinstance(telemetry, dict):
            return QueryResult(
                "I need telemetry in context.telemetry to answer that performance question.",
                QueryType.PERFORMANCE,
                0.0,
                [],
            )

        lap_times = telemetry.get("lap_times") or []
        sector_times = telemetry.get("sector_times") or {}
        braking = telemetry.get("braking_consistency")
        throttle = telemetry.get("throttle_aggressiveness")
        parts: List[str] = []
        evidence: Dict[str, Any] = {}

        if lap_times:
            values = [float(v) for v in lap_times]
            best = min(values)
            latest = values[-1]
            parts.append(f"Latest lap: {latest:.3f}s; best supplied lap: {best:.3f}s; delta: {latest - best:+.3f}s.")
            evidence["lap_times"] = values

        sector_match = re.search(r"sector\s*(\d+)", query.lower())
        if sector_match and sector_times:
            sector = sector_match.group(1)
            candidate = sector_times.get(sector) if isinstance(sector_times, dict) else None
            if candidate is None and isinstance(sector_times, list):
                idx = int(sector) - 1
                if 0 <= idx < len(sector_times):
                    candidate = sector_times[idx]
            if candidate is not None:
                parts.append(f"Supplied Sector {sector} time: {float(candidate):.3f}s.")
                evidence[f"sector_{sector}"] = float(candidate)

        if braking is not None:
            parts.append(f"Braking consistency input: {float(braking):.2f}.")
            evidence["braking_consistency"] = float(braking)
        if throttle is not None:
            parts.append(f"Throttle aggressiveness input: {float(throttle):.2f}.")
            evidence["throttle_aggressiveness"] = float(throttle)

        if not parts:
            return QueryResult(
                "Telemetry was supplied, but it does not contain the fields needed to answer this question.",
                QueryType.PERFORMANCE,
                0.1,
                ["telemetry"],
            )

        return QueryResult(" ".join(parts), QueryType.PERFORMANCE, 0.8, ["telemetry"], evidence)

    @staticmethod
    def _handle_regulatory_query(query: str, context: Dict[str, Any]) -> QueryResult:
        from core_modules.rule_checker.fia_rag_agent import query_fia_regulations_detailed

        result = query_fia_regulations_detailed(query)
        if not result.get("grounded") and str(result.get("answer", "")).startswith("FIA RAG is unavailable:"):
            raise RuntimeError(result["answer"])
        return QueryResult(
            answer=result["answer"],
            query_type=QueryType.REGULATORY,
            confidence=float(result.get("confidence", 0.0)),
            data_sources=["fia_regulations"],
            additional_context={
                "citations": result.get("citations", []),
                "retrieved_passages": result.get("retrieved_passages", []),
                "top_retrieval_score": result.get("top_retrieval_score", 0.0),
                "grounded": result.get("grounded", False),
            },
        )

    @staticmethod
    def _handle_technical_query(query: str, context: Dict[str, Any]) -> QueryResult:
        required = ("driver_preferences", "track_profile", "weather")
        if not all(key in context for key in required):
            return QueryResult(
                "For a setup recommendation I need driver_preferences, track_profile and weather in the query context.",
                QueryType.TECHNICAL,
                0.0,
                [],
            )
        from core_modules.setup_optimizer.setup_recommender import recommend_setup

        setup = recommend_setup(context["driver_preferences"], context["track_profile"], context["weather"])
        answer = (
            f"Recommended ride height {setup['ride_height']:.1f} mm, front/rear wing "
            f"{setup['front_wing_angle']:.1f}°/{setup['rear_wing_angle']:.1f}°, and brake bias {setup['brake_bias']:.1f}%. "
            f"{setup['reasoning']}"
        )
        return QueryResult(answer, QueryType.TECHNICAL, float(setup["confidence"]), ["setup_optimizer"], setup)

    @staticmethod
    def _handle_strategy_query(query: str, context: Dict[str, Any]) -> QueryResult:
        required = ("telemetry", "car_status", "driver_profile", "tire_data", "race_state", "competition")
        if not all(key in context for key in required):
            return QueryResult(
                "To generate a race strategy I need telemetry, car_status, driver_profile, tire_data, race_state and competition in the query context.",
                QueryType.STRATEGY,
                0.0,
                [],
            )

        from core_modules.strategy_optimizer.strategy_engine import (
            CarStatus,
            Competitor,
            DriverProfile,
            RaceState,
            TireCompound,
            TireData,
            WeatherCondition,
            generate_strategy,
        )

        race_data = dict(context["race_state"])
        race_data["weather"] = WeatherCondition(race_data["weather"]) if isinstance(race_data.get("weather"), str) else race_data["weather"]
        tyre_data: Dict[TireCompound, TireData] = {}
        for key, value in context["tire_data"].items():
            compound = TireCompound(key) if isinstance(key, str) else key
            item = dict(value)
            raw_compound = item.get("compound", compound.value)
            item["compound"] = TireCompound(raw_compound) if isinstance(raw_compound, str) else raw_compound
            item["peak_performance_window"] = tuple(item["peak_performance_window"])
            tyre_data[compound] = TireData(**item)
        competitors = []
        for value in context["competition"]:
            item = dict(value)
            if isinstance(item.get("tire_compound"), str):
                item["tire_compound"] = TireCompound(item["tire_compound"])
            competitors.append(Competitor(**item))

        strategies = generate_strategy(
            telemetry=context["telemetry"],
            car_status=CarStatus(**context["car_status"]),
            driver_profile=DriverProfile(**context["driver_profile"]),
            tire_data=tyre_data,
            race_state=RaceState(**race_data),
            competition=competitors,
        )
        best = strategies[0]
        answer = (
            f"Best candidate is {best.strategy_id}: {len(best.pit_laps)} stop(s), "
            f"compounds {' -> '.join(c.value for c in best.tire_compounds)}, pit laps {best.pit_laps}, "
            f"projected remaining-race time {best.projected_race_time:.1f}s."
        )
        return QueryResult(
            answer,
            QueryType.STRATEGY,
            best.confidence_score,
            ["strategy_engine"],
            {"strategy_id": best.strategy_id, "pit_laps": best.pit_laps},
        )

    @staticmethod
    def _handle_emotion_query(query: str, context: Dict[str, Any]) -> QueryResult:
        audio = context.get("audio_file")
        if not audio:
            return QueryResult(
                "I need audio_file in the query context to analyse driver-radio emotion.",
                QueryType.EMOTION,
                0.0,
                [],
            )
        from core_modules.driver_emotion.emotion_classifier import classify_emotion_detailed

        result = classify_emotion_detailed(str(audio), transcribe=bool(context.get("transcribe", False)))
        return QueryResult(
            f"The audio classifier returned {result['emotion']} with similarity confidence {result['confidence']:.2f}.",
            QueryType.EMOTION,
            float(result["confidence"]),
            ["driver_emotion"],
            result,
        )

    @staticmethod
    def _handle_general_query(query: str, context: Dict[str, Any]) -> QueryResult:
        return QueryResult(
            "I could not map that question to a supported module. Ask about FIA regulations, telemetry performance, race strategy, car setup, or driver-radio emotion.",
            QueryType.GENERAL,
            0.0,
            [],
        )


_query_processor: Optional[NaturalQueryProcessor] = None


def get_query_processor() -> NaturalQueryProcessor:
    global _query_processor
    if _query_processor is None:
        _query_processor = NaturalQueryProcessor()
    return _query_processor


def process_natural_query(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = get_query_processor().process_natural_query(query, context)
    return {
        "answer": result.answer,
        "query_type": result.query_type.value,
        "confidence": result.confidence,
        "data_sources": result.data_sources,
        "additional_context": result.additional_context,
    }
