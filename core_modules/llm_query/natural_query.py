#!/usr/bin/env python3
"""
Natural Query Processor
Processes natural language queries and routes them to appropriate modules
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    PERFORMANCE = "performance"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"
    STRATEGY = "strategy"
    EMOTION = "emotion"
    GENERAL = "general"


@dataclass
class QueryResult:
    """Result of natural language query processing"""
    answer: str
    query_type: QueryType
    confidence: float
    data_sources: List[str]
    additional_context: Optional[Dict[str, Any]] = None


class NaturalQueryProcessor:
    """Processes natural language queries about F1 racing"""
    
    def __init__(self):
        self.performance_keywords = [
            "lost time", "sector", "lap time", "pace", "performance",
            "braking", "throttle", "speed", "acceleration", "cornering"
        ]
        self.regulatory_keywords = [
            "rule", "regulation", "penalty", "fia", "violation",
            "track limits", "unsafe release", "collision", "blocking"
        ]
        self.technical_keywords = [
            "setup", "tire", "fuel", "engine", "aero", "brake",
            "suspension", "wing", "diff", "ers", "battery"
        ]
        self.strategy_keywords = [
            "strategy", "pit stop", "tire compound", "undercut", "overcut",
            "stint", "fuel load", "weather", "safety car"
        ]
        self.emotion_keywords = [
            "driver", "radio", "emotion", "frustrated", "angry", "calm",
            "focused", "panicked", "excited"
        ]
    
    def process_natural_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Process a natural language query
        
        Args:
            query: Natural language query
            context: Additional context data
            
        Returns:
            QueryResult with answer and metadata
        """
        # Determine query type
        query_type = self._classify_query(query)
        
        # Route to appropriate handler
        if query_type == QueryType.PERFORMANCE:
            return self._handle_performance_query(query, context)
        elif query_type == QueryType.REGULATORY:
            return self._handle_regulatory_query(query, context)
        elif query_type == QueryType.TECHNICAL:
            return self._handle_technical_query(query, context)
        elif query_type == QueryType.STRATEGY:
            return self._handle_strategy_query(query, context)
        elif query_type == QueryType.EMOTION:
            return self._handle_emotion_query(query, context)
        else:
            return self._handle_general_query(query, context)
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the type of query based on keywords"""
        query_lower = query.lower()
        
        # Count keyword matches for each category
        performance_score = sum(1 for keyword in self.performance_keywords if keyword in query_lower)
        regulatory_score = sum(1 for keyword in self.regulatory_keywords if keyword in query_lower)
        technical_score = sum(1 for keyword in self.technical_keywords if keyword in query_lower)
        strategy_score = sum(1 for keyword in self.strategy_keywords if keyword in query_lower)
        emotion_score = sum(1 for keyword in self.emotion_keywords if keyword in query_lower)
        
        # Find the highest scoring category
        scores = {
            QueryType.PERFORMANCE: performance_score,
            QueryType.REGULATORY: regulatory_score,
            QueryType.TECHNICAL: technical_score,
            QueryType.STRATEGY: strategy_score,
            QueryType.EMOTION: emotion_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            for query_type, score in scores.items():
                if score == max_score:
                    return query_type
        
        return QueryType.GENERAL
    
    def _handle_performance_query(self, query: str, context: Optional[Dict[str, Any]]) -> QueryResult:
        """Handle performance-related queries"""
        # Extract performance metrics from query
        metrics = self._extract_performance_metrics(query)
        
        if "sector" in query.lower():
            sector = self._extract_sector_number(query)
            answer = f"Performance analysis for Sector {sector}: "
            if context and "telemetry" in context:
                answer += "Based on telemetry data, the driver lost time in braking zones and had suboptimal throttle application."
            else:
                answer += "Detailed sector analysis requires telemetry data for accurate assessment."
        elif "lap time" in query.lower():
            answer = "Lap time analysis shows variations due to tire degradation and track evolution."
        else:
            answer = "Performance analysis requires real-time telemetry data for accurate assessment."
        
        return QueryResult(
            answer=answer,
            query_type=QueryType.PERFORMANCE,
            confidence=0.8,
            data_sources=["telemetry", "lap_times", "sector_times"]
        )
    
    def _handle_regulatory_query(self, query: str, context: Optional[Dict[str, Any]]) -> QueryResult:
        """Handle regulatory queries using FIA RAG agent"""
        from core_modules.rule_checker.fia_rag_agent import query_fia_regulations
        
        # Route to FIA RAG agent
        fia_answer = query_fia_regulations(query)
        
        return QueryResult(
            answer=fia_answer,
            query_type=QueryType.REGULATORY,
            confidence=0.85,
            data_sources=["fia_regulations", "penalty_precedents"]
        )
    
    def _handle_technical_query(self, query: str, context: Optional[Dict[str, Any]]) -> QueryResult:
        """Handle technical queries about car setup and components"""
        if "tire" in query.lower():
            answer = "Tire analysis shows optimal compound selection based on track temperature and degradation curves."
        elif "setup" in query.lower():
            answer = "Car setup recommendations consider track characteristics, weather conditions, and driver preferences."
        elif "fuel" in query.lower():
            answer = "Fuel strategy optimization balances performance requirements with regulatory constraints."
        else:
            answer = "Technical analysis requires detailed component data and setup parameters."
        
        return QueryResult(
            answer=answer,
            query_type=QueryType.TECHNICAL,
            confidence=0.75,
            data_sources=["car_setup", "tire_data", "fuel_data"]
        )
    
    def _handle_strategy_query(self, query: str, context: Optional[Dict[str, Any]]) -> QueryResult:
        """Handle strategy-related queries"""
        if "pit stop" in query.lower():
            answer = "Pit stop strategy optimization considers tire degradation, track position, and competitor strategies."
        elif "undercut" in query.lower() or "overcut" in query.lower():
            answer = "Undercut/overcut opportunities are identified based on tire age gaps and track position analysis."
        else:
            answer = "Strategy analysis requires real-time race data and competitor information."
        
        return QueryResult(
            answer=answer,
            query_type=QueryType.STRATEGY,
            confidence=0.8,
            data_sources=["strategy_engine", "competitor_data", "race_state"]
        )
    
    def _handle_emotion_query(self, query: str, context: Optional[Dict[str, Any]]) -> QueryResult:
        """Handle emotion-related queries"""
        answer = "Driver emotion analysis requires audio data from radio communications for accurate assessment."
        
        return QueryResult(
            answer=answer,
            query_type=QueryType.EMOTION,
            confidence=0.6,
            data_sources=["radio_audio", "emotion_classifier"]
        )
    
    def _handle_general_query(self, query: str, context: Optional[Dict[str, Any]]) -> QueryResult:
        """Handle general queries"""
        answer = "This query requires additional context or specific data sources for accurate analysis."
        
        return QueryResult(
            answer=answer,
            query_type=QueryType.GENERAL,
            confidence=0.5,
            data_sources=["general_knowledge"]
        )
    
    def _extract_performance_metrics(self, query: str) -> Dict[str, Any]:
        """Extract performance metrics from query"""
        metrics = {}
        
        # Extract sector numbers
        sector_match = re.search(r'sector\s*(\d+)', query.lower())
        if sector_match:
            metrics['sector'] = int(sector_match.group(1))
        
        # Extract lap numbers
        lap_match = re.search(r'lap\s*(\d+)', query.lower())
        if lap_match:
            metrics['lap'] = int(lap_match.group(1))
        
        # Extract time deltas
        time_match = re.search(r'(\d+\.?\d*)\s*s', query.lower())
        if time_match:
            metrics['time_delta'] = float(time_match.group(1))
        
        return metrics
    
    def _extract_sector_number(self, query: str) -> int:
        """Extract sector number from query"""
        sector_match = re.search(r'sector\s*(\d+)', query.lower())
        if sector_match:
            return int(sector_match.group(1))
        return 1  # Default to sector 1


# Global query processor instance
_query_processor = None

def get_query_processor() -> NaturalQueryProcessor:
    """Get or create query processor instance"""
    global _query_processor
    if _query_processor is None:
        _query_processor = NaturalQueryProcessor()
    return _query_processor

def process_natural_query(
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process natural language query
    
    Args:
        query: Natural language query
        context: Additional context data
        
    Returns:
        Query result with answer and metadata
    """
    processor = get_query_processor()
    result = processor.process_natural_query(query, context)
    
    return {
        "answer": result.answer,
        "query_type": result.query_type.value,
        "confidence": result.confidence,
        "data_sources": result.data_sources,
        "additional_context": result.additional_context
    }


# Example usage and testing
if __name__ == "__main__":
    print("🏁 Natural Query Processor Test")
    print("=" * 50)
    
    processor = NaturalQueryProcessor()
    
    # Test queries
    test_queries = [
        "Why did we lose 0.7s in Sector 2 on Lap 14?",
        "Which rule applies to unsafe pit release?",
        "What's the optimal tire compound for this track?",
        "Should we pit now for the undercut?",
        "How is the driver feeling based on radio communication?",
        "What's the weather forecast for the next 10 laps?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        result = processor.process_natural_query(query)
        
        print(f"   📋 Type: {result.query_type.value}")
        print(f"   🎯 Confidence: {result.confidence:.2f}")
        print(f"   📊 Data Sources: {', '.join(result.data_sources)}")
        print(f"   💬 Answer: {result.answer}")
        print("-" * 50) 