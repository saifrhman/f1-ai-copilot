#!/usr/bin/env python3
"""Natural-language query router for F1 AI Copilot modules.

Routing is a transparent keyword heuristic, not a language model:

1. The query is lower-cased, accents are stripped ("fermé" -> "ferme"),
   Unicode dashes become "-" and possessive "'s" is removed
   ("driver's radio" -> "driver radio"). ``classify_query`` rejects
   non-strings, queries longer than MAX_QUERY_CHARS and queries that Unicode
   normalisation expands beyond MAX_NORMALISED_QUERY_CHARS.
2. Vocabulary terms are matched as whole words/phrases (no substring hits, so
   "following" is not "wing", "space" is not "pace", "Mafia" is not "FIA").
   A space inside a term matches a space, a hyphen or nothing ("one-stop",
   "one stop", "onestop"). A hyphen inside a term must be written as a hyphen
   or slash ("stop-go", "drive-through"), because the spaced forms are
   ordinary verbs ("drive through turn 3"). One plural ending is accepted:
   "es" after s/x/z/ch/sh ("box" -> "boxes"), otherwise "s", so a term never
   matches a verb form such as "goes".
   A few regex terms (PATTERN_TERMS) cover phrasings a fixed phrase cannot,
   such as "lose the most time", "how much time do we lose in the pit lane",
   "Article 33.4" or "the car gets upset"; they are matched first. Then longer
   phrases are matched before shorter ones. Every match claims its characters,
   so "pit lane" and "speed limit" are not also counted as "pit" or "speed".
3. Each distinct matched term adds its weight to its query type: intent cues
   (e.g. "rule", "strategy", "setup", "telemetry", "radio") weigh 2, topic
   words weigh 1. A few neutral phrases ("rule out", "rule of thumb", "as a
   rule") are claimed without scoring. Words with a common non-specialist
   meaning are deliberately only topics: "article" (unless a number follows),
   "breach of", and "nervous"/"upset" (a car can be nervous or upset too; a
   car-handling phrasing such as "the rear feels nervous" is a technical cue).
4. Decision, in this fixed order:
   a. no match -> GENERAL;
   b. any regulatory cue (rule, regulation, "article <number>", penalty,
      allowed, permitted, legal, steward, FIA, stop-go, ...) -> REGULATORY,
      whatever else the query mentions: a question about what the rules say
      must be answered from the regulations, never from telemetry or a
      strategy simulation;
   c. otherwise the highest score wins;
   d. a tie that includes REGULATORY goes to REGULATORY;
   e. other ties go to the tied type whose required context is supplied
      (e.g. telemetry for PERFORMANCE), when exactly one of them has it;
   f. remaining ties follow ROUTING_PRIORITY
      (regulatory > strategy > technical > emotion > performance), because
      performance words such as "faster" or "pace" are the most incidental.

Every result carries these routing diagnostics (scores, matched terms and the
decision rule used) so that a routing decision can always be inspected.
"""

import math
import numbers
import re
import unicodedata
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


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
    # None when the answering module has no meaningful confidence value.
    confidence: Optional[float]
    data_sources: List[str]
    additional_context: Optional[Dict[str, Any]] = None
    routing: Optional[Dict[str, Any]] = None


MAX_QUERY_CHARS = 2000  # same limit as the FIA RAG question validator
# NFKD can expand one character into up to 18 (e.g. U+FDFA); ordinary text
# (accents, ligatures, full-width forms) stays far below this bound.
MAX_NORMALISED_QUERY_CHARS = 2 * MAX_QUERY_CHARS

CUE = 2  # words that state what the user wants (a module's subject)
TOPIC = 1  # words that only suggest a topic

_REGULATORY_CUES = (
    "rule", "regulation", "regulatory", "penalty", "penalties",
    "penalise", "penalised", "penalize", "penalized", "allowed", "permitted",
    "permissible", "legal", "illegal", "legally", "illegally", "legality",
    "steward", "fia", "sporting regulation", "technical regulation",
    "financial regulation", "sporting code", "infringement",
    "disqualified", "disqualification", "reprimand", "banned",
    # Penalty names only in their penalty sense: "drive through turn 3" and
    # "if the stop goes wrong" are not rules questions.
    "stop-go", "stop-and-go", "stop go penalty", "stop and go penalty",
    "drive-through", "drive through penalty",
)
_REGULATORY_TOPICS = (
    "article", "breach of", "drs", "parc ferme", "power unit", "cost cap",
    "budget cap", "minimum weight", "minimum car weight", "weight limit",
    "yellow flag", "double yellow", "red flag", "blue flag", "black flag",
    "black and white flag", "speed limit", "speeding", "pit lane", "pit entry",
    "pit exit", "unsafe release", "track limit", "collision", "blocking",
    "impeding", "false start", "jump start", "curfew", "homologation",
    "fuel flow", "plank", "skid block", "scrutineering", "protest",
    "racing room", "racing incident", "enough space", "leave space",
    "leaving space", "car width", "driving standard", "formation lap",
)
_STRATEGY_CUES = (
    "strategy", "strategies", "tyre strategy", "tire strategy", "race strategy",
    "pit strategy", "pit", "pitted", "pitting", "pit stop", "pit window", "box",
    "boxing", "undercut", "overcut", "one stop", "two stop", "three stop",
    "1 stop", "2 stop", "3 stop", "stay out", "staying out", "stayed out",
)
# Bare "soft"/"medium"/"hard" are common adjectives ("brake hard"), so the
# compound names count only in a tyre context.
_COMPOUND_PHRASES = tuple(
    f"{compound} {noun}"
    for compound in ("soft", "medium", "hard", "intermediate", "wet")
    for noun in ("tyre", "tire", "compound")
) + tuple(f"{prep} the {compound}" for prep in ("on", "to") for compound in ("soft", "medium", "hard"))
_STRATEGY_TOPICS = (
    "stop", "stint", "compound", "tyre compound", "tire compound", "safety car",
    "virtual safety car", "vsc", "degradation", "tyre degradation",
    "tire degradation", "tyre wear", "tire wear", "new tyre", "new tire",
    "fresh tyre", "fresh tire", "softs", "mediums", "hards", "inters",
    "track position", "tyre change", "tire change", "tyre", "tire",
) + _COMPOUND_PHRASES
_TECHNICAL_CUES = (
    "set up", "ride height", "brake bias", "differential", "wing angle",
    "wing level", "anti roll bar", "camber", "toe", "suspension", "gear ratio",
    "tyre pressure", "tire pressure",
)
_TECHNICAL_TOPICS = (
    "wing", "front wing", "rear wing", "downforce", "drag", "rake", "spring",
    "damper", "stiffness", "understeer", "oversteer", "balance", "handling",
    "snappy", "twitchy", "unsettled", "unstable", "instability", "bottoming",
    "porpoising",
)
_EMOTION_CUES = (
    "emotion", "emotional", "driver emotion", "driver radio", "team radio",
    "radio", "mood", "sentiment", "frustrated", "frustration", "angry", "anger",
    "panicked", "panic", "panicking", "stressed", "tone of voice",
)
_EMOTION_TOPICS = ("calm", "composed", "excited", "focused", "tense", "feeling", "nervous", "upset")
_PERFORMANCE_CUES = ("performance", "telemetry", "time loss", "time lost")
_PERFORMANCE_TOPICS = (
    "lap", "lap time", "laptime", "sector", "braking", "brake point",
    "braking point", "throttle", "acceleration", "pace", "speed", "top speed",
    "corner speed", "cornering", "corner exit", "corner entry", "slow", "fast",
    "slower", "faster", "slowest", "fastest", "quicker", "quickest",
    "consistency", "delta", "racing line", "traction",
)

# Claimed before scoring so that e.g. "rule out a two-stop" is not a rules question
# and "gear box" is not a pit "box".
NEUTRAL_PHRASES = ("rule out", "ruled out", "rules out", "rule of thumb", "as a rule", "gear box")

_CAR_PART = r"(?:car|rear|front|back\s+end|rear\s+end|front\s+end|platform|chassis)"
_HANDLING_WORD = r"(?:nervous|upset|unsettled|twitchy|snappy|unstable)"
_LOSE = r"(?:lose|loses|losing|lost)"

# (term, query type, weight, regex) for phrasings a fixed phrase cannot express.
# Matched in this order, before every phrase term.
PATTERN_TERMS: Tuple[Tuple[str, QueryType, int, str], ...] = (
    (
        "pit loss",
        QueryType.STRATEGY,
        CUE,
        r"pit(?:\s*(?:lane|stop))?\s+(?:time\s+)?loss(?:es)?"
        rf"|(?:time\s+(?:[a-z]+\s+){{0,3}})?(?:{_LOSE}|loss)\s+(?:[a-z]+\s+){{0,3}}?"
        r"(?:in|through|at|during|on)\s+(?:the\s+|a\s+)?pit(?:\s*(?:lane|stop))?s?",
    ),
    (
        "lose time",
        QueryType.PERFORMANCE,
        CUE,
        rf"{_LOSE}\s+(?:[a-z]+\s+){{0,2}}?(?:times?|seconds|tenths|hundredths|a\s+second|half\s+a\s+second)",
    ),
    ("article <number>", QueryType.REGULATORY, CUE, r"articles?\s*\d+(?:\.\d+)*"),
    (
        "car handling complaint",
        QueryType.TECHNICAL,
        CUE,
        rf"{_CAR_PART}\s+(?:[a-z]+\s+){{0,3}}?{_HANDLING_WORD}|{_HANDLING_WORD}\s+{_CAR_PART}",
    ),
)


def _weighted(cues: Sequence[str], topics: Sequence[str]) -> Dict[str, int]:
    terms = dict.fromkeys(cues, CUE)
    terms.update(dict.fromkeys(topics, TOPIC))
    return terms


VOCABULARY: Mapping[QueryType, Mapping[str, int]] = MappingProxyType({
    QueryType.REGULATORY: MappingProxyType(_weighted(_REGULATORY_CUES, _REGULATORY_TOPICS)),
    QueryType.STRATEGY: MappingProxyType(_weighted(_STRATEGY_CUES, _STRATEGY_TOPICS)),
    QueryType.TECHNICAL: MappingProxyType(_weighted(_TECHNICAL_CUES, _TECHNICAL_TOPICS)),
    QueryType.EMOTION: MappingProxyType(_weighted(_EMOTION_CUES, _EMOTION_TOPICS)),
    QueryType.PERFORMANCE: MappingProxyType(_weighted(_PERFORMANCE_CUES, _PERFORMANCE_TOPICS)),
})

# Tie-break order after context (see module docstring).
ROUTING_PRIORITY: Tuple[QueryType, ...] = (
    QueryType.REGULATORY,
    QueryType.STRATEGY,
    QueryType.TECHNICAL,
    QueryType.EMOTION,
    QueryType.PERFORMANCE,
)

# Context keys a module needs before it can answer from evidence.
REQUIRED_CONTEXT: Mapping[QueryType, Tuple[str, ...]] = MappingProxyType({
    QueryType.PERFORMANCE: ("telemetry",),
    QueryType.TECHNICAL: ("driver_preferences", "track_profile", "weather"),
    QueryType.STRATEGY: ("telemetry", "car_status", "driver_profile", "tire_data", "race_state"),  # competition is optional
    QueryType.EMOTION: ("audio_file",),
})


@dataclass(frozen=True)
class _Term:
    text: str
    query_type: Optional[QueryType]  # None for neutral phrases
    weight: int
    pattern: "re.Pattern[str]"


def _whole_word(body: str) -> "re.Pattern[str]":
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")


def _compile_term(term: str) -> "re.Pattern[str]":
    """Space: space/hyphen/nothing; hyphen: required hyphen or slash; one plural ending."""

    words = [r"\s*[-/]\s*".join(re.escape(part) for part in word.split("-")) for word in term.split()]
    plural = "(?:es)?" if term.endswith(("s", "x", "z", "ch", "sh")) else "s?"
    return _whole_word(r"[\s\-]*".join(words) + plural)


def _build_terms() -> Tuple[_Term, ...]:
    patterns = [_Term(text, kind, weight, _whole_word(regex)) for text, kind, weight, regex in PATTERN_TERMS]
    phrases = [_Term(p, None, 0, _compile_term(p)) for p in NEUTRAL_PHRASES]
    seen = {term.text for term in patterns + phrases}
    for query_type, vocabulary in VOCABULARY.items():
        for text, weight in vocabulary.items():
            if text in seen:
                raise RuntimeError(f"Routing term {text!r} is defined more than once")
            seen.add(text)
            phrases.append(_Term(text, query_type, weight, _compile_term(text)))
    # Pattern terms first, then the longest phrases, so they claim their characters first.
    return tuple(patterns) + tuple(sorted(phrases, key=lambda t: (-len(t.text), t.text)))


_TERMS = _build_terms()
_DASHES = str.maketrans(dict.fromkeys("‐‑‒–—―−", "-"))


def normalise_query_text(text: str) -> str:
    """Lower-case, strip accents, unify apostrophes and dashes, drop possessive "'s"."""

    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
    plain = plain.replace("’", "'").replace("‘", "'").translate(_DASHES)
    return re.sub(r"'s(?![a-z0-9])", "", plain)


def _normalised_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters")
    text = normalise_query_text(query)
    if len(text) > MAX_NORMALISED_QUERY_CHARS:
        raise ValueError(
            f"query expands to more than {MAX_NORMALISED_QUERY_CHARS} characters after Unicode normalisation"
        )
    return text


@dataclass(frozen=True)
class RoutingDecision:
    """Result of the keyword-heuristic classification, with diagnostics."""

    query_type: QueryType
    scores: Dict[str, int]
    matched_terms: Dict[str, List[str]]
    decision_rule: str
    tied_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "keyword heuristic (whole-word vocabulary match)",
            "query_type": self.query_type.value,
            "scores": dict(self.scores),
            "matched_terms": {key: list(value) for key, value in self.matched_terms.items()},
            "decision_rule": self.decision_rule,
            "tied_types": list(self.tied_types),
        }


def _match_terms(text: str) -> Dict[QueryType, Dict[str, int]]:
    """Return {query_type: {term: weight}} for non-overlapping whole-word matches."""

    claimed = bytearray(len(text))  # 1 = character already claimed by an earlier match
    hits: Dict[QueryType, Dict[str, int]] = {}
    for term in _TERMS:
        for match in term.pattern.finditer(text):
            start, end = match.span()
            if 1 in claimed[start:end]:
                continue
            claimed[start:end] = b"\x01" * (end - start)
            if term.query_type is not None:
                hits.setdefault(term.query_type, {})[term.text] = term.weight
    return hits


def _has_required_context(query_type: QueryType, context: Mapping[str, Any]) -> bool:
    return all(context.get(key) is not None for key in REQUIRED_CONTEXT.get(query_type, ()))


def classify_query(query: str, context: Optional[Mapping[str, Any]] = None) -> RoutingDecision:
    """Route a query with the documented keyword heuristic (see module docstring).

    Safe for untrusted input: a non-string or over-long query and a non-object
    context raise ValueError, and the work per call is bounded by the length
    limits.
    """

    text = _normalised_query(query)
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise ValueError("context must be an object")
    hits = _match_terms(text)
    scores = {kind.value: sum(hits.get(kind, {}).values()) for kind in ROUTING_PRIORITY}
    matched = {kind.value: sorted(hits[kind]) for kind in ROUTING_PRIORITY if kind in hits}

    def decide(kind: QueryType, rule: str, tied: Sequence[QueryType] = ()) -> RoutingDecision:
        return RoutingDecision(kind, scores, matched, rule, [t.value for t in tied])

    if not hits:
        return decide(QueryType.GENERAL, "no_vocabulary_match")
    if any(weight == CUE for weight in hits.get(QueryType.REGULATORY, {}).values()):
        return decide(QueryType.REGULATORY, "regulatory_cue")

    best = max(scores.values())
    tied = [kind for kind in ROUTING_PRIORITY if scores[kind.value] == best]
    if len(tied) == 1:
        return decide(tied[0], "highest_score")
    if QueryType.REGULATORY in tied:
        return decide(QueryType.REGULATORY, "tie_regulatory_priority", tied)
    supported = [kind for kind in tied if _has_required_context(kind, context)]
    if len(supported) == 1:
        return decide(supported[0], "tie_context_supplied", tied)
    return decide(tied[0], "tie_priority_order", tied)


# ---------------------------------------------------------------------------
# Performance-handler helpers
# ---------------------------------------------------------------------------

MAX_TELEMETRY_LAPS = 1000
MAX_SECTORS = 50
MAX_LAP_SECONDS = 3600.0  # sanity bound; even red-flagged laps are shorter
PERFORMANCE_MAX_CONFIDENCE = 0.8  # the router only restates supplied values
SUPPORTED_TELEMETRY_FIELDS = ("lap_times", "sector_times", "braking_consistency", "throttle_aggressiveness")

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3}
_NUM = r"(?:\d{1,2}|one|two|three)"
_NUMBER_LIST = rf"{_NUM}(?:\s*(?:,\s*(?:and\s+|or\s+)?|&|/|and|or|to|through|-)\s*{_NUM})*"
# Ordinals refer to the three official timing sectors of an F1 lap.
_SECTOR_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "middle": 2,
    "third": 3, "3rd": 3, "final": 3, "last": 3,
}
_ORDINAL = r"(?:first|1st|second|2nd|middle|third|3rd|final|last)"

_SECTOR_LIST_RE = re.compile(rf"(?<![a-z0-9])sectors?\s*({_NUMBER_LIST})(?![a-z0-9])")
# "lap 3", "laps 3 and 5", "laps 3-5", "lap #12": 1-based positions in telemetry.lap_times.
# "3 laps", "lap time(s)" and lap times such as "lap 1:31.2" / "lap 91.2" are not lap numbers.
_LAP_NUM = r"(?:\d{1,4}|one|two|three)"
_LAP_LIST_RE = re.compile(
    rf"(?<![a-z0-9])laps?\s*#?\s*({_LAP_NUM}(?:\s*(?:,\s*(?:and\s+|or\s+)?|&|/|and|or|to|through|-)\s*{_LAP_NUM})*)"
    r"(?![a-z0-9]|[.:]\d)"
)
_LAP_ASPECT_RE = re.compile(r"lap_(\d+)")
MAX_LAP_RANGE = 50  # "laps 3-10" is expanded; a longer range keeps only its end points
_SECTOR_SHORT_RE = re.compile(r"(?<![a-z0-9])s(\d{1,2})(?![a-z0-9])")
_SECTOR_ORDINAL_RE = re.compile(
    rf"(?<![a-z0-9])({_ORDINAL}(?:\s*(?:,|&|and|or)\s*{_ORDINAL})*)\s+(?:timed\s+|timing\s+)?sectors?(?![a-z0-9])"
)
_ANY_SECTOR_RE = re.compile(r"(?<![a-z0-9])sectors?(?![a-z0-9])")
_TURN_RE = re.compile(rf"(?<![a-z0-9])(?:turns?\s*({_NUMBER_LIST})|t(\d{{1,2}}))(?![a-z0-9])")
_CORNER_RE = re.compile(
    r"(?<![a-z0-9])(?:corners?|cornering|chicanes?|hairpins?|bends?|apex(?:es)?|apices|kerbs?|curbs?"
    r"|turn[\s\-]?ins?|turns(?!\s*(?:\d|one|two|three)))(?![a-z0-9])"
)
_STRAIGHT_RE = re.compile(
    r"(?<![a-z0-9])(?:straights|(?:the|a)\s+(?:back\s+|main\s+|pit\s+|long\s+)?straight"
    r"|straight[\s\-]?line|drs\s*zones?|speed\s*traps?)(?![a-z0-9])"
)
_LAP_RE = re.compile(
    r"(?<![a-z0-9])(?:laps?|lap\s*times?|laptimes?|pace|slow(?:er|est)?|fast(?:er|est)?|quick(?:er|est)?|delta)(?![a-z0-9])"
)
_BRAKING_RE = re.compile(r"(?<![a-z0-9])brak(?:e|es|ing)(?![a-z0-9])")
_THROTTLE_RE = re.compile(r"(?<![a-z0-9])(?:throttle|accelerat\w*)")
_SPEED_RE = re.compile(r"(?<![a-z0-9])speeds?(?![a-z0-9])")

# Comparisons against something outside the supplied telemetry ("slower than
# Verstappen", "compared to my teammate", "vs the car ahead").
_COMPARISON_RE = re.compile(
    r"(?<![a-z0-9])(?:(?<!rather )than|vs\.?|versus|against|relative\s+to"
    r"|compar(?:e|es|ed|ing)\s+(?:[a-z0-9]+\s+){0,3}?(?:to|with|against))"
    r"\s+([a-z0-9]{1,30})(?:\s+([a-z0-9]{1,30}))?"
)
_RIVAL_RE = re.compile(
    r"(?<![a-z0-9])(?:team\s*mates?|rivals?|opponents?|competitors?|other\s+(?:drivers?|cars?|teams?)"
    r"|cars?\s+(?:ahead|behind|in\s+front)|pole\s*sitters?|race\s+leaders?|the\s+leaders?|the\s+field"
    r"|everyone\s+else)(?![a-z0-9])"
)
# Comparison targets that refer to the driver's own supplied laps or to a quantity.
_OWN_TARGETS = frozenset({
    "i", "me", "mine", "myself", "we", "us", "ours", "it", "that", "this", "those", "these",
    "before", "earlier", "previously", "usual", "usually", "normal", "normally", "expected",
    "planned", "last", "previous", "first", "other", "lap", "laps", "sector", "sectors",
    "turn", "turns", "corner", "corners", "half", "one", "two", "three", "four", "five", "ten",
})
_DETERMINERS = frozenset({"my", "our", "the", "a", "an"})
_OWN_AFTER_DETERMINER = frozenset({
    "best", "fastest", "quickest", "slowest", "previous", "last", "latest", "first", "second",
    "third", "other", "own", "usual", "average", "earlier", "opening", "final", "lap", "laps",
    "time", "times", "sector", "sectors", "personal", "pb", "rest", "same", "start", "one",
    "tenth", "tenths", "hundredth", "hundredths", "thousandth", "thousandths", "bit", "lot",
    "little", "few", "couple", "half",
})
_OWN_NUMBER_RE = re.compile(r"(?:s|t|lap)?\d+(?:s|ms)?")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _duration(value: Any, name: str) -> float:
    seconds = _finite_number(value, name)
    if not 0.0 < seconds <= MAX_LAP_SECONDS:
        raise ValueError(f"{name} must be greater than 0 and at most {MAX_LAP_SECONDS:g} seconds")
    return seconds


def _unit_score(value: Any, name: str) -> float:
    score = _finite_number(value, name)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return score


def _validate_lap_times(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("telemetry.lap_times must be a list of lap times in seconds")
    if not raw:
        raise ValueError("telemetry.lap_times must contain at least one lap time")
    if len(raw) > MAX_TELEMETRY_LAPS:
        raise ValueError(f"telemetry.lap_times may contain at most {MAX_TELEMETRY_LAPS} laps")
    return [_duration(value, f"telemetry.lap_times[{index}]") for index, value in enumerate(raw)]


def _validate_sector_times(raw: Any) -> Optional[Dict[int, float]]:
    """Accept {"1": 27.1, ...} / {1: 27.1, ...} or a list ordered from sector 1."""

    if raw is None:
        return None
    if isinstance(raw, list):
        items = [(index + 1, value) for index, value in enumerate(raw)]
    elif isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            text = str(key).strip()
            if isinstance(key, bool) or not (text.isascii() and text.isdigit()):
                raise ValueError("telemetry.sector_times keys must be sector numbers such as '1', '2', '3'")
            items.append((int(text), value))
    else:
        raise ValueError("telemetry.sector_times must be an object keyed by sector number or a list")
    if not items:
        raise ValueError("telemetry.sector_times must contain at least one sector time")
    if len(items) > MAX_SECTORS:
        raise ValueError(f"telemetry.sector_times may contain at most {MAX_SECTORS} sectors")
    sectors: Dict[int, float] = {}
    for sector, value in items:
        if not 1 <= sector <= MAX_SECTORS:
            raise ValueError(f"telemetry.sector_times sector numbers must be between 1 and {MAX_SECTORS}")
        if sector in sectors:
            raise ValueError(f"telemetry.sector_times contains sector {sector} more than once")
        sectors[sector] = _duration(value, f"telemetry.sector_times[{sector}]")
    return sectors


def _parse_number_list(text: str, max_span: int = MAX_SECTORS) -> List[int]:
    """'1 and 2' -> [1, 2]; '1-3' / '1 to 3' -> [1, 2, 3]; 'one, two' -> [1, 2].

    A range longer than ``max_span`` keeps only its end points.
    """

    result: List[int] = []
    in_range = False
    for token in re.findall(r"\d+|one|two|three|through|to|-", text):
        if token in ("to", "through", "-"):
            in_range = bool(result)
            continue
        value = _NUMBER_WORDS[token] if token in _NUMBER_WORDS else int(token)
        if in_range and result[-1] < value <= result[-1] + max_span:
            result.extend(range(result[-1] + 1, value + 1))
        else:
            result.append(value)
        in_range = False
    return result


def _requested_laps(text: str) -> List[int]:
    found: List[int] = []
    for match in _LAP_LIST_RE.finditer(text):
        found.extend(_parse_number_list(match.group(1), MAX_LAP_RANGE))
    return sorted(set(found))


def _requested_sectors(text: str) -> List[int]:
    found: List[int] = []
    for match in _SECTOR_LIST_RE.finditer(text):
        found.extend(_parse_number_list(match.group(1)))
    found.extend(int(number) for number in _SECTOR_SHORT_RE.findall(text))
    for match in _SECTOR_ORDINAL_RE.finditer(text):
        found.extend(_SECTOR_ORDINALS[word] for word in re.findall(_ORDINAL, match.group(1)))
    return sorted(set(found))


def _requested_turns(text: str) -> List[int]:
    found: List[int] = []
    for match in _TURN_RE.finditer(text):
        found.extend(_parse_number_list(match.group(1)) if match.group(1) else [int(match.group(2))])
    return sorted(set(found))


def _comparison_targets(text: str) -> List[str]:
    """Comparison targets outside the supplied telemetry (another driver, car or reference)."""

    targets = [match.group(0) for match in _RIVAL_RE.finditer(text)]
    for match in _COMPARISON_RE.finditer(text):
        first, second = match.group(1), match.group(2)
        if first in _DETERMINERS:
            if second is None or second in _OWN_AFTER_DETERMINER or _OWN_NUMBER_RE.fullmatch(second):
                continue
            targets.append(f"{first} {second}")
        elif first not in _OWN_TARGETS and not _OWN_NUMBER_RE.fullmatch(first):
            targets.append(first)
    return list(dict.fromkeys(targets))


# Aspects no supported telemetry field can answer; always reported as not supplied.
_UNSUPPORTED_ASPECTS = ("speed", "corners", "straights", "comparison_target")


def _is_unsupported_aspect(aspect: str) -> bool:
    return aspect in _UNSUPPORTED_ASPECTS or aspect.startswith("turn_")


def _requested_aspects(text: str) -> List[str]:
    """Which telemetry aspects the question asks about (lap times by default).

    Heuristic keyword parsing of the normalised question. Lap references
    ("lap 3", "laps 3 and 5", "laps 3-5") become ``lap_N`` aspects: N is the
    1-based position in telemetry.lap_times. Sector references
    may be numbers or lists ("sectors 1 and 2", "sectors 1-3", "S2") or
    ordinals of the three official timing sectors ("second sector" = 2,
    "middle" = 2, "final"/"last" = 3). Turns, corners, straights, speed and
    comparisons with other drivers or external references are recorded as
    aspects that the supported telemetry fields cannot answer.
    """

    aspects = [f"lap_{number}" for number in _requested_laps(text)]
    sectors = _requested_sectors(text)
    aspects += [f"sector_{number}" for number in sectors]
    if not sectors and _ANY_SECTOR_RE.search(text):
        aspects.append("sector_times")
    if _BRAKING_RE.search(text):
        aspects.append("braking_consistency")
    if _THROTTLE_RE.search(text):
        aspects.append("throttle_aggressiveness")
    if _SPEED_RE.search(text):
        aspects.append("speed")
    aspects.extend(f"turn_{number}" for number in _requested_turns(text))
    if _CORNER_RE.search(text):
        aspects.append("corners")
    if _STRAIGHT_RE.search(text):
        aspects.append("straights")
    if _comparison_targets(text):
        aspects.append("comparison_target")
    if _LAP_RE.search(text) or not aspects:
        aspects.insert(0, "lap_times")
    return aspects


LAP_NUMBERING = "lap N = the N-th value of telemetry.lap_times, counting from 1"


def _named_lap_parts(
    numbers: Sequence[int], lap_times: Optional[List[float]], evidence: Dict[str, Any], missing: List[str]
) -> List[str]:
    """Answer the laps a question names ("lap 3", "laps 3 and 5") from telemetry.lap_times.

    Supplied laps are restated with their differences; lap numbers beyond the supplied list
    are added to ``missing`` and named in the answer.
    """

    parts: List[str] = []
    found: Dict[int, float] = {}
    for number in numbers:
        if lap_times and 1 <= number <= len(lap_times):
            found[number] = lap_times[number - 1]
            evidence[f"lap_{number}"] = found[number]
            continue
        missing.append(f"lap_{number}")
        if lap_times:
            supplied = f"telemetry.lap_times holds {len(lap_times)} lap{'s' if len(lap_times) != 1 else ''}"
        else:
            supplied = "telemetry.lap_times was not supplied"
        parts.append(f"Lap {number} time was not supplied: {supplied} ({LAP_NUMBERING}).")
    if not found:
        return parts

    listed = "; ".join(f"lap {number} {seconds:.3f}s" for number, seconds in found.items())
    sentence = f"Supplied times for the laps you named: {listed}."
    if len(found) >= 2:
        slow, slow_time = max(found.items(), key=lambda item: item[1])
        fast, fast_time = min(found.items(), key=lambda item: item[1])
        if slow_time == fast_time:
            sentence += f" They are equal ({slow_time:.3f}s)."
        elif len(found) == 2:
            sentence += f" Lap {slow} was {slow_time - fast_time:.3f}s slower than lap {fast}."
        else:
            sentence += (
                f" Slowest of these: lap {slow}; fastest: lap {fast}; "
                f"lap {slow} was {slow_time - fast_time:.3f}s slower than lap {fast}."
            )
    return [sentence] + parts


class NaturalQueryProcessor:
    """Classify a query and route it to a module that has the required evidence.

    The processor holds no mutable state, so one instance can serve concurrent
    requests.
    """

    def process_natural_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryResult:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters")
        if context is None:
            context = {}
        if not isinstance(context, dict):
            raise ValueError("context must be an object")

        decision = classify_query(query, context)
        handlers = {
            QueryType.PERFORMANCE: self._handle_performance_query,
            QueryType.REGULATORY: self._handle_regulatory_query,
            QueryType.TECHNICAL: self._handle_technical_query,
            QueryType.STRATEGY: self._handle_strategy_query,
            QueryType.EMOTION: self._handle_emotion_query,
            QueryType.GENERAL: self._handle_general_query,
        }
        result = handlers[decision.query_type](query, context)
        return replace(result, routing=decision.to_dict())

    def _handle_performance_query(self, query: str, context: Dict[str, Any]) -> QueryResult:
        """Restate supplied telemetry values; never infer or invent missing ones.

        Confidence is an evidence-coverage heuristic: PERFORMANCE_MAX_CONFIDENCE
        times the share of the aspects the question asks about (see
        ``_requested_aspects``: lap times, lap N, sector N, braking, throttle,
        speed, turns/corners, straights, comparisons with others) that the
        supplied telemetry covers. Lap N is the N-th value of
        telemetry.lap_times; a lap number beyond that list is not supplied.
        Aspects outside SUPPORTED_TELEMETRY_FIELDS are named in the answer and
        listed under "unsupported" and "not_supplied".
        """

        telemetry = context.get("telemetry")
        if telemetry is None:
            return QueryResult(
                "I need telemetry in context.telemetry to answer that performance question.",
                QueryType.PERFORMANCE,
                0.0,
                [],
            )
        if not isinstance(telemetry, dict):
            raise ValueError("context.telemetry must be an object")

        lap_times = _validate_lap_times(telemetry.get("lap_times"))
        sector_times = _validate_sector_times(telemetry.get("sector_times"))
        scores = {
            name: _unit_score(telemetry[name], f"telemetry.{name}")
            for name in ("braking_consistency", "throttle_aggressiveness")
            if telemetry.get(name) is not None
        }
        text = normalise_query_text(query)
        requested = _requested_aspects(text)
        fields = ", ".join(SUPPORTED_TELEMETRY_FIELDS)

        parts: List[str] = []
        missing: List[str] = []
        evidence: Dict[str, Any] = {}

        summary: Optional[str] = None
        if lap_times:
            best, latest = min(lap_times), lap_times[-1]
            summary = f"Latest lap: {latest:.3f}s; best supplied lap: {best:.3f}s; delta: {latest - best:+.3f}s."
            evidence["lap_times"] = lap_times
        elif "lap_times" in requested:
            missing.append("lap_times")
            summary = "Lap times were not supplied (telemetry.lap_times), so I cannot compare laps."
        named_laps = [int(match.group(1)) for match in map(_LAP_ASPECT_RE.fullmatch, requested) if match]
        # The laps the question names come first; the general lap summary follows as context.
        parts.extend(_named_lap_parts(named_laps, lap_times, evidence, missing))
        if summary:
            parts.append(summary)

        for aspect in requested:
            if aspect == "sector_times":
                if sector_times:
                    listed = ", ".join(f"S{n} {t:.3f}s" for n, t in sorted(sector_times.items()))
                    parts.append(f"Supplied sector times: {listed}.")
                    evidence["sector_times"] = {str(n): t for n, t in sorted(sector_times.items())}
                else:
                    missing.append(aspect)
                    parts.append("Sector times were not supplied (telemetry.sector_times).")
            elif aspect.startswith("sector_"):
                number = int(aspect.split("_")[1])
                if sector_times and number in sector_times:
                    parts.append(f"Supplied Sector {number} time: {sector_times[number]:.3f}s.")
                    evidence[aspect] = sector_times[number]
                else:
                    missing.append(aspect)
                    parts.append(
                        f"Sector {number} time was not supplied (telemetry.sector_times), "
                        f"so I cannot say anything specific about sector {number}."
                    )
            elif aspect == "speed":
                missing.append(aspect)
                parts.append(f"Speed data is not analysed here; supported telemetry fields are {fields}.")
            elif aspect.startswith("turn_"):
                number = int(aspect.split("_")[1])
                missing.append(aspect)
                parts.append(
                    f"Turn {number} data was not supplied: corner-by-corner data is not part of the supported "
                    f"telemetry ({fields}), so I cannot say anything specific about turn {number}."
                )
            elif aspect == "corners":
                missing.append(aspect)
                parts.append(
                    f"Corner-level data was not supplied: it is not part of the supported telemetry ({fields}), "
                    "so I cannot analyse individual corners."
                )
            elif aspect == "straights":
                missing.append(aspect)
                parts.append(
                    f"Straight-line data was not supplied: it is not part of the supported telemetry ({fields}), "
                    "so I cannot analyse the straights."
                )
            elif aspect == "comparison_target":
                missing.append(aspect)
                targets = ", ".join(f"'{target}'" for target in _comparison_targets(text))
                parts.append(
                    f"Data for the comparison with {targets} was not supplied: the telemetry holds only your own "
                    "laps, so I cannot compare against another driver, car or external reference."
                )

        for name, label in (("braking_consistency", "Braking consistency"), ("throttle_aggressiveness", "Throttle aggressiveness")):
            if name in scores:
                parts.append(f"{label} input: {scores[name]:.2f}.")
                evidence[name] = scores[name]
            elif name in requested:
                missing.append(name)
                parts.append(f"{label} was not supplied (telemetry.{name}).")

        unsupported = [aspect for aspect in missing if _is_unsupported_aspect(aspect)]
        if not evidence:
            if not any(telemetry.get(name) is not None for name in SUPPORTED_TELEMETRY_FIELDS):
                parts.insert(0, f"Telemetry was supplied, but it contains none of the supported fields ({fields}).")
            return QueryResult(
                " ".join(parts),
                QueryType.PERFORMANCE,
                0.0,
                [],
                {"requested": requested, "not_supplied": missing, "unsupported": unsupported, "evidence": {}},
            )

        coverage = (len(requested) - len(missing)) / len(requested)
        details: Dict[str, Any] = {
            "requested": requested,
            "not_supplied": missing,
            "unsupported": unsupported,
            "evidence": evidence,
            "method": "restates supplied telemetry values; no causal analysis",
            "confidence_basis": (
                "heuristic: share of the requested aspects covered by supplied telemetry "
                f"x {PERFORMANCE_MAX_CONFIDENCE}; not a statistical probability"
            ),
        }
        if named_laps:
            details["lap_numbering"] = LAP_NUMBERING
        return QueryResult(
            " ".join(parts),
            QueryType.PERFORMANCE,
            round(PERFORMANCE_MAX_CONFIDENCE * coverage, 3),
            ["telemetry"],
            details,
        )

    @staticmethod
    def _handle_regulatory_query(query: str, context: Dict[str, Any]) -> QueryResult:
        # RAGUnavailableError propagates so callers can report "service unavailable"
        # instead of presenting a failure as an answer.
        from core_modules.rule_checker.fia_rag import get_fia_rag

        result = get_fia_rag().answer(query)
        return QueryResult(
            answer=result["answer"],
            query_type=QueryType.REGULATORY,
            confidence=float(result["confidence"]),
            data_sources=["fia_regulations"] if result["grounded"] else [],
            additional_context={
                "grounded": result["grounded"],
                "decline_reason": result["decline_reason"],
                "citations": result["citations"],
                "referenced_rules": result["referenced_rules"],
                "retrieved_passages": result["retrieved_passages"],
                "top_retrieval_score": result["top_retrieval_score"],
            },
        )

    @staticmethod
    def _handle_technical_query(query: str, context: Dict[str, Any]) -> QueryResult:
        """Setup search on the supplied context, validated exactly like /api/setup/recommend."""

        required = REQUIRED_CONTEXT[QueryType.TECHNICAL]
        if not all(context.get(key) is not None for key in required):
            return QueryResult(
                "For a setup recommendation I need driver_preferences, track_profile and weather in the query context.",
                QueryType.TECHNICAL,
                0.0,
                [],
            )
        from core_modules.setup_optimizer.schemas import SetupRequest
        from core_modules.setup_optimizer.setup_recommender import recommend_setup_from_inputs

        fields = (*required, "n_trials", "seed")
        request = SetupRequest.model_validate({key: context[key] for key in fields if key in context})
        setup = recommend_setup_from_inputs(request.to_engine_inputs())
        answer = (
            f"Heuristic setup search suggests ride height {setup['ride_height']:.1f} mm, front/rear wing "
            f"{setup['front_wing_angle']:.1f}°/{setup['rear_wing_angle']:.1f}° and brake bias {setup['brake_bias']:.1f}% "
            f"(objective {setup['objective_value']:.3f}; baseline {setup['baseline_objective_value']:.3f}). {setup['reasoning']}"
        )
        return QueryResult(answer, QueryType.TECHNICAL, float(setup["confidence"]), ["setup_optimizer"], setup)

    @staticmethod
    def _handle_strategy_query(query: str, context: Dict[str, Any]) -> QueryResult:
        """Strategy candidates for the supplied context, validated exactly like /api/strategy/generate."""

        required = REQUIRED_CONTEXT[QueryType.STRATEGY]
        if not all(context.get(key) is not None for key in required):
            return QueryResult(
                "To generate a race strategy I need " + ", ".join(required) + " in the query context.",
                QueryType.STRATEGY,
                0.0,
                [],
            )
        from core_modules.strategy_optimizer.schemas import StrategyRequest, strategy_result_to_dict
        from core_modules.strategy_optimizer.strategy_engine import generate_strategy

        result = generate_strategy(**StrategyRequest.from_context(context).to_engine_inputs())
        best = result.best
        stops = f"{best.stop_count} stop" + ("" if best.stop_count == 1 else "s")
        laps = [str(lap) for lap in best.pit_laps]
        pit_laps = (" and ".join([", ".join(laps[:-1]), laps[-1]]) if len(laps) > 1 else "".join(laps)) or "none"
        answer = (
            f"Fastest heuristic plan: {best.strategy_id} with {stops}, compounds "
            f"{' → '.join(c.value for c in best.tire_compounds)}, pit laps {pit_laps}, projected "
            f"remaining-race time {best.projected_race_time:.1f}s. {best.notes[0] if best.notes else ''}"
        ).strip()
        # The engine reports time margins between plans, not a probability, so no confidence is claimed.
        return QueryResult(answer, QueryType.STRATEGY, None, ["strategy_engine"], strategy_result_to_dict(result))

    @staticmethod
    def _handle_emotion_query(query: str, context: Dict[str, Any]) -> QueryResult:
        """Classify base64 audio from context.audio_file; server file paths are refused."""

        audio = context.get("audio_file")
        if audio is None or (isinstance(audio, str) and not audio.strip()):
            return QueryResult(
                "I need audio_file (base64 audio or a base64 data URI) in the query context to analyse driver-radio emotion.",
                QueryType.EMOTION,
                0.0,
                [],
            )
        if not isinstance(audio, str):
            raise ValueError("context.audio_file must be a base64 string or base64 data URI")
        transcribe = context.get("transcribe", False)
        if not isinstance(transcribe, bool):
            raise ValueError("context.transcribe must be a boolean")

        from core_modules.driver_emotion.emotion_classifier import classify_emotion_detailed

        result = classify_emotion_detailed(audio, transcribe=transcribe, allow_local_paths=False)
        confidence = result["confidence"]
        return QueryResult(
            f"The {result['classifier']} classifier labelled the clip '{result['emotion']}' "
            f"(heuristic score {confidence:.2f}, not a calibrated probability).",
            QueryType.EMOTION,
            confidence,
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


_query_processor = NaturalQueryProcessor()


def process_natural_query(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = _query_processor.process_natural_query(query, context)
    return {
        "answer": result.answer,
        "query_type": result.query_type.value,
        "confidence": result.confidence,
        "data_sources": result.data_sources,
        "additional_context": result.additional_context,
        "routing": result.routing,
    }
