"""Retrieval: question -> ranked evidence passages, without any answer generation.

This stage depends only on :class:`RetrievalConfig`, an embedding service and
the vector index. ``top_k`` controls retrieval depth; ``min_score`` is a
per-passage cosine-similarity floor (higher similarity = more relevant), and
passages below it are reported separately and never treated as evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import RetrievalConfig, validate_min_score, validate_top_k
from .embeddings import EmbeddingService
from .glossary import GlossaryEntry
from .index import QdrantIndex

MAX_QUESTION_CHARS = 2000


@dataclass(frozen=True)
class RetrievedPassage:
    chunk_id: str
    text: str
    score: float
    source: str
    page: Optional[int]
    page_label: Optional[str] = None
    section: Optional[str] = None
    source_url: Optional[str] = None
    nearest_rule: Optional[str] = None
    rule_ids: Tuple[str, ...] = ()
    # "regulation": a retrieved chunk (score = cosine similarity); "definition": the official
    # definition of a term used in the retrieved chunks (not retrieved by similarity; score 0).
    kind: str = "regulation"
    defined_term: Optional[str] = None

    @classmethod
    def from_glossary(cls, entry: GlossaryEntry) -> "RetrievedPassage":
        term = f"{entry.term} ({entry.abbreviation})" if entry.abbreviation and entry.abbreviation != entry.term else entry.term
        return cls(
            chunk_id=f"definition:{entry.section or '?'}:{entry.term}",
            text=entry.definition,
            score=0.0,
            source=entry.source,
            page=entry.page,
            page_label=entry.page_label,
            section=entry.section,
            source_url=entry.source_url,
            kind="definition",
            defined_term=term,
        )

    @classmethod
    def from_payload(cls, point_id: Any, score: float, payload: Dict[str, Any]) -> "RetrievedPassage":
        page = payload.get("page")
        return cls(
            chunk_id=str(payload.get("chunk_id") or point_id),
            text=str(payload.get("text", "")).strip(),
            score=float(score),
            source=str(payload.get("source", "unknown")),
            page=int(page) if page is not None else None,
            page_label=payload.get("page_label"),
            section=payload.get("section"),
            source_url=payload.get("source_url"),
            nearest_rule=payload.get("nearest_rule"),
            rule_ids=tuple(payload.get("rule_ids") or ()),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rule_ids"] = list(self.rule_ids)
        data["score"] = round(self.score, 4)
        return data


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    top_k: int
    min_score: float
    passages: List[RetrievedPassage]
    below_threshold: List[RetrievedPassage] = field(default_factory=list)
    duplicates_removed: int = 0
    # Definitions of terms used in ``passages``, attached by the pipeline from the index glossary.
    definitions: List[RetrievedPassage] = field(default_factory=list)

    def with_threshold(self, min_score: float) -> "RetrievalResult":
        """Re-split the same ranked passages at another threshold (no new embedding or search).

        Definitions are dropped: they depend on which passages are accepted, and
        the pipeline derives them again when the result is answered.
        """

        threshold = validate_min_score(min_score)
        ranked = sorted(self.passages + self.below_threshold, key=lambda p: p.score, reverse=True)
        return RetrievalResult(
            question=self.question,
            top_k=self.top_k,
            min_score=threshold,
            passages=[p for p in ranked if p.score >= threshold],
            below_threshold=[p for p in ranked if p.score < threshold],
            duplicates_removed=self.duplicates_removed,
        )

    @property
    def top_score(self) -> float:
        scores = [p.score for p in self.passages + self.below_threshold]
        return max(scores) if scores else 0.0

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        def render(passage: RetrievedPassage) -> Dict[str, Any]:
            data = passage.to_dict()
            if not include_text:
                data.pop("text")
            return data

        return {
            "question": self.question,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "top_score": round(self.top_score, 4),
            "passages": [render(p) for p in self.passages],
            "below_threshold": [render(p) for p in self.below_threshold],
            "definitions": [render(p) for p in self.definitions],
            "duplicates_removed": self.duplicates_removed,
        }


def validate_question(question: str) -> str:
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters")
    return question


def _dedup_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


class Retriever:
    def __init__(self, embedder: EmbeddingService, index: QdrantIndex, config: RetrievalConfig, vector_size: Optional[int] = None):
        self.embedder = embedder
        self.index = index
        self.config = config
        self.vector_size = vector_size

    def retrieve(self, question: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> RetrievalResult:
        question = validate_question(question)
        k = self.config.top_k if top_k is None else validate_top_k(top_k)
        threshold = self.config.min_score if min_score is None else validate_min_score(min_score)

        vector = self.embedder.embed_query(question)
        hits = self.index.search(vector, limit=k, expected_size=self.vector_size)

        ranked = sorted(
            (RetrievedPassage.from_payload(hit.id, hit.score, hit.payload or {}) for hit in hits),
            key=lambda passage: passage.score,
            reverse=True,
        )
        unique: List[RetrievedPassage] = []
        seen = set()
        duplicates = 0
        for passage in ranked:
            if not passage.text:
                continue
            key = _dedup_key(passage.text)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            unique.append(passage)

        accepted = [p for p in unique if p.score >= threshold]
        rejected = [p for p in unique if p.score < threshold]
        return RetrievalResult(
            question=question,
            top_k=k,
            min_score=threshold,
            passages=accepted,
            below_threshold=rejected,
            duplicates_removed=duplicates,
        )
