"""Orchestration of the FIA regulation RAG stages.

``build_index``: documents -> chunks -> embeddings -> Qdrant, plus the definitions
                 glossary (explicit, offline step)
``retrieve``:    question -> ranked evidence + definitions of the terms it uses
                 (no answer model involved)
``answer``:      retrieve -> grounded generation -> validation

Components are created lazily from :class:`RAGSettings`; tests and experiments
can inject their own embeddings/chat model/Qdrant client. Production code never
substitutes fake components: missing configuration raises an error.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import RAGSettings
from .embeddings import EmbeddingService, create_openai_embeddings, open_embedding_cache
from .errors import IndexNotReadyError, RAGUnavailableError
from .generation import GroundedAnswerGenerator, create_chat_model
from .glossary import GLOSSARY_VERSION, GlossaryEntry, definitions_for, extract_glossary, with_frequencies
from .index import IndexState, QdrantIndex, compute_fingerprint, create_qdrant_client
from .ingestion import DiscoveryResult, discover_documents, load_and_chunk, load_chunks_and_pages
from .retrieval import RetrievalResult, RetrievedPassage, Retriever

logger = logging.getLogger(__name__)

BUILD_COMMAND = "python scripts/build_fia_index.py"


@dataclass
class IndexBuildReport:
    status: str  # "up_to_date" | "rebuilt" | "glossary_rebuilt"
    fingerprint: str
    chunks: int
    definitions: int = 0
    documents: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "fingerprint": self.fingerprint,
            "chunks": self.chunks,
            "definitions": self.definitions,
            "documents": self.documents,
            "warnings": self.warnings,
            "seconds": round(self.seconds, 1),
        }


class FIARegulationRAG:
    def __init__(self, settings: RAGSettings, *, embeddings=None, llm=None, qdrant_client=None):
        self.settings = settings
        self._embeddings_override = embeddings
        self._llm_override = llm
        self._qdrant_override = qdrant_client
        self._embedder: Optional[EmbeddingService] = None
        self._llm = None
        self._index: Optional[QdrantIndex] = None
        self._glossary: Optional[Tuple[str, List[GlossaryEntry]]] = None
        self._lock = threading.RLock()
        self.last_error: Optional[str] = None
        self.last_error_type: Optional[str] = None
        self.last_error_at: Optional[str] = None
        self.last_success_at: Optional[str] = None
        self.cache_problem: Optional[str] = None

    def _record_failure(self, exc: Exception) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.last_error_type = type(exc).__name__
        self.last_error_at = datetime.now(timezone.utc).isoformat()

    def _record_success(self) -> None:
        # A later success means the component recovered; do not keep reporting an old error.
        self.last_error = None
        self.last_error_type = None
        self.last_error_at = None
        self.last_success_at = datetime.now(timezone.utc).isoformat()

    def _rebuild_hint(self) -> str:
        if self.settings.qdrant.mode == "local":
            return (
                f"Run `{BUILD_COMMAND}` to (re)build it (stop the API first: embedded Qdrant storage "
                "can only be opened by one process; or set QDRANT_URL to use a Qdrant server)."
            )
        return f"Run `{BUILD_COMMAND}` to (re)build it (if a rebuild is already running, wait for it to finish)."

    # ----------------------------------------------------------------- components
    def embedder(self) -> EmbeddingService:
        with self._lock:
            if self._embedder is None:
                embeddings = self._embeddings_override or create_openai_embeddings(
                    self.settings.embedding, self.settings.provider
                )
                cache = None
                if self.settings.embedding.cache_path:
                    cache, self.cache_problem = open_embedding_cache(self.settings.embedding.cache_path)
                self._embedder = EmbeddingService(embeddings, self.settings.embedding, cache)
            return self._embedder

    def chat_model(self):
        with self._lock:
            if self._llm is None:
                self._llm = self._llm_override or create_chat_model(self.settings.generation, self.settings.provider)
            return self._llm

    def index(self) -> QdrantIndex:
        with self._lock:
            if self._index is None:
                client = self._qdrant_override or create_qdrant_client(self.settings.qdrant)
                self._index = QdrantIndex(client, self.settings.qdrant.collection)
            return self._index

    # ----------------------------------------------------------------- indexing
    def discover(self) -> DiscoveryResult:
        return discover_documents(self.settings.docs_path)

    def fingerprint(self, discovery: Optional[DiscoveryResult] = None) -> str:
        discovery = discovery or self.discover()
        return compute_fingerprint(discovery.documents, self.settings.chunking, self.settings.embedding.model)

    @staticmethod
    def glossary_key(fingerprint: str) -> str:
        return f"{fingerprint}:glossary-v{GLOSSARY_VERSION}"

    def glossary(self, fingerprint: str) -> Optional[List[GlossaryEntry]]:
        """The stored glossary for this index fingerprint (cached in memory), or None if it was not built."""

        key = self.glossary_key(fingerprint)
        with self._lock:
            if self._glossary is not None and self._glossary[0] == key:
                return self._glossary[1]
            entries = self.index().load_glossary(key)
            if entries is not None:
                self._glossary = (key, entries)
            return entries

    def _store_glossary(self, fingerprint: str, chunks, pages) -> int:
        entries = with_frequencies(extract_glossary(pages), [chunk.text for chunk in chunks])
        key = self.glossary_key(fingerprint)
        count = self.index().rebuild_glossary(entries, key)
        self._glossary = (key, entries)
        return count

    def plan_index(self) -> Dict[str, Any]:
        """Parse and chunk the documents without calling the embedding API (a dry run)."""

        discovery = self.discover()
        chunks, reports = load_and_chunk(discovery.documents, self.settings.chunking)
        characters = sum(len(chunk.text_for_embedding()) for chunk in chunks)
        batch = self.settings.embedding.batch_size
        return {
            "fingerprint": self.fingerprint(discovery),
            "documents": [report.to_dict() for report in reports],
            "warnings": discovery.warnings,
            "chunks": len(chunks),
            "characters": characters,
            "estimated_embedding_tokens": characters // 4,
            "embedding_requests": -(-len(chunks) // batch),
            "embedding_model": self.settings.embedding.model,
        }

    def build_index(
        self, force: bool = False, progress: Optional[Callable[[int, int], None]] = None
    ) -> IndexBuildReport:
        """Build the index unless an index for exactly these inputs already exists."""

        started = time.monotonic()
        with self._lock:
            discovery = self.discover()
            fingerprint = self.fingerprint(discovery)
            index = self.index()
            state = index.inspect()
            if not force and state.status_for(fingerprint) == "current":
                glossary = self.glossary(fingerprint)
                if glossary is not None:
                    return IndexBuildReport(
                        status="up_to_date",
                        fingerprint=fingerprint,
                        chunks=state.points,
                        definitions=len(glossary),
                        warnings=discovery.warnings,
                        seconds=time.monotonic() - started,
                    )
                # Chunks are current but the glossary is missing or from an older
                # extractor version: rebuild only the glossary (no embedding calls).
                chunks, reports, pages = load_chunks_and_pages(discovery.documents, self.settings.chunking)
                definitions = self._store_glossary(fingerprint, chunks, pages)
                return IndexBuildReport(
                    status="glossary_rebuilt",
                    fingerprint=fingerprint,
                    chunks=state.points,
                    definitions=definitions,
                    documents=[report.to_dict() for report in reports],
                    warnings=discovery.warnings,
                    seconds=time.monotonic() - started,
                )
            chunks, reports, pages = load_chunks_and_pages(discovery.documents, self.settings.chunking)
            # Embed everything before touching the existing collection, so a
            # provider failure leaves the previous index intact.
            vectors = self.embedder().embed_documents([chunk.text_for_embedding() for chunk in chunks], progress=progress)
            # The glossary goes live first, so that readers in other processes never see the new
            # chunk index without its glossary (each switch-over is atomic, the pair is not).
            definitions = self._store_glossary(fingerprint, chunks, pages)
            count = index.rebuild(chunks, vectors, fingerprint)
            self._record_success()
            return IndexBuildReport(
                status="rebuilt",
                fingerprint=fingerprint,
                chunks=count,
                definitions=definitions,
                documents=[report.to_dict() for report in reports],
                warnings=discovery.warnings,
                seconds=time.monotonic() - started,
            )

    def _ready_index_state(self) -> Tuple[IndexState, str]:
        fingerprint = self.fingerprint()
        state = self.index().inspect()
        status = state.status_for(fingerprint)
        if status != "current":
            explanation = {
                "missing": "has not been built",
                "empty": "is empty",
                "stale": "was built from different documents, chunking settings or embedding model",
                "incomplete": f"is incomplete ({state.points} of {state.expected_points} chunks)",
            }[status]
            raise IndexNotReadyError(f"The FIA regulation index {explanation}. {self._rebuild_hint()}")
        if self.settings.retrieval.max_definitions and self.glossary(fingerprint) is None:
            raise IndexNotReadyError(
                "The FIA regulation definitions glossary has not been built for this index "
                f"(or was built by an older version). {self._rebuild_hint()} "
                "Or set FIA_RAG_MAX_DEFINITIONS=0 to answer without definitions."
            )
        return state, fingerprint

    def definitions(self, retrieval: RetrievalResult, fingerprint: Optional[str] = None) -> List[RetrievedPassage]:
        """Official definitions of the abbreviations used in the accepted passages / terms named in the question."""

        limit = self.settings.retrieval.max_definitions
        if not limit or not retrieval.passages:
            return []
        glossary = self.glossary(fingerprint or self.fingerprint()) or []
        entries = definitions_for(
            glossary,
            [p.text for p in retrieval.passages],
            [p.section for p in retrieval.passages],
            retrieval.question,
            limit,
        )
        return [RetrievedPassage.from_glossary(entry) for entry in entries]

    # ----------------------------------------------------------------- querying
    def retrieve(self, question: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> RetrievalResult:
        try:
            state, fingerprint = self._ready_index_state()
            retriever = Retriever(self.embedder(), self.index(), self.settings.retrieval, vector_size=state.vector_size)
            result = retriever.retrieve(question, top_k=top_k, min_score=min_score)
            result = replace(result, definitions=self.definitions(result, fingerprint))
        except RAGUnavailableError as exc:
            self._record_failure(exc)
            raise
        self._record_success()
        return result

    def answer(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        return self.answer_from_retrieval(self.retrieve(question, top_k=top_k))

    def answer_from_retrieval(self, retrieval: RetrievalResult) -> Dict[str, Any]:
        """Generate and validate an answer for an already-inspected retrieval result."""

        question = retrieval.question
        try:
            # Recomputed rather than taken from ``retrieval``: the accepted passages may have been re-split.
            definitions = self.definitions(retrieval)
            chat_model = self.chat_model() if retrieval.passages else None
            result = GroundedAnswerGenerator(chat_model, self.settings.generation).generate(
                question, retrieval.passages, definitions
            )
        except RAGUnavailableError as exc:
            self._record_failure(exc)
            raise
        self._record_success()
        validation = result.validation
        cited = set(validation.citations)
        passages = []
        for label, passage in result.labelled.items():
            item = passage.to_dict()
            item["label"] = label
            item["cited"] = label in cited
            passages.append(item)
        # Definitions are not retrieved by similarity, so they do not contribute to the evidence score.
        cited_scores = [p.score for label, p in result.labelled.items() if label in cited and p.kind == "regulation"]
        return {
            "question": retrieval.question,
            "answer": validation.answer,
            "grounded": validation.grounded,
            "status": validation.status,
            "decline_reason": validation.reason,
            # Evidence-strength proxy (best cosine similarity among cited regulation passages),
            # not a calibrated probability that the answer is correct.
            "confidence": round(max(cited_scores), 4) if validation.grounded and cited_scores else 0.0,
            "citations": validation.citations,
            "referenced_rules": validation.referenced_rules,
            "retrieved_passages": passages,
            "top_retrieval_score": round(retrieval.top_score, 4),
            "retrieval": {
                "top_k": retrieval.top_k,
                "min_score": retrieval.min_score,
                "passages_above_threshold": len(retrieval.passages),
                "definitions_added": len(definitions),
                "below_threshold": [p.to_dict() for p in retrieval.below_threshold],
                "duplicates_removed": retrieval.duplicates_removed,
            },
            "validation": {
                "invalid_citations": validation.invalid_citations,
                "unsupported_rules": validation.unsupported_rules,
                "uncited_claims": validation.uncited_claims,
                "unsupported_numbers": validation.unsupported_numbers,
                "unverified_claims": validation.unverified_claims,
                # None when FIA_RAG_VERIFY_CLAIMS is off or the answer was declined before verification.
                "claim_verification": result.verification,
                # Raw model text is only exposed when it was rejected, for inspection.
                "rejected_model_output": None if validation.grounded else validation.model_output,
            },
            "models": {
                "embedding": self.settings.embedding.model,
                "generation": self.settings.generation.model if retrieval.passages else None,
            },
            "source": "fia_rag",
        }

    # ----------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        """Report configuration and index readiness without calling the model APIs."""

        report: Dict[str, Any] = {"settings": self.settings.summary()}
        problems: List[str] = []
        needs_key = self._embeddings_override is None or self._llm_override is None
        if needs_key and not self.settings.provider.api_key:
            problems.append("OPENAI_API_KEY is not set")
        documents: List[Dict[str, Any]] = []
        fingerprint = None
        try:
            discovery = self.discover()
            documents = [
                {"filename": d.filename, "section": d.section, "sha256": d.sha256, "bytes": d.size_bytes}
                for d in discovery.documents
            ]
            report["document_warnings"] = discovery.warnings
            fingerprint = self.fingerprint(discovery)
        except RAGUnavailableError as exc:
            problems.append(str(exc))
        report["documents"] = documents
        index_status = "unknown"
        state: Optional[IndexState] = None
        if fingerprint is not None:
            try:
                state = self.index().inspect()
                index_status = state.status_for(fingerprint)
            except RAGUnavailableError as exc:
                problems.append(str(exc))
                index_status = "unavailable"
        if index_status not in ("current", "unknown", "unavailable"):
            problems.append(f"index is {index_status}. {self._rebuild_hint()}")
        glossary_report: Dict[str, Any] = {"status": "unknown", "entries": None}
        if index_status == "current" and fingerprint is not None:
            try:
                entries = self.glossary(fingerprint)
                glossary_report = {"status": "current" if entries is not None else "missing", "entries": len(entries or [])}
            except RAGUnavailableError as exc:
                problems.append(str(exc))
                glossary_report["status"] = "unavailable"
            if glossary_report["status"] == "missing" and self.settings.retrieval.max_definitions:
                problems.append(f"the definitions glossary is missing. {self._rebuild_hint()}")
        # The most recent model-provider call failed and nothing has succeeded since.
        provider_failing = self.last_error_type == "ProviderError"
        if provider_failing:
            problems.append(f"the most recent model-provider call failed: {self.last_error}")
        report["index"] = {
            "status": index_status,
            "expected_fingerprint": fingerprint,
            **(state.to_dict() if state else {}),
            "glossary": glossary_report,
        }
        report["ready"] = not problems and index_status == "current"
        report["problems"] = problems
        report["provider_status"] = "failing" if provider_failing else ("ok" if self.last_success_at else "unknown")
        report["embedding_cache_problem"] = self.cache_problem
        report["last_error"] = self.last_error
        report["last_error_at"] = self.last_error_at
        report["last_success_at"] = self.last_success_at
        return report


_instance: Optional[FIARegulationRAG] = None
_instance_lock = threading.Lock()


def get_fia_rag() -> FIARegulationRAG:
    """Process-wide pipeline built from environment variables (raises on invalid settings)."""

    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = FIARegulationRAG(RAGSettings.from_env())
        return _instance


def reset_fia_rag() -> None:
    global _instance
    with _instance_lock:
        _instance = None
