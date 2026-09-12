#!/usr/bin/env python3
"""Retrieval-augmented QA over FIA Formula 1 regulation PDFs.

The pipeline deliberately keeps ingestion/indexing, retrieval, and generation
separate so each stage can be inspected and tuned independently.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:  # compatibility with the pinned legacy LangChain release
        from langchain.chat_models import ChatOpenAI
        from langchain.embeddings import OpenAIEmbeddings
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    RAG_DEPENDENCIES_AVAILABLE = True
    RAG_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # allows the rest of the application to import cleanly
    RAG_DEPENDENCIES_AVAILABLE = False
    RAG_IMPORT_ERROR = exc


DECLINE_ANSWER = (
    "I cannot answer that from the indexed FIA regulations because the retrieved "
    "context does not contain sufficient evidence."
)


@dataclass(frozen=True)
class RetrievedPassage:
    text: str
    score: float
    source: str
    page: Optional[int]
    chunk_id: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["page"] = self.page
        return data


class FIAKnowledgeBase:
    """FIA regulation RAG pipeline backed by Qdrant."""

    def __init__(
        self,
        fia_docs_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> None:
        self.fia_docs_path = Path(fia_docs_path or os.getenv("FIA_DOCS_PATH", "data/fia_docs"))
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.collection_name = collection_name or os.getenv("FIA_RAG_COLLECTION", "fia_regulations")
        self.top_k = top_k or int(os.getenv("FIA_RAG_TOP_K", "5"))
        self.min_score = min_score if min_score is not None else float(os.getenv("FIA_RAG_MIN_SCORE", "0.30"))
        self.embedding_model = os.getenv("FIA_RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("FIA_RAG_MODEL", "gpt-4o-mini")
        self.chunk_size = int(os.getenv("FIA_RAG_CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("FIA_RAG_CHUNK_OVERLAP", "200"))

        if self.top_k <= 0:
            raise ValueError("FIA_RAG_TOP_K must be positive")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("FIA_RAG_MIN_SCORE must be between 0 and 1")
        if self.chunk_size <= 0:
            raise ValueError("FIA_RAG_CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("FIA_RAG_CHUNK_OVERLAP must be >= 0 and smaller than chunk size")

        self.embeddings = None
        self.llm = None
        self.qdrant = None
        self.is_initialized = False
        self.indexed_chunks = 0
        self.last_error: Optional[str] = None

    # ------------------------- initialization / ingestion -------------------------
    def initialize(self, force_reindex: bool = False) -> None:
        """Connect dependencies and build the vector index if necessary."""
        if self.is_initialized and not force_reindex:
            return
        if not RAG_DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"RAG dependencies are unavailable: {RAG_IMPORT_ERROR}")
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for FIA RAG queries")

        pdf_files = sorted(glob.glob(str(self.fia_docs_path / "*.pdf")))
        if not pdf_files:
            raise RuntimeError(
                f"No FIA regulation PDFs found in {self.fia_docs_path}. "
                "Run scripts/fetch_fia_regulations.py before querying the RAG system."
            )

        try:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model, openai_api_key=self.openai_api_key)
            self.llm = ChatOpenAI(model=self.chat_model, openai_api_key=self.openai_api_key, temperature=0)
            self.qdrant = self._make_qdrant_client()

            collection_exists = self._collection_exists()
            if force_reindex or not collection_exists:
                self._build_index(pdf_files)
            else:
                self.indexed_chunks = self._collection_count()
                if self.indexed_chunks == 0:
                    self._build_index(pdf_files)

            self.is_initialized = True
            self.last_error = None
        except Exception as exc:
            self.is_initialized = False
            self.last_error = str(exc)
            logger.exception("Failed to initialize FIA RAG pipeline")
            raise

    def _make_qdrant_client(self):
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        if qdrant_url:
            return QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)
        return QdrantClient(path=os.getenv("QDRANT_PATH", ".qdrant"))

    def _collection_exists(self) -> bool:
        try:
            self.qdrant.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _collection_count(self) -> int:
        result = self.qdrant.count(collection_name=self.collection_name, exact=True)
        return int(result.count)

    def _load_and_chunk_documents(self, pdf_files: List[str]):
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = Path(pdf_file).name
            documents.extend(pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(documents)

    def _build_index(self, pdf_files: List[str]) -> None:
        chunks = self._load_and_chunk_documents(pdf_files)
        if not chunks:
            raise RuntimeError("FIA PDFs were found, but no text could be extracted")

        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        if not vectors or not vectors[0]:
            raise RuntimeError("Embedding generation returned no vectors")

        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )

        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            source = str(chunk.metadata.get("source", "unknown"))
            page = chunk.metadata.get("page")
            chunk_key = f"{source}:{page}:{index}:{chunk.page_content}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={"text": chunk.page_content, "source": source, "page": page, "chunk_id": chunk_id},
                )
            )

        batch_size = 64
        for start in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )
        self.indexed_chunks = len(points)
        logger.info("Indexed %s FIA regulation chunks", self.indexed_chunks)

    # -------------------------------- retrieval ---------------------------------
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedPassage]:
        """Return top-k regulation passages without invoking the answer model."""
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        requested_k = top_k or self.top_k
        if requested_k <= 0:
            raise ValueError("top_k must be positive")

        self.initialize()
        query_vector = self.embeddings.embed_query(question)
        hits = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=requested_k,
            with_payload=True,
        )

        passages: List[RetrievedPassage] = []
        for hit in hits:
            payload = hit.payload or {}
            page = payload.get("page")
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            passages.append(
                RetrievedPassage(
                    text=text,
                    score=float(hit.score),
                    source=str(payload.get("source", "unknown")),
                    page=(int(page) + 1) if page is not None else None,
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                )
            )
        return passages

    # -------------------------------- generation --------------------------------
    @staticmethod
    def _format_context(passages: List[RetrievedPassage]) -> str:
        blocks = []
        for i, passage in enumerate(passages, start=1):
            page = f", page {passage.page}" if passage.page is not None else ""
            blocks.append(
                f"[S{i}] {passage.source}{page}\n"
                f"Retrieval score: {passage.score:.3f}\n"
                f"{passage.text}"
            )
        return "\n\n".join(blocks)

    def generate_answer(self, question: str, passages: List[RetrievedPassage]) -> str:
        """Generate an answer from already-retrieved evidence."""
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        if not passages or max(p.score for p in passages) < self.min_score:
            return DECLINE_ANSWER

        context = self._format_context(passages)
        prompt = f"""You are an FIA Formula 1 regulations assistant.

Use ONLY the retrieved regulation excerpts below. Do not use memory or general F1 knowledge.
If the excerpts do not contain enough evidence to answer the question, reply exactly:
{DECLINE_ANSWER}

When you do answer:
- cite the relevant article/section number exactly as it appears in the excerpts;
- attach one or more source labels such as [S1] or [S2] to each material claim;
- do not invent article numbers, penalties, thresholds, dates, or exceptions;
- distinguish a regulation statement from any inference you make.

Retrieved regulation excerpts:
{context}

Question: {question}
Answer:"""
        response = self.llm.invoke(prompt)
        return str(getattr(response, "content", response)).strip()

    def query(self, question: str) -> Dict[str, Any]:
        """Retrieve evidence first, then generate a grounded answer."""
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        try:
            passages = self.retrieve(question)
            top_score = max((p.score for p in passages), default=0.0)
            enough_evidence = bool(passages) and top_score >= self.min_score
            answer = self.generate_answer(question, passages)
            grounded = enough_evidence and answer != DECLINE_ANSWER
            # This is an evidence-strength proxy based on vector similarity, not a
            # calibrated probability that the generated statement is correct.
            confidence = max(0.0, min(1.0, top_score)) if grounded else 0.0
            return {
                "answer": answer,
                "source": "fia_rag_agent",
                "grounded": grounded,
                "confidence": round(confidence, 4),
                "top_retrieval_score": round(top_score, 4),
                "retrieved_passages": [p.to_dict() for p in passages],
                "referenced_rules": self._extract_rules(answer),
                "citations": sorted(set(re.findall(r"\[S\d+\]", answer))),
            }
        except ValueError:
            raise
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("FIA RAG query failed")
            return {
                "answer": f"FIA RAG is unavailable: {exc}",
                "source": "fia_rag_agent",
                "grounded": False,
                "confidence": 0.0,
                "top_retrieval_score": 0.0,
                "retrieved_passages": [],
                "referenced_rules": [],
                "citations": [],
            }

    def status(self) -> Dict[str, Any]:
        pdf_count = len(glob.glob(str(self.fia_docs_path / "*.pdf")))
        configured = RAG_DEPENDENCIES_AVAILABLE and bool(self.openai_api_key) and pdf_count > 0
        return {
            "ready": self.is_initialized,
            "configured": configured,
            "dependencies_available": RAG_DEPENDENCIES_AVAILABLE,
            "api_key_configured": bool(self.openai_api_key),
            "docs_path": str(self.fia_docs_path),
            "pdf_count": pdf_count,
            "collection": self.collection_name,
            "indexed_chunks": self.indexed_chunks,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "last_error": self.last_error,
        }

    @staticmethod
    def _extract_rules(answer: str) -> List[str]:
        matches = re.findall(
            r"\b(?:Article|Regulation|Section)\s+\d+(?:\.\d+)*",
            answer,
            flags=re.IGNORECASE,
        )
        return sorted(set(matches))


_fia_kb: Optional[FIAKnowledgeBase] = None


def get_fia_knowledge_base() -> FIAKnowledgeBase:
    global _fia_kb
    if _fia_kb is None:
        _fia_kb = FIAKnowledgeBase()
    return _fia_kb


def query_fia_regulations(question: str) -> str:
    return get_fia_knowledge_base().query(question)["answer"]


def query_fia_regulations_detailed(question: str) -> Dict[str, Any]:
    return get_fia_knowledge_base().query(question)


def retrieve_fia_passages(question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Expose retrieval independently for debugging and evaluation."""
    passages = get_fia_knowledge_base().retrieve(question, top_k=top_k)
    return [passage.to_dict() for passage in passages]


if __name__ == "__main__":
    kb = get_fia_knowledge_base()
    print(kb.status())
    for question in (
        "What are the rules for an unsafe release?",
        "When can a driver use DRS?",
        "What does the regulation say about track limits?",
    ):
        print("\nQuestion:", question)
        print(kb.query(question))
