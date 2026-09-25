"""Configuration for the FIA regulation RAG pipeline.

Each stage receives only its own settings object:

* :class:`ChunkingConfig`   -> ingestion (PDF text -> chunks)
* :class:`EmbeddingConfig`  -> embedding model used for indexing *and* queries
* :class:`RetrievalConfig`  -> top-k depth and minimum similarity score
* :class:`GenerationConfig` -> chat model used to write grounded answers

so chunking, retrieval depth and prompting can be changed independently.
Relative paths are resolved against the project root, not the current working
directory, so the API and scripts agree on where documents and the index live.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from .errors import RAGConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: str | Path) -> str:
    """Path for API output and messages: project-relative, or only the final name when outside the project."""

    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def redact_url(url: Optional[str]) -> Optional[str]:
    """URL without user:password, query string or fragment (they can carry credentials)."""

    if not url:
        return url
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _valid_timeout(value: float, name: str, maximum: float = 600.0) -> None:
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise RAGConfigurationError(f"{name} must be a finite number of seconds in (0, {maximum:g}]")


@dataclass(frozen=True)
class ChunkingConfig:
    """Character-based chunking of extracted page text."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Chunks with fewer alphanumeric characters than this are discarded as
    # layout debris (stray page numbers, isolated list markers).
    min_chunk_chars: int = 20

    def __post_init__(self) -> None:
        if self.chunk_size < 50:
            raise RAGConfigurationError("FIA_RAG_CHUNK_SIZE must be at least 50 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise RAGConfigurationError("FIA_RAG_CHUNK_OVERLAP must be >= 0 and smaller than FIA_RAG_CHUNK_SIZE")
        if not 0 <= self.min_chunk_chars < self.chunk_size:
            raise RAGConfigurationError("min_chunk_chars must be >= 0 and smaller than the chunk size")


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    # Number of texts sent per embedding request while indexing.
    batch_size: int = 128
    # SQLite cache of provider vectors keyed by model + exact text; None disables it.
    cache_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise RAGConfigurationError("FIA_RAG_EMBEDDING_MODEL cannot be empty")
        if not 1 <= self.batch_size <= 2048:
            raise RAGConfigurationError("FIA_RAG_EMBEDDING_BATCH_SIZE must be between 1 and 2048")


@dataclass(frozen=True)
class RetrievalConfig:
    # 8 was chosen from the end-to-end evaluation: at 5, comparative questions
    # spanning two documents lost the second document's passage (ranked 7th).
    top_k: int = 8
    # Minimum cosine similarity for a passage to count as evidence. The useful
    # value depends on the embedding model and should be calibrated for it.
    min_score: float = 0.30
    # Official definitions of abbreviations used in the accepted passages (or of terms
    # named in the question) added to the evidence; 0 disables the glossary.
    max_definitions: int = 3

    MAX_TOP_K = 50
    MAX_DEFINITIONS = 10

    def __post_init__(self) -> None:
        try:
            validate_top_k(self.top_k)
            validate_min_score(self.min_score)
        except ValueError as exc:
            raise RAGConfigurationError(f"Invalid FIA_RAG_TOP_K/FIA_RAG_MIN_SCORE setting: {exc}") from exc
        if isinstance(self.max_definitions, bool) or not isinstance(self.max_definitions, int) or not (
            0 <= self.max_definitions <= self.MAX_DEFINITIONS
        ):
            raise RAGConfigurationError(f"FIA_RAG_MAX_DEFINITIONS must be an integer between 0 and {self.MAX_DEFINITIONS}")


def validate_top_k(value: int) -> int:
    """Validate a retrieval depth (ValueError: a caller error; config wraps it as RAGConfigurationError)."""

    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= RetrievalConfig.MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {RetrievalConfig.MAX_TOP_K}")
    return value


def validate_min_score(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("min_score must be a number between 0 and 1")
    return float(value)


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    # Second model call that checks each sentence of an accepted answer against the
    # excerpts it cites (entailment); doubles the chat requests per answered question.
    verify_claims: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise RAGConfigurationError("FIA_RAG_MODEL cannot be empty")
        if not math.isfinite(self.temperature) or not 0.0 <= self.temperature <= 2.0:
            raise RAGConfigurationError("FIA_RAG_TEMPERATURE must be between 0 and 2")
        if self.max_output_tokens is not None and self.max_output_tokens < 16:
            raise RAGConfigurationError("FIA_RAG_MAX_OUTPUT_TOKENS must be at least 16")


@dataclass(frozen=True)
class ProviderConfig:
    """OpenAI-compatible endpoint used for embeddings and chat (OpenAI, OpenRouter, ...)."""

    api_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = None
    timeout_seconds: float = 120.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        _valid_timeout(self.timeout_seconds, "FIA_RAG_TIMEOUT_SECONDS")
        if not 0 <= self.max_retries <= 10:
            raise RAGConfigurationError("FIA_RAG_MAX_RETRIES must be between 0 and 10")


@dataclass(frozen=True)
class QdrantConfig:
    collection: str = "fia_regulations"
    url: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    path: Path = field(default_factory=lambda: resolve_project_path(".qdrant"))
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not _COLLECTION_NAME.match(self.collection):
            raise RAGConfigurationError("FIA_RAG_COLLECTION may only contain letters, digits, '_' and '-'")
        _valid_timeout(self.timeout_seconds, "Qdrant timeout")

    @property
    def mode(self) -> str:
        return "remote" if self.url else "local"


@dataclass(frozen=True)
class RAGSettings:
    docs_path: Path = field(default_factory=lambda: resolve_project_path("data/fia_docs"))
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RAGSettings":
        """Build settings from environment variables, rejecting invalid values explicitly."""

        env = os.environ if env is None else env
        reader = _EnvReader(env)
        return cls(
            docs_path=resolve_project_path(reader.text("FIA_DOCS_PATH", "data/fia_docs")),
            chunking=ChunkingConfig(
                chunk_size=reader.integer("FIA_RAG_CHUNK_SIZE", 1000),
                chunk_overlap=reader.integer("FIA_RAG_CHUNK_OVERLAP", 200),
            ),
            embedding=EmbeddingConfig(
                model=reader.text("FIA_RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
                batch_size=reader.integer("FIA_RAG_EMBEDDING_BATCH_SIZE", 128),
                cache_path=reader.optional_path("FIA_RAG_EMBEDDING_CACHE", ".cache/fia_embeddings.sqlite3"),
            ),
            retrieval=RetrievalConfig(
                top_k=reader.integer("FIA_RAG_TOP_K", 8),
                min_score=reader.number("FIA_RAG_MIN_SCORE", 0.30),
                max_definitions=reader.integer("FIA_RAG_MAX_DEFINITIONS", 3),
            ),
            generation=GenerationConfig(
                model=reader.text("FIA_RAG_MODEL", "gpt-4o-mini"),
                temperature=reader.number("FIA_RAG_TEMPERATURE", 0.0),
                max_output_tokens=reader.optional_integer("FIA_RAG_MAX_OUTPUT_TOKENS"),
                verify_claims=reader.boolean("FIA_RAG_VERIFY_CLAIMS", False),
            ),
            provider=ProviderConfig(
                api_key=reader.optional_text("OPENAI_API_KEY"),
                base_url=reader.optional_text("OPENAI_BASE_URL"),
                timeout_seconds=reader.number("FIA_RAG_TIMEOUT_SECONDS", 120.0),
                max_retries=reader.integer("FIA_RAG_MAX_RETRIES", 2),
            ),
            qdrant=QdrantConfig(
                collection=reader.text("FIA_RAG_COLLECTION", "fia_regulations"),
                url=reader.optional_text("QDRANT_URL"),
                api_key=reader.optional_text("QDRANT_API_KEY"),
                path=resolve_project_path(reader.text("QDRANT_PATH", ".qdrant")),
            ),
        )

    def summary(self) -> dict:
        """Non-secret view of the effective configuration."""

        return {
            "docs_path": display_path(self.docs_path),
            "collection": self.qdrant.collection,
            "qdrant_mode": self.qdrant.mode,
            "qdrant_location": redact_url(self.qdrant.url) if self.qdrant.url else display_path(self.qdrant.path),
            "chunk_size": self.chunking.chunk_size,
            "chunk_overlap": self.chunking.chunk_overlap,
            "top_k": self.retrieval.top_k,
            "min_score": self.retrieval.min_score,
            "max_definitions": self.retrieval.max_definitions,
            "embedding_model": self.embedding.model,
            "embedding_cache": display_path(self.embedding.cache_path) if self.embedding.cache_path else None,
            "generation_model": self.generation.model,
            "verify_claims": self.generation.verify_claims,
            "provider_base_url": redact_url(self.provider.base_url) or "https://api.openai.com/v1",
            "api_key_configured": bool(self.provider.api_key),
        }


class _EnvReader:
    def __init__(self, env: Mapping[str, str]):
        self.env = env

    def _raw(self, name: str) -> Optional[str]:
        value = self.env.get(name)
        if value is None:
            return None
        value = value.strip()
        return value or None

    def text(self, name: str, default: str) -> str:
        return self._raw(name) or default

    def optional_text(self, name: str) -> Optional[str]:
        return self._raw(name)

    def integer(self, name: str, default: int) -> int:
        raw = self._raw(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise RAGConfigurationError(f"{name} must be an integer, got {raw!r}") from exc

    def boolean(self, name: str, default: bool) -> bool:
        raw = self._raw(name)
        if raw is None:
            return default
        value = raw.lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise RAGConfigurationError(f"{name} must be true or false, got {raw!r}")

    def optional_path(self, name: str, default: str) -> Optional[Path]:
        """A project-relative path; "off"/"none"/"false" disables the feature."""

        raw = self._raw(name)
        if raw is not None and raw.lower() in {"off", "none", "false", "0"}:
            return None
        return resolve_project_path(raw or default)

    def optional_integer(self, name: str) -> Optional[int]:
        raw = self._raw(name)
        return None if raw is None else self.integer(name, 0)

    def number(self, name: str, default: float) -> float:
        raw = self._raw(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError as exc:
            raise RAGConfigurationError(f"{name} must be a number, got {raw!r}") from exc
