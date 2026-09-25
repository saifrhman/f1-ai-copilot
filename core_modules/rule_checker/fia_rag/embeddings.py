"""Embedding generation with explicit validation of provider output.

The same model embeds indexed chunks and incoming questions; the model name is
part of the index fingerprint, so switching models forces a rebuild instead of
comparing vectors from different embedding spaces.
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import threading
from array import array
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import EmbeddingConfig, ProviderConfig
from .errors import ProviderError, RAGConfigurationError, describe_provider_error

logger = logging.getLogger(__name__)


def create_openai_embeddings(config: EmbeddingConfig, provider: ProviderConfig):
    """LangChain ``OpenAIEmbeddings`` for OpenAI or any OpenAI-compatible endpoint."""

    if not provider.api_key:
        raise RAGConfigurationError("OPENAI_API_KEY is required for FIA RAG embeddings")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=config.model,
        api_key=provider.api_key,
        base_url=provider.base_url,
        chunk_size=config.batch_size,
        # Send plain strings. Token-array inputs (tiktoken pre-tokenisation) are
        # an OpenAI-only feature that OpenAI-compatible providers reject; chunks
        # are far below embedding context limits anyway.
        check_embedding_ctx_length=False,
        max_retries=provider.max_retries,
        request_timeout=provider.timeout_seconds,
    )


class EmbeddingCache:
    """Persistent store of provider embeddings keyed by (model, SHA-256 of the exact text).

    Document vectors are kept indefinitely (they are what an index rebuild needs).
    Query vectors go to a separate table capped at ``max_queries`` rows (oldest
    evicted), so arbitrary questions cannot grow the file without bound. Vectors
    are stored as float32, the precision in which the OpenAI SDK returns them.
    A damaged row is treated as a cache miss and removed; database errors never
    fail a request - the provider is called instead.
    """

    def __init__(self, path: Path, max_queries: int = 5000):
        self.path = Path(path)
        self.max_queries = max_queries
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            for table in ("embeddings", "query_embeddings"):
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} (model TEXT, text_sha256 TEXT, vector BLOB, "
                    "PRIMARY KEY (model, text_sha256))"
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _read(self, tables: Sequence[str], model: str, texts: Sequence[str]) -> List[Optional[List[float]]]:
        keys = [self._key(text) for text in texts]
        found: Dict[str, List[float]] = {}
        damaged: List[Tuple[str, str]] = []
        try:
            with self._lock, self._connect() as connection:
                for table in tables:
                    wanted = [key for key in keys if key not in found]
                    for start in range(0, len(wanted), 500):
                        batch = wanted[start : start + 500]
                        rows = connection.execute(
                            f"SELECT text_sha256, vector FROM {table} WHERE model = ? AND text_sha256 IN ({','.join('?' * len(batch))})",
                            [model, *batch],
                        ).fetchall()
                        for key, blob in rows:
                            if isinstance(blob, bytes) and blob and len(blob) % 4 == 0:
                                found[key] = array("f", blob).tolist()
                            else:
                                damaged.append((table, key))
                for table, key in damaged:
                    connection.execute(f"DELETE FROM {table} WHERE model = ? AND text_sha256 = ?", (model, key))
        except sqlite3.Error as exc:
            logger.warning("Embedding cache read failed (%s); calling the provider instead", exc)
            return [None] * len(keys)
        return [found.get(key) for key in keys]

    def _write(self, table: str, model: str, texts: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        rows = [(model, self._key(text), array("f", vector).tobytes()) for text, vector in zip(texts, vectors)]
        try:
            with self._lock, self._connect() as connection:
                connection.executemany(f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?)", rows)
                if table == "query_embeddings":
                    connection.execute(
                        "DELETE FROM query_embeddings WHERE rowid NOT IN "
                        "(SELECT rowid FROM query_embeddings ORDER BY rowid DESC LIMIT ?)",
                        (self.max_queries,),
                    )
        except sqlite3.Error as exc:
            logger.warning("Embedding cache write failed (%s); continuing without caching", exc)

    def get_many(self, model: str, texts: Sequence[str]) -> List[Optional[List[float]]]:
        return self._read(["embeddings"], model, texts)

    def put_many(self, model: str, texts: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        self._write("embeddings", model, texts, vectors)

    def get_query(self, model: str, text: str) -> Optional[List[float]]:
        # Identical model + text gives an identical vector, so a document row may serve a query too.
        return self._read(["query_embeddings", "embeddings"], model, [text])[0]

    def put_query(self, model: str, text: str, vector: Sequence[float]) -> None:
        self._write("query_embeddings", model, [text], [vector])


def open_embedding_cache(path: Path) -> Tuple[Optional[EmbeddingCache], Optional[str]]:
    """Open the cache, or return (None, reason) when it cannot be used (the RAG then works uncached)."""

    try:
        return EmbeddingCache(path), None
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Embedding cache %s unavailable: %s", path, exc)
        return None, f"{type(exc).__name__}: {exc}"


class EmbeddingService:
    """Wraps a LangChain ``Embeddings`` object; never substitutes vectors on failure."""

    def __init__(self, embeddings, config: EmbeddingConfig, cache: Optional[EmbeddingCache] = None):
        self._embeddings = embeddings
        self.config = config
        self.cache = cache
        self.provider_requests = 0

    @property
    def model(self) -> str:
        return self.config.model

    def embed_documents(
        self, texts: Sequence[str], progress: Optional[Callable[[int, int], None]] = None
    ) -> List[List[float]]:
        total = len(texts)
        cached = self.cache.get_many(self.model, texts) if self.cache else [None] * total
        vectors: List[Optional[List[float]]] = list(cached)
        missing = [i for i, vector in enumerate(cached) if vector is None]
        if progress and total - len(missing):
            progress(total - len(missing), total)
        for start in range(0, len(missing), self.config.batch_size):
            indices = missing[start : start + self.config.batch_size]
            batch = [texts[i] for i in indices]
            result = self._call(lambda: self._embeddings.embed_documents(batch), "embed_documents")
            self.provider_requests += 1
            if not isinstance(result, list) or len(result) != len(batch):
                raise ProviderError(
                    f"Embedding provider returned {len(result) if isinstance(result, list) else type(result).__name__} "
                    f"vectors for {len(batch)} inputs"
                )
            validated = [self._validate(vector) for vector in result]
            self._check_dimensions([v for v in vectors if v is not None] + validated)
            if self.cache:
                self.cache.put_many(self.model, batch, validated)
            for index, vector in zip(indices, validated):
                vectors[index] = vector
            if progress:
                progress(total - len(missing) + start + len(batch), total)
        final = [self._validate(vector) for vector in vectors]  # cached vectors are re-validated too
        self._check_dimensions(final)
        return final

    def embed_query(self, text: str) -> List[float]:
        if self.cache:
            hit = self.cache.get_query(self.model, text)
            if hit is not None:
                try:
                    return self._validate(hit)
                except ProviderError:
                    logger.warning("Ignoring an invalid cached query vector")
        vector = self._validate(self._call(lambda: self._embeddings.embed_query(text), "embed_query"))
        self.provider_requests += 1
        if self.cache:
            self.cache.put_query(self.model, text, vector)
        return vector

    def _call(self, fn, operation: str):
        try:
            return fn()
        except (RAGConfigurationError, ProviderError):
            raise
        except Exception as exc:  # classify SDK/network/auth failures at the provider boundary
            raise ProviderError(
                f"Embedding provider call ({operation}, model {self.model}) failed: {describe_provider_error(exc)}"
            ) from exc

    @staticmethod
    def _validate(vector) -> List[float]:
        try:
            values = [float(x) for x in vector]
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"Embedding provider returned a non-numeric vector: {exc}") from exc
        if not values:
            raise ProviderError("Embedding provider returned an empty vector")
        if not all(math.isfinite(x) for x in values):
            raise ProviderError("Embedding provider returned non-finite values")
        if not any(values):
            raise ProviderError("Embedding provider returned an all-zero vector")
        return values

    @staticmethod
    def _check_dimensions(vectors: Sequence[Sequence[float]]) -> None:
        sizes = {len(v) for v in vectors}
        if len(sizes) > 1:
            raise ProviderError(f"Embedding provider returned vectors of inconsistent sizes: {sorted(sizes)}")


