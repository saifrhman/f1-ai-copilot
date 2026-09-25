"""Qdrant vector index with build fingerprints.

Every point stores the fingerprint of the inputs the index was built from
(document hashes, chunking settings, embedding model, ingestion version) and
the expected number of points. An index is only used when both match, so a
changed chunk size, a new regulation issue, a different embedding model or an
interrupted build is detected instead of silently serving stale vectors.
Rebuilding replaces the collection, so repeated ingestion never duplicates chunks;
the swap to the new collection is atomic for concurrent readers (see QdrantIndex).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

from .config import ChunkingConfig, QdrantConfig, display_path, redact_url
from .errors import VectorStoreError
from .glossary import GlossaryEntry
from .ingestion import INGESTION_VERSION, Chunk, SourceDocument

logger = logging.getLogger(__name__)

INDEX_SCHEMA_VERSION = "2"
_UPSERT_BATCH = 128
_GLOSSARY_META_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "f1-ai-copilot/glossary-meta"))
# Failures raised by qdrant-client for remote (HTTP) and local (embedded) modes.
_QDRANT_ERRORS = (ApiException, httpx.HTTPError, OSError, RuntimeError, ValueError)

_local_clients: Dict[str, QdrantClient] = {}
_local_clients_lock = threading.Lock()


def create_qdrant_client(config: QdrantConfig) -> QdrantClient:
    """Remote client when QDRANT_URL is set, otherwise embedded persistent storage.

    Embedded storage can be opened by only one client at a time, so a single
    client per storage path is shared within the process.
    """

    try:
        if config.url:
            return QdrantClient(url=config.url, api_key=config.api_key, timeout=int(config.timeout_seconds))
        key = str(Path(config.path).resolve())
        with _local_clients_lock:
            client = _local_clients.get(key)
            if client is None:
                Path(key).mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=key)
                _local_clients[key] = client
            return client
    except _QDRANT_ERRORS as exc:
        hint = ""
        if "already accessed" in str(exc):
            hint = (
                " Another process (for example the API server) is using the local Qdrant storage; "
                "stop it or run a Qdrant server and set QDRANT_URL for multi-process access."
            )
        location = redact_url(config.url) if config.url else display_path(config.path)
        message = str(exc).replace(str(Path(config.path).resolve()), location)
        raise VectorStoreError(f"Could not open Qdrant ({location}): {message}.{hint}") from exc


def close_qdrant_clients() -> None:
    """Release embedded-storage locks (on shutdown, or before reopening the same path)."""

    with _local_clients_lock:
        for client in _local_clients.values():
            client.close()
        _local_clients.clear()


def compute_fingerprint(
    documents: Sequence[SourceDocument], chunking: ChunkingConfig, embedding_model: str
) -> str:
    payload = {
        "schema": INDEX_SCHEMA_VERSION,
        "ingestion": INGESTION_VERSION,
        # Manifest metadata is stored in every payload, so it is part of the build inputs too.
        "documents": sorted([doc.filename, doc.sha256, doc.section or "", doc.source_url or ""] for doc in documents),
        "chunk_size": chunking.chunk_size,
        "chunk_overlap": chunking.chunk_overlap,
        "min_chunk_chars": chunking.min_chunk_chars,
        "embedding_model": embedding_model,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IndexState:
    exists: bool
    points: int = 0
    expected_points: Optional[int] = None
    fingerprint: Optional[str] = None
    vector_size: Optional[int] = None
    distance: Optional[str] = None

    def status_for(self, fingerprint: str) -> str:
        """``missing`` | ``empty`` | ``incomplete`` | ``stale`` | ``current``."""

        if not self.exists:
            return "missing"
        if self.points == 0:
            return "empty"
        if self.fingerprint != fingerprint:
            return "stale"
        if self.expected_points != self.points:
            return "incomplete"
        return "current"

    def to_dict(self) -> Dict[str, object]:
        return {
            "exists": self.exists,
            "points": self.points,
            "expected_points": self.expected_points,
            "fingerprint": self.fingerprint,
            "vector_size": self.vector_size,
            "distance": self.distance,
        }


class QdrantIndex:
    """The chunk index and its glossary, each served under a Qdrant alias.

    ``collection`` (and ``<collection>_glossary``) name aliases, not collections.
    A rebuild writes a new backing collection ``<name>__<12 hex>``, checks the
    stored point count, and then repoints the alias with a single
    ``update_collection_aliases`` call. Readers in other processes (API workers
    during ``build_fia_index.py --force`` against a Qdrant server) therefore see
    either the complete previous index or the complete new one, never an empty
    or half-uploaded collection. A failed or interrupted rebuild deletes its own
    backing collection and leaves the previous index serving; only a killed
    process (SIGKILL) can leave an unused ``<name>__<hex>`` collection behind.
    Two rebuilds must not run at the same time: the later swap wins.
    """

    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection
        self._lock = threading.RLock()

    def inspect(self) -> IndexState:
        with self._lock:
            try:
                if not self.client.collection_exists(self.collection):
                    return IndexState(exists=False)
                info = self.client.get_collection(self.collection)
                vectors = info.config.params.vectors
                vector_size = getattr(vectors, "size", None)
                distance = getattr(getattr(vectors, "distance", None), "value", None)
                points = int(self.client.count(self.collection, exact=True).count)
                sample, _ = self.client.scroll(
                    self.collection,
                    limit=1,
                    with_payload=["index_fingerprint", "index_chunk_count"],
                    with_vectors=False,
                )
            except _QDRANT_ERRORS as exc:
                raise VectorStoreError(f"Qdrant inspection of collection {self.collection!r} failed: {exc}") from exc
        payload = (sample[0].payload or {}) if sample else {}
        expected = payload.get("index_chunk_count")
        return IndexState(
            exists=True,
            points=points,
            expected_points=int(expected) if expected is not None else None,
            fingerprint=payload.get("index_fingerprint"),
            vector_size=vector_size,
            distance=distance,
        )

    def alias_target(self, alias: str) -> Optional[str]:
        """The backing collection ``alias`` points to, or None (no alias, or a pre-alias plain collection)."""

        for description in self.client.get_aliases().aliases:
            if description.alias_name == alias:
                return description.collection_name
        return None

    def _replace(
        self, alias: str, vector_params: models.VectorParams, batches: Iterable[List[models.PointStruct]], count: int
    ) -> None:
        """Write ``batches`` (``count`` points) to a new backing collection, verify it, then point ``alias`` at it."""

        backing = f"{alias}__{uuid.uuid4().hex[:12]}"
        self.client.create_collection(backing, vectors_config=vector_params)
        published = False
        try:
            for batch in batches:
                self.client.upsert(backing, points=batch, wait=True)
            stored = int(self.client.count(backing, exact=True).count)
            if stored != count:
                raise VectorStoreError(f"Qdrant stored {stored} points but {count} were written")
            previous = self.alias_target(alias)
            operations: List[models.AliasOperations] = []
            if previous is not None:
                operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
            elif self.client.collection_exists(alias):
                # Built before aliases were used: a plain collection owns the name. Readers see the
                # index as missing (an explicit 503) between this delete and the alias creation below.
                self.client.delete_collection(alias)
            operations.append(
                models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=backing, alias_name=alias))
            )
            self.client.update_collection_aliases(change_aliases_operations=operations)
            published = True
        finally:
            if not published:
                self._discard(alias, backing)
        if previous is not None and previous != backing:
            try:
                self.client.delete_collection(previous)
            except _QDRANT_ERRORS as exc:
                logger.warning(
                    "New index is live, but the replaced collection %s could not be deleted: %s", previous, exc
                )

    def _discard(self, alias: str, backing: str) -> None:
        """Best-effort removal of an unpublished backing collection (never the one the alias serves)."""

        try:
            if self.alias_target(alias) != backing:
                self.client.delete_collection(backing)
        except _QDRANT_ERRORS as exc:
            logger.warning("Could not delete the unfinished collection %s: %s", backing, exc)

    def rebuild(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], fingerprint: str) -> int:
        """Replace the index with exactly these chunks (cosine distance); see the class docstring for the swap."""

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            raise ValueError("Cannot build an index without chunks")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Chunk IDs are not unique; refusing to build an index that would silently drop chunks")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("Vectors have inconsistent dimensions")

        count = len(chunks)
        batches = (
            [
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=list(vector),
                    payload={
                        **chunk.metadata,
                        "text": chunk.text,
                        "index_fingerprint": fingerprint,
                        "index_chunk_count": count,
                    },
                )
                for chunk, vector in zip(chunks[start : start + _UPSERT_BATCH], vectors[start : start + _UPSERT_BATCH])
            ]
            for start in range(0, count, _UPSERT_BATCH)
        )
        with self._lock:
            try:
                vector_params = models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                self._replace(self.collection, vector_params, batches, count)
            except VectorStoreError:
                raise
            except _QDRANT_ERRORS as exc:
                raise VectorStoreError(f"Qdrant indexing into {self.collection!r} failed: {exc}") from exc
        logger.info("Indexed %s chunks into Qdrant collection %s", count, self.collection)
        return count

    @property
    def glossary_collection(self) -> str:
        return f"{self.collection}_glossary"

    def rebuild_glossary(self, entries: Sequence[GlossaryEntry], fingerprint: str) -> int:
        """Store the definitions glossary (no embeddings: points carry a constant 1-d vector)."""

        name = self.glossary_collection
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{fingerprint}:{entry.source}:{entry.term}:{index}")),
                vector=[1.0],
                payload={**entry.to_payload(), "index_fingerprint": fingerprint},
            )
            for index, entry in enumerate(entries)
        ]
        points.append(
            models.PointStruct(
                id=_GLOSSARY_META_ID,
                vector=[1.0],
                payload={"meta": True, "index_fingerprint": fingerprint, "index_entry_count": len(entries)},
            )
        )
        with self._lock:
            try:
                batches = (points[start : start + _UPSERT_BATCH] for start in range(0, len(points), _UPSERT_BATCH))
                self._replace(name, models.VectorParams(size=1, distance=models.Distance.DOT), batches, len(points))
            except VectorStoreError:
                raise
            except _QDRANT_ERRORS as exc:
                raise VectorStoreError(f"Qdrant glossary indexing into {name!r} failed: {exc}") from exc
        return len(entries)

    def load_glossary(self, fingerprint: str) -> Optional[List[GlossaryEntry]]:
        """The stored glossary if it was built from the same inputs and is complete, else None."""

        name = self.glossary_collection
        with self._lock:
            try:
                if not self.client.collection_exists(name):
                    return None
                meta = self.client.retrieve(name, ids=[_GLOSSARY_META_ID], with_payload=True)
                if not meta or (meta[0].payload or {}).get("index_fingerprint") != fingerprint:
                    return None
                expected = int((meta[0].payload or {}).get("index_entry_count", -1))
                entries: List[GlossaryEntry] = []
                offset = None
                while True:
                    batch, offset = self.client.scroll(name, limit=512, offset=offset, with_payload=True, with_vectors=False)
                    for point in batch:
                        payload = point.payload or {}
                        if not payload.get("meta") and payload.get("index_fingerprint") == fingerprint:
                            entries.append(GlossaryEntry.from_payload(payload))
                    if offset is None:
                        break
            except _QDRANT_ERRORS as exc:
                raise VectorStoreError(f"Qdrant glossary read from {name!r} failed: {exc}") from exc
        return entries if len(entries) == expected else None

    def search(self, vector: Sequence[float], limit: int, expected_size: Optional[int] = None) -> List[models.ScoredPoint]:
        """Top-``limit`` points by cosine similarity (higher is more similar), best first."""

        if expected_size is not None and len(vector) != expected_size:
            raise VectorStoreError(
                f"Query embedding has {len(vector)} dimensions but the collection stores {expected_size}; "
                "the index was built with a different embedding model. Rebuild the index."
            )
        with self._lock:
            try:
                response = self.client.query_points(
                    self.collection,
                    query=list(vector),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except _QDRANT_ERRORS as exc:
                raise VectorStoreError(f"Qdrant search in {self.collection!r} failed: {exc}") from exc
        return list(response.points)
