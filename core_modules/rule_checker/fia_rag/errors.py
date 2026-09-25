"""Exception types for the FIA regulation RAG pipeline.

Every condition that makes the RAG system unable to serve a request derives
from :class:`RAGUnavailableError`, so callers (for example the API) can map it
to "service unavailable" without matching on message strings. Programming
errors are deliberately *not* wrapped and propagate unchanged.
"""

from __future__ import annotations

import re


class RAGUnavailableError(RuntimeError):
    """The RAG system cannot answer: configuration, documents, index or provider problem."""


class RAGConfigurationError(RAGUnavailableError):
    """Missing or invalid configuration (API key, paths, numeric settings)."""


class DocumentError(RAGUnavailableError):
    """A regulation document is missing, corrupt, modified or has no extractable text."""


class IndexNotReadyError(RAGUnavailableError):
    """The vector index is missing, incomplete or was built from different inputs."""


class VectorStoreError(RAGUnavailableError):
    """Qdrant could not be reached or rejected a request."""


class ProviderError(RAGUnavailableError):
    """The embedding or chat-model provider failed (auth, rate limit, network, bad response)."""


_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-*]{2}|Bearer\s+)[A-Za-z0-9_\-*.]{4,}")


def describe_provider_error(exc: BaseException, limit: int = 300) -> str:
    """Short, secret-free description of a provider/SDK exception for logs and API responses."""

    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return f"{name} (HTTP {status}): the provider rejected the credentials"
    message = _SECRET.sub(lambda m: m.group(1) + "***", str(exc).strip() or name)
    if len(message) > limit:
        message = message[:limit] + "..."
    return f"{name} (HTTP {status}): {message}" if status else f"{name}: {message}"
