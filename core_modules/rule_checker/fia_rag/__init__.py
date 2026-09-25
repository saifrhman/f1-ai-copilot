"""Retrieval-augmented question answering over FIA Formula 1 regulation PDFs.

Stages (each in its own module, each configured independently):

    ingestion   PDF validation -> page text -> chunks with source/page/article metadata
    embeddings  LangChain OpenAIEmbeddings (OpenAI or an OpenAI-compatible endpoint)
    index       Qdrant collection with build fingerprints (no stale or duplicate chunks)
    retrieval   question -> top-k passages -> per-passage similarity threshold
    generation  evidence-only prompt with [S#] labels -> chat model
    grounding   deterministic validation of citations and article numbers
    pipeline    orchestration, status reporting and the process-wide instance
"""

from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    ProviderConfig,
    QdrantConfig,
    RAGSettings,
    RetrievalConfig,
)
from .errors import (
    DocumentError,
    IndexNotReadyError,
    ProviderError,
    RAGConfigurationError,
    RAGUnavailableError,
    VectorStoreError,
)
from .grounding import DECLINE_ANSWER
from .pipeline import FIARegulationRAG, get_fia_rag, reset_fia_rag
from .retrieval import RetrievedPassage, RetrievalResult

__all__ = [
    "ChunkingConfig",
    "DECLINE_ANSWER",
    "DocumentError",
    "EmbeddingConfig",
    "FIARegulationRAG",
    "GenerationConfig",
    "IndexNotReadyError",
    "ProviderConfig",
    "ProviderError",
    "QdrantConfig",
    "RAGConfigurationError",
    "RAGSettings",
    "RAGUnavailableError",
    "RetrievalConfig",
    "RetrievedPassage",
    "RetrievalResult",
    "VectorStoreError",
    "get_fia_rag",
    "reset_fia_rag",
]
