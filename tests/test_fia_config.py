"""RAG settings: environment parsing, validation and path resolution."""

import pytest

from core_modules.rule_checker.fia_rag import RAGConfigurationError, RAGSettings
from core_modules.rule_checker.fia_rag.config import PROJECT_ROOT
from core_modules.rule_checker.fia_rag.embeddings import create_openai_embeddings
from core_modules.rule_checker.fia_rag.generation import create_chat_model


def test_defaults_and_project_relative_paths():
    settings = RAGSettings.from_env({})
    assert settings.docs_path == PROJECT_ROOT / "data" / "fia_docs"
    assert settings.qdrant.path == PROJECT_ROOT / ".qdrant"
    assert (settings.retrieval.top_k, settings.retrieval.min_score) == (8, 0.30)
    assert (settings.chunking.chunk_size, settings.chunking.chunk_overlap) == (1000, 200)
    assert settings.qdrant.mode == "local" and settings.provider.api_key is None


def test_environment_overrides_every_stage_independently(tmp_path):
    settings = RAGSettings.from_env(
        {
            "FIA_DOCS_PATH": str(tmp_path),
            "FIA_RAG_CHUNK_SIZE": "600",
            "FIA_RAG_CHUNK_OVERLAP": "60",
            "FIA_RAG_TOP_K": "8",
            "FIA_RAG_MIN_SCORE": "0.25",
            "FIA_RAG_EMBEDDING_MODEL": "nvidia/nemotron-3-embed-1b:free",
            "FIA_RAG_MODEL": "some/chat-model",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
            "QDRANT_URL": "http://localhost:6333",
        }
    )
    assert settings.docs_path == tmp_path
    assert settings.chunking.chunk_size == 600 and settings.retrieval.top_k == 8
    assert settings.embedding.model.startswith("nvidia/") and settings.generation.model == "some/chat-model"
    assert settings.qdrant.mode == "remote"
    summary = settings.summary()
    assert summary["api_key_configured"] is True and "sk-test" not in str(summary) and "sk-test" not in repr(settings)


@pytest.mark.parametrize(
    "env",
    [
        {"FIA_RAG_TOP_K": "abc"},
        {"FIA_RAG_TOP_K": "0"},
        {"FIA_RAG_MIN_SCORE": "1.5"},
        {"FIA_RAG_MIN_SCORE": "high"},
        {"FIA_RAG_CHUNK_SIZE": "500", "FIA_RAG_CHUNK_OVERLAP": "500"},
        {"FIA_RAG_EMBEDDING_BATCH_SIZE": "0"},
        {"FIA_RAG_COLLECTION": "bad name/with slash"},
        {"FIA_RAG_TIMEOUT_SECONDS": "-1"},
    ],
)
def test_invalid_settings_are_rejected(env):
    with pytest.raises(RAGConfigurationError):
        RAGSettings.from_env(env)


def test_provider_clients_require_a_key_and_honour_the_base_url():
    unconfigured = RAGSettings.from_env({})
    with pytest.raises(RAGConfigurationError, match="OPENAI_API_KEY"):
        create_openai_embeddings(unconfigured.embedding, unconfigured.provider)
    with pytest.raises(RAGConfigurationError, match="OPENAI_API_KEY"):
        create_chat_model(unconfigured.generation, unconfigured.provider)

    configured = RAGSettings.from_env(
        {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "https://openrouter.ai/api/v1", "FIA_RAG_TIMEOUT_SECONDS": "30"}
    )
    # Constructing the real LangChain clients must work with the pinned openai/httpx versions (no network call).
    embeddings = create_openai_embeddings(configured.embedding, configured.provider)
    chat = create_chat_model(configured.generation, configured.provider)
    assert embeddings.openai_api_base == chat.openai_api_base == "https://openrouter.ai/api/v1"
    assert embeddings.check_embedding_ctx_length is False
    assert embeddings.chunk_size == configured.embedding.batch_size
    assert chat.temperature == 0.0 and chat.request_timeout == 30.0
