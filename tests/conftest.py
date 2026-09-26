"""Hermetic test environment.

The test session never talks to a real model provider, the developer's local
index or a running API: provider credentials, the UI's API address and the API's
CORS origins are removed, ``.env`` loading is disabled and every
data/index/artifact path points into a temporary directory that is deleted when
the session ends.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_SESSION_DIR = Path(tempfile.mkdtemp(prefix="f1-copilot-tests-"))
atexit.register(shutil.rmtree, _SESSION_DIR, ignore_errors=True)

for _name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "QDRANT_URL", "QDRANT_API_KEY", "F1_API_URL", "CORS_ORIGINS", "WHISPER_MODEL", "WHISPER_CACHE_DIR"):
    os.environ.pop(_name, None)
for _name in [key for key in os.environ if key.startswith("FIA_RAG_")]:
    os.environ.pop(_name, None)
os.environ["F1_COPILOT_LOAD_DOTENV"] = "0"
os.environ["FIA_DOCS_PATH"] = str(_SESSION_DIR / "fia_docs_unconfigured")
os.environ["QDRANT_PATH"] = str(_SESSION_DIR / "qdrant_unconfigured")
os.environ["F1_ARTIFACTS_DIR"] = str(_SESSION_DIR / "artifacts")
os.environ["FIA_RAG_EMBEDDING_CACHE"] = str(_SESSION_DIR / "embedding_cache.sqlite3")


@pytest.fixture(autouse=True)
def _fresh_rag_singleton():
    from core_modules.rule_checker.fia_rag import reset_fia_rag

    reset_fia_rag()
    yield
    reset_fia_rag()


@pytest.fixture
def qdrant():
    """An in-memory Qdrant client, closed after the test."""

    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture
def installed_rag(tmp_path, monkeypatch):
    """``tests.helpers.build_test_rag`` installed as the API's process-wide pipeline; yields ``(rag, llm)``.

    The index is not built yet (call ``rag.build_index()``).
    """

    import core_modules.rule_checker.fia_rag.pipeline as rag_pipeline
    from tests.helpers import build_test_rag

    rag, llm, client = build_test_rag(tmp_path / "fia_docs")
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    yield rag, llm
    client.close()
