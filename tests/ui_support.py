"""Shared harness for the Streamlit UI tests (imported by tests/test_ui_*.py, not collected itself).

Pages run in-process with Streamlit's ``AppTest``. Their ``ApiClient`` is bound to the
real FastAPI app through Starlette's ``TestClient``, so every request goes through the
real routing, validation and modules. For RAG pages, ``ui_rag`` installs a real pipeline
over generated PDFs with an in-memory Qdrant: only the embedding and chat services are
replaced (by the deterministic stand-ins from ``tests.helpers``).

Usage in a test module::

    from tests.ui_support import assert_no_exception, page_text, run_page
    from tests.ui_support import ui_api, ui_rag  # noqa: F401 (pytest fixtures)

    def test_page(ui_api):
        at = run_page("strategy")
        assert_no_exception(at)
        assert "Heuristic" in page_text(at)

Never import a module from ui/views in a test: importing a page script runs it, outside
AppTest, against whatever API F1_API_URL points to. Pure helpers belong in ui/components.py.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from streamlit.testing.v1 import AppTest

import core_modules.rule_checker.fia_rag.pipeline as rag_pipeline
from app.main import app
from core_modules.rule_checker.fia_rag import (
    ChunkingConfig,
    EmbeddingConfig,
    FIARegulationRAG,
    QdrantConfig,
    RAGSettings,
    RetrievalConfig,
)
from tests.helpers import HashingEmbeddings, ScriptedChatModel, fia_page, write_pdf
from tests.test_fia_generation import cite_passage_containing
from tests.test_fia_index_retrieval import FUEL_FLOW, PIT_LANE, REAR_WING, UNSAFE_RELEASE
from ui.api_client import ApiClient, clear_client_override, set_client_override

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = REPO_ROOT / "ui" / "streamlit_app.py"
VIEWS_DIR = REPO_ROOT / "ui" / "views"
# AppTest's own default (3 s) is too short for the first import of the API's modules or a setup search.
DEFAULT_TIMEOUT_S = 60.0

_TEXT_ELEMENTS = (
    "title", "header", "subheader", "markdown", "caption", "text", "code", "latex", "json",
    "info", "success", "warning", "error", "exception",
)  # fmt: skip


def closed_port() -> int:
    """A local TCP port with nothing listening on it (connections are refused)."""

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def install_client(client: ApiClient) -> ApiClient:
    set_client_override(client)
    return client


@pytest.fixture
def ui_api() -> Iterator[ApiClient]:
    """The UI's client, bound to the real FastAPI app in this process."""

    client = install_client(ApiClient(http=TestClient(app)))
    try:
        yield client
    finally:
        clear_client_override()


@pytest.fixture
def ui_api_down() -> Iterator[ApiClient]:
    """The UI's client, pointed at a local port where no API is listening."""

    client = install_client(ApiClient(f"http://127.0.0.1:{closed_port()}"))
    try:
        yield client
    finally:
        clear_client_override()
        client.close()


def build_test_rag(docs_path: Path, reply=None) -> Tuple[FIARegulationRAG, ScriptedChatModel, QdrantClient]:
    """A real pipeline over two generated regulation PDFs; the index is NOT built yet (call ``build_index``).

    The scripted model answers pit-lane questions by citing the passage containing "80km/h" unless
    ``reply`` (a string or ``messages -> str``) is given.
    """

    write_pdf(docs_path / "section_b_sporting.pdf", [fia_page(1, PIT_LANE), None, fia_page(3, UNSAFE_RELEASE)])
    write_pdf(docs_path / "section_c_technical.pdf", [FUEL_FLOW, REAR_WING])
    settings = RAGSettings(
        docs_path=docs_path,
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=40),
        embedding=EmbeddingConfig(model="hashing-test-512", batch_size=8),
        retrieval=RetrievalConfig(top_k=4, min_score=0.2),
        qdrant=QdrantConfig(collection="ui_test"),
    )
    llm = ScriptedChatModel(reply if reply is not None else cite_passage_containing("80km/h"))
    qdrant = QdrantClient(":memory:")
    rag = FIARegulationRAG(settings, embeddings=HashingEmbeddings(), llm=llm, qdrant_client=qdrant)
    return rag, llm, qdrant


@pytest.fixture
def ui_rag(tmp_path, monkeypatch) -> Iterator[Tuple[FIARegulationRAG, ScriptedChatModel]]:
    """Install ``build_test_rag`` as the API's process-wide pipeline; yields ``(rag, llm)``."""

    rag, llm, qdrant = build_test_rag(tmp_path / "fia_docs")
    monkeypatch.setattr(rag_pipeline, "_instance", rag)
    try:
        yield rag, llm
    finally:
        qdrant.close()


def run_page(page: Union[str, Path], timeout: float = DEFAULT_TIMEOUT_S) -> AppTest:
    """Run a page (name under ui/views, or a script path) once and return the AppTest."""

    path = page if isinstance(page, Path) else VIEWS_DIR / f"{page}.py"
    at = AppTest.from_file(str(path), default_timeout=timeout)
    return at.run()


def page_text(at: AppTest) -> str:
    """All visible text of the last run (main area and sidebar), for substring assertions."""

    parts: List[str] = []
    for kind in _TEXT_ELEMENTS:
        parts.extend(str(element.value) for element in getattr(at, kind))
    parts.extend(f"{metric.label}: {metric.value}" for metric in at.metric)
    parts.extend(str(expander.label) for expander in at.expander)
    parts.extend(str(block.label) for block in at.status)  # AppTest lists an expander with an icon as a status
    parts.extend(str(button.label) for button in at.button)
    parts.extend(frame.value.to_string() for frame in at.dataframe)
    return "\n".join(parts)


def column_formats(frame: Any) -> Dict[str, str]:
    """The number format of each formatted column of a displayed table (``at.dataframe[i]``)."""

    columns = json.loads(frame.proto.columns or "{}")
    types = {name: config.get("type_config", {}) for name, config in columns.items()}
    return {name: config["format"] for name, config in types.items() if "format" in config}


def assert_no_exception(at: AppTest) -> None:
    assert not at.exception, "\n".join(str(element.value) for element in at.exception)
