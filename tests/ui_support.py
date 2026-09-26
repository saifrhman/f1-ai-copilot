"""Shared harness for the Streamlit UI tests (imported by tests/test_ui_*.py, not collected itself).

Pages run in-process with Streamlit's ``AppTest``. Their ``ApiClient`` is bound to the
real FastAPI app through Starlette's ``TestClient``, so every request goes through the
real routing, validation and modules. For RAG pages, the ``installed_rag`` fixture
(tests/conftest.py) installs a real pipeline over generated PDFs with an in-memory Qdrant:
only the embedding and chat services are replaced (by the deterministic stand-ins from
``tests.helpers``).

Usage in a test module::

    from tests.ui_support import assert_no_exception, page_text, run_page
    from tests.ui_support import ui_api  # noqa: F401 (pytest fixture)

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
from typing import Any, Dict, Iterator, List, Union

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.main import app
from ui.api_client import ApiClient, clear_client_override, set_client_override
from ui.components import VIEWS_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = REPO_ROOT / "ui" / "streamlit_app.py"
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
    parts.extend(expander_labels(at))
    parts.extend(str(button.label) for button in at.button)
    parts.extend(frame.value.to_string() for frame in at.dataframe)
    return "\n".join(parts)


def markdown_text(block: Any) -> str:
    """The st.markdown text of a run or block (normal contrast: st.caption is not included)."""

    return "\n".join(str(element.value) for element in block.markdown)


def sidebar_text(at: AppTest, captions: bool = False) -> str:
    """The sidebar's Markdown (the API state and module badges), and its captions when asked."""

    elements = [*at.sidebar.markdown, *(at.sidebar.caption if captions else [])]
    return "\n".join(str(element.value) for element in elements)


def expanders(block: Any) -> List[Any]:
    """The st.expander blocks of a run or block: AppTest lists an expander that has an icon as a status element."""

    return [*block.expander, *block.status]


def expander_labels(block: Any) -> List[str]:
    return [str(node.label) for node in expanders(block)]


def metrics(at: AppTest) -> Dict[str, str]:
    """The value of every metric by its label."""

    return {metric.label: metric.value for metric in at.metric}


def frame(at: AppTest, column: str) -> pd.DataFrame:
    """The first displayed table that has ``column``."""

    return next(element.value for element in at.dataframe if column in element.value.columns)


def table_rows(at: AppTest, column: str) -> List[Dict[str, Any]]:
    """The rows of the first displayed table that has ``column``."""

    return frame(at, column).to_dict("records")


def record_calls(monkeypatch: pytest.MonkeyPatch, client: ApiClient, method: str) -> List[Dict[str, Any]]:
    """Wrap ``client.<method>``: every attempt is recorded with its arguments and, when it succeeds, its result."""

    calls: List[Dict[str, Any]] = []
    original = getattr(client, method)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        calls[-1]["result"] = original(*args, **kwargs)
        return calls[-1]["result"]

    monkeypatch.setattr(client, method, wrapper)
    return calls


def column_formats(frame: Any) -> Dict[str, str]:
    """The number format of each formatted column of a displayed table (``at.dataframe[i]``)."""

    columns = json.loads(frame.proto.columns or "{}")
    types = {name: config.get("type_config", {}) for name, config in columns.items()}
    return {name: config["format"] for name, config in types.items() if "format" in config}


def assert_no_exception(at: AppTest) -> None:
    assert not at.exception, "\n".join(str(element.value) for element in at.exception)
