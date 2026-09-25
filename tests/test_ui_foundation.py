"""Web UI foundation: API client, shared components, overview page, navigation and the launcher."""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from streamlit.proto.Block_pb2 import Block
from streamlit.testing.v1 import AppTest

import app.main
import ui.api_client as api_client
import ui.components as components
from scripts import run_app
from tests.helpers import fia_page, write_pdf
from tests.test_fia_index_retrieval import PIT_LANE
from tests.ui_support import (  # noqa: F401 (pytest fixtures)
    ENTRY_POINT,
    REPO_ROOT,
    VIEWS_DIR,
    assert_no_exception,
    closed_port,
    page_text,
    run_page,
    ui_api,
    ui_api_down,
    ui_rag,
)
from ui.api_client import ApiClient, ApiError, ApiUnavailable, RequestNotSent
from ui.components import (
    MODULES,
    NAV_SECTIONS,
    PAGES,
    docs_reference,
    local_time,
    md_text,
    rag_fix_steps,
    uses_compose,
)

TRIAGE_REQUEST = {"incident_type": "unsafe_release", "track_condition": "dry", "intent": "accidental"}
HEURISTIC_MODULES = ("strategy", "setup", "ghost", "emotion", "natural_query", "penalty_triage")
STAND_IN_ROUTES = {
    "/": {"message": "F1 AI Copilot API", "version": "9.9.9", "docs": "/docs", "health": "/health"},
    "/health": {"status": "healthy", "version": "9.9.9", "modules": {"strategy": "ready"}, "details": {}},
}


def _lap(speed_kmh: float, seconds: float, lap_number: int) -> dict:
    samples = int(seconds * 2) + 1
    return {"timestamps": [i / 2 for i in range(samples)], "speed": [speed_kmh] * samples, "lap_number": lap_number}


def _mock_client(handler) -> ApiClient:
    return ApiClient(http=httpx.Client(base_url="http://mock-api", transport=httpx.MockTransport(handler)))


def _raising(exc: Exception):
    def handler(request):
        raise exc

    return handler


def _next_steps(at: AppTest) -> str:
    return next(element.value for element in at.markdown if element.value.startswith("**Next steps**"))


def _counting(monkeypatch, client: ApiClient, method: str) -> list:
    calls: list = []
    original = getattr(client, method)
    monkeypatch.setattr(client, method, lambda *args: calls.append(1) or original(*args))
    return calls


@pytest.fixture(autouse=True)
def _no_api_url_from_the_environment(monkeypatch):
    monkeypatch.delenv("F1_API_URL", raising=False)
    api_client._shared_client.cache_clear()
    yield
    api_client.clear_client_override()
    api_client._shared_client.cache_clear()


@pytest.fixture
def stand_in_api():
    """A local HTTP server answering / and /health like the API; yields (url, headers of each request)."""

    seen: list = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append({key.lower(): value for key, value in self.headers.items()})
            known = self.path in STAND_IN_ROUTES
            body = json.dumps(STAND_IN_ROUTES[self.path] if known else {"detail": "Not Found"}).encode()
            self.send_response(200 if known else 404)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------ API client


def test_client_returns_the_api_json(ui_api):
    assert ui_api.root()["docs"] == "/docs"
    assert ui_api.health()["modules"]["strategy"] == "ready"
    body = ui_api.predict_penalty(TRIAGE_REQUEST)
    assert body["method"] == "transparent heuristic triage" and body["referenced_rule"] is None


def test_list_validation_errors_become_readable_lines(ui_api):
    with pytest.raises(ApiError) as caught:
        ui_api.predict_penalty({**TRIAGE_REQUEST, "track_condition": "sunny", "weather": "hot"})
    error = caught.value
    assert error.status_code == 422 and str(error).startswith("HTTP 422: ")
    assert any(line.startswith("track_condition: Input should be") and line.endswith("(got 'sunny')") for line in error.errors)
    assert any(line.startswith("weather: Extra inputs are not permitted") for line in error.errors)
    assert error.detail == "; ".join(error.errors)


def test_string_error_detail_is_kept(ui_api):
    with pytest.raises(ApiError) as caught:
        ui_api.classify_emotion("data:text/plain;base64,aGVsbG8=")
    assert caught.value.status_code == 422 and caught.value.errors == []
    assert "audio" in caught.value.detail


def test_unconfigured_rag_is_a_503_with_the_api_reason(ui_api):
    with pytest.raises(ApiError) as caught:
        ui_api.fia_query("What is the pit lane speed limit?")
    assert caught.value.status_code == 503
    assert caught.value.detail.startswith("FIA RAG is unavailable") and caught.value.error_type


def test_error_body_parsing_edge_cases():
    nested = {
        "detail": [
            {
                "type": "missing", "loc": ["body", "race_state", "total_laps"],
                "msg": "Field required", "input": "<object with 3 keys>",
            },
            {
                "type": "float_type", "loc": ["body", "telemetry", "lap_times", 2],
                "msg": "Input should be a finite number", "input": "nan",
            },
            {"type": "json_invalid", "loc": ["body"], "msg": "JSON decode error", "input": None},
        ],
        "more_errors": 3,
    }  # fmt: skip
    client = _mock_client(lambda request: httpx.Response(422, json=nested))
    with pytest.raises(ApiError) as caught:
        client.generate_strategy({})
    assert caught.value.errors == [
        "race_state.total_laps: Field required",
        "telemetry.lap_times[2]: Input should be a finite number (got 'nan')",
        "request body: JSON decode error",
        "... and 3 more",
    ]
    proxy = _mock_client(lambda request: httpx.Response(502, text="<html>\n  Bad   gateway</html>"))
    with pytest.raises(ApiError) as caught:
        proxy.health()
    assert (caught.value.status_code, caught.value.detail) == (502, "<html> Bad gateway</html>")


@pytest.mark.parametrize("body", [["ok"], "OK", None])
def test_a_success_that_is_not_a_json_object_is_reported_not_guessed(body):
    client = _mock_client(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ApiError, match="did not return a JSON object.*check F1_API_URL"):
        client.health()
    html = _mock_client(lambda request: httpx.Response(200, text="<html>streamlit</html>"))
    with pytest.raises(ApiError, match="did not return a JSON object"):
        html.health()


def test_another_json_service_on_the_api_port_does_not_break_the_ui():
    api_client.set_client_override(_mock_client(lambda request: httpx.Response(200, json=["ok"])))
    at = run_page(ENTRY_POINT)
    assert_no_exception(at)
    sidebar = "\n".join(str(element.value) for element in [*at.sidebar.markdown, *at.sidebar.caption])
    assert ":red-badge[unreachable] API status" in sidebar and "did not return a JSON object" in sidebar


def test_connection_refused_is_api_unavailable_with_a_start_hint():
    port = closed_port()
    client = ApiClient(f"http://127.0.0.1:{port}")
    with pytest.raises(ApiUnavailable) as caught:
        client.health()
    assert caught.value.reason.startswith("connection failed")
    assert f"`uvicorn app.main:app --port {port}`" in caught.value.hint and "scripts/run_app.py" in caught.value.hint


def test_transport_failures_are_api_unavailable_with_hints():
    cases = [
        (httpx.ConnectTimeout("timed out"), "no connection within 5 s", "API on mock-api is running"),
        (httpx.RemoteProtocolError("Server disconnected"), "the connection broke (RemoteProtocolError: Server", "mock-api"),
        (httpx.ReadTimeout("timed out"), "no answer within 60 s", "did not finish in time"),
    ]
    for exc, reason, hint in cases:
        with pytest.raises(ApiUnavailable) as caught:
            _mock_client(_raising(exc)).health()
        assert caught.value.reason.startswith(reason) and hint in caught.value.hint


def test_a_body_json_cannot_carry_is_reported_and_never_sent():
    sent: list = []
    client = _mock_client(lambda request: sent.append(request) or httpx.Response(200, json={}))
    cases = [
        # The same words on every Python version (3.11 does not name the value).
        ({"lap_times": [95.6, float("inf")]}, "it contains a number JSON cannot carry (NaN or infinity)"),
        ({"lap_times": [float("nan")]}, "it contains a number JSON cannot carry (NaN or infinity)"),
        ({"query": "\ud800"}, "it contains text that is not valid Unicode"),
    ]
    for payload, reason in cases:
        with pytest.raises(RequestNotSent) as caught:
            client.generate_strategy(payload)
        assert caught.value.detail == f"the request could not be encoded as JSON: {reason}"
        assert isinstance(caught.value, ApiError)  # every page that handles API errors handles it too
    assert sent == []
    client.generate_strategy({"lap_times": [95.6], "track": "Autódromo"})
    assert json.loads(sent[0].content) == {"lap_times": [95.6], "track": "Autódromo"}
    assert sent[0].headers["content-type"] == "application/json"


def test_the_start_hint_follows_the_api_address():
    assert "`uvicorn app.main:app`, or stop this UI" in api_client.start_api_hint("http://127.0.0.1:8000")
    assert "`uvicorn app.main:app --port 18999`" in api_client.start_api_hint("http://localhost:18999")
    assert "`uvicorn app.main:app --host ::1 --port 9000`" in api_client.start_api_hint("http://[::1]:9000")
    remote = api_client.start_api_hint("http://192.168.1.20:8000")
    assert "API on 192.168.1.20 is running" in remote and "`docker logs <container>`" in remote
    assert "uvicorn" not in remote and "docker compose" not in remote
    # The API service of this project's Docker Compose stack.
    compose = api_client.start_api_hint("http://api:8000")
    assert "`docker compose logs api`" in compose and "`docker compose up -d api`" in compose
    assert "uvicorn" not in compose and "F1_API_URL" not in compose
    for hint in (api_client.start_api_hint("http://127.0.0.1:8000"), remote):
        # Shown as Markdown: an address outside a code span would become a link to someone else's machine.
        assert "set `F1_API_URL` (e.g. `F1_API_URL=http://192.168.1.20:8000`)" in hint
        assert "://" not in re.sub(r"`[^`]*`", "", hint)


def test_a_silent_api_times_out_as_api_unavailable():
    with socket.socket() as server:  # accepts the connection, never answers
        server.bind(("127.0.0.1", 0))
        server.listen()
        client = ApiClient(f"http://127.0.0.1:{server.getsockname()[1]}")
        with pytest.raises(ApiUnavailable) as caught:
            client._send("GET", "/health", read_timeout=0.3)
    assert caught.value.reason.startswith("no answer within")
    assert "did not finish in time" in caught.value.hint


def test_proxy_settings_from_the_environment_are_ignored(stand_in_api, monkeypatch):
    url, _ = stand_in_api
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{closed_port()}")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")  # would need the optional socksio package
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert ApiClient(url).health()["version"] == "9.9.9"
    monkeypatch.setenv("F1_API_URL", url)
    at = run_page(ENTRY_POINT)
    assert_no_exception(at)
    assert ":green-badge[healthy] API 9.9.9" in "\n".join(str(element.value) for element in at.sidebar.markdown)


def test_credentials_in_the_api_url_are_sent_but_never_shown(stand_in_api, monkeypatch):
    url, seen = stand_in_api
    secret_url = url.replace("http://", "http://admin:s3cr3t-pass@")
    client = ApiClient(secret_url)
    assert client.base_url == url
    client.health()
    assert seen[-1]["authorization"] == "Basic " + base64.b64encode(b"admin:s3cr3t-pass").decode()

    with pytest.raises(ApiUnavailable) as caught:
        ApiClient(f"http://admin:s3cr3t-pass@127.0.0.1:{closed_port()}").health()
    assert "s3cr3t" not in f"{caught.value} {caught.value.base_url} {caught.value.hint}"
    for bad in (
        "ftp://admin:s3cr3t-pass@example.org",
        "http://example.org/?token=s3cr3t-pass",
        "http://admin:s3cr3t-pass@example.org:abc",
        "http://admin:s3cr3t-pass@example.org:99999",
    ):
        with pytest.raises(ValueError, match="F1_API_URL") as caught:
            api_client.normalise_base_url(bad)
        assert "s3cr3t" not in str(caught.value)

    monkeypatch.setenv("F1_API_URL", secret_url)
    at = run_page(ENTRY_POINT)
    assert_no_exception(at)
    assert at.sidebar.code[0].value == url and "s3cr3t" not in page_text(at)
    assert seen[-1]["authorization"].startswith("Basic ")  # the UI's own requests still authenticate


def test_ghost_png_is_fetched_through_the_client(ui_api):
    body = ui_api.generate_ghost({"lap1_telemetry": _lap(200.0, 60, 1), "lap2_telemetry": _lap(190.0, 63.2, 2)})
    image = ui_api.fetch_ghost_image(body["visualization_url"])
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    for url in ("/etc/passwd", "/artifacts/ghost/../../app/main.py", "http://example.org/artifacts/ghost/x.png"):
        with pytest.raises(ValueError):
            ui_api.fetch_ghost_image(url)
    with pytest.raises(ApiError) as caught:
        ui_api.fetch_ghost_image("/artifacts/ghost/missing.png")
    assert caught.value.status_code == 404


def test_a_ghost_image_must_be_a_png():
    login_page = _mock_client(lambda request: httpx.Response(200, text="<html>login</html>"))
    with pytest.raises(ApiError, match="Expected a PNG image, got text/plain"):
        login_page.fetch_ghost_image("/artifacts/ghost/lap.png")


def test_endpoint_detection_and_examples_use_the_openapi_schema(ui_api):
    schema = ui_api.get_openapi()
    offers = components.offers_endpoint
    assert offers(schema, "/api/fia/query") and offers(schema, "/health", "GET")
    assert offers(schema, api_client.CALIBRATE_TYRES_PATH)
    assert not offers(schema, "/api/fia/query", "get") and not offers(schema, "/api/nothing-here") and not offers({}, "/")
    example = components.documented_example(schema, "StrategyRequest")
    assert example == schema["components"]["schemas"]["StrategyRequest"]["examples"][0]
    example["race_state"]["current_lap"] = -1  # a copy: the schema keeps its example
    assert schema["components"]["schemas"]["StrategyRequest"]["examples"][0]["race_state"]["current_lap"] != -1
    assert components.documented_example(schema, "PenaltyRequest") is None  # documents no example
    assert components.documented_example({}, "StrategyRequest") is None


def _read_schema_twice():
    import streamlit as st

    from ui.components import api_schema

    first, second = api_schema(), api_schema()
    st.caption(f"same document: {first is second}; paths: {len(first['paths'])}")


def test_the_openapi_document_is_fetched_once_per_session(ui_api, monkeypatch):
    calls = _counting(monkeypatch, ui_api, "get_openapi")
    at = AppTest.from_function(_read_schema_twice, default_timeout=30).run()
    assert_no_exception(at)
    at.run()
    assert len(calls) == 1 and at.caption[0].value.startswith("same document: True; paths: ")


def test_base_url_resolution(monkeypatch, tmp_path):
    assert api_client.resolve_base_url() == api_client.DEFAULT_API_URL  # tests never read .env
    monkeypatch.setenv("F1_API_URL", " localhost:9000/ ")
    assert api_client.resolve_base_url() == "http://localhost:9000"
    monkeypatch.setenv("F1_API_URL", "ftp://example.org")
    with pytest.raises(ValueError, match="F1_API_URL"):
        api_client.resolve_base_url()

    monkeypatch.delenv("F1_API_URL")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-test-not-a-key\nF1_API_URL=http://10.1.2.3:8000/\n")
    monkeypatch.setattr(api_client, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("F1_COPILOT_LOAD_DOTENV", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert api_client.resolve_base_url() == "http://10.1.2.3:8000"
    assert "OPENAI_API_KEY" not in os.environ  # only F1_API_URL is read from .env


def test_get_client_is_cached_and_overridable(ui_api):
    api_client.clear_client_override()
    first = api_client.get_client()
    assert api_client.get_client() is first and first.base_url == api_client.DEFAULT_API_URL
    api_client.set_client_override(ui_api)
    assert api_client.get_client() is ui_api


# ------------------------------------------------------------------ shared components


def test_md_text_shows_api_text_literally_but_keeps_code_spans():
    assert md_text("a_b *c* [S1] <project> `python scripts/build_fia_index.py`") == (
        "a\\_b \\*c\\* \\[S1\\] \\<project\\> `python scripts/build_fia_index.py`"
    )
    assert md_text("unbalanced ` tick_x") == "unbalanced \\` tick\\_x"


def _render_errors():
    import streamlit as st

    from ui.api_client import ApiError, ApiUnavailable
    from ui.components import show_api_error

    show_api_error(ApiUnavailable("http://127.0.0.1:1", "connection failed (refused)", "Start the API."), "Loading")
    show_api_error(ApiError(422, "x", errors=["race_state.current_lap: Input should be a valid integer (got 5.5)"]))
    show_api_error(
        ApiError(503, "FIA RAG is unavailable: The FIA regulation index is missing. Run `x`.", error_type="IndexNotReadyError")
    )
    show_api_error(ApiError(500, "Internal server error"))
    show_api_error(ApiError(413, "Request body too large (limit 8 MiB)"))
    show_api_error(ApiError(503, "Transcription failed: ffmpeg is not installed"))
    st.session_state["_f1_health_snapshot"] = (0.0, 0.0, {"status": "healthy"})
    show_api_error(ApiError(503, "FIA RAG is unavailable: Embedding provider call failed", error_type="ProviderError"))
    st.caption(f"health cache cleared: {'_f1_health_snapshot' not in st.session_state}")


def test_api_errors_are_explained_with_next_steps():
    at = AppTest.from_function(_render_errors, default_timeout=30).run()
    assert_no_exception(at)
    captions = [element.value for element in at.caption]
    assert at.error[0].value == "Loading could not reach the API at `http://127.0.0.1:1`: connection failed (refused)."
    assert at.info[0].value == "Start the API."
    assert "rejected by the API (HTTP 422)" in at.error[1].value
    assert "- race\\_state.current\\_lap\\: Input should be a valid integer (got 5.5)" in page_text(at)
    assert at.warning[0].value.startswith("Service unavailable (HTTP 503): FIA RAG is unavailable")
    assert captions[1] == (
        "The regulation index is missing, incomplete or out of date. The Overview page shows its state and the exact fix."
    )
    assert "HTTP 500" in at.error[2].value and "server-side error" in captions[2]
    assert at.error[3].value.startswith("The request is too large for the API (HTTP 413)")
    assert "Whisper transcription is optional" in captions[3]
    assert captions[4].startswith("The model provider call failed") and "the index itself is fine" in captions[4]
    assert "index is missing" not in captions[4]
    assert captions[-1] == "health cache cleared: True"


def _render_more_errors():
    from ui.api_client import ApiError, RequestNotSent
    from ui.components import show_api_error

    show_api_error(RequestNotSent("it contains a number JSON cannot carry (inf)"), "The question")
    pydantic_text = (
        "1 validation error for StrategyRequest\nrace_state\n  Value error, current_lap (99) must be <= total_laps (57) "
        "[type=value_error, input_value={...}, input_type=dict]\n    For further information visit "
        "https://errors.pydantic.dev/2.11/v/value_error"
    )
    show_api_error(ApiError(422, pydantic_text), "The question")


def test_unsent_requests_and_plain_text_validation_errors_are_readable():
    at = AppTest.from_function(_render_more_errors, default_timeout=30).run()
    assert_no_exception(at)
    reason = "the request could not be encoded as JSON: it contains a number JSON cannot carry (inf)"
    assert at.error[0].value == f"The question was not sent: {md_text(reason)}."
    assert at.caption[0].value == "Correct that value and submit again."
    assert at.error[1].value == "The question was rejected by the API (HTTP 422):"
    # One line per error, without pydantic's type codes and documentation links.
    assert at.markdown[0].value == "- " + md_text("race_state: current_lap (99) must be <= total_laps (57)")
    assert components.validation_lines("Model provider failed\nline two") == []  # any other text is kept as it is


def test_field_labels_use_the_form_words():
    fields = components.FieldLabels(
        {"tire_data": "Tyre model", "peak_performance_window": "Peak window", "competition": "Competitors",
         "tire_age": "Tyre age", "telemetry": "Telemetry", "lap_times": "Recent lap times"},
        table_lists=("competition",),
        index_names={"peak_performance_window": ("Window start", "Window end")},
    )  # fmt: skip
    assert fields.label("tire_data.hard.peak_performance_window[1]") == "Tyre model › hard › Window end"
    assert fields.label("competition[0].tire_age") == "Competitors › row 1 › Tyre age"
    assert fields.label("telemetry.lap_times[2]") == "Telemetry › Recent lap times › entry 3"
    assert fields.message("Value error, tire_data has no compound usable in wet") == "Tyre model has no compound usable in wet"
    assert fields.errors(
        [
            "competition[1].tire_age: Input should be >= 0 (got -1)",
            "telemetry: telemetry.lap_times must not be empty",  # names its own field: shown once, in bold
            "request body: Field required",
            "no separator",
        ]
    ) == [
        "**Competitors › row 2 › Tyre age**: Input should be \\>= 0 (got -1)",
        "**Telemetry › Recent lap times** must not be empty",
        "Field required",
        "no separator",
    ]


def test_tied_emotion_profiles_are_named():
    tied = {"calm": 1.0, "angry": 0.628, "focused": 1.0}
    assert components.tied_profiles(tied) == ["calm", "focused"]
    assert components.tied_profiles({"calm": 0.754, "focused": 0.752}) == [] and components.tied_profiles({}) == []
    note = components.profile_tie_note(tied, "calm")
    assert note.startswith("**Tied profiles:** calm and focused both score 1.000, so the audio does not separate them.")
    three = components.profile_tie_note({"calm": 0.9, "focused": 0.9, "excited": 0.9}, "calm")
    assert "calm, focused and excited all score 0.900" in three
    # The API reports an exact tie for the best profile as neutral with confidence 0; the note says why.
    assert components.profile_tie_note({"calm": 0.3, "focused": 0.3}, "neutral") == (
        "**Tied profiles:** calm and focused both score 0.300, so the audio does not separate them: the acoustic "
        "label is **neutral** with confidence 0."
    )
    assert components.profile_tie_note({"calm": 0.9, "focused": 0.5}, "calm") is None


def test_close_emotion_profiles_get_a_caution():
    close = {"frustrated": 1.0, "focused": 0.991, "calm": 0.4}
    note = components.profile_tie_note(close, "frustrated")
    assert note.startswith("**Close profiles:** frustrated 1.000 leads focused 0.991 by only 0.009 (under 0.05)")
    assert "barely separates them" in note
    assert components.profile_tie_note({"frustrated": 1.0, "focused": 0.95}, "frustrated") is None  # a clear lead
    assert components.profile_tie_note(close, "neutral") is None  # no profile won (below the threshold)
    assert components.profile_tie_note({"calm": 0.9}, "calm") is None  # no runner-up


def _render_one_error(error):
    from ui.components import show_api_error

    show_api_error(error, "The search")


def _break_the_provider(rag, monkeypatch) -> None:
    def unreachable(text):
        raise ConnectionError("provider unreachable")

    monkeypatch.setattr(rag.embedder()._embeddings, "embed_query", unreachable)


def test_real_provider_failure_is_not_called_a_missing_index(ui_api, ui_rag, monkeypatch):
    rag, _ = ui_rag
    rag.build_index()
    _break_the_provider(rag, monkeypatch)
    with pytest.raises(ApiError) as caught:
        ui_api.fia_retrieve("How long is a drive-through penalty?")
    assert caught.value.error_type == "ProviderError"
    at = AppTest.from_function(_render_one_error, args=(caught.value,), default_timeout=30).run()
    assert_no_exception(at)
    assert "provider unreachable" in at.warning[0].value
    assert at.caption[0].value.startswith("The model provider call failed")


def test_health_snapshots_expire(ui_api, ui_api_down, monkeypatch):
    now = [1000.0]
    clock = SimpleNamespace(monotonic=lambda: now[0], time=time.time, strftime=time.strftime, localtime=time.localtime)
    monkeypatch.setattr(components, "time", clock)
    for client, ttl in ((ui_api, components.HEALTH_TTL_S), (ui_api_down, components.FAILED_HEALTH_TTL_S)):
        api_client.set_client_override(client)
        calls = _counting(monkeypatch, client, "health")
        at = run_page("overview")
        at.run()
        assert len(calls) == 1
        now[0] += ttl - 1
        at.run()
        assert len(calls) == 1
        now[0] += 2
        at.run()
        assert_no_exception(at)
        assert len(calls) == 2, client.base_url


def test_navigation_covers_every_page_once():
    names = [name for section in NAV_SECTIONS.values() for name in section]
    assert sorted(names) == sorted(PAGES) and len(names) == len(set(names))
    assert all((VIEWS_DIR / f"{name}.py").is_file() for name in PAGES)
    assert {spec.page for spec in MODULES.values()} <= set(PAGES)


def test_no_pages_directory_next_to_the_entry_point():
    # A "pages" folder beside the entry script switches Streamlit to its legacy multipage mode: a page URL
    # opened first after a server start then bypasses streamlit_app.py (no sidebar, no layout, import errors).
    assert not (ENTRY_POINT.parent / "pages").exists()
    assert components.VIEWS_DIR == ENTRY_POINT.parent / "views"


# ------------------------------------------------------------------ overview page


def test_overview_explains_an_unconfigured_rag(ui_api):
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    assert at.title[0].value == "F1 AI Copilot"
    assert ":orange-badge[degraded]" in text and ":red-badge[not configured] Regulation QA" in text
    assert "OPENAI\\_API\\_KEY is not set" in text  # the API's own problem text
    assert "Download the official PDFs: `python scripts/fetch_fia_regulations.py`" in text
    assert "Set `OPENAI_API_KEY` in `.env`" in text and "`python scripts/build_fia_index.py`" in text
    steps = _next_steps(at)
    assert steps.index("Stop the API first") < steps.index("Build the index") < steps.index("Start the API again")
    assert "Restart the API" not in steps  # starting it again after the build reads the new settings
    assert ":green-badge[ready] Strategy engine :orange-badge[heuristic]" in text
    assert "Documents: 0" in text and "Model provider: unknown" in text


def _metric_parent(at: AppTest, label: str):
    """The layout block that directly holds the metric ``label``."""

    def walk(node):
        for child in getattr(node, "children", {}).values():
            if child.type == "metric" and child.label == label:
                return node
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(at.main)


def test_overview_index_metrics_wrap_on_a_phone(ui_api):
    at = run_page("overview")
    assert_no_exception(at)
    # One wrapping row of metrics (like the regulations page), not five columns that stack into five rows on a phone.
    row = _metric_parent(at, "Documents")
    assert row.proto.flex_container.direction == Block.FlexContainer.Direction.HORIZONTAL and row.proto.flex_container.wrap
    assert [child.label for child in row.children.values()] == [
        "Documents", "Index", "Indexed passages", "Definitions", "Model provider",
    ]  # fmt: skip
    # "ok" can follow a search answered from cached embeddings, which calls no provider: the help says so.
    provider = next(metric for metric in at.metric if metric.label == "Model provider")
    assert provider.help == components.PROVIDER_STATUS_HELP
    assert "not a live check" in provider.help and "cached embeddings counts too" in provider.help


def test_every_heuristic_module_is_labelled(ui_api):
    assert "heuristic" in ui_api.natural_query("Which tyre compound for the next stint?")["routing"]["method"]
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    for key in HEURISTIC_MODULES:
        assert f"{MODULES[key].label} :orange-badge[heuristic]" in text, key
    for key in ("fia_rag", "emotion_transcription"):
        assert f"{MODULES[key].label} :orange-badge[heuristic]" not in text, key


def test_overview_asks_only_for_what_is_missing(ui_api, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-never-sent")  # status checks make no provider calls
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[not configured] Regulation QA" in text and "OPENAI\\_API\\_KEY is not set" not in text
    assert "Download the official PDFs" in text and "Set `OPENAI_API_KEY`" not in text


def test_overview_reports_a_ready_index_without_calling_the_model(ui_api, ui_rag):
    rag, llm = ui_rag
    rag.build_index()
    points = rag.status()["index"]["points"]
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    assert ":green-badge[healthy]" in text and ":green-badge[ready] Regulation QA" in text
    assert at.success[0].value.startswith(f"Ready: {points} indexed passages from 2 documents")
    assert f"Indexed passages: {points}" in text and "Index: current" in text
    assert "Next steps" not in text and not at.warning
    assert "section_b_sporting.pdf" in text and "hashing-test-512" in text
    assert llm.calls == []


def test_overview_gives_the_build_command_for_a_missing_index(ui_api, ui_rag):
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[index missing] Regulation QA" in text
    assert "`python scripts/build_fia_index.py --dry-run`" in text
    steps = _next_steps(at)
    assert steps.index("Stop the API first") < steps.index("Build the index") < steps.index("Start the API again")
    assert "Download the official PDFs" not in text and "Set `OPENAI_API_KEY`" not in text
    assert "docker compose" not in text and "Documents: 2" in text


def test_overview_explains_a_stale_index(ui_api, ui_rag):
    rag, _ = ui_rag
    rag.build_index()
    write_pdf(rag.settings.docs_path / "section_d_financial.pdf", [fia_page(1, PIT_LANE)])
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[index stale] Regulation QA" in text and "Documents: 3" in text
    assert "The documents or the chunking/embedding settings changed" in text and "Build the index" in text


def test_overview_explains_a_misconfigured_rag(ui_api, monkeypatch):
    monkeypatch.setenv("FIA_RAG_TOP_K", "abc")
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[misconfigured] Regulation QA" in text and "FIA\\_RAG\\_TOP\\_K must be an integer" in text
    assert "Correct the setting named above in `.env` or the environment, then restart the API" in text


def test_overview_explains_a_failing_provider(ui_api, ui_rag, monkeypatch):
    rag, llm = ui_rag
    rag.build_index()
    _break_the_provider(rag, monkeypatch)
    with pytest.raises(ApiError):
        ui_api.fia_retrieve("How long is a drive-through penalty?")
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[provider failing] Regulation QA" in text and "Model provider: failing" in text
    assert "provider unreachable" in text and "your provider quota; after changing `.env`, restart the API" in text
    failed_at = datetime.fromisoformat(rag.last_error_at).astimezone().strftime("%H:%M:%S")
    assert f"Last provider error at {failed_at}." in [element.value for element in at.caption]
    assert llm.calls == []


def test_overview_notes_unwritable_artifacts_and_missing_transcription(ui_api, monkeypatch):
    monkeypatch.setattr(app.main, "_artifacts_writable", lambda: False)
    monkeypatch.setattr(app.main, "_transcription_availability", lambda: (False, "openai-whisper is not installed"))
    at = run_page("overview")
    assert_no_exception(at)
    text = page_text(at)
    assert ":red-badge[artifacts not writable] Ghost comparison" in text
    assert "make `F1_ARTIFACTS_DIR` (default `outputs`) writable" in at.warning[-1].value
    assert "Radio transcription is optional and unavailable on this API: openai-whisper is not installed" in text


def test_overview_never_contradicts_itself_after_the_index_changes(ui_api, ui_rag, monkeypatch):
    rag, _ = ui_rag
    calls = _counting(monkeypatch, ui_api, "health")
    at = run_page(ENTRY_POINT)
    assert_no_exception(at)
    assert ":red-badge[index missing] FIA regulation QA" in "\n".join(m.value for m in at.sidebar.markdown)
    rag.build_index()  # within the 30 s health cache
    at.run()
    assert_no_exception(at)
    sidebar = "\n".join(str(element.value) for element in at.sidebar.markdown)
    main = page_text(at)
    assert ":green-badge[ready] FIA regulation QA" in sidebar and "index missing" not in sidebar
    assert ":green-badge[ready] FIA regulation QA" in main and ":green-badge[ready] Regulation QA" in main
    assert "index missing" not in main and len(calls) == 2  # one refresh, no rerun loop
    at.run()
    assert len(calls) == 2


def test_overview_refreshes_a_disagreeing_status_only_once(ui_api, monkeypatch):
    calls = _counting(monkeypatch, ui_api, "health")
    status = ui_api.fia_status()
    monkeypatch.setattr(ui_api, "fia_status", lambda: {**status, "state": "index_missing"})  # never matches /health
    at = run_page("overview", timeout=15)
    assert_no_exception(at)
    assert len(calls) == 2 and ":red-badge[index missing] Regulation QA" in page_text(at)


def test_overview_when_the_api_is_down(ui_api_down, monkeypatch):
    attempts = _counting(monkeypatch, ui_api_down, "health")
    at = run_page("overview")
    at.run()
    assert_no_exception(at)
    assert len(attempts) == 1  # the failed check is reused briefly, not retried on every rerun
    text = page_text(at)
    assert at.error[0].value.startswith(f"The status check could not reach the API at `{ui_api_down.base_url}`")
    assert "uvicorn app.main:app --port" in at.info[0].value
    assert ":gray-badge[not checked] Strategy engine" in text and "Shown when the API is reachable." in text
    assert ":green-badge" not in text and "Documents:" not in text  # nothing is made up


def test_overview_caches_health_per_session_and_refreshes_on_demand(ui_api, monkeypatch):
    calls = _counting(monkeypatch, ui_api, "health")
    at = run_page("overview")
    at.run()
    assert len(calls) == 1
    at.button(key="overview_refresh").click().run()
    assert_no_exception(at)
    assert len(calls) == 2 and at.title[0].value == "F1 AI Copilot"


def test_fix_steps_for_docker_compose():
    compose_settings = {"qdrant_mode": "server", "qdrant_location": "http://qdrant:6333", "api_key_configured": False}
    assert uses_compose("http://api:8000", {}) and uses_compose("http://127.0.0.1:8000", compose_settings)
    assert not uses_compose("http://127.0.0.1:8000", {"qdrant_mode": "local", "qdrant_location": ".qdrant"})
    unconfigured = {"state": "not_configured", "documents": [], "settings": compose_settings, "index": {"status": "missing"}}
    steps = rag_fix_steps(unconfigured, compose=True)
    assert steps[0] == (
        "Download the official PDFs: `docker compose run --rm api python scripts/fetch_fia_regulations.py` "
        "(saved to `data/fia_docs`)."
    )
    assert "`docker compose run --rm api python scripts/build_fia_index.py --dry-run`" in steps[2]
    assert steps[3].startswith("Apply `.env` changes with `docker compose up -d api` (`docker compose restart` keeps")
    assert not any("Stop the API" in step or "`python scripts/" in step for step in steps)
    failing = rag_fix_steps({"state": "provider_failing", "settings": compose_settings}, compose=True)
    assert "`docker compose up -d api`" in failing[0]


def test_fix_steps_name_the_folder_the_api_reads():
    def fetch_step(docs_path, compose=False):
        settings = {"qdrant_mode": "local", "api_key_configured": True, "docs_path": docs_path}
        status = {"state": "not_configured", "documents": [], "settings": settings, "index": {"status": "missing"}}
        return rag_fix_steps(status, compose=compose)[0]

    assert (
        fetch_step("data/fia_docs")
        == "Download the official PDFs: `python scripts/fetch_fia_regulations.py` (saved to `data/fia_docs`)."
    )
    # FIA_DOCS_PATH elsewhere: the script saves to FIA_DOCS_PATH, so it must run with the API's setting.
    assert fetch_step("<external>/docs") == (
        "Download the official PDFs: `python scripts/fetch_fia_regulations.py`. It saves to `FIA_DOCS_PATH` (from the "
        "environment or `.env`, default `data/fia_docs`) and this API reads `<external>/docs`, so run it with the "
        "API's `FIA_DOCS_PATH`."
    )
    assert "this API reads `data/regs`" in fetch_step("data/regs")
    # In Docker Compose the script runs in the API container, with the API's own settings.
    assert fetch_step("data/regs", compose=True).endswith("fetch_fia_regulations.py` (saved to `data/regs`).")


def test_fix_steps_for_a_missing_glossary_stop_the_api_first():
    status = {
        "state": "unavailable",
        "settings": {"qdrant_mode": "local", "api_key_configured": True},
        "index": {"status": "current", "glossary": {"status": "missing"}},
    }
    steps = rag_fix_steps(status)
    assert steps[0].startswith("Stop the API first") and steps[2] == "Start the API again."
    assert steps[1].startswith("Rebuild the index to add the definitions glossary: `python scripts/build_fia_index.py --dry-run`")
    server = {**status, "settings": {"qdrant_mode": "server", "api_key_configured": True}}
    assert len(rag_fix_steps(server)) == 1  # a Qdrant server needs no API stop


def test_timestamps_and_docs_link():
    today = datetime.now(timezone.utc).replace(microsecond=268857)
    assert local_time(today.isoformat()) == today.astimezone().strftime("%H:%M:%S")
    earlier = today - timedelta(days=3)
    assert local_time(earlier.isoformat()) == earlier.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    assert local_time("not a time_stamp") == "not a time\\_stamp"
    assert docs_reference("http://127.0.0.1:8000", "http://localhost:8501/") == "http://127.0.0.1:8000/docs"
    # Docker Compose: the UI reaches the API as http://api:8000; the browser uses its published port.
    compose = docs_reference("http://api:8000", "http://127.0.0.1:8501/")
    assert compose == "http://127.0.0.1:8000/docs (the API container of Docker Compose)"
    for base_url, browser_url in (
        ("http://api:8000", "http://192.168.1.5:8501/"),  # a browser elsewhere cannot use 127.0.0.1
        ("http://127.0.0.1:8000", "http://192.168.1.5:8501/"),  # LAN mode, browser on another device
        ("http://127.0.0.1:8000", None),
    ):
        assert docs_reference(base_url, browser_url) == f"`{base_url}/docs` (the address the UI server uses)"


# ------------------------------------------------------------------ entry point and navigation


def test_entry_point_runs_every_page_with_the_status_sidebar(ui_api):
    at = run_page(ENTRY_POINT)
    assert_no_exception(at)
    assert at.title[0].value == "F1 AI Copilot"  # the default page is the overview
    sidebar = "\n".join(str(element.value) for element in at.sidebar.markdown)
    assert at.sidebar.code[0].value == "http://testserver"
    assert ":orange-badge[degraded] API" in sidebar and ":red-badge[not configured] FIA regulation QA" in sidebar
    for name in PAGES:
        at.switch_page(f"views/{name}.py").run()
        assert_no_exception(at)
        assert at.title, f"page {name} rendered no title"
        if name != "overview":  # the overview is titled with the project name
            assert at.title[0].value == PAGES[name].title  # the navigation and the page use the same name


def test_entry_point_explains_a_malformed_api_url(monkeypatch):
    monkeypatch.setenv("F1_API_URL", "ftp://example.org")
    at = run_page(ENTRY_POINT)
    assert_no_exception(at)
    message = at.error[0].value
    assert md_text("F1_API_URL must look like http://host:port") in message
    assert "Set `F1_API_URL` to the API address (e.g. `http://127.0.0.1:8000`)" in message
    # An address outside a code span has its ":" escaped, so it does not become a link.
    assert not re.search(r"(?<!\\):/", re.sub(r"`[^`]*`", "", message))


def test_sidebar_shows_an_unreachable_api(ui_api_down):
    at = run_page(ENTRY_POINT)
    assert_no_exception(at)
    sidebar = "\n".join(str(element.value) for element in [*at.sidebar.markdown, *at.sidebar.caption])
    assert ":red-badge[unreachable] API status" in sidebar and "Status check failed: connection failed" in sidebar


def test_docker_context_excludes_secrets_in_every_folder():
    # .dockerignore patterns match from the context root: without "**/" a nested .env would be baked in.
    patterns = [line.strip() for line in (REPO_ROOT / ".dockerignore").read_text().splitlines()]
    for pattern in ("**/.env", "**/.env.*", "**/.envrc", "**/*.key", "**/*.pem", "**/.streamlit/secrets.toml"):
        assert pattern in patterns, pattern
    assert patterns.index("!.env.example") > patterns.index("**/.env.*")
    assert not {".env", ".env.*", "*.key", "*.pem", ".streamlit/secrets.toml"} & set(patterns)  # root-only forms


def test_streamlit_settings_sit_next_to_the_entry_point():
    config = tomllib.loads((ENTRY_POINT.parent / ".streamlit" / "config.toml").read_text())
    assert config["server"]["address"] == "127.0.0.1" and config["browser"]["gatherUsageStats"] is False
    assert config["server"]["maxUploadSize"] == 20


# ------------------------------------------------------------------ launcher


def _get(url: str) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=2) as response:
        return response.status


def _wait_for(url: str, deadline: float, process: subprocess.Popen) -> None:
    while time.monotonic() < deadline:
        assert process.poll() is None, "process exited early"
        try:
            if _get(url) == 200:
                return
        except OSError:
            pass
        time.sleep(0.25)
    pytest.fail(f"{url} did not answer")


def _refuses_connections(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _two_free_ports() -> tuple:
    first, second = closed_port(), closed_port()
    while second == first:
        second = closed_port()
    return first, second


def _gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


class _Output:
    """Collects a child's combined output on a thread."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.lines: list = []
        self._thread = threading.Thread(target=lambda: self.lines.extend(process.stdout), daemon=True)
        self._thread.start()

    def join(self, timeout: float = 10) -> str:
        """All output, once the pipe is closed (the process and its children have exited)."""

        self._thread.join(timeout)
        return self.text

    def wait_for(self, text: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while text not in self.text and time.monotonic() < deadline:
            time.sleep(0.05)
        assert text in self.text, self.text
        return self.text

    @property
    def text(self) -> str:
        return "".join(self.lines)


# A stand-in server for the launcher's children (answers 200 to everything, stops on SIGTERM).
_FAKE_SERVER = (
    "import http.server, sys\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
    "    def log_message(self, *args): pass\n"
    "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n"
)


def _launcher(api_command: str, ui_command: str, *args: str) -> subprocess.Popen:
    """run_app.py in its own interpreter, with the two server commands replaced (Python expressions of host, port)."""

    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from scripts import run_app\n"
        f"FAKE = {_FAKE_SERVER!r}\n"
        f"run_app.api_command = lambda host, port: {api_command}\n"
        f"run_app.ui_command = lambda host, port: {ui_command}\n"
        "sys.exit(run_app.main(sys.argv[1:]))\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code, "--no-browser", "--stop-timeout", "5", *args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip


def test_launcher_arguments():
    args = run_app.parse_args([])
    assert (args.host, args.api_port, args.ui_port, args.open_browser) == ("127.0.0.1", 8000, 8501, True)
    args = run_app.parse_args(["--host", "0.0.0.0", "--api-port", "9000", "--ui-port", "9001", "--no-browser"])
    assert (args.api_port, args.ui_port, args.open_browser) == (9000, 9001, False)
    assert run_app.parse_args(["--host", "[::1]"]).host == "::1"
    for bad in (["--api-port", "70000"], ["--ui-port", "http"], ["--api-port", "9000", "--ui-port", "9000"]):
        with pytest.raises(SystemExit):
            run_app.parse_args(bad)
    with pytest.raises(SystemExit):
        run_app.parse_args(["--stop-timeout", "0"])
    assert run_app.local_url("0.0.0.0", 8000) == "http://127.0.0.1:8000"
    assert run_app.local_url("::", 8000) == "http://[::1]:8000"
    assert run_app.local_url("192.168.1.20", 8501) == "http://192.168.1.20:8501"
    assert run_app.api_command("127.0.0.1", 8000)[:4] == [sys.executable, "-m", "uvicorn", "app.main:app"]
    assert run_app.ui_command("127.0.0.1", 8501)[:3] == [sys.executable, "-m", "streamlit"]
    lan = run_app.lan_address()
    assert lan is None or (re.fullmatch(r"\d+\.\d+\.\d+\.\d+", lan) and not lan.startswith("127."))


@pytest.mark.parametrize("option", ["--api-port", "--ui-port"])
def test_launcher_refuses_a_busy_port(option, capsys):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        other = "--ui-port" if option == "--api-port" else "--api-port"
        assert run_app.main([option, str(port), other, str(closed_port()), "--no-browser"]) == 1
    error = capsys.readouterr().err
    label = "API" if option == "--api-port" else "UI"
    assert f"Port {port} on 127.0.0.1 is already in use." in error and f"pick another {label} port with {option}" in error


def test_launcher_names_an_earlier_run_holding_the_port(stand_in_api, capsys):
    url, _ = stand_in_api
    port = int(url.rsplit(":", 1)[1])
    assert run_app.main(["--api-port", str(port), "--ui-port", str(closed_port()), "--no-browser"]) == 1
    error = capsys.readouterr().err
    assert "already in use by an F1 AI Copilot API; an earlier run of this launcher may still be running" in error
    assert f":{port}`" in error  # how to find the process


def test_launcher_rejects_an_address_that_is_not_local(capsys):
    with socket.socket() as probe:  # TEST-NET-1 is never assigned, unless the kernel allows non-local binds
        try:
            probe.bind(("192.0.2.77", 0))
        except OSError:
            pass
        else:
            pytest.skip("this system allows binding addresses it does not have (ip_nonlocal_bind)")
    assert run_app.main(["--host", "192.0.2.77", "--no-browser"]) == 2
    error = capsys.readouterr().err
    assert error.startswith("Cannot listen on --host 192.0.2.77") and "already in use" not in error


def test_launcher_restores_signal_handlers(monkeypatch):
    handled = [signal.SIGTERM] + [getattr(signal, name) for name in ("SIGHUP", "SIGBREAK") if hasattr(signal, name)]
    before = {signum: signal.getsignal(signum) for signum in [signal.SIGINT, *handled]}
    during: dict = {}

    class Exited:
        pid, returncode = 4242, 7

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def fake_start(command, env):
        during.update({signum: signal.getsignal(signum) for signum in handled})
        return Exited()

    monkeypatch.setattr(run_app, "start", fake_start)
    assert run_app.main(["--api-port", str(closed_port()), "--ui-port", str(closed_port()), "--no-browser"]) == 7
    assert during and all(handler is run_app._raise_stop for handler in during.values())
    assert {signum: signal.getsignal(signum) for signum in before} == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and process groups")
def test_launcher_kills_a_child_that_ignores_sigterm(capsys):
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"
    child = subprocess.Popen([sys.executable, "-c", stubborn], stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        started = time.monotonic()
        run_app.stop({"stubborn server": child}, timeout=0.5)
        assert child.returncode == -signal.SIGKILL and time.monotonic() - started < 5
        assert "The stubborn server did not stop within 0.5 s; killing it." in capsys.readouterr().err
    finally:
        if child.poll() is None:
            child.kill()
        child.stdout.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_launcher_stops_the_other_server_when_one_exits():
    api_port, ui_port = _two_free_ports()
    launcher = _launcher(
        "[sys.executable, '-c', 'raise SystemExit(3)']", "[sys.executable, '-c', FAKE, str(port)]",
        "--api-port", str(api_port), "--ui-port", str(ui_port),
    )  # fmt: skip
    try:
        output, _ = launcher.communicate(timeout=30)
    finally:
        if launcher.poll() is None:
            launcher.kill()
    assert launcher.returncode == 3, output
    assert "The API exited unexpectedly (exit code 3); stopping the other server." in output
    assert _refuses_connections(ui_port)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_servers_stop_when_the_launcher_is_killed():
    api_port, ui_port = _two_free_ports()
    fake = "[sys.executable, '-c', FAKE, str(port)]"
    launcher = _launcher(fake, fake, "--api-port", str(api_port), "--ui-port", str(ui_port))
    output = _Output(launcher)
    try:
        log = output.wait_for("Press Ctrl+C", timeout=30)
        pids = [int(pid) for pid in re.findall(r"pid (\d+)", log)]
        assert len(pids) == 2, log
        for pid in pids:  # own session: a terminal's Ctrl+C reaches only the launcher
            assert os.getpgid(pid) == pid != os.getpgid(launcher.pid)
        launcher.send_signal(signal.SIGKILL)
        launcher.wait(timeout=10)
        assert all(_gone(pid, timeout=10) for pid in pids), "servers outlived the launcher"
        assert _refuses_connections(api_port) and _refuses_connections(ui_port)
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="signals the launcher with SIGTERM")
def test_launcher_starts_both_servers_and_stops_them(tmp_path):
    api_port, ui_port = _two_free_ports()
    env = {**os.environ, "F1_ARTIFACTS_DIR": str(tmp_path / "artifacts"), "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none"}
    process = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_app.py"), "--api-port", str(api_port), "--ui-port", str(ui_port),
         "--no-browser", "--stop-timeout", "8"],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    output = _Output(process)
    try:
        deadline = time.monotonic() + 60
        _wait_for(f"http://127.0.0.1:{api_port}/", deadline, process)
        _wait_for(f"http://127.0.0.1:{ui_port}/_stcore/health", deadline, process)
        output.wait_for("Press Ctrl+C", timeout=max(1.0, deadline - time.monotonic()))
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 0, output.text
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    log = output.join()
    pids = [int(pid) for pid in re.findall(r"pid (\d+)", log)]
    assert len(pids) == 2, log
    assert f"Web UI: http://127.0.0.1:{ui_port}" in log and "Stopping the API and the web UI" in log
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert _refuses_connections(api_port) and _refuses_connections(ui_port)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_ui_started_from_another_folder_keeps_its_settings(tmp_path):
    """The Streamlit config applies from any working directory: local address only, no usage statistics."""

    port = closed_port()
    env = {**os.environ, "F1_API_URL": f"http://127.0.0.1:{closed_port()}", "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none"}
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ENTRY_POINT), "--server.port", str(port), "--server.headless", "true"],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    output = _Output(process)
    try:
        _wait_for(f"http://127.0.0.1:{port}/_stcore/health", time.monotonic() + 60, process)
        log = output.wait_for(f"URL: http://127.0.0.1:{port}", timeout=10)
    finally:
        process.terminate()
        process.wait(timeout=20)
    assert "Network URL" not in log and "External URL" not in log, log
