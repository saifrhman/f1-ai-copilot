"""HTTP client for the F1 AI Copilot API; the only place the UI talks to the server.

The UI never imports the API or its modules: every request goes over HTTP, so the
API stays the single validated entry point (and the only process that opens the
embedded Qdrant index). Errors surface as two exception types:

* ``ApiUnavailable``: the API could not be reached or did not answer in time;
* ``ApiError``: the API answered with an HTTP error (4xx/5xx), or with a body that is
  not a JSON object (another service at that address); ``detail`` carries the API's
  own explanation in readable form. Its subclass ``RequestNotSent`` means the request
  body could not be encoded as JSON (e.g. an infinite number), so nothing was sent.

The base URL comes from ``F1_API_URL`` (environment, then ``.env``), default
``http://127.0.0.1:8000``. Proxy settings from the environment (HTTP_PROXY, ALL_PROXY, ...)
are ignored: the API is the user's own server on this computer or network. ``user:password@``
in the URL is sent as Basic auth but never displayed (``ApiClient.base_url`` omits it).
Tests inject a client bound to the ASGI app with ``set_client_override``.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import httpx
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://127.0.0.1:8000"

CONNECT_TIMEOUT_S = 5.0
DEFAULT_READ_TIMEOUT_S = 60.0
# Model-provider calls (embedding, answer, claim verification), Optuna search and Whisper can be slow.
SLOW_READ_TIMEOUT_S = 300.0

CALIBRATE_TYRES_PATH = "/api/strategy/calibrate-tyres"  # optional endpoint: pages check the OpenAPI schema
GHOST_ARTIFACT_PREFIX = "/artifacts/ghost/"
_USERINFO = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*://)[^/?#]*@")


class ApiUnavailable(Exception):
    """The API could not be reached, or did not answer within the timeout."""

    def __init__(self, base_url: str, reason: str, hint: str) -> None:
        super().__init__(f"Cannot use the API at {base_url}: {reason}")
        self.base_url = base_url
        self.reason = reason
        self.hint = hint


class ApiError(Exception):
    """The API answered with an HTTP error; ``detail`` is its readable explanation."""

    def __init__(
        self, status_code: int, detail: str, *, errors: Optional[List[str]] = None, error_type: Optional[str] = None
    ) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.errors = errors or []  # one readable line per validation error (HTTP 422)
        self.error_type = error_type


class RequestNotSent(ApiError):
    """The request body could not be encoded as JSON, so nothing was sent (``status_code`` 0)."""

    def __init__(self, reason: str) -> None:
        super().__init__(0, f"the request could not be encoded as JSON: {reason}")


def encode_json(payload: Any) -> bytes:
    """``payload`` as a JSON request body, encoded as httpx would; raises ``RequestNotSent``."""

    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate such as "\ud800"
        raise RequestNotSent("it contains text that is not valid Unicode") from None
    except ValueError as exc:  # NaN or infinity ("Out of range float values ..."), or a circular reference
        reason = str(exc)
        if reason.startswith("Out of range float"):  # the value is named only by Python 3.12+
            reason = "it contains a number JSON cannot carry (NaN or infinity)"
        raise RequestNotSent(reason) from None
    except RecursionError:
        raise RequestNotSent("it is nested too deeply") from None


def redact_url(url: str) -> str:
    """``url`` without ``user:password@`` (httpx sends it as Basic auth; it must never be displayed)."""

    return _USERINFO.sub(r"\1", url)


def is_loopback(url: str) -> bool:
    """Whether ``url`` points at this computer (localhost, 127.0.0.0/8 or ::1)."""

    host = httpx.URL(url).host
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def start_api_hint(base_url: str) -> str:
    """How to get the API at ``base_url`` running, for connection failures."""

    url = httpx.URL(base_url)
    elsewhere = "If the API runs elsewhere, set `F1_API_URL` (e.g. `F1_API_URL=http://192.168.1.20:8000`) and restart the UI."
    if url.host == "api":  # the API service of this project's Docker Compose stack
        return (
            "Check the API container with `docker compose ps` and `docker compose logs api` in the project folder, "
            "and start it with `docker compose up -d api`."
        )
    if not is_loopback(base_url):
        return (
            f"Make sure the API on {url.host} is running and reachable from this computer (in a container: "
            f"`docker ps` and `docker logs <container>`). {elsewhere}"
        )
    options = f" --host {url.host}" if url.host == "::1" else ""
    port = url.port or (443 if url.scheme == "https" else 80)
    options += f" --port {port}" if port != 8000 else ""
    return (
        f"Start it from the project folder with `uvicorn app.main:app{options}`, or stop this UI and start both "
        f"with `python scripts/run_app.py`. {elsewhere}"
    )


def normalise_base_url(url: str) -> str:
    """``host:port`` or ``http(s)://[user:password@]host[:port][/prefix]`` without a trailing slash."""

    url = url.strip().rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise ValueError(f"F1_API_URL {redact_url(url)!r} is not a valid URL: {exc}") from exc
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise ValueError(f"F1_API_URL must look like http://host:port, got {redact_url(url)!r}")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError(f"F1_API_URL port must be 1-65535, got {redact_url(url)!r}")
    if parsed.query or parsed.fragment:  # not echoed: a query string can carry a token
        raise ValueError("F1_API_URL must not contain a query string (?...) or fragment (#...)")
    return url


def resolve_base_url() -> str:
    """F1_API_URL from the environment, else from the project's .env (only that key is read), else the default."""

    url = os.getenv("F1_API_URL", "").strip()
    if not url and os.getenv("F1_COPILOT_LOAD_DOTENV", "1") != "0":
        url = (dotenv_values(PROJECT_ROOT / ".env").get("F1_API_URL") or "").strip()
    return normalise_base_url(url or DEFAULT_API_URL)


def format_validation_error(error: Any) -> str:
    """One FastAPI validation error as ``field.path[0]: message (got value)``."""

    if not isinstance(error, dict):
        return str(error)
    loc = list(error.get("loc") or [])
    if loc and loc[0] in ("body", "query", "path"):
        loc = loc[1:]
    location = ""
    for part in loc:
        location += f"[{part}]" if isinstance(part, int) else (f".{part}" if location else str(part))
    text = f"{location or 'request body'}: {error.get('msg', 'invalid value')}"
    value = error.get("input")
    summarised = isinstance(value, str) and value.startswith(("<object with", "<array with"))
    if error.get("type") != "missing" and value is not None and not summarised:
        text += f" (got {value!r})"
    return text


def _error_from_response(response: Any) -> ApiError:
    status = response.status_code
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and "detail" in body:
        error_type = body.get("error_type") if isinstance(body.get("error_type"), str) else None
        detail = body["detail"]
        if isinstance(detail, list):
            errors = [format_validation_error(item) for item in detail]
            more = body.get("more_errors")
            if isinstance(more, int) and more > 0:
                errors.append(f"... and {more} more")
            return ApiError(status, "; ".join(errors) or "invalid request", errors=errors, error_type=error_type)
        return ApiError(status, str(detail), error_type=error_type)
    text = " ".join(response.text.split())[:300]
    return ApiError(status, text or f"HTTP {status} {response.reason_phrase}".strip())


class ApiClient:
    """One method per API endpoint; every call returns the decoded JSON body.

    ``http`` may be any httpx-compatible client (for tests: Starlette's ``TestClient``), used
    with its own timeouts; by default an ``httpx.Client`` for ``base_url`` (or
    ``resolve_base_url()``) is created and every request gets the timeouts defined above.
    ``base_url`` is the address for display: it never contains credentials.
    """

    def __init__(self, base_url: Optional[str] = None, *, http: Any = None) -> None:
        self._per_request_timeouts = http is None
        if http is None:
            url = normalise_base_url(base_url) if base_url else resolve_base_url()
            http = httpx.Client(base_url=url, follow_redirects=False, trust_env=False)
        else:
            url = (base_url or str(http.base_url)).rstrip("/")
        self.base_url = redact_url(url)
        self._http = http

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ transport

    def _send(self, method: str, path: str, *, payload: Any = None, read_timeout: float = DEFAULT_READ_TIMEOUT_S) -> Any:
        options: Dict[str, Any] = {}
        if payload is not None:  # encoded here, so a value JSON cannot carry is reported instead of raised by httpx
            options.update(content=encode_json(payload), headers={"Content-Type": "application/json"})
        if self._per_request_timeouts:
            options["timeout"] = httpx.Timeout(CONNECT_TIMEOUT_S, read=read_timeout, write=read_timeout)
        try:
            response = self._http.request(method, path, **options)
        except httpx.ConnectTimeout as exc:
            raise ApiUnavailable(
                self.base_url, f"no connection within {CONNECT_TIMEOUT_S:.0f} s", start_api_hint(self.base_url)
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiUnavailable(
                self.base_url,
                f"no answer within {read_timeout:.0f} s",
                "The API is running but did not finish in time (model provider, search or transcription "
                "can be slow). Try again, or check the API log for the request.",
            ) from exc
        except httpx.ConnectError as exc:
            raise ApiUnavailable(self.base_url, f"connection failed ({exc})", start_api_hint(self.base_url)) from exc
        except httpx.TransportError as exc:
            raise ApiUnavailable(
                self.base_url, f"the connection broke ({type(exc).__name__}: {exc})", start_api_hint(self.base_url)
            ) from exc
        if not 200 <= response.status_code < 300:
            raise _error_from_response(response)
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        """The response body, which every endpoint returns as a JSON object."""

        response = self._send(method, path, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise ApiError(
                response.status_code,
                f"{path} did not return a JSON object; is {self.base_url} the F1 AI Copilot API (check F1_API_URL)?",
            )
        return body

    def _post(self, path: str, payload: Mapping[str, Any], read_timeout: float = DEFAULT_READ_TIMEOUT_S) -> Dict[str, Any]:
        return self._json("POST", path, payload=dict(payload), read_timeout=read_timeout)

    # ------------------------------------------------------------------ service

    def root(self) -> Dict[str, Any]:
        return self._json("GET", "/")

    def health(self) -> Dict[str, Any]:
        """Readiness of every module (``status`` is ``healthy`` or ``degraded``)."""

        return self._json("GET", "/health")

    def get_openapi(self) -> Dict[str, Any]:
        return self._json("GET", "/openapi.json")

    # ------------------------------------------------------------------ FIA regulation RAG

    def fia_status(self) -> Dict[str, Any]:
        return self._json("GET", "/api/fia/status")

    def fia_query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"question": question}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._post("/api/fia/query", payload, SLOW_READ_TIMEOUT_S)

    def fia_retrieve(self, question: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"question": question}
        if top_k is not None:
            payload["top_k"] = top_k
        if min_score is not None:
            payload["min_score"] = float(min_score)
        return self._post("/api/fia/retrieve", payload, SLOW_READ_TIMEOUT_S)

    # ------------------------------------------------------------------ heuristic modules

    def generate_strategy(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post("/api/strategy/generate", request, SLOW_READ_TIMEOUT_S)

    def calibrate_tyres(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post(CALIBRATE_TYRES_PATH, request, SLOW_READ_TIMEOUT_S)

    def recommend_setup(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post("/api/setup/recommend", request, SLOW_READ_TIMEOUT_S)

    def generate_ghost(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post("/api/ghost/generate", request, SLOW_READ_TIMEOUT_S)

    def fetch_ghost_image(self, visualization_url: str) -> bytes:
        """PNG bytes of a ghost comparison (``visualization_url`` from ``generate_ghost``)."""

        if not visualization_url.startswith(GHOST_ARTIFACT_PREFIX) or ".." in visualization_url:
            raise ValueError(f"Not a ghost artifact URL: {visualization_url!r}")
        response = self._send("GET", visualization_url)
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/png"):
            raise ApiError(response.status_code, f"Expected a PNG image, got {content_type or 'no content type'}")
        return response.content

    def classify_emotion(self, audio_file: str, transcribe: bool = False) -> Dict[str, Any]:
        """``audio_file`` is base64 audio or a ``data:audio/<type>;base64,...`` URI."""

        return self._post("/api/emotion/classify", {"audio_file": audio_file, "transcribe": transcribe}, SLOW_READ_TIMEOUT_S)

    def natural_query(self, query: str, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query}
        if context is not None:
            payload["context"] = dict(context)
        return self._post("/api/query/natural", payload, SLOW_READ_TIMEOUT_S)

    def predict_penalty(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Heuristic incident triage (not an FIA steward-decision predictor)."""

        return self._post("/api/penalty/predict", request)


_override: Optional[ApiClient] = None


@lru_cache(maxsize=4)
def _shared_client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


def get_client() -> ApiClient:
    """The client for this process (the test override when one is installed)."""

    return _override if _override is not None else _shared_client(resolve_base_url())


def set_client_override(client: ApiClient) -> None:
    global _override
    _override = client


def clear_client_override() -> None:
    global _override
    _override = None
