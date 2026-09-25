"""Wire-level contract tests for the RAG's OpenAI code path.

``FIARegulationRAG`` creates the real LangChain ``OpenAIEmbeddings`` and
``ChatOpenAI`` clients (and through them the ``openai`` SDK and its HTTP
stack). Here they talk over HTTP to a local server that implements the part of
the OpenAI REST contract the pipeline uses - ``POST /v1/embeddings`` and
``POST /v1/chat/completions`` - with OpenAI's response shapes, error bodies
and status codes, and basic request validation (a subset of OpenAI's). The
server records every request, so the tests assert on what actually goes over
the wire (headers, JSON bodies, batching, retries) and on how the pipeline
reacts to provider responses and failures.

The server is a stand-in for the provider, not a model: its embeddings are the
deterministic hashed bag of words from ``tests.helpers.HashingEmbeddings``
(cosine similarity reflects word overlap, so retrieval is meaningful), and chat
replies are scripted per test. DNS lookups and connections for anything but
the loopback interface are refused while these tests run, so no request can
reach a real provider.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import pytest
from qdrant_client import QdrantClient

from core_modules.rule_checker.fia_rag import (
    ChunkingConfig,
    EmbeddingConfig,
    FIARegulationRAG,
    GenerationConfig,
    ProviderConfig,
    ProviderError,
    QdrantConfig,
    RAGSettings,
    RetrievalConfig,
)
from core_modules.rule_checker.fia_rag.embeddings import create_openai_embeddings
from core_modules.rule_checker.fia_rag.generation import SYSTEM_PROMPT
from core_modules.rule_checker.fia_rag.grounding import DeclineReason
from tests.helpers import HashingEmbeddings, fia_page, write_pdf
from tests.test_fia_generation import cite_passage_containing
from tests.test_fia_index_retrieval import FUEL_FLOW, PIT_LANE, REAR_WING, UNSAFE_RELEASE

API_KEY = "sk-test-wire"
EMBEDDING_MODEL = "text-embedding-wire-test"
CHAT_MODEL = "gpt-wire-test-chat"
DIMENSION = 384
BATCH_SIZE = 3
MAX_OUTPUT_TOKENS = 256
EMBEDDINGS = "/v1/embeddings"
CHAT = "/v1/chat/completions"
PIT_QUESTION = "What is the speed limit in the pit lane?"
_CHAT_ROLES = {"system", "developer", "user", "assistant", "tool"}

ChatReply = Callable[[Dict[str, Any]], Tuple[str, str]]


# ------------------------------------------------------------------ the local OpenAI-compatible server


@dataclass(frozen=True)
class RecordedRequest:
    path: str
    headers: Dict[str, str]  # lower-case header names
    body: Any  # decoded JSON (None when the body was not valid JSON)


@dataclass
class ScriptedResponse:
    """Overrides the next request to one endpoint: an error reply and/or a delay.

    During ``delay`` the server sends nothing at all (no status line, no bytes);
    with ``status=None`` it then answers normally.
    """

    status: Optional[int] = None
    error: Optional[Dict[str, Any]] = None  # OpenAI error object, sent as {"error": ...}
    headers: Dict[str, str] = field(default_factory=dict)
    delay: float = 0.0


def openai_error(message: str, error_type: str, code: Optional[str] = None, param: Optional[str] = None) -> Dict[str, Any]:
    return {"message": message, "type": error_type, "param": param, "code": code}


def rate_limit(retry_after_ms: int = 10) -> ScriptedResponse:
    return ScriptedResponse(
        status=429,
        error=openai_error("Rate limit reached for requests. Please try again shortly.", "requests", "rate_limit_exceeded"),
        # The SDK honours retry-after-ms, which keeps the retry fast.
        headers={"retry-after-ms": str(retry_after_ms)},
    )


def float32(vector: List[float]) -> List[float]:
    """The values a float32 wire encoding can carry (what OpenAI's embeddings are)."""

    return list(struct.unpack(f"<{len(vector)}f", struct.pack(f"<{len(vector)}f", *vector)))


class OpenAIStubServer:
    """Threaded HTTP server implementing OpenAI's embeddings and chat-completions endpoints.

    * Authentication: ``Authorization: Bearer <api_key>`` is required; any other
      key gets OpenAI's 401 ``invalid_api_key`` error, whose message echoes the
      presented key (so tests can check that it never leaks into our errors).
    * Embeddings: ``input`` may be a string or a list of strings/token arrays
      (as OpenAI accepts); ``encoding_format`` ``"float"`` returns JSON numbers,
      ``"base64"`` returns base64 of little-endian float32 values. Both carry
      the same float32-rounded vector.
    * Chat: non-streaming completions only (``stream: true`` is rejected with a
      400, so a switch to streaming would be noticed); ``chat_reply(body)``
      returns ``(content, finish_reason)``.
    * ``usage`` counts whitespace-separated words, not tokens.

    HTTP/1.0 is used so every request has its own connection: no handler thread
    outlives a test waiting on an idle keep-alive connection.
    """

    def __init__(self, api_key: str = API_KEY, dimension: int = DIMENSION):
        self.api_key = api_key
        self._embedder = HashingEmbeddings(dimension)
        self.requests: List[RecordedRequest] = []
        self.scripted: Dict[str, Deque[ScriptedResponse]] = {EMBEDDINGS: deque(), CHAT: deque()}
        self.chat_reply: ChatReply = lambda body: ("INSUFFICIENT_EVIDENCE", "stop")
        self._completions = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, format, *args):  # keep test output clean
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                headers = {name.lower(): value for name, value in self.headers.items()}
                status, payload, extra_headers = stub.respond(urlsplit(self.path).path, headers, body)
                data = json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    for name, value in extra_headers.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up waiting (timeout tests)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        # A short poll interval keeps shutdown() (and so each test's teardown) fast.
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, name="openai-wire-stub", daemon=True
        )

    # -------------------------------------------------------------- lifecycle and inspection
    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> "OpenAIStubServer":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()  # releases handlers that are sleeping to simulate a slow provider
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    def script(self, path: str, *responses: ScriptedResponse) -> None:
        with self._lock:
            self.scripted[path].extend(responses)

    def requests_to(self, path: str) -> List[RecordedRequest]:
        with self._lock:
            return [request for request in self.requests if request.path == path]

    def vector(self, text: str) -> List[float]:
        """The embedding this server returns for ``text``."""

        return float32(self._embedder.embed_query(text))

    # -------------------------------------------------------------- request handling
    def respond(self, path: str, headers: Dict[str, str], body: Any) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        with self._lock:
            self.requests.append(RecordedRequest(path, headers, body))
            queue = self.scripted.get(path)
            step = queue.popleft() if queue else None
        if step and step.delay:
            self._stop.wait(step.delay)
        if step and step.status is not None:
            return step.status, {"error": step.error}, step.headers
        presented = headers.get("authorization", "")
        if presented != f"Bearer {self.api_key}":
            key = presented.removeprefix("Bearer ").strip()
            message = (
                f"Incorrect API key provided: {key}. You can find your API key at https://platform.openai.com/account/api-keys."
                if key
                else "You didn't provide an API key. You need to provide your API key in an Authorization header."
            )
            return 401, {"error": openai_error(message, "invalid_request_error", "invalid_api_key")}, {}
        if path == EMBEDDINGS:
            return self._embeddings(body)
        if path == CHAT:
            return self._chat(body)
        return 404, {"error": openai_error(f"Invalid URL (POST {path})", "invalid_request_error")}, {}

    @staticmethod
    def _bad_request(message: str, param: Optional[str] = None) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        return 400, {"error": openai_error(message, "invalid_request_error", param=param)}, {}

    def _embeddings(self, body: Any) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"]:
            return self._bad_request("you must provide a model parameter", "model")
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= 2048:
            return self._bad_request("'$.input' is invalid: expected a string or 1-2048 inputs", "input")
        encoding = body.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            return self._bad_request(f"Invalid encoding_format {encoding!r}; expected 'float' or 'base64'", "encoding_format")
        data = []
        for position, item in enumerate(inputs):
            if isinstance(item, str) and item:
                text = item
            elif isinstance(item, list) and item and all(isinstance(token, int) for token in item):
                text = " ".join(str(token) for token in item)  # token arrays are accepted, as OpenAI does
            else:
                return self._bad_request(f"'$.input[{position}]' must be a non-empty string or token array", "input")
            vector = self.vector(text)
            packed = struct.pack(f"<{len(vector)}f", *vector)
            embedding: Any = base64.b64encode(packed).decode("ascii") if encoding == "base64" else vector
            data.append({"object": "embedding", "index": position, "embedding": embedding})
        words = sum(len(str(item).split()) for item in inputs)
        return 200, {
            "object": "list",
            "data": data,
            "model": body["model"],
            "usage": {"prompt_tokens": words, "total_tokens": words},
        }, {}

    def _chat(self, body: Any) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"]:
            return self._bad_request("you must provide a model parameter", "model")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(
            isinstance(message, dict) and message.get("role") in _CHAT_ROLES for message in messages
        ):
            return self._bad_request("'messages' must be a non-empty list of role/content objects", "messages")
        if body.get("stream"):
            return self._bad_request("this server implements only non-streaming chat completions", "stream")
        temperature = body.get("temperature", 1)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            return self._bad_request("temperature must be between 0 and 2", "temperature")
        content, finish_reason = self.chat_reply(body)
        with self._lock:
            self._completions += 1
            number = self._completions
        prompt_words = sum(len(str(message.get("content", "")).split()) for message in messages)
        completion_words = len(content.split())
        return 200, {
            "id": f"chatcmpl-wire{number}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "refusal": None},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_words,
                "completion_tokens": completion_words,
                "total_tokens": prompt_words + completion_words,
            },
            "system_fingerprint": None,
        }, {}


def replying_with(responder: Callable[[list], str], finish_reason: str = "stop") -> ChatReply:
    """Adapt a ``test_fia_generation`` responder (messages with ``.content``) to a wire request body."""

    def reply(body: Dict[str, Any]) -> Tuple[str, str]:
        return responder([SimpleNamespace(content=message["content"]) for message in body["messages"]]), finish_reason

    return reply


# ------------------------------------------------------------------ fixtures


def _is_loopback(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch):
    """Refuse DNS lookups and connections to non-loopback hosts; drop proxy/tracing settings."""

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_loopback(host):
            raise socket.gaierror(socket.EAI_NONAME, f"wire tests must not resolve {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):
            raise OSError(f"wire tests must not open network connections (attempted {address[0]})")
        return real_connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def server():
    stub = OpenAIStubServer().start()
    yield stub
    stub.close()


@pytest.fixture
def qdrant():
    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "fia_docs"
    write_pdf(folder / "section_b_sporting.pdf", [fia_page(1, PIT_LANE), None, fia_page(3, UNSAFE_RELEASE)])
    write_pdf(folder / "section_c_technical.pdf", [FUEL_FLOW, REAR_WING])
    return folder


@pytest.fixture
def make_rag(docs, tmp_path, server, qdrant):
    """A pipeline whose embeddings and chat model are the real LangChain/OpenAI clients."""

    def make(
        api_key: str = API_KEY, timeout: float = 5.0, max_retries: int = 1, temperature: Optional[float] = None
    ) -> FIARegulationRAG:
        generation: Dict[str, Any] = {"model": CHAT_MODEL, "max_output_tokens": MAX_OUTPUT_TOKENS}
        if temperature is not None:  # otherwise GenerationConfig's default applies
            generation["temperature"] = temperature
        settings = RAGSettings(
            docs_path=docs,
            chunking=ChunkingConfig(chunk_size=400, chunk_overlap=40),
            embedding=EmbeddingConfig(model=EMBEDDING_MODEL, batch_size=BATCH_SIZE, cache_path=None),
            retrieval=RetrievalConfig(top_k=4, min_score=0.2),
            generation=GenerationConfig(**generation),
            provider=ProviderConfig(
                api_key=api_key, base_url=server.base_url, timeout_seconds=timeout, max_retries=max_retries
            ),
            qdrant=QdrantConfig(collection="fia_wire_test", path=tmp_path / "unused"),
        )
        # No embeddings/llm overrides: FIARegulationRAG creates the real clients itself.
        return FIARegulationRAG(settings, qdrant_client=qdrant)

    return make


# ------------------------------------------------------------------ embeddings over the wire


def test_index_build_sends_openai_embedding_requests(make_rag, server):
    rag = make_rag()
    report = rag.build_index()
    requests = server.requests_to(EMBEDDINGS)

    assert report.status == "rebuilt" and report.chunks == 4
    expected_sizes = [min(BATCH_SIZE, report.chunks - start) for start in range(0, report.chunks, BATCH_SIZE)]
    assert [len(request.body["input"]) for request in requests] == expected_sizes == [3, 1]
    for request in requests:
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        assert request.headers["content-type"].startswith("application/json")
        assert request.body["model"] == EMBEDDING_MODEL
        # Plain strings, not tiktoken token arrays (which OpenAI-compatible providers reject).
        assert all(isinstance(text, str) and text for text in request.body["input"])
        # Neither the pipeline nor LangChain sets a format, so the openai SDK asks for base64.
        assert request.body["encoding_format"] == "base64"
    sent = [text for request in requests for text in request.body["input"]]
    assert len(set(sent)) == report.chunks
    assert any("80km/h" in text for text in sent) and any("fuel mass flow" in text for text in sent)
    assert rag.status()["index"]["vector_size"] == DIMENSION


def test_base64_vectors_are_decoded_exactly_and_drive_retrieval(make_rag, server):
    rag = make_rag()
    rag.build_index()

    # float32 -> Python float is exact, so the decoded vector must equal what was sent bit for bit.
    assert rag.embedder().embed_query("pit lane speed limit") == server.vector("pit lane speed limit")

    pit = rag.retrieve(PIT_QUESTION)
    assert server.requests_to(EMBEDDINGS)[-1].body["input"] == [PIT_QUESTION]
    assert "80km/h" in pit.passages[0].text
    fuel = rag.retrieve("What is the maximum fuel mass flow per hour?")
    assert "fuel mass flow" in fuel.passages[0].text and "80km/h" not in fuel.passages[0].text


def test_server_encodings_agree_through_the_real_sdk(server):
    """Self-check of the stub: base64 and float replies decode to the same vector in the openai SDK."""

    embeddings = create_openai_embeddings(
        EmbeddingConfig(model=EMBEDDING_MODEL, batch_size=4),
        ProviderConfig(api_key=API_KEY, base_url=server.base_url, timeout_seconds=5.0, max_retries=0),
    )
    text = "rear wing flap position"
    via_base64 = embeddings.embed_query(text)
    via_float = embeddings.client.create(input=[text], model=EMBEDDING_MODEL, encoding_format="float").data[0].embedding

    assert [request.body["encoding_format"] for request in server.requests_to(EMBEDDINGS)] == ["base64", "float"]
    assert via_base64 == via_float == server.vector(text)
    assert len(via_base64) == DIMENSION


# ------------------------------------------------------------------ chat completions over the wire


@pytest.mark.parametrize(
    "temperature, sent_temperature",
    [
        pytest.param(None, 0, id="default-temperature"),  # the production default is temperature 0
        pytest.param(0.3, 0.3, id="configured-temperature"),  # a configured value is sent unchanged
    ],
)
def test_grounded_answer_round_trips_over_the_chat_api(make_rag, server, temperature, sent_temperature):
    server.chat_reply = replying_with(cite_passage_containing("80km/h"))
    rag = make_rag(temperature=temperature)
    rag.build_index()
    result = rag.answer(PIT_QUESTION)

    [request] = server.requests_to(CHAT)
    body = request.body
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert body["model"] == CHAT_MODEL
    assert body["temperature"] == sent_temperature
    assert body["max_completion_tokens"] == MAX_OUTPUT_TOKENS
    assert not body.get("stream")
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    user = body["messages"][1]["content"]
    assert '<excerpt label="S1"' in user and f"<question>\n{PIT_QUESTION}\n</question>" in user

    assert result["grounded"] is True and result["status"] == "answered"
    cited = [p for p in result["retrieved_passages"] if p["cited"]]
    assert result["citations"] == [cited[0]["label"]] and "80km/h" in cited[0]["text"]
    assert result["answer"] == f"The regulations state: 80km/h [{cited[0]['label']}]."
    assert result["models"] == {"embedding": EMBEDDING_MODEL, "generation": CHAT_MODEL}
    assert rag.status()["provider_status"] == "ok"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_incomplete_completion_is_declined(make_rag, server, finish_reason):
    """OpenAI's finish reasons for a cut-off reply: the token limit, or content omitted by its filter."""

    server.chat_reply = lambda body: ("The pit lane speed limit is 80km/h [S1] unless the", finish_reason)
    rag = make_rag()
    rag.build_index()
    result = rag.answer(PIT_QUESTION)

    assert len(server.requests_to(CHAT)) == 1
    assert result["grounded"] is False
    assert result["decline_reason"] == DeclineReason.TRUNCATED == "truncated_model_output"
    assert result["validation"]["rejected_model_output"].startswith(f"[finish_reason={finish_reason}] The pit lane")
    assert result["citations"] == [] and result["confidence"] == 0.0


# ------------------------------------------------------------------ provider failures


def _embed_query(rag: FIARegulationRAG) -> None:
    rag.retrieve(PIT_QUESTION)


def _complete_chat(rag: FIARegulationRAG) -> None:
    rag.answer(PIT_QUESTION)


# Each LangChain client the pipeline creates, and a call that reaches it once the index exists.
# The embeddings client is the one an index build uses too.
EACH_CLIENT = [
    pytest.param(EMBEDDINGS, _embed_query, id="embeddings"),
    pytest.param(CHAT, _complete_chat, id="chat"),
]


def test_rejected_key_on_embeddings_fails_without_leaking_it(make_rag, server):
    leaked_key = "sk-proj-WIRECANARY0123456789abcdef"
    with pytest.raises(ProviderError) as build_error:
        make_rag(api_key=leaked_key).build_index()

    [request] = server.requests_to(EMBEDDINGS)  # a 401 is not retried
    assert request.headers["authorization"] == f"Bearer {leaked_key}"
    assert "HTTP 401" in str(build_error.value) and "rejected the credentials" in str(build_error.value)

    make_rag().build_index()
    rag = make_rag(api_key=leaked_key)  # a key revoked after the index was built: query embedding fails
    with pytest.raises(ProviderError) as query_error:
        rag.retrieve(PIT_QUESTION)
    status = rag.status()
    assert status["provider_status"] == "failing"
    for text in (str(build_error.value), str(query_error.value), status["last_error"]):
        assert leaked_key not in text and "WIRECANARY" not in text


def test_rejected_key_on_chat_fails_without_leaking_it(make_rag, server):
    rag = make_rag()
    rag.build_index()
    server.script(
        CHAT,
        ScriptedResponse(
            status=401,
            error=openai_error(f"Incorrect API key provided: {API_KEY}. The key was revoked.", "invalid_request_error", "invalid_api_key"),
        ),
    )
    with pytest.raises(ProviderError) as caught:
        rag.answer(PIT_QUESTION)

    assert len(server.requests_to(CHAT)) == 1
    assert "rejected the credentials" in str(caught.value) and CHAT_MODEL in str(caught.value)
    assert API_KEY not in str(caught.value) and API_KEY not in rag.status()["last_error"]


def test_rate_limit_once_is_retried_on_both_endpoints(make_rag, server):
    server.chat_reply = replying_with(cite_passage_containing("80km/h"))
    server.script(EMBEDDINGS, rate_limit())
    server.script(CHAT, rate_limit())
    rag = make_rag(max_retries=1)

    report = rag.build_index()
    result = rag.answer(PIT_QUESTION)

    embedding_requests = server.requests_to(EMBEDDINGS)
    assert report.status == "rebuilt"
    # 2 document batches + 1 query, plus the retried first batch with an identical body.
    assert len(embedding_requests) == 4 and embedding_requests[0].body == embedding_requests[1].body
    chat_requests = server.requests_to(CHAT)
    assert len(chat_requests) == 2 and chat_requests[0].body == chat_requests[1].body
    assert result["grounded"] is True
    assert rag.status()["provider_status"] == "ok"


@pytest.mark.parametrize("path, call", EACH_CLIENT)
def test_persistent_rate_limit_is_a_provider_error_after_bounded_retries(make_rag, server, path, call):
    rag = make_rag(max_retries=3)  # differs from the SDK default (2), so the setting is seen to apply
    rag.build_index()
    before = len(server.requests_to(path))
    server.script(path, *[rate_limit() for _ in range(6)])
    with pytest.raises(ProviderError, match="HTTP 429"):
        call(rag)
    assert len(server.requests_to(path)) - before == 4  # the first attempt + max_retries
    assert rag.status()["provider_status"] == "failing"


def test_network_guard_refuses_non_loopback_hosts():
    """The autouse guard is what makes "no request can reach a real provider" hold for this module."""

    with pytest.raises(OSError, match="must not resolve"):
        socket.create_connection(("api.openai.com", 443), timeout=1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError, match="must not open network connections"):
            sock.connect(("192.0.2.1", 443))  # TEST-NET-1: refused before any packet is sent


@pytest.mark.parametrize("path, call", EACH_CLIENT)
def test_provider_silent_for_longer_than_the_timeout_is_a_provider_error(make_rag, server, path, call):
    """A provider that sends nothing for longer than the timeout fails the call.

    The openai SDK hands ``timeout_seconds`` to httpx, which applies it to each
    wait separately (connecting, or waiting for the next bytes of a response).
    It is not a deadline for the whole call, so a response that keeps arriving
    slowly is not cut off; this test covers only a provider that goes silent.
    """

    rag = make_rag(timeout=0.5, max_retries=1)
    rag.build_index()
    before = len(server.requests_to(path))
    server.script(path, ScriptedResponse(delay=10.0), ScriptedResponse(delay=10.0))
    started = time.monotonic()
    with pytest.raises(ProviderError, match="(?i)timed out"):
        call(rag)
    elapsed = time.monotonic() - started

    assert len(server.requests_to(path)) - before == 2  # the timed-out request was retried once
    assert elapsed < 5.0, f"the 0.5 s timeout was not applied (took {elapsed:.1f} s)"
    assert rag.status()["provider_status"] == "failing"
