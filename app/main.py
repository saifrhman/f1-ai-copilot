#!/usr/bin/env python3
"""FastAPI entry point for F1 AI Copilot.

Run from the repository root with ``uvicorn app.main:app``, ``python -m app.main``
or ``python app/main.py``. Settings come from the environment, and from ``.env``
in the project root for anything not already set in the environment.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # support `python app/main.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

if os.getenv("F1_COPILOT_LOAD_DOTENV", "1") != "0":
    load_dotenv(PROJECT_ROOT / ".env")  # never overrides variables that are already set

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError  # noqa: E402

from core_modules.driver_emotion.emotion_classifier import (  # noqa: E402
    TranscriptionError,
    classify_emotion_detailed,
    get_transcriber,
)
from core_modules.driver_emotion.schemas import EmotionRequest, EmotionResponse  # noqa: E402
from core_modules.ghost_car.ghost_car_visualizer import generate_ghost_comparison  # noqa: E402
from core_modules.ghost_car.schemas import GhostCarRequest  # noqa: E402
from core_modules.llm_query.natural_query import MAX_QUERY_CHARS, process_natural_query  # noqa: E402
from core_modules.rule_checker.fia_rag import RAGUnavailableError, get_fia_rag  # noqa: E402
from core_modules.rule_checker.fia_rag.config import RetrievalConfig, resolve_project_path  # noqa: E402
from core_modules.rule_checker.fia_rag.index import close_qdrant_clients  # noqa: E402
from core_modules.rule_checker.fia_rag.retrieval import MAX_QUESTION_CHARS  # noqa: E402
from core_modules.rule_checker.penalty_predictor import predict_penalty  # noqa: E402
from core_modules.rule_checker.schemas import PenaltyRequest  # noqa: E402
from core_modules.setup_optimizer.schemas import SetupRequest  # noqa: E402
from core_modules.setup_optimizer.setup_recommender import recommend_setup_from_inputs  # noqa: E402
from core_modules.strategy_optimizer.schemas import (  # noqa: E402
    StrategyRequest,
    TyreCalibrationRequest,
    calibrate_tyres_response,
    generate_strategy_response,
)

logger = logging.getLogger("f1_copilot.api")

API_VERSION = "1.2.0"
# Request body limits, enforced while the body streams in (Content-Length or chunked).
# Only the routes that accept base64 audio need room for a 120 s clip.
MAX_BODY_BYTES = 40 * 1024 * 1024
ROUTE_BODY_LIMITS = {
    "/api/emotion/classify": MAX_BODY_BYTES,
    "/api/query/natural": MAX_BODY_BYTES,
    "/api/ghost/generate": 8 * 1024 * 1024,
}
DEFAULT_BODY_LIMIT = 1024 * 1024
ARTIFACTS_DIR = resolve_project_path(os.getenv("F1_ARTIFACTS_DIR", "outputs"))
GHOST_ARTIFACTS_DIR = ARTIFACTS_DIR / "ghost"


def _cors_origins() -> List[str]:
    """'*' (the default, also for an empty value) or explicit http(s) origins without a trailing slash."""

    raw = [value.strip() for value in os.getenv("CORS_ORIGINS", "").split(",") if value.strip()]
    if not raw:
        return ["*"]
    if "*" in raw:
        if len(raw) > 1:
            raise RuntimeError("CORS_ORIGINS must be either '*' or a list of explicit origins, not both")
        return raw
    origins = []
    for value in raw:
        if not re.fullmatch(r"https?://[^/\s]+/?", value):
            raise RuntimeError(f"CORS_ORIGINS entry {value!r} must look like https://host[:port]")
        origins.append(value.rstrip("/"))
    return origins


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        GHOST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # reported by /health as artifacts_not_writable; other modules keep working
        logger.warning("Artifacts directory %s is not usable: %s", GHOST_ARTIFACTS_DIR, exc)
    yield
    close_qdrant_clients()


app = FastAPI(
    title="F1 AI Copilot",
    description=(
        "Formula 1 analysis API: evidence-grounded FIA regulation QA (RAG over the official PDFs) plus "
        "explicitly heuristic strategy, setup, telemetry comparison, driver-radio and incident-triage modules."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


class BodyLimitMiddleware:
    """Reject request bodies above the route's limit while they stream in (413), chunked uploads included."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = ROUTE_BODY_LIMITS.get(scope.get("path", ""), DEFAULT_BODY_LIMIT)
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            return await self._reject(send, limit)
        received = 0
        responded = False

        async def guarded_send(message):
            if not responded:
                await send(message)

        async def limited_receive():
            nonlocal received, responded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # Answer 413 now and tell the app the client went away, so it stops reading.
                    if not responded:
                        responded = True
                        await self._reject(send, limit)
                    return {"type": "http.disconnect"}
            return message

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not responded:
                raise

    @staticmethod
    async def _reject(send, limit: int) -> None:
        body = json.dumps({"detail": f"Request body exceeds the {limit}-byte limit for this endpoint"}).encode()
        await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


# Middleware added last runs first: CORS stays outermost so 413/500 responses carry CORS headers.
app.add_middleware(BodyLimitMiddleware)
CORS_ORIGINS = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Only generated ghost-comparison images are public; nothing else in F1_ARTIFACTS_DIR is served.
app.mount("/artifacts/ghost", StaticFiles(directory=GHOST_ARTIFACTS_DIR, check_dir=False), name="ghost-artifacts")


def _public_message(exc: BaseException) -> str:
    """Error text for API clients, without absolute server paths."""

    return str(exc).replace(str(PROJECT_ROOT), "<project>")


def _safe_text(value: str, max_chars: int = 200) -> str:
    text = value if len(value) <= max_chars else value[:max_chars] + f"... ({len(value)} characters)"
    return text.encode("utf-8", "backslashreplace").decode("utf-8")  # lone surrogates cannot be rendered


def _summarise_input(value: Any) -> Any:
    """Bounded echo of an invalid value: never whole objects, never non-finite floats or lone surrogates."""

    if isinstance(value, dict):
        return f"<object with {len(value)} keys>"
    if isinstance(value, (list, tuple)):
        return f"<array with {len(value)} items>"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str):
        return _safe_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe_text(repr(value))


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default handler re-encodes every error with its full input: a NaN literal or a lone
    # surrogate then breaks rendering (HTTP 500), and a large invalid body is echoed back many times.
    errors = exc.errors()
    detail = []
    for error in errors[:20]:
        detail.append(
            {
                "type": _safe_text(str(error.get("type", ""))),
                "loc": [item if isinstance(item, int) else _safe_text(str(item), 80) for item in list(error.get("loc", ()))[:10]],
                "msg": _safe_text(str(error.get("msg", ""))),
                "input": _summarise_input(error.get("input")),
            }
        )
    content: Dict[str, Any] = {"detail": detail}
    if len(errors) > 20:
        content["more_errors"] = len(errors) - 20
    return JSONResponse(status_code=422, content=content)


@app.exception_handler(RAGUnavailableError)
async def rag_unavailable_handler(_: Request, exc: RAGUnavailableError) -> JSONResponse:
    logger.warning("FIA RAG unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": f"FIA RAG is unavailable: {_public_message(exc)}", "error_type": type(exc).__name__},
    )


@app.exception_handler(TranscriptionError)
async def transcription_error_handler(_: Request, exc: TranscriptionError) -> JSONResponse:
    # Whisper was requested and available but failed (model download, ffmpeg): the audio itself was valid.
    return JSONResponse(status_code=503, content={"detail": f"Transcription failed: {_public_message(exc)}"})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error_type": type(exc).__name__})


def _unprocessable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=_public_message(exc))


# ------------------------------------------------------------------------ request / response models


class FIAQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS, description="Natural-language question")
    top_k: Optional[StrictInt] = Field(
        None, ge=1, le=RetrievalConfig.MAX_TOP_K, description="Retrieval depth for this request (default FIA_RAG_TOP_K)"
    )


class FIARetrieveRequest(FIAQueryRequest):
    min_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, strict=True, allow_inf_nan=False,
        description="Similarity threshold for this request (default FIA_RAG_MIN_SCORE)",
    )


class RetrievedPassageOut(BaseModel):
    label: str = Field(description="Citation label used in the answer, e.g. S1")
    cited: bool = Field(description="Whether the validated answer cites this passage")
    chunk_id: str
    text: str
    score: float = Field(
        description="Cosine similarity between question and passage (higher is more similar); 0 for definitions"
    )
    source: str = Field(description="PDF file name")
    page: Optional[int] = Field(description="1-based page in the PDF")
    page_label: Optional[str] = Field(description="Printed page label, e.g. B10")
    section: Optional[str] = Field(description="Regulation section letter A-F")
    source_url: Optional[str] = Field(description="Official FIA URL of the PDF")
    nearest_rule: Optional[str] = Field(description="Last numbered rule heading at or before the passage start")
    rule_ids: List[str] = Field(description="Rule identifiers mentioned in the passage")
    kind: str = Field(
        description="regulation (retrieved chunk) | definition (official definition of a term the chunks use)"
    )
    defined_term: Optional[str] = Field(description="The defined term, for definition passages")


class CitationSpanOut(BaseModel):
    start: int = Field(description="Offset in `answer` (Unicode code points) where the citation is shown")
    end: int = Field(description="Offset in `answer` just after it")
    labels: List[str] = Field(description="Labels it names, ranges expanded, e.g. [S1-S3] -> S1, S2, S3")


class FIAAnswerResponse(BaseModel):
    question: str
    answer: str = Field(description="Validated answer with [S#] labels, or the standard decline message")
    grounded: bool = Field(description="True only when the answer passed citation and rule-number validation")
    status: str = Field(description="answered | declined")
    decline_reason: Optional[str] = Field(
        description=(
            "no_evidence_above_threshold | model_declined | empty_model_output | missing_citation | "
            "invalid_citation | unsupported_rule_reference | unsupported_number | uncited_claim | "
            "truncated_model_output | unverified_claim"
        )
    )
    confidence: float = Field(
        description="Best similarity among cited regulation passages (evidence-strength proxy, not a probability)"
    )
    citations: List[str] = Field(description="Labels cited by the answer; each maps to a retrieved_passages entry")
    citation_spans: List[CitationSpanOut] = Field(
        description=(
            "Each citation as written in `answer`, in text order: the whole token ([S1], (source S2)), or only the "
            "label of a prose citation (the S2 of 'source S2'). Empty for a decline"
        )
    )
    referenced_rules: List[str] = Field(description="Rule identifiers in the answer, all verified against the evidence")
    retrieved_passages: List[RetrievedPassageOut] = Field(
        description="Evidence given to the model: passages above the threshold, then definitions of terms they use"
    )
    top_retrieval_score: float
    retrieval: Dict[str, Any] = Field(
        description=(
            "top_k, min_score, passages_above_threshold, definitions_added, below_threshold (the passages under "
            "min_score) and duplicates_removed"
        )
    )
    validation: Dict[str, Any] = Field(
        description=(
            "invalid_citations, unsupported_rules, uncited_claims, unsupported_numbers, unverified_claims, "
            "claim_verification (null when FIA_RAG_VERIFY_CLAIMS is off or the answer was declined before "
            "verification), rejected_model_output (the model text, only when it failed validation) and "
            "finish_reason (what stopped a truncated_model_output reply: length, max_tokens or content_filter; "
            "null otherwise)"
        )
    )
    models: Dict[str, Optional[str]]
    source: str


class NaturalQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    context: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Evidence for the routed module: telemetry (performance), the strategy request fields (strategy), "
            "driver_preferences/track_profile/weather (setup), audio_file (emotion). Never inferred when absent."
        ),
    )


# ------------------------------------------------------------------------ health


@lru_cache(maxsize=1)
def _transcription_availability() -> Tuple[bool, Optional[str]]:
    # Importing Whisper loads PyTorch, so the check runs once per process.
    return get_transcriber().availability()


def _fia_health() -> Tuple[str, Dict[str, Any]]:
    try:
        status = get_fia_rag().status()
    except RAGUnavailableError as exc:  # invalid FIA_RAG_* settings
        return "misconfigured", {"ready": False, "problems": [_public_message(exc)]}
    if status["ready"]:
        return "ready", status
    if not status["documents"] or any("OPENAI_API_KEY" in problem for problem in status["problems"]):
        return "not_configured", status
    index_status = status["index"]["status"]
    if index_status in {"missing", "empty", "stale", "incomplete"}:
        return f"index_{index_status}", status
    if status.get("provider_status") == "failing":
        return "provider_failing", status
    return "unavailable", status


def _artifacts_writable() -> bool:
    target = next((path for path in (GHOST_ARTIFACTS_DIR, ARTIFACTS_DIR, ARTIFACTS_DIR.parent) if path.exists()), None)
    return target is not None and os.access(target, os.W_OK)


@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "F1 AI Copilot API", "version": API_VERSION, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Readiness of every component. HTTP 200 with status "degraded" when a component is not ready."""

    fia_state, fia_status = _fia_health()
    transcription_ok, transcription_reason = _transcription_availability()
    artifacts_ok = _artifacts_writable()
    modules = {
        "fia_rag": fia_state,
        "strategy": "ready",
        "setup": "ready",
        "penalty_triage": "ready",
        "natural_query": "ready",
        "emotion": "ready",
        "emotion_transcription": "ready" if transcription_ok else "unavailable",
        "ghost": "ready" if artifacts_ok else "artifacts_not_writable",
    }
    degraded = fia_state != "ready" or not artifacts_ok
    return {
        "status": "degraded" if degraded else "healthy",
        "version": API_VERSION,
        "modules": modules,
        "details": {
            "fia_rag": {
                key: fia_status.get(key)
                for key in ("ready", "problems", "index", "provider_status", "last_error", "last_error_at", "embedding_cache_problem")
            },
            "emotion_transcription": transcription_reason,
            "artifacts_dir_writable": artifacts_ok,
        },
    }


# ------------------------------------------------------------------------ FIA regulation RAG


@app.get("/api/fia/status")
def fia_rag_status() -> Dict[str, Any]:
    state, status = _fia_health()
    return {"state": state, **status}


@app.post("/api/fia/query", response_model=FIAAnswerResponse)
def query_fia_rules(request: FIAQueryRequest) -> Dict[str, Any]:
    """Retrieve evidence, generate a grounded answer and validate its citations (503 when the RAG is not ready)."""

    try:
        return get_fia_rag().answer(request.question, top_k=request.top_k)
    except ValueError as exc:
        raise _unprocessable(exc) from exc


@app.post("/api/fia/retrieve")
def retrieve_fia_passages(request: FIARetrieveRequest) -> Dict[str, Any]:
    """Retrieval only: ranked passages with scores, without calling the answer model."""

    try:
        return get_fia_rag().retrieve(request.question, top_k=request.top_k, min_score=request.min_score).to_dict()
    except ValueError as exc:
        raise _unprocessable(exc) from exc


# ------------------------------------------------------------------------ heuristic modules


@app.post("/api/strategy/generate")
def generate_race_strategy(request: StrategyRequest) -> Dict[str, Any]:
    try:
        return generate_strategy_response(request)
    except ValueError as exc:
        raise _unprocessable(exc) from exc


@app.post("/api/strategy/calibrate-tyres")
def calibrate_tyres(request: TyreCalibrationRequest) -> Dict[str, Any]:
    """Estimate the strategy engine's tyre parameters (``tire_data``) from observed lap times."""

    try:
        return calibrate_tyres_response(request)
    except ValueError as exc:
        raise _unprocessable(exc) from exc


@app.post("/api/setup/recommend")
def recommend_car_setup(request: SetupRequest) -> Dict[str, Any]:
    try:
        return recommend_setup_from_inputs(request.to_engine_inputs())
    except ValueError as exc:
        raise _unprocessable(exc) from exc


@app.post("/api/ghost/generate")
def generate_ghost_car(request: GhostCarRequest) -> Dict[str, Any]:
    try:
        result = generate_ghost_comparison(request, output_dir=GHOST_ARTIFACTS_DIR)
    except ValueError as exc:
        raise _unprocessable(exc) from exc
    except OSError as exc:
        logger.error("Ghost artifact could not be written: %s", exc)
        raise HTTPException(status_code=503, detail="Artifact storage is not writable (see F1_ARTIFACTS_DIR)") from exc
    result["visualization_url"] = f"/artifacts/ghost/{result['artifact_path']}"
    return result


@app.post("/api/emotion/classify", response_model=EmotionResponse)
def classify_driver_emotion(request: EmotionRequest) -> Dict[str, Any]:
    try:
        # Only base64/data-URI audio is accepted over HTTP; server paths are never opened.
        return classify_emotion_detailed(request.audio_file, transcribe=request.transcribe, allow_local_paths=False)
    except ValueError as exc:
        raise _unprocessable(exc) from exc


@app.post("/api/query/natural")
def process_query(request: NaturalQueryRequest) -> Dict[str, Any]:
    try:
        return process_natural_query(request.query, request.context)
    except ValidationError as exc:
        # The strategy and setup handlers validate the context with their endpoints' request models:
        # report those errors like every other 422 (a capped list, located under body.context).
        raise RequestValidationError(
            [{**error, "loc": ("body", "context", *error["loc"])} for error in exc.errors()]
        ) from exc
    except ValueError as exc:
        raise _unprocessable(exc) from exc


@app.post("/api/penalty/predict")
def predict_incident_penalty(request: PenaltyRequest) -> Dict[str, Any]:
    """Heuristic incident triage; not an FIA steward-decision predictor."""

    try:
        return predict_penalty(**request.to_predictor_kwargs())
    except ValueError as exc:
        raise _unprocessable(exc) from exc


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
