"""Test helpers: real PDF files, a deterministic lexical embedder, a scripted chat model, and the
generated regulation corpora and pipelines the RAG, API and UI tests share.

Only the two network services (embedding API, chat API) are replaced in tests.
PDF parsing, chunking, Qdrant indexing/search and answer validation are real.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient

from core_modules.rule_checker.fia_rag import (
    ChunkingConfig,
    EmbeddingConfig,
    FIARegulationRAG,
    GenerationConfig,
    QdrantConfig,
    RAGSettings,
    RetrievalConfig,
)

PageSpec = Union[str, Sequence[str], None]


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(text: str, width: int = 90) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words, current = paragraph.split(), ""
        for word in words:
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        lines.append(current)
    return lines


def write_pdf(path: Path, pages: Sequence[PageSpec]) -> Path:
    """Write a valid PDF with one Helvetica text page per entry (``None``/``""`` = blank page)."""

    objects: List[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    pages_id = add(b"")  # placeholder, filled in below
    page_ids = []
    for spec in pages:
        text = spec if isinstance(spec, str) or spec is None else "\n".join(spec)
        commands = ["BT", "/F1 9 Tf", "11 TL", "40 800 Td"]
        for line in _wrap(text or ""):
            commands.append(f"({_pdf_escape(line)}) Tj T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("cp1252")  # WinAnsiEncoding
        content_id = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 595 842] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (pages_id, font_id, content_id)
            )
        )
    kids = " ".join(f"{pid} 0 R" for pid in page_ids).encode()
    objects[pages_id - 1] = b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % len(page_ids)
    catalog_id = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog_id, xref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


_STOPWORDS = set(
    "a an the of to in on for and or is are be by with at as it its this that from what which when "
    "does do can may must shall will any all their there than into per if not no".split()
)


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: cosine similarity reflects word overlap."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.document_calls = 0
        self.query_calls = 0
        self.documents_embedded = 0

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in _STOPWORDS:
                continue
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [x / norm for x in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_calls += 1
        self.documents_embedded += len(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        self.query_calls += 1
        return self._embed(text)


class ScriptedChatModel:
    """Chat model stand-in: returns a fixed string or ``responder(messages)`` and records prompts."""

    def __init__(self, reply: Union[str, Callable[[list], str]] = "INSUFFICIENT_EVIDENCE", finish_reason: str = "stop"):
        self.reply = reply
        self.finish_reason = finish_reason
        self.calls: List[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        text = self.reply(messages) if callable(self.reply) else self.reply
        return AIMessage(content=text, response_metadata={"finish_reason": self.finish_reason})

    @property
    def last_prompt(self) -> Optional[str]:
        if not self.calls:
            return None
        return "\n".join(str(message.content) for message in self.calls[-1])


FIA_STYLE_HEADER = (
    "SECTION B: SPORTING REGULATIONS\n0\nB\n"
    "B{page} 2026 Formula 1: Sporting Regulations ©2026 Fédération Internationale de l'Automobile "
    "05 August 2026 Issue 08"
)


def fia_page(page: int, body: str) -> str:
    return FIA_STYLE_HEADER.format(page=page) + "\n" + body


# ------------------------------------------------------------------ regulation corpus and pipeline

PIT_LANE = (
    "B1.6 Pit Lane Speed\nB1.6.3 Driving in the Pit Entry Road, Pit Lane and Pit Exit Road "
    "a. A speed limit of 80km/h will be imposed in the pit lane during all sessions."
)
UNSAFE_RELEASE = (
    "B4.2 Unsafe Release\nB4.2.1 A car must not be released from its pit stop position in an unsafe "
    "condition. Competitors are responsible for releasing cars only when it is safe."
)
FUEL_FLOW = "C5.4 Fuel Flow\nC5.4.2 The fuel mass flow must not exceed one hundred kilograms per hour above 10500 rpm."
REAR_WING = "C3.9 Rear Wing\nC3.9.1 The rear wing flap position may be adjusted by the driver only when the adjustable wing is enabled."


def write_regulation_corpus(folder: Path) -> Path:
    """Two regulation PDFs: Section B (pit lane, a blank page, unsafe release) and Section C (fuel flow, rear wing)."""

    write_pdf(folder / "section_b_sporting.pdf", [fia_page(1, PIT_LANE), None, fia_page(3, UNSAFE_RELEASE)])
    write_pdf(folder / "section_c_technical.pdf", [FUEL_FLOW, REAR_WING])
    return folder


def cite_passage_containing(needle, template="{claim} [{label}]."):
    """A well-behaved model: cites the excerpt that actually contains the fact."""

    def responder(messages):
        user = messages[1].content
        for match in re.finditer(r'<excerpt label="(S\d+)"[^>]*>\n(.*?)\n</excerpt>', user, re.DOTALL):
            if needle in match.group(2):
                return template.format(claim=f"The regulations state: {needle}", label=match.group(1))
        return "INSUFFICIENT_EVIDENCE"

    return responder


def rag_settings(docs_path: Path, **overrides: Any) -> RAGSettings:
    """The test pipelines' settings: 400-character chunks, hashing embeddings, the top 4 passages above 0.2.

    ``overrides`` replace whole settings objects (``retrieval=RetrievalConfig(...)``, ``qdrant=...``).
    """

    values: Dict[str, Any] = {
        "docs_path": docs_path,
        "chunking": ChunkingConfig(chunk_size=400, chunk_overlap=40),
        "embedding": EmbeddingConfig(model="hashing-test-512", batch_size=8),
        "retrieval": RetrievalConfig(top_k=4, min_score=0.2),
        "qdrant": QdrantConfig(collection="test"),
    }
    return RAGSettings(**{**values, **overrides})


def build_test_rag(docs_path: Path, reply=None) -> Tuple[FIARegulationRAG, ScriptedChatModel, QdrantClient]:
    """A real pipeline over ``write_regulation_corpus``; the index is NOT built yet (call ``build_index``).

    The scripted model answers pit-lane questions by citing the passage containing "80km/h" unless
    ``reply`` (a string or ``messages -> str``) is given. The caller closes the returned Qdrant client.
    """

    llm = ScriptedChatModel(reply if reply is not None else cite_passage_containing("80km/h"))
    qdrant = QdrantClient(":memory:")
    rag = FIARegulationRAG(
        rag_settings(write_regulation_corpus(docs_path)), embeddings=HashingEmbeddings(), llm=llm, qdrant_client=qdrant
    )
    return rag, llm, qdrant


# ------------------------------------------------------------------ definitions corpus and pipeline

PIT_PENALTY = (
    "B1.6 Pit Lane Speed\nB1.6.4 Speeding in the pit lane during a TTCS will be penalised with a drive through "
    "penalty. Speeding in the pit lane during an LTCS will be penalised with a fine."
)
DEFINITIONS = (
    "APPENDIX B1 DEFINITIONS\n"
    "“Total Time Classified Session” (or “TTCS”) is any track running session during which the "
    "classification is determined by the total time taken. Total Time Classified Sessions include the Sprint "
    "session and the Race session.\n"
    "“Lap Time Classified Session” (or “LTCS”) is any session classified by the fastest lap "
    "time of each driver, such as Qualifying.\n"
    "“Official” means any of the persons listed in the Code.\n"
    "“Cost Cap” has the meaning set out in Article D4.1.2."
)
DEFINITIONS_QUESTION = "What happens when speeding in the pit lane?"


def write_definitions_corpus(folder: Path) -> Path:
    """One Section B PDF: a pit-lane penalty that uses TTCS and LTCS (page 1) and their definitions (page 85)."""

    write_pdf(folder / "section_b_sporting.pdf", [fia_page(1, PIT_PENALTY), fia_page(85, DEFINITIONS)])
    return folder


def make_definitions_rag(docs, tmp_path, qdrant, llm=None, embeddings=None, max_definitions=3, verify=False):
    """A pipeline over ``write_definitions_corpus`` that retrieves one passage and adds its definitions."""

    settings = RAGSettings(
        docs_path=docs,
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=0),
        embedding=EmbeddingConfig(model="hashing-test-512"),
        retrieval=RetrievalConfig(top_k=1, min_score=0.1, max_definitions=max_definitions),
        generation=GenerationConfig(verify_claims=verify),
        qdrant=QdrantConfig(collection="fia_test", path=tmp_path / "qdrant"),
    )
    return FIARegulationRAG(settings, embeddings=embeddings or HashingEmbeddings(), llm=llm, qdrant_client=qdrant)
