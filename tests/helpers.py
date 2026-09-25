"""Test helpers: real PDF files, a deterministic lexical embedder and a scripted chat model.

Only the two network services (embedding API, chat API) are replaced in tests.
PDF parsing, chunking, Qdrant indexing/search and answer validation are real.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

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
