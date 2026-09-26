"""Ingestion: validate regulation PDFs, extract page text and split it into chunks.

This stage depends only on :class:`ChunkingConfig`; it knows nothing about
embeddings, Qdrant or answer generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from core_modules.rule_checker.fia_files import MANIFEST_NAME, fia_section

from .config import ChunkingConfig, display_path
from .errors import DocumentError, RAGConfigurationError
from .rules import extract_headings, extract_rule_ids

logger = logging.getLogger(__name__)

# Bumped whenever text extraction/normalisation/chunk metadata changes, so an
# index built by an older ingestion version is detected as stale.
INGESTION_VERSION = "4"
_CHUNK_NAMESPACE = uuid.UUID("5b0f7a52-6a1e-4a4e-9a55-f1a0c0de2026")
_SPLITTER_SEPARATORS = ["\n\n", "\n", ". ", "; ", " ", ""]
# A page with fewer letters and digits than this (after the header is removed) counts as blank.
_MIN_PAGE_CHARS = 20


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    filename: str
    sha256: str
    size_bytes: int
    section: Optional[str] = None
    source_url: Optional[str] = None


@dataclass(frozen=True)
class DiscoveryResult:
    documents: List[SourceDocument]
    warnings: List[str] = field(default_factory=list)


@dataclass
class DocumentReport:
    filename: str
    pages: int
    pages_with_text: int
    empty_pages: List[int]
    contents_pages: List[int] = field(default_factory=list)
    chunks: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "filename": self.filename,
            "pages": self.pages,
            "pages_with_text": self.pages_with_text,
            "empty_pages": self.empty_pages,
            "contents_pages_skipped": self.contents_pages,
            "chunks": self.chunks,
        }


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str  # verbatim extracted text, shown to the model and returned as evidence
    metadata: Dict[str, object]
    embedding_text: str  # text + short document/article context, used only for the vector


# --------------------------------------------------------------------------- discovery


_hash_cache: Dict[Tuple[str, int, int], str] = {}


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _hash_cache.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    _hash_cache[key] = value
    return value


def _load_manifest(docs_path: Path) -> Dict[str, dict]:
    manifest_path = docs_path / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["documents"]
        return {str(entry["filename"]): entry for entry in entries if "filename" in entry}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DocumentError(f"{manifest_path} is not a valid FIA download manifest: {exc}") from exc


def discover_documents(docs_path: Path) -> DiscoveryResult:
    """List the PDFs to index, validating each file before any parsing happens.

    Every ``*.pdf`` in ``docs_path`` is used. When the downloader's manifest
    lists a file, its SHA-256 must still match (the file was not modified or
    truncated after download) and its section/source URL become metadata.
    Byte-identical duplicates are indexed once. With a manifest, a listed file
    that is missing, or a second issue of the same FIA section that the manifest
    does not list, is an error: indexing it would mix regulation issues.
    """

    if not docs_path.exists():
        raise RAGConfigurationError(
            f"FIA document directory {display_path(docs_path)} does not exist. "
            "Run `python scripts/fetch_fia_regulations.py` to download the regulations."
        )
    if not docs_path.is_dir():
        raise RAGConfigurationError(f"FIA_DOCS_PATH {display_path(docs_path)} is not a directory")
    try:
        pdf_paths = sorted(p for p in docs_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    except OSError as exc:
        raise RAGConfigurationError(f"FIA document directory {display_path(docs_path)} cannot be read: {exc.strerror}") from exc
    if not pdf_paths:
        raise RAGConfigurationError(
            f"No FIA regulation PDFs found in {display_path(docs_path)}. "
            "Run `python scripts/fetch_fia_regulations.py` to download the regulations."
        )

    manifest = _load_manifest(docs_path)
    if manifest:
        missing = sorted(name for name in manifest if not (docs_path / name).is_file())
        if missing:
            raise DocumentError(
                f"{MANIFEST_NAME} lists files that are missing: {', '.join(missing)}. Re-run the downloader."
            )
        listed_sections = {fia_section(name): name for name in manifest if fia_section(name)}
        for path in pdf_paths:
            section = fia_section(path.name)
            if path.name not in manifest and section in listed_sections:
                raise DocumentError(
                    f"{path.name} is another issue of FIA {section[0]} Section {section[1]} than the one in "
                    f"{MANIFEST_NAME} ({listed_sections[section]}); remove the superseded file or re-run the downloader."
                )

    documents: List[SourceDocument] = []
    warnings: List[str] = []
    seen_hashes: Dict[str, str] = {}
    for path in pdf_paths:
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                head = handle.read(1024)
            sha256 = _sha256_file(path) if size else ""
        except OSError as exc:
            raise DocumentError(f"{path.name} cannot be read: {exc.strerror}") from exc
        if size == 0:
            raise DocumentError(f"{path.name} is empty (0 bytes)")
        if b"%PDF-" not in head:
            snippet = head[:40].decode("latin-1", errors="replace").strip()
            raise DocumentError(f"{path.name} is not a PDF file (starts with {snippet!r})")
        entry = manifest.get(path.name, {})
        expected = entry.get("sha256")
        if expected and expected != sha256:
            raise DocumentError(
                f"{path.name} does not match the SHA-256 recorded in {MANIFEST_NAME}; "
                "it was modified or truncated after download. Re-run the downloader."
            )
        if sha256 in seen_hashes:
            warnings.append(f"{path.name} is byte-identical to {seen_hashes[sha256]} and was not indexed twice")
            continue
        seen_hashes[sha256] = path.name
        documents.append(
            SourceDocument(
                path=path,
                filename=path.name,
                sha256=sha256,
                size_bytes=size,
                section=entry.get("section"),
                source_url=entry.get("source_url"),
            )
        )
    return DiscoveryResult(documents=documents, warnings=warnings)


# --------------------------------------------------------------------------- text extraction

# Page furniture printed on every page of the 2026 FIA regulations, e.g.
# "SECTION B: SPORTING REGULATIONS 0 B B21 2026 Formula 1: Sporting Regulations
#  ©2026 Fédération Internationale de l'Automobile 05 August 2026 Issue 08".
_HEADER_START = re.compile(r"^\s*SECTION\s+([A-F])\s*:", re.IGNORECASE)
_HEADER_END_MARKERS = re.compile(
    r"Issue\s*\d(?:\s?\d)?(?!\d)|F[ée]d[ée]ration\s+Internationale\s+de\s+l\s*['’]\s*Automobile",
    re.IGNORECASE,
)
_HEADER_WINDOW = 450
_HEADER_DECORATION = re.compile(r"^\s*0\s+(?:[A-F]\s+)?")


def split_page_header(text: str) -> Tuple[Optional[str], str]:
    """Return ``(page_label, body)``; text without the FIA page header is returned unchanged."""

    match = _HEADER_START.match(text)
    if not match:
        return None, text
    window = text[:_HEADER_WINDOW]
    ends = [m.end() for m in _HEADER_END_MARKERS.finditer(window)]
    if not ends:
        return None, text
    cut = max(ends)
    header, body = text[:cut], text[cut:]
    body = _HEADER_DECORATION.sub("", body, count=1)
    letter = match.group(1).upper()
    label_match = re.search(rf"\b{letter}\s?(\d{{1,3}})\b", header[match.end():])
    page_label = f"{letter}{label_match.group(1)}" if label_match else None
    return page_label, body


# The FIA PDFs encode some "ff"/"ffi" ligatures with glyphs that extract as a
# backtick or a left single quote inside words ("o`icials", "su‘iciently").
_BROKEN_LIGATURE = re.compile(r"(?<=[A-Za-z])[`‘](?=[a-z])")
# Table-of-contents line: "B3.5 Pre-Sprint & Pre-Race Parc Fermé 29" / "ARTICLE C4: MASS 59".
_CONTENTS_ENTRY = re.compile(
    r"(?:(?:ARTICLE|APPENDIX)\s+[A-F]\d{1,2}\s*:?|[A-F]\d{1,2}(?:\.\d{1,3})+)\s+[^\n]{2,90}?\s\d{1,3}(?=\s|$)"
)


def looks_like_contents_page(text: str) -> bool:
    """Contents pages list many "rule-id title page-number" entries and no prose.

    They are keyword-dense but carry no regulatory content, so indexing them
    would let retrieval return a page reference instead of the rule itself.
    """

    entries = len(_CONTENTS_ENTRY.findall(text))
    words = max(1, len(text.split()))
    sentences = len(re.findall(r"[a-z]{3,}\.(?=\s|$)", text))
    if entries >= 8 and entries / words > 0.04 and sentences <= 2:
        return True
    # Short continuation of a contents list: every line is "title ... page-number".
    lines = [line for line in text.splitlines() if line.strip()]
    numbered = sum(1 for line in lines if re.search(r"\s\d{1,3}\s*$", line))
    return entries >= 1 and len(lines) >= 3 and numbered / len(lines) >= 0.8 and sentences == 0


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _BROKEN_LIGATURE.sub("ff", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _alnum_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


@contextmanager
def _quiet_pypdf():
    """pypdf logs benign xref repairs ("Ignoring wrong pointing object") for the FIA files."""

    pypdf_logger = logging.getLogger("pypdf")
    previous = pypdf_logger.level
    pypdf_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        pypdf_logger.setLevel(previous)


def load_pages(document: SourceDocument) -> Tuple[List[Document], DocumentReport]:
    """Extract one LangChain ``Document`` per page that contains text.

    Unreadable PDFs and per-page extraction failures raise :class:`DocumentError`
    instead of silently producing a partial index. Pages without text are
    recorded in the report (they are legitimately blank in some documents),
    but a document without any extractable text is rejected.
    """

    with _quiet_pypdf():
        return _load_pages(document)


def _load_pages(document: SourceDocument) -> Tuple[List[Document], DocumentReport]:
    try:
        reader = PdfReader(str(document.path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError(f"{document.filename} is encrypted and cannot be read")
        page_count = len(reader.pages)
    except DocumentError:
        raise
    except (PdfReadError, OSError, ValueError, KeyError, TypeError) as exc:
        raise DocumentError(f"{document.filename} could not be parsed as a PDF: {exc}") from exc
    if page_count == 0:
        raise DocumentError(f"{document.filename} contains no pages")

    pages: List[Document] = []
    empty_pages: List[int] = []
    contents_pages: List[int] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # pypdf raises many exception types for damaged content streams
            raise DocumentError(f"{document.filename}: text extraction failed on page {number}: {exc}") from exc
        page_label, body = split_page_header(raw)
        text = normalise_text(body)
        if _alnum_count(text) < _MIN_PAGE_CHARS:
            empty_pages.append(number)
            continue
        if looks_like_contents_page(text):
            contents_pages.append(number)
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": document.filename,
                    "document_sha256": document.sha256,
                    "section": document.section,
                    "source_url": document.source_url,
                    "page": number,  # 1-based physical page number
                    "page_label": page_label,
                },
            )
        )

    if not pages:
        raise DocumentError(
            f"{document.filename}: none of its {page_count} pages contain extractable text "
            "(image-only/scanned PDFs need OCR before indexing)"
        )
    report = DocumentReport(
        filename=document.filename,
        pages=page_count,
        pages_with_text=len(pages),
        empty_pages=empty_pages,
        contents_pages=contents_pages,
    )
    return pages, report


# --------------------------------------------------------------------------- chunking


def make_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=_SPLITTER_SEPARATORS,
        keep_separator="end",  # sentence punctuation stays with its sentence
        add_start_index=True,
    )


def document_title(filename: str) -> str:
    """Readable title from an FIA file name, e.g. "fia 2026 f1 regulations section b sporting"."""

    stem = Path(filename).stem.lower()
    stem = re.split(r"[_ -]+iss(?:ue)?[_ -]*\d", stem)[0]
    return re.sub(r"[_\s-]+", " ", stem).strip()


def chunk_pages(pages: Sequence[Document], config: ChunkingConfig) -> List[Chunk]:
    """Split pages (in document order) into chunks carrying source/page/article metadata.

    ``nearest_rule`` is the last numbered heading at or before the start of the
    chunk, carried across page boundaries within the same document, so a chunk
    that starts mid-article still records which article it belongs to.
    """

    splitter = make_splitter(config)
    chunks: List[Chunk] = []
    current_rule: Dict[str, Optional[str]] = {}
    for page in pages:
        source = str(page.metadata["source"])
        headings = extract_headings(page.page_content)
        carried = current_rule.get(source)
        parts = splitter.split_documents([page])
        chunk_index = 0
        for part in parts:
            text = part.page_content.strip()
            if _alnum_count(text) < config.min_chunk_chars:
                continue
            start = int(part.metadata.get("start_index", 0))
            nearest = carried
            for position, rule in headings:
                if position <= start:
                    nearest = rule
                else:
                    break
            if nearest is None and headings and headings[0][0] < start + len(text):
                nearest = headings[0][1]
            metadata = dict(page.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "start_index": start,
                    "nearest_rule": nearest,
                    "rule_ids": extract_rule_ids(text),
                }
            )
            key = f"{metadata['document_sha256']}:{metadata['page']}:{chunk_index}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            chunk_id = str(uuid.uuid5(_CHUNK_NAMESPACE, key))
            metadata["chunk_id"] = chunk_id
            context = document_title(source) + (f" | {nearest}" if nearest else "")
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata, embedding_text=f"{context}\n{text}"))
            chunk_index += 1
        if headings:
            current_rule[source] = headings[-1][1]
    return chunks


def load_chunks_and_pages(
    documents: Sequence[SourceDocument], config: ChunkingConfig
) -> Tuple[List[Chunk], List[DocumentReport], List[Document]]:
    all_chunks: List[Chunk] = []
    all_pages: List[Document] = []
    reports: List[DocumentReport] = []
    for document in documents:
        pages, report = load_pages(document)
        chunks = chunk_pages(pages, config)
        if not chunks:
            raise DocumentError(f"{document.filename} produced no chunks with meaningful text")
        report.chunks = len(chunks)
        all_chunks.extend(chunks)
        all_pages.extend(pages)
        reports.append(report)
        logger.info(
            "Parsed %s: %s pages (%s with text), %s chunks",
            document.filename,
            report.pages,
            report.pages_with_text,
            report.chunks,
        )
    return all_chunks, reports, all_pages
