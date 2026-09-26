#!/usr/bin/env python3
"""Download the current official FIA Formula 1 regulation PDFs for local RAG use.

The FIA website terms reserve copyright in FIA Publications and limit copying to
private/non-commercial use unless FIA gives prior written consent. For that
reason this script downloads files into the git-ignored data/fia_docs directory
and the project does not redistribute the PDFs.

The FIA category page lists every historical issue of each section. For each
requested section the script selects the newest issue (by the date and issue
number in the file name), validates every download, and only then replaces the
previous set: superseded files recorded in the previous manifest are removed so
the RAG index never mixes old and new issues. Files not created by this script
are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Standard library only: the discovery CI job installs just requests and pypdf.
from core_modules.rule_checker.fia_files import MANIFEST_NAME, fia_issue, fia_section  # noqa: E402

DEFAULT_CATEGORY_URL = "https://www.fia.com/regulation/category/2182"
DEFAULT_SECTIONS = ("A", "B", "C", "D", "E", "F")
USER_AGENT = "f1-ai-copilot/1.2 (local FIA regulation downloader)"
MAX_PDF_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class Candidate:
    section: str
    url: str
    issue: Optional[int]
    date: Optional[str]
    page_order: int

    @property
    def rank(self) -> Tuple[str, int, int]:
        # Newest date first, then highest issue, then earliest position on the page.
        return (self.date or "", self.issue if self.issue is not None else -1, -self.page_order)


def _filename_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "fia_regulations.pdf"


def _section_from_url(url: str, year: int) -> Optional[str]:
    """Section letter of an official FIA regulation file of ``year`` (see ``fia_files.fia_section``)."""

    found = fia_section(html.unescape(unquote(Path(urlparse(url).path).name)), year)
    return found[1] if found else None


def _issue_and_date(url: str) -> Tuple[Optional[int], Optional[str]]:
    name = unquote(urlparse(url).path).lower()
    date = re.findall(r"(20\d\d-\d\d-\d\d)", name)
    return fia_issue(name), date[-1] if date else None


def discover_candidates(category_html: str, category_url: str, year: int) -> List[Candidate]:
    """Every linked regulation PDF for ``year`` that belongs to one of sections A-F."""

    candidates: List[Candidate] = []
    seen = set()
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', category_html, flags=re.IGNORECASE)
    for order, href in enumerate(hrefs):
        value = html.unescape(href).strip()
        absolute = urljoin(category_url, value)
        if absolute in seen or not urlparse(absolute).path.lower().endswith(".pdf"):
            continue
        section = _section_from_url(absolute, year)
        if section not in DEFAULT_SECTIONS:
            continue
        seen.add(absolute)
        issue, date = _issue_and_date(absolute)
        candidates.append(Candidate(section, absolute, issue, date, order))
    return candidates


def latest_candidates(candidates: Iterable[Candidate]) -> Dict[str, Candidate]:
    """The newest issue among ``candidates`` for each section (see ``Candidate.rank``)."""

    latest: Dict[str, Candidate] = {}
    for candidate in candidates:
        current = latest.get(candidate.section)
        if current is None or candidate.rank > current.rank:
            latest[candidate.section] = candidate
    return latest


def validate_pdf_bytes(content: bytes, url: str, content_type: Optional[str] = None) -> int:
    """Reject HTML error pages, truncated or unparsable downloads; return the page count."""

    if not content:
        raise RuntimeError(f"Empty download from {url}")
    if not content.startswith(b"%PDF-"):
        raise RuntimeError(f"Expected a PDF from {url}, received {content_type or 'unknown content'} ({content[:15]!r}...)")
    if b"%%EOF" not in content[-2048:]:
        raise RuntimeError(f"Download from {url} is truncated (no %%EOF marker)")
    try:
        from io import BytesIO

        from pypdf import PdfReader

        pages = len(PdfReader(BytesIO(content)).pages)
    except Exception as exc:  # pypdf raises many exception types for damaged files
        raise RuntimeError(f"Download from {url} is not a readable PDF: {exc}") from exc
    if pages == 0:
        raise RuntimeError(f"Download from {url} contains no pages")
    return pages


def download_pdf(session: requests.Session, url: str, destination: Path) -> Dict[str, object]:
    response = session.get(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    content = response.content
    if len(content) > MAX_PDF_BYTES:
        raise RuntimeError(f"Download from {url} exceeds {MAX_PDF_BYTES} bytes")
    pages = validate_pdf_bytes(content, url, response.headers.get("content-type"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return {
        "download_url": response.url,
        "filename": destination.name,
        "bytes": len(content),
        "pages": pages,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _previous_documents(output_dir: Path, year: int) -> List[Dict[str, object]]:
    """Entries of the previous manifest; if it is unreadable, official FIA files found on disk."""

    path = output_dir / MANIFEST_NAME
    if path.exists():
        try:
            documents = json.loads(path.read_text(encoding="utf-8")).get("documents", [])
            if isinstance(documents, list) and all(isinstance(doc, dict) and "filename" in doc for doc in documents):
                return documents
        except (OSError, ValueError, AttributeError):
            pass
        print(f"WARNING: {path} is unreadable; recognising previous downloads by their official file names")
    found = []
    for pdf in sorted(output_dir.glob("*.pdf")):
        section = _section_from_url(pdf.name, year)
        if section:
            found.append({"filename": pdf.name, "section": section, "year": year})
    return found


def _write_json_atomic(path: Path, data: object) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".json")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def fetch_regulations(
    output_dir: Path,
    year: int,
    sections: Iterable[str],
    category_url: str = DEFAULT_CATEGORY_URL,
    dry_run: bool = False,
    session: Optional[requests.Session] = None,
) -> Dict[str, object]:
    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8"})

    category_response = session.get(category_url, timeout=60)
    category_response.raise_for_status()
    candidates = discover_candidates(category_response.text, category_response.url, year)
    latest = latest_candidates(candidates)

    requested = list(dict.fromkeys(section.upper() for section in sections))
    missing = [section for section in requested if section not in latest]
    if missing:
        raise RuntimeError(
            "Could not discover official FIA PDF links for section(s): "
            + ", ".join(missing)
            + ". The FIA page structure may have changed."
        )
    filenames = {section: _filename_from_url(latest[section].url) for section in requested}
    if len(set(filenames.values())) != len(filenames):
        raise RuntimeError(f"Two sections resolved to the same file name: {filenames}")

    entries: List[Dict[str, object]] = []
    for section in requested:
        chosen = latest[section]
        entries.append(
            {
                "section": section,
                "year": year,
                "issue": chosen.issue,
                "issue_date": chosen.date,
                "source_category": category_response.url,
                "source_url": chosen.url,
                "filename": filenames[section],
                "superseded_issues_on_page": sorted(
                    c.url for c in candidates if c.section == section and c.url != chosen.url
                ),
            }
        )

    manifest: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "year": year,
        "official_category": category_response.url,
        "usage_note": (
            "Downloaded directly from FIA for local/private project use. "
            "Do not redistribute these FIA Publications without the rights holder's permission."
        ),
        "documents": entries,
    }
    if dry_run:
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=".download-"))
    try:
        # Download and validate everything before touching the existing files.
        for entry in entries:
            entry.update(download_pdf(session, str(entry["source_url"]), staging / str(entry["filename"])))
        hashes = [entry["sha256"] for entry in entries]
        if len(set(hashes)) != len(hashes):
            raise RuntimeError("Two sections downloaded byte-identical files; refusing to write an ambiguous set")

        previous = _previous_documents(output_dir, year)
        for entry in entries:
            os.replace(staging / str(entry["filename"]), output_dir / str(entry["filename"]))
        new_files = {(entry["year"], entry["section"]): str(entry["filename"]) for entry in entries}
        removed, kept = [], []
        for old in previous:
            name = str(old["filename"])
            key = (old.get("year", year), old.get("section"))
            if key in new_files:
                # Same section and year: replaced by the newly downloaded issue.
                stale = output_dir / name
                if name != new_files[key] and stale.is_file() and stale.parent == output_dir:
                    stale.unlink()
                    removed.append(name)
            elif (output_dir / name).is_file():
                kept.append(old)  # a section (or year) this run did not request stays as it was
        manifest["documents"] = entries + kept
        manifest["removed_superseded_files"] = removed
        _write_json_atomic(output_dir / MANIFEST_NAME, manifest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


def _default_output() -> Path:
    """FIA_DOCS_PATH (from the environment or .env), resolved like the RAG resolves it."""

    if os.getenv("F1_COPILOT_LOAD_DOTENV", "1") != "0":
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:  # the discovery-only CI job installs just requests and pypdf
            pass
    configured = Path(os.getenv("FIA_DOCS_PATH") or "data/fia_docs").expanduser()
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--sections", nargs="+", choices=list(DEFAULT_SECTIONS), default=list(DEFAULT_SECTIONS))
    parser.add_argument(
        "--output", type=Path, default=_default_output(), help="Target directory (default: FIA_DOCS_PATH or data/fia_docs)"
    )
    parser.add_argument("--category-url", default=DEFAULT_CATEGORY_URL)
    parser.add_argument("--dry-run", action="store_true", help="Discover and print links without downloading PDFs")
    args = parser.parse_args()

    try:
        manifest = fetch_regulations(
            output_dir=args.output,
            year=args.year,
            sections=args.sections,
            category_url=args.category_url,
            dry_run=args.dry_run,
        )
    except (requests.RequestException, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
