#!/usr/bin/env python3
"""Download current official FIA Formula 1 regulation PDFs for local RAG use.

The FIA website terms reserve copyright in FIA Publications and limit copying to
private/non-commercial use unless FIA gives prior written consent. For that
reason this script downloads files into the git-ignored data/fia_docs directory
and the project does not redistribute the PDFs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urljoin, urlparse

import requests

FIA_BASE_URL = "https://www.fia.com"
DEFAULT_CATEGORY_URL = "https://www.fia.com/regulation/category/2182"
DEFAULT_SECTIONS = ("A", "B", "C", "D", "E", "F")
USER_AGENT = "f1-ai-copilot/1.1 (local FIA regulation downloader)"


def _filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "fia_regulations.pdf"


def _section_from_url(url: str, year: int) -> str | None:
    decoded = html.unescape(url).lower()
    patterns = (
        rf"fia[_ -]?{year}[_ -]?f1[_ -]?regulations.*?section[_ -]?([a-f])",
        rf"section[_ -]?([a-f]).*?{year}",
    )
    for pattern in patterns:
        match = re.search(pattern, decoded)
        if match:
            return match.group(1).upper()
    return None


def discover_pdf_urls(category_html: str, category_url: str, year: int) -> Dict[str, str]:
    """Discover the latest linked regulation PDF for each A-F section."""

    hrefs = re.findall(r'href=["\']([^"\']+)["\']', category_html, flags=re.IGNORECASE)
    candidates: List[str] = []
    for href in hrefs:
        value = html.unescape(href).strip()
        if ".pdf" not in value.lower():
            continue
        absolute = urljoin(category_url, value)
        if str(year) in absolute and "f1" in absolute.lower() and "regulation" in absolute.lower():
            candidates.append(absolute)

    # Preserve page order. The FIA category lists current issues first.
    discovered: Dict[str, str] = {}
    for url in candidates:
        section = _section_from_url(url, year)
        if section in DEFAULT_SECTIONS and section not in discovered:
            discovered[section] = url

    # Some FIA layouts wrap titles around file links. Try a bounded nearby-text
    # fallback when the filename itself does not contain a section marker.
    missing = [section for section in DEFAULT_SECTIONS if section not in discovered]
    for section in missing:
        title_pattern = re.compile(
            rf"FIA\s+{year}\s+F1\s+Regulations\s*-\s*Section\s+{section}\b",
            flags=re.IGNORECASE,
        )
        for title_match in title_pattern.finditer(category_html):
            window = category_html[title_match.start() : title_match.start() + 5000]
            pdf_match = re.search(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', window, flags=re.IGNORECASE)
            if pdf_match:
                discovered[section] = urljoin(category_url, html.unescape(pdf_match.group(1)))
                break

    return discovered


def download_pdf(session: requests.Session, url: str, destination: Path) -> Dict[str, object]:
    response = session.get(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF"):
        raise RuntimeError(f"Expected a PDF from {url}, received {response.headers.get('content-type')}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return {
        "download_url": response.url,
        "filename": destination.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def fetch_regulations(
    output_dir: Path,
    year: int,
    sections: Iterable[str],
    category_url: str = DEFAULT_CATEGORY_URL,
    dry_run: bool = False,
) -> Dict[str, object]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8"})

    category_response = session.get(category_url, timeout=60)
    category_response.raise_for_status()
    discovered = discover_pdf_urls(category_response.text, category_response.url, year)

    requested = [section.upper() for section in sections]
    missing = [section for section in requested if section not in discovered]
    if missing:
        raise RuntimeError(
            "Could not discover official FIA PDF links for section(s): "
            + ", ".join(missing)
            + ". The FIA page structure may have changed."
        )

    manifest_entries = []
    for section in requested:
        url = discovered[section]
        filename = _filename_from_url(url)
        entry: Dict[str, object] = {
            "section": section,
            "year": year,
            "source_category": category_response.url,
            "source_url": url,
        }
        if dry_run:
            entry["filename"] = filename
        else:
            entry.update(download_pdf(session, url, output_dir / filename))
        manifest_entries.append(entry)

    manifest: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "year": year,
        "official_category": category_response.url,
        "usage_note": (
            "Downloaded directly from FIA for local/private project use. "
            "Do not redistribute these FIA Publications without the rights holder's permission."
        ),
        "documents": manifest_entries,
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--sections", nargs="+", choices=list(DEFAULT_SECTIONS), default=list(DEFAULT_SECTIONS))
    parser.add_argument("--output", type=Path, default=Path("data/fia_docs"))
    parser.add_argument("--category-url", default=DEFAULT_CATEGORY_URL)
    parser.add_argument("--dry-run", action="store_true", help="Discover and print links without downloading PDFs")
    args = parser.parse_args()

    manifest = fetch_regulations(
        output_dir=args.output,
        year=args.year,
        sections=args.sections,
        category_url=args.category_url,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
