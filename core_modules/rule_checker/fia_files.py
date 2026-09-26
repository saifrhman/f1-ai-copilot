"""File names of the official FIA regulation downloads, shared by the downloader and the RAG.

Standard library only: ``scripts/fetch_fia_regulations.py`` imports this module in a CI job that
installs just requests and pypdf, so nothing here may import the RAG package (LangChain, Qdrant).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

MANIFEST_NAME = "manifest.json"


def fia_section(filename: str, year: Optional[int] = None) -> Optional[Tuple[str, str]]:
    """(regulation year, section letter) of an official "FIA <year> F1 Regulations - Section X" file name.

    The year is the regulation-year prefix of the name; dates elsewhere in it (the issue date) never
    count, so a 2027 document issued in 2026 is not taken for a 2026 regulation. With ``year``, only
    a file of that regulation year matches.
    """

    year_pattern = r"\d{4}" if year is None else str(year)
    match = re.match(
        rf"fia[_ -]*({year_pattern})[_ -]*(?:f1|formula[_ -]*1)[_ -]*regulations[_ -]*-?[_ -]*section[_ -]*([a-f])(?![a-z])",
        filename.lower(),
    )
    return (match.group(1), match.group(2).upper()) if match else None


def fia_issue(filename: str) -> Optional[int]:
    """Issue number in a regulation file name ("..._-_iss_08_-_2026-08-05.pdf" -> 8), or None."""

    match = re.search(r"iss(?:ue)?[_ -]?0*(\d{1,3})(?!\d)", filename.lower())
    return int(match.group(1)) if match else None
