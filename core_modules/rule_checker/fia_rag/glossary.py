"""Defined terms of the regulations, used to add their official definitions to the evidence.

The regulations define terms once, in definition appendices, and then use them
(often as abbreviations such as "TTCS") throughout. A passage that says
"during a TTCS, a Stop-and-Go Penalty will be imposed" does not say that a
TTCS includes the Race; the definition does. At index time the definitions are
extracted verbatim from the PDFs; at question time, definitions of the
abbreviations used in the retrieved passages (and of terms named in the
question) are added to the evidence as additional, citable excerpts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

# '"Total Time Classified Session" (or "TTCS") is any track running session ...'
# '"Control" means the power to conduct ...' / '"Cost Cap" shall mean ...'
_DEFINITION = re.compile(
    r"[“\"]\s*(?P<term>[A-Z][^”\"\n]{1,80}?)\s*[”\"]\s*"
    r"(?:\(\s*or\s*[“\"]\s*(?P<abbr>[A-Za-z0-9&/-]{2,12})\s*[”\"]\s*\)\s*)?"
    r"(?:is|are|means|shall\s+mean|has\s+the\s+meaning)\b"
)
_MAX_DEFINITION_CHARS = 700
# Part of the stored glossary key: bump when extraction changes so old glossaries are rebuilt.
GLOSSARY_VERSION = "1"


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    abbreviation: Optional[str]
    definition: str
    source: str
    page: Optional[int]
    page_label: Optional[str]
    section: Optional[str]
    source_url: Optional[str]
    # Share of indexed chunks that use the abbreviation; near-ubiquitous ones ("FIA") are not worth a slot.
    frequency: float = 0.0

    def to_payload(self) -> Dict[str, object]:
        return {
            "term": self.term,
            "abbreviation": self.abbreviation,
            "definition": self.definition,
            "source": self.source,
            "page": self.page,
            "page_label": self.page_label,
            "section": self.section,
            "source_url": self.source_url,
            "frequency": self.frequency,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "GlossaryEntry":
        return cls(
            term=str(payload["term"]),
            abbreviation=payload.get("abbreviation") or None,
            definition=str(payload["definition"]),
            source=str(payload.get("source", "unknown")),
            page=payload.get("page"),
            page_label=payload.get("page_label"),
            section=payload.get("section"),
            source_url=payload.get("source_url"),
            frequency=float(payload.get("frequency") or 0.0),
        )


# Abbreviations used in more than this share of all chunks ("FIA": 41% of the 2026 regulations)
# are general vocabulary; their definitions would crowd out the specific ones.
MAX_TERM_FREQUENCY = 0.2
# Below this many chunks a share says nothing about how common a term is.
MIN_CHUNKS_FOR_FREQUENCY = 50


def with_frequencies(entries: Sequence[GlossaryEntry], chunk_texts: Sequence[str]) -> List[GlossaryEntry]:
    """Attach, per abbreviation, the share of chunks that use it (0 for corpora too small to tell)."""

    total = len(chunk_texts)
    result = []
    for entry in entries:
        frequency = 0.0
        if entry.abbreviation and total >= MIN_CHUNKS_FOR_FREQUENCY:
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(entry.abbreviation)}(?![A-Za-z0-9])")
            frequency = sum(1 for text in chunk_texts if pattern.search(text)) / total
        result.append(replace(entry, frequency=round(frequency, 4)))
    return result


def _cut(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_DEFINITION_CHARS:
        return text
    cut = text[:_MAX_DEFINITION_CHARS]
    end = cut.rfind(". ")
    return (cut[: end + 1] if end > 200 else cut).strip()


def extract_glossary(pages: Sequence[Document]) -> List[GlossaryEntry]:
    """Verbatim definitions ("Term" (or "ABBR") is/means ...) found in the page texts."""

    entries: List[GlossaryEntry] = []
    seen = set()
    for page in pages:
        text = page.page_content
        matches = list(_DEFINITION.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            definition = _cut(text[match.start() : end])
            term = re.sub(r"\s+", " ", match.group("term")).strip()
            abbreviation = match.group("abbr")
            inner = re.fullmatch(r"(.+?)\s*\(([A-Z][A-Za-z0-9&/-]{1,11})\)", term)  # "Accepted Breach Agreement (ABA)"
            if inner and not abbreviation:
                term, abbreviation = inner.group(1), inner.group(2)
            elif not abbreviation and re.fullmatch(r"[A-Z][A-Z0-9&/-]{1,7}", term):  # "ASN"
                abbreviation = term
            key = (term.lower(), page.metadata.get("section"), page.metadata.get("source"))
            if key in seen or len(definition) < len(match.group(0)) + 10:
                continue
            seen.add(key)
            entries.append(
                GlossaryEntry(
                    term=term,
                    abbreviation=abbreviation,
                    definition=definition,
                    source=str(page.metadata.get("source")),
                    page=page.metadata.get("page"),
                    page_label=page.metadata.get("page_label"),
                    section=page.metadata.get("section"),
                    source_url=page.metadata.get("source_url"),
                )
            )
    return entries


# '"Cost Cap" has the meaning set out in Article D4.1.2.' only points elsewhere; it adds no content.
_REFERENTIAL = re.compile(r"has\s+the\s+meaning\s+(?:given|set\s+out|ascribed|assigned)", re.IGNORECASE)


def _useful(entry: GlossaryEntry) -> bool:
    return not (_REFERENTIAL.search(entry.definition) and len(entry.definition) < 200)


def _find(key: str, text: str, ignore_case: bool = False) -> int:
    """Offset of ``key`` as a whole word in ``text``, or -1."""

    flags = re.IGNORECASE if ignore_case else 0
    match = re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])", text, flags)
    return match.start() if match else -1


def definitions_for(
    glossary: Sequence[GlossaryEntry],
    passage_texts: Sequence[str],
    passage_sections: Sequence[Optional[str]],
    question: str,
    limit: int,
) -> List[GlossaryEntry]:
    """Definitions of terms named in the question and of abbreviations used in the passages.

    * Abbreviations ("TTCS") are matched case-sensitively as whole words, in the
      question and in the passages. Abbreviations used in a large share of all
      chunks ("FIA", see :data:`MAX_TERM_FREQUENCY`) are skipped.
    * Full terms are matched only in the question (the passages use many
      capitalised defined terms): multi-word terms case-insensitively, single
      words case-sensitively (so "official" does not select "Official").
    * Purely referential definitions ("has the meaning set out in ...") are skipped,
      as is a definition whose text already appears in a passage.
    * Where a term is defined in several sections, the section of the passage
      that uses it (for question terms: the best-ranked passage's section) wins.

    Results are ordered by first use: question terms first, then by the rank of the
    first passage using the abbreviation and the position within that passage.
    """

    if limit <= 0 or not glossary:
        return []
    usable = [entry for entry in glossary if _useful(entry)]
    by_abbreviation: Dict[str, List[GlossaryEntry]] = {}
    by_term: Dict[str, List[GlossaryEntry]] = {}
    for entry in usable:
        by_term.setdefault(entry.term.lower(), []).append(entry)
        if entry.abbreviation and entry.frequency <= MAX_TERM_FREQUENCY:
            by_abbreviation.setdefault(entry.abbreviation, []).append(entry)

    ranked_sections = [section for section in passage_sections if section]
    # (passage rank, offset in text, candidates, preferred sections); rank -1 = the question
    found: List[tuple] = []
    for key, candidates in by_abbreviation.items():
        offset = _find(key, question)
        if offset >= 0:
            found.append((-1, offset, candidates, ranked_sections))
            continue
        for position, text in enumerate(passage_texts):
            offset = _find(key, text)
            if offset >= 0:
                section = passage_sections[position] if position < len(passage_sections) else None
                found.append((position, offset, candidates, [section, *ranked_sections]))
                break
    for term, candidates in by_term.items():
        original = candidates[0].term
        if len(original) < 4 or original == candidates[0].abbreviation:
            continue
        multi_word = " " in original.strip()
        offset = _find(term if multi_word else original, question, ignore_case=multi_word)
        if offset >= 0:
            found.append((-1, offset, candidates, ranked_sections))

    joined = "\n".join(passage_texts)
    selected: List[GlossaryEntry] = []
    used = set()
    for _, _, candidates, preferred in sorted(found, key=lambda item: (item[0], item[1])):
        entry = next(
            (c for section in preferred for c in candidates if c.section == section),
            candidates[0],
        )
        identity = entry.term.lower()
        if identity in used or entry.definition[:120] in joined:
            continue
        used.add(identity)
        selected.append(entry)
        if len(selected) >= limit:
            break
    return selected
