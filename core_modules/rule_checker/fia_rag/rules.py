"""Recognise FIA rule identifiers such as ``B2.3.4c``, ``Article C3.2.4`` or ``Appendix D1``.

The 2026 FIA Formula 1 Regulations are split into Sections A-F and number their
articles with the section letter as a prefix (``B1.3.7``, ``C3.5.10``,
``E4.1.1.a``). Older regulation sets use purely numeric articles
(``Article 12.4``). This module is shared by ingestion (chunk metadata) and by
answer validation, where every rule identifier an answer mentions must occur in
the evidence. The grammar is therefore deliberately broad for answers: keyword
forms (``Article``, ``Art.``, ``Rule``, ``Clause``, ``Paragraph``, ``Section``,
``Regulation``) with or without a dot or space, enumerations (``Articles 12.2
and 12.4``), letter-prefixed identifiers in any case and with any letter, and
appendices by letter, number or roman numeral. Appendix identifiers are kept
distinct from article identifiers (``Appendix B2`` is not ``Article B2``).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Tuple

# Components after a dot may have up to 4 digits so an over-long invented id ("B1.6.3.1000") is
# captured whole (and then rejected) instead of being read as its supported parent.
_ID_CORE = r"(?:[A-Za-z]\d{1,3}(?:\.\d{1,4}){0,5}|\d{1,3}(?:\.\d{1,4}){0,5})"
# Optional lettered item ("B2.3.4c", "E4.1.1.a") that is not the start of a word or unit ("155mm").
_ITEM = r"(?:\.?[a-z](?![A-Za-z]))?"
# An identifier must not continue with more digits, so "B1.6.3.1000" is never read as "B1.6.3".
_END = r"(?![A-Za-z0-9])(?!\.\d)"  # "_" may follow (markdown emphasis: __B9.9.9__)
_ID = rf"(?:{_ID_CORE}{_ITEM}{_END})"
# Roman numerals and letter-only appendices are matched case-sensitively, so words ("of") and
# single section letters ("Section C") are not taken for identifiers.
_ROMAN = r"(?-i:[IVXLC]{2,6})(?![\w])"
_LETTERS = r"(?-i:[A-Z]{1,3})(?![\w])"

_ARTICLE_KEYWORDS = r"(?:articles?|arts?\.?|rules?|clauses?|paragraphs?|paras?\.?|regulations?|regs?\.?)"
_SECTION_KEYWORDS = r"(?:sections?)"
_APPENDIX_KEYWORDS = r"(?:appendix|appendices|app\.|annex(?:es)?)"
_ANY_KEYWORD = rf"(?:{_ARTICLE_KEYWORDS}|{_SECTION_KEYWORDS}|{_APPENDIX_KEYWORDS})"

_ARTICLE_REF = re.compile(rf"\b(?:{_ARTICLE_KEYWORDS}\s*({_ID}|{_ROMAN})|{_SECTION_KEYWORDS}\s*({_ID}))", re.IGNORECASE)
_APPENDIX_REF = re.compile(rf"\b{_APPENDIX_KEYWORDS}\s*({_ID}|{_ROMAN}|{_LETTERS})", re.IGNORECASE)
# Continuation of an enumeration after a keyword: "Articles 12.2, 12.3 and 12.4", "B1.2/B1.3", "C3.5 to C3.12".
_LIST_CONTINUATION = re.compile(
    rf"\s*(?:,|;|&|/|\band\b|\bor\b|\bto\b|-|–)\s*(?:({_ANY_KEYWORD})\s*)?({_ID})", re.IGNORECASE
)
# Identifier without a keyword: any letter followed by a dotted number ("B1.6.3", "b9.9.9", "G7.1").
# A preceding "." is allowed ("Art.A6.1.2") but not a preceding digit or letter.
_BARE_REF = re.compile(rf"(?<![A-Za-z0-9])([A-Za-z]\d{{1,3}}(?:\.\d{{1,4}}){{1,5}}{_ITEM}{_END})")
# A purely numeric reference into a named document: "34.7 of the Sporting Regulations".
_NUMERIC_OF_REF = re.compile(
    r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){1,5})\s+of\s+(?:the\s+)?(?:[A-Z][\w-]*\s+){0,4}(?:Regulations|Code|ISC|Rules)\b"
)

# Headings: "ARTICLE D2: OBLIGATIONS ..." / "APPENDIX B1: DEFINITIONS" (position = start of the keyword),
# or a dotted identifier followed by a title-case word ("B2.3.5 Sprint Session ...").
_KEYWORD_HEADING = re.compile(r"\b(ARTICLE|APPENDIX)\s+([A-F]\d{1,2}(?:\.\d{1,3}){0,5})\s*:?\s+(?=[A-Z])")
_DOTTED_HEADING = re.compile(r"(?:^|(?<=\s))([A-F]\d{1,2}(?:\.\d{1,3}){1,5})\s+(?=[A-Z](?:[a-z]|[A-Z][a-z]|\s+[a-z])|[\"“])")
# Words that make a following identifier a cross-reference rather than a heading.
_REFERENCE_CONTEXT = re.compile(
    r"(?:\b(?:articles?|arts?\.?|appendix|appendices|sections?|paragraphs?|in|under|of|and|or|to|see|with|per|by|"
    r"from|pursuant|as|than|within)\s*|[,&/(\-–]\s*)$",
    re.IGNORECASE,
)

# PDF extraction sometimes splits an identifier after a dot: "Article C5. 2.8".
_SPLIT_ID = re.compile(r"\b([A-F]\d{1,2}(?:\.\d{1,3})*)\. (?=\d{1,3}(?:\.\d|\b))")


def normalise_rule_id(raw: str) -> str:
    """Canonical form: no spaces or trailing punctuation, upper-case leading letter; appendices keep their kind."""

    value = re.sub(r"\s+", " ", raw).strip().rstrip(".,;:)")
    if value.lower().startswith("appendix "):
        return "Appendix " + value.split(" ", 1)[1].replace(" ", "").upper()
    value = value.replace(" ", "")
    if value and value[0].isalpha():
        value = value[0].upper() + value[1:]
    return value


def _appendix(raw: str) -> str:
    return normalise_rule_id("Appendix " + raw)


def extract_rule_ids(text: str) -> List[str]:
    """Rule identifiers mentioned in free text, in order of first appearance, without duplicates."""

    text = _SPLIT_ID.sub(r"\1.", text)
    spans: List[Tuple[int, str]] = []

    def walk_list(end: int, first: str, appendix: bool) -> None:
        # After "Articles 5 and" a plain integer continues the list; after a dotted or
        # lettered identifier only dotted/lettered ones do ("Article B1.6.3 and 10 seconds").
        plain_list = first.isdigit()
        while True:
            more = _LIST_CONTINUATION.match(text, end)
            if not more:
                return
            keyword, value = more.group(1), more.group(2)
            if keyword:  # "Appendix A7, Paragraph 2.1": an explicit keyword sets the kind
                appendix = bool(re.fullmatch(_APPENDIX_KEYWORDS, keyword, re.IGNORECASE))
            elif value.isdigit() and not plain_list:
                return
            spans.append((more.start(2), _appendix(value) if appendix else normalise_rule_id(value)))
            end = more.end()

    for match in _ARTICLE_REF.finditer(text):
        group = 1 if match.group(1) else 2
        spans.append((match.start(group), normalise_rule_id(match.group(group))))
        walk_list(match.end(), match.group(group), appendix=False)
    for match in _APPENDIX_REF.finditer(text):
        spans.append((match.start(1), _appendix(match.group(1))))
        walk_list(match.end(), match.group(1), appendix=True)
    claimed = {position for position, _ in spans}
    for match in _BARE_REF.finditer(text):
        if match.start(1) not in claimed:
            spans.append((match.start(1), normalise_rule_id(match.group(1))))
    for match in _NUMERIC_OF_REF.finditer(text):
        if match.start(1) not in claimed:
            spans.append((match.start(1), normalise_rule_id(match.group(1))))

    found: List[str] = []
    seen: Set[str] = set()
    for _, rule in sorted(spans, key=lambda item: item[0]):
        if rule not in seen:
            seen.add(rule)
            found.append(rule)
    return found


def extract_headings(text: str) -> List[Tuple[int, str]]:
    """Start positions and identifiers of numbered headings, in text order.

    ``ARTICLE B2: FORMAT ...`` yields ``B2`` positioned at ``ARTICLE`` (so a chunk
    that begins with the heading belongs to that article); ``APPENDIX B1: ...``
    yields ``Appendix B1``. A dotted identifier counts as a heading only when a
    title-case word follows and it is not preceded by referencing words
    ("in", "under", "Articles ... to", ...), so cross-references and table
    cells are not mistaken for headings.
    """

    headings: List[Tuple[int, str]] = []
    for match in _KEYWORD_HEADING.finditer(text):
        value = match.group(2)
        headings.append((match.start(), _appendix(value) if match.group(1) == "APPENDIX" else normalise_rule_id(value)))
    keyword_ids = {match.start(2) for match in _KEYWORD_HEADING.finditer(text)}
    for match in _DOTTED_HEADING.finditer(text):
        position = match.start(1)
        if position in keyword_ids or _REFERENCE_CONTEXT.search(text[max(0, position - 30) : position]):
            continue
        headings.append((position, normalise_rule_id(match.group(1))))
    return sorted(headings, key=lambda item: item[0])


def item_letters(text: str) -> Set[str]:
    """Letters used as list-item markers in regulation text ("a. ...", "(b)")."""

    return {m.lower() for m in re.findall(r"(?:^|[\s(])([a-z])(?:\.\s|\))", text)}


def is_supported(rule: str, evidence_rules: Iterable[str], evidence_items: Optional[Set[str]] = None) -> bool:
    """A cited rule is supported if the evidence contains it or one of its sub-rules.

    ``B2.3`` is supported by evidence containing ``B2.3.5`` (a parent article),
    but ``B2.3.5.1`` is *not* supported by evidence that only contains ``B2.3.5``,
    and ``Appendix B2`` never supports ``Article B2``. A lettered item
    (``B1.6.3a``) is supported by its article when ``evidence_items`` is not
    given, or when that letter is used as an item marker in the evidence.
    """

    target = normalise_rule_id(rule).lower()
    evidence = [normalise_rule_id(candidate).lower() for candidate in evidence_rules]
    if _supported(target, evidence):
        return True
    # A list-item suffix ("B1.6.3a" / "B1.6.3.a") refers to a lettered item of
    # the article, which the evidence shows as "B1.6.3 a." - check the article.
    base = re.sub(r"\.?[a-z]$", "", target)
    if base == target or not re.search(r"\d", base) or not _supported(base, evidence):
        return False
    return evidence_items is None or target[-1] in evidence_items


def _supported(target: str, evidence: Iterable[str]) -> bool:
    for value in evidence:
        if value == target:
            return True
        if value.startswith(target) and len(value) > len(target) and value[len(target)] in ".abcdefghijklmnopqrstuvwxyz":
            return True
    return False
