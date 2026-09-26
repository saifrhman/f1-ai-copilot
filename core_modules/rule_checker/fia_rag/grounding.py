"""Deterministic validation of a model answer against the evidence it was given.

Prompting asks the model to cite sources and decline without evidence; this
module *enforces* it. An answer is only accepted when:

* it is not a decline (the ``INSUFFICIENT_EVIDENCE`` sentinel or the decline message),
* the model finished its answer (no truncated output; checked in ``generation``),
* every source label it uses, in any citation style (``[S2]``, ``(S2)``, ``[2]``,
  ``[source: S2]``, ``【S2】``, ranges ``[S1-S3]``, "excerpt 2"), refers to an
  excerpt that was actually supplied,
* it cites at least one excerpt,
* no statement is left uncited - text after the last citation of a paragraph (or a
  paragraph without any) must not state anything, unless it is explicitly marked as
  an inference ("Thus", "Therefore", ...) or only talks about the excerpts
  ("The excerpts do not specify ..."),
* every article/section/appendix identifier it mentions occurs in the passages it cites,
* every number it states (limits, penalties, amounts, times) occurs in the text of the
  passages it cites - list markers, citation labels, rule identifiers, inference-marked
  sentences and references to where a cited passage is (its page, issue, file name or
  regulation year, each compared with that kind of metadata of the passage) excepted.

Anything else is replaced by the standard decline message, with the reason and
the raw model output kept for inspection. The checks establish citation, identifier
and number integrity; they cannot prove that a cited passage entails a sentence
(the optional claim verifier in ``generation`` adds a model-based entailment check).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from core_modules.rule_checker.fia_files import fia_issue, fia_section

from .retrieval import RetrievedPassage
from .rules import extract_rule_ids, is_supported, item_letters, rule_id_spans

DECLINE_ANSWER = (
    "I cannot answer that from the indexed FIA regulations because the retrieved "
    "context does not contain sufficient evidence."
)
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# The upper-case token is the model's decline. Prose such as "the excerpts give insufficient
# evidence on X" inside a cited partial answer is not a decline; a reply that consists only of
# a decline phrase (any case, with or without citations) is.
_SENTINEL = re.compile(r"INSUFFICIENT[\s_-]*EVIDENCE")
_DECLINE_ONLY = re.compile(r"^(?:insufficient[\s_-]*evidence|no\s+(?:sufficient\s+)?evidence)$")

_NUMBER = r"\d{1,6}"
_ITEM_SEP = r"\s*(?:,|;|/|&|\band\b)\s*"
_RANGE_SEP = r"\s*(?:-|–|—|\bto\b)\s*"
_LABEL_TOKEN = rf"(?:(?:sources?|src|refs?|excerpts?|passages?)\s*[:#]?\s*)?S?\s*{_NUMBER}"
_LABEL_LIST = rf"{_LABEL_TOKEN}(?:(?:{_ITEM_SEP}|{_RANGE_SEP}){_LABEL_TOKEN})*"
# [S1], [s2], [S1, S3], [S1-S3], [2], [source: S7], [Source 7] (bracket content must be only labels)
_BRACKET_CITATION = re.compile(rf"\[\s*({_LABEL_LIST})\s*\]", re.IGNORECASE)
# (S7), (source S7): an explicit S or "source" is required, so "one (1) hour" is not a citation
_PAREN_CITATION = re.compile(
    rf"\(\s*((?:(?:sources?|excerpts?)\s*[:#]?\s*S?\s*|S\s*){_NUMBER}(?:(?:{_ITEM_SEP}|{_RANGE_SEP}){_LABEL_TOKEN})*)\s*\)",
    re.IGNORECASE,
)
# "excerpt 3 says", "according to source S2"
_PROSE_CITATION = re.compile(rf"\b(?:excerpts?|sources?|passages?)\s+(S?\s*{_NUMBER})\b", re.IGNORECASE)
# Each style and the part of a match that shows its labels: a prose citation keeps its word ("source S2").
_CITATION_STYLES = ((_BRACKET_CITATION, 0), (_PAREN_CITATION, 0), (_PROSE_CITATION, 1))
_MAX_RANGE = 50

# An explicit inference marker at the start of a sentence, also inside an opening parenthesis
# ("(Inference: the difference is ...)"), exempts that sentence from the citation and number checks.
_INFERENCE_MARKER = re.compile(
    r"^[\s*_>#(-]*(?:thus|therefore|hence|so|consequently|this means|in other words|in summary|overall|"
    r"inference|by inference|it follows)\b",
    re.IGNORECASE,
)
_HAS_NUMBER = re.compile(r"\d")
# Where one sentence or list item ends: whitespace after ".", "!" or "?", or a line break.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
# Sentences about the limits of the supplied evidence, not about the regulations:
# "The excerpts do not specify ...", "No excerpt covers ...", "This is not stated in the provided text."
_EVIDENCE_WORD = re.compile(
    r"\b(?:excerpts?|sources?|passages?|provided\s+(?:text|context|regulations?)|supplied\s+(?:text|regulations?))\b",
    re.IGNORECASE,
)
_LIMITATION_WORD = re.compile(
    r"\b(?:not|no|none|neither|nor|only|silent|beyond|outside|without|unclear|cannot|can't|don't|doesn't|isn't|aren't)\b",
    re.IGNORECASE,
)
_META = re.compile(
    r"^(?:I|we)\s+(?:cannot|can't|could\s+not|couldn't|did\s+not|didn't|do\s+not|don't)\b"
    r"|\b(?:is|are)\s+not\s+(?:specified|stated|covered|mentioned|addressed|defined|given)\b",
    re.IGNORECASE,
)
_MIN_CLAIM_WORDS = 4

# A number as written in an answer or passage: "80", "0.30", "1,000", "135,000,000".
_NUMBER_TOKEN = re.compile(r"(?<![A-Za-z0-9.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)*(?![A-Za-z]\d)")
_SCALE = re.compile(r"\s*(million|mn|billion|bn)\b", re.IGNORECASE)
_SCALES = {"million": 10**6, "mn": 10**6, "billion": 10**9, "bn": 10**9}
_NUMBER_WORDS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen twenty".split()
    )
}
_NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
                      "hundred": 100, "thousand": 1000, "half": 0.5})
_NUMBER_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)
_LIST_MARKER = re.compile(r"(?m)^[ \t>*_-]*\(?\d{1,2}[.)](?=\s)")
# Where a passage is, not what it says: "page 9", "p. 9", "PDF page 9", "printed page B9", "Issue 08" (capital I:
# the verb "issue" is no reference), a PDF file name, and the year of a document ("the 2026 Sporting Regulations",
# "Regulations 2026").
_LOCATOR = re.compile(
    r"(?P<page>\b(?:(?:pdf|printed)\s+)?(?:pages?|pp?\.)\s*[A-F]?\d{1,4}\b)"
    r"|(?P<issue>\b(?-i:Iss(?:ue)?)\.?\s*\d{1,3}\b)"
    r"|(?P<file>[\w.-]+\.pdf\b)"
    r"|(?P<year>\b(?:19|20)\d\d(?=\s+(?:(?-i:[A-Z0-9][\w-]*)\s+){0,4}(?:regulations?|season|championship|edition)\b)"
    r"|\b(?:regulations?|season|championship)\s+(?:19|20)\d\d\b)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d\d(?!\d)")


class Outcome:
    ANSWERED = "answered"
    DECLINED = "declined"


class DeclineReason:
    NO_EVIDENCE = "no_evidence_above_threshold"
    MODEL_DECLINED = "model_declined"
    EMPTY_OUTPUT = "empty_model_output"
    MISSING_CITATION = "missing_citation"
    INVALID_CITATION = "invalid_citation"
    UNSUPPORTED_RULE = "unsupported_rule_reference"
    UNCITED_CLAIM = "uncited_claim"
    UNSUPPORTED_NUMBER = "unsupported_number"
    TRUNCATED = "truncated_model_output"
    UNVERIFIED_CLAIM = "unverified_claim"


@dataclass
class ValidatedAnswer:
    status: str
    answer: str
    reason: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    referenced_rules: List[str] = field(default_factory=list)
    unsupported_rules: List[str] = field(default_factory=list)
    uncited_claims: List[str] = field(default_factory=list)
    unsupported_numbers: List[str] = field(default_factory=list)
    unverified_claims: List[str] = field(default_factory=list)
    model_output: Optional[str] = None
    # The provider's finish_reason of a reply that stopped before it finished (TRUNCATED only).
    finish_reason: Optional[str] = None

    @property
    def grounded(self) -> bool:
        return self.status == Outcome.ANSWERED


def label_passages(passages: Sequence[RetrievedPassage]) -> Dict[str, RetrievedPassage]:
    """Assign ``S1..Sn`` in rank order; these are the only valid citation labels."""

    return {f"S{i}": passage for i, passage in enumerate(passages, start=1)}


_BRACKETS = str.maketrans({"【": "[", "】": "]", "〔": "[", "〕": "]", "（": "(", "）": ")"})


def _normalise(text: str) -> str:
    """Fold full-width brackets/digits (【S2】, ［S２］) to ASCII so every citation style is seen."""

    return unicodedata.normalize("NFKC", text).translate(_BRACKETS)


def _normalise_with_origins(text: str) -> Tuple[str, List[int]]:
    """``_normalise`` applied character by character, and the offset in ``text`` of each resulting character."""

    pieces: List[str] = []
    origins: List[int] = []
    for offset, char in enumerate(text):
        piece = _normalise(char)
        pieces.append(piece)
        origins.extend([offset] * len(piece))
    return "".join(pieces), origins


def _labels_in(group: str) -> List[str]:
    labels: List[str] = []
    tokens = re.findall(rf"({_NUMBER})|({_RANGE_SEP})", group, flags=re.IGNORECASE)
    numbers: List[int] = []
    pending_range = False
    for number, range_sep in tokens:
        if range_sep:
            pending_range = True
            continue
        value = int(number)
        if pending_range and numbers:
            low = numbers[-1]
            if value < low or value - low > _MAX_RANGE:
                labels.append(f"S{value}")  # nonsensical range: validated as a (likely invalid) label
            else:
                labels.extend(f"S{n}" for n in range(low + 1, value + 1))
        else:
            labels.append(f"S{value}")
        numbers.append(value)
        pending_range = False
    return labels


def _citations(text: str) -> List[Tuple[int, int, int, List[str]]]:
    """(start, end, start of the shown labels, labels) of every citation-like token, in text order.

    Matching runs on the normalised text, so 【S2】 and ［Ｓ２］ are citations, and the offsets are mapped
    back into ``text``. Overlapping matches count once, as the earliest and outermost one: "[Source S1]"
    and "(source S1)" also contain the prose citation "source S1".
    """

    normalised, origins = _normalise_with_origins(text)
    matches = sorted(
        (
            (origins[match.start()], origins[match.end() - 1] + 1, origins[match.start(shown)], match.group(1))
            for pattern, shown in _CITATION_STYLES
            for match in pattern.finditer(normalised)
        ),
        key=lambda match: (match[0], -match[1]),
    )
    citations: List[Tuple[int, int, int, List[str]]] = []
    for start, end, shown_start, group in matches:
        if not citations or start >= citations[-1][1]:
            citations.append((start, end, shown_start, _labels_in(group)))
    return citations


def citation_spans(text: str) -> List[Tuple[int, int, List[str]]]:
    """(start, end, labels) of every citation-like token in ``text``, in text order, never overlapping."""

    return [(start, end, labels) for start, end, _, labels in _citations(text)]


def shown_citation_spans(text: str) -> List[Dict[str, Any]]:
    """Where ``text`` shows each citation, for clients that mark them: ``start``/``end`` offsets into
    ``text`` (the whole citation, or only the label of a prose citation such as "source S2") and the
    ``labels`` it names, ranges expanded."""

    return [{"start": start, "end": end, "labels": labels} for _, end, start, labels in _citations(text)]


def _without_citations(text: str, replacement: str = "") -> str:
    """``text`` normalised, with every citation replaced by ``replacement``."""

    text = _normalise(text)
    for start, end, _ in reversed(citation_spans(text)):
        text = text[:start] + replacement + text[end:]
    return text


def extract_citations(text: str) -> List[str]:
    """Normalised labels (``S3``) in order of first appearance."""

    labels: List[str] = []
    for _, _, group in citation_spans(text):
        for label in group:
            if label not in labels:
                labels.append(label)
    return labels


def _is_meta(sentence: str) -> bool:
    """A heading, list introduction or remark about the evidence rather than a regulatory statement."""

    stripped = sentence.strip(" \t*_#>")
    if stripped.endswith((":", "?")):
        return not (_HAS_NUMBER.search(stripped) or extract_rule_ids(stripped))
    words = re.findall(r"[A-Za-z]+", stripped)
    if len(words) < _MIN_CLAIM_WORDS and not _HAS_NUMBER.search(stripped):
        return True
    about_evidence = bool(_EVIDENCE_WORD.search(stripped) and _LIMITATION_WORD.search(stripped)) or bool(_META.search(stripped))
    return about_evidence and not (_HAS_NUMBER.search(stripped) or extract_rule_ids(stripped))


_SOURCES_LABEL = re.compile(r"(?:sources?|references?|citations?|evidence)\s*:?", re.IGNORECASE)
# A sentence fragment that is only a list marker ("1", "b", "ii") left by sentence splitting.
_BARE_MARKER = re.compile(r"\(?(?:\d{1,2}|[A-Za-z]|[ivxIVX]{1,4})")


def _is_sources_block(block: str) -> bool:
    """A paragraph that consists only of citations, optionally labelled ("Sources: [S1][S3]")."""

    if not citation_spans(block):
        return False
    rest = _SOURCES_LABEL.sub(" ", _without_citations(block))
    return not rest.strip(" \t\n.,;:*_-()[]")


def uncited_claims(text: str) -> List[str]:
    """Statements that no citation covers.

    A citation covers the text of its paragraph (block separated by a blank
    line) up to the citation; a final paragraph that consists only of citations
    (a sources block) covers everything before it. Other text - whatever follows
    a paragraph's last citation, or a whole paragraph without citations - must
    not make statements, unless a sentence starts with an explicit inference
    marker or is only a heading, a list marker or introduction, or a remark
    about the excerpts themselves.
    """

    text = _normalise(text)
    blocks = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    if blocks and _is_sources_block(blocks[-1]):
        return []
    claims: List[str] = []
    for block in blocks:
        spans = citation_spans(block)
        tail = block[spans[-1][1]:] if spans else block
        tail_stripped = tail.strip(" \t\n.,;:!?)*_")
        if not tail_stripped or _INFERENCE_MARKER.match(tail_stripped):
            continue
        # A trailing sentence may itself start with an inference marker after earlier cited text.
        for sentence in SENTENCE_BREAK.split(tail.strip()):
            sentence = sentence.strip(" \t.,;)*_")
            if not sentence or _BARE_MARKER.fullmatch(sentence):
                continue
            if _INFERENCE_MARKER.match(sentence) or _is_meta(sentence):
                continue
            claims.append(sentence[:200])
    return claims


def _token_values(text: str, match: "re.Match[str]") -> Set[Decimal]:
    """Value of one number token ("135 million" -> {135, 135000000}); empty for identifiers like "1.6.3"."""

    token = match.group(0)
    if token.count(".") > 1:
        return set()
    try:
        value = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return set()
    values = {value.normalize()}
    scale = _SCALE.match(text, match.end())
    if scale:
        values.add((value * _SCALES[scale.group(1).lower()]).normalize())
    return values


def _numbers(text: str) -> Set[Decimal]:
    """Numeric values in ``text`` (digits and number words; "135 million" also as 135000000)."""

    values: Set[Decimal] = set()
    for match in _NUMBER_TOKEN.finditer(text):
        values |= _token_values(text, match)
    for match in _NUMBER_WORD.finditer(text):
        values.add(Decimal(str(_NUMBER_WORDS[match.group(1).lower()])).normalize())
    return values


def _without_locators(text: str, cited: Sequence[RetrievedPassage]) -> str:
    """Blank the references to where a cited passage is (page, issue, file name, regulation year).

    Each reference is compared with its own kind of metadata of the cited passages: a page with
    their page numbers and labels, an issue with the issue number in their file names, a year with
    their regulation year, a file name with their file names. So "(page 9)" of a page-9 passage is
    exempt from the number check, but "(page 99)", or a "5" taken from a file dated 2026-08-05, is not.
    """

    located: Dict[str, Set[Any]] = {"page": set(), "issue": set(), "file": set(), "year": set()}
    for passage in cited:
        for value in (passage.page_label, None if passage.page is None else str(passage.page)):
            if value:
                located["page"] |= _numbers(value)
        issue = fia_issue(passage.source)
        if issue is not None:
            located["issue"].add(Decimal(issue))
        located["file"].add(passage.source.lower())
        # An official name's regulation year, not the year of its issue date; otherwise any year in the name.
        official = fia_section(passage.source)
        located["year"] |= {Decimal(year) for year in ([official[0]] if official else _YEAR.findall(passage.source))}

    def blank(match: "re.Match[str]") -> str:
        kind, reference = match.lastgroup or "", match.group(0)
        exempt = reference.lower() in located["file"] if kind == "file" else _numbers(reference) <= located[kind]
        return " " if exempt else reference

    return _LOCATOR.sub(blank, text)


def _strip_non_quantities(text: str) -> str:
    """Remove citation labels, list markers, inference-marked sentences and numeric rule identifiers."""

    text = _LIST_MARKER.sub(" ", _without_citations(text, " "))
    kept = []
    for sentence in SENTENCE_BREAK.split(text):
        if not _INFERENCE_MARKER.match(sentence.strip(" \t*_#>")):
            kept.append(sentence)
    # Identifiers split after a dot are not joined here: "under Article B1.6. 3 penalty points" states a 3.
    text = " ".join(kept)
    for start, end, rule in rule_id_spans(text):
        # Purely numeric identifiers ("Article 34.7") look like quantities; letter-prefixed ones ("B1.6.3",
        # "Appendix B2") are never read as numbers. Only the identifier itself is blanked, so the "12" of
        # "Article 12 ... a 12 place grid penalty" is still checked.
        if rule.split()[-1][:1].isdigit():
            text = text[:start] + " " * (end - start) + text[end:]
    return text


def passage_numbers(passages: Sequence[RetrievedPassage]) -> Set[Decimal]:
    """Numbers stated by the passages' text (and their nearest rule heading), not by their metadata."""

    values: Set[Decimal] = set()
    for passage in passages:
        values |= _numbers(passage.text)
        if passage.nearest_rule:
            values |= _numbers(passage.nearest_rule)
    return values


def unsupported_numbers(text: str, cited: Sequence[RetrievedPassage]) -> List[str]:
    """Numbers stated in the answer that do not occur in any cited passage."""

    evidence = passage_numbers(cited)
    stripped = _strip_non_quantities(_without_locators(_normalise(text), cited))
    missing: List[str] = []
    for match in _NUMBER_TOKEN.finditer(stripped):
        values = _token_values(stripped, match)
        if values and not values & evidence and match.group(0) not in missing:
            missing.append(match.group(0))
    return missing


def evidence_rule_ids(passages: Sequence[RetrievedPassage]) -> Set[str]:
    rules: Set[str] = set()
    for passage in passages:
        rules.update(extract_rule_ids(passage.text))
        rules.update(passage.rule_ids)
        if passage.nearest_rule:
            rules.add(passage.nearest_rule)
    return rules


def declined(reason: str, model_output: Optional[str] = None, **details) -> ValidatedAnswer:
    return ValidatedAnswer(status=Outcome.DECLINED, answer=DECLINE_ANSWER, reason=reason, model_output=model_output, **details)


def _folded(text: str) -> str:
    """Lower case, with runs of whitespace, punctuation and emphasis as single spaces."""

    return re.sub(r"[\s.,;:!*_\-]+", " ", text).strip().lower()


_FOLDED_DECLINE = _folded(DECLINE_ANSWER)


def _is_decline(text: str) -> bool:
    if _SENTINEL.search(text):
        return True
    core = _folded(_without_citations(text))
    return bool(_DECLINE_ONLY.match(core)) or _FOLDED_DECLINE in core


def validate_answer(model_output: str, labelled: Dict[str, RetrievedPassage]) -> ValidatedAnswer:
    text = (model_output or "").strip()
    if not text:
        return declined(DeclineReason.EMPTY_OUTPUT, model_output)
    if _is_decline(text):
        return declined(DeclineReason.MODEL_DECLINED, model_output)

    citations = extract_citations(text)
    invalid = [label for label in citations if label not in labelled]
    valid = [label for label in citations if label in labelled]
    rules = extract_rule_ids(_normalise(text))
    # Rule identifiers must be supported by the passages the answer actually cites.
    cited = [labelled[label] for label in valid]
    cited_evidence = evidence_rule_ids(cited)
    cited_items = set().union(*(item_letters(p.text) for p in cited)) if cited else set()
    unsupported = [rule for rule in rules if not is_supported(rule, cited_evidence, cited_items)]
    uncited = uncited_claims(text)
    numbers = unsupported_numbers(text, cited) if cited else []
    details = {
        "citations": valid,
        "invalid_citations": invalid,
        "referenced_rules": rules,
        "unsupported_rules": unsupported,
        "uncited_claims": uncited,
        "unsupported_numbers": numbers,
    }
    if invalid:
        return declined(DeclineReason.INVALID_CITATION, model_output, **details)
    if not valid:
        return declined(DeclineReason.MISSING_CITATION, model_output, **details)
    if unsupported:
        return declined(DeclineReason.UNSUPPORTED_RULE, model_output, **details)
    if uncited:
        return declined(DeclineReason.UNCITED_CLAIM, model_output, **details)
    if numbers:
        return declined(DeclineReason.UNSUPPORTED_NUMBER, model_output, **details)
    return ValidatedAnswer(status=Outcome.ANSWERED, answer=text, **details)
