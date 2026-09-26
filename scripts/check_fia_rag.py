#!/usr/bin/env python3
"""End-to-end evaluation of the FIA regulation RAG system against real documents and models.

Prerequisites:
  1. python scripts/fetch_fia_regulations.py      (official PDFs in data/fia_docs)
  2. python scripts/build_fia_index.py            (embeddings + Qdrant index)
  3. OPENAI_API_KEY (and optionally OPENAI_BASE_URL, model names) in the environment or .env

The run has two phases so retrieval can be inspected independently of generation:
  1. retrieval for every question (one embedding call each, served from the embedding
     cache when repeated), printed with similarity scores for threshold calibration;
  2. grounded generation from the passages that clear the threshold, followed by strict
     checks per category (answerable, paraphrased, cross-document, unanswerable,
     adversarial). Any failed check makes the script exit with status 1.

    python scripts/check_fia_rag.py                    # full evaluation
    python scripts/check_fia_rag.py --retrieval-only   # phase 1 only (no chat-model calls)
    python scripts/check_fia_rag.py --calibrate        # choose FIA_RAG_MIN_SCORE (no chat-model calls)
    python scripts/check_fia_rag.py --ids answerable-minimum-mass trap-drs

``--calibrate`` sweeps the similarity threshold over the retrieval results: for
each answerable question it finds the best-scoring retrieved passage that carries
each expected fact / expected section, and for each unanswerable question the top
score. It reports, per threshold, how many answerable questions keep all their
evidence and how many unanswerable ones are declined before the chat model is
called, and recommends the highest threshold that keeps all answerable evidence.
With a small question set this is a sanity check, not a statistically robust calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from core_modules.rule_checker.fia_rag import RAGUnavailableError, get_fia_rag  # noqa: E402
from core_modules.rule_checker.fia_rag.config import validate_min_score  # noqa: E402
from core_modules.rule_checker.fia_rag.index import close_qdrant_clients  # noqa: E402

QUESTIONS_FILE = Path(__file__).with_name("fia_rag_eval_questions.json")
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "fia_rag_eval.json"
CALIBRATION_REPORT = PROJECT_ROOT / "outputs" / "fia_rag_calibration.json"
ANSWERABLE_CATEGORIES = {"answerable", "paraphrased", "cross_document"}


def _normalise(text: str) -> str:
    text = text.lower().replace("\u202f", " ").replace("\u00a0", " ")
    text = re.sub(r"[\u2010\u2011\u2012\u2043]", "-", text)  # typographic / non-breaking hyphens
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[a-z])- (?=[a-z])", "-", text)  # PDF line break inside a hyphenated word: "stop-\nand-go"
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 215,000,000 -> 215000000


def _contains(answer: str, fact: str) -> bool:
    """Whole-token match: "80" does not match "800", "9" does not match "29"."""

    answer_n, fact_n = _normalise(answer), _normalise(fact)
    for haystack, needle in ((answer_n, fact_n), (answer_n.replace(" ", ""), fact_n.replace(" ", ""))):
        pattern = re.escape(needle)
        if needle[:1].isdigit():
            pattern = r"(?<![\d.])" + pattern  # "80" is not the end of "180"
        elif needle[:1].isalpha():
            pattern = r"(?<![a-z])" + pattern  # word start; plurals ("kerbs") still match
        if needle[-1:].isdigit():
            pattern += r"(?!\d|[.,]\d)"  # "80" is not the start of "800" or "80.5"; units may follow ("726kg")
        if re.search(pattern, haystack):
            return True
    return False


def evaluate(spec: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
    """Return failed checks for one question (empty list = pass)."""

    failures: List[str] = []
    category = spec["category"]
    answer = result["answer"]
    cited = [p for p in result["retrieved_passages"] if p["cited"]]
    labels = {p["label"] for p in result["retrieved_passages"]}
    if any(label not in labels for label in result["citations"]):
        failures.append("a citation label does not map to a retrieved passage")
    if result["validation"]["unsupported_rules"] and result["grounded"]:
        failures.append("grounded answer references rules absent from the evidence")

    if category in ANSWERABLE_CATEGORIES:
        if not result["grounded"]:
            failures.append(f"expected a grounded answer, got decline ({result['decline_reason']})")
            return failures
        cited_sections = {p.get("section") for p in cited}
        for section in spec.get("expected_sections", []):
            if section not in cited_sections:
                failures.append(f"no cited passage from Section {section} (cited sections: {sorted(s for s in cited_sections if s)})")
        for alternatives in spec.get("expected_facts_any", []):
            if not any(_contains(answer, fact) for fact in alternatives):
                failures.append(f"answer lacks expected fact {alternatives}")
            elif not any(_contains(p["text"], fact) for p in cited for fact in alternatives):
                failures.append(f"expected fact {alternatives} is not in the cited passages")
    elif category == "unanswerable":
        if result["grounded"]:
            failures.append("expected a decline, got a grounded answer")
    elif category == "adversarial":
        for fact in spec.get("forbidden_facts", []):
            if result["grounded"] and _contains(answer, fact):
                failures.append(f"grounded answer contains forbidden/invented content {fact!r}")
    return failures


def required_score(spec: Dict[str, Any], passages: List[Any]) -> Optional[float]:
    """Highest threshold at which the retrieved passages still carry all expected evidence.

    Every expected-fact group needs a passage containing one of its alternatives and
    every expected section a passage from that section; the question keeps its
    evidence while each of these best passages clears the threshold. None when some
    evidence was not retrieved at all (a retrieval miss at this top_k).
    """

    needs: List[Optional[float]] = []
    for alternatives in spec.get("expected_facts_any", []):
        scores = [p.score for p in passages if any(_contains(p.text, fact) for fact in alternatives)]
        needs.append(max(scores) if scores else None)
    for section in spec.get("expected_sections", []):
        scores = [p.score for p in passages if p.section == section]
        needs.append(max(scores) if scores else None)
    if not needs or any(need is None for need in needs):
        return None
    return min(needs)  # type: ignore[type-var]


def calibrate(specs: List[Dict[str, Any]], retrievals: Dict[str, Any]) -> Dict[str, Any]:
    """Threshold sweep over retrieval results (see module docstring)."""

    answerable, unanswerable = {}, {}
    for spec in specs:
        ranked = retrievals[spec["id"]].passages + retrievals[spec["id"]].below_threshold
        if spec["category"] in ANSWERABLE_CATEGORIES:
            answerable[spec["id"]] = required_score(spec, ranked)
        elif spec["category"] == "unanswerable":
            unanswerable[spec["id"]] = retrievals[spec["id"]].top_score
    rows = []
    for step in range(101):
        threshold = step / 100
        kept = sum(1 for need in answerable.values() if need is not None and need >= threshold)
        rejected = sum(1 for top in unanswerable.values() if top < threshold)
        rows.append({"threshold": threshold, "answerable_kept": kept, "unanswerable_rejected": rejected})
    reachable = [need for need in answerable.values() if need is not None]
    recommended = None
    if answerable and len(reachable) == len(answerable):
        # Highest threshold (2-decimal grid) that keeps every answerable question's evidence.
        recommended = max(row["threshold"] for row in rows if row["answerable_kept"] == len(answerable))
    return {
        "answerable_required_scores": {k: None if v is None else round(v, 4) for k, v in answerable.items()},
        "retrieval_misses": sorted(k for k, v in answerable.items() if v is None),
        "unanswerable_top_scores": {k: round(v, 4) for k, v in unanswerable.items()},
        "sweep": rows,
        "recommended_min_score": recommended,
        "unanswerable_rejected_at_recommended": None
        if recommended is None
        else sum(1 for top in unanswerable.values() if top < recommended),
        "questions": {"answerable": len(answerable), "unanswerable": len(unanswerable)},
    }


def print_calibration(report: Dict[str, Any], current: float) -> None:
    print("\n=== Threshold calibration (no chat-model calls) ===")
    for question, need in report["answerable_required_scores"].items():
        print(f"  answerable   {question:<36} keeps its evidence up to min_score={need if need is not None else 'MISS (not in top_k)'}")
    for question, top in report["unanswerable_top_scores"].items():
        print(f"  unanswerable {question:<36} top score {top} (declined before generation if min_score > {top})")
    total_a, total_u = report["questions"]["answerable"], report["questions"]["unanswerable"]
    print(f"\n  {'min_score':>9} | answerable kept (of {total_a}) | unanswerable rejected by retrieval (of {total_u})")
    marks = {round(current, 2), report["recommended_min_score"]}
    for row in report["sweep"]:
        if round(row["threshold"] * 100) % 5 == 0 or row["threshold"] in marks:
            note = "  <- current" if row["threshold"] == round(current, 2) else ""
            note += "  <- recommended" if row["threshold"] == report["recommended_min_score"] else ""
            print(f"  {row['threshold']:>9.2f} | {row['answerable_kept']:>28} | {row['unanswerable_rejected']:>40}{note}")
    if report["retrieval_misses"]:
        print(f"\n  Retrieval misses at this top_k (no threshold helps): {report['retrieval_misses']}")
    elif report["recommended_min_score"] is not None:
        print(
            f"\n  Recommended FIA_RAG_MIN_SCORE: {report['recommended_min_score']} (highest value keeping all answerable "
            f"evidence; rejects {report['unanswerable_rejected_at_recommended']} of {total_u} unanswerable questions at "
            "retrieval). Unanswerable questions above it are left to the grounded generator to decline."
        )


def chat_requests(result: Dict[str, Any]) -> int:
    """Chat-model requests behind one answer: the answer call (only with evidence) plus the claim verifier's."""

    if not result["retrieval"]["passages_above_threshold"]:
        return 0
    return 1 + (result["validation"]["claim_verification"] is not None)


def min_score_argument(value: str) -> float:
    """``--threshold``: checked when the arguments are parsed, not after the retrieval phase."""

    try:
        return validate_min_score(float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", help="only run these question ids")
    parser.add_argument("--threshold", type=min_score_argument, help="override FIA_RAG_MIN_SCORE for this run (0 to 1)")
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    parser.add_argument("--calibrate", action="store_true", help="sweep FIA_RAG_MIN_SCORE over the retrieval results (no chat calls)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="where to write the JSON results")
    args = parser.parse_args()

    if os.getenv("F1_COPILOT_LOAD_DOTENV", "1") != "0":
        load_dotenv(PROJECT_ROOT / ".env")  # never overrides variables that are already set
    specs = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    if args.ids:
        specs = [spec for spec in specs if spec["id"] in set(args.ids)]
        if not specs:
            print("No matching question ids")
            return 2

    try:
        rag = get_fia_rag()
        status = rag.status()
        print(json.dumps({"settings": status["settings"], "index": status["index"]["status"], "chunks": status["index"].get("points")}, indent=2))
        if not status["ready"]:
            print("FAIL: RAG is not ready:\n  - " + "\n  - ".join(status["problems"]))
            return 1
        threshold = rag.settings.retrieval.min_score if args.threshold is None else args.threshold

        print(f"\n=== Phase 1: retrieval (top_k={rag.settings.retrieval.top_k}, threshold={threshold}) ===")
        retrievals = {}
        for spec in specs:
            retrieval = rag.retrieve(spec["question"], min_score=0.0)
            retrievals[spec["id"]] = retrieval
            top = retrieval.passages[:3]
            summary = ", ".join(f"{p.section or '?'}:{p.page_label or p.page}={p.score:.3f}" for p in top)
            above = sum(p.score >= threshold for p in retrieval.passages)
            print(f"{spec['category']:>14} | {spec['id']:<36} top={retrieval.top_score:.3f} above={above} | {summary}")
        if args.calibrate:
            report = calibrate(specs, retrievals)
            print_calibration(report, rag.settings.retrieval.min_score)
            CALIBRATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
            CALIBRATION_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nEmbedding provider requests this run: {rag.embedder().provider_requests}")
            print(f"Report written to {CALIBRATION_REPORT}")
            return 0
        if args.retrieval_only:
            print(f"\nEmbedding provider requests this run: {rag.embedder().provider_requests}")
            return 0

        print("\n=== Phase 2: grounded generation and checks ===")
        results, failures_total, chat_calls = [], 0, 0
        for spec in specs:
            retrieval = retrievals[spec["id"]].with_threshold(threshold)
            started = time.monotonic()
            result = rag.answer_from_retrieval(retrieval)
            chat_calls += chat_requests(result)
            failures = evaluate(spec, result)
            verdict = "REVIEW" if spec["category"] == "review" else ("PASS" if not failures else "FAIL")
            failures_total += bool(failures) and verdict != "REVIEW"
            print("\n" + "=" * 100)
            print(f"[{verdict}] {spec['id']} ({spec['category']}, {time.monotonic() - started:.1f}s)")
            print(f"Q: {spec['question']}")
            print(f"A: {result['answer']}")
            print(
                f"grounded={result['grounded']} decline_reason={result['decline_reason']} citations={result['citations']} "
                f"rules={result['referenced_rules']} top_score={result['top_retrieval_score']}"
            )
            for passage in result["retrieved_passages"]:
                marker = "*" if passage["cited"] else " "
                preview = passage["text"].replace("\n", " ")[:160]
                print(
                    f"  {marker}[{passage['label']}] {passage['section'] or '?'} p.{passage['page']} ({passage['page_label']}) "
                    f"rule={passage['nearest_rule']} score={passage['score']:.3f}: {preview}"
                )
            if result["validation"]["finish_reason"]:
                print(f"  finish reason: {result['validation']['finish_reason']}")
            if result["validation"]["rejected_model_output"]:
                print(f"  rejected model output: {result['validation']['rejected_model_output'][:400]!r}")
            for failure in failures:
                print(f"  CHECK FAILED: {failure}")
            results.append({"spec": spec, "verdict": verdict, "failures": failures, "result": result})

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n" + "=" * 100)
        counts: Dict[str, int] = {}
        for item in results:
            counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
        print(f"Summary: {counts}; embedding requests={rag.embedder().provider_requests}, chat requests={chat_calls}")
        print(f"Report written to {args.report}")
        return 1 if failures_total else 0
    except RAGUnavailableError as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        close_qdrant_clients()


if __name__ == "__main__":
    raise SystemExit(main())
