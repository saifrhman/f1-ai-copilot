#!/usr/bin/env python3
"""End-to-end smoke test for the FIA regulation RAG component.

Prerequisites:
  1. Put official FIA regulation PDFs in data/fia_docs (or set FIA_DOCS_PATH).
  2. Set OPENAI_API_KEY.
  3. Optionally set QDRANT_URL, otherwise local Qdrant storage is used.
"""

import json
import sys

from core_modules.rule_checker.fia_rag_agent import get_fia_knowledge_base


QUESTIONS = [
    "What do the regulations say about an unsafe release?",
    "What do the regulations say about track limits?",
    "What conditions govern use of DRS?",
]


def main() -> int:
    kb = get_fia_knowledge_base()
    print("Initial status:")
    print(json.dumps(kb.status(), indent=2))

    try:
        kb.initialize()
    except Exception as exc:
        print(f"\nFAIL: FIA RAG could not initialize: {exc}")
        return 1

    print("\nIndexed status:")
    print(json.dumps(kb.status(), indent=2))

    failures = 0
    for question in QUESTIONS:
        result = kb.query(question)
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("ANSWER:", result["answer"])
        print("TOP SCORE:", result["top_retrieval_score"])
        print("CITATIONS:", result["citations"])
        print("RULES:", result["referenced_rules"])
        print("EVIDENCE:")
        for passage in result["retrieved_passages"]:
            preview = passage["text"].replace("\n", " ")[:220]
            print(
                f"  - {passage['source']} p.{passage['page']} "
                f"score={passage['score']:.3f}: {preview}"
            )

        if not result["retrieved_passages"]:
            print("FAIL: no passages retrieved")
            failures += 1
        elif result["grounded"] and not result["citations"]:
            print("FAIL: grounded answer did not include source labels")
            failures += 1

    if failures:
        print(f"\nFAIL: {failures} smoke-test checks failed")
        return 1

    print("\nPASS: FIA RAG ingestion, retrieval, and grounded generation are operational")
    return 0


if __name__ == "__main__":
    sys.exit(main())
