#!/usr/bin/env python3
"""Build (or verify) the Qdrant index for the FIA regulation RAG system.

    python scripts/build_fia_index.py --dry-run   # parse + chunk only, no API calls, shows cost estimate
    python scripts/build_fia_index.py             # embed + index, skipped if an identical index exists
    python scripts/build_fia_index.py --force     # rebuild even if the index is current

The index is only rebuilt when the documents, chunking settings or embedding
model differ from the ones it was built with, so re-running this command does
not re-embed (or pay for) unchanged documents.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from core_modules.rule_checker.fia_rag import RAGUnavailableError, get_fia_rag  # noqa: E402
from core_modules.rule_checker.fia_rag.index import close_qdrant_clients  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild even if the existing index is current")
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk only; do not call the embedding API")
    args = parser.parse_args()

    if os.getenv("F1_COPILOT_LOAD_DOTENV", "1") != "0":
        load_dotenv(PROJECT_ROOT / ".env")  # never overrides variables that are already set
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        rag = get_fia_rag()
        print(json.dumps(rag.settings.summary(), indent=2))
        if args.dry_run:
            print(json.dumps(rag.plan_index(), indent=2))
            return 0

        def progress(done: int, total: int) -> None:
            print(f"  embedded {done}/{total} chunks", flush=True)

        report = rag.build_index(force=args.force, progress=progress)
        print(json.dumps(report.to_dict(), indent=2))
        print(json.dumps(rag.status()["index"], indent=2))
        return 0
    except RAGUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        close_qdrant_clients()


if __name__ == "__main__":
    raise SystemExit(main())
