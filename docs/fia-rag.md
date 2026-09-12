# FIA Regulations RAG

This component answers natural-language questions using retrieved evidence from FIA Formula 1 regulation PDFs. Ingestion/indexing, retrieval and answer generation are deliberately separate so chunking, retrieval depth, thresholds and prompts can be inspected and changed independently.

## Pipeline

1. `PyPDFLoader` extracts page text and page metadata from each PDF in `FIA_DOCS_PATH`.
2. `RecursiveCharacterTextSplitter` creates overlapping chunks.
3. `OpenAIEmbeddings` generates one vector per chunk.
4. Qdrant stores each vector with chunk text, source filename, page number and chunk ID.
5. `retrieve()` embeds a question and returns top-k Qdrant matches without calling the answer model.
6. `generate_answer()` receives only the retrieved passages. The prompt requires source labels such as `[S1]`, exact article/section references from the evidence and refusal when evidence is insufficient.
7. `query()` combines retrieval and generation and exposes the answer, grounding state, evidence-strength proxy, retrieval score and retrieved source metadata.

## Obtain the FIA documents

FIA Publications are third-party copyrighted material and are not redistributed by this repository. `data/` is git-ignored.

For local/private use, discover and download the current official 2026 Formula 1 regulation Sections A-F directly from FIA:

```bash
python scripts/fetch_fia_regulations.py
```

Files are written to:

```text
data/fia_docs/
```

The downloader also writes `manifest.json` containing the official category URL, source URL, final download URL, byte count and SHA-256 digest for every downloaded PDF.

To check the official FIA links without downloading the documents:

```bash
python scripts/fetch_fia_regulations.py --dry-run
```

The repository CI performs this discovery check so an FIA page-layout change becomes visible.

## Configure the RAG system

Copy `.env.example` to `.env` or export the settings. At minimum:

```bash
export OPENAI_API_KEY="..."
export FIA_DOCS_PATH="data/fia_docs"
```

Qdrant can run in either mode:

- **Local persistent mode:** leave `QDRANT_URL` empty; data is stored under `QDRANT_PATH` (default `.qdrant`).
- **Qdrant server/cloud:** set `QDRANT_URL` and, if required, `QDRANT_API_KEY`.

## Tests

Unit tests do not require an OpenAI key or real FIA PDFs:

```bash
pip install -r requirements-rag.txt
python -m pytest -q tests/test_fia_rag_agent.py
```

The real external smoke test requires downloaded FIA PDFs and an OpenAI key:

```bash
python scripts/check_fia_rag.py
```

It exercises PDF parsing, indexing, question embedding, top-k retrieval, page/source metadata and grounded answer generation.

## API

Start the application:

```bash
python app/main.py
```

Inspect RAG status:

```bash
curl http://localhost:8000/api/fia/status
```

Query the regulations:

```bash
curl -X POST http://localhost:8000/api/fia/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What do the regulations say about an unsafe release?"}'
```

A successful detailed response includes:

- `answer`
- `grounded`
- `confidence` (an evidence-strength proxy from the top retrieval similarity, not a calibrated correctness probability)
- `top_retrieval_score`
- `retrieved_passages` with source, page, score, text and chunk ID
- `referenced_rules`
- `citations`

When the API key, documents or external dependencies are unavailable, `/api/fia/query` returns HTTP 503 instead of silently returning a mock answer.

## Tuning controls

- `FIA_RAG_CHUNK_SIZE`
- `FIA_RAG_CHUNK_OVERLAP`
- `FIA_RAG_TOP_K`
- `FIA_RAG_MIN_SCORE`
- `FIA_RAG_EMBEDDING_MODEL`
- `FIA_RAG_MODEL`

Because retrieval is exposed independently through `retrieve_fia_passages()`, retrieval quality can be evaluated without answer generation. This makes it possible to compare chunk sizes, overlaps, top-k values, embedding models and score thresholds before changing the generation prompt.
