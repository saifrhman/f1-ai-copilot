# FIA Regulations RAG

This component answers natural-language questions using retrieved evidence from FIA regulation PDFs. It is intentionally split into independent ingestion/indexing, retrieval, and generation stages so chunking, retrieval depth, thresholds, and prompts can be changed and evaluated separately.

## Pipeline

1. `PyPDFLoader` extracts page text and page metadata from each PDF in `FIA_DOCS_PATH`.
2. `RecursiveCharacterTextSplitter` creates overlapping chunks.
3. `OpenAIEmbeddings` generates one vector per chunk.
4. Qdrant stores vectors together with chunk text, source filename, page number, and chunk ID.
5. `retrieve()` embeds the question and returns the top-k Qdrant matches without calling the answer model.
6. `generate_answer()` receives only those passages. It must answer from them, cite source labels such as `[S1]`, preserve regulation article/section numbers, and decline when the retrieved evidence is insufficient.
7. `query()` combines retrieval and generation and returns the answer together with retrieval scores and source metadata for inspection.

## Setup

The repository does not commit FIA PDFs because `data/` is ignored. Add official FIA regulation PDFs locally:

```text
data/
└── fia_docs/
    ├── sporting_regulations.pdf
    └── technical_regulations.pdf
```

Copy `.env.example` to `.env` or export the required variables. At minimum:

```bash
export OPENAI_API_KEY="..."
export FIA_DOCS_PATH="data/fia_docs"
```

Qdrant can run in either mode:

- **Local persistent mode:** leave `QDRANT_URL` empty. The client stores its collection under `QDRANT_PATH` (default `.qdrant`).
- **Qdrant server/cloud:** set `QDRANT_URL` and, when required, `QDRANT_API_KEY`.

## Run the tests

Unit tests do not require an OpenAI key or real FIA PDFs:

```bash
pip install -r requirements-rag.txt
python -m pytest -q tests/test_fia_rag_agent.py
```

The end-to-end smoke test does require the PDFs and an OpenAI key:

```bash
python scripts/check_fia_rag.py
```

It verifies initialization and indexing, top-k retrieval, source/page metadata, grounded answer generation, and source-label citations.

## API

Start the application:

```bash
python app/main.py
```

Inspect RAG configuration/status:

```bash
curl http://localhost:8000/api/fia/status
```

Query the regulations:

```bash
curl -X POST http://localhost:8000/api/fia/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What do the regulations say about an unsafe release?"}'
```

The response includes:

- `answer`
- `grounded`
- `top_retrieval_score`
- `retrieved_passages` with source, page, score, text, and chunk ID
- `referenced_rules`
- `citations`

## Main tuning controls

- `FIA_RAG_CHUNK_SIZE`
- `FIA_RAG_CHUNK_OVERLAP`
- `FIA_RAG_TOP_K`
- `FIA_RAG_MIN_SCORE`
- `FIA_RAG_EMBEDDING_MODEL`
- `FIA_RAG_MODEL`

Because retrieval is exposed through `retrieve_fia_passages()`, retrieval quality can be evaluated without generation. This is useful for testing different chunk sizes, overlaps, top-k values, embedding models, and score thresholds before changing the prompting layer.
