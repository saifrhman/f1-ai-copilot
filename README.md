# F1 AI Copilot

F1 AI Copilot is a modular Formula 1 analysis project with a FastAPI interface. It combines a regulation RAG system with heuristic race-strategy analysis, setup search, telemetry comparison, natural-language routing, driver-radio audio features, and incident triage.

The repository is intentionally explicit about what is evidence-grounded and what is heuristic. It is a research/demo project, not an FIA steward tool, a validated vehicle-dynamics simulator, or a production race-engineering system.

## Implemented modules

| Module | What it does | Method / scope |
| --- | --- | --- |
| FIA regulation QA | Answers natural-language questions from locally downloaded official FIA F1 regulation PDFs | PDF ingestion → chunking → OpenAI embeddings → Qdrant top-k retrieval → evidence threshold → grounded answer with source labels |
| Strategy engine | Generates and ranks multi-stint candidate strategies from race state, tyre data, telemetry, competitors and car condition | Transparent heuristic simulation; complete stint accounting, weather and damage/wear adjustments |
| Setup recommender | Searches setup parameters for track/weather/driver targets | Reproducible Optuna TPE search over a documented heuristic objective |
| Ghost comparison | Compares two telemetry traces and produces a visualization artifact | Validated time-series comparison using supplied x/y, speed, braking and DRS data |
| Driver-radio emotion | Extracts acoustic features and returns a coarse emotion similarity label | Librosa acoustic heuristic; optional local OpenAI Whisper transcription, never a mock transcript |
| Natural-query router | Routes questions to regulatory, performance, setup, strategy or emotion modules | Keyword routing with evidence requirements; it declines when required context is missing |
| Incident triage | Gives a preliminary severity/outcome category for an incident | Transparent deterministic heuristic; does **not** invent FIA article numbers or claim to predict steward decisions |

## FIA RAG architecture

The regulation component keeps retrieval and generation separate so they can be tested and tuned independently:

```text
Official FIA PDFs (local files)
        ↓
PyPDFLoader
        ↓
Recursive text chunking
        ↓
OpenAI embeddings
        ↓
Qdrant vector collection
        ↓
Top-k retrieval + score threshold
        ↓
Retrieved source/page passages
        ↓
Grounded answer generation
```

Answers are instructed to use only the retrieved passages, retain article/section numbers exactly as found in the evidence, add `[S1]`, `[S2]`, etc. source labels, and decline when the retrieved evidence is insufficient.

## FIA regulation documents

The FIA PDFs are **not committed to this repository**. FIA's website terms state that copyright in FIA Publications, including championship technical and sporting regulations, belongs to FIA and that reproduction/distribution requires prior written consent, while private non-commercial copies are permitted under the stated terms.

For local/private use, fetch the current official 2026 Formula 1 regulation PDFs directly from FIA:

```bash
python scripts/fetch_fia_regulations.py
```

This downloads the current Sections A-F into:

```text
data/fia_docs/
```

and writes `data/fia_docs/manifest.json` with the official source URL, final download URL, file size and SHA-256 hash. The whole `data/` directory is git-ignored.

To inspect the official links without downloading files:

```bash
python scripts/fetch_fia_regulations.py --dry-run
```

If the FIA changes its website layout, the live CI source-discovery check will make that visible.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/saifrhman/f1-ai-copilot.git
cd f1-ai-copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment configuration:

```bash
cp .env.example .env
```

For the FIA RAG component, set an OpenAI API key:

```bash
export OPENAI_API_KEY="..."
```

By default Qdrant runs in local persistent mode under `.qdrant`. To use a Qdrant server instead:

```bash
export QDRANT_URL="http://localhost:6333"
```

Optional local Whisper transcription requires the additional package and `ffmpeg`:

```bash
pip install -r requirements-whisper.txt
```

On Ubuntu/Debian:

```bash
sudo apt-get install ffmpeg
```

The acoustic emotion classifier itself does not require Whisper.

## Run the API

```bash
python app/main.py
```

or:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

OpenAPI/Swagger documentation is available at:

```text
http://localhost:8000/docs
```

Generated ghost-comparison images are served under `/artifacts/...`.

## Main endpoints

### Health

```http
GET /health
```

The service reports `degraded` when optional/configured components such as FIA RAG are not ready instead of pretending they are healthy.

### FIA regulation QA

```http
POST /api/fia/query
Content-Type: application/json

{
  "question": "What does the current regulation say about an unsafe release?"
}
```

The detailed response contains the answer, grounding state, retrieval score, retrieved passages with source/page metadata, referenced rule strings and source labels.

RAG setup/index status:

```http
GET /api/fia/status
```

### Race strategy

```http
POST /api/strategy/generate
```

Inputs include telemetry, car status, driver profile, per-compound tyre data, race state and competitors. The engine returns ranked candidates with every stint, compound sequence, pit laps, estimated remaining-race time, opportunities, risk label and explicit heuristic notes.

### Setup recommendation

```http
POST /api/setup/recommend
```

The response includes setup values, confidence, reasoning, optimization method, trial count and heuristic objective value.

### Telemetry / ghost comparison

```http
POST /api/ghost/generate
```

Telemetry timestamps must be non-empty and strictly increasing. Parallel telemetry arrays must have matching sample counts. The endpoint returns comparison arrays plus a served PNG artifact.

### Driver-radio emotion

```http
POST /api/emotion/classify
```

`audio_file` may be an existing local path or base64/data-URI audio. Set `transcribe: true` to request Whisper transcription when the optional dependency is installed. Invalid audio is rejected instead of being replaced with mock data.

### Natural query

```http
POST /api/query/natural
```

The router only makes module-specific claims when the required context is present. Regulatory questions route to FIA RAG.

### Incident triage

```http
POST /api/penalty/predict
```

This endpoint returns a severity/outcome review category. It is deliberately labelled as heuristic triage and does not claim a specific FIA sanction or article without retrieved evidence.

## Verification

Run the complete automated test set:

```bash
python -m pytest -q tests/test_fia_rag_agent.py tests/test_project_modules.py
```

The tests cover:

- independent RAG retrieval and generation behavior
- RAG evidence threshold, citations and refusal behavior
- strategy string-enum API conversion and complete stint accounting
- reproducible/bounded Optuna setup search
- ghost telemetry validation and artifact creation
- real audio feature extraction from a generated WAV
- natural-query evidence use
- incident triage not fabricating FIA article numbers
- FIA regulation-link discovery
- health/status behavior

GitHub Actions runs both the module/API suite and a live discovery check against the official FIA regulations page.

For a real FIA RAG smoke test after downloading the regulations and setting `OPENAI_API_KEY`:

```bash
python scripts/check_fia_rag.py
```

That smoke test exercises the external stages that unit tests cannot reproduce without credentials: PDF parsing, embedding API calls, Qdrant indexing/retrieval, and answer generation.

## Configuration

Useful FIA RAG variables:

```text
OPENAI_API_KEY
FIA_DOCS_PATH=data/fia_docs
FIA_RAG_COLLECTION=fia_regulations
FIA_RAG_TOP_K=5
FIA_RAG_MIN_SCORE=0.30
FIA_RAG_CHUNK_SIZE=1000
FIA_RAG_CHUNK_OVERLAP=200
FIA_RAG_EMBEDDING_MODEL=text-embedding-3-small
FIA_RAG_MODEL=gpt-4o-mini
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_PATH=.qdrant
CORS_ORIGINS=*
```

Chunk size, overlap, retrieval depth and prompting remain separate from the answer-generation step so they can be changed independently during experiments.

## Important limitations

The strategy and setup modules use transparent heuristic models, not proprietary F1 team models or validated vehicle simulation. Emotion labels are coarse acoustic/text similarity labels and should not be treated as psychological assessment. Incident triage is not an FIA decision predictor. FIA regulation QA is only as current as the PDFs downloaded into `data/fia_docs`, and retrieved evidence should be checked against the official source for high-stakes interpretation.

## Repository structure

```text
app/
  main.py                         FastAPI application
core_modules/
  strategy_optimizer/
  rule_checker/
  llm_query/
  driver_emotion/
  ghost_car/
  setup_optimizer/
scripts/
  fetch_fia_regulations.py       official-source local downloader
  check_fia_rag.py               real RAG smoke test
tests/
  test_fia_rag_agent.py
  test_project_modules.py
docs/
  fia-rag.md
```

## License

Project source code is provided under the repository's project license where applicable. FIA Publications downloaded by the helper script are third-party copyrighted material and are not covered by the project license or redistributed by this repository.
