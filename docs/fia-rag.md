# FIA Regulations RAG

Question answering over the official FIA Formula 1 Regulations (2026, Sections A–F). Answers are generated only from retrieved regulation passages (plus the official definitions of the defined terms those passages use), cite them with `[S#]` labels, and are **validated after generation**. An answer is replaced by a decline when it:

* cites a passage that was not supplied (in any citation style), or cites nothing;
* leaves a statement uncited;
* names an article, section or appendix identifier that does not occur in the passages it cites;
* states a number that does not occur in the passages it cites;
* was cut off by the model's output limit;
* is flagged by the optional model-based claim verifier.

The system says it cannot answer rather than presenting unsupported text as a regulation.

## Pipeline and where it is implemented

All code is in `core_modules/rule_checker/fia_rag/`, except the official file-name conventions (section, year and issue of a file name, `manifest.json`), which `core_modules/rule_checker/fia_files.py` shares with the downloader. Each stage lives in its own module and receives only its own settings object, so chunking, retrieval depth and prompting can be changed independently.

| Stage | Implementation | Settings |
| --- | --- | --- |
| Official PDFs | `scripts/fetch_fia_regulations.py` (`discover_candidates`, `latest_candidates`, `fetch_regulations`, `validate_pdf_bytes`) | `--year`, `--sections`, `--output` (default `FIA_DOCS_PATH`) |
| Document validation | `ingestion.discover_documents` – missing/unreadable directory or file, non-PDF (e.g. an HTML error page), empty file, SHA-256 mismatch with `manifest.json`, listed files missing, two issues of the same FIA section, byte-identical duplicates | `FIA_DOCS_PATH` |
| PDF parsing | `ingestion.load_pages` – pypdf, one LangChain `Document` per page; strips the FIA running header, keeps the printed page label (`B10`), repairs two broken ligature glyphs, skips table-of-contents pages, reports blank pages; unreadable PDFs or pages raise `DocumentError` | – |
| Chunking | `ingestion.chunk_pages` – LangChain `RecursiveCharacterTextSplitter`; metadata: source, 1-based page, page label, section, source URL, `nearest_rule` (last numbered heading at or before the chunk start, carried across pages; cross-references and table cells are not headings), `rule_ids`, deterministic `chunk_id` | `ChunkingConfig`: `FIA_RAG_CHUNK_SIZE`, `FIA_RAG_CHUNK_OVERLAP` |
| Embeddings | `embeddings.EmbeddingService` around LangChain `OpenAIEmbeddings` (OpenAI or any OpenAI-compatible endpoint); validates vector count, finiteness and dimensions; optional `EmbeddingCache`. Each chunk is embedded with a one-line context header (document title and `nearest_rule`); the stored and displayed text stays verbatim | `EmbeddingConfig`: `FIA_RAG_EMBEDDING_MODEL`, `FIA_RAG_EMBEDDING_BATCH_SIZE`, `FIA_RAG_EMBEDDING_CACHE` |
| Vector index | `index.QdrantIndex` – cosine collection, payload = chunk text + metadata + build fingerprint | `QdrantConfig`: `QDRANT_URL`/`QDRANT_API_KEY` or `QDRANT_PATH`, `FIA_RAG_COLLECTION` |
| Definitions glossary | `glossary.extract_glossary` – verbatim definitions (`"Total Time Classified Session" (or "TTCS") is ...`) extracted from the PDF pages at index time and stored in a second Qdrant collection `<collection>_glossary` (no embeddings); `glossary.definitions_for` selects definitions at question time | – |
| Retrieval | `retrieval.Retriever.retrieve` – embeds the question, top-k Qdrant search, exact-duplicate removal, per-passage similarity threshold; the pipeline then attaches the definitions of the abbreviations the accepted passages use and of terms named in the question | `RetrievalConfig`: `FIA_RAG_TOP_K`, `FIA_RAG_MIN_SCORE`, `FIA_RAG_MAX_DEFINITIONS` |
| Context + generation | `generation.build_messages` / `GroundedAnswerGenerator` – system rules + labelled, HTML-escaped `<excerpt>` blocks (definitions marked `kind="definition"`) + escaped question; LangChain `ChatOpenAI`; truncated output (`finish_reason` `length`/`content_filter`/`max_tokens`) is declined; optional claim verifier | `GenerationConfig`: `FIA_RAG_MODEL`, `FIA_RAG_TEMPERATURE`, `FIA_RAG_MAX_OUTPUT_TOKENS`, `FIA_RAG_VERIFY_CLAIMS` |
| Citation / rule / number validation | `grounding.validate_answer` (identifier grammar in `rules.py`) | – |
| Orchestration, status | `pipeline.FIARegulationRAG` (`build_index`, `retrieve`, `answer`, `answer_from_retrieval`, `status`) | `RAGSettings.from_env()` |
| API | `app/main.py`: `GET /api/fia/status`, `POST /api/fia/query`, `POST /api/fia/retrieve` | – |

What each setting affects:

* Chunking settings, the embedding model, the documents and their manifest metadata (section, source URL) and the ingestion code version form the **index fingerprint**. Changing any of them marks the index `stale`; queries are refused (HTTP 503) until `scripts/build_fia_index.py` rebuilds it. Generation code is not involved.
* `FIA_RAG_TOP_K` and `FIA_RAG_MIN_SCORE` affect retrieval only. They can also be set per request (`top_k` on `/api/fia/query`; `top_k` and `min_score` on `/api/fia/retrieve`). No re-indexing is needed.
* `FIA_RAG_MAX_DEFINITIONS` (default 3; 0 disables) sets how many definitions are added to the evidence. The glossary is built together with the index and stored under the index fingerprint plus a glossary version, so it can never describe other documents than the chunks.
* The prompt, the chat model and `FIA_RAG_VERIFY_CLAIMS` affect generation only. They do not change the index fingerprint.

`tests/test_fia_generation.py::test_generation_settings_and_top_k_do_not_affect_the_index`, `tests/test_fia_index_retrieval.py::test_changed_chunking_makes_index_stale_until_rebuilt` and `tests/test_fia_hardening.py::test_each_chunking_field_alone_makes_the_index_stale` check these properties.

## 1. Download the regulations

The FIA PDFs are third-party copyrighted material: the FIA keeps their copyright. The repository includes unmodified copies of the issues listed in `data/fia_docs/manifest.json`, credited to the FIA in the README's [Acknowledgements](../README.md#acknowledgements). To replace them with the newest official issues:

```bash
python scripts/fetch_fia_regulations.py --dry-run   # show what would be downloaded
python scripts/fetch_fia_regulations.py             # download into FIA_DOCS_PATH (default data/fia_docs)
```

The FIA category page lists every historical issue of every section; the downloader picks the newest issue per section from the issue number and date in the file name (not from the page order), and only accepts files whose name starts with the requested regulation year (a 2027 document issued in 2026 is not taken for a 2026 regulation). Every download must start with `%PDF-`, end with `%%EOF` and parse with pypdf. Downloads are staged and only replace the existing set when all of them succeeded. A previous file is removed only when a newer issue of the same section and year replaced it; sections not requested in this run (`--sections A`) are kept and stay in the manifest. If the previous `manifest.json` is unreadable, previous downloads are recognised by their official file names. Files that are not official FIA regulation files are never removed.

`manifest.json` records the section, issue, issue date, official source URL, final download URL, byte count, page count and SHA-256 of each file. The RAG refuses to use the directory when a file no longer matches its SHA-256, when a listed file is missing, or when a second issue of a listed FIA section is present, so issues are never mixed silently.

## 2. Configure a model provider

Copy `.env.example` to `.env` (git-ignored) or export the variables. Real environment variables take precedence over `.env`.

OpenAI:

```bash
OPENAI_API_KEY=sk-...
FIA_RAG_EMBEDDING_MODEL=text-embedding-3-small
FIA_RAG_MODEL=gpt-4o-mini
```

Any OpenAI-compatible endpoint, for example OpenRouter:

```bash
OPENAI_API_KEY=sk-or-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
FIA_RAG_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b:free
FIA_RAG_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
FIA_RAG_EMBEDDING_BATCH_SIZE=256
```

Without a key the RAG endpoints return HTTP 503; nothing falls back to fake embeddings or canned answers. Provider errors are reported without credentials (HTTP 401/403 bodies are not echoed, key-like strings are masked).

Qdrant runs in one of two modes:

* **Embedded (default):** `QDRANT_PATH` (default `.qdrant`). Embedded storage can be opened by **one process at a time**: stop the API before running `build_fia_index.py`, or use a server.
* **Server / Qdrant Cloud:** set `QDRANT_URL` (and `QDRANT_API_KEY`). Required for several API workers or processes; see [Qdrant server mode](#qdrant-server-mode-several-processes).

## 3. Build the index

```bash
python scripts/build_fia_index.py --dry-run   # parse + chunk only: chunk count, token estimate, request count
python scripts/build_fia_index.py             # embed + index (skipped when an identical index exists)
python scripts/build_fia_index.py --force     # rebuild anyway
```

For the six 2026 documents (592 pages; 19 contents pages and 8 blank pages skipped) the default 1000/200 chunking produces 1,938 chunks, about 431k embedding tokens (including the one-line context header per chunk) and 8 embedding requests at a batch size of 256 (16 at the default 128). Parsing takes about a minute because pypdf is slow on the 254-page technical regulations.

The build also extracts the definitions glossary (541 definitions from the six 2026 documents, 34 of them with an abbreviation) from the same parsed pages; this needs no embedding requests. When the chunks are current but the glossary is missing or was built by an older extractor version, `build_fia_index.py` rebuilds only the glossary (`"status": "glossary_rebuilt"`).

Every point stores the build fingerprint and the number of chunks written. The index is used only when both match. These states are reported by `/api/fia/status` and must be rebuilt:
* an interrupted build (`incomplete`);
* new documents or settings (`stale`);
* a missing or empty collection.

A rebuild writes a complete new collection and then switches the index over to it, so repeated ingestion never duplicates chunks. Embeddings are computed before anything is written, so a provider failure leaves the previous index intact.

The embedding cache (`.cache/fia_embeddings.sqlite3`; `FIA_RAG_EMBEDDING_CACHE=off` disables it) stores provider vectors, as float32 (the precision the OpenAI SDK returns), keyed by model name and the SHA-256 of the exact text. Rebuilding after a metadata-only change, moving the index to another Qdrant, or repeating a question therefore makes no new embedding request. Question vectors live in a separate table capped at 5,000 rows. A damaged row is treated as a miss; if the cache file cannot be used the RAG runs uncached and `/api/fia/status` reports `embedding_cache_problem`.

## Qdrant server mode (several processes)

Embedded storage allows one process at a time. While the API has it open, `python scripts/build_fia_index.py` exits with `Could not open Qdrant (...): Storage folder ... is already accessed by another instance of Qdrant client ...`. With `uvicorn --workers N` (N > 1), only the worker that opens the storage first can serve the FIA endpoints; the others return 503. Use the Qdrant server in `docker-compose.yml` when more than one process needs the index:

```bash
docker compose up -d --wait qdrant        # qdrant/qdrant:v1.19.0 on 127.0.0.1:6333, data in a named volume
export QDRANT_URL=http://127.0.0.1:6333   # or set it in .env
python scripts/build_fia_index.py         # the server starts empty: build the index once
uvicorn app.main:app --workers 2
docker compose down                       # stop; `docker compose down -v` also deletes the stored index
```

**API key (optional).** Set `QDRANT_API_KEY` in the shell or in `.env`. Compose passes it to the server as `QDRANT__SERVICE__API_KEY`, and the app sends the same variable. Generate the key without `$` (`openssl rand -hex 32`), or single-quote it in `.env`. The reason: Compose expands `$name` in unquoted and double-quoted `.env` values, while python-dotenv (used by the app) does not. So `QDRANT_API_KEY=abc$def` reaches the server as `abc`, and every request fails with `401 (Unauthorized)`. Without a key the server accepts unauthenticated requests; the port is only published on 127.0.0.1.

**Rebuilding while the API is running.** `FIA_RAG_COLLECTION` (and `<collection>_glossary`) name Qdrant *aliases*. Each rebuild:
1. writes a new collection `<name>__<12 hex>`;
2. checks the stored point count;
3. points the alias at the new collection in one atomic `update_collection_aliases` call;
4. deletes the old collection.

The glossary is switched first, so readers never see the new chunks without their glossary. Running API workers answer from the complete old index until the switch and from the complete new one after it.

A load test (2 API workers against a Qdrant server, 6 concurrent clients, 5 forced rebuilds) got HTTP 200 with the correct passage on all 2,941 requests. `tests/test_fia_index_retrieval.py` checks the swap with a reader that queries before every write.

A rebuild that fails or is interrupted with Ctrl+C deletes its unfinished collection, and the old index keeps serving. Limits:
* A killed process (SIGKILL) can leave an unused `<name>__<hex>` collection behind; delete it by hand.
* Do not run two rebuilds at once.
* The first rebuild after upgrading from a version without aliases replaces the old plain collection. Requests in the brief moment between the delete and the alias creation get 503.
* Changed PDFs still make the API return 503 (index stale) until the rebuild switches over.

## 4. Ask questions

```bash
uvicorn app.main:app
curl http://localhost:8000/api/fia/status
curl -X POST http://localhost:8000/api/fia/query -H 'Content-Type: application/json' \
  -d '{"question": "What is the pit lane speed limit?"}'
curl -X POST http://localhost:8000/api/fia/retrieve -H 'Content-Type: application/json' \
  -d '{"question": "What is the pit lane speed limit?", "top_k": 3}'
```

`/api/fia/retrieve` returns ranked passages with similarity scores, and the definitions that would be added, without calling the answer model.

`/api/fia/query` returns:

| Field | Meaning |
| --- | --- |
| `answer` | Validated answer with `[S#]` labels, or the standard decline message |
| `grounded` | `true` only if the answer passed validation |
| `status` / `decline_reason` | `answered`, or `declined` with `no_evidence_above_threshold`, `model_declined`, `empty_model_output`, `invalid_citation`, `missing_citation`, `unsupported_rule_reference`, `uncited_claim`, `unsupported_number`, `truncated_model_output`, `unverified_claim` |
| `citations` | Labels used by the answer; every label is a `retrieved_passages[].label` |
| `citation_spans` | Each citation as written in `answer`, in text order: `start`/`end` offsets into `answer` (Unicode code points) and the `labels` it names, ranges expanded (`[S1-S3]` → S1, S2, S3). A prose citation ("source S2") spans only its label. Clients mark citations from these instead of parsing the answer again (the web UI turns them into badges). Empty for a decline |
| `retrieved_passages` | The evidence given to the model: `label`, `cited`, `kind` (`regulation` for retrieved passages, `definition` for added definitions, which are labelled after the passages), `defined_term`, `text`, `score` (0 for definitions), `source`, `page`, `page_label`, `section`, `source_url`, `nearest_rule`, `rule_ids`, `chunk_id` |
| `referenced_rules` | Rule identifiers in the answer, all found in the passages it cites |
| `confidence` | Best similarity among cited *regulation* passages – an evidence-strength proxy, not a probability of correctness (definitions are not retrieved by similarity and do not count) |
| `retrieval` | `top_k`, `min_score`, passages above the threshold, definitions added, passages below the threshold (never shown to the model), duplicates removed |
| `validation` | `invalid_citations`, `unsupported_rules`, `uncited_claims`, `unsupported_numbers`, `unverified_claims`, `claim_verification` (the verifier's report when enabled), the rejected model output when an answer was declined, and `finish_reason` (`length`, `max_tokens` or `content_filter`) when the reply was cut off |

HTTP 503 means the RAG cannot answer (missing key or documents, index missing/stale/incomplete, Qdrant or provider failure); HTTP 422 means an invalid request.

`/api/fia/status` reports `ready`, `problems`, the index state (`current`, `missing`, `empty`, `stale`, `incomplete`, `unknown` when the documents cannot be read, `unavailable` when Qdrant fails), the glossary state (`index.glossary`: `current` with its entry count, or `missing`, which blocks queries while `FIA_RAG_MAX_DEFINITIONS` > 0), `provider_status` (from the latest regulation request, not a live check: `ok` once one has succeeded, including a search answered entirely from cached embeddings, which makes no provider call; `failing` after a failed model-provider call with no later success; `unknown` until a request has succeeded) and a top-level `state` (`ready`, `not_configured`, `misconfigured`, `index_missing`/`index_empty`/`index_stale`/`index_incomplete`, `provider_failing`, `unavailable`). Paths are shown relative to the project; URLs without credentials.

## How grounding is enforced

1. **Threshold before generation.** Passages below `FIA_RAG_MIN_SCORE` (cosine similarity; higher is more similar) are not shown to the model. If none remain, the answer is declined without calling the model; definitions alone never count as evidence.
2. **Definitions of defined terms.** The regulations define terms once (Appendix B1, Article A1, ...) and then use them, often as abbreviations: a passage says "during a TTCS, a Stop-and-Go Penalty will be imposed", while only the definition says that TTCS include the Race. For every abbreviation used in an accepted passage (case-sensitive, whole word), and every defined term named in the question (multi-word terms in any case, single words with exact case), the verbatim official definition is added as an extra excerpt. Where a term is defined in several sections the passage's own section wins. Abbreviations used in more than 20% of all chunks ("FIA": 41%) are general vocabulary and skipped. So are purely referential definitions ("has the meaning set out in D4.1.2") and definitions a passage already contains.
3. **Evidence-only prompt.** The system message requires evidence-only answers, `[S#]` labels on every factual sentence, article numbers copied exactly, explicitly marked inferences, and the `INSUFFICIENT_EVIDENCE` reply when nothing relevant is present. It also says to cite a definition excerpt when the answer relies on what a term covers. Excerpts and the question are delimited, HTML-escaped and declared to be data, not instructions.
4. **Deterministic validation** (`grounding.validate_answer`), in this order:
   * a reply that is only a decline (the sentinel, "insufficient evidence", or the decline message – with or without citations) is a decline;
   * every citation-like token – `[S2]`, `[s2]`, `[S1, S3]`, `[S1-S3]`, `[2]`, `(S2)`, `[source: S2]`, full-width `【S2】`, "excerpt 2" – must name a supplied passage;
   * at least one supplied passage must be cited;
   * every rule identifier must occur in the passages the answer cites. The grammar in `rules.py` covers:
     * `Article`/`Art.`/`Rule`/`Clause`/`Paragraph`/`Section`/`Regulation` forms, with or without a dot or space;
     * enumerations (`Articles 12.2, 12.4 and 12.5`);
     * bare identifiers with any letter and case (`B1.6.3`, `b9.9.9`, `Z1.1`);
     * appendices by number, letter or roman numeral (`Appendix B2`, `Appendix H`);
     * numeric references into named documents (`34.7 of the Sporting Regulations`).

     `Appendix B2` is not `Article B2`. A parent of an evidence rule (`B2.3` for `B2.3.5`) is accepted, and so is a lettered item that exists in the cited text (`B1.6.3a` when the passage lists `a.`). An invented child (`B2.3.5.1`, `B1.6.3.1000`) or item (`B1.6.3z`) is not;
   * **no uncited statements:** a citation covers its paragraph up to the citation, and a final paragraph consisting only of citations (a sources block such as `[S1]` or `Sources: [S1], [S2]`) covers the whole answer. Any other text must not make a statement. Exceptions: sentences that start with an inference marker ("Thus", "Therefore", ...), headings, list markers, list introductions ending in `:`, and remarks about the evidence itself ("The excerpts do not specify the fine.");
   * **every number must occur in the cited passages:** limits, penalties, amounts, times and counts. Values are compared numerically, so `80 km/h` matches `80km/h`, `0.3` matches `0.30`, `1,000` matches `1000`, `100` matches "one hundred", and `US$135 million` matches `135,000,000`. List markers, citation labels, rule identifiers and inference-marked sentences are exempt, and so are references to where a cited passage is, each compared with its own kind of metadata: `page 9` and `printed page B9` with the passage's page and page label, `Issue 08` (capital I) with the issue number in its file name, its exact file name, and `the 2026 Sporting Regulations` with the regulation year of its file name. The metadata is not evidence otherwise: a `5-place grid drop` or "the stewards may issue 5 penalty points" cited to a file dated `2026-08-05` is declined, and so is `(page 8)` cited to page 9 of Issue 08. A rule identifier split after a dot is not joined for this check, so the `3` of "under Article B1.6. 3 penalty points" is checked. A number the passages do not state – for example a changed limit, or a number repeated from the question – declines the answer (`unsupported_number`).
5. **Optional claim verifier** (`FIA_RAG_VERIFY_CLAIMS=true`, off by default). An answer that passed step 4 is split into sentences and sent, together with only the excerpts it cites, to a second model call. That call returns the sentences the excerpts do not support, as JSON. The answer is declined (`unverified_claim`, flagged sentences listed) when:
   * any sentence is flagged;
   * the verifier's reply is not the requested JSON;
   * the verifier's reply was truncated.

   This doubles the chat requests per answered question.

Steps 1–4 establish citation, identifier and number integrity deterministically. They cannot prove that a cited passage *entails* a sentence without numbers: "teams may keep running their wind tunnels [S1]" passes them even when S1 says the opposite. Step 5 closes that gap with a model judgement, which is itself not infallible. The evaluation below therefore also checks expected facts against the cited text.

## Evaluation against the real documents and models

`scripts/check_fia_rag.py` runs the questions in `scripts/fia_rag_eval_questions.json` (answerable, paraphrased, cross-document, unanswerable, adversarial, and a DRS trap for human review) in two phases: retrieval for all questions (printed with scores, for threshold calibration), then generation with strict checks – grounded answers must cite passages from the expected sections and contain the expected facts, which must also appear in the cited text (numbers are matched as whole numbers: `80` does not match `800`); unanswerable questions must be declined; adversarial questions must not produce the invented rule. It exits non-zero on any failure and writes `outputs/fia_rag_eval.json` (not served by the API).

```bash
python scripts/check_fia_rag.py --retrieval-only   # embeddings only
python scripts/check_fia_rag.py --calibrate        # threshold sweep, no chat calls
python scripts/check_fia_rag.py                    # full run
```

`--calibrate` sweeps `min_score` from 0 to 1 over the retrieval results.
* For each answerable question it finds the best-scoring retrieved passage carrying each expected fact and each expected section. The weakest of these is the highest threshold at which the question keeps all its evidence.
* For each unanswerable question it takes the top score.

It prints how many answerable questions keep their evidence and how many unanswerable ones would be declined before generation. It recommends the highest threshold that keeps all answerable evidence, and reports questions whose evidence is not retrieved at all at the current `top_k`. The report is written to `outputs/fia_rag_calibration.json`.

### Results with real documents and models (2026-09-24 to 26)

Configuration:
* **Documents:** the six current 2026 issues (A iss 03, B iss 08, C iss 20, D iss 07, E iss 06, F iss 10), downloaded by the script.
* **Models:** OpenRouter free tier, with `nvidia/nemotron-3-embed-1b:free` (2,048-dimensional) for embeddings and `nvidia/nemotron-3-ultra-550b-a55b:free` for answers.
* **Retrieval:** embedded Qdrant; `min_score` 0.30, `top_k` 8 from run 2 onwards.

**Retrieval and calibration.** For every answerable, paraphrased and cross-document question the top passage came from the page that contains the answer (similarity 0.50–0.72). The same questions against a Qdrant server (Docker, API key) returned identical rankings and scores. `--calibrate` (0 embedding requests, served from the cache):

| Question | Keeps all evidence up to `min_score` |
| --- | --- |
| cross-document fuel + pit speed (weakest answerable) | 0.47 |
| paraphrased track limits | 0.50 |
| other answerable questions | 0.53–0.72 |

| Unanswerable question | Top score |
| --- | --- |
| race result | 0.24 |
| ticket prices | 0.22 |
| "Article B99.7" | 0.27 |
| fictional rule (plausible wording) | 0.43 |

At 0.30 all eight answerable questions keep their evidence and three of the four unanswerable questions are declined before the chat model is called. The fourth reaches the model and is declined there. The sweep's "highest safe" value, 0.47, would also stop the fourth question, but it sits 0.004 below the weakest answerable question in an eight-question sample. The default therefore stays at 0.30, and the generation stage and validator decline the rest.

**Run 1 (top_k 5): 12 pass, 2 fail, 1 review.** Both failures were declines, not wrong answers:

* *Cost-cap comparison* – all five passages came from Section E. The F1-team cap in Section D (D4.1.2, US Dollars 215,000,000) ranked 7th, so the model correctly said the evidence was insufficient. On the evaluation questions (no new embedding calls), top_k 8 and a diversity (MMR) re-ranking both retrieved the missing passage; the simpler top_k 8 became the default.
* *Unsafe release* – the model declined although the evidence states the rule. The prompt was changed so that a rule stated in the regulations' own terminology is reported rather than refused. This was a generation-only change; the index was not touched.

**Runs 2 and 3 (top_k 8, revised prompt; run 3 after further hardening): 13 pass, 1 fail, 1 review each.** The remaining failure: "What happens if a car is released … in an unsafe condition *during the race*?" was declined. The retrieved passage says "during a TTCS, a Stop-and-Go Penalty will be imposed", but TTCS is defined only in Appendix B1 (page B85), which was not retrieved, and the model would not equate a TTCS with the race.

**Run 4 (definitions glossary, number and uncited-statement checks): 14 pass, 0 fail, 1 review** (12 chat and 0 embedding requests). Answers:
* **Unsafe release, answered for the first time.** The TTCS and LTCS definitions were added to the evidence. The model answered "The race is a Total Time Classified Session (TTCS) [S10]. If an F1 Car is released … in an unsafe condition during a TTCS, a Stop‑and‑Go Penalty will be imposed on the driver [S1][S2] …". It also covered the fine on retirement and the additional penalty, citing the definition and B1.6.2.
* **Other answerable questions, correct and cited:**
  * pit-lane limit of 80 km/h and €100 per km/h up to €1000 (B1.6.3);
  * minimum mass of 726 kg plus Nominal Tyre Mass in qualifying (C4.1);
  * shutdown periods of 14 (or 13) and 9 days (F3.1.1);
  * both paraphrased questions;
  * fuel energy flow of 3000 MJ/h together with the pit-lane limit;
  * the cost caps: D4.1.2, US $215,000,000; E2.3.1, US $190,000,000, citing the added "Inaugural Season" definition. The difference (US $25,000,000) is marked "Inference:".
* **Declines:** all four unanswerable questions, the "invent Article B42.1" request and the DRS trap were declined (DRS does not exist in the 2026 regulations).
* **Forged excerpt:** the model answered from the real rule ("Fuel may not be added to nor removed from a car during a Race", C6.4.4) and pointed out that the forged "S9" in the question is not a supplied excerpt.
* **Harness corrections:** the harness first scored two of these as failures, and neither was an answer error.
  * The model wrote "Stop‑and‑Go" with non-breaking hyphens (U+2011), which the fact matcher did not normalise. The matcher also missed "Stop-\nand-Go" split across a PDF line.
  * The shutdown answer was a numbered list followed by a separate `[S1]` paragraph, which the validator did not treat as covering the list. It now accepts a trailing sources block; every number in the list occurs in F3.1.1.

  Both fixes have regression tests. All 36 saved model outputs from runs 1–3 were re-validated with the final validator: no verdict changed.

**Run 5 (after a code audit that tightened the number and citation checks): 13 pass, 1 fail, 1 review** (12 chat and 0 embedding requests).
* **All answers still correct:** the unsafe-release, shutdown and cost-cap answers were correct again, and the forged excerpt and invented-rule requests were rejected.
* **The one failure was a validator gap, not a wrong answer.** The cost-cap answer cited all its figures and marked the computed difference as an inference, but in parentheses: "(Inference: ... higher ... by US $25,000,000 ...)". The inference marker was only recognised at the very start of a sentence, so the marked difference counted as an uncited claim with an unsupported number.
* **Fix:** markers are now also recognised after an opening parenthesis. `test_a_parenthesised_inference_is_exempt_but_a_parenthesised_claim_is_not` pins this; a parenthesised statement without a marker is still declined.
* **Re-check:** the saved outputs of all five runs were re-validated with the final validator. Run 5 now scores 14 pass, 0 fail, 1 review, and no earlier verdict changed.

**Prompt injection.** In runs 1 and 2 the forged-excerpt question made the chat model answer "Yes, according to Article Z1.1 teams may refuel cars during the race [S9]". The validator rejected it (`invalid_citation`; `Z1.1` is not in the evidence) and the API returned the decline. In runs 3 to 5 the model resisted by itself.

**Claim verifier (3 requests).**
* With `FIA_RAG_VERIFY_CLAIMS=true`, the minimum-mass answer ("726 kg plus the Nominal Tyre Mass [S1]") was verified.
* One sentence was planted in the real shutdown answer: "During both shutdown periods, teams may continue to run their wind tunnels." This contradicts F3.1.2, but it contains no number or identifier and is covered by the sources block. It passed every deterministic check. The verifier flagged exactly that sentence and the answer was declined (`unverified_claim`).

## Tests

`python -m pytest -q` runs the whole suite. The RAG tests (`tests/test_fia_*.py`, 239 tests, plus `tests/test_fetch_fia_regulations.py` and `tests/test_check_fia_rag.py`) use real PDF files generated by `tests/helpers.write_pdf`, real pypdf parsing, the real LangChain splitter and a real embedded Qdrant; only the two network services are replaced – by a deterministic bag-of-words embedder and a scripted model. They need only `requirements-rag.txt`. Deliberately breaking any of the grounding, fingerprint, manifest or cache checks (or the API's request limits) makes at least one test in the suite fail.

### OpenAI wire contract

`tests/test_fia_openai_wire.py` runs the RAG's real OpenAI clients (`OpenAIEmbeddings`, `ChatOpenAI`) over HTTP against a local server. The server implements `POST /v1/embeddings` and `POST /v1/chat/completions` with OpenAI's response shapes, error bodies and status codes. Connections to anything other than loopback are refused during these tests. They check:
* the bearer key and model names sent;
* plain-string embedding inputs in batches of at most `FIA_RAG_EMBEDDING_BATCH_SIZE`;
* exact decoding of base64 embeddings;
* non-streaming chat requests with the configured temperature and token limit;
* declines on `finish_reason` `length` or `content_filter`, reported in `validation.finish_reason`;
* a 401 whose error message never echoes the key;
* bounded retries on persistent 429s;
* timeouts turned into a `ProviderError`, checked separately for the embeddings and the chat client.

`FIA_RAG_TIMEOUT_SECONDS` is not a deadline for the whole call. httpx applies it to each wait separately (connecting, or the next bytes of a response), so a provider that keeps trickling bytes can hold a call for longer than timeout × attempts.

## Limitations

* **Currency.** Answers are only as current as the PDFs in `data/fia_docs`. Re-run the downloader and the index build when FIA publishes new issues (`.github/workflows/fia-source-check.yml` checks weekly that the official PDFs can still be discovered on fia.com). Check high-stakes interpretations against the official documents.
* **PDF extraction** is imperfect:
  * tables are flattened;
  * some words in Sections D/E are split (`Yea r`);
  * some ligatures extract as other letters ("OWicial");
  * image-only pages are skipped.
* **Threshold.** The similarity threshold is model-specific. Re-run `--calibrate` when changing the embedding model; the evaluation set (15 questions) is a sanity check, not a statistically robust calibration.
* **Retrieval.** Single-query dense retrieval can miss one side of a comparative question (see run 1).
* **Definitions** are added only for abbreviations used in the retrieved passages and for terms named in the question. A defined term the passages spell out in full ("Power Unit") is not expanded unless the question names it.
* **Numbers repeated from the question.** The number check declines an answer that repeats a number which appears only in the question (for example, correcting a false premise such as "is the limit 60 km/h?"). This is deliberate: accepting numbers from the question would let a question plant them.
* **Entailment.** The deterministic checks prove citation, identifier and number integrity, not entailment. The optional verifier adds a model judgement of entailment, which can itself err.
