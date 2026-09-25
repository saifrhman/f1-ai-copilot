# Web UI guide

The web UI is a Streamlit app (`ui/streamlit_app.py`) that runs in your browser on your own computer. It is a thin client of the FastAPI API: every result on a page comes from an API response, and the UI never computes, fills in or invents one. When the API cannot answer, the page says why and what to do next.

Installation and start-up are in the README section "Run it on your own computer". In short, from the project folder with the virtual environment active:

```bash
python scripts/run_app.py      # API on http://127.0.0.1:8000, UI on http://127.0.0.1:8501 (opens the browser)
```

Ctrl+C in that terminal stops both servers.

**Contents**

* [How the UI works](#how-the-ui-works)
* [Pages](#pages): [Overview](#overview), [FIA regulations](#fia-regulations), [Race strategy](#race-strategy), [Car setup](#car-setup), [Ghost car](#ghost-car), [Driver radio](#driver-radio), [Ask the copilot](#ask-the-copilot), [Incident triage](#incident-triage)
* [Reading the results](#reading-the-results)
* [Model-provider requests](#model-provider-requests)
* [Troubleshooting](#troubleshooting)
* [UI settings](#ui-settings)

## How the UI works

* **One API, one index.** The UI sends HTTP requests to the API (default `http://127.0.0.1:8000`, set with `F1_API_URL`). Only the API opens the regulation index. Images such as the ghost-car chart are fetched by the UI server, so the browser only needs to reach the UI.
* **Sidebar.** The navigation, the API address, and the readiness of every API component (`ready`, `not configured`, `unavailable`, ...). The status is cached for 30 seconds and updated on your next action after that; **Refresh status** checks it immediately.
* **Forms keep their values** while you switch pages. Each page keeps its last result until you submit again. Reloading the browser tab starts a new session, which resets the forms to their examples.
* **Pre-filled examples.** The strategy, setup and triage forms start with documented example values, and the ghost-car page starts with synthetic laps. A result computed from unchanged example inputs gets a grey **Example inputs · not your data** badge.
* **Raw data.** Every result has a **Raw API response** expander (the JSON the API returned), and the form pages also show the **Request sent to the API**.
* **What leaves your computer.** Only the regulation Q&A calls an outside service: the model provider in `.env`. Search passages sends the question text (for its embedding). Ask sends the question and the retrieved regulation passages. All other pages run locally in the API. `scripts/fetch_fia_regulations.py` downloads from fia.com. Whisper, if you install it, downloads its model once. Streamlit usage statistics are switched off.

## Pages

### Overview

![Overview on a first run: heuristic modules ready, regulation QA not configured, with next steps](images/ui-overview-first-run.png)

* **System status:** the API version and address, and `healthy` or `degraded`. Degraded means at least one component is not ready; the others keep working.
* **Cards:** one per page, with the live state of the modules behind it. The orange **heuristic** label marks modules that use hand-set rules or models that have not been validated against real outcomes.
* **FIA regulation index:**
  * Documents (regulation PDFs found).
  * Index state: `current`, `missing`, `empty`, `stale` or `incomplete`.
  * Indexed passages.
  * Definitions: the number of official definitions extracted from the PDFs.
  * Model provider: `ok`, `failing` or `unknown`. This is the outcome of the API's latest regulation request, not a live check.
* **Next steps:** shown whenever the regulation QA is not ready. They are the exact commands for your setup: host commands, or `docker compose ...` commands when the API runs in Docker Compose. The yellow notes above them are the API's own problem messages.
* **Documents** and **Retrieval and model settings** (expanders): the PDFs and the RAG settings the API uses.
* The link to the interactive API reference (`/docs`).

### FIA regulations

Answers questions **only** from the official FIA Formula 1 Regulations PDFs. The page has two tabs:

* **Ask:** retrieval, one answer-model call, then validation.
* **Search passages:** retrieval only. No answer model is called.

This page needs a model-provider key, the PDFs and a built index; until then it shows the regulation index state and next steps, and the forms are disabled.

**Index card** (top of the page):
* Documents, indexed passages and definitions.
* Claim verifier on or off (`FIA_RAG_VERIFY_CLAIMS`).
* Model provider status.
* The embedding and answer models.
* The default `top_k` and similarity threshold.

**Inputs**

| Input | Range | Meaning |
| --- | --- | --- |
| Question | 1-2,000 characters | Plain English. The example buttons fill in questions from the project's evaluation set. |
| Passages to retrieve (top_k) | 1-50, default `FIA_RAG_TOP_K` (8) | How many passages are retrieved. Every passage above the threshold is given to the model; more passages can bring in other documents. |
| Similarity threshold (min_score) | 0-1, **Search passages only** | Splits the retrieved passages into evidence and non-evidence, for exploring scores. **Ask always uses the API default** (`FIA_RAG_MIN_SCORE`, 0.30). |

#### Reading an answer

A **grounded answer** has a green **Grounded answer** badge. It passed every check:
* every citation `[S1]`, `[S2]`, ... names a passage that was retrieved;
* every statement is cited;
* every article or rule number and every number in the answer occurs in the passages it cites;
* the answer was not cut off.

Each citation is a blue chip. Clicking a chip under the answer shows that passage, with a link to the official PDF opened at its page.

The card says what was checked and what was not. The deterministic checks do not prove that each sentence means what its passage says. With the claim verifier on, a second model call judges that; its verdict is a model's judgement, not proof. **Read the cited passages before relying on an answer.**

The metrics below the answer:
* **Evidence strength:** the best similarity (0 to 1) between the question and the regulation passages the answer cites. It is **not a probability that the answer is correct**; it says how close the cited text is to the question. It is shown as "–" when only definitions were cited, because definitions are not retrieved by similarity.
* **Passages cited:** how many of the passages given to the model the answer cites.
* **Rules referenced:** the article and rule numbers in the answer, each found in the cited passages.

A **declined** question has an orange **Declined: no answer is shown** badge, a headline, what happened, which check failed and what you can do. A decline is a result, not an error: the system found no answer it could support from the regulations. When the model's text failed validation, it is available under **Rejected model output: failed validation, not an answer** for inspection only. It is not a statement of the regulations.

| Reason code | What happened | What you can do |
| --- | --- | --- |
| `no_evidence_above_threshold` | No passage reached the similarity threshold; the answer model was not called | Use the regulations' own terms, or check **Search passages** for the closest passages |
| `model_declined` | The model found the evidence insufficient | Check the evidence shown; rephrase or raise top_k |
| `empty_model_output` | The model returned nothing | Ask again; if it repeats, check the provider settings |
| `invalid_citation` | The answer cited a passage that was not supplied | Rephrase |
| `missing_citation` | The answer cited nothing | Rephrase |
| `uncited_claim` | Part of the answer had no citation | Rephrase |
| `unsupported_rule_reference` | The answer named an article or rule that is not in its cited passages | Rephrase |
| `unsupported_number` | The answer stated a number that is not in its cited passages | Rephrase |
| `truncated_model_output` | The reply hit the output limit or the provider's content filter | Ask a narrower question, or raise `FIA_RAG_MAX_OUTPUT_TOKENS` (not for the content filter) |
| `unverified_claim` | The claim verifier found an unsupported sentence | Rephrase |

**Evidence given to the model** lists the passages the model saw:
* **regulation passages** above the threshold, in rank order, with their similarity;
* then **definitions**: official definitions of terms that the passages use or the question names (for example "TTCS"). Definitions supplement the evidence. They are not retrieved by similarity, so they have no similarity score, and they are never evidence on their own.

For each passage the table shows the label, whether it was cited, the section, printed page, PDF page and a link to the official PDF. Expand a passage to read its verbatim text. Passages **below the threshold** are listed separately, greyed. They were retrieved but not given to the model, so they are not evidence. **Validation details** lists the checks, the claim verifier's report, the retrieval settings and the models.

#### Search passages

![Search passages against the real 2026 index: similarities per passage with the threshold line](images/ui-regulations-search.png)

Search passages shows:
* the best similarity, and the number of passages above and below the threshold;
* the definitions Ask would add;
* a bar chart of each passage's cosine similarity, with the threshold as a dashed line.

These are search results, not an answer. At the API default threshold and the same top_k, the passages above the threshold are exactly the evidence Ask would give the model; at any other threshold the page says that the split differs from Ask's.

### Race strategy

Ranks pit-stop plans for the rest of a race with a **heuristic** lap-time model:
* tyre warm-up, a peak window and linear degradation;
* driver and damage penalties;
* an exact search over pit laps for every compound sequence with up to 3 stops.

It is not a calibrated race simulator. Fuel, traffic, safety cars and weather changes are not modelled. The form starts from the engine's documented mid-race example: lap 18 of 57, with the example competitors HAM and VER.

**Race state**

| Input | Unit / range | Notes |
| --- | --- | --- |
| Current lap | 1-200, at most the total | The next lap to be driven; laps from here to the finish are planned |
| Total laps | 1-200 | Race distance |
| Weather | dry, intermediate, wet | Held constant. In intermediate or wet conditions a stop can only fit intermediate or wet tyres, so add rows for them to the tyre model |
| Track temperature | °C, -10 to 80 | Above 35 °C the model slows soft tyres and helps hard ones |
| Fitted tyres | compound, or "Fresh set" | With "Fresh set" every plan starts on new tyres without a pit cost |
| Laps on the fitted set | 0-100 | Ignored for a fresh set |
| Compounds used before the fitted set | list | For the simplified two-compound rule. Untick **Tyre history known** when you do not know them |
| Your gap to the leader | s, 0-7,200, optional | Needed for the competitor signals |

**Pace, driver and car**

| Input | Unit / range | Notes |
| --- | --- | --- |
| Recent lap times | s (`95.6`) or m:ss (`1:35.6`), 20-600 s each, at most 200 | Their mean is the base lap time |
| Tyre management, risk tolerance, braking consistency, throttle aggressiveness | 0-1 | Heuristic penalties. Each help text gives the thresholds, for example risk tolerance above 0.8 adds 3 % lap time |
| Measured braking consistency / throttle aggressiveness | 0-1, optional | Values from telemetry; they replace the profile values |
| Engine wear, brake wear, front wing, floor and diffuser damage | 0 = as new, 1 = worn out or destroyed | Up to +2 %, +1.5 %, +2.5 % and +4 % lap time |

**Tyre model** (one row per compound available to you). The short headers are explained in each column's help:

| Column | Range | Meaning |
| --- | --- | --- |
| Base perf. | 0.5-2 | Peak performance factor; 1.0 = the base lap time, lower = slower |
| Deg./lap | 0-0.5 | Performance lost per tyre lap after the peak window |
| Warm-up | 0-10 laps | Laps to ramp from 90 % to 100 % |
| Peak from / Peak to | tyre laps | The peak window; degradation starts after "Peak to" |
| Pit loss (s) | 0-120 | Time lost by a stop that fits this compound |

**Competitors** (optional, up to 30): a driver ID (for example `HAM`), their compound, tyre age (laps, 0-100) and gap to the leader (s).

#### Reading the result

![Race strategy result for the example inputs](images/ui-strategy-result.png)

**Recommended plan**
* **Plan id:** for example `2-stop:medium/soft/soft`, the stop count and the compound sequence.
* **Pit laps:** a pit lap is the last lap *before* the stop.
* **Projected time:** covers the remaining laps only (driving time plus pit loss). It is a heuristic estimate.
* **Margin to plan 2:** the model has no uncertainty range, so a small margin does not separate two plans.

**Two-compound rule** values in the plans table:
* `satisfied`;
* `unverified`: the earlier compounds are unknown;
* `waived`: not required, because the plan uses intermediate or wet tyres;
* `violated`: listed only when no plan can satisfy the rule; a warning explains why.

**Plans compared** lists up to 8 plans, and the timeline draws their stints (the gaps are pit stops). **Plan details** shows the stints of one plan: lap times, laps on the set, tyre performance and the laps past the peak window. The API's risk label comes from the number of stops only.

**Model factors used**
* Base lap time.
* Driver multiplier and damage/wear multiplier: 1.0 means no penalty.
* Tyre state: `supplied` (the fitted set continues) or `assumed fresh`.

**Competitor signals**
* `undercut_target`: a car up to 3 s ahead whose tyres are past their peak window.
* `undercut_threat`: a car level with you or up to 3 s behind.

Signals are labels from your inputs. They are not simulated and do not change the ranking.

**Assumptions and limitations** come from the API. It also lists inputs that it accepts but that do not change any number.

#### Calibrate tyre parameters from lap history

The expander at the top estimates the tyre model from timed laps by inverting the engine's own lap-time model. Its table starts with the API's documented example: synthetic laps generated from that same model, so they fit almost exactly. Replace them with real laps; you can paste from a spreadsheet.

| Input | Meaning |
| --- | --- |
| Compound, Tyre age, Lap time (s) | One row per lap. Tyre age is the lap number on that set (1 = first lap on it) |
| Race lap | Needed on unflagged laps only when a fuel correction is set |
| Out-lap, In-lap, Safety car | Flagged laps are excluded from the fit (safety car covers VSC and red-flag laps too) |
| Weather, Track temperature (°C) | The conditions of the whole history (temperature is required) |
| Pit-stop loss (s) | Required; copied into every compound, not estimated |
| Fuel correction (s/lap) | Optional; seconds gained per lap of fuel burnt (typically 0.03-0.06); 0 = none |
| Fixed values | Optional per-compound peak-window end or warm-up laps, instead of detecting them |

The result shows:
* the estimated base lap time;
* the reference compound (base performance 1.0 by definition);
* per compound: a fit status, R², residual standard deviation, the laps used, clean and supplied, and the rejected outliers;
* the estimated tyre model;
* the excluded laps with their reasons;
* the assumptions.

Each compound needs at least 5 clean laps. A compound without enough data is reported as insufficient data with the reason, never guessed.

**Use in the strategy form** copies the calibrated tyre model and base lap time into the form:
* Compounds that were not calibrated are removed from the tyre model, and a warning names them.
* The calibrated base lap time already contains the driver's and car's pace. Keep the driver profile and car condition penalty-free, or the result warns that penalties are counted twice.

### Car setup

Searches eleven setup parameters for the lowest score of a documented **heuristic objective** that trades downforce against drag, balance, bottoming, compliance and tyre wear. It uses a seeded Optuna search, then refines from three start points. It is not a vehicle-dynamics simulator: validate any setup in a simulator or on track. The form starts with the documented Silverstone example.

| Input | Unit / range | Notes |
| --- | --- | --- |
| Track name | text | A label for the reasoning only |
| Track type | high speed, technical, mixed, low speed | Sets kerb and bump severity |
| Lap length | m, 500-25,000 | |
| Average speed | km/h, 20-400 | |
| Corners, high-speed corners, low-speed corners | counts | High- plus low-speed corners cannot exceed the corners |
| Downforce requirement | 0-1 | 0 = low-downforce track, 1 = maximum downforce |
| Track condition | dry, intermediate, wet | Sets grip, standing water and ride-height need |
| Air temperature | °C, -10 to 50 | Sets the cold tyre pressures |
| Humidity, wind speed | %, m/s, optional | Validated but not modelled |
| Risk tolerance, tyre management | 0-1, optional | Empty = the API's default, listed under **Assumed defaults** |
| Pin setup values | ride height 60-85 mm, front wing 0-15°, rear wing 0-20°, differential 0-100 % lock | A pinned value is used as given and not searched |
| Search trials, seed | 16-300 trials (about 1 s at 128); seed ≥ 0 | The same seed and inputs give an identical result |

The results:
* **Objective (lower is better):** a dimensionless heuristic penalty, **not a lap time**.
* **Reduction vs baseline:** compared with the rule-of-thumb baseline. It is a model score, not a predicted lap-time gain.
* **Multi-start agreement:** the API calls it "confidence". It is the share of the refinement start points that reached the same setup (1.0 = all of them). It is not a probability that the setup is right.
* **Setup parameters table:** the recommended value, the baseline, the change, the unit and whether you pinned it. Brake bias is % of braking force on the front axle, 50-70. Differential and suspension use 0-100 scales. **Search limit** marks a value at an end of its 0-100 scale: read it as a direction, not a tuned setting.
* **Tyre pressures:** cold set pressures in psi.
* **Handling balance:** a dimensionless index; above 0 is understeer, below 0 oversteer.
* **Remaining objective penalties:** a chart of what the recommended setup still trades off.
* **Reasoning, assumed defaults and limitations,** and the search details.

### Ghost car

Compares a **reference lap** (lap 1) with a **comparison lap** (lap 2) on a common distance axis. Distance comes from integrating speed over time, or from x/y when both laps have it. Channels a lap does not supply are reported as missing, never filled in. Braking-zone matching and the loss/gain locations are heuristic.

* **Example laps:** two synthetic 10 Hz laps of a made-up 5.2 km circuit, generated by the page. They are clearly labelled and are not recorded telemetry.
* **Upload CSV files:** one CSV per lap. **Download a CSV template** gives the format.

| Column | Required | Values |
| --- | --- | --- |
| `timestamp` (or `time`) | yes | seconds, strictly increasing (any start value) |
| `speed` | yes | km/h, 0-400 |
| `throttle` | no | 0-1 (divide percentages by 100) |
| `brake` | no | 0-1, or true/false |
| `gear` (or `nGear`) | no | whole number 0-8 |
| `drs` | no | true/false or 1/0 (flap open) |
| `x`, `y` | no | metres, both or neither |
| `steering` | no | any unit; stored, not analysed |

Each lap needs 2-20,000 rows. Other columns (such as `distance`) are ignored. From FastF1, use `Time.dt.total_seconds()`, divide `Throttle` by 100, and convert `DRS` with `DRS >= 10`. The form checks the files before anything is sent and lists every problem by row. Optional per lap:
* lap number (1-200);
* official lap time (s): enables the lap-time delta and, when the samples span the lap time, lap-fraction alignment;
* three sector times (s).

**Alignment:**
* **Auto:** lap fraction when both laps run from timing line to timing line, otherwise distance.
* **Distance:** distance from each lap's first sample.
* **Lap fraction:** each lap mapped onto 0-1 of its own distance.

![Ghost-car comparison of the synthetic example laps](images/ui-ghost-result.png)

Every gap is **lap 2 minus lap 1** at the same position: above 0 means lap 2 is behind.
* **Gap at the end:** the gap at the end of the compared distance.
* **Official lap-time delta:** from the lap times or sector times, when both laps have them.
* **Largest loss / gain:** the largest net change of the gap over any stretch, with its start and end. Changes below the stated resolution are reported as none.
* **The chart:** rendered by the API. Speed on top; the gap along the lap below.
* **Tabs:**
  * Laps: samples, distance, lap time and missing channels.
  * Sectors.
  * Braking: brake point delta > 0 means lap 2 brakes later.
  * Throttle, gears, DRS.
  * Aligned traces: the full table behind the chart.

### Driver radio

Gives a **coarse emotion label** for a team-radio clip:
* acoustic features (pitch and energy) are compared with hand-set emotion profiles;
* optionally, a local Whisper transcript is scanned for emotion keywords.

It is **not a validated emotion model** and not a psychological assessment.

* **Clip:** upload a file, or **Record** with the browser microphone. The browser allows the microphone only on `localhost`/`127.0.0.1` or HTTPS pages.
* **Limits:**
  * 0.5-120 s of speech, mono or stereo, 8-96 kHz, at most 20 MiB;
  * WAV, FLAC, OGG and MP3 are decoded natively; M4A and WebM need ffmpeg where the API runs;
  * silent or noise-only clips are rejected, not labelled.
* **Transcribe with Whisper:** optional and slower. It is disabled, with the reason, when Whisper is not installed where the API runs.

The results:
* **Emotion:** the final label (calm, angry, panicked, focused, excited, frustrated or neutral).
* **Confidence:** a heuristic score from 0 to 0.95, **not a probability**. The **combination rule** under the metrics says how it was computed:
  * acoustic label only;
  * transcript agrees;
  * transcript overrides the acoustic label;
  * transcript disagrees too weakly to change it.
* **Acoustic label:** the label from pitch and energy alone. Its acoustic confidence is 0.45 × the best profile similarity + 0.55 × its lead over the runner-up, at most 0.95. Below 0.20 the acoustic label is neutral.
* **Close profiles / Tied profiles:** a warning when the best profile leads the runner-up by less than 0.05, or not at all. The audio then barely separates the two labels.
* **Transcript:** Whisper's text and the emotion keywords found in it (negated keywords are ignored).
* **Profile similarity** and **acoustic features:** the 13 measured features; the four marked "used for the label" feed the profiles.

Energy is measured on the recording as it is: the same speech recorded louder or quieter can get a different label.

### Ask the copilot

A plain-English question is sent to **one** module, chosen by a transparent **keyword** heuristic. It is not a language model. **What can I ask?** lists the topics and the context each module needs.

| Question about | Answered by | Context it needs |
| --- | --- | --- |
| FIA rules (penalties, flags, "allowed", "Article 33") | FIA regulation QA: the same checks and declines as the FIA regulations page | none |
| Race strategy | Strategy engine (heuristic) | `telemetry`, `car_status`, `driver_profile`, `tire_data`, `race_state`; optional `competition` |
| Car setup | Setup search (heuristic) | `driver_preferences`, `track_profile`, `weather` |
| Lap performance | A summary of the telemetry you supply | `telemetry` |
| Driver radio | Radio emotion heuristic | an attached clip |

* **Question:** up to 2,000 characters. A regulatory cue (rule, penalty, allowed, FIA, ...) always sends a question to the regulations.
* **Context:** a JSON object with the evidence for the module.
  * **Strategy example**, **Setup example** and **Telemetry example** insert the API's documented examples. They are marked as example values, not your data.
  * Without context, a module does not guess: it says what it needs.
* **Attach a driver-radio clip:** for radio questions (at most 20 MiB).

In the result:
* **Routed to:** the module.
* **Score:** named for what that module reports:
  * Evidence strength (regulations; 0 for a decline);
  * Multi-start agreement (setup);
  * Coverage (lap performance);
  * Heuristic score (radio).

  The strategy engine reports time margins, not a score. None of these is a probability.
* **Data sources:** the evidence the answer used.
* **How the question was routed:** the decision rule and the keyword scores.

### Incident triage

Sorts an incident into a **review category** with a preliminary severity from a transparent rule table. It is **not an FIA steward-decision predictor**: it does not predict penalties, is not a legal or regulatory determination, and never cites FIA articles. For what the rules say, use the FIA regulations page.

| Input | Values | Effect |
| --- | --- | --- |
| Incident type | track limits, unsafe release, collision, blocking, dangerous driving, technical infringement, speeding in pit, illegal overtaking | Sets the review category and base severity |
| Track condition | dry, wet, intermediate, mixed, unknown | Context only; never changes the severity |
| Intent | accidental, intentional, racing incident, unknown | "Intentional" raises on-track driving cases; "racing incident" lowers car-to-car cases |
| Recent penalties | one per line, optional | Only the count is used: three or more raise the severity |
| Total penalties on record | ≥ 0, optional | Ten or more raise the severity slightly; cannot be below the recent count |

The results:
* **Review category:** the kind of steward review the case needs, not a sanction.
* **Severity band:** low below 0.50, medium from 0.50, high from 0.80.
* **Severity score:** 0-1.
* **Confidence (input completeness):** the share of the severity-relevant inputs you supplied. It is not a probability of any decision.
* **The adjustments** that moved the score, and the reasoning.

## Reading the results

| You see | Meaning |
| --- | --- |
| Orange **Heuristic · not validated** badge | Result of a hand-set heuristic that has not been validated against real outcomes; useful for exploring, not for decisions |
| Grey **Example inputs · not your data** badge | Computed from the unchanged pre-filled example |
| Green **Grounded answer** | A regulation answer that passed every citation, rule-number and number check |
| Orange **Declined** | The regulation QA found no answer it could support; the reason and next step are shown |
| "Nothing was sent to the API" | The form found a problem first (for example an unreadable lap time or CSV row); fix it and submit again |
| HTTP 422 | The API rejected an input; each line names the field in the form's words |
| HTTP 503 | A component is not ready (see [Troubleshooting](#troubleshooting)); nothing is substituted for the result |
| HTTP 413 | The request is larger than the API accepts |
| HTTP 500 | A server-side error; the API's terminal log has the details |

## Model-provider requests

Only the regulation QA uses the model provider. The API retries a failed call up to `FIA_RAG_MAX_RETRIES` more times (default 2), and every retry is another request. This matters on quota-limited plans such as OpenRouter's free tier (50 requests per day).

| Action | Requests |
| --- | --- |
| Overview, sidebar, all heuristic pages | 0 |
| Search passages | 1 query embedding; 0 when the exact same question was embedded before (cache) |
| Ask (and regulatory questions on Ask the copilot) | 1 query embedding (0 if cached), plus 1 answer call when a passage reaches the threshold, plus 1 more with `FIA_RAG_VERIFY_CLAIMS=true` |
| `python scripts/build_fia_index.py --dry-run` | 0 (prints the chunk count and the number of embedding requests) |
| `python scripts/build_fia_index.py` | at most chunks ÷ `FIA_RAG_EMBEDDING_BATCH_SIZE`, rounded up: 1,938 chunks for the 2026 PDFs give 16 requests at 128 and 8 at 256. Texts already in the embedding cache (`.cache/`) are not sent again |
| `python scripts/check_fia_rag.py --calibrate` | 1 embedding per evaluation question (15), 0 for cached ones; no answer calls |

## Troubleshooting

**"... could not reach the API at http://127.0.0.1:8000"** (sidebar: `unreachable`). The API is not running, or runs at another address. The message under the error gives the start command for that address:
* Start the API with `uvicorn app.main:app`, or stop the UI and start both with `python scripts/run_app.py`.
* If the API runs elsewhere, set `F1_API_URL` and restart the UI.
* In Docker Compose: `docker compose ps`, `docker compose logs api`, then `docker compose up -d api`.

"No answer within 300 s" means the API is running but a request (model provider, setup search, transcription) took too long; try again, and check the API log.

**HTTP 503 on the regulation pages.** The Overview page shows the state and the exact next steps.

| State | Cause | Fix |
| --- | --- | --- |
| `not_configured` | No PDFs, no `OPENAI_API_KEY`, or no index yet | `python scripts/fetch_fia_regulations.py`, set the key in `.env`, build the index |
| `misconfigured` | An invalid `FIA_RAG_*`/`QDRANT_*` setting; the message names it | Correct `.env`, restart the API |
| `index_missing`, `index_empty` | No index built for these documents and settings | Stop the API (embedded storage only), run `python scripts/build_fia_index.py --dry-run`, then the same without `--dry-run`, and start the API again |
| `index_incomplete` | A build was interrupted | Rebuild as above |
| `index_stale` | The PDFs, chunking settings or embedding model changed since the build | Rebuild as above; unchanged texts come from the embedding cache |
| `provider_failing` | The latest model-provider call failed (key, `OPENAI_BASE_URL`, model names, quota or rate limit) | Read the error on the Overview page, fix `.env` and restart the API, or wait for the quota to reset; the state clears after the next successful request |
| `unavailable` | The index storage could not be opened (see the Qdrant lock below) | Stop the other process, or use a Qdrant server |

A missing definitions glossary also blocks the regulation QA. Rebuild the index to add it; no embedding requests are needed for that.

**Embedded Qdrant lock ("Storage folder ... is already accessed by another instance of Qdrant client").** The default embedded index (`.qdrant/`) can be opened by only one process at a time. Common cases:
* **An index build while the API runs.** Stop the API first (Ctrl+C in the `run_app.py` terminal stops the UI too), build, then start again.
* **Two API processes.** For example, `run_app.py` started twice, or a `uvicorn` left running in another terminal: the second API reports the index as unavailable. Stop the other process.
* **Several processes need the index at once.** Run the Qdrant server instead: `docker compose up -d --wait qdrant` and `QDRANT_URL=http://127.0.0.1:6333` in `.env`. Then build the index once for the server.

**Provider quota and rate limits.** A quota or rate limit shows as HTTP 503 with a message such as `RateLimitError (HTTP 429): ...`, and the model provider status turns `failing`. OpenRouter's free models allow 50 requests per day; a failed call's automatic retries also count against that. Options:
* wait for the limit to reset;
* lower `FIA_RAG_MAX_RETRIES`;
* use **Search passages**, which makes no answer calls;
* switch provider or plan.

A message ending in `(HTTP 401): the provider rejected the credentials` (or 403) means the key, or the `OPENAI_BASE_URL` it belongs to, is wrong.

**Port already in use.** `run_app.py` refuses to start and names the owner when it can ("by an F1 AI Copilot API; an earlier run of this launcher may still be running"). To free the port:
* find the program with `lsof -i :8000` (macOS/Linux) or `netstat -ano | findstr :8000` (Windows), and stop it;
* or choose other ports: `python scripts/run_app.py --api-port 8010 --ui-port 8601`. The UI is told the new API address automatically.

When you start the servers separately instead, give the UI the new address with `F1_API_URL=http://127.0.0.1:8010`.

**Upload limits.**
* The browser uploader accepts files up to 20 MB (`maxUploadSize` in `ui/.streamlit/config.toml`).
* Radio clips: at most 20 MiB decoded, 0.5-120 s, up to 2 channels, 8-96 kHz. For long recordings, trim them or save them as FLAC, OGG or MP3.
* Ghost CSVs: 2-20,000 rows per lap.
* The API rejects larger request bodies with HTTP 413: 40 MB for the audio routes, 8 MB for the ghost comparison, 1 MB elsewhere.

**Whisper unavailable.** The radio page says "Transcription is unavailable on this API" with the reason, for example that `openai-whisper` or ffmpeg is not installed. The acoustic analysis works without it. To add transcription, install ffmpeg and Whisper where the API runs (README: "Optional: Whisper transcription"), check them with `python scripts/check_whisper.py <clip>`, and restart the API: availability is checked once per API process. "Transcription failed" (HTTP 503) means Whisper is installed but failed on this clip; submit again without transcription and check the API log. M4A and WebM clips need ffmpeg even without Whisper. The Docker image includes ffmpeg but not Whisper.

**Microphone recording does nothing.** Browsers allow the microphone only on `http://localhost`, `http://127.0.0.1` or HTTPS. On another device (LAN mode, `--host 0.0.0.0`), upload a file instead.

**Ghost images cannot be written** (Overview: `artifacts not writable`). Make `F1_ARTIFACTS_DIR` (default `outputs/`) writable by the API process.

**"Page not found" at `/overview`.** The Overview is the default page and lives at `http://127.0.0.1:8501/`; Streamlit briefly shows this notice for `/overview`, then opens the Overview.

**Docker Compose.**
* `.env` changes need `docker compose up -d api`; `docker compose restart api` keeps the old settings.
* On Linux, if the containers cannot write `data/`, `.cache/` or `outputs/`, rebuild with your user id: `APP_UID=$(id -u) docker compose up -d --build`.
* Times in the UI are shown in the container's time zone (UTC).

## UI settings

| Setting | Where | Default | Meaning |
| --- | --- | --- | --- |
| `F1_API_URL` | environment, then `.env` | `http://127.0.0.1:8000` | Address of the API. `run_app.py` and Docker Compose set it themselves. `http://user:password@host:port` works for an API behind an authenticating proxy; the credentials are never displayed |
| `server.address`, `server.port` | `ui/.streamlit/config.toml` | `127.0.0.1`, `8501` | This computer only. Command-line flags override them, e.g. `streamlit run ui/streamlit_app.py --server.port 8600` |
| `server.maxUploadSize` | `ui/.streamlit/config.toml` | 20 (MB) | Largest file the uploaders accept |
| `--host`, `--api-port`, `--ui-port`, `--no-browser` | `python scripts/run_app.py` | `127.0.0.1`, 8000, 8501, opens the browser | `--host 0.0.0.0` makes both servers reachable from your network; neither has authentication, so use it only on a trusted network |

To run only the UI against an API on another machine: `pip install -r requirements-ui.txt`, set `F1_API_URL` (for example `http://192.168.1.20:8000`), then `streamlit run ui/streamlit_app.py`.
