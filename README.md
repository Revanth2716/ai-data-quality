# AI Data Quality & Anomaly Detection API

Upload a CSV, get a quality report: profile, detected issues, statistical anomalies, a
0–100 quality score, and an optional AI-assisted explanation of the computed analysis.

A lightweight, Python-first developer/data-engineering API built as a portfolio project.
Runs **fully offline by default** — with `AI_PROVIDER=mock` every feature works with no
API key, no network and no cost.

## Features

- **Data profiler** — rows/columns, per-column type inference (numeric, categorical,
  text, boolean, datetime), missing values, duplicate rows, unique counts, and a
  seven-number summary for numeric columns.
- **Quality analyzer** — 7 deterministic issue detectors: missing values, duplicate
  rows, inconsistent types (≥80%-numeric columns with unparseable values), sentinel
  values (`N/A`, `?`, `unknown`, `-`, …), empty columns, constant columns, and
  infinite values. Each issue carries severity, count, ratio and example rows/values.
- **Anomaly detector** — IQR fences per numeric column, Z-score (|z| > 3, gated on
  n ≥ 30), and a multivariate IsolationForest pass (scikit-learn, fixed 2%
  contamination, `random_state=42` for determinism).
- **Quality score** — 100 minus capped penalties across five weighted factors
  (completeness 25, uniqueness 20, validity 15, consistency 10, anomalies 30) with
  A–F grade bands and a full penalty breakdown. **Computed entirely in code.**
- **AI explanation (optional)** — the AI only ever *explains* the computed analysis;
  it cannot recalculate or invent scores. Deterministic mock provider by default, or
  any OpenAI-compatible chat endpoint. Provider failures degrade to the mock template.
- **Swagger UI as the interface** — no frontend; upload and explore at `/docs`.

## Architecture

```
CSV upload (multipart)
        │
        ▼
FastAPI validation          extension, size cap, UTF-8, rectangularity check
        │
        ▼
Pipeline (pure functions)
 ├─ profiler.py   profile_dataset(df)
 ├─ quality.py    detect_quality_issues(df, profile)      7 issue types
 ├─ anomaly.py    detect_anomalies(df, profile)           IQR + Z-score + IsolationForest
 ├─ scoring.py    compute_quality_score(...)              5 capped factors → 0–100
 └─ ai/           explain_safely(provider, analysis)      mock | openai-compatible
        │
        ▼
Structured JSON report + optional explanation
```

```
ai-data-quality/
├── app/
│   ├── main.py        FastAPI app factory + error handlers
│   ├── routes.py      GET /health, POST /api/v1/analyze
│   ├── schemas.py     Pydantic v2 response contract
│   ├── config.py      env-driven settings (stdlib dataclass)
│   ├── errors.py      uniform {"error": {code, message}} envelope
│   ├── profiler.py    profiling (pure)
│   ├── quality.py     issue detection (pure)
│   ├── anomaly.py     IQR / Z-score / IsolationForest (pure)
│   ├── scoring.py     score + grade (pure)
│   ├── pipeline.py    orchestration
│   └── ai/            provider protocol, mock, openai-compatible adapter
├── samples/           synthetic demo CSVs + seeded generator script
└── tests/             pytest suite (unit + API integration)
```

The analysis core is a set of pure DataFrame→model functions with no HTTP or I/O
concerns, which makes every stage unit-testable without the API layer.

## Technology stack

| Layer     | Choices                                                     |
| --------- | ----------------------------------------------------------- |
| API       | FastAPI, Uvicorn, Pydantic v2                               |
| Data      | pandas, NumPy                                               |
| ML        | scikit-learn (IsolationForest)                              |
| AI        | httpx against any OpenAI-compatible endpoint; mock fallback  |
| Testing   | pytest (unit + API integration via `TestClient`)            |
| Tooling   | uv, Python 3.12                                             |

Deliberately **not** included: databases, message queues, Docker, auth, frontend
frameworks, LangChain, cloud services. Swagger UI is the interface.

## Local setup

```bash
uv sync                                  # creates .venv and installs everything
uv run uvicorn app.main:app --reload     # → http://localhost:8000/docs
```

Then open **http://localhost:8000/docs**, `POST /api/v1/analyze`, and upload one of
the sample CSVs from `samples/`.

Or with curl:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze?explain=true" \
     -F "file=@samples/customers_dirty.csv"
```

## API

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0", "ai_provider": "mock" }
```

### `POST /api/v1/analyze?explain=false`

Multipart upload (`file`), query param `explain` to attach the AI explanation.
Errors use one uniform envelope: `{"error": {"code", "message"}}` with
`400` (bad/unsupported CSV), `413` (size cap), `422` (missing file part).

Response shape (abridged):

```jsonc
{
  "file":     { "name": "customers_dirty.csv", "size_bytes": 6643 },
  "profile":  { "row_count": 119, "column_count": 10,
                "duplicate_row_count": 6, "missing_cell_ratio": 0.141,
                "columns": [ { "name": "salary", "inferred_type": "numeric",
                               "missing": 13, "unique": 104, "constant": false,
                               "stats": { "mean": 77411.5, "std": 904512.4, ... } } ] },
  "issues":   [ { "type": "inconsistent_types", "severity": "high", "column": "age",
                  "count": 2, "ratio": 0.0168, "examples": ["twenty-five", "??"],
                  "message": "..." } ],
  "anomalies":[ { "method": "iqr", "column": "salary", "count": 1,
                  "bounds": { "lower": -45043.75, "upper": 229756.25 } },
                { "method": "isolation_forest", "column": null, "count": 3,
                  "row_indices": [10, 11, 42] } ],
  "quality_score": { "score": 42, "grade": "D",
                     "breakdown": [ { "factor": "completeness", "penalty": 25.0,
                                      "detail": "168 missing cells (14.1% of all cells)." } ] },
  "explanation": { "provider": "mock", "text": "Quality verdict: ..." }
}
```

## Quality score model

Start at 100, subtract capped deterministic penalties, clamp to 0–100:

| Factor       | Cap | Penalty                                             |
| ------------ | --- | --------------------------------------------------- |
| completeness | 25  | missing-cell ratio × 250                            |
| uniqueness   | 20  | duplicate-row ratio × 200                           |
| validity     | 15  | inf-cell ratio × 100 + 5 per empty/constant column  |
| consistency  | 10  | type-mismatch/sentinel cells / row count × 50       |
| anomalies    | 30  | union of flagged rows / row count × 200             |

Grade bands: A ≥ 90, B ≥ 75, C ≥ 60, D ≥ 40, F < 40. The reference samples score
**95/A** (clean) and **42/D** (dirty).

## AI providers

| `AI_PROVIDER` | Behaviour                                                        |
| ------------- | ---------------------------------------------------------------- |
| `mock` (default) | Deterministic template rendered from the computed analysis — offline, no key, byte-identical across runs. |
| `openai`      | `httpx` POST to `{OPENAI_BASE_URL}/chat/completions` — works with OpenAI, a local Ollama, or any compatible gateway. |

Design rule: the prompt frames the computed analysis as *data* and forbids inventing
numbers — the AI explains, code computes. Any provider error (bad key, timeout,
malformed output) falls back to the mock template and the response is labelled
`"provider": "mock (fallback: <reason>)"`. The report never fails because of the AI.

```bash
cp .env.example .env        # then edit; AI_PROVIDER=mock needs nothing
```

## Sample datasets

Synthetic only — regenerate with `uv run python samples/generate_samples.py` (seeded,
deterministic output):

- `customers_clean.csv` — 80 tidy rows × 8 columns → score ≈ 95 (A), zero issues.
- `customers_dirty.csv` — 119 rows × 10 columns with planted defects: ~8% missing
  cells, 6 exact duplicates, `?`/`-`/`unknown` sentinels, mixed-type ages
  (`"twenty-five"`), a `9,999,999` salary outlier plus an `inf` cell, a constant
  column and an empty column → score ≈ 42 (D), all 7 issue types and all 3 anomaly
  methods fire.

## Testing

```bash
uv run pytest          # 37 tests: profiler, quality, anomaly, scoring, API
```

The API tests drive the real FastAPI stack through `TestClient` (no separate test
server, no asyncio plugin needed) and verify, among other things, that two identical
uploads produce byte-identical mock explanations.

## Design notes

- **Deterministic-first**: every number in the report is computed in code; the AI
  layer is strictly an explainer with a guaranteed offline fallback.
- **Pandas 3 ready**: type checks use `pandas.api.types` helpers rather than dtype
  names, stats run on finite values only, and ragged CSVs are rejected explicitly
  (pandas 3 silently pads mismatched rows).
- **Bounded responses**: row indices are serialized capped (10) while scoring uses
  the full flagged set internally.

## License

[MIT](LICENSE)
