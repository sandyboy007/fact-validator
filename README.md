# Fact Validator

> An AI-powered, source-aware fact-checking system that extracts claims from any article or URL, searches for corroborating or refuting evidence across the web, scores source credibility, and delivers a transparent misinformation-likelihood report.

[![CI](https://github.com/sandyboy007/fact-validator/actions/workflows/ci.yml/badge.svg)](https://github.com/sandyboy007/fact-validator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Frontend Experience](#frontend-experience)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
  - [1 · Clone the repository](#1--clone-the-repository)
  - [2 · Configure environment variables](#2--configure-environment-variables)
  - [3 · Start the API](#3--start-the-api)
  - [4 · Start the Web app](#4--start-the-web-app)
- [Docker Setup](#docker-setup)
- [Environment Variables Reference](#environment-variables-reference)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [CI / CD](#ci--cd)
- [Contributing](#contributing)
- [Code Style](#code-style)
- [Commit Convention](#commit-convention)
- [License](#license)

---

## Overview

Fact Validator takes a **URL or free text** as input and performs the following pipeline automatically:

1. **Fetch & extract** — renders the page and strips boilerplate using Trafilatura.
2. **Claim extraction** — NLP sentence scoring selects the most fact-like claims (up to 6 per article).
3. **Evidence search** — each claim is queried against Google via SerpAPI; results are fetched and ranked.
4. **Source credibility scoring** — every evidence domain is scored 0–100 using a transparent, multi-signal rubric:
  - Heuristic trust signals (reputable domains, institutional suffixes, platform risk markers)
  - OpenSources domain-type tags (e.g., satire, conspiracy, fake) mapped to score deltas
  - Iffy Index (MBFC-backed factuality levels) mapped to score penalties
5. **Verdict** — a baseline NLP verifier classifies each claim as `SUPPORTED`, `REFUTED`, or `NEI` (Not Enough Information), with a confidence score.
6. **Legacy heuristic score** — retained in API exports for backwards compatibility only; it is not a calibrated probability and is not shown as a decision score in the interface.
7. **Persistent storage** — every run is written to a local SQLite database; past results are browsable and exportable.

### Live application versus thesis proxy

The repository contains two related but distinct evaluation surfaces:

- **Fact Validator live application:** the implemented open-web system
  described above.
- **FactValidator-Proxy:** the deterministic model evaluated on the frozen
  5,000-claim thesis benchmark.

The 5,000-claim experiment combines lexical classification, category-level
priors, heuristic semantic signals, deterministic arbitration rules, and a
quality filter. It does **not** execute live SerpAPI retrieval, live
domain-level credibility scoring, SentenceTransformer reranking, or Ollama
debate for every test claim. Benchmark results are therefore labelled proxy
results throughout the final thesis package.

---

## Frontend Experience

The web app now supports two user-focused interface modes:

### User View (default)

- Designed for non-technical users.
- Prioritizes a verdict-first flow with plain-language explanations.
- Hides technical API/report links from the top action bar.
- Keeps advanced evaluation dashboards out of the main reading path.
- Recent verification history is collapsed under a dropdown near the bottom to reduce clutter.

### Analyst View

- Designed for research and thesis workflows.
- Exposes advanced tabs for:
  - Evaluation
  - Operations
  - Governance
  - Defense
- Includes richer comparative and operational summaries used for benchmarking and reporting.
- Makes developer/report endpoints (for example API docs and report JSON links) available for deep inspection.

### Top Navigation Behavior

- **Source Checker** is the primary user-facing quick action.
- Technical/report actions are intentionally de-emphasized in User View.
- Full structured verification controls remain available in the main input panel at the top.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser / Client                     │
│          Next.js 16 · React 19 · Tailwind CSS 4         │
│                  http://localhost:3000                    │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP (REST JSON)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Service                        │
│                 http://localhost:8000                    │
│                                                         │
│  POST /analyze          ← main analysis pipeline        │
│  GET  /source/{domain}  ← credibility score lookup      │
│  GET  /runs             ← list saved runs               │
│  GET  /runs/{id}        ← single saved run              │
│  GET  /runs-export      ← bulk JSON export              │
│  GET  /health           ← liveness probe                │
│                                                         │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │  credibility │  │  debate (LLM)  │  │   storage   │ │
│  │  rubric      │  │  Ollama opt.   │  │   SQLite    │ │
│  └──────────────┘  └────────────────┘  └─────────────┘ │
└──────────────┬─────────────────────────────────────────-┘
               │ SerpAPI (Google Search)
               ▼
         📡 External Web
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | [Next.js 16](https://nextjs.org), React 19, TypeScript 5, Tailwind CSS 4 |
| Backend | [FastAPI](https://fastapi.tiangolo.com), Python 3.10+, Uvicorn |
| NLP / Extraction | [Trafilatura](https://trafilatura.readthedocs.io), NLTK (punkt tokeniser) |
| Search | [SerpAPI](https://serpapi.com) (Google Search) |
| Domain resolution | [tldextract](https://github.com/john-kurkowski/tldextract) |
| LLM verifier (optional) | [Ollama](https://ollama.com) — `llama3.1:8b` (debate mode) |
| Storage | SQLite via Python `sqlite3` / SQLModel |
| Infrastructure | Docker Compose (Postgres + Redis — optional services) |
| Testing | pytest, pytest-asyncio |
| Linting | Ruff (Python), ESLint + tsc (TypeScript) |
| CI | GitHub Actions |

---

## Prerequisites

Ensure the following are installed before you begin:

| Tool | Version | Download |
|---|---|---|
| Python | ≥ 3.10 | https://python.org |
| Node.js | ≥ 20 LTS | https://nodejs.org |
| npm | ≥ 10 | bundled with Node.js |
| Git | any recent | https://git-scm.com |
| Ollama *(optional)* | latest | https://ollama.com |

> **SerpAPI key** — required for live evidence search. Get a free key at https://serpapi.com (100 free searches/month on the free tier).

---

## Project Structure

```
fact-validator/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── apps/
│   └── web/                    # Next.js frontend
│       ├── app/
│       │   ├── page.tsx        # Main analysis UI
│       │   ├── run/[id]/       # Saved-run detail page
│       │   └── source/         # Source credibility lookup page
│       ├── components/
│       │   └── ui.tsx          # Shared UI primitives
│       ├── next.config.ts
│       ├── package.json
│       └── tsconfig.json
├── services/
│   └── api/                    # FastAPI backend
│       ├── app/
│       │   ├── main.py         # Routes, pipeline orchestration
│       │   ├── credibility.py  # Domain credibility rubric + cache
│       │   ├── debate.py       # Ollama LLM debate verifier (optional)
│       │   ├── storage.py      # SQLite persistence helpers
│       │   └── source_routes.py# /source endpoint router
│       ├── data/
│       │   ├── domain_cache.json  # 14-day credibility score cache
│       │   ├── opensources.json   # OpenSources domain-label snapshot
│       │   └── iffy_index.json    # Iffy Index domain snapshot
│       ├── tests/
│       │   └── test_*.py    # Backend unit, integration, and research tests
│       ├── requirements.in  # Direct Python dependencies
│       └── requirements.lock # Hash-pinned resolved environment
├── infra/
│   └── docker-compose.yml      # Optional Postgres + Redis services
├── data/                       # SQLite DB written here at runtime
├── docs/                       # Additional documentation
├── .gitignore
└── README.md
```

---

## Local Setup

### 1 · Clone the repository

```bash
git clone https://github.com/sandyboy007/fact-validator.git
cd fact-validator
```

---

### 2 · Configure environment variables

#### Backend (`services/api/.env`)

```bash
cp services/api/.env.example services/api/.env   # if the example file exists
# or create it manually:
```

```dotenv
# ── Required ──────────────────────────────────────────────────────────────────
SERPAPI_API_KEY=your_serpapi_key_here

# ── Optional: Ollama LLM debate verifier ──────────────────────────────────────
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b

# ── Optional: override default paths ──────────────────────────────────────────
FACT_VALIDATOR_DB=./data/fact_validator.db
FACTVALIDATOR_DOMAIN_CACHE=C:/Fact_Validator/services/api/data/domain_cache.json
```

> `FACT_VALIDATOR_DB` is the canonical SQLite path variable. The legacy alias `FACTVALIDATOR_DB` is still accepted for backward compatibility.

> The app runs without `SERPAPI_API_KEY` — evidence search will be skipped and all claims will resolve to `NEI`.

> Set `FACTVALIDATOR_NLI_ENABLED=true` to enable the local MNLI relation classifier. The first request downloads the model named by `FACTVALIDATOR_NLI_MODEL` (default: `facebook/bart-large-mnli`). Without it, the API reports `heuristic-fallback` in each evidence item's `relation_classifier` metadata.

---

### 3 · Start the API

```bash
cd services/api

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install the hash-pinned environment (Python 3.10)
pip install --require-hashes -r requirements.lock

# Start the development server (hot-reload enabled)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at **http://127.0.0.1:8000**.  
Interactive API docs (Swagger UI): **http://127.0.0.1:8000/docs**.

---

### 4 · Start the Web app

Open a new terminal:

```bash
cd apps/web

# Install dependencies
npm install

# Start the dev server
npm run dev
```

If the frontend needs to talk to a remote or deployed backend, set `NEXT_PUBLIC_API_BASE_URL` in `apps/web/.env.local` before starting the web app. The default remains `http://127.0.0.1:8000` for local development.

The UI will be available at **http://localhost:3000**.

---

## Optional Infrastructure Containers

The `docker-compose.yml` in `infra/` provides optional Postgres and Redis containers (reserved for future persistence layers):

```bash
cd infra
docker compose up -d
```

| Service | Port | Purpose |
|---|---|---|
| `postgres` | 5432 | Optional relational DB (future) |
| `redis` | 6379 | Optional cache layer (future) |

> The application does not connect to these containers. It currently uses
> SQLite and an in-process/file cache. They are development placeholders, not
> evidence of a containerized application deployment.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `SERPAPI_API_KEY` | Recommended | `""` | SerpAPI key for Google Search evidence fetching |
| `OLLAMA_BASE_URL` | No | `http://127.0.0.1:11434` | Base URL of a running Ollama instance |
| `OLLAMA_MODEL` | No | `llama3.1:8b` | Ollama model name used for debate-mode verdicts |
| `FACT_VALIDATOR_DB` | No | `<repo>/data/fact_validator.db` | SQLite database file path |
| `FACTVALIDATOR_DB` | Legacy alias | `""` | Backward-compatible alias for `FACT_VALIDATOR_DB` |
| `FACTVALIDATOR_DOMAIN_CACHE` | No | `services/api/data/domain_cache.json` | Credibility score cache file path |

---

## API Reference

### `POST /analyze`

Run the full fact-checking pipeline on a URL or free text.

**Request body**

```jsonc
{
  "url": "https://example.com/article",   // mutually exclusive with text
  "text": "Your article text here...",     // mutually exclusive with url
  "mode": "live",                          // "live" | "snapshot"
  "verifier": "baseline",                 // "baseline" | "debate" (requires Ollama)
  "max_claims": 6,                        // 1–10, default 6
  "max_evidence_per_claim": 5,            // 1–10, default 5
  "min_source_score": 50,                 // filter evidence by credibility score
  "require_independent_domains": true     // enforce 2+ distinct domains for SUPPORTED
}
```

**Response (abbreviated)**

```jsonc
{
  "domain": "bbc.com",
  "domain_score": 80,
  "domain_label": "HIGH",
  "final_misinformation_likelihood": 0.23,
  "claims": [
    {
      "claim_text": "Global temperatures have risen by 1.5°C...",
      "verdict": "SUPPORTED",
      "confidence": 0.78,
      "evidence": [ { "url": "...", "domain": "reuters.com", "domain_score": 80, "snippet": "..." } ],
      "debate_summary": "Two independent high-credibility sources corroborate the claim."
    }
  ],
  "metadata": { "run_id": 42, "verifier": "baseline", "serpapi_enabled": true }
}
```

---

### `GET /source/{domain}`

Returns the credibility score and reasoning for a domain.

```bash
curl http://127.0.0.1:8000/source/forbes.com
```

```jsonc
{
  "domain": "forbes.com",
  "score": 75,
  "label": "MEDIUM",
  "reasons": {
    "whitelist": "Domain appears in local reputable-source list (+25).",
    "opensources": "Domain found in OpenSources unreliable-news dataset (-5).",
    "iffy_index": "Domain flagged by Iffy Index (MBFC factual rating: Low, -15)."
  }
}
```

Notes:
- `reasons` is sparse by design; only applicable signals are returned for a domain.
- OpenSources and Iffy effects are additive with heuristic rules, then clamped to `0..100`.

---

### `GET /runs`

Lists the most recent analysis runs (default limit: 50).

```bash
curl "http://127.0.0.1:8000/runs?limit=20"
```

---

### `GET /runs/{id}`

Returns the full saved response for a single run by ID.

---

### `GET /runs-export`

Bulk-exports all runs as JSON (useful for research / thesis data collection).

---

### `GET /health`

Liveness probe — returns `{"status": "ok"}`.

---

## Running Tests

All tests run **without** a network connection, SerpAPI key, database, or Ollama instance.

```bash
cd services/api

# Activate your virtual environment first
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Install the reproducible backend environment
pip install --require-hashes -r requirements.lock

# Run the full suite
pytest tests/ -v

# Run a specific test class
pytest tests/test_smoke.py::TestScoreDomainRubric -v

# Run with coverage (requires pytest-cov)
pip install pytest-cov
pytest tests/ --cov=app --cov-report=term-missing
```

Recorded correction-branch verification (Python 3.10.0):

```text
$ pytest --collect-only -q
164 tests collected
$ pytest services/api/tests -q
164 passed
```

Warning counts and timing vary with the local environment. On 31 July 2026,
the current Windows environment reproduced 164 passed with 6 warnings in
77.15 seconds. The machine-readable historical record is stored in
`data/benchmarks/results_5000/reproducibility_audit_report.json`.

---

## CI / CD

The workflow is configured for pushes and pull requests targeting **`main`**.
At the time of this thesis revision, the manuscript correction branch had not
yet been merged into `main`; repository claims therefore identify the exact
branch and commit rather than implying that `main` contains them.
defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job | Runner | Steps |
|---|---|---|
| `api-test` | `ubuntu-latest` / Python 3.10 | Install locked dependencies, lint, and run backend tests |
| `thesis-artifacts` | `ubuntu-latest` / Python 3.10 | Regenerate statistics and validate frozen thesis artifacts |
| `web-build` | `ubuntu-latest` / Node 20 | `npm ci` → ESLint → `tsc --noEmit` → `next build` |

The pipeline validates backend tests, the web build, and the frozen thesis
artifacts. The thesis job regenerates the statistics and fails when split
isolation, prediction alignment, hashes, required manifest fields, or committed
reports differ.

---

## Contributing

Contributions are welcome. Please follow the steps below.

### Getting started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/fact-validator.git
   cd fact-validator
   ```
3. **Create a branch** from `main` with a descriptive name:
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```
4. Follow the [Local Setup](#local-setup) instructions to get the project running.

---

### Making changes

- Keep each pull request focused on **one concern** — a single feature, bug fix, or refactor.
- Add or update tests in `services/api/tests/` for any backend logic you change.
- Run the full test suite before opening a PR and ensure **all tests pass**:
  ```bash
  pytest tests/ -v
  ```
- For frontend changes, verify the build succeeds:
  ```bash
  cd apps/web && npm run build
  ```
- If you update credibility logic (`whitelist`, OpenSources mapping, or Iffy penalties) in `credibility.py`, add corresponding tests in `TestScoreDomainRubric`.

---

### Extending source credibility signals

Credibility scoring combines three signal groups in `services/api/app/credibility.py`:
- Heuristic whitelist/platform rules in `score_domain_rubric()`
- OpenSources type-label deltas (`_OS_TYPE_DELTA`) backed by `services/api/data/opensources.json`
- Iffy Index level deltas (`_IFFY_LEVEL_DELTA`) backed by `services/api/data/iffy_index.json`

Whitelist entries use this shape:

```python
"domain.tld": <bonus_points>,   # 10–35 pts above the 50-point baseline
```

Guidelines for assigning bonus points:

| Bonus | Description |
|---|---|
| 30–35 | Major wire services, national public broadcasters, top-tier academic journals |
| 25 | National newspapers of record, well-established broadcast networks |
| 20 | International news organisations, major NGOs, established think tanks |
| 15 | Trade publications, mixed-quality aggregators with editorial oversight |
| 10 | Open-access platforms with basic peer review |

Entries must have a corresponding test asserting `score >= <expected_minimum>`.

---

### Pull request checklist

Before submitting a pull request, confirm the following:

- [ ] Code is on a feature branch, **not** `main`
- [ ] `pytest tests/ -v` passes with **0 failures**
- [ ] `npm run build` completes with **0 errors** (frontend changes)
- [ ] No `.venv/`, `__pycache__/`, `.pyc`, or `.env` files are included in the diff
- [ ] Commit messages follow the [Commit Convention](#commit-convention)
- [ ] PR description explains **what** changed and **why**

---

## Code Style

### Python

- Formatter: [Black](https://black.readthedocs.io) (line length 100) — `black app/ tests/`
- Linter: [Ruff](https://docs.astral.sh/ruff/) — `ruff check app/ tests/`
- Type hints are encouraged on all public functions. Use `from __future__ import annotations`.

```bash
pip install black ruff
black app/ tests/
ruff check app/ tests/ --fix
```

### TypeScript / React

- Formatter: [Prettier](https://prettier.io) (if added to the project)
- Linter: ESLint with `eslint-config-next`

```bash
cd apps/web
npm run lint
npx tsc --noEmit
```

---

## Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

| Type | Use for |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Maintenance, tooling, config |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `docs` | Documentation only changes |
| `ci` | Changes to CI configuration |
| `perf` | Performance improvement |

**Examples:**

```
feat(credibility): add Bloomberg and WSJ to whitelist
fix(baseline): lower overlap threshold for single high-cred source
test(smoke): add label consistency test for all whitelist domains
docs: add full README with setup and contribution guide
chore: remove tracked .venv and __pycache__ from git index
```

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

> Built as part of an academic thesis on automated misinformation detection. Credibility scores are heuristic signals, not ground truth. Always verify important claims through primary sources.
