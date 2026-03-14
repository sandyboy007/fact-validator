# Release Notes - 2026-03-14

## Highlights

### 1) Trust-aware fact-checking pipeline
- Added claim decomposition with atomic subclaims, entity extraction, numeric checks, year extraction, and expertise profile inference.
- Added evidence quality scoring with:
  - source credibility
  - recency weighting
  - directness overlap
  - primary-source detection
  - quote-grounding
  - anti-manipulation flags
- Added structured verdicts:
  - Supported
  - Likely supported
  - Mixed / disputed
  - Insufficient evidence
  - Likely false
  - False
- Added uncertainty reasons and human-review escalation signals.
- Added claim-memory cache support for re-check workflows.

Files:
- `services/api/app/analysis_features.py`
- `services/api/app/main.py`
- `services/api/app/storage.py`
- `services/api/tests/test_smoke.py`

### 2) Frontend trust signals and explainability UI
- Added display of structured verdict labels, uncertainty reasons, review-needed flags, and trust diagnostics.
- Added evidence-level fields: quality score, stance, source type, primary source signal, recency/directness, quote-grounded, entity/numeric match, and manipulation flags.
- Added benchmark link in the app UI.

File:
- `apps/web/app/page.tsx`

### 3) Credibility dataset integration
- Integrated OpenSources and curated Iffy Index data into credibility scoring.
- Added dataset files and loading logic.
- Cleared stale scoring behavior where many domains stayed near default 50.

Files:
- `services/api/app/credibility.py`
- `services/api/data/opensources.json`
- `services/api/data/iffy_index.json`

### 4) Evaluation benchmark
- Added a bundled benchmark corpus endpoint and dataset for regression/evaluation.

Files:
- `docs/evaluation_benchmark.json`
- `services/api/app/main.py` (`GET /evaluation/benchmark`)

## Recent commits
- `1a8804e` chore: update local fact validator database
- `5ac38c4` feat: add trust-aware fact-checking pipeline
- `e8d6aa8` feat: integrate OpenSources + Iffy Index for genuine domain credibility scores
- `5a9ca39` fix: source credibility scores now show genuine values
- `88ff5eb` fix(credibility): replace hardcoded Windows CACHE_PATH with portable __file__-relative default

## Notes
- Added `.gitignore` protection for local DB files under `data/*.db` to avoid committing local runtime state.
