# Fact Validator Research Project Report

## Executive Summary

Fact Validator is a source-aware fact-checking system that combines web evidence retrieval, explicit source credibility scoring, optional LLM debate, sentiment-based risk adjustment, and persistent run storage. The project has evolved from a small proof-of-concept into a reproducible research codebase with benchmark preparation scripts, comparative analysis tooling, production metric synthesis, and a research-oriented frontend and API.

The strongest evidence available in the repository today comes from a genuine 224-claim benchmark assembled from two local research datasets and evaluated on a deterministic 48-claim holdout split. On that split, the full proxy system achieved 35.4% accuracy. The strongest simple baseline was the majority baseline at 41.7%, which shows that the current system is not yet superior on this dataset, but the credibility component materially improved performance relative to its ablation.

## What Exists Now

### Product and Research Surfaces

- Next.js frontend for URL/text claim verification and research dashboards.
- FastAPI backend for extraction, evidence retrieval, credibility scoring, debate, sentiment analysis, and result persistence.
- Research scripts for benchmark building, ablation studies, baseline comparison, comparative analysis, production metrics, and credibility calibration.
- Research manuscript draft and protocol notes for benchmarking, external comparisons, shortcut analysis, and reproducibility.

### Benchmark Assets

- Local seed benchmark: `docs/evaluation_benchmark.json`.
- Combined genuine research benchmark: `data/benchmarks/research_benchmark_224.json`.
- Canonicalized benchmark: `data/benchmarks/research_benchmark_224_canonical.json`.
- Deterministic splits: `data/benchmarks/splits_224/`.
- Evaluation outputs: `data/benchmarks/results_224/`.

## Architecture

### High-Level Flow

1. User submits a URL or text claim source through the web UI.
2. The API extracts text using Trafilatura and decomposes it into fact-like claims with NLP heuristics.
3. For each claim, the system retrieves search evidence and scores evidence domains using a transparent credibility rubric.
4. Evidence is reranked semantically and classified into `SUPPORTED`, `REFUTED`, or `NEI`.
5. Optional debate mode adds a second reasoning layer for uncertain claims.
6. Sentiment and bias analysis adjust the final misinformation likelihood.
7. The run is written to SQLite and cached in JSON for reproducibility.

### Backend Components

- `services/api/app/main.py`: orchestration, request/response models, analysis pipeline, and final score computation.
- `services/api/app/credibility.py`: domain credibility rubric with cache-backed scoring and external reputation signals.
- `services/api/app/analysis_features.py`: claim decomposition, verdict heuristics, evidence enrichment, and feature extraction.
- `services/api/app/semantic_retrieval.py`: semantic reranking of evidence.
- `services/api/app/sentiment.py`: sentiment, bias, and misinformation-risk adjustment.
- `services/api/app/debate.py`: optional LLM debate verifier.
- `services/api/app/storage.py`: SQLite persistence and export.
- `services/api/app/cache.py`: runtime and score caching.

### Frontend Components

- `apps/web/app/page.tsx`: main analysis interface.
- `apps/web/app/source/page.tsx`: source credibility explorer.
- `apps/web/app/dashboard/page.tsx`: research and operations dashboard.
- `apps/web/lib/api-base.ts`: local API routing and environment-aware API base selection.

## Research Work Completed

### 1. Benchmark Scaling and Preparation

The repository now supports deterministic benchmark preparation and split generation. The work includes scripts for exact-size benchmark assembly, canonical label normalization, and benchmark export. Locally available genuine claims were merged into a 224-claim research benchmark because the repo does not yet contain a 5000-claim corpus.

### 2. Comparative Evaluation

The project includes evaluation scripts and outputs for:

- baseline comparison
- ablation study
- comparative analysis
- production metrics synthesis

### 3. Credibility Scoring Calibration

The credibility rubric was improved so it does not collapse to a neutral 50 by default. Unknown domains now start more cautiously, and text-only analyses can derive a risk signal from the credibility of retrieved evidence domains rather than from a hard-coded midpoint.

### 4. Production Metrics

Production metrics are synthesized from evaluation outputs and include latency, throughput, cost, calibration, and error summaries. The backend also supports runtime telemetry fields for cache hit rate, CPU, memory, concurrency, recovery, and scaling when such telemetry is available.

## Measured Results So Far

All measured values below are from the 224-claim research benchmark, evaluated on a 48-claim holdout split.

### System Accuracy

- Full proxy system: 35.4% accuracy, macro F1 0.361.
- Majority baseline: 41.7% accuracy.
- Random baseline: 37.5% accuracy.
- Length heuristic: 35.4% accuracy.
- Keyword baseline: 29.2% accuracy.
- Sentiment baseline: 29.2% accuracy.

### Ablation Findings

- Removing credibility scoring reduced accuracy to 22.9%, indicating the credibility component contributes useful signal.
- Removing semantic reranking increased accuracy to 39.6% on this split, suggesting the reranker is not yet stable on the current benchmark distribution.
- Debate mode had no measurable lift on the current holdout.

### Comparative Significance

- Full system vs credibility ablation: statistically significant improvement at p = 0.0156.
- Full system vs majority baseline: no statistically significant improvement.
- Full system vs random baseline: no statistically significant improvement.

### Production Characteristics

- Baseline latency: 8.20 sec/claim.
- Debate latency: 72.00 sec/claim.
- Debate mode is 8.78x slower than baseline mode.
- Baseline throughput: 439.02 claims/hour.
- Debate throughput: 50.00 claims/hour.
- Estimated monthly cost reduction with caching: 71.43%.

## What These Results Mean

The current system is clearly more than a prototype, but it is not yet a state-of-the-art fact checker. The evidence shows:

- the architecture is operational and reproducible;
- the credibility module is genuinely useful;
- the reranking and debate layers need further tuning;
- the current benchmark is too small for strong superiority claims;
- a 5000-claim genuine benchmark still requires external public source data.

## Current Gaps

- The repository does not yet contain a genuine 5000-claim corpus.
- External-system prediction dumps for GPT-4o, Gemini, Claude, FacTool, FEVER baseline, and RAG baseline are not present.
- Shortcut-analysis outputs for the full intended benchmark size remain pending.
- Load-test telemetry is not yet captured as a dedicated runtime artifact.

## Recommended Next Steps

1. Ingest FEVER, LIAR, SciFact, and HealthVer style public data to reach the 5000-claim target.
2. Run the shortcut/perturbation/correlation analysis on the larger benchmark.
3. Collect prediction dumps from external systems and rerun the comparison harness.
4. Add load-testing telemetry so production metrics include real concurrency and recovery data.
5. Re-run the manuscript tables with the final benchmark outputs.

## Repository Status

The project is now in a strong research state: it has a working frontend/backend application, a measurable benchmark pipeline, a credible source-scoring subsystem, and a manuscript-ready set of evaluation artifacts. The limiting factor is now dataset scale rather than missing implementation.
