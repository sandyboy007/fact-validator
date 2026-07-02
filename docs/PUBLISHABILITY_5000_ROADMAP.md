# Publishability Roadmap: 5000-Claim Benchmark + Architecture Comparison

## Current Status (Verified)

- Current maximum unique benchmark claims available locally: **224**
- Target test size for publication-grade evaluation: **5000**
- Gap to target: **4776** claims

This was verified by running:

- `services/api/Scripts/build_large_test_benchmark.py`
- Inputs: `research_benchmark_v1.json`, `research_benchmark_v2.json`, `research_benchmark_224_canonical.json`
- Result: `Insufficient unique claims for a 5000-claim test set: found 224 after deduplication.`

## Current Architecture-vs-Architecture Accuracy (Latest Run)

Source artifact:
- `data/benchmarks/results_224/comparative_analysis_summary.md`
- Generated UTC: `2026-07-02T00:31:44.154584`
- Claims compared: `48`

### Ranking by Accuracy

| System | Accuracy | 95% CI |
|---|---:|---:|
| majority | 0.417 | [0.277, 0.556] |
| ablate_semantic_rerank | 0.396 | [0.257, 0.534] |
| ablate_quality_filter | 0.396 | [0.257, 0.534] |
| random | 0.375 | [0.238, 0.512] |
| length | 0.354 | [0.219, 0.489] |
| full_proxy (your full system proxy) | 0.354 | [0.219, 0.489] |
| ablate_debate | 0.354 | [0.219, 0.489] |
| keyword | 0.292 | [0.163, 0.420] |
| sentiment | 0.292 | [0.163, 0.420] |
| ablate_credibility | 0.229 | [0.110, 0.348] |

### Direct Comparison: Full System vs Baselines

- vs majority: **-6.25 pp** (not significant)
- vs random: **-2.08 pp** (not significant)
- vs length: **0.00 pp** (not significant)
- vs keyword: **+6.25 pp** (not significant)
- vs sentiment: **+6.25 pp** (not significant)

Interpretation:
- On the current 48-claim test split, your full proxy architecture is **not yet significantly better** than strong simple baselines.
- This is consistent with a small-sample regime where confidence intervals are wide.

## One-Command Architecture Comparison Runner (Added)

New script added:
- `services/api/Scripts/run_benchmark_architecture_suite.py`

What it does:
1. Runs baseline comparison
2. Runs ablation study
3. Runs comparative analysis

Example (current 224 split):

```bash
cd services/api
python Scripts/run_benchmark_architecture_suite.py \
  --train C:/Fact_Validator/data/benchmarks/splits_224/train.json \
  --test C:/Fact_Validator/data/benchmarks/splits_224/test.json \
  --output-dir C:/Fact_Validator/data/benchmarks/results_224
```

Example (future 5000 split):

```bash
cd services/api
python Scripts/run_benchmark_architecture_suite.py \
  --train C:/Fact_Validator/data/benchmarks/splits_5000/train.json \
  --test C:/Fact_Validator/data/benchmarks/splits_5000/test.json \
  --output-dir C:/Fact_Validator/data/benchmarks/results_5000
```

## What You Need To Reach a Real 5000 Benchmark

You need to provide external genuine claim datasets (with labels), for example:
- FEVER
- LIAR
- SciFact
- HealthVer

Then run:

```bash
cd services/api
python Scripts/run_5000_benchmark_pipeline.py \
  --input fever=C:/path/to/fever.json \
  --input liar=C:/path/to/liar.csv \
  --input scifact=C:/path/to/scifact.json \
  --input healthver=C:/path/to/healthver.csv \
  --target-test-size 5000 \
  --benchmark-output C:/Fact_Validator/data/benchmarks/results/large_benchmark_manifest.json \
  --splits-dir C:/Fact_Validator/data/benchmarks/splits_5000
```

After that, run architecture comparison on the 5000 test split:

```bash
cd services/api
python Scripts/run_benchmark_architecture_suite.py \
  --train C:/Fact_Validator/data/benchmarks/splits_5000/train.json \
  --test C:/Fact_Validator/data/benchmarks/splits_5000/test.json \
  --output-dir C:/Fact_Validator/data/benchmarks/results_5000
```

## Publication Guidance

For a publishable claim of comparative superiority, prioritize:
- At least 5000 genuine, deduplicated claims.
- Label-quality controls (dual annotation/adjudication if feasible).
- Confidence intervals and significance tests in all tables.
- Report both overall and per-domain metrics.

Do not claim superiority from the current 48-claim test setting.
