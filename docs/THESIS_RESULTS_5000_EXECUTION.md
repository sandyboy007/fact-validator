# 5000-Claim Execution and Results (Final)

## What Was Executed

The full external-data to evaluation workflow was executed end-to-end on 2026-07-02.

### 1. External dataset ingestion (automated)

Script:
- `services/api/Scripts/fetch_external_public_benchmarks.py`

Generated normalized CSV inputs:
- `data/benchmarks/external_templates/fever_filled.csv` (8000 rows)
- `data/benchmarks/external_templates/liar_filled.csv` (6000 rows)
- `data/benchmarks/external_templates/scifact_filled.csv` (1200 rows)
- `data/benchmarks/external_templates/healthver_filled.csv` (6000 rows)

Note:
- HealthVer was not directly available via this ingestion route, so `health_fact` was used as a health-domain substitute and normalized into the `healthver_filled.csv` slot for pipeline compatibility.

### 2. 5000 benchmark construction

Command path:
- `services/api/Scripts/run_5000_benchmark_pipeline.py`

Output manifest:
- `data/benchmarks/results/large_benchmark_manifest.json`

Result:
- Retained unique claims: **19983**
- Test claims: **5000**
- Train claims: **11986**
- Validation claims: **2997**

Test label distribution:
- SUPPORTED: 2463
- REFUTED: 1371
- NEI: 1166

### 3. Architecture comparison on 5000 split

Command path:
- `services/api/Scripts/run_benchmark_architecture_suite.py`

Output directory:
- `data/benchmarks/results_5000/`

Primary summary:
- `data/benchmarks/results_5000/comparative_analysis_summary.md`

## Corrected 5000 Comparative Results

Generated UTC: `2026-07-02T02:05:16.689296`
Claims compared: `5000`

| System | Accuracy | 95% CI |
|---|---:|---:|
| length | 0.494 | [0.480, 0.508] |
| majority | 0.493 | [0.479, 0.506] |
| ablate_quality_filter | 0.347 | [0.334, 0.360] |
| random | 0.332 | [0.319, 0.345] |
| sentiment | 0.238 | [0.226, 0.250] |
| full_proxy | 0.236 | [0.224, 0.248] |
| ablate_semantic_rerank | 0.236 | [0.224, 0.248] |
| ablate_debate | 0.236 | [0.224, 0.248] |
| keyword | 0.235 | [0.224, 0.247] |
| ablate_credibility | 0.234 | [0.222, 0.246] |

### Full system vs comparators

`full_proxy` differences (percentage points):

- vs random: -9.58 pp (not significant)
- vs keyword: +0.08 pp (not significant)
- vs length: -25.80 pp (not significant)
- vs sentiment: -0.16 pp (not significant)
- vs majority: -25.64 pp (not significant)

Debate lift:
- Accuracy delta (full - no-debate): +0.00 pp
- Prediction change rate: 0.011

## Important methodological note

A large-n overflow bug in `run_comparative_analysis.py` sign-test computation was fixed (log-space implementation), and the 5000 reports were regenerated after the fix.

A claim-id alignment bug (missing `id` in generated split claims) was also fixed in the 5000 benchmark builder and loaders before re-running metrics.

## Publication-safe interpretation

- The 5000-sample benchmark is now available and reproducibly generated.
- On this benchmark, the current `full_proxy` architecture does **not** outperform the strongest baselines (`length`/`majority`) by accuracy.
- This supports a defensible publication framing around:
  1. reproducible benchmark pipeline,
  2. transparent architecture and ablation protocol,
  3. honest reporting of comparative performance and limitations.
