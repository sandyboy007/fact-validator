# Ablation Study Summary

- Generated UTC: 2026-07-02T02:22:20.202846
- Test claims: 5000
- Train claims: 11986
- Full model variant: full_proxy

| Variant | Accuracy | Macro F1 | Delta Accuracy vs Full | Delta Macro F1 vs Full |
|---|---:|---:|---:|---:|
| full_proxy | 0.508 | 0.438 | +0.000 | +0.000 |
| ablate_credibility | 0.497 | 0.439 | -0.011 | +0.001 |
| ablate_semantic_rerank | 0.507 | 0.437 | -0.001 | -0.001 |
| ablate_debate | 0.513 | 0.443 | +0.005 | +0.004 |
| ablate_quality_filter | 0.509 | 0.439 | +0.000 | +0.000 |
| tune_fever | 0.510 | 0.442 | +0.002 | +0.004 |

## Component Impact

| Component Removed | Relative Importance (%) | Accuracy Drop (%) | Prediction Change Rate (%) |
|---|---:|---:|---:|
| credibility_scoring | 2.13 | 2.13 | 7.50 |
| semantic_reranking | 0.28 | 0.28 | 2.14 |
| debate_mode | -1.02 | -1.02 | 3.00 |
| source_quality_filtering | -0.08 | -0.08 | 0.04 |
| fever_tuning | -0.31 | -0.31 | 0.68 |