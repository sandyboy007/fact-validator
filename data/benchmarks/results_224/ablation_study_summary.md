# Ablation Study Summary

- Generated UTC: 2026-07-01T00:36:56.505238
- Test claims: 48
- Train claims: 133
- Full model variant: full_proxy

| Variant | Accuracy | Macro F1 | Delta Accuracy vs Full | Delta Macro F1 vs Full |
|---|---:|---:|---:|---:|
| full_proxy | 0.354 | 0.361 | +0.000 | +0.000 |
| ablate_credibility | 0.229 | 0.181 | -0.125 | -0.180 |
| ablate_semantic_rerank | 0.396 | 0.382 | +0.042 | +0.021 |
| ablate_debate | 0.354 | 0.361 | +0.000 | +0.000 |
| ablate_quality_filter | 0.396 | 0.321 | +0.042 | -0.040 |

## Component Impact

| Component Removed | Relative Importance (%) | Accuracy Drop (%) | Prediction Change Rate (%) |
|---|---:|---:|---:|
| credibility_scoring | 35.29 | 35.29 | 12.50 |
| semantic_reranking | -11.76 | -11.76 | 12.50 |
| debate_mode | 0.00 | 0.00 | 0.00 |
| source_quality_filtering | -11.76 | -11.76 | 31.25 |