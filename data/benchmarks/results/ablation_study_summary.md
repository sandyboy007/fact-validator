# Ablation Study Summary

- Generated UTC: 2026-03-27T19:41:55.531733
- Test claims: 7
- Train claims: 11
- Full model variant: full_proxy

| Variant | Accuracy | Macro F1 | Delta Accuracy vs Full | Delta Macro F1 vs Full |
|---|---:|---:|---:|---:|
| full_proxy | 0.714 | 0.711 | +0.000 | +0.000 |
| ablate_credibility | 0.571 | 0.579 | -0.143 | -0.132 |
| ablate_semantic_rerank | 0.286 | 0.229 | -0.429 | -0.483 |
| ablate_debate | 0.714 | 0.675 | +0.000 | -0.037 |
| ablate_quality_filter | 0.714 | 0.711 | +0.000 | +0.000 |

## Component Impact

| Component Removed | Relative Importance (%) | Accuracy Drop (%) | Prediction Change Rate (%) |
|---|---:|---:|---:|
| credibility_scoring | 20.00 | 20.00 | 14.29 |
| semantic_reranking | 60.00 | 60.00 | 42.86 |
| debate_mode | 0.00 | 0.00 | 28.57 |
| source_quality_filtering | 0.00 | 0.00 | 14.29 |