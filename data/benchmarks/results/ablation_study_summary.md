# Ablation Study Summary

- Generated UTC: 2026-03-31T19:42:34.619687
- Test claims: 51
- Train claims: 143
- Full model variant: full_proxy

| Variant | Accuracy | Macro F1 | Delta Accuracy vs Full | Delta Macro F1 vs Full |
|---|---:|---:|---:|---:|
| full_proxy | 0.216 | 0.212 | +0.000 | +0.000 |
| ablate_credibility | 0.137 | 0.102 | -0.078 | -0.110 |
| ablate_semantic_rerank | 0.235 | 0.225 | +0.020 | +0.013 |
| ablate_debate | 0.216 | 0.212 | +0.000 | +0.000 |
| ablate_quality_filter | 0.196 | 0.177 | -0.020 | -0.034 |

## Component Impact

| Component Removed | Relative Importance (%) | Accuracy Drop (%) | Prediction Change Rate (%) |
|---|---:|---:|---:|
| credibility_scoring | 36.36 | 36.36 | 7.84 |
| semantic_reranking | -9.09 | -9.09 | 1.96 |
| debate_mode | 0.00 | 0.00 | 0.00 |
| source_quality_filtering | 9.09 | 9.09 | 5.88 |