# Ablation Study Summary

- Generated UTC: 2026-07-02T02:05:14.113845
- Test claims: 5000
- Train claims: 11986
- Full model variant: full_proxy

| Variant | Accuracy | Macro F1 | Delta Accuracy vs Full | Delta Macro F1 vs Full |
|---|---:|---:|---:|---:|
| full_proxy | 0.236 | 0.142 | +0.000 | +0.000 |
| ablate_credibility | 0.234 | 0.134 | -0.002 | -0.008 |
| ablate_semantic_rerank | 0.236 | 0.137 | +0.000 | -0.005 |
| ablate_debate | 0.236 | 0.146 | +0.000 | +0.004 |
| ablate_quality_filter | 0.347 | 0.282 | +0.111 | +0.140 |

## Component Impact

| Component Removed | Relative Importance (%) | Accuracy Drop (%) | Prediction Change Rate (%) |
|---|---:|---:|---:|
| credibility_scoring | 1.02 | 1.02 | 1.96 |
| semantic_reranking | 0.00 | 0.00 | 1.44 |
| debate_mode | 0.00 | 0.00 | 1.10 |
| source_quality_filtering | -46.91 | -46.91 | 26.30 |