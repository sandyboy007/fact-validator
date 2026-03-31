# Baseline Comparison Summary

- Generated UTC: 2026-03-31T19:42:34.430917
- Test claims: 51
- Train claims: 143

| Baseline | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| random | 0.373 | 0.383 | 0.374 | 0.375 |
| keyword | 0.294 | 0.218 | 0.322 | 0.215 |
| length | 0.314 | 0.231 | 0.311 | 0.242 |
| sentiment | 0.294 | 0.098 | 0.333 | 0.152 |
| majority | 0.353 | 0.118 | 0.333 | 0.174 |

## Majority Baseline

- Computed majority class from train split: **REFUTED**