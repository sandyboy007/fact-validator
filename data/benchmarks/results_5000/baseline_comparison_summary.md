# Baseline Comparison Summary

- Generated UTC: 2026-07-02T02:05:12.343401
- Test claims: 5000
- Train claims: 11986

| Baseline | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| random | 0.332 | 0.333 | 0.334 | 0.324 |
| keyword | 0.235 | 0.355 | 0.332 | 0.136 |
| length | 0.494 | 0.422 | 0.395 | 0.348 |
| sentiment | 0.238 | 0.296 | 0.333 | 0.140 |
| majority | 0.493 | 0.164 | 0.333 | 0.220 |

## Majority Baseline

- Computed majority class from train split: **SUPPORTED**