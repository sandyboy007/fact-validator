# Baseline Comparison Summary

- Generated UTC: 2026-07-02T00:34:32.642843
- Test claims: 48
- Train claims: 133

| Baseline | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| random | 0.375 | 0.366 | 0.364 | 0.364 |
| keyword | 0.292 | 0.177 | 0.319 | 0.197 |
| length | 0.354 | 0.217 | 0.298 | 0.227 |
| sentiment | 0.292 | 0.097 | 0.333 | 0.151 |
| majority | 0.417 | 0.139 | 0.333 | 0.196 |

## Majority Baseline

- Computed majority class from train split: **SUPPORTED**