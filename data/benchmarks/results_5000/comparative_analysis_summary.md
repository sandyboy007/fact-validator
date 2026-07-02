# Comparative Analysis Summary

- Generated UTC: 2026-07-02T02:05:16.689296
- Full system variant: full_proxy
- Claims compared: 5000

## System Ranking

| System | Accuracy | 95% CI | Avg Confidence | Calibration Error | ECE |
|---|---:|---:|---:|---:|---:|
| length | 0.494 | [0.480, 0.508] | 48.8 | 0.006 | 0.063 |
| majority | 0.493 | [0.479, 0.506] | 50.0 | 0.007 | 0.007 |
| ablate_quality_filter | 0.347 | [0.334, 0.360] | 36.1 | 0.014 | 0.050 |
| random | 0.332 | [0.319, 0.345] | 50.0 | 0.168 | 0.168 |
| sentiment | 0.238 | [0.226, 0.250] | 40.3 | 0.165 | 0.165 |
| full_proxy | 0.236 | [0.224, 0.248] | 47.2 | 0.236 | 0.240 |
| ablate_semantic_rerank | 0.236 | [0.224, 0.248] | 45.1 | 0.215 | 0.220 |
| ablate_debate | 0.236 | [0.224, 0.248] | 47.2 | 0.236 | 0.240 |
| keyword | 0.235 | [0.224, 0.247] | 23.6 | 0.001 | 0.060 |
| ablate_credibility | 0.234 | [0.222, 0.246] | 30.9 | 0.075 | 0.076 |

## Full System vs Comparators

| Comparator | Delta Accuracy (pp) | p-value | Cohen's d | Significant (0.05) |
|---|---:|---:|---:|:---:|
| random | -9.58 | 1.0000 | -0.151 | no |
| keyword | +0.08 | 0.3620 | 0.007 | no |
| length | -25.80 | 1.0000 | -0.362 | no |
| sentiment | -0.16 | 0.7905 | -0.010 | no |
| majority | -25.64 | 1.0000 | -0.319 | no |
| ablate_credibility | +0.24 | 0.0740 | 0.022 | no |
| ablate_semantic_rerank | +0.00 | 0.5598 | 0.000 | no |
| ablate_debate | +0.00 | 0.5612 | 0.000 | no |
| ablate_quality_filter | -11.08 | 1.0000 | -0.255 | no |

## Debate Lift

- Variant compared: ablate_debate
- Accuracy delta (full - no-debate): +0.00 pp
- Prediction change rate: 0.011