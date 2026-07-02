# Comparative Analysis Summary

- Generated UTC: 2026-07-02T02:22:22.910689
- Full system variant: full_proxy
- Claims compared: 5000

## System Ranking

| System | Accuracy | 95% CI | Avg Confidence | Calibration Error | ECE |
|---|---:|---:|---:|---:|---:|
| ablate_debate | 0.513 | [0.500, 0.527] | 88.4 | 0.370 | 0.370 |
| tune_fever | 0.510 | [0.496, 0.524] | 88.5 | 0.376 | 0.376 |
| ablate_quality_filter | 0.509 | [0.495, 0.522] | 88.3 | 0.375 | 0.375 |
| full_proxy | 0.508 | [0.494, 0.522] | 88.3 | 0.375 | 0.375 |
| ablate_semantic_rerank | 0.507 | [0.493, 0.521] | 87.3 | 0.366 | 0.366 |
| ablate_credibility | 0.497 | [0.484, 0.511] | 78.7 | 0.289 | 0.289 |
| length | 0.494 | [0.480, 0.508] | 48.8 | 0.006 | 0.063 |
| majority | 0.493 | [0.479, 0.506] | 50.0 | 0.007 | 0.007 |
| random | 0.332 | [0.319, 0.345] | 50.0 | 0.168 | 0.168 |
| sentiment | 0.238 | [0.226, 0.250] | 40.3 | 0.165 | 0.165 |
| keyword | 0.235 | [0.224, 0.247] | 23.6 | 0.001 | 0.060 |

## Full System vs Comparators

| Comparator | Delta Accuracy (pp) | p-value | Cohen's d | Significant (0.05) |
|---|---:|---:|---:|:---:|
| random | +17.62 | 0.0000 | 0.253 | yes |
| keyword | +27.28 | 0.0000 | 0.396 | yes |
| length | +1.40 | 0.0231 | 0.029 | yes |
| sentiment | +27.04 | 0.0000 | 0.393 | yes |
| majority | +1.56 | 0.0216 | 0.029 | yes |
| ablate_credibility | +1.08 | 0.0005 | 0.047 | yes |
| ablate_semantic_rerank | +0.14 | 0.2383 | 0.012 | no |
| ablate_debate | -0.52 | 0.9975 | -0.038 | no |
| ablate_quality_filter | -0.04 | 1.0000 | -0.020 | no |
| tune_fever | -0.16 | 0.9793 | -0.025 | no |

## Debate Lift

- Variant compared: ablate_debate
- Accuracy delta (full - no-debate): -0.52 pp
- Prediction change rate: 0.030