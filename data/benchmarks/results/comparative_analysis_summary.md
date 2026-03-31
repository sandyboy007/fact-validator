# Comparative Analysis Summary

- Generated UTC: 2026-03-31T19:42:34.831016
- Full system variant: full_proxy
- Claims compared: 51

## System Ranking

| System | Accuracy | 95% CI | Avg Confidence | Calibration Error | ECE |
|---|---:|---:|---:|---:|---:|
| random | 0.373 | [0.240, 0.505] | 47.7 | 0.104 | 0.134 |
| majority | 0.353 | [0.222, 0.484] | 50.0 | 0.147 | 0.147 |
| length | 0.314 | [0.186, 0.441] | 49.0 | 0.176 | 0.176 |
| keyword | 0.294 | [0.169, 0.419] | 32.5 | 0.031 | 0.165 |
| sentiment | 0.294 | [0.169, 0.419] | 40.0 | 0.106 | 0.106 |
| ablate_semantic_rerank | 0.235 | [0.119, 0.352] | 40.6 | 0.170 | 0.212 |
| full_proxy | 0.216 | [0.103, 0.329] | 47.9 | 0.263 | 0.295 |
| ablate_debate | 0.216 | [0.103, 0.329] | 47.9 | 0.263 | 0.295 |
| ablate_quality_filter | 0.196 | [0.087, 0.305] | 39.2 | 0.196 | 0.196 |
| ablate_credibility | 0.137 | [0.043, 0.232] | 35.1 | 0.214 | 0.214 |

## Full System vs Comparators

| Comparator | Delta Accuracy (pp) | p-value | Cohen's d | Significant (0.05) |
|---|---:|---:|---:|:---:|
| random | -15.69 | 0.9793 | -0.259 | no |
| keyword | -7.84 | 0.9648 | -0.202 | no |
| length | -9.80 | 0.8852 | -0.141 | no |
| sentiment | -7.84 | 0.9102 | -0.151 | no |
| majority | -13.73 | 0.9461 | -0.200 | no |
| ablate_credibility | +7.84 | 0.0625 | 0.292 | no |
| ablate_semantic_rerank | -1.96 | 1.0000 | -0.141 | no |
| ablate_debate | +0.00 | 1.0000 | 0.000 | no |
| ablate_quality_filter | +1.96 | 0.5000 | 0.081 | no |

## Debate Lift

- Variant compared: ablate_debate
- Accuracy delta (full - no-debate): +0.00 pp
- Prediction change rate: 0.000