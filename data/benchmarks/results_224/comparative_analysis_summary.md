# Comparative Analysis Summary

- Generated UTC: 2026-07-02T00:34:32.918031
- Full system variant: full_proxy
- Claims compared: 48

## System Ranking

| System | Accuracy | 95% CI | Avg Confidence | Calibration Error | ECE |
|---|---:|---:|---:|---:|---:|
| majority | 0.417 | [0.277, 0.556] | 50.0 | 0.083 | 0.083 |
| ablate_semantic_rerank | 0.396 | [0.257, 0.534] | 45.4 | 0.058 | 0.178 |
| ablate_quality_filter | 0.396 | [0.257, 0.534] | 43.6 | 0.040 | 0.184 |
| random | 0.375 | [0.238, 0.512] | 47.0 | 0.095 | 0.168 |
| length | 0.354 | [0.219, 0.489] | 49.3 | 0.139 | 0.139 |
| full_proxy | 0.354 | [0.219, 0.489] | 51.8 | 0.164 | 0.180 |
| ablate_debate | 0.354 | [0.219, 0.489] | 51.8 | 0.164 | 0.180 |
| keyword | 0.292 | [0.163, 0.420] | 35.0 | 0.058 | 0.233 |
| sentiment | 0.292 | [0.163, 0.420] | 40.0 | 0.108 | 0.108 |
| ablate_credibility | 0.229 | [0.110, 0.348] | 37.8 | 0.149 | 0.169 |

## Full System vs Comparators

| Comparator | Delta Accuracy (pp) | p-value | Cohen's d | Significant (0.05) |
|---|---:|---:|---:|:---:|
| random | -2.08 | 0.6612 | -0.030 | no |
| keyword | +6.25 | 0.2744 | 0.132 | no |
| length | +0.00 | 0.5806 | 0.000 | no |
| sentiment | +6.25 | 0.3036 | 0.113 | no |
| majority | -6.25 | 0.7709 | -0.081 | no |
| ablate_credibility | +12.50 | 0.0156 | 0.378 | yes |
| ablate_semantic_rerank | -4.17 | 0.8906 | -0.119 | no |
| ablate_debate | +0.00 | 1.0000 | 0.000 | no |
| ablate_quality_filter | -4.17 | 0.7880 | -0.077 | no |

## Debate Lift

- Variant compared: ablate_debate
- Accuracy delta (full - no-debate): +0.00 pp
- Prediction change rate: 0.000