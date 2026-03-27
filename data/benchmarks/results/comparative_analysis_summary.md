# Comparative Analysis Summary

- Generated UTC: 2026-03-27T20:02:29.757013
- Full system variant: full_proxy
- Claims compared: 7

## System Ranking

| System | Accuracy | 95% CI | Avg Confidence | Calibration Error | ECE |
|---|---:|---:|---:|---:|---:|
| full_proxy | 0.714 | [0.380, 1.000] | 45.2 | 0.262 | 0.326 |
| ablate_debate | 0.714 | [0.380, 1.000] | 45.2 | 0.262 | 0.262 |
| ablate_quality_filter | 0.714 | [0.380, 1.000] | 41.8 | 0.296 | 0.296 |
| ablate_credibility | 0.571 | [0.205, 0.938] | 29.6 | 0.276 | 0.317 |
| length | 0.429 | [0.062, 0.795] | 50.0 | 0.071 | 0.071 |
| majority | 0.429 | [0.062, 0.795] | 50.0 | 0.071 | 0.071 |
| random | 0.286 | [0.000, 0.620] | 40.0 | 0.114 | 0.114 |
| keyword | 0.286 | [0.000, 0.620] | 20.0 | 0.086 | 0.086 |
| sentiment | 0.286 | [0.000, 0.620] | 40.0 | 0.114 | 0.114 |
| ablate_semantic_rerank | 0.286 | [0.000, 0.620] | 35.2 | 0.066 | 0.199 |

## Full System vs Comparators

| Comparator | Delta Accuracy (pp) | p-value | Cohen's d | Significant (0.05) |
|---|---:|---:|---:|:---:|
| random | +42.86 | 0.1250 | 0.866 | no |
| keyword | +42.86 | 0.1250 | 0.866 | no |
| length | +28.57 | 0.3125 | 0.408 | no |
| sentiment | +42.86 | 0.1250 | 0.866 | no |
| majority | +28.57 | 0.3125 | 0.408 | no |
| ablate_credibility | +14.29 | 0.5000 | 0.408 | no |
| ablate_semantic_rerank | +42.86 | 0.1250 | 0.866 | no |
| ablate_debate | +0.00 | 0.7500 | 0.000 | no |
| ablate_quality_filter | +0.00 | 1.0000 | 0.000 | no |

## Debate Lift

- Variant compared: ablate_debate
- Accuracy delta (full - no-debate): +0.00 pp
- Prediction change rate: 0.286