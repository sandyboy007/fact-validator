# Reflective Threshold Tuning Summary

- Generated UTC: 2026-04-02T20:24:46.474993Z
- Mode: live-sampled
- Claims evaluated: 30
- Grid size: 64
- Requested sample size: 30
- Selected sample size: 30
- Unique live calls: 30
- Live call failures: 0
- SERPAPI enabled: True

## Recommended Thresholds

- hallucination_quality_min: 45.0
- hallucination_directness_min: 0.15
- strong_quality_min: 56.0
- conflict_quality_gap_max: 6.0
- low_factor_coverage_pct: 24.0

## Best Candidate Metrics

- Objective: 0.1388
- Accuracy: 0.3000
- Macro F1: 0.1538
- Abstention rate: 1.0000
- Abstention precision: 0.3000
- Abstention recall (NEI): 1.0000
- False abstention (non-NEI): 1.0000

## Top Candidates

| Rank | strong_quality_min | low_factor_coverage_pct | conflict_quality_gap_max | Accuracy | Macro F1 | Abstain Precision | Abstain Recall | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56.0 | 24.0 | 6.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 2 | 56.0 | 24.0 | 8.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 3 | 56.0 | 24.0 | 10.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 4 | 56.0 | 24.0 | 12.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 5 | 56.0 | 30.0 | 6.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 6 | 56.0 | 30.0 | 8.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 7 | 56.0 | 30.0 | 10.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 8 | 56.0 | 30.0 | 12.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 9 | 56.0 | 36.0 | 6.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 10 | 56.0 | 36.0 | 8.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 11 | 56.0 | 36.0 | 10.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |
| 12 | 56.0 | 36.0 | 12.0 | 0.300 | 0.154 | 0.300 | 1.000 | 0.139 |