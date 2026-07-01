# Production Metrics Summary

- Generated UTC: 2026-07-01T00:37:05.078211
- Full system variant: full_proxy
- Claims in evaluation split: 48

## Latency & Throughput

| Metric | Value |
|---|---:|
| Baseline avg latency (sec) | 8.20 |
| Debate avg latency (sec) | 72.00 |
| Debate / Baseline latency ratio | 8.78x |
| Baseline throughput (claims/hour) | 439.02 |
| Debate throughput (claims/hour) | 50.00 |

## Cost

| Metric | Value |
|---|---:|
| Monthly claims (assumed) | 1000 |
| Monthly cost without cache (USD) | 77.00 |
| Monthly cost with cache (USD) | 22.00 |
| Monthly savings (USD) | 55.00 |
| Monthly savings (%) | 71.43% |

## Quality & Error

| Metric | Value |
|---|---:|
| Accuracy | 0.354 |
| Error rate | 0.646 |
| Expected errors / 100 claims | 64.58 |
| Macro F1 | 0.361 |
| Calibration error | 0.164 |
| ECE | 0.180 |

## Assumptions

- Baseline latency assumption: 8.2 sec/claim
- Debate latency assumption: 72.0 sec/claim
- Cost per search query: $0.0220
- Calls/claim without cache: 3.5
- Calls/claim with cache: 1.0