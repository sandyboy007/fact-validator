# Limitations Assessment Summary

- Generated UTC: 2026-03-31T19:38:05.278637
- Registered limitations: 6
- High severity items: 2

## Limitations Register

| ID | Limitation | Severity | Impact |
|---|---|---|---|
| L1 | Residual classification errors | high | Incorrect verdicts remain possible in real-world usage. |
| L2 | Category-specific generalization gaps | high | Performance is uneven across claim domains. |
| L3 | Debate arbitration can introduce regressions | low | Debate mode may change verdicts without guaranteed net improvement. |
| L4 | Confidence calibration mismatch | medium | Displayed confidence may overstate or understate true correctness. |
| L5 | Source-selection and credibility-rubric bias | medium | System trust signals may favor mainstream indexed domains and under-represent local/novel sources. |
| L6 | Limited statistical power from small evaluation split | medium | Point estimates and p-values are sensitive to a few cases. |

## Evidence & Mitigation

### L1 - Residual classification errors

- Severity: high
- Evidence: Observed error rate is 0.625 (62.5 errors per 100 claims).
- Mitigation: Route low-confidence outputs to human review; expand evaluation set and tune calibration.

### L2 - Category-specific generalization gaps

- Severity: high
- Evidence: Categories below threshold 0.60: general=0.00 (n=2); technology=0.00 (n=6); health=0.00 (n=3); conflict=0.00 (n=3); science=0.14 (n=7); demographics=0.25 (n=4); politics=0.25 (n=4); work=0.50 (n=4)
- Mitigation: Increase domain-balanced benchmark size and add domain-specific retrieval prompts/models.

### L3 - Debate arbitration can introduce regressions

- Severity: low
- Evidence: Debate changed 0/7 cases (0.00); regressions observed in 0/7 (0.00).
- Mitigation: Trigger debate selectively (only uncertain baseline cases) and validate with guardrail thresholds.

### L4 - Confidence calibration mismatch

- Severity: medium
- Evidence: Calibration error=0.174, ECE=0.223 in current split.
- Mitigation: Apply post-hoc calibration (temperature scaling / isotonic regression) on larger validation data.

### L5 - Source-selection and credibility-rubric bias

- Severity: medium
- Evidence: Credibility scoring relies on domain rubric and search-retrieved evidence, not exhaustive ground truth.
- Mitigation: Introduce external expert calibration panel and diversify retrieval sources beyond a single search index.

### L6 - Limited statistical power from small evaluation split

- Severity: medium
- Evidence: Current evaluated split contains 48 claims.
- Mitigation: Expand benchmark to 100+ claims per major domain and recompute all comparison statistics.
