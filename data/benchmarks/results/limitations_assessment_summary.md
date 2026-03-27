# Limitations Assessment Summary

- Generated UTC: 2026-03-27T20:28:18.084521
- Registered limitations: 6
- High severity items: 1

## Limitations Register

| ID | Limitation | Severity | Impact |
|---|---|---|---|
| L1 | Residual classification errors | medium | Incorrect verdicts remain possible in real-world usage. |
| L2 | Category-specific generalization gaps | medium | Performance is uneven across claim domains. |
| L3 | Debate arbitration can introduce regressions | low | Debate mode may change verdicts without guaranteed net improvement. |
| L4 | Confidence calibration mismatch | medium | Displayed confidence may overstate or understate true correctness. |
| L5 | Source-selection and credibility-rubric bias | medium | System trust signals may favor mainstream indexed domains and under-represent local/novel sources. |
| L6 | Limited statistical power from small evaluation split | high | Point estimates and p-values are sensitive to a few cases. |

## Evidence & Mitigation

### L1 - Residual classification errors

- Severity: medium
- Evidence: Observed error rate is 0.286 (28.6 errors per 100 claims).
- Mitigation: Route low-confidence outputs to human review; expand evaluation set and tune calibration.

### L2 - Category-specific generalization gaps

- Severity: medium
- Evidence: Categories below threshold 0.60: climate=0.00 (n=1); science=0.50 (n=2)
- Mitigation: Increase domain-balanced benchmark size and add domain-specific retrieval prompts/models.

### L3 - Debate arbitration can introduce regressions

- Severity: low
- Evidence: Debate changed 2/7 cases (0.29); regressions observed in 1/7 (0.14).
- Mitigation: Trigger debate selectively (only uncertain baseline cases) and validate with guardrail thresholds.

### L4 - Confidence calibration mismatch

- Severity: medium
- Evidence: Calibration error=0.262, ECE=0.326 in current split.
- Mitigation: Apply post-hoc calibration (temperature scaling / isotonic regression) on larger validation data.

### L5 - Source-selection and credibility-rubric bias

- Severity: medium
- Evidence: Credibility scoring relies on domain rubric and search-retrieved evidence, not exhaustive ground truth.
- Mitigation: Introduce external expert calibration panel and diversify retrieval sources beyond a single search index.

### L6 - Limited statistical power from small evaluation split

- Severity: high
- Evidence: Current evaluated split contains 7 claims.
- Mitigation: Expand benchmark to 100+ claims per major domain and recompute all comparison statistics.
