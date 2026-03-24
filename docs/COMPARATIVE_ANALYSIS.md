# Comparative Analysis Framework - Human Judgment vs. Existing Systems

## Overview

This document outlines the framework for comparing Fact Validator against:
1. **Human annotators** (ground truth)
2. **Existing fact-checking systems** (competitive analysis)
3. **Reference baselines** (relative performance)

---

## 1. Human Evaluation Framework

### 1.1 Annotator Selection

**Recommended Annotators:**
- 3-5 human judges per claim
- Mix of backgrounds:
  - Domain experts (health, science, etc.) - 30%
  - Journalists/researchers - 40%
  - General educated population - 30%
- Screening: Annotators must pass comprehension test

**Compensation:**
- $0.10-0.20 per claim (via MTurk, Prolific, etc.)
- Quality bonus: +20% for >98% accuracy on attention checks

### 1.2 Annotation Protocol

**Setup:**
```
For each claim:
1. Present claim in neutral context
2. Provide optional search results (like real search)
3. Ask for verdict: SUPPORTED / REFUTED / NEI
4. Ask for confidence: 0-100%
5. Ask for reasoning (optional)
6. Record time spent

Time per claim: 60-120 seconds
Total annotation: 2-3 hours for 20 claims
```

### 1.3 Interrater Agreement Analysis

**Metrics:**
- **Cohen's Kappa** (2 judges)
- **Fleiss' Kappa** (3+ judges)
- **Percent Agreement (simple)**
- **Confidence Correlation**

**Interpretation:**
```
Kappa    | Agreement Quality
---------|------------------
< 0.20   | Poor
0.20-0.40| Fair
0.40-0.60| Moderate
0.60-0.80| Substantial
> 0.80   | Excellent
```

**Expected Results:**
- Well-designed benchmark: κ ≥ 0.60 (substantial agreement)
- Poor benchmark: κ < 0.40 (too ambiguous)

---

## 2. Existing System Comparisons

### 2.1 Google Fact Check API

**Status:** Available but limited

**Integration:**
```python
import requests

response = requests.get(
    "https://factchecktools.googleapis.com/v1alpha1/claims:search",
    params={"query": claim_text, "key": api_key}
)

# Extract verdicts from response
for claim in response.json()["claims"]:
    verdict = claim["claimReview"][0]["textualRating"]  # TRUE/FALSE/MIXED
```

**Comparison Metrics:**
- Accuracy against ground truth
- Coverage (% of claims with results)
- Latency (API response time)

**Known Limitations:**
- Only searches Google's fact-check database
- May return 0 results for novel claims
- Predefined verdicts (not continuous scores)

### 2.2 ClaimBuster (CMU)

**Status:** Academic system, API available

**Integration:**
```python
import requests

response = requests.post(
    "https://claimbuster.org/api/v2/score_claim",
    json={"claim": claim_text}
)

claim_score = response.json()["scores"][0]  # 0-1 check-worthiness
```

**Comparison Metrics:**
- Claim extraction performance
- Check-worthiness scoring accuracy

**Known Limitations:**
- Scores check-worthiness, not verdict
- Not a complete fact-checking system
- Limited to English

### 2.3 FEVER Baseline

**Status:** Research dataset, no API

**Comparison Method:**
- Use FEVER-trained models (publicly available)
- Run locally on test claims
- Compare verdict accuracy

**Reference Implementation:**
- BERT-based evidence retrieval + NLI classification
- Publicly available weights: github.com/facebookresearch/FEVER

---

## 3. Comparative Study Design

### 3.1 Experimental Setup

```
┌─────────────────────────────────┐
│  20-Claim Test Set              │
└─────────────────────────────────┘
          │
    ┌─────┼─────┬─────┬─────┐
    │     │     │     │     │
    ▼     ▼     ▼     ▼     ▼
 FactVal Human Google ClaimBuster Baseline
 System  Judges API    System      Random
    │     │     │     │     │
    └─────┴─────┴─────┴─────┘
          │
    Accuracy, Precision, Recall, F1
    Confidence Calibration
    Statistical Significance
```

### 3.2 Metrics Calculated

| Metric | Calculation | Interpretation |
|--------|-------------|-----------------|
| **Accuracy** | (TP + TN) / N | % correct verdicts |
| **Precision** | TP / (TP + FP) | % predicted SUPPORTED actually supported |
| **Recall** | TP / (TP + FN) | % actual SUPPORTED predicted correctly |
| **F1** | 2(P×R)/(P+R) | Harmonic mean |
| **Human Agree** | % agreement with human consensus | Alignment with truth |
| **Latency** | Time to generate verdict | Real-world feasibility |

### 3.3 Expected Comparative Performance

**Hypothetical Scenario:**
```
System          Accuracy  Human Agree  Latency (sec)
────────────────────────────────────────────────────
Fact Validator  0.80      0.75         15-30
RandomBaseline  0.33      0.30         <1
KeywordBaseline 0.45      0.40         <5
Google API      0.70      0.65         5-20
ClaimBuster     0.55*     0.50         <5
Human Judges    0.85      1.00         60-120

* ClaimBuster scores check-worthiness, not verdict
```

---

## 4. Statistical Comparison Procedure

### 4.1 Paired T-Test: System vs Human

**Hypothesis:**
- H₀: System accuracy ≤ Human accuracy
- H₁: System accuracy > Human accuracy
- α = 0.05 (two-tailed)

**Calculation:**
```python
from app.statistics import StatisticalAnalyzer

system_scores = [0.9, 0.8, 0.85, ...]  # Per-claim accuracy
human_scores = [1.0, 0.8, 1.0, ...]    # Human consensus

result = analyzer.paired_t_test(system_scores, human_scores)
print(f"p-value: {result.p_value}")
print(f"Cohen's d: {result.effect_size}")
```

**Interpretation:**
- p < 0.05: System significantly different from humans
- p ≥ 0.05: No significant difference

### 4.2 Confidence Intervals

**Report 95% CI for each system:**
```
System          Accuracy (95% CI)
────────────────────────────────
Fact Validator  0.80 [0.65, 0.95]
Human Judges    0.85 [0.70, 1.00]
Google API      0.70 [0.52, 0.88]
```

Overlapping intervals suggest no significant difference.

### 4.3 Effect Size (Cohen's d)

**Magnitude Interpretation:**
```
d < 0.2:  Negligible - systems roughly equivalent
0.2-0.5:  Small difference but may not matter practically
0.5-0.8:  Medium difference - noticeable
> 0.8:    Large difference - substantial advantage
```

---

## 5. Human-System Alignment Analysis

### 5.1 Disagreement Patterns

**Categorize disagreements:**
```
System SUPPORTED, Human REFUTED:
  - Type 1: Hallucination (system found non-credible evidence)
  - Type 2: Over-confidence (correct but too confident)

System REFUTED, Human SUPPORTED:
  - Type 3: Evidence miss (failed to find supporting evidence)
  - Type 4: Evidence ranking (supporting evidence ranked too low)
```

### 5.2 Confidence Calibration

**Plot system confidence vs human consensus:**
```
100% ┤    *
     │  * * *
80%  │ * * * *
     │ * *   *
60%  ├────────┤
     │ *   * 
40%  │   *
     │
20%  │
     └──────────
       0%  50% 100% (Human % Agreement)
```

**Interpretation:**
- Points on diagonal: well-calibrated
- Points above diagonal: overconfident
- Points below diagonal: underconfident

---

## 6. Comparative Report Template

```markdown
# Comparative Evaluation: Fact Validator vs. References

## Executive Summary

**Test Set:** 20 diverse claims (health, politics, science, etc.)
**Annotators:** 3 human judges (κ = 0.72, substantial agreement)
**Comparison Systems:** Google Fact Check API, ClaimBuster, Random Baseline

## Overall Results

| Metric | FactVal | Humans | Google | ClaimBuster | Random |
|--------|---------|--------|--------|-------------|--------|
| Accuracy | **80%** | 85% | 70% | 55% | 33% |
| Precision | **82%** | 87% | 72% | 60% | 50% |
| Recall | **78%** | 83% | 68% | 50% | 20% |
| F1-Score | **80%** | 85% | 70% | 55% | 29% |

## Per-Category Performance

### Health Claims (4 claims)
- Fact Validator: 100% ✓
- Human Judges: 100%
- Google API: 75%

### Political Claims (4 claims)
- Fact Validator: 75%
- Human Judges: 75%
- Google API: 50%

### Science Claims (4 claims)
- Fact Validator: 75%
- Human Judges: 100%
- Google API: 75%

## Statistical Significance

**Fact Validator vs Human Judges**
```
Cohen's d: 0.35 (small effect)
t-statistic: 1.23, p-value: 0.24 (not significant)
95% CI: [-0.05, 0.15] (overlapping)

Interpretation: Fact Validator performance not significantly 
different from humans (p > 0.05). Result suggests good alignment.
```

**Fact Validator vs Google Fact Check API**
```
Cohen's d: 0.52 (medium effect)
t-statistic: 2.45, p-value: 0.03 (significant)
95% CI: [0.02, 0.18] (non-overlapping)

Interpretation: Fact Validator significantly outperforms
Google API (p < 0.05) with medium effect size.
```

## Error Analysis

### Fact Validator Failures (4 errors)
1. **Claim:** "Vaccines contain mercury"
   - System Verdict: SUPPORTED (confidence: 65%)
   - Correct Verdict: REFUTED
   - Error Type: Misinterpreted thimerosal content evidence

2. **Claim:** "Remote work increases productivity"
   - System Verdict: SUPPORTED (confidence: 72%)
   - Correct Verdict: NEI (mixed evidence)
   - Error Type: Treated mixed evidence as supporting

3. **Claim:** "AI will cause 30% unemployment by 2030"
   - System Verdict: NEI (confidence: 45%)
   - Correct Verdict: NEI
   - Error Type: Correct but very low confidence

4. **Claim:** "The Earth is 6,000 years old"
   - System Verdict: REFUTED (confidence: 95%)
   - Correct Verdict: REFUTED
   - Error Type: Correct verdict, but excessive confidence

### Human Failures (3 errors)
- Humans agreed on wrong verdict for 3 claims
- Suggests benchmark has ambiguous cases

## Recommendations

1. **Strengths to Maintain**
   - Excellent performance on health claims (100%)
   - Good evidence retrieval (78% recall)
   - Appropriate confidence calibration for most cases

2. **Improvements Needed**
   - Better handling of mixed/disputed claims (NEI classification)
   - Improved confidence calibration (slightly overconfident at 70-80%)
   - Edge case handling (rare but important claims)

3. **Deployment Considerations**
   - Suitable for initial screening (80% accuracy)
   - Requires human review for high-stakes claims
   - Particularly reliable for health/science claims
   - Needs improvement on political/controversial claims

## Conclusion

Fact Validator demonstrates competitive performance with 80% accuracy,
significantly outperforming existing systems (Google API: 70%, ClaimBuster: 55%)
while achieving near-human agreement (human: 85%, FactVal: 80%).
Performance is not significantly different from human judges (p=0.24),
suggesting the system reliably replicates human-level fact-checking on
this benchmark.
```

---

## 7. Reproducibility of Comparisons

All comparative analyses are reproducible via:

```bash
# Run comparative evaluation
cd services/api
python -m pytest tests/test_comparative.py -v

# Generate comparative report
python scripts/generate_comparison_report.py \
  --systems fact_validator google_api claimbuster \
  --test_set docs/evaluation_benchmark.json \
  --output reports/comparison_2026-03-24.md
```

---

## 8. References

- [Google Fact Check API Docs](https://toolbox.google.com/factcheck/api/documentation/v1alpha1)
- [ClaimBuster](https://claimbuster.org)
- [FEVER Dataset](https://fever.ai)
- [Cohen's Kappa Calculator](https://statpages.info/kappa.html)
