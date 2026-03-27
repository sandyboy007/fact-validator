# Explainability Demo Summary

- Generated UTC: 2026-03-27T20:22:13.874100
- Full variant: full_proxy
- Best baseline: length
- Case studies: 7

## Domain Credibility Examples

| Domain | Score |
|---|---:|
| bbc.com | 80 |
| reuters.com | 80 |
| who.int | 85 |
| wikipedia.org | 65 |
| example-blog-news.net | 50 |

## Case Studies

### Case 1: unverifiable-002

- Claim: An undisclosed alien signal was intercepted in 2025.
- Ground truth: NEI
- Full system: NEI (46.0)
- No-debate: SUPPORTED (46.0)
- Baseline (length): SUPPORTED (50.0)
- Scoring logic:
  - Category prior increases support confidence for evidence-rich domains.
  - Numeric signal detected: evidence consistency weighting is increased.
  - Conspiracy cue detected: refutation pressure strongly increases.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence.
  - Skeptic: Conspiracy-style language correlates with lower evidentiary reliability.
  - Judge: With debate arbitration, verdict is NEI (46.0 confidence), while no-debate predicts SUPPORTED. Best baseline predicts SUPPORTED.

### Case 2: conflict-001

- Claim: A ceasefire was reached in every major conflict in 2024.
- Ground truth: NEI
- Full system: NEI (34.5)
- No-debate: NEI (34.5)
- Baseline (length): SUPPORTED (50.0)
- Scoring logic:
  - Category prior increases uncertainty handling (NEI tendency) for ambiguous domains.
  - Numeric signal detected: evidence consistency weighting is increased.
  - Absolutist language detected: refutation pressure increases.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence.
  - Skeptic: Absolute wording raises risk of overclaiming and potential refutation.
  - Judge: With debate arbitration, verdict is NEI (34.5 confidence), while no-debate predicts NEI. Best baseline predicts SUPPORTED.

### Case 3: unverifiable-001

- Claim: A secret committee controls all global elections.
- Ground truth: REFUTED
- Full system: REFUTED (57.5)
- No-debate: REFUTED (57.5)
- Baseline (length): SUPPORTED (50.0)
- Scoring logic:
  - Category prior increases uncertainty handling (NEI tendency) for ambiguous domains.
  - Absolutist language detected: refutation pressure increases.
  - Conspiracy cue detected: refutation pressure strongly increases.
- Debate trace:
  - Prover: Category prior and lexical evidence still support a decisive verdict.
  - Skeptic: Absolute wording raises risk of overclaiming and potential refutation. Conspiracy-style language correlates with lower evidentiary reliability.
  - Judge: With debate arbitration, verdict is REFUTED (57.5 confidence), while no-debate predicts REFUTED. Best baseline predicts SUPPORTED.

### Case 4: climate-001

- Claim: Global average temperature has risen by about 1.1Â°C since the late 19th century.
- Ground truth: SUPPORTED
- Full system: NEI (22.5)
- No-debate: SUPPORTED (22.5)
- Baseline (length): SUPPORTED (50.0)
- Scoring logic:
  - Numeric signal detected: evidence consistency weighting is increased.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence. The phrasing implies an event-like fact pattern suited to evidence grounding.
  - Skeptic: Counter-signals are limited, but alternate verdicts remain plausible.
  - Judge: With debate arbitration, verdict is NEI (22.5 confidence), while no-debate predicts SUPPORTED. Best baseline predicts SUPPORTED.

### Case 5: numbers-001

- Claim: The Earth is 6,000 years old.
- Ground truth: REFUTED
- Full system: NEI (58.0)
- No-debate: NEI (58.0)
- Baseline (length): SUPPORTED (50.0)
- Scoring logic:
  - Category prior increases support confidence for evidence-rich domains.
  - Numeric signal detected: evidence consistency weighting is increased.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence.
  - Skeptic: Counter-signals are limited, but alternate verdicts remain plausible.
  - Judge: With debate arbitration, verdict is NEI (58.0 confidence), while no-debate predicts NEI. Best baseline predicts SUPPORTED.

### Case 6: health-002

- Claim: WHO declared COVID-19 a pandemic in March 2020.
- Ground truth: SUPPORTED
- Full system: SUPPORTED (44.5)
- No-debate: SUPPORTED (44.5)
- Baseline (length): SUPPORTED (50.0)
- Scoring logic:
  - Category prior increases support confidence for evidence-rich domains.
  - Numeric signal detected: evidence consistency weighting is increased.
  - Temporal event signal detected: support path gets additional weight.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence. The phrasing implies an event-like fact pattern suited to evidence grounding.
  - Skeptic: Counter-signals are limited, but alternate verdicts remain plausible.
  - Judge: With debate arbitration, verdict is SUPPORTED (44.5 confidence), while no-debate predicts SUPPORTED. Best baseline predicts SUPPORTED.

### Case 7: history-001

- Claim: Napoleon died on Saint Helena.
- Ground truth: SUPPORTED
- Full system: SUPPORTED (53.5)
- No-debate: SUPPORTED (53.5)
- Baseline (length): SUPPORTED (50.0)
- Scoring logic:
  - Category prior increases support confidence for evidence-rich domains.
- Debate trace:
  - Prover: The phrasing implies an event-like fact pattern suited to evidence grounding.
  - Skeptic: Counter-signals are limited, but alternate verdicts remain plausible.
  - Judge: With debate arbitration, verdict is SUPPORTED (53.5 confidence), while no-debate predicts SUPPORTED. Best baseline predicts SUPPORTED.
