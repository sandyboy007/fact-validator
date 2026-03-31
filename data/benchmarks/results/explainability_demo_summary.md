# Explainability Demo Summary

- Generated UTC: 2026-03-31T19:38:05.082093
- Full variant: full_proxy
- Best baseline: random
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

### Case 1: v2-003

- Claim: All climate records before 1950 were fabricated by one institution.
- Ground truth: NEI
- Full system: NEI (61.4)
- No-debate: NEI (61.4)
- Baseline (random): REFUTED (40.5)
- Scoring logic:
  - Numeric signal detected: evidence consistency weighting is increased.
  - Absolutist language detected: refutation pressure increases.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence.
  - Skeptic: Absolute wording raises risk of overclaiming and potential refutation.
  - Judge: With debate arbitration, verdict is NEI (61.4 confidence), while no-debate predicts NEI. Best baseline predicts REFUTED.

### Case 2: v2-014

- Claim: Lightning never strikes the same place twice.
- Ground truth: REFUTED
- Full system: REFUTED (57.0)
- No-debate: REFUTED (57.0)
- Baseline (random): NEI (43.5)
- Scoring logic:
  - Category prior increases support confidence for evidence-rich domains.
  - Absolutist language detected: refutation pressure increases.
- Debate trace:
  - Prover: Category prior and lexical evidence still support a decisive verdict.
  - Skeptic: Absolute wording raises risk of overclaiming and potential refutation.
  - Judge: With debate arbitration, verdict is REFUTED (57.0 confidence), while no-debate predicts REFUTED. Best baseline predicts NEI.

### Case 3: v2-042

- Claim: One hidden empire controlled every continent in 1200 CE.
- Ground truth: NEI
- Full system: NEI (61.4)
- No-debate: NEI (61.4)
- Baseline (random): REFUTED (45.8)
- Scoring logic:
  - Category prior increases support confidence for evidence-rich domains.
  - Numeric signal detected: evidence consistency weighting is increased.
  - Absolutist language detected: refutation pressure increases.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence.
  - Skeptic: Absolute wording raises risk of overclaiming and potential refutation.
  - Judge: With debate arbitration, verdict is NEI (61.4 confidence), while no-debate predicts NEI. Best baseline predicts REFUTED.

### Case 4: v2-052

- Claim: The UK held a general election in 2019.
- Ground truth: SUPPORTED
- Full system: SUPPORTED (69.5)
- No-debate: SUPPORTED (69.5)
- Baseline (random): NEI (42.6)
- Scoring logic:
  - Category prior increases uncertainty handling (NEI tendency) for ambiguous domains.
  - Numeric signal detected: evidence consistency weighting is increased.
  - Temporal event signal detected: support path gets additional weight.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence. The phrasing implies an event-like fact pattern suited to evidence grounding.
  - Skeptic: Counter-signals are limited, but alternate verdicts remain plausible.
  - Judge: With debate arbitration, verdict is SUPPORTED (69.5 confidence), while no-debate predicts SUPPORTED. Best baseline predicts NEI.

### Case 5: v2-063

- Claim: All climate records before 1950 were fabricated by one institution.
- Ground truth: NEI
- Full system: NEI (61.4)
- No-debate: NEI (61.4)
- Baseline (random): REFUTED (31.2)
- Scoring logic:
  - Numeric signal detected: evidence consistency weighting is increased.
  - Absolutist language detected: refutation pressure increases.
- Debate trace:
  - Prover: The claim contains quantifiable elements that can be checked against evidence.
  - Skeptic: Absolute wording raises risk of overclaiming and potential refutation.
  - Judge: With debate arbitration, verdict is NEI (61.4 confidence), while no-debate predicts NEI. Best baseline predicts REFUTED.

### Case 6: v2-117

- Claim: A hidden census proved world population is half the published figure.
- Ground truth: NEI
- Full system: NEI (61.4)
- No-debate: NEI (61.4)
- Baseline (random): SUPPORTED (55.1)
- Scoring logic:
  - Category prior increases support confidence for evidence-rich domains.
- Debate trace:
  - Prover: Category prior and lexical evidence still support a decisive verdict.
  - Skeptic: Counter-signals are limited, but alternate verdicts remain plausible.
  - Judge: With debate arbitration, verdict is NEI (61.4 confidence), while no-debate predicts NEI. Best baseline predicts SUPPORTED.

### Case 7: v2-126

- Claim: A newly found manuscript proves all major medieval timelines are incorrect.
- Ground truth: NEI
- Full system: NEI (53.8)
- No-debate: NEI (53.8)
- Baseline (random): REFUTED (36.2)
- Scoring logic:
  - Category prior increases support confidence for evidence-rich domains.
  - Absolutist language detected: refutation pressure increases.
- Debate trace:
  - Prover: Category prior and lexical evidence still support a decisive verdict.
  - Skeptic: Absolute wording raises risk of overclaiming and potential refutation.
  - Judge: With debate arbitration, verdict is NEI (53.8 confidence), while no-debate predicts NEI. Best baseline predicts REFUTED.
