# Statistical Significance Execution Plan

Date: 2026-04-01

## Why significance is not yet achieved

- Current benchmark has only 20 claims total.
- Re-splitting can increase test size, but this reduces train size and can destabilize model quality.
- Recent rerun with a larger test split produced 18 test claims but only 2 train claims, which is not a reliable training/evaluation setup.

Conclusion: real improvement requires expanding the benchmark with new labeled claims, not only changing split ratios.

## Target sample sizes

From planning script (`Scripts/estimate_sample_size.py`):

- Detecting a large gap (0.714 vs 0.429):
  - Required n per group: ~47
  - Approx total for two independent groups: ~94

- Detecting a moderate gap (0.714 vs 0.571):
  - Required n per group: ~176
  - Approx total for two independent groups: ~352

Recommended thesis target:
- Minimum publishable: 120-150 test claims
- Preferred: 180-250 test claims
- Strong significance and tighter CIs: 300+ test claims

## Data collection requirements

Use the provided template file:
- `data/benchmarks/claim_annotation_template_240.csv`

Required columns:
- id
- claim
- label (`SUPPORTED`, `REFUTED`, `NEI`)
- category
- difficulty (`easy`, `medium`, `hard`)
- source_url
- annotator_1
- annotator_2
- annotator_3
- notes

Quality controls:
- At least 3 annotators/claim
- Resolve disagreements by majority vote
- Track inter-annotator agreement (target kappa >= 0.60)
- Keep label balance close to 1/3 each where possible

## Execution workflow after annotations are complete

From `services/api`:

1. Build benchmark and splits

```
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/prepare_research_benchmark.py --input docs/evaluation_benchmark_v2.json --output data/benchmarks/research_benchmark_v2.json --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --seed 42
```

2. Run evaluation pipeline

```
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/run_baseline_comparison.py
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/run_ablation_study.py
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/run_comparative_analysis.py
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/run_production_metrics.py
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/run_explainability_demo.py
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/run_limitations_assessment.py
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/run_ethics_assessment.py
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/run_defense_talking_points.py
```

3. Re-check sample adequacy

```
c:/Fact_Validator/.venv/Scripts/python.exe Scripts/estimate_sample_size.py --p1 <new_full_accuracy> --p2 <new_baseline_accuracy>
```

## Thesis reporting language once expanded dataset is ready

Use this structure:
- Primary outcome: accuracy and macro F1 on the expanded test set
- Secondary outcomes: calibration error, ECE, category-wise accuracy
- Statistical tests: paired sign test p-values + effect sizes
- Confidence intervals: 95% CI for all main systems

Then state significance only when p < 0.05 with adequate sample size and balanced split.
