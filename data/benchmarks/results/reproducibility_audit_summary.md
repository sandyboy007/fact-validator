# Reproducibility Audit Summary

- Generated UTC: 2026-03-27T20:34:00.398235
- Git commit: 20b5344
- Python: 3.10.0
- Platform: Windows-10-10.0.26200-SP0

## Score

- Reproducibility score: 100.0%
- Passed checks: 29 / 29

## Checklist

| Section | Check | Status | Details |
|---|---|---|---|
| docs | D1 - Required documentation present: README.md | PASS | OK (19065 bytes): C:\Fact_Validator\README.md |
| docs | D2 - Required documentation present: DEPLOYMENT.md | PASS | OK (5852 bytes): C:\Fact_Validator\DEPLOYMENT.md |
| docs | D3 - Required documentation present: METHODS.md | PASS | OK (9249 bytes): C:\Fact_Validator\docs\METHODS.md |
| docs | D4 - Required documentation present: LIMITATIONS.md | PASS | OK (15326 bytes): C:\Fact_Validator\docs\LIMITATIONS.md |
| docs | D5 - Required documentation present: COMPARATIVE_ANALYSIS.md | PASS | OK (11930 bytes): C:\Fact_Validator\docs\COMPARATIVE_ANALYSIS.md |
| docs | D6 - Required documentation present: THESIS_COMPARATIVE_EVALUATION.md | PASS | OK (33612 bytes): C:\Fact_Validator\docs\THESIS_COMPARATIVE_EVALUATION.md |
| scripts | S1 - Pipeline script present: prepare_research_benchmark.py | PASS | OK (3304 bytes): C:\Fact_Validator\services\api\Scripts\prepare_research_benchmark.py |
| scripts | S2 - Pipeline script present: run_baseline_comparison.py | PASS | OK (8076 bytes): C:\Fact_Validator\services\api\Scripts\run_baseline_comparison.py |
| scripts | S3 - Pipeline script present: run_ablation_study.py | PASS | OK (19293 bytes): C:\Fact_Validator\services\api\Scripts\run_ablation_study.py |
| scripts | S4 - Pipeline script present: run_comparative_analysis.py | PASS | OK (16407 bytes): C:\Fact_Validator\services\api\Scripts\run_comparative_analysis.py |
| scripts | S5 - Pipeline script present: run_production_metrics.py | PASS | OK (10817 bytes): C:\Fact_Validator\services\api\Scripts\run_production_metrics.py |
| scripts | S6 - Pipeline script present: run_explainability_demo.py | PASS | OK (17422 bytes): C:\Fact_Validator\services\api\Scripts\run_explainability_demo.py |
| scripts | S7 - Pipeline script present: run_limitations_assessment.py | PASS | OK (10387 bytes): C:\Fact_Validator\services\api\Scripts\run_limitations_assessment.py |
| scripts | S8 - Pipeline script present: run_reproducibility_audit.py | PASS | OK (9857 bytes): C:\Fact_Validator\services\api\Scripts\run_reproducibility_audit.py |
| artifacts | A1 - Generated artifact present: research_benchmark_v1.json | PASS | OK (4092 bytes): C:\Fact_Validator\data\benchmarks\research_benchmark_v1.json |
| artifacts | A2 - Generated artifact present: baseline_comparison_report.json | PASS | OK (8341 bytes): C:\Fact_Validator\data\benchmarks\results\baseline_comparison_report.json |
| artifacts | A3 - Generated artifact present: ablation_study_report.json | PASS | OK (10157 bytes): C:\Fact_Validator\data\benchmarks\results\ablation_study_report.json |
| artifacts | A4 - Generated artifact present: comparative_analysis_report.json | PASS | OK (16225 bytes): C:\Fact_Validator\data\benchmarks\results\comparative_analysis_report.json |
| artifacts | A5 - Generated artifact present: production_metrics_report.json | PASS | OK (1352 bytes): C:\Fact_Validator\data\benchmarks\results\production_metrics_report.json |
| artifacts | A6 - Generated artifact present: explainability_demo_report.json | PASS | OK (11007 bytes): C:\Fact_Validator\data\benchmarks\results\explainability_demo_report.json |
| artifacts | A7 - Generated artifact present: limitations_assessment_report.json | PASS | OK (3000 bytes): C:\Fact_Validator\data\benchmarks\results\limitations_assessment_report.json |
| runtime | R1 - Endpoint available: /health | PASS | HTTP 200: http://127.0.0.1:8000/health |
| runtime | R2 - Endpoint available: /evaluation/benchmark | PASS | HTTP 200: http://127.0.0.1:8000/evaluation/benchmark |
| runtime | R3 - Endpoint available: /evaluation/baselines | PASS | HTTP 200: http://127.0.0.1:8000/evaluation/baselines |
| runtime | R4 - Endpoint available: /evaluation/ablations | PASS | HTTP 200: http://127.0.0.1:8000/evaluation/ablations |
| runtime | R5 - Endpoint available: /evaluation/comparative | PASS | HTTP 200: http://127.0.0.1:8000/evaluation/comparative |
| runtime | R6 - Endpoint available: /evaluation/production-metrics | PASS | HTTP 200: http://127.0.0.1:8000/evaluation/production-metrics |
| runtime | R7 - Endpoint available: /evaluation/explainability | PASS | HTTP 200: http://127.0.0.1:8000/evaluation/explainability |
| runtime | R8 - Endpoint available: /evaluation/limitations | PASS | HTTP 200: http://127.0.0.1:8000/evaluation/limitations |

## Notes

- This audit checks artifact completeness and runtime availability, not semantic correctness of every model decision.
- For stronger reproducibility claims, run the full test suite and record dependency lockfiles in CI.