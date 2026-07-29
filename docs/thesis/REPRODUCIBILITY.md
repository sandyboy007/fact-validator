# Fact Validator thesis reproducibility

This directory contains the final thesis source and compiled PDF for the
`thesis/reproducibility-corrections` branch. The intended post-merge release
tag is `thesis-v1.0.0`.

## Evaluation boundary

The repository contains two related but separate artifacts:

- **Fact Validator live application** implements open-web retrieval, live
  domain credibility scoring, SentenceTransformer reranking, persistence,
  caching, and optional Ollama debate.
- **FactValidator-Proxy** is the deterministic model evaluated on the frozen
  5,000-claim test set. It combines lexical classification, category priors,
  heuristic semantic signals, deterministic arbitration, and a quality
  filter. It does not execute the live components for every benchmark claim.

The 5,000-claim accuracy and macro-F1 values are proxy results. They must not
be attributed to end-to-end live application execution.

## Authoritative immutable inputs

- `data/benchmarks/splits_5000/train.json`
- `data/benchmarks/splits_5000/val.json`
- `data/benchmarks/splits_5000/test.json`
- `data/benchmarks/results_5000/ablation_study_predictions.csv`
- `data/benchmarks/results_5000/baseline_comparison_predictions.csv`

`data/benchmarks/results_5000/run_manifest.json` records SHA-256 hashes,
environment metadata, the random seed, and the proxy/live component boundary.

## Recreate and validate the statistical artifacts

From the repository root with Python 3.10.0:

```powershell
python -m pip install --require-hashes -r services/api/requirements.lock
python services/api/Scripts/run_thesis_statistics.py
python services/api/Scripts/validate_thesis_artifacts.py
pytest --collect-only -q
pytest services/api/tests -q
python services/api/Scripts/run_reproducibility_audit.py
```

The verified local release result is 163 tests collected and 163 passed. The
full output, OS, CPU, GPU, RAM, dependency-lock hash, and artifact hashes are
stored in `reproducibility_audit_report.json` and
`reproducibility_audit_summary.md`.

## Statistical outputs

- `statistics_report.json`
- `confusion_matrix_full_proxy.csv`
- `per_class_metrics.csv`
- `per_dataset_metrics.csv`
- `paired_tests.csv`
- `statistics_summary.md`

The statistical generator computes Wilson accuracy intervals, exact
two-sided McNemar tests, paired bootstrap intervals with seed 42, Holm
correction, paired risk differences, and matched-pair odds ratios.

The unadjusted full-proxy comparisons are:

- majority: p = 0.0433;
- length: p = 0.0462;
- no-debate proxy: p = 0.00955.

None of these three comparisons is significant after the committed Holm
correction, so they are not evidence of robust superiority.

## Confidence and operations

The proxy stores one heuristic score on a 0--100 scale. The code divides the
average score by 100 before comparing it with 0--1 accuracy. The result is
called a **raw-score calibration diagnostic**, not probability calibration.
A proper multiclass Brier score requires stored probabilities for SUPPORTED,
REFUTED, and NEI.

`operational_projection_report.json` contains scenarios, not controlled
load-test measurements. The 439 claims/hour, 50 claims/hour with debate,
USD 77 without caching, USD 22 with caching, and 71.43% savings values must
always be labelled projections.

## Build the thesis

Run twice from the repository root so cross-references and the table of
contents settle:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error `
  -output-directory=docs/thesis docs/thesis/Fact_Validator_Thesis_Final.tex
pdflatex -interaction=nonstopmode -halt-on-error `
  -output-directory=docs/thesis docs/thesis/Fact_Validator_Thesis_Final.tex
```

The source-listing paths are repository-relative, so compilation should be
started from the repository root.
