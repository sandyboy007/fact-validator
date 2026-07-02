# External Dataset Import Templates (5000 Benchmark)

This folder contains templates to normalize external benchmark datasets into a format accepted by:

- `services/api/Scripts/build_large_test_benchmark.py`
- `services/api/Scripts/run_5000_benchmark_pipeline.py`

## Accepted input formats

You can pass each dataset as either:

1. CSV file with at least these fields:
- `id`
- `claim`
- `label`

Optional but recommended fields:
- `category`
- `difficulty`
- `source_url`
- `notes`

2. JSON file with top-level `claims` array. Each claim object should include:
- `id`
- `claim` (or `text`)
- `label` (or `verdict`)

Optional fields:
- `category` (or `topic`)
- `difficulty`
- `source_url` (or `url`)
- `provenance_note` (or `notes`)

## Label normalization

Allowed canonical labels are:
- `SUPPORTED`
- `REFUTED`
- `NEI`

The pipeline also normalizes common aliases:
- `true` -> `SUPPORTED`
- `false` -> `REFUTED`
- `not enough information` -> `NEI`
- `insufficient evidence` -> `NEI`
- `mixed`, `disputed` -> `NEI`

## Recommended workflow

1. Copy one of the template CSV files and populate claims.
2. Ensure all labels are valid and claims are non-empty.
3. Run the 5000 pipeline:

```powershell
cd C:/Fact_Validator/services/api
python Scripts/run_5000_benchmark_pipeline.py `
  --input fever=C:/Fact_Validator/data/benchmarks/external_templates/fever_filled.csv `
  --input liar=C:/Fact_Validator/data/benchmarks/external_templates/liar_filled.csv `
  --input scifact=C:/Fact_Validator/data/benchmarks/external_templates/scifact_filled.csv `
  --input healthver=C:/Fact_Validator/data/benchmarks/external_templates/healthver_filled.csv `
  --target-test-size 5000 `
  --benchmark-output C:/Fact_Validator/data/benchmarks/results/large_benchmark_manifest.json `
  --splits-dir C:/Fact_Validator/data/benchmarks/splits_5000
```

4. After splits are created, run architecture comparison:

```powershell
cd C:/Fact_Validator/services/api
python Scripts/run_benchmark_architecture_suite.py `
  --train C:/Fact_Validator/data/benchmarks/splits_5000/train.json `
  --test C:/Fact_Validator/data/benchmarks/splits_5000/test.json `
  --output-dir C:/Fact_Validator/data/benchmarks/results_5000
```

## Notes for publishability

- Prefer genuine external claims over synthetic claims.
- Keep provenance for each claim (`source_url` and/or `notes`).
- Avoid duplicates across datasets before final run.
- Freeze the final test split once generated.
