# 5,000-Claim FactValidator-Proxy Execution and Results

## Scope

The frozen 5,000-claim experiment evaluates a deterministic
**FactValidator-Proxy**. It combines lexical classification, heuristic
baseline signals, category-level priors, deterministic arbitration rules, and
a quality filter.

It does not execute live SerpAPI retrieval, live domain-level credibility
scoring, SentenceTransformer reranking, or Ollama debate for every claim. The
Fact Validator live application is an implemented open-web architecture and
is evaluated separately through software tests and bounded operational
demonstrations.

## Data

- Train: 11,986 claims
- Validation: 2,997 claims
- Test: 5,000 claims
- Test labels: 2,463 SUPPORTED; 1,371 REFUTED; 1,166 NEI

The benchmark combines FEVER, LIAR, SciFact, and the PUBHEALTH
`health_fact` dataset stored under the legacy `healthver` compatibility name.

## Authoritative predictions

- `data/benchmarks/results_5000/ablation_study_predictions.csv`
- `data/benchmarks/results_5000/baseline_comparison_predictions.csv`

Their SHA-256 hashes and the input split hashes are recorded in
`data/benchmarks/results_5000/run_manifest.json`.

## Final verified results

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| Proxy without debate (`ablate_debate`) | 51.34% | 0.4427 |
| FEVER-tuned proxy (`tune_fever`) | 50.98% | 0.4421 |
| Full proxy (`full_proxy`) | 50.82% | 0.4384 |
| Length heuristic | 49.42% | 0.3483 |
| Majority baseline | 49.26% | 0.2200 |
| Random baseline | 33.20% | 0.3240 |
| Sentiment heuristic | 23.78% | 0.1400 |
| Keyword heuristic | 23.54% | 0.1364 |

The verified full-proxy confusion matrix is:

| Gold / Predicted | SUPPORTED | REFUTED | NEI |
|---|---:|---:|---:|
| SUPPORTED | 1,776 | 180 | 507 |
| REFUTED | 678 | 294 | 399 |
| NEI | 538 | 157 | 471 |

## Paired interpretation

The exact two-sided McNemar p-values before multiple-comparison correction are:

- full proxy vs majority: 0.0433;
- full proxy vs length: 0.0462;
- full proxy vs no-debate proxy: 0.00955.

The first two comparisons are marginal and are not described as robust
superiority after Holm correction. Removing deterministic proxy debate
improves accuracy from 50.82% to 51.34%, a difference of 0.52 percentage
points in favour of no debate.

## Reproduction

```bash
python services/api/Scripts/build_thesis_run_manifest.py
python services/api/Scripts/run_thesis_statistics.py
```

The generated statistical report, confusion matrix, class metrics, dataset
metrics, paired tests, and summary are stored in
`data/benchmarks/results_5000/`.
