# Thesis Draft: Results and Architecture Comparison

## 4. Results and Comparative Evaluation

This section reports the current architecture comparison results for the Fact Validator system on the latest available held-out split, and interprets them conservatively for academic publication.

### 4.1 Evaluation Setup

We evaluated multiple architectures on a held-out test split generated from the 224-claim benchmark pipeline (`n=48` test claims). The comparison includes:

- Full system proxy (`full_proxy`)
- Simple baseline architectures (`majority`, `random`, `length`, `keyword`, `sentiment`)
- Internal ablation variants (`ablate_credibility`, `ablate_semantic_rerank`, `ablate_debate`, `ablate_quality_filter`)

All metrics were produced by the reproducible script chain:

1. `run_baseline_comparison.py`
2. `run_ablation_study.py`
3. `run_comparative_analysis.py`

and can be executed via the unified runner:

- `services/api/Scripts/run_benchmark_architecture_suite.py`

### 4.2 Accuracy Ranking

From `data/benchmarks/results_224/comparative_analysis_summary.md` (generated UTC `2026-07-02T00:31:44.154584`), the ranking by accuracy is:

| Architecture | Accuracy | 95% CI |
|---|---:|---:|
| majority | 0.417 | [0.277, 0.556] |
| ablate_semantic_rerank | 0.396 | [0.257, 0.534] |
| ablate_quality_filter | 0.396 | [0.257, 0.534] |
| random | 0.375 | [0.238, 0.512] |
| length | 0.354 | [0.219, 0.489] |
| full_proxy (proposed full system) | 0.354 | [0.219, 0.489] |
| ablate_debate | 0.354 | [0.219, 0.489] |
| keyword | 0.292 | [0.163, 0.420] |
| sentiment | 0.292 | [0.163, 0.420] |
| ablate_credibility | 0.229 | [0.110, 0.348] |

### 4.3 Full System vs Comparator Architectures

Relative to the full system (`full_proxy`):

- vs `majority`: -6.25 percentage points
- vs `random`: -2.08 percentage points
- vs `length`: +0.00 percentage points
- vs `keyword`: +6.25 percentage points
- vs `sentiment`: +6.25 percentage points

On this split, paired significance tests show no statistically significant superiority of the full system over the strongest simple comparators at $\alpha=0.05$.

### 4.4 Ablation Signal

The most consistent positive signal is the comparison:

- `full_proxy` vs `ablate_credibility`: +12.50 percentage points, significant in this run.

This suggests the credibility component contributes meaningful predictive signal, even though overall architecture-level superiority is not yet established on the current small held-out set.

### 4.5 Interpretation for Publication

The current results should be reported as **provisional comparative evidence** rather than definitive superiority claims. Confidence intervals remain broad due to low test-set size (`n=48`), and ranking instability is expected in this regime.

Accordingly, the strongest publishable framing is:

1. The system provides a reproducible, auditable architecture and benchmark workflow.
2. Credibility-aware components demonstrate measurable internal value (ablation evidence).
3. A larger benchmark is required for robust external-performance claims.

## 5. Required Expansion to 5000 Claims

A direct attempt to construct a 5000-claim test set from currently available local benchmark files returns only **224 unique claims** after deduplication.

Therefore, publication-grade evaluation requires importing genuine external datasets (e.g., FEVER, LIAR, SciFact, HealthVer) and re-running the exact same architecture comparison suite on `splits_5000`.

Suggested command sequence:

```powershell
cd C:/Fact_Validator/services/api
python Scripts/run_5000_benchmark_pipeline.py `
  --input fever=C:/path/to/fever.csv `
  --input liar=C:/path/to/liar.csv `
  --input scifact=C:/path/to/scifact.csv `
  --input healthver=C:/path/to/healthver.csv `
  --target-test-size 5000 `
  --benchmark-output C:/Fact_Validator/data/benchmarks/results/large_benchmark_manifest.json `
  --splits-dir C:/Fact_Validator/data/benchmarks/splits_5000

python Scripts/run_benchmark_architecture_suite.py `
  --train C:/Fact_Validator/data/benchmarks/splits_5000/train.json `
  --test C:/Fact_Validator/data/benchmarks/splits_5000/test.json `
  --output-dir C:/Fact_Validator/data/benchmarks/results_5000
```

This preserves methodological consistency between current and future results, enabling a defensible revision from pilot-scale to publication-scale evidence.
