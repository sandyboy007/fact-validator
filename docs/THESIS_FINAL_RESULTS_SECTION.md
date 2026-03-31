# Thesis-Ready Comparative Evaluation Section

Author: Fact Validator Project Team  
Date: 2026-04-01

## 1. Results and Comparative Narrative

### 1.1 Evaluation Setup

The benchmark pipeline was rerun on the diversified v2 dataset and generated a stratified split with:
- Train claims: 143
- Validation claims: 46
- Test claims: 51

Primary evidence files:
- data/benchmarks/results/comparative_analysis_summary.md
- data/benchmarks/results/ablation_study_summary.md
- data/benchmarks/results/baseline_comparison_summary.md
- data/benchmarks/results/production_metrics_summary.md

Note:
- The v2 dataset is auto-generated and still contains some duplicate patterns.
- Results are useful for stress-testing and pipeline validation, but should be treated as provisional for thesis-grade scientific claims.

### 1.2 Main Quantitative Results (n=51 test claims)

#### Comparative ranking

| System | Accuracy | 95% CI | Avg Confidence | Calibration Error | ECE |
|---|---:|---:|---:|---:|---:|
| random | 0.373 | [0.240, 0.505] | 47.7 | 0.104 | 0.134 |
| majority | 0.353 | [0.222, 0.484] | 50.0 | 0.147 | 0.147 |
| length | 0.314 | [0.186, 0.441] | 49.0 | 0.176 | 0.176 |
| keyword | 0.294 | [0.169, 0.419] | 32.5 | 0.031 | 0.165 |
| sentiment | 0.294 | [0.169, 0.419] | 40.0 | 0.106 | 0.106 |
| ablate_semantic_rerank | 0.235 | [0.119, 0.352] | 40.6 | 0.170 | 0.212 |
| full_proxy | 0.216 | [0.103, 0.329] | 47.9 | 0.263 | 0.295 |
| ablate_debate | 0.216 | [0.103, 0.329] | 47.9 | 0.263 | 0.295 |
| ablate_quality_filter | 0.196 | [0.087, 0.305] | 39.2 | 0.196 | 0.196 |
| ablate_credibility | 0.137 | [0.043, 0.232] | 35.1 | 0.214 | 0.214 |

Interpretation:
- In this run, the full system does not outperform simple baselines.
- Most pairwise comparisons remain non-significant (p >= 0.05).
- This indicates benchmark realism is currently the limiting factor, not just sample size.

#### Baseline-only snapshot

| Baseline | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| random | 0.373 | 0.383 | 0.374 | 0.375 |
| keyword | 0.294 | 0.218 | 0.322 | 0.215 |
| length | 0.314 | 0.231 | 0.311 | 0.242 |
| sentiment | 0.294 | 0.098 | 0.333 | 0.152 |
| majority | 0.353 | 0.118 | 0.333 | 0.174 |

### 1.3 Ablation Findings

| Variant | Accuracy | Macro F1 | Delta Accuracy vs Full | Delta Macro F1 vs Full |
|---|---:|---:|---:|---:|
| full_proxy | 0.216 | 0.212 | +0.000 | +0.000 |
| ablate_credibility | 0.137 | 0.102 | -0.078 | -0.110 |
| ablate_semantic_rerank | 0.235 | 0.225 | +0.020 | +0.013 |
| ablate_debate | 0.216 | 0.212 | +0.000 | +0.000 |
| ablate_quality_filter | 0.196 | 0.177 | -0.020 | -0.034 |

Interpretation:
- Credibility removal hurts performance, suggesting credibility contributes value.
- Debate has neutral effect in this run.
- Semantic reranking impact is small and unstable under current synthetic distribution.

### 1.4 Production and Cost Metrics

| Metric | Value |
|---|---:|
| Baseline latency (sec) | 8.20 |
| Debate latency (sec) | 72.00 |
| Debate/Baseline ratio | 8.78x |
| Baseline throughput (claims/hour) | 439.02 |
| Debate throughput (claims/hour) | 50.00 |
| Monthly cost without cache (USD) | 77.00 |
| Monthly cost with cache (USD) | 22.00 |
| Monthly savings (USD) | 55.00 |
| Monthly savings (%) | 71.43% |

Interpretation:
- Engineering and cost-efficiency claims remain strong and repeatable.
- Quality claims remain benchmark-sensitive.

### 1.5 Defensible Claims vs Non-Defensible Claims

Defensible now:
- The evaluation pipeline is reproducible and runs at larger scale than initial n=7.
- Cost and operational metrics are stable and well-documented.
- Statistical-testing infrastructure is in place.

Not yet defensible:
- Superiority of full model over baselines on benchmark v2.
- Broad generalization claims against prior literature from auto-generated claims.

## 2. Threats to Validity

### 2.1 Internal validity
- Generated claims still include duplicate or highly similar patterns.
- Distribution artifacts may bias model behavior and ablation outcomes.

### 2.2 External validity
- Synthetic claims are not a complete proxy for real-world misinformation.
- Domain complexity and linguistic diversity are still limited.

### 2.3 Statistical conclusion validity
- While n=51 improves over n=7, it is still insufficient for robust publication claims with this data quality.
- Non-significant comparisons indicate uncertain inferential strength.

## 3. Next Step Required for Thesis-Strong Evidence

To produce publishable-quality comparative claims:
1. Replace synthetic rows with manually curated and independently annotated claims.
2. Eliminate near-duplicate statements.
3. Maintain at least 150-250 test claims after cleaning.
4. Re-run this exact pipeline and update this section from fresh outputs.

---

## Recommended Final Thesis Paragraph (Ready to Paste)

This work delivers a reproducible, deployment-ready fact-validation pipeline with transparent reporting, statistical comparison tooling, and measurable operational efficiency. Expanded evaluation runs demonstrate system-level robustness and cost effectiveness, but model-quality findings remain sensitive to benchmark realism and claim diversity. Therefore, the primary validated contribution is methodological and infrastructural, with definitive comparative performance claims deferred pending evaluation on a curated, independently annotated benchmark.
