# Publication-Safe Claim Statement

## Required terminology

**Fact Validator live application** is the implemented open-web system with
retrieval, domain credibility scoring, semantic reranking, optional Ollama
debate, caching, persistence, and user-facing reports.

**FactValidator-Proxy** is the deterministic model evaluated on 5,000 frozen
claims. It does not execute the live components for each test item.

## Defensible main claim

> The deterministic FactValidator-Proxy obtains 50.82% accuracy and 0.4384
> macro-F1 on a frozen 5,000-claim benchmark assembled from FEVER, LIAR,
> SciFact, and the PUBHEALTH `health_fact` dataset. A proxy variant without
> deterministic debate obtains the best observed result, 51.34% accuracy and
> 0.4427 macro-F1. The contribution is an auditable, deployment-oriented
> verification architecture and a transparent proxy evaluation, not a claim
> of state-of-the-art predictive performance.

## Statistical boundary

Unadjusted exact two-sided McNemar comparisons of the full proxy against
majority and length produce p-values of 0.0433 and 0.0462. These are marginal
and are not described as robust superiority after correcting for multiple
comparisons. The comparison with the no-debate proxy gives p = 0.00955 and
favours removing always-on deterministic debate.

## Dataset terminology

The health subset is:

> PUBHEALTH `health_fact` dataset, stored under the legacy `healthver`
> compatibility name.

It must not be called the original HealthVer dataset.

## Novelty statement

> Unlike work focused primarily on factual-error correction or benchmark
> classification, Fact Validator contributes an auditable and
> deployment-oriented verification architecture. It combines explicit source
> credibility reasoning, evidence retrieval, persistent run records,
> operational analysis, optional deliberation, and provenance-aware evidence
> auditing. Its novelty is trustworthy systems integration rather than
> state-of-the-art classification accuracy.

## Claims not supported

Do not claim that:

- the live retrieval application achieved the 5,000-claim proxy result;
- Fact Validator empirically outperforms FacTool or reference research systems;
- always-on debate improves accuracy;
- the raw confidence score is a calibrated probability;
- the evidence-graph prototype improved the main 5,000-claim evaluation;
- operational projections are controlled load-test measurements.

## Artifact references

- `data/benchmarks/results_5000/run_manifest.json`
- `data/benchmarks/results_5000/statistics_report.json`
- `data/benchmarks/results_5000/statistics_summary.md`
- `services/api/Scripts/run_thesis_statistics.py`
