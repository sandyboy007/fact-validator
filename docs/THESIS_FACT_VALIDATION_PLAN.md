# Fact Validation Thesis Plan

## Recommendation

The proposed thesis direction is good and feasible if the contribution is framed as an evaluated method, not as a claim that the current application already performs semantic or argumentative verification.

**Working contribution statement:** This thesis evaluates whether passage-grounded evidence graphs with explicit support, refutation, undercutting, and audit-forced abstention improve open-domain fact validation over lexical, NLI-only, and flat evidence baselines.

The central contribution should be the conflict-aware evidence graph plus justified abstention. Multi-agent debate and correction are secondary ablations, not the thesis core.

## Implemented Vertical Slice

The API now includes the following research-facing behaviour:

- short factual statements and dates are retained during candidate extraction;
- search-result URLs are fetched and a claim-relevant passage is selected before verification;
- returned evidence records include the passage, retrieval status, retrieval timestamp, and SHA-256 content hash;
- retrieval failure, unavailable search, and no results are represented separately;
- a typed graph records `DEPENDS_ON`, `CITES`, `SUPPORTS`, `REFUTES`, and `UNDERCUTS` relations;
- graph adjudication returns `CONFLICTING` when similarly strong direct support and attack relations remain unresolved;
- the grounding gate returns `NEI` when no fetched passage establishes a relation or retrieval was unavailable/failed;
- the user interface no longer presents the legacy misinformation heuristic as a risk probability.

These are deterministic, inspectable mechanisms. Current relation classification remains heuristic and is therefore a baseline implementation, not a validated semantic NLI method.

## Research Questions

1. Does atomic claim decomposition with full-passage grounding improve evidence and verdict accuracy over sentence-level snippet matching?
2. Do typed support, refutation, and undercut graphs improve validation when evidence is contradictory or multi-step?
3. Does a grounding auditor reduce selective risk by abstaining on missing, failed, weak, or conflicting evidence?

## Required Next Work

1. Add a benchmark adapter for AVeriTeC and select FEVER or SciFact as the secondary diagnostic benchmark.
2. Implement frozen retrieval manifests containing query, ranked URLs, raw search response hashes, passage hashes, model/prompt versions, and run date.
3. Evaluate the configurable local MNLI relation baseline independently from retrieval using `services/api/Scripts/run_relation_baseline.py` and a frozen claim-passage relation set. The default API mode remains an explicitly labelled heuristic fallback until the NLI model is enabled.
4. Add graph annotations for a contradictory-evidence challenge set, including undercuts and missing-premise cases.
5. Run the required baselines under identical evidence and model/token budgets: lexical baseline, BM25 plus NLI, dense retrieval plus NLI, evidence-constrained LLM, graph without undercuts, graph without auditor, and full system.
6. Report claim extraction F1, Recall@k, MRR, evidence-relation macro-F1, verdict macro-F1, joint evidence-verdict score, graph-edge F1, risk-coverage/AURC, false-accept rate, cost, latency, and paired confidence intervals.

## Implemented Audit Extensions

- Evidence selection now keeps the best matching sentence with its immediate context window, reducing loss of qualifiers and exceptions.
- Retrieved passages are clustered by publisher and near-duplicate content before corroboration is interpreted.
- The graph auditor checks retrieval status, URL/hash provenance, relation grounding, numeric coverage, unresolved conflict, and correlation among decisive sources.
- The API exposes `graph_audit` and `independence_clusters` in every graph-enabled claim result. An audit failure requests human review; unavailable/ungrounded evidence yields `NEI`, and unresolved opposition yields `CONFLICTING`.
- `data/challenges/conflict_undercut_v1_seed.json` and its annotation protocol provide a versioned seed for future expert-annotated graph and robustness evaluation. They are not benchmark evidence.
- `services/api/Scripts/run_retrieval_robustness.py` replays only frozen, explicitly supplied evidence corruptions and compares graph-only with full audited-graph coverage, selective risk, and false-accept rate.

## Thesis Claims To Avoid

- Do not describe the legacy heuristic as calibrated misinformation probability.
- Do not claim semantic entailment, correction faithfulness, or multi-agent reasoning unless each is evaluated with ground truth.
- Do not claim that the graph method improves factuality until the controlled ablations above are complete.

## Current Status

This repository is now positioned as a passage-grounded, conflict-aware prototype and a transparent lexical baseline. It is not yet a completed experimental thesis system because benchmark adapters, semantic relation baselines, annotated graph data, and controlled evaluation remain outstanding.