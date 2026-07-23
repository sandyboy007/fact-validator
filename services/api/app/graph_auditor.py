from __future__ import annotations

from typing import Any, Dict, List


DECISIVE_RELATIONS = {"SUPPORTS", "REFUTES", "UNDERCUTS"}


def audit_evidence_graph(
    graph: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    claim_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate that an automated claim decision has a complete evidence basis."""
    checked_rules = [
        "retrieval-status",
        "provenance",
        "relation-grounding",
        "numeric-factor-coverage",
        "conflict-resolution",
        "source-independence",
    ]
    violations: List[str] = []
    status = str(graph.get("retrieval_status") or "")
    relation_counts = dict(graph.get("relation_counts") or {})
    retrieved = [item for item in evidence if item.get("retrieval_status") == "retrieved"]
    decisive_edges = sum(int(relation_counts.get(relation, 0)) for relation in DECISIVE_RELATIONS)

    if status in {"search_failed", "search_unavailable"}:
        violations.append("Retrieval did not complete successfully.")
    if not retrieved:
        violations.append("No fetched evidence passage is available for this claim.")
    if any(not item.get("content_hash") or not item.get("url") for item in retrieved):
        violations.append("A retrieved passage lacks a URL or content hash.")
    if retrieved and decisive_edges == 0:
        violations.append("No retrieved passage establishes a grounded argumentative relation.")
    if claim_profile.get("numbers") and not any(bool(item.get("numeric_match")) for item in retrieved):
        violations.append("Numeric claim factors are not matched by retrieved evidence.")
    if graph.get("unresolved_conflict"):
        violations.append("Comparable support and attack relations remain unresolved.")

    decisive_items = [
        item
        for item in retrieved
        if str(item.get("stance") or "") in {"support", "refute"}
    ]
    clusters = {item.get("independence_cluster") for item in decisive_items if item.get("independence_cluster")}
    if len(decisive_items) >= 2 and len(clusters) < 2:
        violations.append("Multiple decisive passages are correlated within one evidence cluster.")

    decision = "PROCEED"
    if status in {"search_failed", "search_unavailable"} or decisive_edges == 0:
        decision = "NEI"
    elif graph.get("unresolved_conflict"):
        decision = "CONFLICTING"
    elif violations:
        decision = "HUMAN_REVIEW"

    return {
        "version": "graph-auditor-v1",
        "passed": decision == "PROCEED",
        "decision": decision,
        "violations": violations,
        "checked_rules": checked_rules,
        "evidence_clusters": sorted(cluster for cluster in clusters if cluster),
    }