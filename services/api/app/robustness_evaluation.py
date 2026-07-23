from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any, Dict, List

from app.analysis_features import decompose_claim, enrich_evidence
from app.evidence_graph import adjudicate_graph, build_evidence_graph
from app.evidence_independence import cluster_evidence
from app.graph_auditor import audit_evidence_graph
from app.relation_classifier import classify_relation


DECISIVE = {"SUPPORTED", "REFUTED"}


def _default_evidence(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    passage = str(item.get("passage") or "").strip()
    if not passage:
        return []
    return [{
        "url": str(item.get("url") or f"https://seed.example/{item.get('id', 'item')}").strip(),
        "title": str(item.get("title") or "Frozen evidence passage"),
        "snippet": passage,
        "passage": passage,
        "content_hash": sha256(passage.encode("utf-8")).hexdigest(),
        "retrieval_status": "retrieved",
        "domain": str(item.get("domain") or "seed.example"),
        "base_domain": str(item.get("base_domain") or "seed.example"),
        "domain_score": int(item.get("domain_score") or 80),
        "overlap": 0,
    }]


def _prepare_evidence(claim: str, raw_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    profile = decompose_claim(claim)
    prepared: List[Dict[str, Any]] = []
    for item in deepcopy(raw_evidence):
        passage = str(item.get("passage") or item.get("snippet") or "").strip()
        if not passage:
            continue
        item.setdefault("snippet", passage)
        item.setdefault("passage", passage)
        item.setdefault("content_hash", sha256(passage.encode("utf-8")).hexdigest())
        item.setdefault("retrieval_status", "retrieved")
        item.setdefault("domain", "unknown.example")
        item.setdefault("base_domain", item["domain"])
        item.setdefault("domain_score", 50)
        item.setdefault("overlap", 0)
        enriched = enrich_evidence(claim, profile, item)
        stance, metadata = classify_relation(claim, passage)
        enriched["stance"] = stance
        enriched["relation_classifier"] = metadata
        prepared.append(enriched)
    cluster_evidence(prepared)
    return prepared


def _graph_decision(graph: Dict[str, Any]) -> str:
    abstention, _ = adjudicate_graph(graph)
    if abstention:
        return abstention
    counts = dict(graph.get("relation_counts") or {})
    support = int(counts.get("SUPPORTS", 0))
    attack = int(counts.get("REFUTES", 0)) + int(counts.get("UNDERCUTS", 0))
    if support > attack:
        return "SUPPORTED"
    if attack > support:
        return "REFUTED"
    return "NEI"


def evaluate_case(item: Dict[str, Any], scenario_name: str, raw_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    claim = str(item.get("claim") or "").strip()
    profile = decompose_claim(claim)
    evidence = _prepare_evidence(claim, raw_evidence)
    status = "ok" if evidence else "no_results"
    graph = build_evidence_graph(claim, list(profile.get("atomic_claims") or [claim]), evidence, status)
    graph["independence_clusters"] = cluster_evidence(evidence)
    graph_only = _graph_decision(graph)
    audit = audit_evidence_graph(graph, evidence, profile)
    audited = graph_only
    if audit["decision"] in {"NEI", "CONFLICTING"}:
        audited = audit["decision"]
    elif audit["decision"] == "HUMAN_REVIEW":
        audited = "HUMAN_REVIEW"

    return {
        "id": item.get("id"),
        "scenario": scenario_name,
        "claim": claim,
        "expected_verdict": item.get("expected_verdict"),
        "graph_only": graph_only,
        "full_audited_graph": audited,
        "audit": audit,
        "evidence_count": len(evidence),
        "independence_clusters": graph["independence_clusters"],
    }


def build_scenarios(item: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Use only frozen, explicitly supplied evidence for non-clean corruptions."""
    scenarios = {"clean": list(item.get("evidence") or _default_evidence(item))}
    for name, evidence in dict(item.get("corruptions") or {}).items():
        if isinstance(evidence, list):
            scenarios[str(name)] = evidence
    return scenarios


def selective_metrics(rows: List[Dict[str, Any]], system: str) -> Dict[str, float]:
    if not rows:
        return {"coverage": 0.0, "selective_risk": 0.0, "false_accept_rate": 0.0}
    decisions = [row[system] for row in rows]
    decisive = [row for row in rows if row[system] in DECISIVE]
    incorrect = [row for row in decisive if row[system] != row.get("expected_verdict")]
    return {
        "coverage": round(len(decisive) / len(rows), 4),
        "selective_risk": round(len(incorrect) / max(1, len(decisive)), 4),
        "false_accept_rate": round(len(incorrect) / len(rows), 4),
    }