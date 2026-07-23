from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List, Tuple


RELATION_TYPES = {"SUPPORTS", "REFUTES", "UNDERCUTS", "DEPENDS_ON", "CITES"}


def passage_hash(text: str) -> str:
    return sha256((text or "").encode("utf-8")).hexdigest()


def _evidence_id(index: int, evidence: Dict[str, Any]) -> str:
    return f"evidence:{index}:{passage_hash(str(evidence.get('passage') or evidence.get('snippet') or ''))[:12]}"


def _relation_for_evidence(evidence: Dict[str, Any]) -> str | None:
    if evidence.get("retrieval_status") != "retrieved":
        return None
    if not evidence.get("numeric_match"):
        return None

    directness = float(evidence.get("directness_score") or 0.0)
    stance = str(evidence.get("stance") or "neutral")
    passage = str(evidence.get("passage") or evidence.get("snippet") or "").lower()

    if stance == "support" and directness >= 0.30:
        return "SUPPORTS"
    if stance == "refute" and directness >= 0.20:
        if any(marker in passage for marker in ("context", "exception", "only", "not necessarily", "depends on")):
            return "UNDERCUTS"
        return "REFUTES"
    return None


def build_evidence_graph(
    claim_text: str,
    atomic_claims: List[str],
    evidence: List[Dict[str, Any]],
    retrieval_status: str,
) -> Dict[str, Any]:
    claim_id = f"claim:{passage_hash(claim_text)[:12]}"
    nodes: List[Dict[str, Any]] = [{"id": claim_id, "type": "CLAIM", "text": claim_text}]
    edges: List[Dict[str, Any]] = []

    for index, atomic_claim in enumerate(atomic_claims):
        atomic_id = f"atomic:{index}:{passage_hash(atomic_claim)[:12]}"
        nodes.append({"id": atomic_id, "type": "ATOMIC_CLAIM", "text": atomic_claim})
        edges.append({"source": claim_id, "target": atomic_id, "type": "DEPENDS_ON"})

    relation_counts = {relation: 0 for relation in RELATION_TYPES}
    for index, item in enumerate(evidence):
        evidence_id = _evidence_id(index, item)
        nodes.append(
            {
                "id": evidence_id,
                "type": "EVIDENCE_PASSAGE",
                "url": item.get("url"),
                "content_hash": item.get("content_hash") or passage_hash(str(item.get("passage") or "")),
                "retrieval_status": item.get("retrieval_status"),
            }
        )
        edges.append({"source": claim_id, "target": evidence_id, "type": "CITES"})
        relation_counts["CITES"] += 1

        relation = _relation_for_evidence(item)
        if relation:
            edges.append(
                {
                    "source": evidence_id,
                    "target": claim_id,
                    "type": relation,
                    "quality": item.get("quality_score"),
                    "reason": "Passage-level alignment with matched claim factors.",
                }
            )
            relation_counts[relation] += 1

    support = [
        float(item.get("quality_score") or 0.0)
        for item in evidence
        if _relation_for_evidence(item) == "SUPPORTS"
    ]
    attacks = [
        float(item.get("quality_score") or 0.0)
        for item in evidence
        if _relation_for_evidence(item) in {"REFUTES", "UNDERCUTS"}
    ]
    conflict = bool(support and attacks and abs(max(support) - max(attacks)) <= 12.0)

    return {
        "version": "1.0",
        "retrieval_status": retrieval_status,
        "nodes": nodes,
        "edges": edges,
        "relation_counts": relation_counts,
        "unresolved_conflict": conflict,
        "decision_basis": "fetched_passages_only",
    }


def adjudicate_graph(graph: Dict[str, Any]) -> Tuple[str | None, List[str]]:
    status = str(graph.get("retrieval_status") or "")
    counts = dict(graph.get("relation_counts") or {})
    reasons: List[str] = []
    if status in {"search_failed", "search_unavailable"}:
        return "NEI", ["Evidence retrieval failed or was unavailable; absence of results is not evidence of absence."]
    if graph.get("unresolved_conflict"):
        return "CONFLICTING", ["Direct support and attack relations remain unresolved in the evidence graph."]
    if not any(int(counts.get(key, 0)) for key in ("SUPPORTS", "REFUTES", "UNDERCUTS")):
        return "NEI", ["No fetched passage grounded a support, refutation, or undercut relation."]
    return None, reasons