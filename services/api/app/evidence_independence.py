from __future__ import annotations

from typing import Any, Dict, List
import re


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split()
        if len(token) >= 4
    }


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens.union(right_tokens))


def cluster_evidence(evidence: List[Dict[str, Any]], duplicate_threshold: float = 0.78) -> List[Dict[str, Any]]:
    """Assign transparent source-independence clusters to retrieved passages."""
    clusters: List[Dict[str, Any]] = []
    for item in evidence:
        if item.get("retrieval_status") != "retrieved":
            item["independence_cluster"] = None
            item["independence_reason"] = "Passage was not retrieved."
            continue

        domain = str(item.get("base_domain") or item.get("domain") or "").lower()
        passage = str(item.get("passage") or "")
        matched_cluster = None
        match_reason = ""
        for cluster in clusters:
            if domain and domain == cluster["domain"]:
                matched_cluster = cluster
                match_reason = "Same base domain."
                break
            if _similarity(passage, cluster["representative_passage"]) >= duplicate_threshold:
                matched_cluster = cluster
                match_reason = "Near-duplicate passage content."
                break

        if matched_cluster is None:
            matched_cluster = {
                "id": f"cluster:{len(clusters) + 1}",
                "domain": domain,
                "representative_passage": passage,
                "members": [],
            }
            clusters.append(matched_cluster)
            match_reason = "New independent source cluster."

        matched_cluster["members"].append(item)
        item["independence_cluster"] = matched_cluster["id"]
        item["independence_reason"] = match_reason

    return [
        {
            "id": cluster["id"],
            "domain": cluster["domain"],
            "evidence_count": len(cluster["members"]),
            "urls": [member.get("url") for member in cluster["members"]],
        }
        for cluster in clusters
    ]