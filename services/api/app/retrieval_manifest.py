from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Dict, List


MANIFEST_VERSION = "retrieval-manifest-v1"


def _canonical_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def build_retrieval_manifest(
    claim: str,
    query: str,
    retrieval_status: str,
    raw_results: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    search_results = [
        {
            "rank": index,
            "title": item.get("title"),
            "url": item.get("link"),
            "snippet": item.get("snippet") or "",
        }
        for index, item in enumerate(raw_results, start=1)
    ]
    passages = [
        {
            "url": item.get("url"),
            "content_hash": item.get("content_hash"),
            "retrieval_status": item.get("retrieval_status"),
            "retrieved_at_utc": item.get("retrieved_at_utc"),
            "relation_classifier": item.get("relation_classifier"),
        }
        for item in evidence
    ]
    payload = {
        "version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim": claim,
        "query": query,
        "retrieval_status": retrieval_status,
        "search_results": search_results,
        "passages": passages,
    }
    payload["search_results_hash"] = _canonical_hash(search_results)
    payload["manifest_hash"] = _canonical_hash(payload)
    return payload