from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import os
import re


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    stop = {
        "the", "and", "for", "this", "that", "with", "from", "into", "about", "have", "has",
        "been", "were", "which", "when", "where", "while", "than", "also", "their", "they", "them",
    }
    return [t for t in text.split() if len(t) >= 3 and t not in stop]


def _lexical_semantic_score(claim: str, text: str) -> float:
    c = set(_tokenize(claim))
    t = set(_tokenize(text))
    if not c or not t:
        return 0.0
    jaccard = len(c.intersection(t)) / len(c.union(t))
    coverage = len(c.intersection(t)) / max(1, len(c))
    return max(0.0, min(1.0, 0.45 * jaccard + 0.55 * coverage))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (math.sqrt(na) * math.sqrt(nb))))


_MODEL = None
_MODEL_LOAD_ERROR: Optional[str] = None


def _get_sentence_transformer_model():
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        return None

    # Disabled by default - use lexical fallback for speed
    enabled = os.getenv("FACTVALIDATOR_EMBEDDINGS_ENABLED", "false").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        _MODEL_LOAD_ERROR = "disabled-by-default"
        return None

    model_name = os.getenv("FACTVALIDATOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _MODEL = SentenceTransformer(model_name)
        return _MODEL
    except Exception as ex:  # pragma: no cover - environment dependent
        _MODEL_LOAD_ERROR = f"model-load-failed: {ex.__class__.__name__}"
        return None


def semantic_rerank(
    claim: str,
    evidence_items: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Re-rank evidence by semantic similarity.

    - Preferred path: sentence-transformer embeddings cosine similarity.
    - Fallback path: lexical overlap similarity.
    """
    if not evidence_items:
        return [], {"enabled": True, "method": "none", "reason": "no-evidence"}

    items = [dict(e) for e in evidence_items]
    texts = [
        " ".join(
            [
                str(e.get("title") or ""),
                str(e.get("snippet") or ""),
                str(e.get("domain") or ""),
            ]
        ).strip()
        for e in items
    ]

    model = _get_sentence_transformer_model()
    method = "lexical-fallback"

    if model is not None:
        try:
            emb = model.encode([claim, *texts], normalize_embeddings=True)
            claim_vec = [float(v) for v in emb[0]]
            doc_vecs = [[float(v) for v in row] for row in emb[1:]]
            for e, vec in zip(items, doc_vecs):
                e["semantic_score"] = round(_cosine(claim_vec, vec), 4)
            method = "sentence-transformer"
        except Exception:  # pragma: no cover - runtime/model dependent
            for e, txt in zip(items, texts):
                e["semantic_score"] = round(_lexical_semantic_score(claim, txt), 4)
            method = "lexical-fallback"
    else:
        for e, txt in zip(items, texts):
            e["semantic_score"] = round(_lexical_semantic_score(claim, txt), 4)

    items.sort(
        key=lambda e: (
            float(e.get("semantic_score") or 0.0),
            int(e.get("domain_score") or 0),
            int(e.get("overlap") or 0),
        ),
        reverse=True,
    )

    if isinstance(top_k, int) and top_k > 0:
        items = items[:top_k]

    meta = {
        "enabled": True,
        "method": method,
        "model": os.getenv("FACTVALIDATOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "scores": [float(e.get("semantic_score") or 0.0) for e in items[:5]],
        "error": _MODEL_LOAD_ERROR,
    }
    return items, meta
