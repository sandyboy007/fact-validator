from __future__ import annotations

from typing import Any, Dict, Tuple
import os

from app.analysis_features import detect_stance


DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"
_PIPELINE: Any = None
_LOAD_ERROR: str | None = None


def _nli_enabled() -> bool:
    return os.getenv("FACTVALIDATOR_NLI_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _get_pipeline() -> Any:
    global _PIPELINE, _LOAD_ERROR
    if _PIPELINE is not None:
        return _PIPELINE
    if _LOAD_ERROR is not None or not _nli_enabled():
        return None
    try:
        from transformers import pipeline  # type: ignore

        _PIPELINE = pipeline(
            "text-classification",
            model=os.getenv("FACTVALIDATOR_NLI_MODEL", DEFAULT_NLI_MODEL).strip(),
            return_all_scores=True,
        )
        return _PIPELINE
    except Exception as exc:  # pragma: no cover - model/environment dependent
        _LOAD_ERROR = f"{exc.__class__.__name__}: {exc}"
        return None


def _map_label(label: str) -> str:
    normalized = (label or "").strip().lower()
    if "entail" in normalized:
        return "support"
    if "contradict" in normalized:
        return "refute"
    return "neutral"


def classify_relation(claim: str, passage: str) -> Tuple[str, Dict[str, Any]]:
    """Classify a retrieved passage as support, refutation, or neutral evidence."""
    classifier = _get_pipeline()
    model_name = os.getenv("FACTVALIDATOR_NLI_MODEL", DEFAULT_NLI_MODEL).strip()
    if classifier is None:
        return detect_stance(claim, passage), {
            "method": "heuristic-fallback",
            "model": None,
            "enabled": _nli_enabled(),
            "reason": _LOAD_ERROR or "nli-disabled",
        }

    try:
        results = classifier({"text": passage, "text_pair": claim}, truncation=True)
        scores = results[0] if results and isinstance(results[0], list) else results
        best = max(scores, key=lambda item: float(item.get("score") or 0.0))
        relation = _map_label(str(best.get("label") or ""))
        return relation, {
            "method": "mnli",
            "model": model_name,
            "enabled": True,
            "label": best.get("label"),
            "score": round(float(best.get("score") or 0.0), 4),
        }
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        return detect_stance(claim, passage), {
            "method": "heuristic-fallback",
            "model": model_name,
            "enabled": True,
            "reason": f"inference-failed:{exc.__class__.__name__}",
        }