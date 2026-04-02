from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from app.analysis_features import overlap_ratio, tokenize_for_overlap


_FACTOR_STOP_TERMS = {
    "no",
    "all",
    "every",
    "any",
    "always",
    "never",
    "source",
    "sources",
    "credible",
    "expert",
    "experts",
    "dataset",
    "datasets",
    "evidence",
    "report",
    "reports",
    "region",
    "regions",
    "country",
    "countries",
    "world",
    "worldwide",
    "time",
    "history",
    "claim",
}


DEFAULT_REFLECTIVE_THRESHOLDS: Dict[str, float] = {
    "hallucination_quality_min": 45.0,
    "hallucination_directness_min": 0.15,
    "strong_quality_min": 56.0,
    "conflict_quality_gap_max": 6.0,
    "low_factor_coverage_pct": 24.0,
}


def _resolve_thresholds(thresholds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    cfg = dict(DEFAULT_REFLECTIVE_THRESHOLDS)
    if thresholds:
        for key, value in thresholds.items():
            if key in cfg:
                try:
                    cfg[key] = float(value)
                except (TypeError, ValueError):
                    continue
    return cfg


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _first_sentence(text: str, max_len: int = 220) -> str:
    text = _safe_text(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    sent = parts[0].strip() if parts else text
    sent = sent[:max_len].strip(" ,")
    if sent and sent[-1] not in ".!?":
        sent += "."
    return sent


def _build_factor_inventory(claim_text: str, claim_profile: Dict[str, Any]) -> Dict[str, Any]:
    entities = [
        str(e).lower()
        for e in (claim_profile.get("entities") or [])
        if str(e).strip()
        and str(e).strip().lower() not in _FACTOR_STOP_TERMS
        and len(str(e).strip()) >= 3
    ]
    numbers = [str(n) for n in (claim_profile.get("numbers") or []) if str(n).strip()]
    years = [str(y) for y in (claim_profile.get("years") or []) if str(y).strip()]

    term_source = " ".join(claim_profile.get("atomic_claims") or [claim_text])
    terms = [
        t
        for t in tokenize_for_overlap(term_source)
        if len(t) >= 4 and t not in _FACTOR_STOP_TERMS
    ]
    if not terms:
        terms = tokenize_for_overlap(claim_text)

    # Keep stable and compact factor space.
    terms = terms[:14]
    entities = entities[:10]
    numbers = numbers[:8]
    years = years[:6]

    factors: List[Dict[str, str]] = []
    seen = set()

    for et in entities:
        key = ("entity", et)
        if key not in seen:
            seen.add(key)
            factors.append({"type": "entity", "value": et})

    for num in numbers:
        key = ("number", num)
        if key not in seen:
            seen.add(key)
            factors.append({"type": "number", "value": num})

    for yr in years:
        key = ("year", yr)
        if key not in seen:
            seen.add(key)
            factors.append({"type": "year", "value": yr})

    for term in terms:
        key = ("term", term)
        if key not in seen:
            seen.add(key)
            factors.append({"type": "term", "value": term})

    return {
        "entities": entities,
        "numbers": numbers,
        "years": years,
        "terms": terms,
        "factors": factors,
    }


def _factor_in_evidence(factor: Dict[str, str], evidence_item: Dict[str, Any]) -> bool:
    val = factor["value"]
    factor_type = factor["type"]

    text = f"{_safe_text(evidence_item.get('title'))} {_safe_text(evidence_item.get('snippet'))}".lower()

    if factor_type in {"entity", "term"}:
        return val in text

    if factor_type in {"number", "year"}:
        nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
        return val in nums

    return False


def _compute_reflective_metrics(
    claim_text: str,
    claim_profile: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    cfg = _resolve_thresholds(thresholds)
    inventory = _build_factor_inventory(claim_text, claim_profile)
    factors = inventory["factors"]
    n_gt = max(1, len(factors))

    used_factors: List[Dict[str, str]] = []
    omitted_factors: List[Dict[str, str]] = []

    for factor in factors:
        if any(_factor_in_evidence(factor, e) for e in evidence):
            used_factors.append(factor)
        else:
            omitted_factors.append(factor)

    n_used = len(used_factors)
    factor_utilization_recall = (n_used / n_gt) * 100.0

    hallucination_risk = 0
    for e in evidence:
        stance = str(e.get("stance") or "neutral")
        quality = float(e.get("quality_score") or 0.0)
        directness = float(e.get("directness_score") or 0.0)
        manip = list(e.get("manipulation_flags") or [])
        entity_match = bool(e.get("entity_match", False))
        numeric_match = bool(e.get("numeric_match", False))

        if stance in {"support", "refute"}:
            if (
                quality < cfg["hallucination_quality_min"]
                or directness < cfg["hallucination_directness_min"]
                or (not entity_match and not numeric_match)
            ):
                hallucination_risk += 1
        if manip:
            hallucination_risk += 1

    hallucination_accuracy = (1.0 - (hallucination_risk / n_gt)) * 100.0
    hallucination_accuracy = max(0.0, min(100.0, hallucination_accuracy))

    support_items = [e for e in evidence if str(e.get("stance") or "") == "support"]
    refute_items = [e for e in evidence if str(e.get("stance") or "") == "refute"]
    strong_items = [
        e
        for e in evidence
        if float(e.get("quality_score") or 0.0) >= cfg["strong_quality_min"]
        and str(e.get("stance") or "") in {"support", "refute"}
    ]

    avg_support = (
        sum(float(e.get("quality_score") or 0.0) for e in support_items) / max(1, len(support_items))
    )
    avg_refute = (
        sum(float(e.get("quality_score") or 0.0) for e in refute_items) / max(1, len(refute_items))
    )

    support_domains = {
        str(e.get("base_domain") or e.get("domain") or "")
        for e in support_items
        if str(e.get("base_domain") or e.get("domain") or "")
    }
    refute_domains = {
        str(e.get("base_domain") or e.get("domain") or "")
        for e in refute_items
        if str(e.get("base_domain") or e.get("domain") or "")
    }

    has_numeric_requirements = bool(claim_profile.get("numbers"))
    has_entity_requirements = bool(claim_profile.get("entities"))
    numeric_covered = any(bool(e.get("numeric_match")) for e in evidence)
    entity_covered = any(bool(e.get("entity_match")) for e in evidence)

    conflict = bool(
        support_items
        and refute_items
        and abs(avg_support - avg_refute) <= cfg["conflict_quality_gap_max"]
    )
    low_coverage = factor_utilization_recall < cfg["low_factor_coverage_pct"]

    terminate_reason: Optional[str] = None
    if not evidence:
        terminate_reason = "No evidence was retrieved."
    elif not strong_items:
        terminate_reason = "No strong support/refute evidence met quality threshold."
    elif has_numeric_requirements and not numeric_covered:
        terminate_reason = "Numeric elements in the claim were not grounded in evidence."
    elif has_entity_requirements and not entity_covered and low_coverage:
        terminate_reason = "Claim entities were weakly grounded and factor coverage is low."
    elif conflict and len(support_domains) <= 1 and len(refute_domains) <= 1:
        terminate_reason = "Support and refute evidence conflict without enough independent corroboration."

    decision = "TERMINATE" if terminate_reason else "PROCEED"

    analyst_issues: List[str] = []
    if low_coverage:
        analyst_issues.append("Low factor utilization coverage.")
    if conflict:
        analyst_issues.append("Conflicting evidence signals detected.")
    if has_numeric_requirements and not numeric_covered:
        analyst_issues.append("Numeric mismatch across evidence.")
    if has_entity_requirements and not entity_covered:
        analyst_issues.append("Entity grounding is weak.")

    polisher_suggestions: List[str] = []
    if omitted_factors:
        omitted_preview = ", ".join(f["value"] for f in omitted_factors[:4])
        polisher_suggestions.append(f"Address omitted claim factors: {omitted_preview}.")
    if conflict:
        polisher_suggestions.append("Add independent sources to resolve support/refute conflict.")
    if not polisher_suggestions:
        polisher_suggestions.append("Evidence coverage and grounding are acceptable for automated output.")

    return {
        "decision": decision,
        "abstain_reason": terminate_reason,
        "factor_utilization_recall": round(factor_utilization_recall, 2),
        "hallucination_accuracy": round(hallucination_accuracy, 2),
        "abstention_ratio": 100.0 if decision == "TERMINATE" else 0.0,
        "factor_coverage": {
            "total_factors": len(factors),
            "used_count": len(used_factors),
            "omitted_count": len(omitted_factors),
            "used": used_factors,
            "omitted": omitted_factors,
        },
        "factor_analyst": {
            "issues": analyst_issues,
            "supporting_evidence_count": len(support_items),
            "refuting_evidence_count": len(refute_items),
            "strong_evidence_count": len(strong_items),
            "support_domains": sorted(support_domains),
            "refute_domains": sorted(refute_domains),
            "avg_support_quality": round(avg_support, 2),
            "avg_refute_quality": round(avg_refute, 2),
            "conflict_signal": conflict,
        },
        "argument_polisher": {
            "suggestions": polisher_suggestions,
        },
    }


def reflective_analysis(
    claim_text: str,
    claim_profile: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    return _compute_reflective_metrics(claim_text, claim_profile, evidence, thresholds=thresholds)


def _template_correction(claim_text: str) -> Optional[str]:
    low = claim_text.lower().strip()

    m = re.match(r"^no credible source has ever measured (.+?) at any time in history\.?$", low)
    if m:
        return f"Credible sources have measured {m.group(1)}."

    m = re.match(r"^every published claim about (.+?) after \d{4} has been fabricated\.?$", low)
    if m:
        return f"Not every published claim about {m.group(1)} has been fabricated."

    m = re.match(r"^all experts agree that (.+?) has never changed in any region since \d{4}\.?$", low)
    if m:
        return f"Experts do not all agree that {m.group(1)} has never changed."

    m = re.match(r"^every dataset proves that (.+?) is always identical across all countries\.?$", low)
    if m:
        return f"Datasets do not prove that {m.group(1)} is always identical across all countries."

    return None


def generate_faithful_correction(
    claim_text: str,
    claim_profile: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    reflective_report: Dict[str, Any],
    verdict: str,
) -> Optional[Dict[str, Any]]:
    if verdict != "REFUTED":
        return None

    if reflective_report.get("decision") == "TERMINATE":
        return None

    if not evidence:
        return None

    sorted_items = sorted(
        evidence,
        key=lambda e: (float(e.get("quality_score") or 0.0), float(e.get("directness_score") or 0.0)),
        reverse=True,
    )

    candidates: List[Dict[str, Any]] = []

    template = _template_correction(claim_text)
    if template:
        top = sorted_items[0]
        rel = overlap_ratio(claim_text, template)
        faith = 0.55 + min(0.3, float(top.get("quality_score") or 0.0) / 300.0)
        score = max(0.0, min(1.0, 0.6 * faith + 0.4 * rel))
        candidates.append(
            {
                "text": template,
                "score": round(score, 3),
                "method": "template_inversion",
                "source_url": top.get("url"),
                "source_domain": top.get("domain"),
            }
        )

    for item in sorted_items[:4]:
        sent = _first_sentence(item.get("snippet") or "")
        if len(sent) < 32:
            continue

        rel = overlap_ratio(claim_text, sent)
        faith = (
            float(item.get("directness_score") or 0.0) * 0.4
            + (1.0 if item.get("quote_grounded") else 0.0) * 0.2
            + (1.0 if item.get("primary_source") else 0.0) * 0.15
            + (float(item.get("quality_score") or 0.0) / 100.0) * 0.25
        )
        score = max(0.0, min(1.0, 0.6 * faith + 0.4 * rel))

        candidates.append(
            {
                "text": sent,
                "score": round(score, 3),
                "method": "evidence_sentence",
                "source_url": item.get("url"),
                "source_domain": item.get("domain"),
            }
        )

    if not candidates:
        return None

    best = max(candidates, key=lambda c: float(c.get("score") or 0.0))
    if float(best.get("score") or 0.0) < 0.45:
        return None

    return {
        "proposed_correction": best["text"],
        "score": best["score"],
        "method": best["method"],
        "source_url": best.get("source_url"),
        "source_domain": best.get("source_domain"),
        "alternatives": [c for c in sorted(candidates, key=lambda x: float(x.get("score") or 0.0), reverse=True)[:3]],
    }
