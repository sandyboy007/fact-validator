from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
import math
import re
from urllib.parse import urlparse


LegacyVerdict = Literal["SUPPORTED", "REFUTED", "NEI"]
StructuredVerdict = Literal[
    "Supported",
    "Likely supported",
    "Mixed / disputed",
    "Insufficient evidence",
    "Likely false",
    "False",
]


_TIME_SENSITIVE_PROFILES = {"health", "politics", "finance", "science", "climate", "conflict"}
_LOADED_LANGUAGE_TERMS = {
    "shocking",
    "bombshell",
    "exposed",
    "must see",
    "everyone knows",
    "mainstream media",
    "cover-up",
    "coverup",
    "wake up",
    "traitor",
    "evil",
    "miracle",
    "secret",
    "guaranteed",
    "proof",
}
_NEGATION_CUES = [
    "false",
    "hoax",
    "debunk",
    "misleading",
    "not true",
    "no evidence",
    "incorrect",
    "misinformation",
    "disinformation",
    "fabricated",
    "baseless",
    "refuted",
    "denied",
]
_SUPPORT_CUES = [
    "according to",
    "confirmed",
    "reported",
    "found that",
    "data show",
    "study found",
    "official figures",
    "evidence shows",
]
_EXPERTISE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "health": ("health", "vaccine", "virus", "disease", "hospital", "cdc", "nih", "medical", "covid", "pandemic", "who"),
    "science": ("study", "scientist", "research", "journal", "peer reviewed", "experiment"),
    "climate": ("climate", "warming", "carbon", "emissions", "temperature", "co2"),
    "finance": ("stock", "market", "economy", "inflation", "gdp", "earnings", "revenue", "dollar"),
    "politics": ("president", "election", "senate", "minister", "government", "policy", "vote"),
    "conflict": ("war", "military", "attack", "missile", "troops", "ceasefire", "invasion"),
    "history": ("century", "historical", "founded", "empire", "ancient", "born", "died"),
}
_PRIMARY_DOMAIN_HINTS = (".gov", ".edu", ".int", ".mil")
_PRIMARY_PATH_HINTS = ("report", "dataset", "data", "court", "filing", "press-release", "statistics", "official")
_PRIMARY_TITLE_HINTS = ("report", "official", "dataset", "filing", "court", "statistics", "study")
_SOURCE_TYPE_HINTS = {
    "official": (".gov", ".edu", ".int", ".mil"),
    "reference": ("wikipedia.org", "britannica.com"),
    "journal": ("nature.com", "science.org", "thelancet.com", "nejm.org", "jamanetwork.com", "bmj.com"),
    "newswire": ("reuters.com", "apnews.com", "afp.com"),
    "news": (),
}


def normalize_claim_key(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:300]


def tokenize_for_overlap(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    stop = {
        "this", "that", "with", "from", "were", "have", "has", "been", "into", "also",
        "their", "they", "them", "than", "more", "most", "such", "some", "many", "would",
        "could", "should", "about", "after", "before", "because", "while", "where", "when",
    }
    toks = [t for t in text.split() if len(t) >= 4 and t not in stop]
    return toks[:100]


def overlap_count(a: str, b: str) -> int:
    return len(set(tokenize_for_overlap(a)).intersection(tokenize_for_overlap(b)))


def overlap_ratio(a: str, b: str) -> float:
    a_toks = set(tokenize_for_overlap(a))
    if not a_toks:
        return 0.0
    return len(a_toks.intersection(tokenize_for_overlap(b))) / max(1, len(a_toks))


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text or "")


def extract_years(text: str) -> List[int]:
    years = []
    for m in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text or ""):
        try:
            years.append(int(m))
        except ValueError:
            continue
    return years


def extract_entities(text: str) -> List[str]:
    candidates = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,})\b", text or "")
    out: List[str] = []
    seen = set()
    for c in candidates:
        cc = c.strip()
        low = cc.lower()
        if low in {"the", "a", "an", "and", "or", "but"}:
            continue
        if low not in seen:
            seen.add(low)
            out.append(cc)
    return out[:12]


def split_atomic_claims(text: str) -> List[str]:
    raw = re.split(r"\b(?:and|but|while|although|because)\b|[;:]", text or "")
    out: List[str] = []
    for part in raw:
        s = " ".join((part or "").split()).strip(" ,.-")
        if 20 <= len(s) <= 220:
            out.append(s)
    return out[:4] or ([text.strip()] if text.strip() else [])


def infer_expertise_profile(text: str) -> str:
    low = (text or "").lower()
    best = (0, "general")
    for profile, words in _EXPERTISE_KEYWORDS.items():
        score = sum(1 for w in words if w in low)
        if score > best[0]:
            best = (score, profile)
    return best[1]


def loaded_language_terms(text: str) -> List[str]:
    low = (text or "").lower()
    return sorted(term for term in _LOADED_LANGUAGE_TERMS if term in low)


def decompose_claim(claim: str) -> Dict[str, Any]:
    return {
        "normalized_claim": normalize_claim_key(claim),
        "atomic_claims": split_atomic_claims(claim),
        "entities": extract_entities(claim),
        "numbers": extract_numbers(claim),
        "years": extract_years(claim),
        "expertise_profile": infer_expertise_profile(claim),
        "loaded_language_terms": loaded_language_terms(claim),
    }


def guess_source_type(domain: str, url: str = "", title: str = "") -> str:
    dom = (domain or "").lower()
    url_low = (url or "").lower()
    title_low = (title or "").lower()
    for source_type, hints in _SOURCE_TYPE_HINTS.items():
        if any(dom.endswith(h) or dom == h for h in hints):
            return source_type
    if any(dom.endswith(h) for h in _PRIMARY_DOMAIN_HINTS):
        return "official"
    if any(h in url_low for h in ("/blog", "substack", "medium.com")):
        return "commentary"
    if any(h in title_low for h in ("opinion", "editorial", "analysis")):
        return "commentary"
    return "news"


def detect_primary_source(domain: str, url: str = "", title: str = "") -> Tuple[bool, str]:
    dom = (domain or "").lower()
    parsed_path = (urlparse(url).path or "").lower() if url else ""
    title_low = (title or "").lower()

    if any(dom.endswith(suffix) for suffix in _PRIMARY_DOMAIN_HINTS):
        return True, "Official institution domain"
    if any(h in parsed_path for h in _PRIMARY_PATH_HINTS):
        return True, "URL suggests report/dataset/filing"
    if any(h in title_low for h in _PRIMARY_TITLE_HINTS):
        return True, "Title suggests direct report or dataset"
    if guess_source_type(dom, url, title) in {"official", "journal"}:
        return True, "Institutional or journal source"
    return False, "No strong primary-source signal"


def extract_snippet_year(title: str, snippet: str) -> Optional[int]:
    years = extract_years(f"{title} {snippet}")
    if not years:
        return None
    current_year = datetime.utcnow().year + 1
    valid = [y for y in years if 1900 <= y <= current_year]
    return max(valid) if valid else None


def recency_score(year: Optional[int], expertise_profile: str) -> float:
    if year is None:
        return 0.35 if expertise_profile in _TIME_SENSITIVE_PROFILES else 0.55
    age = max(0, datetime.utcnow().year - year)
    if expertise_profile in _TIME_SENSITIVE_PROFILES:
        if age <= 1:
            return 1.0
        if age <= 3:
            return 0.75
        if age <= 5:
            return 0.5
        return 0.2
    if age <= 2:
        return 0.85
    if age <= 5:
        return 0.7
    if age <= 10:
        return 0.55
    return 0.4


def exact_phrase_overlap(claim: str, snippet: str, min_terms: int = 4) -> bool:
    claim_words = tokenize_for_overlap(claim)
    if len(claim_words) < min_terms:
        return False
    for i in range(0, len(claim_words) - min_terms + 1):
        phrase = " ".join(claim_words[i : i + min_terms])
        if phrase and phrase in (snippet or "").lower():
            return True
    return False


def detect_stance(claim: str, snippet: str) -> Literal["support", "refute", "neutral"]:
    low = (snippet or "").lower()
    ov = overlap_count(claim, snippet)
    if ov < 2:
        return "neutral"
    if any(cue in low for cue in _NEGATION_CUES):
        return "refute"
    if any(cue in low for cue in _SUPPORT_CUES) or overlap_ratio(claim, snippet) >= 0.45:
        return "support"
    return "neutral"


def detect_manipulation_flags(claim: str, snippet: str, domain: str = "") -> List[str]:
    flags: List[str] = []
    low = f"{claim} {snippet}".lower()
    terms = loaded_language_terms(low)
    if terms:
        flags.append("loaded-language")
    if any(cue in low for cue in ("satire", "parody")):
        flags.append("satire-or-parody")
    if '"' in (snippet or "") and "according to" not in low:
        flags.append("quote-with-limited-context")
    if domain.endswith(".co") and any(fake in domain.lower() for fake in ("news", "daily", "report")):
        flags.append("lookalike-domain-risk")
    return flags


def expertise_match_score(expertise_profile: str, domain: str, title: str = "", snippet: str = "") -> float:
    low = f"{domain} {title} {snippet}".lower()
    if expertise_profile == "general":
        return 0.6
    words = _EXPERTISE_KEYWORDS.get(expertise_profile, ())
    hits = sum(1 for w in words if w in low)
    return min(1.0, 0.35 + hits * 0.18)


def numeric_match(claim_numbers: List[str], snippet: str) -> bool:
    if not claim_numbers:
        return True
    snip_numbers = set(extract_numbers(snippet))
    return any(n in snip_numbers for n in claim_numbers)


def entity_match(entities: List[str], snippet: str) -> bool:
    if not entities:
        return True
    low = (snippet or "").lower()
    return sum(1 for e in entities if e.lower() in low) >= max(1, math.ceil(len(entities) / 3))


def enrich_evidence(claim: str, claim_profile: Dict[str, Any], ev: Dict[str, Any]) -> Dict[str, Any]:
    title = ev.get("title") or ""
    snippet = ev.get("snippet") or ""
    title_and_snippet = f"{title} {snippet}".strip()
    domain = ev.get("domain") or ""
    url = ev.get("url") or ""
    domain_score = int(ev.get("domain_score") or 0)
    semantic_score = float(ev.get("semantic_score") or 0.0)
    primary_source, primary_reason = detect_primary_source(domain, url, title)
    source_type = guess_source_type(domain, url, title)
    year = extract_snippet_year(title, snippet)
    recent = recency_score(year, claim_profile.get("expertise_profile", "general"))
    directness = overlap_ratio(claim, title_and_snippet)
    quote_grounded = exact_phrase_overlap(claim, title_and_snippet) or directness >= 0.5
    stance = detect_stance(claim, title_and_snippet)
    manipulation_flags = detect_manipulation_flags(claim, title_and_snippet, domain)
    expertise_score = expertise_match_score(claim_profile.get("expertise_profile", "general"), domain, title, snippet)
    numbers_ok = numeric_match(claim_profile.get("numbers", []), title_and_snippet)
    entities_ok = entity_match(claim_profile.get("entities", []), title_and_snippet)

    quality = (
        domain_score * 0.46
        + min(int(ev.get("overlap") or 0), 10) * 2.1
        + semantic_score * 16
        + (12 if primary_source else 0)
        + recent * 12
        + directness * 14
        + expertise_score * 8
        + (6 if quote_grounded else 0)
        + (4 if numbers_ok else -5)
        + (4 if entities_ok else -4)
        - len(manipulation_flags) * 4
    )
    quality = max(0.0, min(100.0, quality))

    enriched = dict(ev)
    enriched.update(
        {
            "source_type": source_type,
            "semantic_score": round(semantic_score, 4),
            "primary_source": primary_source,
            "primary_source_reason": primary_reason,
            "published_year": year,
            "recency_score": round(recent, 2),
            "directness_score": round(directness, 2),
            "quote_grounded": bool(quote_grounded),
            "stance": stance,
            "manipulation_flags": manipulation_flags,
            "expertise_match": round(expertise_score, 2),
            "numeric_match": bool(numbers_ok),
            "entity_match": bool(entities_ok),
            "quality_score": round(quality, 1),
        }
    )
    return enriched


def summarize_evidence(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    supports = [e for e in evidence if e.get("stance") == "support"]
    refutes = [e for e in evidence if e.get("stance") == "refute"]
    primary = [e for e in evidence if e.get("primary_source")]
    high_cred = [e for e in evidence if int(e.get("domain_score") or 0) >= 80]
    years = [int(y) for y in (e.get("published_year") for e in evidence) if isinstance(y, int)]
    oldest = min(years) if years else None
    newest = max(years) if years else None
    conflict_gap = abs(len(supports) - len(refutes))
    if supports and refutes and conflict_gap <= 1:
        conflict_level = "high"
    elif supports and refutes:
        conflict_level = "medium"
    else:
        conflict_level = "low"

    return {
        "evidence_count": len(evidence),
        "high_credibility_sources": len(high_cred),
        "primary_source_count": len(primary),
        "primary_source_present": bool(primary),
        "supporting_items": len(supports),
        "refuting_items": len(refutes),
        "conflict_level": conflict_level,
        "distinct_domains": len({e.get("base_domain") or e.get("domain") for e in evidence}),
        "oldest_citation_year": oldest,
        "newest_citation_year": newest,
        "average_quality_score": round(sum(float(e.get("quality_score") or 0.0) for e in evidence) / max(1, len(evidence)), 1),
    }


def map_structured_to_legacy(verdict: StructuredVerdict) -> LegacyVerdict:
    if verdict in {"Supported", "Likely supported"}:
        return "SUPPORTED"
    if verdict in {"Likely false", "False"}:
        return "REFUTED"
    return "NEI"


def determine_verdict(claim: str, claim_profile: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize_evidence(evidence)
    supports = sorted((e for e in evidence if e.get("stance") == "support"), key=lambda x: float(x.get("quality_score") or 0), reverse=True)
    refutes = sorted((e for e in evidence if e.get("stance") == "refute"), key=lambda x: float(x.get("quality_score") or 0), reverse=True)
    best_support = float(supports[0].get("quality_score")) if supports else 0.0
    best_refute = float(refutes[0].get("quality_score")) if refutes else 0.0
    support_domains = {e.get("base_domain") for e in supports if e.get("base_domain")}
    refute_domains = {e.get("base_domain") for e in refutes if e.get("base_domain")}
    uncertainty: List[str] = []

    if not evidence:
        verdict: StructuredVerdict = "Insufficient evidence"
        confidence = 0.42
        uncertainty.append("No evidence was retrieved for this claim.")
    elif supports and refutes and abs(best_support - best_refute) <= 12:
        verdict = "Mixed / disputed"
        confidence = 0.58
        uncertainty.append("Credible evidence points in both directions.")
    elif best_refute >= 78 and (len(refute_domains) >= 2 or any(e.get("primary_source") for e in refutes[:2])):
        verdict = "False"
        confidence = min(0.9, 0.72 + best_refute / 500)
    elif best_refute >= 63:
        verdict = "Likely false"
        confidence = min(0.82, 0.58 + best_refute / 600)
        if len(refute_domains) < 2:
            uncertainty.append("Refutation evidence is not yet corroborated by multiple independent domains.")
    elif best_support >= 82 and (len(support_domains) >= 2 or any(e.get("primary_source") for e in supports[:2])):
        verdict = "Supported"
        confidence = min(0.9, 0.7 + best_support / 500)
    elif best_support >= 65:
        verdict = "Likely supported"
        confidence = min(0.82, 0.56 + best_support / 650)
        if len(support_domains) < 2 and not any(e.get("primary_source") for e in supports[:2]):
            uncertainty.append("Support comes from limited independent corroboration.")
    else:
        verdict = "Insufficient evidence"
        confidence = 0.5
        uncertainty.append("Retrieved evidence was too weak or indirect for a stronger verdict.")

    if claim_profile.get("numbers") and not any(e.get("numeric_match") for e in evidence):
        uncertainty.append("No evidence directly matched the numeric part of the claim.")
    if claim_profile.get("entities") and not any(e.get("entity_match") for e in evidence):
        uncertainty.append("Key named entities in the claim were not matched clearly in the evidence.")
    if summary.get("primary_source_present") is False and claim_profile.get("expertise_profile") in {"health", "finance", "science", "conflict"}:
        uncertainty.append("No primary source was found for a domain-sensitive claim.")
    if any(e.get("manipulation_flags") for e in evidence):
        uncertainty.append("Some evidence items show manipulation-risk signals or limited context.")

    needs_human_review = (
        verdict in {"Mixed / disputed", "Insufficient evidence"}
        or len(uncertainty) >= 2
        or summary.get("conflict_level") == "high"
    )
    human_review_reason = None
    if needs_human_review:
        human_review_reason = uncertainty[0] if uncertainty else "The tool cannot reach a stable automated conclusion."

    explanation_parts = [
        f"Top support quality: {best_support:.1f}",
        f"top refute quality: {best_refute:.1f}",
        f"primary sources: {summary.get('primary_source_count', 0)}",
        f"conflict level: {summary.get('conflict_level', 'low')}",
    ]

    return {
        "structured_verdict": verdict,
        "legacy_verdict": map_structured_to_legacy(verdict),
        "confidence": round(float(max(0.05, min(0.95, confidence))), 2),
        "explanation": "; ".join(explanation_parts),
        "uncertainty_reasons": uncertainty[:4],
        "needs_human_review": needs_human_review,
        "human_review_reason": human_review_reason,
        "evidence_summary": summary,
    }
