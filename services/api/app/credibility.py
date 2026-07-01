from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import json
import os
import time
import tldextract


# Default cache path is relative to this file so it works on any OS.
# Override via the FACTVALIDATOR_DOMAIN_CACHE environment variable.
_DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "domain_cache.json",
)
CACHE_PATH = os.getenv("FACTVALIDATOR_DOMAIN_CACHE", _DEFAULT_CACHE_PATH)
CACHE_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days
CACHE_VERSION = 2

_DEFAULT_OPENSOURCES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "opensources.json",
)
_DEFAULT_IFFY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "iffy_index.json",
)

# OpenSources type label → score delta (most-negative wins when multiple types present)
_OS_TYPE_DELTA: Dict[str, int] = {
    "reliable": 10,
    "political": 0,
    "bias": -5,
    "satire": -10,
    "clickbait": -10,
    "rumor": -15,
    "state": -15,
    "unreliable": -15,
    "junksci": -20,
    "conspiracy": -20,
    "fake news": -30,
    "fake": -30,
    "hate": -30,
}

# Iffy Index MBFC factual-reporting level → score delta
_IFFY_LEVEL_DELTA: Dict[str, int] = {
    "VL": -25,  # Very Low
    "L": -15,   # Low
    "M": -5,    # Mixed
}


def _load_opensources() -> Dict[str, Dict]:
    try:
        if not os.path.exists(_DEFAULT_OPENSOURCES_PATH):
            return {}
        with open(_DEFAULT_OPENSOURCES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_iffy() -> Dict[str, Dict]:
    try:
        if not os.path.exists(_DEFAULT_IFFY_PATH):
            return {}
        with open(_DEFAULT_IFFY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("_meta", None)
        return data
    except Exception:
        return {}


def _build_os_lookup(raw: Dict[str, Dict]) -> Dict[str, int]:
    """Map base_domain -> worst-type delta from OpenSources data."""
    lookup: Dict[str, int] = {}
    for domain_key, info in raw.items():
        bd = base_domain(domain_key.strip())
        if not bd:
            continue
        types = [
            str(info.get("type", "")).lower().strip(),
            str(info.get("2nd type", "")).lower().strip(),
            str(info.get("3rd type", "")).lower().strip(),
        ]
        worst_delta = 0
        for t in types:
            d = _OS_TYPE_DELTA.get(t, 0)
            if d < worst_delta:
                worst_delta = d
        if worst_delta != 0:
            if bd not in lookup or worst_delta < lookup[bd]:
                lookup[bd] = worst_delta
    return lookup


def _build_iffy_lookup(raw: Dict[str, Dict]) -> Dict[str, Tuple[int, str]]:
    """Map base_domain -> (delta, level_label) from Iffy Index data."""
    lookup: Dict[str, Tuple[int, str]] = {}
    for domain_key, info in raw.items():
        bd = base_domain(domain_key.strip())
        if not bd:
            continue
        level = str(info.get("level", "")).upper().strip()
        delta = _IFFY_LEVEL_DELTA.get(level, 0)
        if delta != 0:
            if bd not in lookup or delta < lookup[bd][0]:
                lookup[bd] = (delta, level)
    return lookup


@dataclass
class CredibilityScore:
    score: int
    label: str
    reasons: Dict[str, str]


def base_domain(domain: str) -> str:
    ext = tldextract.extract(domain or "")
    if not ext.domain or not ext.suffix:
        return (domain or "").lower()
    return f"{ext.domain}.{ext.suffix}".lower()


# Module-level lookups — built once at import time (after base_domain is defined)
_OS_LOOKUP: Dict[str, int] = _build_os_lookup(_load_opensources())
_IFFY_LOOKUP: Dict[str, Tuple[int, str]] = _build_iffy_lookup(_load_iffy())


def _load_cache() -> Dict[str, Dict]:
    try:
        if not os.path.exists(CACHE_PATH):
            return {}
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    cache_dir = os.path.dirname(os.path.abspath(CACHE_PATH))
    os.makedirs(cache_dir, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _label(score: int) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def score_domain_rubric(domain: str) -> CredibilityScore:
    """
    Transparent rubric inspired by source credibility checklists.
    This is NOT NewsGuard; it's a thesis-safe, explainable rubric.
    """
    d = (domain or "").lower().strip()
    bd = base_domain(d)

    cache = _load_cache()
    now = int(time.time())

    cached = cache.get(bd)
    if cached and cached.get("version") == CACHE_VERSION and (now - int(cached.get("ts", 0))) < CACHE_TTL_SECONDS:
        return CredibilityScore(
            score=int(cached["score"]),
            label=cached["label"],
            reasons=cached["reasons"],
        )

    # Rubric components (0..100)
    score = 45
    reasons: Dict[str, str] = {
        "neutral": "No direct source signal matched; conservative default starts at 45 rather than 50."
    }

    # Strong signals – covers .gov, .edu, and international gov variants
    # e.g. gov.uk, gc.ca, gouv.fr, govt.nz, gov.au, etc.
    _gov_patterns = (
        ".gov", ".edu",                      # US
        "gov.uk", "gov.au", "gov.nz",        # Commonwealth
        "gc.ca", "gouv.fr", "governo.it",    # Others
        "bund.de", "government.se",
    )
    if any(bd.endswith(p) for p in _gov_patterns) or bd in _gov_patterns:
        score += 35
        reasons["tld"] = "Government/Education domain suffix is a strong trust signal (+35)."

    # Known reputable references (extend over time)
    whitelist = {
        # Institutions / references
        "who.int": 35,
        "nih.gov": 35,
        "cdc.gov": 35,
        "ipcc.ch": 30,
        "un.org": 30,
        "oecd.org": 30,
        "worldbank.org": 30,
        "ec.europa.eu": 30,

        # Science
        "nature.com": 25,
        "science.org": 25,
        "sciencedirect.com": 15,

        # Data publishers
        "ourworldindata.org": 20,
        "iea.org": 20,
        "unep.org": 20,

        # Encyclopedia / general reference
        "britannica.com": 20,
        "wikipedia.org": 15,

        # Major news wires / broadcasters
        "bbc.com": 30,
        "bbc.co.uk": 30,
        "reuters.com": 30,
        "apnews.com": 30,
        "afp.com": 30,

        # US national newspapers / magazines
        "nytimes.com": 25,
        "washingtonpost.com": 25,
        "wsj.com": 25,
        "usatoday.com": 20,
        "newsweek.com": 20,
        "time.com": 20,
        "theatlantic.com": 20,

        # US business / finance press
        "forbes.com": 25,
        "bloomberg.com": 25,
        "ft.com": 25,
        "economist.com": 25,
        "marketwatch.com": 20,
        "businessinsider.com": 15,
        "cnbc.com": 20,
        "investopedia.com": 15,

        # US broadcast / public media
        "cnn.com": 25,
        "nbcnews.com": 25,
        "abcnews.go.com": 25,
        "cbsnews.com": 25,
        "npr.org": 30,
        "pbs.org": 30,
        "vox.com": 15,
        "thehill.com": 15,
        "politico.com": 20,

        # International quality outlets
        "theguardian.com": 25,
        "bbc.in": 25,
        "dw.com": 25,
        "aljazeera.com": 20,
        "france24.com": 20,
        "rfi.fr": 20,
        "lemonde.fr": 20,
        "elpais.com": 20,
        "derspiegel.de": 20,
        "corriere.it": 20,

        # Major Indian news (English language)
        "thehindu.com": 20,
        "hindustantimes.com": 20,
        "ndtv.com": 20,
        "livemint.com": 20,
        "economictimes.com": 15,
        "indiatoday.in": 15,

        # Academic publishers
        "springer.com": 20,
        "wiley.com": 20,
        "cell.com": 20,
        "thelancet.com": 25,
        "nejm.org": 30,
        "jamanetwork.com": 25,
        "bmj.com": 25,
        "pubmed.ncbi.nlm.nih.gov": 35,
        "scholar.google.com": 20,
        "jstor.org": 20,
        "ssrn.com": 15,
        "arxiv.org": 15,
        "researchgate.net": 10,

        # NGOs / think tanks
        "amnesty.org": 20,
        "hrw.org": 20,
        "weforum.org": 20,
        "brookings.edu": 30,
        "rand.org": 25,
        "pewresearch.org": 25,
        "cfr.org": 20,
        "chathamhouse.org": 20,
    }

    if bd in whitelist:
        score += whitelist[bd]
        reasons["whitelist"] = f"Domain appears in local reputable-source list (+{whitelist[bd]})."

    # Weak signals / risks
    risk_markers = [
        ("blogspot.", -20, "User-generated blog hosting is higher risk (-20)."),
        ("wordpress.", -20, "User-generated blog hosting is higher risk (-20)."),
        ("medium.com", -10, "Open publishing platform: quality varies (-10)."),
        ("substack.com", -10, "Newsletter platform: quality varies (-10)."),
        ("facebook.com", -20, "Social platform: high repost/misinformation risk (-20)."),
        ("tiktok.com", -20, "Social platform: high repost/misinformation risk (-20)."),
        ("x.com", -20, "Social platform: high repost/misinformation risk (-20)."),
        ("twitter.com", -20, "Social platform: high repost/misinformation risk (-20)."),
        ("reddit.com", -15, "Forum platform: quality varies (-15)."),
        ("youtube.com", -10, "Video platform: varies widely (-10)."),
    ]
    for marker, delta, msg in risk_markers:
        if marker in d:
            score += delta
            reasons["platform_risk"] = msg
            break

    # --- OpenSources dataset signal ---
    if bd in _OS_LOOKUP:
        delta = _OS_LOOKUP[bd]
        score += delta
        sign = "+" if delta >= 0 else ""
        reasons["opensources"] = (
            f"Domain found in OpenSources unreliable-news dataset ({sign}{delta})."
        )

    # --- Iffy Index (MBFC-backed) signal ---
    if bd in _IFFY_LOOKUP:
        delta, level = _IFFY_LOOKUP[bd]
        level_label = {"VL": "Very Low", "L": "Low", "M": "Mixed"}.get(level, level)
        score += delta
        sign = "+" if delta >= 0 else ""
        reasons["iffy_index"] = (
            f"Domain flagged by Iffy Index (MBFC factual rating: {level_label}, {sign}{delta})."
        )

    # Obvious spammy/low-quality keyword markers
    low_markers = [
        ("hoax", -30),
        ("clickbait", -25),
        ("rumor", -25),
        ("conspiracy", -25),
    ]
    for marker, delta in low_markers:
        if marker in d:
            score += delta
            reasons["keyword_risk"] = f"Domain contains marker '{marker}' indicating higher risk ({delta})."
            break

    score = max(0, min(score, 100))
    label = _label(score)

    cache[bd] = {"version": CACHE_VERSION, "score": score, "label": label, "reasons": reasons, "ts": now}
    _save_cache(cache)

    return CredibilityScore(score=score, label=label, reasons=reasons)
