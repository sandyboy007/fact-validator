from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, ValidationError
from typing import Optional, List, Literal, Dict, Any, Tuple
from datetime import datetime
import json
from urllib.parse import urljoin, urlparse
import os
import re
import asyncio
import time
from pathlib import Path
import ipaddress
import socket
from hashlib import sha256

import httpx
import trafilatura
from dotenv import load_dotenv
import tldextract

# Load environment variables before importing modules that evaluate env at import time.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

import nltk
from nltk.tokenize import sent_tokenize

from app.analysis_features import (
    decompose_claim,
    determine_verdict,
    enrich_evidence,
    normalize_claim_key,
)
from app.credibility import score_domain_rubric
from app.reflective import reflective_analysis, generate_faithful_correction
from app.semantic_retrieval import semantic_rerank
from app.sentiment import analyze_sentiment, estimate_bias_risk, calculate_sentiment_misinformation_adjustment, get_sentiment_summary
from app.source_routes import router as source_router
from app.storage import (
    save_run,
    list_runs,
    get_run,
    export_runs,
    get_claim_memory,
    save_claim_memory,
    init_db,
)
from app.debate import llm_debate_verdict, llm_final_judge
from app.logger import log_analyze_start, log_analyze_complete, log_debate_started, log_debate_error
from app.config import Config
from app.cache import get_cache
from app.security import ollama_health, rate_limiter
from app.evidence_graph import build_evidence_graph, adjudicate_graph
from app.relation_classifier import classify_relation
from app.retrieval_manifest import MANIFEST_VERSION, build_retrieval_manifest
from app.evidence_independence import cluster_evidence
from app.graph_auditor import audit_evidence_graph

 

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()


class EvidenceSearchError(RuntimeError):
    pass


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find("tokenizers/punkt_tab/english.pickle")
    except Exception:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


_ensure_nltk()

app = FastAPI(title="Fact Validator API", version="0.8.2")

# Initialize database on startup
init_db()

# Add CORS middleware FIRST (before other middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://192.168.0.106:3000",
        "http://192.168.0.106:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

# Add rate limiting middleware (skip OPTIONS for CORS preflight)
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for OPTIONS requests (CORS preflight)
    if request.method == "OPTIONS":
        return await call_next(request)
    
    if Config.FEATURE_RATE_LIMITING:
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Max 100 requests per minute."}
            )
    response = await call_next(request)
    return response

app.include_router(source_router)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = PROJECT_ROOT / "docs"
BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "data" / "benchmarks" / "results"


def _load_json_report(file_path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback


def _get_model_fields(model_class):
    """Get model fields from Pydantic v1 or v2 models."""
    if hasattr(model_class, 'model_fields'):
        return model_class.model_fields
    elif hasattr(model_class, '__fields__'):
        return model_class.__fields__
    return {}


# ----------------------------
# Models
# ----------------------------

class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    mode: Literal["live", "snapshot"] = "live"
    verifier: Literal["baseline", "debate"] = "baseline"

    # optional control knobs
    enable_weighted_confidence: bool = False
    min_source_score: int = 50
    require_independent_domains: bool = True
    min_overlap: int = 6
    max_sources_per_base_domain: int = 2

    max_claims: int = 6
    max_evidence_per_claim: int = 5
    max_debate_claims: int = 2
    enable_reflective_abstention: bool = True
    enable_faithful_correction: bool = True
    
    @validator("url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) > Config.MAX_INPUT_URL_LENGTH:
            raise ValueError(f"URL exceeds max length of {Config.MAX_INPUT_URL_LENGTH}")
        return v
    
    @validator("text")
    @classmethod
    def validate_text(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) > Config.MAX_INPUT_TEXT_LENGTH:
            raise ValueError(f"Text exceeds max length of {Config.MAX_INPUT_TEXT_LENGTH}")
        return v
    
    @validator("max_claims")
    @classmethod
    def validate_max_claims(cls, v: int) -> int:
        if v > Config.MAX_CLAIMS_HARD_LIMIT:
            raise ValueError(f"max_claims exceeds limit of {Config.MAX_CLAIMS_HARD_LIMIT}")
        if v < 1:
            raise ValueError("max_claims must be at least 1")
        return v
    
    @validator("max_evidence_per_claim")
    @classmethod
    def validate_max_evidence(cls, v: int) -> int:
        if v > Config.MAX_EVIDENCE_PER_CLAIM_LIMIT:
            raise ValueError(f"max_evidence_per_claim exceeds limit of {Config.MAX_EVIDENCE_PER_CLAIM_LIMIT}")
        if v < 1:
            raise ValueError("max_evidence_per_claim must be at least 1")
        return v


class EvidenceItem(BaseModel):
    url: str
    title: Optional[str] = None
    snippet: str
    passage: Optional[str] = None
    content_hash: Optional[str] = None
    retrieval_status: Optional[Literal["retrieved", "fetch_failed", "not_fetched"]] = None
    retrieved_at_utc: Optional[str] = None
    domain: str
    domain_score: int
    overlap: int = 0
    semantic_score: Optional[float] = None
    quality_score: Optional[float] = None
    stance: Optional[Literal["support", "refute", "neutral"]] = None
    source_type: Optional[str] = None
    primary_source: Optional[bool] = None
    primary_source_reason: Optional[str] = None
    published_year: Optional[int] = None
    recency_score: Optional[float] = None
    directness_score: Optional[float] = None
    quote_grounded: Optional[bool] = None
    expertise_match: Optional[float] = None
    numeric_match: Optional[bool] = None
    entity_match: Optional[bool] = None
    manipulation_flags: Optional[List[str]] = None
    independence_cluster: Optional[str] = None
    independence_reason: Optional[str] = None


class ClaimResult(BaseModel):
    claim_text: str
    verdict: Literal["SUPPORTED", "REFUTED", "NEI", "CONFLICTING"]
    confidence: float
    evidence: List[EvidenceItem]
    debate_summary: Optional[str] = None
    structured_verdict: Optional[str] = None
    uncertainty_reasons: Optional[List[str]] = None
    needs_human_review: Optional[bool] = None
    human_review_reason: Optional[str] = None
    claim_profile: Optional[Dict[str, Any]] = None

    # optional Step 11 outputs
    adjusted_verdict: Optional[Literal["SUPPORTED", "REFUTED", "NEI"]] = None
    adjusted_confidence: Optional[float] = None
    evidence_summary: Optional[Dict[str, Any]] = None
    reflective: Optional[Dict[str, Any]] = None
    faithful_correction: Optional[Dict[str, Any]] = None
    evidence_graph: Optional[Dict[str, Any]] = None
    retrieval_manifest: Optional[Dict[str, Any]] = None
    graph_audit: Optional[Dict[str, Any]] = None


class AnalyzeResponse(BaseModel):
    input_type: Literal["url", "text"]
    domain: Optional[str] = None
    extracted_text_chars: int
    extracted_text_preview: str
    domain_score: int
    domain_label: Literal["HIGH", "MEDIUM", "LOW"]
    final_misinformation_likelihood: float
    claims: List[ClaimResult]
    timestamp_utc: str
    metadata: Dict[str, Any]


# ----------------------------
# Helpers
# ----------------------------

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "https://" + u
    return u


def extract_domain(url: str) -> Optional[str]:
    try:
        u = normalize_url(url)
        if not u:
            return None
        parsed = urlparse(u)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def domain_from_any_url(url: str) -> str:
    return extract_domain(url) or ""


def base_domain(domain: str) -> str:
    ext = tldextract.extract(domain or "")
    if not ext.domain or not ext.suffix:
        return (domain or "").lower()
    return f"{ext.domain}.{ext.suffix}".lower()


def is_blocked_domain(domain: str) -> bool:
    bd = base_domain((domain or "").lower())
    blocked = {
        "facebook.com", "x.com", "twitter.com", "tiktok.com",
        "instagram.com", "reddit.com", "pinterest.com",
        "worldarticledatabase.com", "wecanfigurethisout.org",
    }
    return bd in blocked


async def fetch_html(url: str) -> Tuple[str, str]:
    current_url = normalize_url(url)
    if not current_url:
        return "", ""

    try:
        headers = {
            "User-Agent": "FactValidatorBot/0.8.2 (thesis demo)",
            "Accept": "text/html,application/xhtml+xml",
        }
        timeout = httpx.Timeout(connect=20.0, read=30.0, write=20.0, pool=30.0)
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=False,
            timeout=timeout,
            trust_env=False,
        ) as client:
            for _ in range(5):
                if not await is_public_http_url(current_url):
                    return "", ""
                r = await client.get(current_url)
                if r.is_redirect:
                    location = r.headers.get("location")
                    if not location:
                        return "", ""
                    current_url = urljoin(current_url, location)
                    continue
                if r.status_code >= 400:
                    return "", ""
                return str(r.url), r.text
    except Exception:
        return "", ""
    return "", ""


async def is_public_http_url(url: str) -> bool:
    parsed = urlparse(normalize_url(url))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    host = parsed.hostname
    if host.lower() == "localhost":
        return False
    try:
        addresses = await asyncio.get_running_loop().run_in_executor(
            None,
            socket.getaddrinfo,
            host,
            None,
        )
    except OSError:
        return False

    try:
        resolved = {ipaddress.ip_address(item[4][0]) for item in addresses}
    except ValueError:
        return False
    return bool(resolved) and all(
        not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        )
        for address in resolved
    )


def extract_readable_text_from_html(html: str, url: str = "") -> str:
    try:
        extracted = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
        return (extracted or "").strip()
    except Exception:
        return ""


def heuristic_claim_score(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    length = len(s)
    length_score = 1.0 - min(abs(length - 160) / 160.0, 1.0)
    has_number = any(ch.isdigit() for ch in s)
    number_bonus = 0.15 if has_number else 0.0
    has_relation = any(tok in s.lower() for tok in [" is ", " are ", " was ", " were ", " has ", " have ", " with "])
    relation_bonus = 0.10 if has_relation else 0.0
    score = 0.55 * length_score + number_bonus + relation_bonus
    return max(0.05, min(score, 0.95))


def clean_text_for_claims(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bExplore Data\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bResearch\s*&\s*Writing\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def extract_claim_candidates(text: str, max_claims: int = 6) -> List[str]:
    text = clean_text_for_claims(text)
    if not text:
        return []

    blocks = [b.strip() for b in text.split("\n") if b.strip()]
    candidates: List[str] = []
    seen = set()

    boilerplate_markers = [
        "this topic page can be cited as",
        "published online at",
        "cite this work",
        "reuse this work",
        "license terms",
        "creative commons",
        "open access",
        "data produced by third parties",
        "the underlying data for this chart",
    ]

    for b in blocks[:250]:
        for s in sent_tokenize(b):
            s2 = " ".join(s.split()).strip()
            if len(s2) < 8 or len(s2) > 280:
                continue
            low = s2.lower()
            if any(m in low for m in boilerplate_markers):
                continue
            key = s2.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(s2)
            if len(candidates) >= 120:
                break
        if len(candidates) >= 120:
            break

    ranked = sorted(candidates, key=heuristic_claim_score, reverse=True)
    return ranked[:max_claims]


async def serpapi_search(query: str, num: int = 5) -> List[Dict[str, Any]]:
    if not SERPAPI_API_KEY:
        return []

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": str(num),
        "hl": "en",
        "gl": "us",
        "no_cache": "true",
    }

    try:
        timeout = httpx.Timeout(connect=20.0, read=45.0, write=20.0, pool=45.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            r = await client.get("https://serpapi.com/search", params=params)
            r.raise_for_status()
            data = r.json()
            organic = data.get("organic_results", []) or []
            out = []
            for item in organic[:num]:
                out.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet") or "",
                    }
                )
            return out
    except Exception as exc:
        raise EvidenceSearchError("Search provider request failed.") from exc


def tokenize_for_overlap(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 4]
    stop = {
        "this", "that", "with", "from", "were", "have", "has", "been", "into", "also",
        "their", "they", "them", "than", "more", "most", "such", "some", "many"
    }
    toks = [t for t in toks if t not in stop]
    return toks[:80]


def compute_overlap(claim: str, snippet: str) -> int:
    claim_toks = set(tokenize_for_overlap(claim))
    if not claim_toks:
        return 0
    snip_toks = set(tokenize_for_overlap(snippet))
    return len(claim_toks.intersection(snip_toks))


def select_evidence_passage(claim: str, text: str, max_chars: int = 1200) -> str:
    sentences = [" ".join(sentence.split()).strip() for sentence in sent_tokenize(text or "")]
    candidates = [sentence for sentence in sentences if len(sentence) >= 30]
    if not candidates:
        return " ".join((text or "").split())[:max_chars]
    best_index = max(range(len(candidates)), key=lambda index: compute_overlap(claim, candidates[index]))
    return " ".join(candidates[max(0, best_index - 1):min(len(candidates), best_index + 2)])[:max_chars]


async def fetch_evidence_passage(url: str, claim: str) -> Tuple[str, str, str]:
    final_url, html = await fetch_html(url)
    if not html:
        return "", final_url or url, "fetch_failed"
    passage = select_evidence_passage(claim, extract_readable_text_from_html(html, final_url or url))
    if not passage:
        return "", final_url or url, "fetch_failed"
    return passage, final_url or url, "retrieved"


def baseline_verdict(
    claim: str, evidence: List[EvidenceItem]
) -> Tuple[Literal["SUPPORTED", "REFUTED", "NEI"], float, str]:
    if not evidence:
        return "NEI", 0.55, "No evidence retrieved."

    claim_toks = set(tokenize_for_overlap(claim))
    if not claim_toks:
        return "NEI", 0.55, "Claim tokenization empty."

    neg_cues = ["false", "hoax", "debunk", "misleading", "not true", "no evidence",
                "incorrect", "misinformation", "disinformation", "fabricated", "baseless"]
    overlaps = []
    for e in evidence:
        ev_toks = set(tokenize_for_overlap(e.snippet))
        overlap = len(claim_toks.intersection(ev_toks))
        overlaps.append((overlap, e.domain_score, base_domain(e.domain), (e.snippet or "").lower()))

    # ------------------------------------------------------------------ #
    # REFUTATION: any credible source (score >= 65) contains a rebuttal cue
    # ------------------------------------------------------------------ #
    for ov, ds, bd, snip_low in overlaps:
        if ds >= 65 and any(cue in snip_low for cue in neg_cues):
            conf = min(0.90, 0.65 + ds / 1000)  # scales slightly with credibility
            return "REFUTED", round(conf, 2), (
                f"Credible source '{bd}' (score {ds}) contains a refutation signal."
            )

    # ------------------------------------------------------------------ #
    # SUPPORT: 2+ independent credible domains corroborate
    # ------------------------------------------------------------------ #
    strong_support_domains = {bd for (ov, ds, bd, _) in overlaps if ds >= 65 and ov >= 5}
    if len(strong_support_domains) >= 2:
        return "SUPPORTED", 0.78, (
            "Two or more independent medium/high-credibility domains corroborate the claim."
        )

    # ------------------------------------------------------------------ #
    # SUPPORT: single HIGH-credibility source (score >= 80) with overlap >= 5
    # ------------------------------------------------------------------ #
    best_high = max(
        ((ov, ds, bd) for (ov, ds, bd, _) in overlaps if ds >= 80),
        key=lambda x: (x[0], x[1]),
        default=None,
    )
    if best_high and best_high[0] >= 5:
        ov, ds, bd = best_high
        conf = round(min(0.82, 0.60 + ds / 1000 + ov * 0.008), 2)
        return "SUPPORTED", conf, (
            f"High-credibility source '{bd}' (score {ds}) corroborates the claim."
        )

    # ------------------------------------------------------------------ #
    # SUPPORT: single MEDIUM-credibility source (score >= 65) with overlap >= 7
    # ------------------------------------------------------------------ #
    best_med = max(
        ((ov, ds, bd) for (ov, ds, bd, _) in overlaps if ds >= 65),
        key=lambda x: (x[0], x[1]),
        default=None,
    )
    if best_med and best_med[0] >= 7:
        ov, ds, bd = best_med
        conf = round(min(0.72, 0.52 + ds / 1000 + ov * 0.007), 2)
        return "SUPPORTED", conf, (
            f"Medium-credibility source '{bd}' (score {ds}) has strong keyword overlap with the claim."
        )

    # ------------------------------------------------------------------ #
    # NEI: confidence weighted by the best available source quality
    # ------------------------------------------------------------------ #
    best_any = max(overlaps, key=lambda x: (x[1], x[0]))
    nei_conf = round(min(0.58, 0.40 + best_any[1] / 1000), 2)
    return "NEI", nei_conf, "Evidence retrieved but insufficient strength for a definitive verdict."


def estimate_misinformation_likelihood(
    claims: List[ClaimResult],
    input_domain_score: int = 50,
) -> float:
    """
    Misinformation likelihood anchored to source credibility.

    Domain-score mapping (base likelihood before claim adjustments):
      score=100  →  0.10  (very credible source)
      score=80   →  0.26
      score=75   →  0.30
      score=65   →  0.38
      score=50   →  0.50  (neutral / unknown)
      score=25   →  0.70
      score=0    →  0.90  (low-credibility source)
    """
    # Anchor: linear map of domain credibility → base misinformation prior
    base = round(0.90 - (min(max(input_domain_score, 0), 100) / 100.0) * 0.80, 4)

    if not claims:
        return float(max(0.05, min(base, 0.95)))

    adjustment = 0.0
    for c in claims:
        w = max(0.1, float(c.confidence))  # confidence-weighted adjustment
        if c.verdict == "REFUTED":
            adjustment += 0.12 * w
        elif c.verdict == "SUPPORTED":
            adjustment -= 0.09 * w
            # Extra bonus when supporting evidence comes from high-credibility sources
            if c.evidence:
                avg_ev_score = sum(
                    getattr(e, "domain_score", 0) for e in c.evidence
                ) / len(c.evidence)
                if avg_ev_score >= 75:
                    adjustment -= 0.04
        # NEI verdict: no adjustment — uncertainty already captured by the base

    return float(max(0.05, min(base + adjustment, 0.95)))


def derive_input_domain_signal(
    claims: List[ClaimResult],
    domain: Optional[str],
) -> Tuple[int, str, Dict[str, Any]]:
    """Derive a transparent source-quality signal without treating it as a verdict."""
    if domain:
        credibility = score_domain_rubric(domain)
        return int(credibility.score), str(credibility.label), {
            "source": "input_domain",
            "domain": domain,
            "reasons": credibility.reasons,
        }

    evidence_scores = [
        int(item.domain_score)
        for claim in claims
        for item in claim.evidence
        if item.domain_score is not None
    ]
    if evidence_scores:
        score = round(sum(evidence_scores) / len(evidence_scores))
        return score, _label(score), {
            "source": "evidence_aggregate",
            "evidence_items": len(evidence_scores),
        }
    return 50, "MEDIUM", {"source": "neutral_default", "reason": "no input domain or evidence sources"}


def _label(score: int) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def build_dashboard_summary(runs: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    input_type_counts: Dict[str, int] = {}
    verifier_counts: Dict[str, int] = {}
    verdict_counts: Dict[str, int] = {
        "SUPPORTED": 0,
        "REFUTED": 0,
        "NEI": 0,
    }
    domain_counts: Dict[str, int] = {}
    likelihood_values: List[float] = []
    claims_analyzed = 0
    claims_requiring_human_review = 0

    for run in runs:
        input_type = str(run.get("input_type") or "unknown").lower()
        input_type_counts[input_type] = input_type_counts.get(input_type, 0) + 1

        verifier = str(run.get("verifier") or "unknown").lower()
        verifier_counts[verifier] = verifier_counts.get(verifier, 0) + 1

        domain = str(run.get("domain") or "").strip().lower()
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        response = run.get("response")
        if not isinstance(response, dict):
            continue

        like_raw = response.get("final_misinformation_likelihood")
        try:
            if like_raw is not None:
                likelihood_values.append(float(like_raw))
        except (TypeError, ValueError):
            pass

        claims = response.get("claims")
        if not isinstance(claims, list):
            continue

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claims_analyzed += 1
            if bool(claim.get("needs_human_review")):
                claims_requiring_human_review += 1

            verdict = str(claim.get("verdict") or "NEI").upper()
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    avg_likelihood = None
    if likelihood_values:
        avg_likelihood = round(sum(likelihood_values) / len(likelihood_values), 3)

    top_domains = [
        {"domain": domain, "count": count}
        for domain, count in sorted(domain_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    ]

    return {
        "limit": limit,
        "total_runs": len(runs),
        "last_run_utc": runs[0].get("created_utc") if runs else None,
        "claims_analyzed": claims_analyzed,
        "claims_requiring_human_review": claims_requiring_human_review,
        "avg_misinformation_likelihood": avg_likelihood,
        "input_type_counts": input_type_counts,
        "verifier_counts": verifier_counts,
        "verdict_counts": verdict_counts,
        "top_domains": top_domains,
    }


# ----------------------------
# Routes
# ----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/deep")
async def health_deep():
    """Deep health check including Ollama availability."""
    ollama_available = await ollama_health.is_available()
    return {
        "status": "ok",
        "ollama_available": ollama_available,
        "debate_enabled": Config.FEATURE_DEBATE_MODE,
        "config": Config.get_all(),
    }


@app.get("/evaluation/benchmark")
def evaluation_benchmark():
    return _load_json_report(
        DOCS_DIR / "evaluation_benchmark.json",
        {"claims": []},
    )


@app.get("/evaluation/baselines")
def evaluation_baselines():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "baseline_comparison_report.json",
        {"metadata": {}, "results": {}},
    )


@app.get("/evaluation/ablations")
def evaluation_ablations():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "ablation_study_report.json",
        {"metadata": {}, "variants": {}},
    )


@app.get("/evaluation/comparative")
def evaluation_comparative():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "comparative_analysis_report.json",
        {"metadata": {}, "ranking": [], "comparisons": [], "debate_lift": {}},
    )


@app.get("/evaluation/production-metrics")
def evaluation_production_metrics():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "production_metrics_report.json",
        {
            "metadata": {},
            "latency": {},
            "throughput": {},
            "cost": {},
            "quality": {},
        },
    )


@app.get("/evaluation/explainability")
def evaluation_explainability():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "explainability_demo_report.json",
        {"metadata": {}, "case_studies": []},
    )


@app.get("/evaluation/limitations")
def evaluation_limitations():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "limitations_assessment_report.json",
        {"metadata": {}, "limitations": []},
    )


@app.get("/evaluation/reproducibility")
def evaluation_reproducibility():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "reproducibility_audit_report.json",
        {
            "metadata": {},
            "summary": {"passed_checks": 0, "total_checks": 0},
            "score": {"score_percent": 0.0},
        },
    )


@app.get("/evaluation/ethics")
def evaluation_ethics():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "ethics_assessment_report.json",
        {"metadata": {}, "ethical_risks": []},
    )


@app.get("/evaluation/defense")
def evaluation_defense():
    return _load_json_report(
        BENCHMARK_RESULTS_DIR / "defense_talking_points_report.json",
        {"metadata": {}, "qa": [], "metrics_cheatsheet": []},
    )


@app.get("/runs")
def runs(limit: int = 50):
    return {"items": list_runs(limit=limit)}


@app.get("/runs/{run_id}")
def run_detail(run_id: int):
    r = get_run(run_id)
    if not r:
        return JSONResponse(status_code=404, content={"detail": "Run not found"})
    return r


@app.get("/runs-export")
def runs_export(limit: int = 500):
    items = export_runs(limit=limit)
    return {"items": items, "exported_at_utc": datetime.utcnow().isoformat() + "Z"}


@app.get("/dashboard/summary")
def dashboard_summary(limit: int = 200):
    safe_limit = max(1, min(int(limit), 1000))
    runs = export_runs(limit=safe_limit)
    return build_dashboard_summary(runs, limit=safe_limit)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    start_time = time.time()
    
    has_url = bool(req.url and req.url.strip())
    has_text = bool(req.text and req.text.strip())

    input_type: Literal["url", "text"] = "url" if has_url else "text"
    domain = extract_domain(req.url) if has_url else None
    
    log_analyze_start(input_type, domain or "(text)", req.max_claims)

    extracted_text = ""
    final_url = None

    if has_url:
        url_norm = normalize_url(req.url or "")
        final_url, html = await fetch_html(url_norm)
        if html:
            extracted_text = extract_readable_text_from_html(html, url=final_url or url_norm)

    if not extracted_text and has_text:
        extracted_text = (req.text or "").strip()

    preview = extracted_text[:400].replace("\n", " ").strip()
    chars = len(extracted_text)

    claim_texts = extract_claim_candidates(extracted_text, max_claims=req.max_claims)
    if not claim_texts and has_text:
        # Fallback for short natural-language queries (e.g. "true or false" style input)
        # so evidence search can still run on at least one candidate claim.
        fallback_claim = " ".join((req.text or "").split()).strip()
        if len(fallback_claim) >= 10 and any(ch.isalpha() for ch in fallback_claim):
            claim_texts = [fallback_claim[:280]]

    claims: List[Dict[str, Any]] = []
    debate_meta: Dict[str, Any] = {
        "enabled": req.verifier == "debate",
        "claims_debated": 0,
        "items": [],
        "memory_hits": 0,
        "semantic_retrieval": {"enabled": True, "method": "unknown"},
    }

    for ct in claim_texts:
        claim_profile = decompose_claim(ct)
        claim_key = normalize_claim_key(ct)
        memory_entry = get_claim_memory(claim_key)
        memory_hit = bool(memory_entry)

        if req.mode == "snapshot" and memory_entry and memory_entry.get("payload"):
            cached = dict(memory_entry["payload"])
            cached.setdefault("claim_profile", claim_profile)
            cached.setdefault("debate_summary", "Loaded from fact-check memory cache.")
            cached.setdefault("memory", {})
            cached["memory"].update(
                {
                    "hit": True,
                    "mode": "snapshot",
                    "updated_utc": memory_entry.get("updated_utc"),
                }
            )
            claims.append(cached)
            debate_meta["memory_hits"] += 1
            continue

        # Snapshot raw search results before any ranking so an analysis can be reproduced.
        raw_results = None
        retrieval_status = "ok"
        cache = get_cache() if Config.FEATURE_CACHING else None
        if cache:
            raw_results = cache.get(ct)
        
        if raw_results is None:
            try:
                raw_results = await serpapi_search(ct[:220], num=req.max_evidence_per_claim)
                if raw_results:
                    retrieval_status = "ok"
                else:
                    retrieval_status = "no_results" if SERPAPI_API_KEY else "search_unavailable"
            except EvidenceSearchError:
                raw_results = []
                retrieval_status = "search_failed"
            if cache and raw_results:
                cache.set(ct, raw_results)
        else:
            retrieval_status = "cached_results" if raw_results else "no_results"

        ev_items: List[Dict[str, Any]] = []
        failed_items: List[Dict[str, Any]] = []
        seen_base_domains: Dict[str, int] = {}
        for rr in raw_results:
            link = rr.get("link") or ""
            dom = domain_from_any_url(link)
            bd = base_domain(dom)

            if not dom:
                continue
            if is_blocked_domain(dom):
                continue

            seen_base_domains.setdefault(bd, 0)
            if seen_base_domains[bd] >= max(1, int(req.max_sources_per_base_domain)):
                continue
            seen_base_domains[bd] += 1

            snippet = (rr.get("snippet") or "").strip()

            cred = score_domain_rubric(dom)
            passage, resolved_url, passage_status = await fetch_evidence_passage(link, ct)
            record = {
                "url": resolved_url,
                "title": rr.get("title"),
                "snippet": snippet,
                "passage": passage or None,
                "content_hash": sha256((passage or "").encode("utf-8")).hexdigest() if passage else None,
                "retrieval_status": passage_status,
                "retrieved_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain": dom,
                "domain_score": int(cred.score),
                "overlap": int(compute_overlap(ct, passage or snippet)),
                "base_domain": bd,
            }
            if passage_status != "retrieved":
                failed_items.append(record)
                continue

            ev_items.append(record)

        reranked_items, semantic_meta = semantic_rerank(
            ct,
            ev_items,
            top_k=req.max_evidence_per_claim,
        )
        debate_meta["semantic_retrieval"] = semantic_meta

        enriched_items = [enrich_evidence(ct, claim_profile, e) for e in reranked_items]
        for item in enriched_items:
            relation, relation_metadata = classify_relation(ct, str(item.get("passage") or ""))
            item["stance"] = relation
            item["relation_classifier"] = relation_metadata
        evidence_clusters = cluster_evidence(enriched_items)
        enriched_items.sort(
            key=lambda e: (float(e.get("quality_score") or 0.0), int(e.get("domain_score") or 0), int(e.get("overlap") or 0)),
            reverse=True,
        )

        ev_for_baseline = [
            EvidenceItem(**{k: v for k, v in e.items() if k in _get_model_fields(EvidenceItem)})
            for e in enriched_items
        ]
        baseline_verdict_value, baseline_conf, baseline_summary = baseline_verdict(ct, ev_for_baseline)
        structured = determine_verdict(ct, claim_profile, enriched_items)
        reflective_report = reflective_analysis(ct, claim_profile, enriched_items)
        evidence_graph = build_evidence_graph(
            ct,
            list(claim_profile.get("atomic_claims") or [ct]),
            [*enriched_items, *failed_items],
            retrieval_status,
        )
        evidence_graph["independence_clusters"] = evidence_clusters
        graph_audit = audit_evidence_graph(evidence_graph, [*enriched_items, *failed_items], claim_profile)
        retrieval_manifest = build_retrieval_manifest(
            claim=ct,
            query=ct[:220],
            retrieval_status=retrieval_status,
            raw_results=list(raw_results or []),
            evidence=[*enriched_items, *failed_items],
        )
        graph_verdict, graph_reasons = adjudicate_graph(evidence_graph)
        if graph_audit["decision"] == "NEI":
            graph_verdict = "NEI"
            graph_reasons = list(graph_audit["violations"])
        elif graph_audit["decision"] == "CONFLICTING":
            graph_verdict = "CONFLICTING"
            graph_reasons = list(graph_audit["violations"])
        elif graph_audit["decision"] == "HUMAN_REVIEW":
            structured["needs_human_review"] = True
            structured["human_review_reason"] = graph_audit["violations"][0]
            uncertainty = list(structured.get("uncertainty_reasons") or [])
            for violation in graph_audit["violations"]:
                if violation not in uncertainty:
                    uncertainty.append(violation)
            structured["uncertainty_reasons"] = uncertainty[:4]
        if graph_verdict:
            structured["legacy_verdict"] = graph_verdict
            structured["structured_verdict"] = (
                "Mixed / disputed" if graph_verdict == "CONFLICTING" else "Insufficient evidence"
            )
            structured["confidence"] = 0.5
            structured["uncertainty_reasons"] = graph_reasons
            structured["needs_human_review"] = True
            structured["human_review_reason"] = graph_reasons[0]
            structured["explanation"] = f"Graph adjudication: {graph_reasons[0]}"

        if (
            req.enable_reflective_abstention
            and reflective_report.get("decision") == "TERMINATE"
            and structured.get("legacy_verdict") != "CONFLICTING"
        ):
            abstain_reason = str(
                reflective_report.get("abstain_reason")
                or "Reflective factor analysis requested abstention."
            )
            structured["legacy_verdict"] = "NEI"
            structured["structured_verdict"] = "Insufficient evidence"
            structured["confidence"] = round(
                min(float(structured.get("confidence") or baseline_conf), 0.55),
                2,
            )
            uncertainty = list(structured.get("uncertainty_reasons") or [])
            if abstain_reason not in uncertainty:
                uncertainty.insert(0, abstain_reason)
            structured["uncertainty_reasons"] = uncertainty[:4]
            structured["needs_human_review"] = True
            structured["human_review_reason"] = abstain_reason
            structured["explanation"] = f"Reflective abstention: {abstain_reason}"

        verdict = structured["legacy_verdict"]
        conf = structured["confidence"]
        summary = structured["explanation"]
        if baseline_verdict_value != verdict:
            summary = summary + f" | baseline={baseline_verdict_value} ({baseline_conf:.2f})"
        if memory_hit:
            summary = summary + " | prior run found in claim memory; result refreshed with current evidence."
            debate_meta["memory_hits"] += 1

        adjusted_verdict = None
        adjusted_conf = None
        evidence_summary = dict(structured["evidence_summary"])

        if req.enable_weighted_confidence:
            filtered = [
                e for e in enriched_items
                if int(e["domain_score"]) >= int(req.min_source_score) and int(e["overlap"]) >= int(req.min_overlap)
            ]
            strong_domains = sorted({e["base_domain"] for e in filtered})
            distinct_domains = len(strong_domains)

            reason = "ok"
            if not filtered:
                reason = "no_evidence_after_filters"
            if req.require_independent_domains and distinct_domains < 2:
                reason = "insufficient_independent_domains"

            ev_for_adjusted = [
                EvidenceItem(**{k: v for k, v in e.items() if k in _get_model_fields(EvidenceItem)})
                for e in filtered
            ]
            av, ac, atext = baseline_verdict(ct, ev_for_adjusted)

            if req.require_independent_domains and distinct_domains < 2:
                av = "NEI"
                ac = max(0.50, float(ac))
                atext = atext + " | independence constraint triggered"

            adjusted_verdict = av
            adjusted_conf = round(float(ac), 2)
            evidence_summary.update(
                {
                    "min_source_score": int(req.min_source_score),
                    "min_overlap": int(req.min_overlap),
                    "require_independent_domains": bool(req.require_independent_domains),
                    "distinct_base_domains": distinct_domains,
                    "strong_base_domains": strong_domains,
                    "filter_reason": reason,
                    "filter_note": atext,
                }
            )

        if req.verifier == "debate":
            summary = summary + " | Debate mode currently uses enriched counter-evidence scoring before optional LLM debate."
        
        # Wire the professional AI final judge if enabled
        debate_result_raw = None
        ollama_available = False
        if (
            req.verifier == "debate"
            and Config.FEATURE_DEBATE_MODE
            and enriched_items
            and (not req.enable_reflective_abstention or reflective_report.get("decision") != "TERMINATE")
        ):
            try:
                log_debate_started(ct)
                ollama_available = await ollama_health.is_available()
                
                if ollama_available:
                    # Try the AI final judge first; fall back to the older debate stack if needed.
                    ai_timeout = max(10, min(int(Config.OLLAMA_TIMEOUT), 30))
                    debate_task = asyncio.create_task(
                        llm_final_judge(
                            ct,
                            enriched_items,
                            verdict,
                            conf,
                            structured["structured_verdict"],
                            claim_profile,
                        )
                    )
                    try:
                        debate_result_raw = await asyncio.wait_for(
                            debate_task, 
                            timeout=ai_timeout
                        )
                    except asyncio.TimeoutError:
                        log_debate_error(ct, Exception("Final judge timeout, using baseline verdict"))
                        debate_result_raw = None
            except Exception as e:
                log_debate_error(ct, e)
                debate_result_raw = None
        
        # Apply debate result if available
        if debate_result_raw:
            debate_verdict, debate_conf, debate_msg, debate_debug = debate_result_raw
            verdict = debate_verdict
            conf = debate_conf
            summary = f"AI Final: {debate_msg} | {summary}"
            debate_meta["claims_debated"] += 1
            debate_meta["items"].append({
                "claim": ct[:100],
                "debate_verdict": debate_verdict,
                "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                "debug": debate_debug,
            })

        faithful_correction = None
        if req.enable_faithful_correction:
            correction_reflective = (
                reflective_report
                if req.enable_reflective_abstention
                else {"decision": "PROCEED"}
            )
            faithful_correction = generate_faithful_correction(
                ct,
                claim_profile,
                enriched_items,
                correction_reflective,
                verdict,
            )

        # Perform sentiment analysis on the claim
        sentiment_result = analyze_sentiment(ct)
        bias_risk = estimate_bias_risk(
            sentiment_result.label,
            sentiment_result.emotional_intensity,
            sentiment_result.flags
        )

        claim_output = {
            "claim_text": ct,
            "verdict": verdict,
            "confidence": round(float(conf), 2),
            "structured_verdict": structured["structured_verdict"],
            "uncertainty_reasons": structured["uncertainty_reasons"],
            "needs_human_review": structured["needs_human_review"],
            "human_review_reason": structured["human_review_reason"],
            "claim_profile": claim_profile,
            "evidence": [{k: v for k, v in e.items() if k != "base_domain"} for e in [*enriched_items, *failed_items]],
            "evidence_graph": evidence_graph,
            "graph_audit": graph_audit,
            "retrieval_manifest": retrieval_manifest,
            "debate_summary": summary,
            "debate_used": req.verifier == "debate",
            "debate_available": ollama_available,
            "adjusted_verdict": adjusted_verdict,
            "adjusted_confidence": adjusted_conf,
            "evidence_summary": evidence_summary,
            "reflective": reflective_report,
            "faithful_correction": faithful_correction,
            "sentiment": {
                "score": sentiment_result.score,
                "label": sentiment_result.label,
                "emotional_intensity": sentiment_result.emotional_intensity,
                "bias_risk": bias_risk,
                "manipulation_flags": sentiment_result.flags,
            },
            "memory": {
                "hit": memory_hit,
                "mode": "live" if req.mode == "live" else "snapshot-refresh",
                "updated_utc": memory_entry.get("updated_utc") if memory_entry else None,
            },
        }
        claims.append(claim_output)
        save_claim_memory(claim_key, claim_output)

    # input domain score
    if domain:
        input_cred = score_domain_rubric(domain)
        input_domain_score = int(input_cred.score)
        input_domain_label = str(input_cred.label)
        input_domain_reasons = input_cred.reasons
    else:
        input_domain_score = 50  # neutral prior — no domain to judge
        input_domain_label = "MEDIUM"
        input_domain_reasons = {}

    final_like = estimate_misinformation_likelihood(
        [ClaimResult(**c) for c in claims],
        input_domain_score=input_domain_score,
    )

    # Apply sentiment-based adjustment to misinformation likelihood
    sentiment_adjustments = []
    for claim in claims:
        if "sentiment" in claim:
            sent = claim["sentiment"]
            # Create SentimentResult-like object for adjustment calculation
            from app.sentiment import SentimentResult
            sent_result = SentimentResult(
                score=sent["score"],
                label=sent["label"],
                emotional_intensity=sent["emotional_intensity"],
                flags=sent.get("manipulation_flags", [])
            )
            adjustment = calculate_sentiment_misinformation_adjustment(sent_result)
            sentiment_adjustments.append(adjustment)
    
    if sentiment_adjustments:
        avg_sentiment_adjustment = sum(sentiment_adjustments) / len(sentiment_adjustments)
        # Adjust final misinformation likelihood (max +0.4, min -0.05)
        final_like = min(1.0, max(0.0, final_like + avg_sentiment_adjustment))
        debate_meta["sentiment_adjustment_applied"] = round(avg_sentiment_adjustment, 3)

    response_dict: Dict[str, Any] = {
        "input_type": input_type,
        "domain": domain,
        "extracted_text_chars": chars,
        "extracted_text_preview": preview,
        "domain_score": input_domain_score,
        "domain_label": input_domain_label,
        "final_misinformation_likelihood": round(float(final_like), 2),
        "claims": claims,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "metadata": {
            "mode": req.mode,
            "final_url": final_url,
            "extraction_success": bool(extracted_text) and chars > 0,
            "claims_extracted": len(claims),
            "serpapi_enabled": bool(SERPAPI_API_KEY),
            "domain_score_reasons": input_domain_reasons,
            "verifier": req.verifier,
            "debate": debate_meta,
            "claim_memory_enabled": True,
            "reflective_abstention_enabled": req.enable_reflective_abstention,
            "faithful_correction_enabled": req.enable_faithful_correction,
            "retrieval": {
                "manifest_version": MANIFEST_VERSION,
                "query_limit_chars": 220,
                "evidence_basis": "fetched_passages_only",
                "search_provider": "SerpAPI",
                "raw_results_frozen_in_claim_snapshots": True,
            },
            "trust_features": [
                "claim_decomposition",
                "evidence_quality_ranking",
                "primary_source_detection",
                "recency_scoring",
                "reflective_factor_analyst",
                "abstention_gate",
                "structured_verdicts",
                "faithful_correction_candidates",
                "counter_evidence_requirement",
                "quote_grounding",
                "uncertainty_explanations",
                "entity_and_number_checks",
                "anti_manipulation_flags",
                "expertise_profiles",
                "human_review_mode",
                "atomic_claim_candidates",
                "full_passage_provenance",
                "typed_evidence_graph",
                "source_independence_clusters",
                "deterministic_graph_auditor",
                "conflict_aware_abstention",
                "sentiment_analysis",
                "emotional_bias_detection",
            ],
            "benchmark_endpoint": "/evaluation/benchmark",
            "note": "Runs are saved to SQLite; list at GET /runs and export at GET /runs-export.",
        },
    }

    run_id = save_run(
        input_type=input_type,
        url=req.url,
        text=req.text,
        domain=domain,
        mode=req.mode,
        verifier=req.verifier,
        response=response_dict,
    )
    response_dict["metadata"]["run_id"] = run_id
    
    # Log completion
    duration_ms = (time.time() - start_time) * 1000
    log_analyze_complete(run_id, len(claims), response_dict["final_misinformation_likelihood"], duration_ms)

    return response_dict
