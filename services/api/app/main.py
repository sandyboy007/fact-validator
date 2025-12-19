from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse
import os
import re

import httpx
import trafilatura
import nltk
from nltk.tokenize import sent_tokenize

from dotenv import load_dotenv
import tldextract

# --- Step 5 DB ---
from sqlmodel import select
from app.db.database import init_db, get_session
from app.db.models import Run

# Load env from services/api/.env
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()

app = FastAPI(title="Fact Validator API", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    init_db()


class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    mode: Literal["live", "snapshot"] = "live"
    max_claims: int = 6
    max_evidence_per_claim: int = 5


class EvidenceItem(BaseModel):
    url: str
    title: Optional[str] = None
    snippet: str
    domain: str
    domain_score: int


class ClaimResult(BaseModel):
    claim_text: str
    verdict: Literal["SUPPORTED", "REFUTED", "NEI"]
    confidence: float
    evidence: List[EvidenceItem]
    debate_summary: Optional[str] = None


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
        return domain or ""
    return f"{ext.domain}.{ext.suffix}"


def score_domain(domain: str) -> int:
    d = (domain or "").lower().strip()
    bd = base_domain(d)

    high_suffix = (bd.endswith(".gov") or bd.endswith(".edu"))
    high_known = bd in {
        "who.int",
        "nih.gov",
        "cdc.gov",
        "nasa.gov",
        "europa.eu",
        "un.org",
        "oecd.org",
        "worldbank.org",
        "wikipedia.org",
        "britannica.com",
        "ourworldindata.org",
        "epa.gov",
    }

    low_markers = [
        "blogspot.", "wordpress.", "medium.com", "substack.com",
        "rumor", "hoax", "clickbait", "conspiracy"
    ]

    if high_suffix or high_known:
        return 90
    if any(m in d for m in low_markers):
        return 35
    return 65


def label_from_score(score: int) -> Literal["HIGH", "MEDIUM", "LOW"]:
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def is_blocked_domain(domain: str) -> bool:
    bd = base_domain((domain or "").lower())
    blocked = {
        "facebook.com", "x.com", "twitter.com", "tiktok.com",
        "instagram.com", "reddit.com", "pinterest.com",
        "worldarticledatabase.com", "wecanfigurethisout.org",
    }
    return bd in blocked


async def fetch_html(url: str) -> Tuple[str, str]:
    try:
        headers = {
            "User-Agent": "FactValidatorBot/0.5.0 (thesis demo)",
            "Accept": "text/html,application/xhtml+xml",
        }
        timeout = httpx.Timeout(20.0, connect=10.0)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
            r = await client.get(url)
            if r.status_code >= 400:
                return "", ""
            return str(r.url), r.text
    except Exception:
        return "", ""


def extract_readable_text_from_html(html: str, url: str = "") -> str:
    try:
        downloaded = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
        return (downloaded or "").strip()
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
    t = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
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
            if len(s2) < 60 or len(s2) > 280:
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
        timeout = httpx.Timeout(25.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
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
    except Exception:
        return []


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


def baseline_verdict(claim: str, evidence: List[EvidenceItem]) -> Tuple[Literal["SUPPORTED", "REFUTED", "NEI"], float, str]:
    if not evidence:
        return "NEI", 0.55, "No evidence retrieved."

    claim_toks = set(tokenize_for_overlap(claim))
    if not claim_toks:
        return "NEI", 0.55, "Claim tokenization empty."

    neg_cues = ["false", "hoax", "debunk", "misleading", "not true", "no evidence", "incorrect"]
    overlaps = []

    for e in evidence:
        ev_toks = set(tokenize_for_overlap(e.snippet))
        overlap = len(claim_toks.intersection(ev_toks))
        overlaps.append((overlap, e.domain_score, base_domain(e.domain), (e.snippet or "").lower()))

    for ov, ds, bd, snip_low in overlaps:
        if ds >= 65 and any(cue in snip_low for cue in neg_cues):
            return "REFUTED", 0.70, "Evidence snippet contains refutation cue from a medium/high source."

    strong_support_domains = {bd for (ov, ds, bd, _) in overlaps if ds >= 65 and ov >= 6}
    if len(strong_support_domains) >= 2:
        return "SUPPORTED", 0.75, "At least two distinct medium/high domains have strong keyword overlap."

    best = max(overlaps, key=lambda x: (x[0], x[1]))
    if best[1] >= 90 and best[0] >= 7:
        return "SUPPORTED", 0.72, "Single high-score domain has strong keyword overlap."

    return "NEI", 0.56, "Evidence retrieved but insufficient strength for a decision (baseline)."


def estimate_misinformation_likelihood(claims: List[ClaimResult]) -> float:
    if not claims:
        return 0.5
    score = 0.5
    for c in claims:
        if c.verdict == "REFUTED":
            score += 0.10
        elif c.verdict == "SUPPORTED":
            score -= 0.07
    return float(max(0.05, min(score, 0.95)))


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Step 5 endpoints: history ---
@app.get("/runs")
def list_runs(limit: int = 20):
    with get_session() as session:
        rows = session.exec(select(Run).order_by(Run.id.desc()).limit(limit)).all()
        return [
            {
                "id": r.id,
                "created_utc": r.created_utc,
                "input_type": r.input_type,
                "input_url": r.input_url,
                "input_domain": r.input_domain,
                "extracted_text_chars": r.extracted_text_chars,
            }
            for r in rows
        ]


@app.get("/runs/{run_id}")
def get_run(run_id: int):
    with get_session() as session:
        r = session.get(Run, run_id)
        if not r:
            raise HTTPException(status_code=404, detail="Run not found")
        return r.get_result()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    has_url = bool(req.url and req.url.strip())
    has_text = bool(req.text and req.text.strip())

    input_type: Literal["url", "text"] = "url" if has_url else "text"
    domain = extract_domain(req.url) if has_url else None

    extracted_text = ""
    final_url = None

    if has_url:
        url_norm = normalize_url(req.url)
        final_url, html = await fetch_html(url_norm)
        if html:
            extracted_text = extract_readable_text_from_html(html, url=final_url or url_norm)

    if not extracted_text and has_text:
        extracted_text = req.text.strip()

    preview = extracted_text[:400].replace("\n", " ").strip()
    chars = len(extracted_text)

    claim_texts = extract_claim_candidates(extracted_text, max_claims=req.max_claims)

    claims: List[ClaimResult] = []
    for ct in claim_texts:
        query = ct[:220]
        raw_results = await serpapi_search(query, num=req.max_evidence_per_claim)

        ev_items: List[EvidenceItem] = []
        seen_domains = set()

        for rr in raw_results:
            link = rr.get("link") or ""
            dom = domain_from_any_url(link)
            bd = base_domain(dom)

            if not dom:
                continue
            if is_blocked_domain(dom):
                continue
            if bd in seen_domains:
                continue
            seen_domains.add(bd)

            snippet = (rr.get("snippet") or "").strip()
            if len(snippet) < 40:
                continue

            dscore = score_domain(dom)

            ev_items.append(
                EvidenceItem(
                    url=link,
                    title=rr.get("title"),
                    snippet=snippet,
                    domain=dom,
                    domain_score=dscore,
                )
            )

        verdict, conf, summary = baseline_verdict(ct, ev_items)

        claims.append(
            ClaimResult(
                claim_text=ct,
                verdict=verdict,
                confidence=round(conf, 2),
                evidence=ev_items,
                debate_summary=summary if SERPAPI_API_KEY else "SERPAPI_API_KEY not set. Evidence retrieval disabled.",
            )
        )

    input_domain_score = score_domain(domain or "") if domain else 65
    input_domain_label = label_from_score(input_domain_score)

    final_like = estimate_misinformation_likelihood(claims)

    response = AnalyzeResponse(
        input_type=input_type,
        domain=domain,
        extracted_text_chars=chars,
        extracted_text_preview=preview,
        domain_score=input_domain_score,
        domain_label=input_domain_label,
        final_misinformation_likelihood=round(final_like, 2),
        claims=claims,
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        metadata={
            "mode": req.mode,
            "final_url": final_url,
            "extraction_success": bool(extracted_text) and chars > 0,
            "claims_extracted": len(claims),
            "serpapi_enabled": bool(SERPAPI_API_KEY),
            "note": "Step 5 enabled: snapshot saves runs; /runs and /runs/{id} endpoints added",
        },
    )

    # --- Snapshot saving ---
    if req.mode == "snapshot":
        with get_session() as session:
            run = Run(
                input_type=input_type,
                input_url=req.url if has_url else None,
                input_domain=domain,
                extracted_text_chars=chars,
                extracted_text_preview=preview,
                result_json="{}",
            )
            run.set_result(response.dict())
            session.add(run)
            session.commit()
            session.refresh(run)
            response.metadata["run_id"] = run.id

    return response
