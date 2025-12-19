from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse

import httpx
import trafilatura
import nltk
from nltk.tokenize import sent_tokenize

app = FastAPI(title="Fact Validator API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    mode: Literal["live", "snapshot"] = "live"


class EvidenceItem(BaseModel):
    url: str
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


async def fetch_html(url: str) -> Tuple[str, str]:
    try:
        headers = {
            "User-Agent": "FactValidatorBot/0.3 (thesis demo; contact: none)",
            "Accept": "text/html,application/xhtml+xml",
        }
        timeout = httpx.Timeout(15.0, connect=10.0)
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
    """
    Simple heuristic for ranking claim candidates.
    Not a model — just a baseline that we will replace later.
    """
    s = s.strip()
    if not s:
        return 0.0

    length = len(s)
    # Prefer mid-length sentences for claim-likeness
    length_score = 1.0 - min(abs(length - 160) / 160.0, 1.0)

    has_number = any(ch.isdigit() for ch in s)
    number_bonus = 0.15 if has_number else 0.0

    has_entity_hint = any(tok in s.lower() for tok in [" is ", " are ", " was ", " were ", " has ", " have ", " with "])
    entity_bonus = 0.10 if has_entity_hint else 0.0

    score = 0.55 * length_score + number_bonus + entity_bonus
    return max(0.05, min(score, 0.95))


def extract_claim_candidates(text: str, max_claims: int = 6) -> List[str]:
    """
    Extract sentences as claim candidates with simple filtering.
    """
    text = (text or "").strip()
    if not text:
        return []

    # NLTK sentence split
    sents = sent_tokenize(text)

    cleaned: List[str] = []
    seen = set()

    for s in sents:
        s2 = " ".join(s.split()).strip()
        if len(s2) < 50:
            continue
        if len(s2) > 350:
            continue
        key = s2.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s2)

        if len(cleaned) >= 60:
            break  # cap processing

    # Rank by heuristic score
    ranked = sorted(cleaned, key=heuristic_claim_score, reverse=True)
    return ranked[:max_claims]


@app.get("/health")
def health():
    return {"status": "ok"}


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

    # NEW: real claim candidates
    claim_texts = extract_claim_candidates(extracted_text, max_claims=6)

    # For now: verdict is NEI for all claims (evidence retrieval comes next)
    claims: List[ClaimResult] = []
    for ct in claim_texts:
        claims.append(
            ClaimResult(
                claim_text=ct,
                verdict="NEI",
                confidence=round(heuristic_claim_score(ct), 2),
                evidence=[],
                debate_summary="Claim extracted from article text. Evidence retrieval not enabled yet.",
            )
        )

    return AnalyzeResponse(
        input_type=input_type,
        domain=domain,
        extracted_text_chars=chars,
        extracted_text_preview=preview,
        domain_score=70,
        domain_label="MEDIUM",
        final_misinformation_likelihood=0.42,
        claims=claims,
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        metadata={
            "mode": req.mode,
            "final_url": final_url,
            "extraction_success": bool(extracted_text) and chars > 0,
            "claims_extracted": len(claims),
            "note": "Step 3.3: sentence-based claim extraction baseline",
        },
    )
