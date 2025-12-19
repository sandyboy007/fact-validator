from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse

import httpx
import trafilatura

app = FastAPI(title="Fact Validator API", version="0.2.0")

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

    # NEW: show live extraction results
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
    """
    Returns (final_url, html) or ("", "") if failed.
    """
    try:
        headers = {
            "User-Agent": "FactValidatorBot/0.2 (thesis demo; contact: none)",
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
    """
    Uses trafilatura to extract main article text.
    Returns "" if extraction fails.
    """
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    has_url = bool(req.url and req.url.strip())
    has_text = bool(req.text and req.text.strip())

    input_type: Literal["url", "text"] = "url" if has_url else "text"

    domain = extract_domain(req.url) if has_url else None

    # --- NEW: live URL -> text extraction ---
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

    # Mock claim output for now (next steps will replace this with real claim extraction)
    mock_claim = ClaimResult(
        claim_text="Example claim extracted from the input.",
        verdict="NEI",
        confidence=0.55,
        evidence=[
            EvidenceItem(
                url="https://example.com",
                snippet="Example evidence snippet. (Mock)",
                domain="example.com",
                domain_score=70,
            )
        ],
        debate_summary="Prover: insufficient proof. Skeptic: no reliable sources. Judge: NEI.",
    )

    return AnalyzeResponse(
        input_type=input_type,
        domain=domain,
        extracted_text_chars=chars,
        extracted_text_preview=preview,
        domain_score=70,
        domain_label="MEDIUM",
        final_misinformation_likelihood=0.42,
        claims=[mock_claim],
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        metadata={
            "mode": req.mode,
            "note": "mock response (now includes live url->text extraction)",
            "input_url": req.url if has_url else None,
            "final_url": final_url,
            "has_text": has_text,
            "extraction_success": bool(extracted_text) and chars > 0,
        },
    )
