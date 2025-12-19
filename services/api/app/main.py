from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse
import os
import re
import json

import httpx
import trafilatura
import nltk
from nltk.tokenize import sent_tokenize

from dotenv import load_dotenv
import tldextract

# OpenAI (optional)
from openai import OpenAI

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-pro").strip()  # :contentReference[oaicite:2]{index=2}

openai_client = OpenAI() if OPENAI_API_KEY else None  # SDK reads OPENAI_API_KEY from env :contentReference[oaicite:3]{index=3}

app = FastAPI(title="Fact Validator API", version="0.5.0")

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
    max_claims: int = 4
    max_evidence_per_claim: int = 4
    use_llm_debate: bool = True  # turn off if you want zero LLM cost


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
    }
    low_markers = ["blogspot.", "wordpress.", "medium.com", "substack.com", "rumor", "hoax", "clickbait", "conspiracy"]

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


async def fetch_html(url: str) -> Tuple[str, str]:
    try:
        headers = {"User-Agent": "FactValidatorBot/0.5 (thesis demo)", "Accept": "text/html,application/xhtml+xml"}
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


def extract_claim_candidates(text: str, max_claims: int = 4) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = sent_tokenize(text)

    cleaned: List[str] = []
    seen = set()

    for s in sents:
        s2 = " ".join(s.split()).strip()
        if len(s2) < 50 or len(s2) > 350:
            continue
        key = s2.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s2)
        if len(cleaned) >= 120:
            break

    ranked = sorted(cleaned, key=heuristic_claim_score, reverse=True)
    return ranked[:max_claims]


async def serpapi_search(query: str, num: int = 4) -> List[Dict[str, Any]]:
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


def baseline_verdict(claim: str, evidence: List[EvidenceItem]) -> Tuple[Literal["SUPPORTED", "REFUTED", "NEI"], float, str]:
    if not evidence:
        return "NEI", 0.55, "No evidence retrieved."
    # conservative default
    return "NEI", 0.56, "Evidence retrieved but baseline verifier is conservative (LLM debate recommended)."


def safe_json_parse(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        # try to extract first {...}
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def run_llm_debate(claim: str, evidence: List[EvidenceItem]) -> Tuple[Literal["SUPPORTED", "REFUTED", "NEI"], float, str]:
    """
    Prover vs Skeptic vs Judge.
    Judge returns strict JSON: {"verdict": "...", "confidence": 0.xx, "summary": "..."}.

    Uses OpenAI Responses API via Python SDK. :contentReference[oaicite:4]{index=4}
    """
    if not openai_client:
        return baseline_verdict(claim, evidence)

    # Limit evidence for cost/stability: take top by domain_score
    ev_sorted = sorted(evidence, key=lambda e: e.domain_score, reverse=True)[:4]
    ev_block = "\n".join(
        [f"[E{i+1}] ({e.domain_score}) {e.domain} | {e.title or ''}\nURL: {e.url}\nSnippet: {e.snippet}"
         for i, e in enumerate(ev_sorted)]
    )

    prover_prompt = f"""
You are the Prover. Your goal: argue the claim is SUPPORTED using only the evidence snippets.
If evidence is insufficient, say so.

Claim:
{claim}

Evidence:
{ev_block}

Output: 4-7 bullets, and cite evidence like [E1], [E2].
""".strip()

    skeptic_prompt = f"""
You are the Skeptic. Your goal: argue the claim is REFUTED or at least NOT PROVEN (NEI),
pointing out contradictions, weak sources, ambiguity, or missing context. Use only the evidence snippets.

Claim:
{claim}

Evidence:
{ev_block}

Output: 4-7 bullets, and cite evidence like [E1], [E2].
""".strip()

    prover = openai_client.responses.create(model=OPENAI_MODEL, input=prover_prompt, max_output_tokens=350).output_text
    skeptic = openai_client.responses.create(model=OPENAI_MODEL, input=skeptic_prompt, max_output_tokens=350).output_text

    judge_prompt = f"""
You are the Judge. Decide whether the claim is SUPPORTED, REFUTED, or NEI based only on the evidence.
Be conservative: if evidence does not clearly support/refute, choose NEI.

Claim:
{claim}

Evidence:
{ev_block}

Prover argument:
{prover}

Skeptic argument:
{skeptic}

Return STRICT JSON ONLY with keys:
verdict: one of "SUPPORTED","REFUTED","NEI"
confidence: number 0.0-1.0
summary: short explanation (1-3 sentences) with citations like [E1]
""".strip()

    judge = openai_client.responses.create(model=OPENAI_MODEL, input=judge_prompt, max_output_tokens=250).output_text
    obj = safe_json_parse(judge) or {}

    verdict = obj.get("verdict", "NEI")
    if verdict not in ["SUPPORTED", "REFUTED", "NEI"]:
        verdict = "NEI"

    try:
        conf = float(obj.get("confidence", 0.60))
    except Exception:
        conf = 0.60
    conf = max(0.05, min(conf, 0.95))

    summary = obj.get("summary") or "LLM judge returned no summary."
    return verdict, conf, summary


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
        raw_results = await serpapi_search(ct, num=req.max_evidence_per_claim)

        ev_items: List[EvidenceItem] = []
        for rr in raw_results:
            link = rr.get("link") or ""
            dom = extract_domain(link) or ""
            dscore = score_domain(dom)
            ev_items.append(
                EvidenceItem(
                    url=link,
                    title=rr.get("title"),
                    snippet=(rr.get("snippet") or "").strip(),
                    domain=dom,
                    domain_score=dscore,
                )
            )

        # Step 4.4: Debate (LLM) with fallback
        if req.use_llm_debate and openai_client:
            verdict, conf, summary = run_llm_debate(ct, ev_items)
        else:
            verdict, conf, summary = baseline_verdict(ct, ev_items)

        claims.append(
            ClaimResult(
                claim_text=ct,
                verdict=verdict,
                confidence=round(conf, 2),
                evidence=ev_items,
                debate_summary=summary,
            )
        )

    input_domain_score = score_domain(domain or "") if domain else 65
    input_domain_label = label_from_score(input_domain_score)

    final_like = estimate_misinformation_likelihood(claims)

    return AnalyzeResponse(
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
            "serpapi_enabled": bool(SERPAPI_API_KEY),
            "openai_enabled": bool(openai_client),
            "openai_model": OPENAI_MODEL,
            "note": "Step 4.4 implemented: Prover vs Skeptic vs Judge debate (fallback to baseline if no OPENAI_API_KEY).",
        },
    )
