from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime

app = FastAPI(title="Fact Validator API", version="0.1.0")

# Allow your website (Next.js) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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
    domain_score: int
    domain_label: Literal["HIGH", "MEDIUM", "LOW"]
    final_misinformation_likelihood: float
    claims: List[ClaimResult]
    timestamp_utc: str
    metadata: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    input_type = "url" if (req.url and req.url.strip()) else "text"

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
        domain="example.com" if input_type == "url" else None,
        domain_score=70,
        domain_label="MEDIUM",
        final_misinformation_likelihood=0.42,
        claims=[mock_claim],
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        metadata={"mode": req.mode, "note": "mock response"},
    )
