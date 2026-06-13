from __future__ import annotations

from typing import List, Dict, Any, Literal, Tuple
import os
import json
import httpx

Verdict = Literal["SUPPORTED", "REFUTED", "NEI"]


def _env_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()


def _env_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()


def _compact_evidence(evidence: List[Dict[str, Any]], max_items: int = 2) -> List[Dict[str, Any]]:
    """
    Keep evidence small and deterministic to reduce Ollama latency.
    The caller may pass many evidence items, but we only keep top-N.
    """
    out: List[Dict[str, Any]] = []
    for e in evidence[:max_items]:
        out.append(
            {
                "domain": e.get("domain"),
                "domain_score": e.get("domain_score"),
                "snippet": (e.get("snippet") or "")[:450],
                "url": e.get("url"),
            }
        )
    return out


def _normalize_verdict(value: Any) -> Verdict:
    verdict = str(value or "NEI").strip().upper()
    if verdict not in ("SUPPORTED", "REFUTED", "NEI"):
        return "NEI"
    return verdict  # type: ignore[return-value]


def _compact_final_context(
    evidence: List[Dict[str, Any]],
    max_items: int = 4,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in evidence[:max_items]:
        out.append(
            {
                "domain": e.get("domain"),
                "domain_score": e.get("domain_score"),
                "stance": e.get("stance"),
                "quality_score": e.get("quality_score"),
                "primary_source": e.get("primary_source"),
                "snippet": (e.get("snippet") or "")[:320],
                "url": e.get("url"),
            }
        )
    return out


def _safe_json_parse(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass

    # If the model wrapped JSON with extra text, salvage the first {...} block.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return {}
    return {}


async def _ollama_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    num_ctx: int = 1536,
    num_predict: int = 220,
) -> str:
    """
    Calls Ollama /api/chat with conservative settings to avoid timeouts.
    - trust_env=False avoids Windows proxy env vars breaking localhost calls
    - long read timeout accommodates slower machines
    - num_predict limits output length (major speed-up)
    """
    base_url = _env_ollama_base_url()
    model = _env_ollama_model()

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    timeout = httpx.Timeout(connect=30.0, read=900.0, write=120.0, pool=120.0)

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        r = await client.post(f"{base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "") or ""


async def llm_debate_verdict(
    claim_text: str,
    evidence_items: List[Dict[str, Any]],
) -> Tuple[Verdict, float, str, Dict[str, Any]]:
    """
    Prover vs Skeptic vs Judge debate.
    Returns:
      verdict, confidence, short_summary, debug_trace
    """
    compact = _compact_evidence(evidence_items, max_items=2)

    system = (
        "You are part of a fact-checking debate system.\n"
        "You must ONLY use the provided evidence snippets.\n"
        "If evidence is insufficient or ambiguous, choose NEI.\n"
        "Do NOT invent sources.\n"
        "Be concise.\n"
    )

    prover = await _ollama_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Role: PROVER\n"
                    "Goal: Argue the claim is SUPPORTED using ONLY evidence.\n"
                    "Output 3-5 concise bullet points referencing evidence by (domain, domain_score).\n"
                    "If you cannot support, write: cannot support.\n\n"
                    f"CLAIM:\n{claim_text}\n\n"
                    f"EVIDENCE(JSON):\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
                ),
            },
        ],
        temperature=0.2,
        num_ctx=1536,
        num_predict=220,
    )

    skeptic = await _ollama_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Role: SKEPTIC\n"
                    "Goal: Argue the claim is REFUTED or at least NOT proven using ONLY evidence.\n"
                    "Output 3-5 concise bullet points referencing evidence by (domain, domain_score).\n"
                    "Point out missing support, ambiguity, or weak sources.\n\n"
                    f"CLAIM:\n{claim_text}\n\n"
                    f"EVIDENCE(JSON):\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
                ),
            },
        ],
        temperature=0.2,
        num_ctx=1536,
        num_predict=220,
    )

    judge_prompt = (
        "Role: JUDGE\n"
        "Decide the verdict based ONLY on evidence.\n"
        "Return STRICT JSON ONLY, no markdown, no extra text.\n"
        "Schema:\n"
        "{\n"
        '  "verdict": "SUPPORTED|REFUTED|NEI",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "summary": "one short sentence justification",\n'
        '  "key_domains": ["domain1","domain2"]\n'
        "}\n\n"
        f"CLAIM:\n{claim_text}\n\n"
        f"PROVER_ARGUMENT:\n{prover}\n\n"
        f"SKEPTIC_ARGUMENT:\n{skeptic}\n\n"
        f"EVIDENCE(JSON):\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
    )

    judge_raw = await _ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": judge_prompt}],
        temperature=0.1,
        num_ctx=1536,
        num_predict=180,
    )

    j = _safe_json_parse(judge_raw)

    verdict: Verdict = j.get("verdict", "NEI")
    confidence = j.get("confidence", 0.55)
    summary = j.get("summary", "Insufficient evidence to decide.")
    key_domains = j.get("key_domains", [])

    if verdict not in ("SUPPORTED", "REFUTED", "NEI"):
        verdict = "NEI"

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.55

    confidence = max(0.05, min(confidence, 0.95))

    debug = {
        "provider": "ollama",
        "base_url": _env_ollama_base_url(),
        "model": _env_ollama_model(),
        "compact_evidence": compact,
        "prover": prover[:4000],
        "skeptic": skeptic[:4000],
        "judge_raw": judge_raw[:4000],
        "judge_json": j,
        "key_domains": key_domains,
    }

    return verdict, confidence, summary, debug


async def llm_final_judge(
    claim_text: str,
    evidence_items: List[Dict[str, Any]],
    baseline_verdict: str,
    baseline_confidence: float,
    structured_verdict: str,
    claim_profile: Dict[str, Any] | None = None,
) -> Tuple[Verdict, float, str, Dict[str, Any]]:
    """
    Professional final-answer judge.

    This is a single-pass JSON judge that turns the retrieved evidence into a
    user-facing final answer. It uses the heuristic verdict as a guardrail but
    is free to override it when the evidence supports a different conclusion.
    """
    compact = _compact_final_context(evidence_items, max_items=4)

    system = (
        "You are a professional fact-checking final judge.\n"
        "Use only the provided claim, evidence, and metadata.\n"
        "Prefer caution when evidence is weak, conflicting, or incomplete.\n"
        "Return strict JSON only. No markdown, no extra prose.\n"
    )

    prompt = (
        "Decide the final verdict for the claim.\n"
        "The output must be concise, professional, and suitable for end users.\n"
        "You must choose one verdict from SUPPORTED, REFUTED, or NEI.\n"
        "If the evidence does not clearly decide, choose NEI.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "verdict": "SUPPORTED|REFUTED|NEI",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "summary": "one sentence, professional and neutral",\n'
        '  "key_points": ["short bullet-like reason 1", "reason 2"],\n'
        '  "risk_notes": ["optional caution 1", "optional caution 2"]\n'
        "}\n\n"
        f"CLAIM:\n{claim_text}\n\n"
        f"CLAIM_PROFILE_JSON:\n{json.dumps(claim_profile or {}, ensure_ascii=False, indent=2)}\n\n"
        f"BASELINE_VERDICT:\n{baseline_verdict} ({baseline_confidence:.2f})\n\n"
        f"STRUCTURED_VERDICT:\n{structured_verdict}\n\n"
        f"EVIDENCE_JSON:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
    )

    raw = await _ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.1,
        num_ctx=2048,
        num_predict=220,
    )

    parsed = _safe_json_parse(raw)
    verdict = _normalize_verdict(parsed.get("verdict"))

    try:
        confidence = float(parsed.get("confidence", baseline_confidence))
    except Exception:
        confidence = baseline_confidence

    confidence = max(0.05, min(confidence, 0.95))
    summary = str(parsed.get("summary") or "Professional AI final judge completed.").strip()
    if not summary:
        summary = "Professional AI final judge completed."

    debug = {
        "provider": "ollama",
        "base_url": _env_ollama_base_url(),
        "model": _env_ollama_model(),
        "compact_evidence": compact,
        "raw": raw[:4000],
        "json": parsed,
        "key_points": parsed.get("key_points", []),
        "risk_notes": parsed.get("risk_notes", []),
    }

    return verdict, confidence, summary, debug
