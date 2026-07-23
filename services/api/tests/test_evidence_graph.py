from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.evidence_graph import adjudicate_graph, build_evidence_graph
from app.evidence_independence import cluster_evidence
from app.graph_auditor import audit_evidence_graph
from app.main import app, clean_text_for_claims, extract_claim_candidates, is_public_http_url, select_evidence_passage


def _passage(stance: str, quality: float = 85.0):
    return {
        "url": f"https://example.org/{stance}",
        "passage": f"The claim is {'false' if stance == 'refute' else 'confirmed'}.",
        "content_hash": stance,
        "retrieval_status": "retrieved",
        "stance": stance,
        "entity_match": True,
        "numeric_match": True,
        "directness_score": 0.8,
        "quality_score": quality,
    }


def test_short_claims_and_dates_are_retained():
    claim = "The Earth is flat."
    assert "March 11, 2020" in clean_text_for_claims("March 11, 2020: WHO made an announcement.")
    assert claim in extract_claim_candidates(claim, max_claims=1)


def test_passage_selection_retains_neighboring_context():
    text = (
        "The report describes the annual survey methodology in detail. "
        "The survey found that vaccination reduced hospitalization by 90 percent. "
        "The result applies only to the studied adult population."
    )
    passage = select_evidence_passage("Vaccination reduced hospitalization by 90 percent.", text)

    assert "annual survey methodology" in passage
    assert "reduced hospitalization by 90 percent" in passage
    assert "only to the studied adult population" in passage


@pytest.mark.asyncio
async def test_private_urls_are_rejected_before_fetching():
    assert await is_public_http_url("http://127.0.0.1:8000/admin") is False
    assert await is_public_http_url("http://localhost:8000/admin") is False


def test_graph_marks_unresolved_direct_conflict():
    graph = build_evidence_graph(
        "The Earth is flat.",
        ["The Earth is flat."],
        [_passage("support"), _passage("refute")],
        "ok",
    )
    verdict, reasons = adjudicate_graph(graph)

    assert graph["relation_counts"]["SUPPORTS"] == 1
    assert graph["relation_counts"]["REFUTES"] == 1
    assert verdict == "CONFLICTING"
    assert reasons


def test_auditor_marks_duplicate_decisive_evidence_for_human_review():
    evidence = [
        {**_passage("support"), "url": "https://example.org/a", "domain": "example.org", "base_domain": "example.org"},
        {**_passage("support"), "url": "https://example.org/b", "domain": "example.org", "base_domain": "example.org"},
    ]
    clusters = cluster_evidence(evidence)
    graph = build_evidence_graph("The Earth is flat.", ["The Earth is flat."], evidence, "ok")
    graph["independence_clusters"] = clusters
    audit = audit_evidence_graph(graph, evidence, {})

    assert len(clusters) == 1
    assert audit["decision"] == "HUMAN_REVIEW"
    assert any("correlated" in violation for violation in audit["violations"])


def test_auditor_explains_empty_evidence_abstention():
    graph = build_evidence_graph("A claim.", ["A claim."], [], "no_results")
    audit = audit_evidence_graph(graph, [], {})

    assert audit["decision"] == "NEI"
    assert "No fetched evidence passage" in audit["violations"][0]


@pytest.mark.asyncio
async def test_analyze_records_passage_provenance_and_conflict():
    results = [
        {"title": "Claim confirmed", "link": "https://example.org/support", "snippet": "A result snippet."},
        {"title": "Claim rejected", "link": "https://example.org/refute", "snippet": "Another result snippet."},
    ]

    async def passage(url: str, claim: str):
        if "support" in url:
            return "The claim that the Earth is flat is confirmed.", url, "retrieved"
        return "The claim that the Earth is flat is false.", url, "retrieved"

    with patch("app.main.SERPAPI_API_KEY", "test"), patch(
        "app.main.serpapi_search", new=AsyncMock(return_value=results)
    ), patch("app.main.fetch_evidence_passage", side_effect=passage), patch("app.main.get_cache", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/analyze", json={"text": "The Earth is flat.", "max_claims": 1})

    assert response.status_code == 200
    claim = response.json()["claims"][0]
    assert claim["verdict"] == "CONFLICTING"
    assert claim["evidence_graph"]["decision_basis"] == "fetched_passages_only"
    assert claim["graph_audit"]["version"] == "graph-auditor-v1"
    assert all(item["content_hash"] for item in claim["evidence"])
    assert all(item["relation_classifier"]["method"] == "heuristic-fallback" for item in claim["evidence"])
    manifest = claim["retrieval_manifest"]
    assert manifest["version"] == "retrieval-manifest-v1"
    assert manifest["search_results"] == [
        {"rank": 1, "title": "Claim confirmed", "url": "https://example.org/support", "snippet": "A result snippet."},
        {"rank": 2, "title": "Claim rejected", "url": "https://example.org/refute", "snippet": "Another result snippet."},
    ]
    assert manifest["manifest_hash"] and manifest["search_results_hash"]