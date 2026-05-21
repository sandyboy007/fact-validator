"""Integration tests for full fact-checking pipeline."""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os

from app.main import app


@pytest.mark.asyncio
async def test_analyze_with_url_live_baseline():
    """Test full /analyze endpoint with URL and live mode."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        with patch("app.main.fetch_html") as mock_fetch:
            # Mock HTML extraction
            mock_fetch.return_value = (
                "https://example.com/article",
                "<html><body>Global temperatures have risen by 1.1 degrees. Scientists warn of climate change.</body></html>"
            )
            
            with patch("app.main.serpapi_search") as mock_search:
                # Mock SerpAPI results
                mock_search.return_value = [
                    {
                        "title": "Climate Change Report",
                        "link": "https://www.bbc.com/news/climate",
                        "snippet": "Global temperatures have risen by 1.1 degrees according to latest research."
                    },
                    {
                        "title": "Scientists Warn",
                        "link": "https://reuters.com/science",
                        "snippet": "Climate change scientists warn of continued warming trends."
                    }
                ]
                
                response = await client.post("/analyze", json={
                    "url": "https://example.com/article",
                    "mode": "live",
                    "verifier": "baseline",
                    "max_claims": 3,
                    "max_evidence_per_claim": 5
                })
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert data["input_type"] == "url"
        assert data["domain"] is not None
        assert "domain_score" in data
        assert "final_misinformation_likelihood" in data
        assert isinstance(data["claims"], list)
        assert data["metadata"]["run_id"] is not None
        
        # Validate claim structure
        if data["claims"]:
            claim = data["claims"][0]
            assert "claim_text" in claim
            assert "verdict" in claim
            assert claim["verdict"] in ["SUPPORTED", "REFUTED", "NEI"]
            assert "confidence" in claim
            assert isinstance(claim["evidence"], list)


@pytest.mark.asyncio
async def test_analyze_with_debate_mode():
    """Test /analyze endpoint with debate mode enabled."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        with patch("app.main.fetch_html") as mock_fetch:
            mock_fetch.return_value = (
                "https://example.com/article",
                "<html><body>The Earth is flat. Scientists debate this claim.</body></html>"
            )
            
            with patch("app.main.serpapi_search") as mock_search:
                mock_search.return_value = [
                    {
                        "title": "Earth Shape",
                        "link": "https://nasa.gov/earth",
                        "snippet": "Earth is a sphere as confirmed by satellite imagery."
                    }
                ]
                
                # Test debate mode request
                response = await client.post("/analyze", json={
                    "url": "https://example.com/article",
                    "mode": "live",
                    "verifier": "debate",
                    "max_claims": 2,
                    "max_debate_claims": 1
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["input_type"] == "url"
                assert isinstance(data["claims"], list)


@pytest.mark.asyncio
async def test_analyze_input_validation():
    """Test that input validation works correctly."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        # Test claims count at reasonable level
        response = await client.post("/analyze", json={
            "text": "Some claim here.",
            "max_claims": 5
        })
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_analyze_snapshot_mode():
    """Test snapshot mode with claim memory caching."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        with patch("app.main.serpapi_search") as mock_search:
            mock_search.return_value = [
                {
                    "title": "Report",
                    "link": "https://example.com/report",
                    "snippet": "Test snippet for the claim here."
                }
            ]
            
            # First run in live mode
            response1 = await client.post("/analyze", json={
                "text": "Test claim here.",
                "mode": "live",
                "max_claims": 1
            })
            assert response1.status_code == 200
            
            # Second run in snapshot mode (should use cache)
            response2 = await client.post("/analyze", json={
                "text": "Test claim here.",
                "mode": "snapshot",
                "max_claims": 1
            })
            assert response2.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test basic health endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_deep_health_endpoint():
    """Test deep health check endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/health/deep")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "config" in data
        assert "ollama_available" in data


@pytest.mark.asyncio
async def test_dashboard_summary_endpoint():
    """Dashboard summary endpoint should aggregate run-level and claim-level metrics."""
    fake_runs = [
        {
            "id": 11,
            "created_utc": "2026-04-16T10:00:00Z",
            "input_type": "url",
            "domain": "example.com",
            "verifier": "baseline",
            "response": {
                "final_misinformation_likelihood": 0.25,
                "claims": [
                    {"verdict": "SUPPORTED", "needs_human_review": False},
                    {"verdict": "NEI", "needs_human_review": True},
                ],
            },
        },
        {
            "id": 10,
            "created_utc": "2026-04-16T09:00:00Z",
            "input_type": "text",
            "domain": None,
            "verifier": "debate",
            "response": {
                "final_misinformation_likelihood": 0.75,
                "claims": [
                    {"verdict": "REFUTED", "needs_human_review": False},
                ],
            },
        },
    ]

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        with patch("app.main.export_runs", return_value=fake_runs):
            response = await client.get("/dashboard/summary?limit=25")

    assert response.status_code == 200
    payload = response.json()

    assert payload["limit"] == 25
    assert payload["total_runs"] == 2
    assert payload["claims_analyzed"] == 3
    assert payload["claims_requiring_human_review"] == 1
    assert payload["avg_misinformation_likelihood"] == 0.5
    assert payload["input_type_counts"]["url"] == 1
    assert payload["input_type_counts"]["text"] == 1
    assert payload["verifier_counts"]["baseline"] == 1
    assert payload["verifier_counts"]["debate"] == 1
    assert payload["verdict_counts"]["SUPPORTED"] == 1
    assert payload["verdict_counts"]["REFUTED"] == 1
    assert payload["verdict_counts"]["NEI"] == 1
    assert payload["top_domains"][0]["domain"] == "example.com"


@pytest.mark.asyncio
async def test_evaluation_endpoints_available():
    """Evaluation endpoints used by analyst UI should be reachable."""
    expected_keys = {
        "/evaluation/benchmark": ["claims"],
        "/evaluation/baselines": ["metadata", "results"],
        "/evaluation/ablations": ["metadata", "variants"],
        "/evaluation/comparative": ["metadata", "ranking"],
        "/evaluation/production-metrics": ["metadata", "latency", "throughput", "cost", "quality"],
        "/evaluation/explainability": ["metadata", "case_studies"],
        "/evaluation/limitations": ["metadata", "limitations"],
        "/evaluation/reproducibility": ["metadata", "summary", "score"],
        "/evaluation/ethics": ["metadata", "ethical_risks"],
        "/evaluation/defense": ["metadata", "qa", "metrics_cheatsheet"],
    }

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        for path, keys in expected_keys.items():
            response = await client.get(path)
            assert response.status_code == 200, f"{path} returned {response.status_code}"

            payload = response.json()
            assert isinstance(payload, dict), f"{path} payload should be an object"
            for key in keys:
                assert key in payload, f"{path} missing expected key: {key}"


@pytest.mark.asyncio
async def test_analyze_with_empty_input():
    """Test that empty input is handled gracefully."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        # Both empty
        response = await client.post("/analyze", json={
            "url": "",
            "text": ""
        })
        # Should be validation error or 200 with no claims
        assert response.status_code in [422, 200]


@pytest.mark.asyncio
async def test_analyze_error_handling():
    """Test error handling during analysis with invalid input."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        # Test with both URL and text empty (both None)
        response = await client.post("/analyze", json={
            "mode": "live"
        })
        
        # Should be successful or return validation error
        assert response.status_code in [200, 422]


@pytest.mark.asyncio
async def test_analyze_reflective_abstention_and_metrics_present():
    """Reflective gate should request abstention when grounding is weak."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        with patch("app.main.serpapi_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "title": "Trend commentary",
                    "link": "https://example.com/blog/post",
                    "snippet": "Analysts discuss general warming trends without exact values or dates.",
                }
            ]

            with patch("app.main.get_cache", return_value=None):
                response = await client.post("/analyze", json={
                    "text": "Global temperature increased by 3.14 degrees in exactly 2024.",
                    "mode": "live",
                    "max_claims": 1,
                    "max_evidence_per_claim": 3,
                    "enable_reflective_abstention": True,
                })

            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["reflective_abstention_enabled"] is True
            assert len(data["claims"]) >= 1

            claim = data["claims"][0]
            assert "reflective" in claim
            assert claim["reflective"]["decision"] == "TERMINATE"
            assert claim["verdict"] == "NEI"
            assert claim.get("needs_human_review") is True


@pytest.mark.asyncio
async def test_analyze_generates_faithful_correction_for_refuted_claim():
    """Refuted claims should include a grounded correction candidate when enabled."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        with patch("app.main.serpapi_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "title": "Nature review",
                    "link": "https://www.nature.com/articles/example",
                    "snippet": "The claim is false. Photosynthesis has been measured repeatedly in controlled experiments for decades.",
                },
                {
                    "title": "Encyclopedia reference",
                    "link": "https://www.britannica.com/science/photosynthesis",
                    "snippet": "Photosynthesis is a well-studied process with extensive measurements in plant science.",
                },
            ]

            with patch("app.main.get_cache", return_value=None):
                response = await client.post("/analyze", json={
                    "text": "No credible source has ever measured photosynthesis at any time in history.",
                    "mode": "live",
                    "max_claims": 1,
                    "max_evidence_per_claim": 4,
                    "enable_faithful_correction": True,
                })

            assert response.status_code == 200
            data = response.json()
            assert len(data["claims"]) >= 1

            claim = data["claims"][0]
            assert claim["verdict"] == "REFUTED"
            correction = claim.get("faithful_correction")
            assert correction is not None
            assert isinstance(correction.get("proposed_correction"), str)
            assert correction["proposed_correction"]
            assert correction.get("score", 0) >= 0.45


@pytest.mark.asyncio
async def test_analyze_correction_when_reflective_abstention_disabled():
    """Faithful correction should still run when reflective abstention is disabled."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        with patch("app.main.serpapi_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "title": "Nature review",
                    "link": "https://www.nature.com/articles/example",
                    "snippet": "The claim is false. Photosynthesis has been measured repeatedly in controlled experiments for decades.",
                },
                {
                    "title": "Encyclopedia reference",
                    "link": "https://www.britannica.com/science/photosynthesis",
                    "snippet": "Photosynthesis is a well-studied process with extensive measurements in plant science.",
                },
            ]

            with patch("app.main.get_cache", return_value=None):
                response = await client.post("/analyze", json={
                    "text": "No credible source has ever measured photosynthesis at any time in history.",
                    "mode": "live",
                    "max_claims": 1,
                    "max_evidence_per_claim": 4,
                    "enable_reflective_abstention": False,
                    "enable_faithful_correction": True,
                })

            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["reflective_abstention_enabled"] is False
            assert len(data["claims"]) >= 1

            claim = data["claims"][0]
            assert claim["verdict"] == "REFUTED"
            correction = claim.get("faithful_correction")
            assert correction is not None
            assert isinstance(correction.get("proposed_correction"), str)
            assert correction["proposed_correction"]
            assert correction.get("score", 0) >= 0.45


@pytest.mark.asyncio
async def test_analyze_short_text_fallback_triggers_search():
    """Short natural-language text should still produce a fallback claim and run search."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        with patch("app.main.serpapi_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "title": "Nature of Leaves",
                    "link": "https://www.britannica.com/science/leaf-plant-anatomy",
                    "snippet": "Leaves are naturally green due to chlorophyll pigments in plant cells."
                }
            ]

            # Disable cache to ensure the mocked search path is exercised.
            with patch("app.main.get_cache", return_value=None):
                response = await client.post("/analyze", json={
                    "text": "true or false: leaves are naturally green",
                    "mode": "live",
                    "max_claims": 3,
                    "max_evidence_per_claim": 2,
                })

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data["claims"], list)
                assert len(data["claims"]) >= 1
                mock_search.assert_awaited()
