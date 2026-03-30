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
