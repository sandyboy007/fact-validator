#!/usr/bin/env python
"""Quick test to check if API endpoints work"""
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

print("Testing API endpoints...\n")

# Test 1: Health check
print("1. Testing GET /docs (Swagger UI)")
response = client.get("/docs")
print(f"   Status: {response.status_code}")

# Test 2: Analyze with text
print("\n2. Testing POST /analyze with text")
response = client.post("/analyze", json={
    "text": "The Earth orbits the Sun",
    "mode": "live",
    "max_claims": 1
})
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"   Claims found: {len(data.get('claims', []))}")
    print(f"   Response keys: {list(data.keys())}")
else:
    print(f"   Error: {response.text}")

# Test 3: Analyze with URL
print("\n3. Testing POST /analyze with URL")
response = client.post("/analyze", json={
    "url": "https://en.wikipedia.org/wiki/Earth",
    "mode": "live",
    "max_claims": 1
})
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"   Claims found: {len(data.get('claims', []))}")
else:
    print(f"   Error: {response.text[:200]}")

print("\n✓ API tests complete")
