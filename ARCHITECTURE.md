# Fact Validator - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Layer                                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Next.js Frontend (React 19 + TypeScript + Tailwind CSS)       │ │
│  │  - Claim Analysis UI                                            │ │
│  │  - URL/Text Input Interface                                     │ │
│  │  - Results Dashboard & History                                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↕ (HTTP/REST)
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  FastAPI Backend (Python 3.10+)                                │ │
│  │  - Analyze Endpoint (POST /analyze)                             │ │
│  │  - Source Credibility Routes                                    │ │
│  │  - Results Management & Export                                  │ │
│  │  - CORS Middleware & Rate Limiting                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. Content Extraction (Trafilatura)                          │  │
│  │    └─→ Clean HTML parsing & boilerplate removal              │  │
│  │                                                                │  │
│  │ 2. Claim Decomposition (NLP)                                 │  │
│  │    └─→ Sentence tokenization & scoring                        │  │
│  │    └─→ Select top 6 fact-like claims                          │  │
│  │                                                                │  │
│  │ 3. Evidence Retrieval (SerpAPI)                              │  │
│  │    └─→ Web search for each claim                              │  │
│  │    └─→ Fetch & parse result documents                         │  │
│  │                                                                │  │
│  │ 4. Source Credibility Scoring                                │  │
│  │    └─→ Domain reputation heuristics                           │  │
│  │    └─→ Score 0-100 per domain                                 │  │
│  │                                                                │  │
│  │ 5. Semantic Reranking (Sentence Transformers)                │  │
│  │    └─→ Relevance scoring of evidence                          │  │
│  │    └─→ Top-K filtering                                        │  │
│  │                                                                │  │
│  │ 6. Verdict Classification (NLP)                              │  │
│  │    └─→ SUPPORTED / REFUTED / NEI                              │  │
│  │    └─→ Confidence scores                                      │  │
│  │                                                                │  │
│  │ 7. LLM Debate Mode (Optional - Ollama)                       │  │
│  │    └─→ Multi-turn LLM reasoning on claims                     │  │
│  │    └─→ Enhanced verdict with reasoning                        │  │
│  │                                                                │  │
│  │ 8. Sentiment & Bias Analysis                                 │  │
│  │    └─→ Sentiment polarity detection                           │  │
│  │    └─→ Bias risk estimation                                   │  │
│  │    └─→ Misinformation likelihood adjustment                   │  │
│  │                                                                │  │
│  │ 9. Final Scoring                                             │  │
│  │    └─→ Aggregate misinformation score (0-100%)                │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Layer                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  SQLite Database (Local Persistence)                         │  │
│  │  - Run history storage                                        │  │
│  │  - Claim memory cache                                         │  │
│  │  - CSV/JSON export capabilities                               │  │
│  │                                                                │  │
│  │  JSON Cache (data/cache/)                                     │  │
│  │  - Result caching for repeated queries                        │  │
│  │  - Domain cache for credibility scores                        │  │
│  │  - IFFy Index (credibility reference data)                    │  │
│  │  - OpenSources domain reputation data                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────────┐
│                    External Services                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  SerpAPI → Google Search API (Evidence Search)                │  │
│  │  Ollama → Local LLM Service (Debate Mode)                     │  │
│  │  Hugging Face Models → Semantic Transformers                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

### Analyze Request Flow
```
User Input (URL/Text)
    ↓
API /analyze endpoint
    ↓
Content Extraction (Trafilatura)
    ↓
Claim Decomposition (NLP tokenization + scoring)
    ↓
[For each top claim]:
    ├─→ SerpAPI search query
    ├─→ Parse & fetch evidence links
    ├─→ Credibility scoring per domain
    └─→ Semantic reranking
    ↓
Verdict Determination
    ↓
[Optional] LLM Debate Processing
    ↓
Sentiment & Bias Analysis
    ↓
Final Misinformation Score Calculation
    ↓
Store in SQLite + Cache JSON
    ↓
Return JSON Response to Frontend
```

## Component Breakdown

### Backend Components (FastAPI)

**Core Modules:**
- `main.py` - FastAPI app initialization, endpoints, middleware
- `analysis_features.py` - NLP analysis: claim decomposition, verdict determination
- `credibility.py` - Domain credibility scoring heuristics
- `semantic_retrieval.py` - Sentence transformer-based reranking
- `sentiment.py` - Sentiment analysis and bias risk scoring
- `debate.py` - LLM-backed debate reasoning via Ollama
- `storage.py` - SQLite database wrapper
- `cache.py` - JSON-based caching layer
- `security.py` - Rate limiting, health checks
- `config.py` - Configuration management
- `logger.py` - Structured logging

**Key Endpoints:**
- `POST /analyze` - Fact-check URL or text
- `GET /runs` - List all analysis runs
- `GET /runs/{run_id}` - Get specific run details
- `POST /export` - Export results (CSV/JSON)
- `GET /source/{domain}` - Source credibility info
- `GET /health` - API health check

### Frontend Components (Next.js)

**Layout & Pages:**
- `app/layout.tsx` - Root layout with global styles
- `app/page.tsx` - Home page / Analyze interface
- `app/run/[id]/page.tsx` - Individual result view
- `app/source/page.tsx` - Source credibility explorer
- `components/ui.tsx` - Reusable UI components

**Features:**
- Real-time claim analysis submission
- Results history & pagination
- Source credibility lookup
- Result export functionality
- Responsive design (Tailwind CSS)

### Database Schema

**Runs Table:**
```sql
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    input TEXT,
    source_domain TEXT,
    source_credibility REAL,
    overall_score REAL,
    verdict_summary TEXT,
    claims TEXT,  -- JSON array
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

**Claim Memory Table:**
```sql
CREATE TABLE IF NOT EXISTS claim_memory (
    claim_key TEXT PRIMARY KEY,
    claim TEXT,
    verdict TEXT,
    confidence REAL,
    evidence_count INTEGER,
    cached_at TIMESTAMP
)
```

## Technology Stack

### Backend
- **Framework:** FastAPI 0.11x (async Python web framework)
- **Server:** Uvicorn (ASGI server)
- **NLP:** NLTK (tokenization, sentence processing)
- **ML Models:** Sentence Transformers (semantic similarity)
- **Web Scraping:** Trafilatura (content extraction)
- **APIs:** SerpAPI (search), Ollama (local LLM)
- **Database:** SQLite (local persistence)
- **Validation:** Pydantic (data validation)
- **Logging:** Python JSON Logger (structured logs)
- **Rate Limiting:** SlowAPI
- **HTTP Client:** HTTPX (async HTTP)

### Frontend
- **Framework:** Next.js 16 (React meta-framework)
- **Runtime:** Node.js
- **Language:** TypeScript
- **Styling:** Tailwind CSS 4
- **Component Library:** React 19
- **Build Tool:** Next.js built-in (Webpack)

### Infrastructure
- **Docker Compose:** Multi-service orchestration
- **PostgreSQL 16:** Optional persistent database (in compose)
- **Redis 7:** Optional caching layer (in compose)
- **Containerization:** Docker support for both services

## Credibility Scoring Algorithm

Domains are scored 0-100 based on:

1. **Known Reputable Publishers** (+30 pts)
   - BBC, Reuters, NPR, Forbes, AP News, Washington Post, Guardian, etc.

2. **Institutional Domains** (+25 pts)
   - `.gov` - Government official
   - `.edu` - Educational institutions
   - `.org` - Registered non-profits

3. **Academic Markers** (+20 pts)
   - journals.org, scholar.google.com, academia.edu, arxiv.org
   - Research publication indicators

4. **Trust Signals** (+15 pts)
   - Published bylines, timestamps, structured metadata

5. **Penalties** (-points)
   - Social media platforms: -30
   - Blog hosts (Medium, Blogger): -20
   - Shortened URLs: -15
   - No HTTPS: -10

## Deployment Architecture

### Local Development
- FastAPI dev server on `localhost:8000`
- Next.js dev server on `localhost:3000`
- SQLite in `data/fact_validator.db`
- CORS enabled for local ports

### Docker Deployment
- API container: Python 3.10 + FastAPI + Uvicorn
- Web container: Node.js + Next.js
- PostgreSQL service (optional)
- Redis service (optional)
- Shared volumes for data persistence

### Environment Variables
```
# API Configuration
FACT_VALIDATOR_DB=data/fact_validator.db
SERPAPI_API_KEY=<your-key>
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Optional
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

## Performance Considerations

1. **Caching:** Results cached per unique input to prevent redundant processing
2. **Async Operations:** FastAPI async handlers for non-blocking I/O
3. **Rate Limiting:** 100 requests/minute per IP (configurable)
4. **Semantic Reranking:** Top-K filtering reduces processing overhead
5. **Query Optimization:** SerpAPI result limiting (20 results max)

## Security Features

1. **CORS Middleware:** Whitelist local development + production origins
2. **Rate Limiting:** SlowAPI-based throttling per IP
3. **Health Checks:** Ollama connectivity verification
4. **Input Validation:** Pydantic models for all requests
5. **Error Handling:** Safe error responses without leaking internals

## Future Enhancements

1. **Multi-Model Support** - Expand to GPT-4, Claude alternatives
2. **GraphQL API** - Alternative query interface
3. **WebSocket Support** - Real-time streaming of analysis
4. **Advanced Caching** - Redis integration for distributed setups
5. **Analytics Dashboard** - Track most-checked claims
6. **Fact Database** - Integration with known fact-check repositories
7. **Custom Models** - Fine-tuned models for domain-specific fact-checking
