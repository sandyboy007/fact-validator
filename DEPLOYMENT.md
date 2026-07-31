# Deployment Guide

This document distinguishes the implemented local prototype from deployment
work that remains to be completed. The repository does not currently contain
API or frontend Dockerfiles, Kubernetes manifests, production monitoring,
distributed rate limiting, or a production database configuration.

## Environment Variables

### Core Configuration
- `SERPAPI_API_KEY` - Required for Google Search evidence retrieval (get from https://serpapi.com)
- `OLLAMA_BASE_URL` - Ollama service URL for debate mode (default: http://127.0.0.1:11434)
- `OLLAMA_MODEL` - Model to use for debate (default: llama3.1:8b)
- `OLLAMA_ENABLED` - Enable Ollama-based features (default: false)
- `FEATURE_DEBATE_MODE` - Enable LLM debate verifier (default: true)
- `FEATURE_CACHING` - Enable SerpAPI result caching (default: true)
- `FEATURE_RATE_LIMITING` - Enable rate limiting (default: true)
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR; default: INFO)

### Database
- `FACT_VALIDATOR_DB` - SQLite database path (default: `<repo>/data/fact_validator.db`)
- `FACTVALIDATOR_DB` - Legacy alias for `FACT_VALIDATOR_DB` (for backward compatibility)

### Path Overrides
- `FACTVALIDATOR_DOMAIN_CACHE` - Credibility score cache path (default: `services/api/data/domain_cache.json`)

## Optional Infrastructure Containers

### Quick Start with Docker Compose

```bash
# Optional: Start Postgres + Redis (currently not used, reserved for future)
cd infra
docker compose up -d

```

Postgres and Redis are not connected to the application. The implemented
local run path is documented in `README.md`. Containerizing the API and web
application is future work and must not be described as a completed deployment.

## Illustrative Kubernetes Planning Notes

The following values are design notes only. No Helm chart is included, and
these values are not a tested deployment artifact:

```yaml
# values.yaml
api:
  replicas: 3
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"
  env:
    SERPAPI_API_KEY: "${SERPAPI_API_KEY}"
    OLLAMA_ENABLED: "true"
    FEATURE_RATE_LIMITING: "true"
    LOG_LEVEL: "INFO"

web:
  replicas: 2
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
```

## Performance Tuning

### For High Throughput
- Enable caching: `FEATURE_CACHING=true`
- Adjust rate limits in code (default: 100 req/min)
- Use Redis for distributed caching (future enhancement)

### For Lower Latency  
- Run Ollama on same machine as API for debate mode
- Pre-warm domain credibility cache by running a few analyses
- Use high-quality GPU for Ollama if available

### For Cost Optimization
- Cache SerpAPI results: `FEATURE_CACHING=true`
- Disable debate mode if not needed: `FEATURE_DEBATE_MODE=false`
- Use lite Ollama models: `OLLAMA_MODEL=tinyllama` or `neural-chat:7b`

## Monitoring

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Deep health with Ollama status
curl http://localhost:8000/health/deep
```

### Logging

Structured logs are output to stdout with format: `timestamp - logger - level - message`

Analyze request logs include run_id for tracing through system.

## Database Migration

Current system uses SQLite by default. To migrate to production database:

1. Current: SQLite in `data/fact_validator.db`
2. Future: PostgreSQL via SQLModel (infrastructure in place but not activated)

To enable PostgreSQL:
- Implement `app/db/database.py` integration with StorageAdapter pattern
- Update `storage.py` to use SQLModel instead of raw sqlite3

## Security Checklist

- [ ] Set strong `SERPAPI_API_KEY` access restrictions
- [ ] Enable rate limiting: `FEATURE_RATE_LIMITING=true`
- [ ] Use HTTPS in production
- [ ] Validate all input (done via Pydantic validators)
- [ ] Set appropriate `LOG_LEVEL` (INFO or WARNING in production)
- [ ] Restrict CORS origins in production
- [ ] Monitor for unusual SerpAPI quota usage
- [ ] Set resource limits on containers

## Troubleshooting

### Debate mode not working
1. Check Ollama is running: `curl http://127.0.0.1:11434/api/tags`
2. Check model is installed: `ollama pull llama3.1:8b`
3. Check logs: requests with `"debate_available": false` in response

### Slow responses
1. Enable caching: `FEATURE_CACHING=true`
2. Check SerpAPI quota not exhausted
3. Monitor Ollama GPU usage if using debate mode
4. Consider parallel evidence fetching (future enhancement)

### Database errors
1. Ensure write permissions on data directory
2. Check disk space for SQLite growth
3. Backup before major migrations

## Zero-Downtime Deployment

1. Deploy new API version with canary (10% traffic)
2. Verify no regression in response quality
3. Gradually roll out to 100%
4. Keep old instance running during transition
5. Database schema is backward-compatible with versioning

## Backup Strategy

```bash
# Daily backup of SQLite database
tar czf fact_validator_backup_$(date +%Y%m%d).tar.gz \
  ./data/fact_validator.db \
  ./services/api/data/domain_cache.json

# Upload to S3
aws s3 cp fact_validator_backup_*.tar.gz s3://your-bucket/backups/
```
