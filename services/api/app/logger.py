"""Structured logging utilities."""
import logging
import os
from typing import Any, Dict
from datetime import datetime

# Configure logging
logging_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=getattr(logging, logging_level),
    format=log_format,
)

logger = logging.getLogger("fact_validator")


def log_analyze_start(request_type: str, domain: str, claims_count: int) -> None:
    """Log start of analysis."""
    logger.info(
        f"Analyze request started: type={request_type}, domain={domain}, max_claims={claims_count}"
    )


def log_analyze_complete(
    run_id: int, claims_extracted: int, misinformation_likelihood: float, duration_ms: float
) -> None:
    """Log successful analysis completion."""
    logger.info(
        f"Analyze request completed: run_id={run_id}, claims={claims_extracted}, "
        f"likelihood={misinformation_likelihood:.2f}, duration_ms={duration_ms:.1f}"
    )


def log_debate_started(claim_text: str, max_length: int = 100) -> None:
    """Log start of debate for a claim."""
    truncated = claim_text[:max_length] + ("..." if len(claim_text) > max_length else "")
    logger.info(f"Debate started for claim: {truncated}")


def log_debate_error(claim_text: str, error: Exception) -> None:
    """Log debate error with context."""
    logger.error(f"Debate failed for claim (fallback to baseline): {str(error)}")


def log_cache_hit(key: str, item_type: str = "item") -> None:
    """Log cache hit."""
    logger.debug(f"Cache hit: {item_type} key={key[:50] if len(key) > 50 else key}")


def log_cache_miss(key: str, item_type: str = "item") -> None:
    """Log cache miss."""
    logger.debug(f"Cache miss: {item_type} key={key[:50] if len(key) > 50 else key}")


def log_validation_error(field: str, reason: str) -> None:
    """Log input validation error."""
    logger.warning(f"Validation error: field={field}, reason={reason}")


def log_request_context(request_id: str, endpoint: str, timestamp: str) -> None:
    """Log request context for tracing."""
    logger.info(f"Request: id={request_id}, endpoint={endpoint}, timestamp={timestamp}")
