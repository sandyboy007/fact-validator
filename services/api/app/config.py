"""Feature flags and configuration management."""
import os
from typing import Dict, Any


class Config:
    """Application configuration with feature flags."""
    
    # Feature flags
    FEATURE_DEBATE_MODE = os.getenv("FEATURE_DEBATE_MODE", "true").lower() in ("true", "1", "yes")
    FEATURE_CACHING = os.getenv("FEATURE_CACHING", "true").lower() in ("true", "1", "yes")
    FEATURE_RATE_LIMITING = os.getenv("FEATURE_RATE_LIMITING", "true").lower() in ("true", "1", "yes")
    FEATURE_STRUCTURED_LOGGING = os.getenv("FEATURE_STRUCTURED_LOGGING", "true").lower() in ("true", "1", "yes")
    
    # Limits
    MAX_INPUT_URL_LENGTH = 2048
    MAX_INPUT_TEXT_LENGTH = 50000
    MAX_CLAIMS_HARD_LIMIT = 20
    MAX_EVIDENCE_PER_CLAIM_LIMIT = 10
    
    # Timeouts (seconds)
    SERPAPI_TIMEOUT = int(os.getenv("SERPAPI_TIMEOUT", "30"))
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    # Ollama config
    OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "false").lower() in ("true", "1", "yes")
    
    @staticmethod
    def get_all() -> Dict[str, Any]:
        """Return all config as dict for logging/debugging."""
        return {
            "feature_debate_mode": Config.FEATURE_DEBATE_MODE,
            "feature_caching": Config.FEATURE_CACHING,
            "feature_rate_limiting": Config.FEATURE_RATE_LIMITING,
            "feature_structured_logging": Config.FEATURE_STRUCTURED_LOGGING,
            "max_input_url_length": Config.MAX_INPUT_URL_LENGTH,
            "max_input_text_length": Config.MAX_INPUT_TEXT_LENGTH,
            "ollama_enabled": Config.OLLAMA_ENABLED,
        }
