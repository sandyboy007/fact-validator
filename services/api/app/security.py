"""Security and health check utilities."""
import httpx
import os
from typing import Tuple, Optional
from datetime import datetime, timedelta


class OllamaHealthCheck:
    """Check Ollama service availability."""
    
    def __init__(self):
        self.last_check: Optional[datetime] = None
        self.last_result: bool = False
        self.check_cache_ttl = 30  # Cache result for 30 seconds
    
    async def is_available(self) -> bool:
        """Check if Ollama service is available."""
        now = datetime.utcnow()
        
        # Use cached result if still fresh
        if self.last_check and (now - self.last_check).seconds < self.check_cache_ttl:
            return self.last_result
        
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
        
        try:
            timeout = httpx.Timeout(connect=5.0, read=5.0)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                response = await client.get(f"{ollama_url}/api/tags", follow_redirects=True)
                success = response.status_code == 200
        except Exception:
            success = False
        
        self.last_check = now
        self.last_result = success
        return success


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times: dict = {}  # ip -> [timestamps]
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request from client is allowed."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        
        if client_ip not in self.request_times:
            self.request_times[client_ip] = []
        
        # Remove old timestamps
        self.request_times[client_ip] = [
            ts for ts in self.request_times[client_ip] if ts > cutoff
        ]
        
        # Check limit
        if len(self.request_times[client_ip]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.request_times[client_ip].append(now)
        return True


# Global instances
ollama_health = OllamaHealthCheck()
rate_limiter = RateLimiter(requests_per_minute=100)
