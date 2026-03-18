"""Caching utilities for API results and evidence."""
import json
import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path


class ResultCache:
    """Simple in-memory + filesystem cache for SerpAPI results."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            repo_root = Path(__file__).resolve().parents[3]
            self.cache_dir = repo_root / "data" / "cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ttl_seconds = 60 * 60 * 24  # 24 hour default
    
    def _hash_claim(self, claim: str) -> str:
        """Create cache key from claim text."""
        import hashlib
        return hashlib.md5(claim.lower().encode()).hexdigest()
    
    def get(self, claim: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached evidence for a claim."""
        key = self._hash_claim(claim)
        
        # Check memory first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry["ts"] < self.ttl_seconds:
                return entry["results"]
            else:
                del self.memory_cache[key]
        
        # Check filesystem
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    entry = json.load(f)
                if time.time() - entry["ts"] < self.ttl_seconds:
                    # Populate memory cache
                    self.memory_cache[key] = entry
                    return entry["results"]
                else:
                    cache_file.unlink()
            except Exception:
                pass
        
        return None
    
    def set(self, claim: str, results: List[Dict[str, Any]]) -> None:
        """Cache evidence for a claim."""
        key = self._hash_claim(claim)
        entry = {"ts": time.time(), "results": results}
        
        # Store in memory
        self.memory_cache[key] = entry
        
        # Store on disk
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(entry, f, ensure_ascii=False)
        except Exception:
            pass  # Silently fail on write


# Global cache instance
_result_cache: Optional[ResultCache] = None


def get_cache() -> ResultCache:
    """Get or create global cache instance."""
    global _result_cache
    if _result_cache is None:
        _result_cache = ResultCache()
    return _result_cache


def clear_cache() -> None:
    """Clear cache instance."""
    global _result_cache
    _result_cache = None
