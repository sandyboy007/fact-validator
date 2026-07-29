"""Caching utilities for API results and evidence."""
import json
import logging
import os
import shutil
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


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

    def _quarantine_invalid_file(self, cache_file: Path, reason: str) -> None:
        """Move an invalid cache record aside so it cannot be reused as evidence."""
        quarantine_dir = self.cache_dir / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / cache_file.name
        counter = 1
        while destination.exists():
            destination = quarantine_dir / f"{cache_file.stem}.{counter}{cache_file.suffix}"
            counter += 1
        try:
            shutil.move(str(cache_file), str(destination))
            logger.warning(
                "Quarantined invalid evidence cache %s -> %s (%s)",
                cache_file,
                destination,
                reason,
            )
        except OSError:
            logger.exception(
                "Invalid evidence cache could not be quarantined: %s (%s)",
                cache_file,
                reason,
            )
    
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
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if not isinstance(entry, dict):
                    raise ValueError("cache root must be a JSON object")
                if not isinstance(entry.get("ts"), (int, float)):
                    raise ValueError("cache record requires numeric ts")
                if not isinstance(entry.get("results"), list):
                    raise ValueError("cache record requires results list")
                if time.time() - entry["ts"] < self.ttl_seconds:
                    # Populate memory cache
                    self.memory_cache[key] = entry
                    return entry["results"]
                else:
                    cache_file.unlink()
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                self._quarantine_invalid_file(cache_file, str(exc))
            except OSError:
                logger.exception("Failed to read evidence cache: %s", cache_file)
        
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
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
        except OSError:
            logger.exception("Failed to write evidence cache: %s", cache_file)


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
