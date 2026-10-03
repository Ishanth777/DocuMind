"""
DocuMind In-Memory Caching Module
Caches frequent RAG query answers with TTL to:
1. Deliver sub-millisecond response times (< 1ms) for repeat queries
2. Drastically reduce LLM API tokens & costs
3. Prevent rate-limiting on LLM Gateways (Groq/OpenAI)
"""

import os
import time
import hashlib
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("DocuMind.Cache")

DEFAULT_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default


class RAGCache:
    """
    High-performance In-Memory TTL & LRU Cache for RAG responses.
    """

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS, max_entries: int = 1000):
        self.ttl = ttl
        self.max_entries = max_entries
        self.backend = "memory"
        self._store: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        logger.info("[Cache] Initialized In-Memory Cache layer.")

    def _generate_key(self, query: str, top_k: int) -> str:
        """Normalized hash key for deterministic lookup."""
        normalized = " ".join(query.lower().strip().split())
        raw_key = f"documind:{top_k}:{normalized}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, query: str, top_k: int = 2) -> Optional[Dict[str, Any]]:
        """Retrieves cached response if available and not expired."""
        key = self._generate_key(query, top_k)
        now = time.time()

        entry = self._store.get(key)
        if entry:
            if entry["expires_at"] > now:
                self.hits += 1
                logger.info(f"[Cache HIT] Served from in-memory cache for: '{query[:30]}...'")
                return entry["data"]
            else:
                del self._store[key]

        self.misses += 1
        return None

    def set(self, query: str, top_k: int, result: Dict[str, Any], ttl: Optional[int] = None):
        """Stores RAG response in cache with TTL and LRU pruning."""
        key = self._generate_key(query, top_k)
        ttl = ttl or self.ttl
        now = time.time()

        # Simple LRU eviction if cache exceeds capacity
        if len(self._store) >= self.max_entries:
            oldest_key = min(self._store.keys(), key=lambda k: self._store[k]["expires_at"])
            del self._store[oldest_key]

        self._store[key] = {
            "data": result,
            "expires_at": now + ttl,
        }

    def clear(self):
        """Clears all cached queries."""
        self._store.clear()
        logger.info("[Cache] In-memory cache cleared.")

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "backend": "memory",
            "cached_entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "total_queries": total,
            "hit_rate_pct": round(hit_rate, 1),
        }
