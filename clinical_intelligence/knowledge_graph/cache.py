from __future__ import annotations
import time
from typing import Any, Dict, Optional


class GraphQueryCache:
    """Configurable TTL-based query caching layer with telemetry tracking."""

    def __init__(self, default_ttl_seconds: float = 300.0) -> None:
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (value, expire_timestamp)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached query response if present and unexpired. Otherwise return None."""
        if key not in self._cache:
            self.misses += 1
            return None

        val, expire_time = self._cache[key]
        if time.perf_counter() > expire_time:
            # Item expired, clean up
            del self._cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return val

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Cache a query result with a specific TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expire_time = time.perf_counter() + ttl
        self._cache[key] = (value, expire_time)

    def clear(self) -> None:
        """Purge all entries from cache."""
        self._cache.clear()

    def reset_metrics(self) -> None:
        """Reset hit and miss telemetry counters."""
        self.hits = 0
        self.misses = 0

    @property
    def hit_ratio(self) -> float:
        """Compute the ratio of cache hits to total lookups [0.0, 1.0]."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return float(self.hits) / total

    @property
    def miss_ratio(self) -> float:
        """Compute the ratio of cache misses to total lookups [0.0, 1.0]."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return float(self.misses) / total
