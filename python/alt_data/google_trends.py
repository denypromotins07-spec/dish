"""
Async Google Trends Data Fetcher for Alternative Data Integration.
Normalizes low-frequency search volume into high-frequency feature store
using forward-fill and exponential decay. Designed for <14GB RAM.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
import aiohttp


@dataclass
class TrendsDataPoint:
    """Single Google Trends data point."""
    keyword: str
    timestamp_ms: int
    value: int  # 0-100 scale
    is_partial: bool = False
    
    @property
    def normalized_value(self) -> float:
        """Normalize to 0-1 range."""
        return self.value / 100.0


class ExponentialDecayInterpolator:
    """
    Interpolates low-frequency data using exponential decay.
    Converts daily/weekly trends to per-minute features.
    """
    
    def __init__(
        self,
        half_life_minutes: int = 60,  # Value halves every hour
        max_history_minutes: int = 10080,  # 7 days max
    ):
        self.half_life_minutes = half_life_minutes
        self.max_history_minutes = max_history_minutes
        
        # Decay constant: lambda = ln(2) / half_life
        self._decay_constant = 0.693147 / half_life_minutes
        
        # Store recent data points per keyword
        self._history: Dict[str, deque] = {}
    
    def add_point(self, keyword: str, timestamp_ms: int, value: float):
        """Add a new data point for a keyword."""
        if keyword not in self._history:
            self._history[keyword] = deque(maxlen=self.max_history_minutes)
        
        self._history[keyword].append((timestamp_ms, value))
    
    def get_current_value(self, keyword: str, current_time_ms: Optional[int] = None) -> float:
        """
        Get interpolated current value using exponential decay.
        More recent points have higher weight.
        """
        if current_time_ms is None:
            current_time_ms = int(time.time() * 1000)
        
        if keyword not in self._history or not self._history[keyword]:
            return 0.0
        
        history = self._history[keyword]
        weighted_sum = 0.0
        weight_total = 0.0
        
        for ts_ms, value in reversed(history):
            age_minutes = (current_time_ms - ts_ms) / 60000.0
            
            if age_minutes > self.max_history_minutes:
                continue
            
            # Exponential decay weight
            weight = 2.0 ** (-age_minutes / self.half_life_minutes)
            weighted_sum += value * weight
            weight_total += weight
        
        if weight_total == 0:
            return 0.0
        
        return weighted_sum / weight_total
    
    def get_all_current_values(self) -> Dict[str, float]:
        """Get current interpolated values for all keywords."""
        current_time_ms = int(time.time() * 1000)
        return {
            keyword: self.get_current_value(keyword, current_time_ms)
            for keyword in self._history.keys()
        }
    
    def forward_fill(
        self,
        keyword: str,
        start_ms: int,
        end_ms: int,
        interval_ms: int = 60000,  # 1 minute default
    ) -> List[Tuple[int, float]]:
        """
        Generate forward-filled time series at regular intervals.
        Returns list of (timestamp, value) tuples.
        """
        result = []
        current_ms = start_ms
        
        while current_ms <= end_ms:
            value = self.get_current_value(keyword, current_ms)
            result.append((current_ms, value))
            current_ms += interval_ms
        
        return result


class GoogleTrendsFetcher:
    """
    Asynchronous Google Trends data fetcher.
    Implements strict memory controls and efficient caching.
    """
    
    # Default crypto-related keywords to track
    DEFAULT_KEYWORDS = [
        "Bitcoin",
        "Ethereum",
        "Cryptocurrency",
        "Crypto crash",
        "Bitcoin price",
        "Buy Bitcoin",
        "Crypto news",
        "BTC",
        "ETH",
        "DeFi",
    ]
    
    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        cache_ttl_seconds: int = 3600,
        max_cache_size: int = 100,
    ):
        self._session = session
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_cache_size = max_cache_size
        
        self._cache: Dict[str, Tuple[int, List[TrendsDataPoint]]] = {}
        self._interpolator = ExponentialDecayInterpolator()
        self._stats = {
            "fetches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is initialized."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=5, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session
    
    def _check_cache(self, keyword: str) -> Optional[List[TrendsDataPoint]]:
        """Check cache for existing data."""
        if keyword in self._cache:
            cached_time, data = self._cache[keyword]
            if time.time() - cached_time < self._cache_ttl_seconds:
                self._stats["cache_hits"] += 1
                return data
            else:
                del self._cache[keyword]
        self._stats["cache_misses"] += 1
        return None
    
    def _set_cache(self, keyword: str, data: List[TrendsDataPoint]):
        """Set data in cache with eviction policy."""
        # Evict oldest if over limit
        if len(self._cache) >= self._max_cache_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        
        self._cache[keyword] = (int(time.time()), data)
    
    async def fetch_keyword(
        self,
        keyword: str,
        timeframe: str = "now 7-d",  # Last 7 days
        geo: str = "US",
    ) -> List[TrendsDataPoint]:
        """
        Fetch Google Trends data for a single keyword.
        
        Note: In production, this would use the pytrends library or
        unofficial Google Trends API. This implementation shows the structure.
        """
        # Check cache first
        cached = self._check_cache(keyword)
        if cached:
            return cached
        
        self._stats["fetches"] += 1
        
        try:
            session = await self._ensure_session()
            
            # In production, use actual API endpoint
            # For demo, we'll simulate the structure
            # Real implementation would use pytrends or similar
            
            # Simulated response structure
            now_ms = int(time.time() * 1000)
            data = []
            
            # Generate simulated daily data points
            for i in range(7):
                timestamp_ms = now_ms - (i * 24 * 60 * 60 * 1000)
                # Simulate realistic search volume (would come from API)
                value = 50 + int(20 * ((i % 3) - 1))  # Varies between 30-70
                
                data.append(TrendsDataPoint(
                    keyword=keyword,
                    timestamp_ms=timestamp_ms,
                    value=value,
                    is_partial=(i == 0),  # Current day is partial
                ))
            
            # Cache the result
            self._set_cache(keyword, data)
            
            # Add to interpolator
            for point in data:
                self._interpolator.add_point(
                    keyword, 
                    point.timestamp_ms, 
                    point.normalized_value
                )
            
            return data
            
        except Exception as e:
            self._stats["errors"] += 1
            raise
    
    async def fetch_all_keywords(
        self,
        keywords: Optional[List[str]] = None,
    ) -> Dict[str, List[TrendsDataPoint]]:
        """Fetch trends for multiple keywords concurrently."""
        keywords = keywords or self.DEFAULT_KEYWORDS
        
        tasks = [self.fetch_keyword(kw) for kw in keywords]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for keyword, result in zip(keywords, results):
            if isinstance(result, list):
                output[keyword] = result
            else:
                output[keyword] = []  # Error case
        
        return output
    
    def get_interpolated_values(self) -> Dict[str, float]:
        """Get current interpolated values for all tracked keywords."""
        return self._interpolator.get_all_current_values()
    
    def get_composite_interest(self, keywords: Optional[List[str]] = None) -> float:
        """
        Get composite search interest score across keywords.
        Useful as a single feature for ML models.
        """
        values = self.get_interpolated_values()
        
        if keywords:
            values = {k: v for k, v in values.items() if k in keywords}
        
        if not values:
            return 0.0
        
        # Weighted average (could be customized)
        return sum(values.values()) / len(values)
    
    def get_stats(self) -> Dict[str, int]:
        """Get fetcher statistics."""
        return self._stats.copy()
    
    async def close(self):
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def main():
    """Example usage of Google Trends fetcher."""
    print("Google Trends Fetcher")
    print("=" * 50)
    
    fetcher = GoogleTrendsFetcher()
    
    print(f"Default keywords: {fetcher.DEFAULT_KEYWORDS[:5]}...")
    print(f"Cache TTL: {fetcher._cache_ttl_seconds}s")
    print(f"Max cache size: {fetcher._max_cache_size}")
    
    # Show interpolation example
    print("\nExponential Decay Interpolator:")
    print(f"  Half-life: {fetcher._interpolator.half_life_minutes} minutes")
    print(f"  Max history: {fetcher._interpolator.max_history_minutes // 60} hours")
    
    print("\nFetcher initialized successfully!")


if __name__ == "__main__":
    main()
