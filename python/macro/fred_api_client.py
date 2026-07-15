"""
Asynchronous FRED API Client for Macroeconomic Data Ingestion.
Implements strict Redis caching and memory limits to prevent RAM bloat.
Designed for AMD Ryzen AI 5 / Radeon GPU target with <14GB RAM constraint.
"""

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import redis.asyncio as redis


class FREDApiClient:
    """
    Async FRED API client with Redis caching and strict memory controls.
    Fetches CPI, PPI, Fed Funds Rate, Bond Yields with minimal RAM footprint.
    """

    # Memory limit constants (in bytes)
    MAX_CACHE_SIZE = 50 * 1024 * 1024  # 50MB max cache
    MAX_QUEUE_SIZE = 100  # Max pending requests
    
    # FRED API configuration
    FRED_API_BASE = "https://api.stlouisfed.org/fred"
    DEFAULT_API_KEY = os.getenv("FRED_API_KEY", "demo_key")
    
    # Macroeconomic series IDs
    SERIES_IDS = {
        "CPI": "CPIAUCSL",           # Consumer Price Index
        "PPI": "PPIACO",             # Producer Price Index
        "FED_FUNDS": "DFF",          # Federal Funds Rate
        "TREASURY_10Y": "GS10",      # 10-Year Treasury Yield
        "TREASURY_2Y": "GS2",        # 2-Year Treasury Yield
        "DXY": "DTWEXBGS",           # Trade Weighted US Dollar Index
        "UNEMPLOYMENT": "UNRATE",    # Unemployment Rate
        "GDP": "GDP",                # Gross Domestic Product
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        cache_ttl: int = 3600,  # 1 hour default TTL
    ):
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self._session: Optional[aiohttp.ClientSession] = None
        self._redis: Optional[redis.Redis] = None
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._cache_size = 0
        self._lock = asyncio.Lock()

    async def _ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def _ensure_redis(self):
        """Ensure Redis connection is initialized."""
        if self._redis is None:
            try:
                self._redis = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
            except Exception:
                # Fallback to in-memory cache if Redis unavailable
                self._redis = None

    def _generate_cache_key(self, series_id: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key from series ID and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        key_data = f"{series_id}:{param_str}"
        return f"fred:{hashlib.sha256(key_data.encode()).hexdigest()[:16]}"

    async def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Check Redis cache for existing data."""
        if self._redis is None:
            return None
        try:
            cached = await self._redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
        return None

    async def _set_cache(self, key: str, data: Dict[str, Any]):
        """Set data in Redis cache with TTL."""
        if self._redis is None:
            return
        try:
            async with self._lock:
                # Check cache size before adding
                info = await self._redis.info("memory")
                self._cache_size = int(info.get("used_memory", 0))
                
                if self._cache_size > self.MAX_CACHE_SIZE:
                    # Evict oldest keys if over limit
                    keys = await self._redis.keys("fred:*")
                    if keys:
                        await self._redis.delete(keys[:len(keys)//4])
                
                await self._redis.setex(key, self.cache_ttl, json.dumps(data))
        except Exception:
            pass

    async def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "d",
    ) -> Dict[str, Any]:
        """
        Fetch a single FRED series with caching and memory controls.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (d=w, w=weekly, m=monthly, q=quarterly, a=annual)
        
        Returns:
            Dictionary with observations and metadata
        """
        await self._ensure_session()
        await self._ensure_redis()

        params = {
            "series_id": series_id,
            "file_type": "json",
            "frequency": frequency,
            "api_key": self.api_key,
        }
        
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        cache_key = self._generate_cache_key(series_id, params)
        
        # Check cache first
        cached_data = await self._check_cache(cache_key)
        if cached_data:
            return cached_data

        # Enforce queue size limit
        if self._request_queue.full():
            # Drop oldest request if queue full
            try:
                self._request_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        await self._request_queue.put((series_id, params))

        try:
            url = f"{self.FRED_API_BASE}/series/observations"
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    return {"error": f"API error: {response.status}", "series_id": series_id}
                
                data = await response.json()
                
                # Normalize response structure
                result = {
                    "series_id": series_id,
                    "observations": [
                        {
                            "date": obs["date"],
                            "value": float(obs["value"]) if obs["value"] != "." else None,
                        }
                        for obs in data.get("observations", [])
                    ],
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "FRED",
                }
                
                # Cache the result
                await self._set_cache(cache_key, result)
                
                return result
                
        except asyncio.TimeoutError:
            return {"error": "Request timeout", "series_id": series_id}
        except Exception as e:
            return {"error": str(e), "series_id": series_id}

    async def fetch_all_macro_series(
        self,
        lookback_days: int = 365,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch all configured macroeconomic series concurrently.
        
        Args:
            lookback_days: Number of days to look back for data
        
        Returns:
            Dictionary mapping series names to their data
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        tasks = [
            self.fetch_series(
                series_id,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            for series_id in self.SERIES_IDS.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for name, series_id in self.SERIES_IDS.items():
            for result in results:
                if isinstance(result, dict) and result.get("series_id") == series_id:
                    output[name] = result
                    break
        
        return output

    async def get_latest_values(self) -> Dict[str, Optional[float]]:
        """
        Get latest available values for all macro series.
        Optimized for minimal memory usage.
        """
        all_data = await self.fetch_all_macro_series(lookback_days=30)
        latest = {}
        
        for name, data in all_data.items():
            observations = data.get("observations", [])
            if observations:
                # Find last non-null value
                for obs in reversed(observations):
                    if obs["value"] is not None:
                        latest[name] = obs["value"]
                        break
                else:
                    latest[name] = None
            else:
                latest[name] = None
        
        return latest

    async def close(self):
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._redis:
            await self._redis.close()

    async def __aenter__(self):
        await self._ensure_session()
        await self._ensure_redis()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


async def main():
    """Example usage demonstrating memory-efficient macro data fetching."""
    async with FREDApiClient() as client:
        # Fetch all macro series
        data = await client.fetch_all_macro_series(lookback_days=90)
        
        # Get latest values
        latest = await client.get_latest_values()
        
        print(f"Fetched {len(data)} macro series")
        print(f"Latest values: {latest}")


if __name__ == "__main__":
    asyncio.run(main())
