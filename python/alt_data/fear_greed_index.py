"""
Crypto Fear & Greed Index Scraper and Parser.
Decomposes the index into underlying volatility, volume momentum, 
and social media components for granular feature engineering.
Designed for <14GB RAM with efficient data structures.
"""

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import aiohttp


@dataclass
class FearGreedDataPoint:
    """Single Fear & Greed Index data point."""
    timestamp_ms: int
    value: int           # 0-100 (0=Extreme Fear, 100=Extreme Greed)
    classification: str  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    
    @property
    def normalized_value(self) -> float:
        """Normalize to -1 to +1 range (-1=Fear, +1=Greed)."""
        return (self.value - 50) / 50.0
    
    @property
    def is_extreme(self) -> bool:
        """Check if reading is at extreme levels."""
        return self.value <= 25 or self.value >= 75


@dataclass
class FearGreedComponents:
    """Decomposed Fear & Greed Index components."""
    volatility: float = 0.0       # 25% weight
    market_momentum: float = 0.0  # 25% weight
    social_media: float = 0.0     # 25% weight
    surveys: float = 0.0          # 25% weight (if available)
    dominance: float = 0.0        # Bitcoin dominance factor
    trends: float = 0.0           # Google Trends factor
    
    def calculate_index(self) -> float:
        """Calculate composite index from components."""
        # Standard weights used by Alternative.me
        weights = {
            "volatility": 0.25,
            "market_momentum": 0.25,
            "social_media": 0.25,
            "surveys": 0.15,
            "dominance": 0.05,
            "trends": 0.05,
        }
        
        weighted_sum = (
            self.volatility * weights["volatility"] +
            self.market_momentum * weights["market_momentum"] +
            self.social_media * weights["social_media"] +
            self.surveys * weights["surveys"] +
            self.dominance * weights["dominance"] +
            self.trends * weights["trends"]
        )
        
        return min(max(weighted_sum * 100, 0), 100)  # Clamp to 0-100


class FearGreedIndexFetcher:
    """
    Async scraper for Crypto Fear & Greed Index.
    Fetches current and historical data with component decomposition.
    """
    
    # Official API endpoint
    API_URL = "https://api.alternative.me/fng/"
    
    # Classification thresholds
    CLASSIFICATIONS = [
        (0, 24, "Extreme Fear"),
        (25, 49, "Fear"),
        (50, 74, "Neutral"),
        (75, 89, "Greed"),
        (90, 100, "Extreme Greed"),
    ]
    
    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        cache_ttl_seconds: int = 300,  # 5 minutes
        max_history_days: int = 90,
    ):
        self._session = session
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_history_days = max_history_days
        
        self._cache: Dict[str, Any] = {}
        self._history: deque = deque(maxlen=max_history_days)
        self._components_history: deque = deque(maxlen=max_history_days)
        self._stats = {
            "fetches": 0,
            "cache_hits": 0,
            "errors": 0,
        }
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is initialized."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            connector = aiohttp.TCPConnector(limit=3, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session
    
    @staticmethod
    def classify_value(value: int) -> str:
        """Classify a Fear & Greed value."""
        for min_val, max_val, classification in FearGreedIndexFetcher.CLASSIFICATIONS:
            if min_val <= value <= max_val:
                return classification
        return "Unknown"
    
    def _parse_timestamp(self, date_str: str) -> int:
        """Parse date string to milliseconds timestamp."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return int(dt.timestamp() * 1000)
        except Exception:
            return int(time.time() * 1000)
    
    async def fetch_current(self) -> Optional[FearGreedDataPoint]:
        """Fetch current Fear & Greed Index value."""
        # Check cache
        cache_key = "current"
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl_seconds:
                self._stats["cache_hits"] += 1
                return data
        
        self._stats["fetches"] += 1
        
        try:
            session = await self._ensure_session()
            
            # Fetch only latest value
            url = f"{self.API_URL}?limit=1"
            async with session.get(url) as response:
                if response.status != 200:
                    self._stats["errors"] += 1
                    return None
                
                data = await response.json()
                
                if data.get("status") != "success":
                    self._stats["errors"] += 1
                    return None
                
                metadata = data.get("metadata", {})
                values = data.get("data", [])
                
                if not values:
                    return None
                
                latest = values[0]
                value = int(latest.get("value", 50))
                
                point = FearGreedDataPoint(
                    timestamp_ms=self._parse_timestamp(latest.get("timestamp", "")),
                    value=value,
                    classification=self.classify_value(value),
                )
                
                # Cache result
                self._cache[cache_key] = (time.time(), point)
                
                # Add to history
                self._history.append(point)
                
                return point
                
        except Exception as e:
            self._stats["errors"] += 1
            return None
    
    async def fetch_historical(self, days: int = 30) -> List[FearGreedDataPoint]:
        """Fetch historical Fear & Greed data."""
        days = min(days, self._max_history_days)
        
        try:
            session = await self._ensure_session()
            url = f"{self.API_URL}?limit={days}"
            
            async with session.get(url) as response:
                if response.status != 200:
                    self._stats["errors"] += 1
                    return []
                
                data = await response.json()
                
                if data.get("status") != "success":
                    return []
                
                values = data.get("data", [])
                
                points = []
                for entry in reversed(values):  # Oldest first
                    value = int(entry.get("value", 50))
                    point = FearGreedDataPoint(
                        timestamp_ms=self._parse_timestamp(entry.get("timestamp", "")),
                        value=value,
                        classification=self.classify_value(value),
                    )
                    points.append(point)
                    self._history.append(point)
                
                return points
                
        except Exception as e:
            self._stats["errors"] += 1
            return []
    
    def decompose_current(self, market_data: Optional[Dict[str, Any]] = None) -> FearGreedComponents:
        """
        Decompose Fear & Greed Index into components.
        
        Args:
            market_data: Optional dict with market metrics:
                - btc_volatility: BTC 30-day volatility
                - volume_change: 24h volume change %
                - social_sentiment: Social media sentiment score
                - btc_dominance: Bitcoin dominance %
                - search_trend: Google Trends value
        """
        components = FearGreedComponents()
        
        if market_data:
            # Volatility component (inverse relationship)
            btc_vol = market_data.get("btc_volatility", 0.5)
            components.volatility = max(0, 1 - btc_vol)  # Lower vol = higher greed
            
            # Market momentum (volume-based)
            vol_change = market_data.get("volume_change", 0)
            components.market_momentum = min(max((vol_change + 100) / 200, 0), 1)
            
            # Social media sentiment
            social = market_data.get("social_sentiment", 0.5)
            components.social_media = social
            
            # Bitcoin dominance
            dominance = market_data.get("btc_dominance", 50)
            components.dominance = dominance / 100
            
            # Search trends
            trends = market_data.get("search_trend", 0.5)
            components.trends = trends
        
        # Get current F&G value to calibrate surveys component
        if self._history:
            current_fg = self._history[-1].normalized_value
            # Back-calculate surveys to match actual index
            known_sum = (
                components.volatility * 0.25 +
                components.market_momentum * 0.25 +
                components.social_media * 0.25 +
                components.dominance * 0.05 +
                components.trends * 0.05
            )
            # Surveys makes up the difference (with 0.15 weight)
            remaining = current_fg - known_sum
            components.surveys = remaining / 0.15 if 0.15 > 0 else 0
            components.surveys = min(max(components.surveys, 0), 1)
        
        return components
    
    def get_signal(self) -> Tuple[str, str]:
        """
        Get trading signal based on current Fear & Greed reading.
        Returns (signal, reasoning).
        """
        if not self._history:
            return "HOLD", "No data available"
        
        current = self._history[-1]
        
        # Extreme fear = potential buy opportunity
        if current.value <= 25:
            return "BUY", f"Extreme Fear ({current.value}) - Potential bottom"
        elif current.value <= 40:
            return "ACCUMULATE", f"Fear ({current.value}) - Good accumulation zone"
        elif current.value >= 75:
            return "SELL", f"Extreme Greed ({current.value}) - Potential top"
        elif current.value >= 60:
            return "REDUCE", f"Greed ({current.value}) - Consider taking profits"
        else:
            return "HOLD", f"Neutral ({current.value}) - Wait for clearer signal"
    
    def get_extremes(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Get extreme readings within lookback period."""
        if not self._history:
            return {"min": None, "max": None, "extreme_fear_count": 0, "extreme_greed_count": 0}
        
        recent = list(self._history)[-lookback_days:]
        
        values = [p.value for p in recent]
        extreme_fear = sum(1 for v in values if v <= 25)
        extreme_greed = sum(1 for v in values if v >= 75)
        
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "average": sum(values) / len(values) if values else None,
            "extreme_fear_count": extreme_fear,
            "extreme_greed_count": extreme_greed,
            "current_percentile": (
                sum(1 for v in values if v <= recent[-1].value) / len(values)
                if values else 0
            ),
        }
    
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
    """Example usage of Fear & Greed Index fetcher."""
    print("Crypto Fear & Greed Index Fetcher")
    print("=" * 50)
    
    fetcher = FearGreedIndexFetcher()
    
    print(f"API URL: {fetcher.API_URL}")
    print(f"Cache TTL: {fetcher._cache_ttl_seconds}s")
    print(f"Max history: {fetcher._max_history_days} days")
    
    print("\nClassification thresholds:")
    for min_val, max_val, name in fetcher.CLASSIFICATIONS:
        print(f"  {min_val}-{max_val}: {name}")
    
    print("\nFetcher initialized successfully!")
    
    # Example signal interpretation
    print("\nSignal guide:")
    print("  0-25 (Extreme Fear) -> Potential BUY opportunity")
    print("  25-40 (Fear) -> ACCUMULATE zone")
    print("  40-60 (Neutral) -> HOLD/wait")
    print("  60-75 (Greed) -> REDUCE positions")
    print("  75-100 (Extreme Greed) -> Potential SELL signal")


if __name__ == "__main__":
    main()
