"""
Macro data fetchers and lightweight Sentiment analysis.
CPI, PPI, DXY, Bond yields, Fear & Greed, News/X API polling.
Runs on isolated background threads to avoid blocking the trading loop.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import orjson


@dataclass
class MacroDataPoint:
    """Immutable macroeconomic data point."""
    __slots__ = ('timestamp_ns', 'metric_type', 'value', 'previous', 'forecast', 'impact')
    
    timestamp_ns: int
    metric_type: str
    value: float
    previous: float
    forecast: float
    impact: str  # 'low', 'medium', 'high'


@dataclass  
class SentimentReading:
    """Sentiment indicator snapshot."""
    __slots__ = ('timestamp_ns', 'fear_greed', 'source', 'metadata')
    
    timestamp_ns: int
    fear_greed: int  # 0-100
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MacroDataFetcher:
    """Fetch macroeconomic indicators from various sources."""
    
    __slots__ = ('_session', '_cache', '_cache_ttl_ns', '_executor')
    
    # Economic calendar API endpoints (production would use actual APIs)
    ENDPOINTS = {
        'cpi': 'https://api.example.com/economic/cpi',
        'ppi': 'https://api.example.com/economic/ppi',
        'dxy': 'https://api.example.com/forex/dxy',
        'treasury_10y': 'https://api.example.com/bonds/us10y',
        'treasury_2y': 'https://api.example.com/bonds/us2y',
        'unemployment': 'https://api.example.com/economic/unemployment',
        'gdp': 'https://api.example.com/economic/gdp',
    }
    
    def __init__(self, session: aiohttp.ClientSession, cache_ttl_sec: int = 300):
        self._session = session
        self._cache: Dict[str, MacroDataPoint] = {}
        self._cache_ttl_ns = cache_ttl_sec * 1_000_000_000
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='macro_fetcher')
    
    async def fetch_indicator(self, indicator: str) -> Optional[MacroDataPoint]:
        """Fetch a specific economic indicator."""
        now_ns = time.time_ns()
        
        # Check cache first
        if indicator in self._cache:
            cached = self._cache[indicator]
            if now_ns - cached.timestamp_ns < self._cache_ttl_ns:
                return cached
        
        url = self.ENDPOINTS.get(indicator)
        if not url:
            return None
        
        try:
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    data = orjson.loads(await resp.read())
                    
                    point = MacroDataPoint(
                        timestamp_ns=now_ns,
                        metric_type=indicator,
                        value=float(data.get('value', 0)),
                        previous=float(data.get('previous', 0)),
                        forecast=float(data.get('forecast', 0)),
                        impact=data.get('impact', 'medium'),
                    )
                    
                    self._cache[indicator] = point
                    return point
        except Exception:
            pass
        
        return self._cache.get(indicator)
    
    async def fetch_all_indicators(self) -> Dict[str, MacroDataPoint]:
        """Fetch all tracked indicators concurrently."""
        tasks = [self.fetch_indicator(ind) for ind in self.ENDPOINTS.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for indicator, result in zip(self.ENDPOINTS.keys(), results):
            if isinstance(result, MacroDataPoint):
                output[indicator] = result
        
        return output
    
    def get_dxy(self) -> Optional[float]:
        """Get current DXY value."""
        point = self._cache.get('dxy')
        return point.value if point else None
    
    def get_treasury_yield(self, tenor: str = '10y') -> Optional[float]:
        """Get treasury yield for specified tenor."""
        key = f'treasury_{tenor}'
        point = self._cache.get(key)
        return point.value if point else None
    
    def get_inflation_surprise(self) -> float:
        """Calculate CPI surprise vs forecast."""
        cpi = self._cache.get('cpi')
        if not cpi:
            return 0.0
        if cpi.forecast == 0:
            return 0.0
        return (cpi.value - cpi.forecast) / cpi.forecast * 100


class SentimentAnalyzer:
    """Lightweight sentiment analysis from multiple sources."""
    
    __slots__ = ('_session', '_fear_greed_cache', '_news_cache', '_executor')
    
    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    
    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        self._fear_greed_cache: deque = deque(maxlen=100)
        self._news_cache: deque = deque(maxlen=50)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='sentiment_worker')
    
    async def fetch_fear_greed_index(self) -> Optional[SentimentReading]:
        """Fetch Crypto Fear & Greed Index."""
        try:
            async with self._session.get(
                self.FEAR_GREED_URL,
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status == 200:
                    data = orjson.loads(await resp.read())
                    value = int(data.get('data', [{}])[0].get('value', 50))
                    
                    reading = SentimentReading(
                        timestamp_ns=time.time_ns(),
                        fear_greed=value,
                        source='alternative.me',
                        metadata={
                            'classification': self._classify_sentiment(value),
                            'timestamp': data.get('data', [{}])[0].get('timestamp', ''),
                        }
                    )
                    
                    self._fear_greed_cache.append(reading)
                    return reading
        except Exception:
            pass
        
        # Return last cached value
        return self._fear_greed_cache[-1] if self._fear_greed_cache else None
    
    async def poll_news_sentiment(self, keywords: List[str] = None) -> Dict[str, float]:
        """Poll news sources for sentiment on given keywords."""
        if keywords is None:
            keywords = ['bitcoin', 'ethereum', 'crypto', 'fed', 'sec']
        
        # Simulated - in production use actual news APIs (CryptoPanic, NewsAPI)
        sentiment_scores = {}
        
        for keyword in keywords:
            # Placeholder for actual API calls
            score = 0.5  # Neutral default
            sentiment_scores[keyword] = score
        
        return sentiment_scores
    
    def _classify_sentiment(self, value: int) -> str:
        """Classify fear & greed value."""
        if value <= 25:
            return 'extreme_fear'
        elif value <= 45:
            return 'fear'
        elif value <= 55:
            return 'neutral'
        elif value <= 75:
            return 'greed'
        else:
            return 'extreme_greed'
    
    def get_sentiment_trend(self, window: int = 10) -> str:
        """Determine sentiment trend over recent readings."""
        if len(self._fear_greed_cache) < window:
            return 'insufficient_data'
        
        values = [r.fear_greed for r in list(self._fear_greed_cache)[-window:]]
        
        # Simple linear regression slope
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = sum(i * i for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 'flat'
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        if slope > 0.5:
            return 'improving'
        elif slope < -0.5:
            return 'worsening'
        else:
            return 'stable'
    
    def get_average_sentiment(self, window: int = 10) -> float:
        """Get average fear & greed over recent window."""
        if not self._fear_greed_cache:
            return 50.0
        
        values = [r.fear_greed for r in list(self._fear_greed_cache)[-window:]]
        return sum(values) / len(values)


class MacroSentimentEngine:
    """Combined macro and sentiment analysis engine."""
    
    __slots__ = ('_session', '_macro_fetcher', '_sentiment_analyzer', '_running', '_update_interval')
    
    def __init__(self, update_interval_sec: float = 60.0):
        self._session: Optional[aiohttp.ClientSession] = None
        self._macro_fetcher: Optional[MacroDataFetcher] = None
        self._sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self._running = False
        self._update_interval = update_interval_sec
        self._callbacks: List[Callable] = []
    
    async def start(self) -> None:
        """Start the engine and background polling."""
        connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=5)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        self._macro_fetcher = MacroDataFetcher(self._session)
        self._sentiment_analyzer = SentimentAnalyzer(self._session)
        
        self._running = True
        asyncio.create_task(self._background_polling())
    
    async def stop(self) -> None:
        """Stop the engine."""
        self._running = False
        if self._session:
            await self._session.close()
    
    async def _background_polling(self) -> None:
        """Background task to periodically update data."""
        while self._running:
            try:
                await asyncio.gather(
                    self._macro_fetcher.fetch_all_indicators(),
                    self._sentiment_analyzer.fetch_fear_greed_index(),
                    return_exceptions=True
                )
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback()
                    except Exception:
                        pass
                
            except Exception:
                pass
            
            await asyncio.sleep(self._update_interval)
    
    def register_callback(self, callback: Callable) -> None:
        """Register callback for data updates."""
        self._callbacks.append(callback)
    
    def get_macro_snapshot(self) -> Dict[str, Any]:
        """Get current macro data snapshot."""
        if not self._macro_fetcher:
            return {}
        
        return {
            'dxy': self._macro_fetcher.get_dxy(),
            'treasury_10y': self._macro_fetcher.get_treasury_yield('10y'),
            'treasury_2y': self._macro_fetcher.get_treasury_yield('2y'),
            'inflation_surprise': self._macro_fetcher.get_inflation_surprise(),
            'timestamp_ns': time.time_ns(),
        }
    
    def get_sentiment_snapshot(self) -> Dict[str, Any]:
        """Get current sentiment snapshot."""
        if not self._sentiment_analyzer:
            return {}
        
        return {
            'fear_greed': self._sentiment_analyzer.get_average_sentiment(),
            'trend': self._sentiment_analyzer.get_sentiment_trend(),
            'timestamp_ns': time.time_ns(),
        }
    
    def get_combined_signal(self) -> Dict[str, Any]:
        """Generate combined macro + sentiment signal."""
        macro = self.get_macro_snapshot()
        sentiment = self.get_sentiment_snapshot()
        
        # Simple scoring logic
        score = 0.0
        
        # DXY inverse relationship (high DXY = bearish for crypto)
        dxy = macro.get('dxy')
        if dxy:
            if dxy > 105:
                score -= 0.3
            elif dxy < 95:
                score += 0.3
        
        # Treasury yields (high yields = risk-off)
        yield_10y = macro.get('treasury_10y')
        if yield_10y:
            if yield_10y > 4.5:
                score -= 0.2
            elif yield_10y < 3.0:
                score += 0.2
        
        # Sentiment
        fg = sentiment.get('fear_greed', 50)
        if fg < 30:
            score += 0.3  # Contrarian buy
        elif fg > 70:
            score -= 0.3  # Contrarian sell
        
        return {
            'score': score,
            'signal': 'bullish' if score > 0.2 else ('bearish' if score < -0.2 else 'neutral'),
            'macro': macro,
            'sentiment': sentiment,
            'timestamp_ns': time.time_ns(),
        }


# Singleton instance
_engine: Optional[MacroSentimentEngine] = None


def get_engine() -> MacroSentimentEngine:
    """Get or create singleton engine instance."""
    global _engine
    if _engine is None:
        _engine = MacroSentimentEngine()
    return _engine
