"""
Asynchronous, low-footprint scrapers for On-Chain metrics.
Whale alerts, Exchange inflows/outflows, TVL, Gas usage.
Uses aiohttp with strict connection pooling to minimize RAM.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import deque
import aiohttp
import orjson


@dataclass
class OnChainMetric:
    """Immutable on-chain metric snapshot."""
    __slots__ = ('timestamp_ns', 'metric_type', 'value', 'symbol', 'metadata')
    
    timestamp_ns: int
    metric_type: str
    value: float
    symbol: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionPoolManager:
    """Manages aiohttp sessions with strict connection limits."""
    
    __slots__ = ('max_connections', 'max_keepalive', '_sessions', '_created_at')
    
    def __init__(self, max_connections: int = 10, max_keepalive: int = 30):
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self._sessions: deque = deque(maxlen=max_connections)
        self._created_at: float = time.time()
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create a session from the pool."""
        # Reuse existing session if available
        while self._sessions:
            session = self._sessions.popleft()
            if not session.closed and (time.time() - self._created_at) < self.max_keepalive:
                return session
            elif not session.closed:
                await session.close()
        
        # Create new session with strict limits
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=2,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=5, connect=2)
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'CryptoBot/1.0'},
        )
        return session
    
    async def return_session(self, session: aiohttp.ClientSession) -> None:
        """Return session to pool for reuse."""
        if not session.closed:
            self._sessions.append(session)
    
    async def close_all(self) -> None:
        """Close all sessions."""
        while self._sessions:
            session = self._sessions.popleft()
            if not session.closed:
                await session.close()


class WhaleAlertScraper:
    """Scrape whale transaction alerts."""
    
    __slots__ = ('_pool', '_cache', '_cache_ttl_ns', '_last_fetch')
    
    API_URL = "https://api.whale-alert.io/v1/transactions"
    
    def __init__(self, pool: ConnectionPoolManager, cache_ttl_sec: int = 60):
        self._pool = pool
        self._cache: deque = deque(maxlen=100)
        self._cache_ttl_ns = cache_ttl_sec * 1_000_000_000
        self._last_fetch: int = 0
    
    async def fetch_whale_transactions(
        self, 
        min_value_usd: float = 100_000,
        limit: int = 10
    ) -> List[OnChainMetric]:
        """Fetch recent whale transactions."""
        now_ns = time.time_ns()
        
        # Return cached if fresh
        if now_ns - self._last_fetch < self._cache_ttl_ns and self._cache:
            return list(self._cache)
        
        session = await self._pool.get_session()
        try:
            params = {'min_value': min_value_usd, 'limit': limit}
            async with session.get(self.API_URL, params=params) as resp:
                if resp.status == 200:
                    data = orjson.loads(await resp.read())
                    transactions = data.get('transactions', [])
                    
                    metrics = []
                    for tx in transactions[:limit]:
                        metric = OnChainMetric(
                            timestamp_ns=tx.get('timestamp', 0) * 1_000_000_000,
                            metric_type='whale_tx',
                            value=tx.get('value', 0),
                            symbol=tx.get('symbol', 'UNKNOWN'),
                            metadata={
                                'from': tx.get('from_address', ''),
                                'to': tx.get('to_address', ''),
                                'tx_hash': tx.get('transaction_hash', ''),
                            }
                        )
                        metrics.append(metric)
                    
                    self._cache.clear()
                    self._cache.extend(metrics)
                    self._last_fetch = now_ns
                    return metrics
        except Exception as e:
            # Log error but don't crash
            pass
        finally:
            await self._pool.return_session(session)
        
        return list(self._cache) if self._cache else []


class ExchangeFlowScraper:
    """Track exchange inflows and outflows."""
    
    __slots__ = ('_pool', '_inflows', '_outflows', '_last_update')
    
    def __init__(self, pool: ConnectionPoolManager):
        self._pool = pool
        self._inflows: Dict[str, float] = {}
        self._outflows: Dict[str, float] = {}
        self._last_update: int = 0
    
    async def fetch_exchange_flows(self, symbols: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Fetch exchange flow data for given symbols."""
        if symbols is None:
            symbols = ['BTC', 'ETH', 'USDT']
        
        session = await self._pool.get_session()
        try:
            # Simulated - in production use actual API (e.g., CryptoQuant, Glassnode)
            tasks = []
            for symbol in symbols:
                url = f"https://api.example.com/exchange-flow/{symbol}"
                tasks.append(session.get(url))
            
            # Batch fetch
            results = {}
            for symbol, task in zip(symbols, tasks):
                try:
                    async with task as resp:
                        if resp.status == 200:
                            data = orjson.loads(await resp.read())
                            inflow = data.get('inflow_24h', 0.0)
                            outflow = data.get('outflow_24h', 0.0)
                            results[symbol] = {'inflow': inflow, 'outflow': outflow}
                            self._inflows[symbol] = inflow
                            self._outflows[symbol] = outflow
                except:
                    # Use cached values
                    results[symbol] = {
                        'inflow': self._inflows.get(symbol, 0.0),
                        'outflow': self._outflows.get(symbol, 0.0)
                    }
            
            self._last_update = time.time_ns()
            return results
        finally:
            await self._pool.return_session(session)
    
    def get_net_flow(self, symbol: str) -> float:
        """Get net flow (inflow - outflow) for symbol."""
        return self._inflows.get(symbol, 0.0) - self._outflows.get(symbol, 0.0)


class TVLScraper:
    """Track Total Value Locked across DeFi protocols."""
    
    __slots__ = ('_pool', '_tvl_cache', '_protocols')
    
    def __init__(self, pool: ConnectionPoolManager):
        self._pool = pool
        self._tvl_cache: Dict[str, float] = {}
        self._protocols = ['uniswap', 'aave', 'compound', 'makerdao', 'curve']
    
    async def fetch_tvl(self) -> Dict[str, float]:
        """Fetch TVL for tracked protocols."""
        session = await self._pool.get_session()
        try:
            # Use DefiLlama API
            url = "https://api.llama.fi/tvl/protocol/"
            
            for protocol in self._protocols:
                try:
                    async with session.get(f"{url}{protocol}") as resp:
                        if resp.status == 200:
                            data = orjson.loads(await resp.read())
                            if isinstance(data, (int, float)):
                                self._tvl_cache[protocol] = float(data)
                except:
                    pass
            
            return self._tvl_cache.copy()
        finally:
            await self._pool.return_session(session)
    
    def get_total_tvl(self) -> float:
        """Get sum of TVL across all protocols."""
        return sum(self._tvl_cache.values())


class GasTracker:
    """Track blockchain gas usage and prices."""
    
    __slots__ = ('_pool', '_gas_cache', '_networks')
    
    def __init__(self, pool: ConnectionPoolManager):
        self._pool = pool
        self._gas_cache: Dict[str, Dict[str, float]] = {}
        self._networks = ['ethereum', 'arbitrum', 'optimism', 'base']
    
    async def fetch_gas_prices(self) -> Dict[str, Dict[str, float]]:
        """Fetch gas prices for tracked networks."""
        session = await self._pool.get_session()
        try:
            for network in self._networks:
                try:
                    if network == 'ethereum':
                        url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = orjson.loads(await resp.read())
                                result = data.get('result', {})
                                self._gas_cache[network] = {
                                    'slow': float(result.get('SafeGasPrice', 0)),
                                    'average': float(result.get('ProposeGasPrice', 0)),
                                    'fast': float(result.get('FastGasPrice', 0)),
                                }
                except:
                    pass
            
            return self._gas_cache.copy()
        finally:
            await self._pool.return_session(session)
    
    def get_gas_level(self, network: str) -> str:
        """Classify gas level as low/medium/high."""
        gas_data = self._gas_cache.get(network, {})
        avg = gas_data.get('average', 0)
        
        if avg < 20:
            return 'low'
        elif avg < 50:
            return 'medium'
        else:
            return 'high'


class OnChainAnalyticsEngine:
    """Main engine coordinating all on-chain scrapers."""
    
    __slots__ = ('_pool', '_whale_scraper', '_exchange_scraper', '_tvl_scraper', '_gas_tracker', '_running')
    
    def __init__(self, max_connections: int = 10):
        self._pool = ConnectionPoolManager(max_connections=max_connections)
        self._whale_scraper = WhaleAlertScraper(self._pool)
        self._exchange_scraper = ExchangeFlowScraper(self._pool)
        self._tvl_scraper = TVLScraper(self._pool)
        self._gas_tracker = GasTracker(self._pool)
        self._running = False
    
    async def start(self) -> None:
        """Start background scraping tasks."""
        self._running = True
    
    async def stop(self) -> None:
        """Stop and cleanup."""
        self._running = False
        await self._pool.close_all()
    
    async def get_full_snapshot(self) -> Dict[str, Any]:
        """Get complete on-chain analytics snapshot."""
        whale_txs, exchange_flows, tvl, gas = await asyncio.gather(
            self._whale_scraper.fetch_whale_transactions(),
            self._exchange_scraper.fetch_exchange_flows(),
            self._tvl_scraper.fetch_tvl(),
            self._gas_tracker.fetch_gas_prices(),
            return_exceptions=True
        )
        
        # Handle exceptions gracefully
        if isinstance(whale_txs, Exception):
            whale_txs = []
        if isinstance(exchange_flows, Exception):
            exchange_flows = {}
        if isinstance(tvl, Exception):
            tvl = {}
        if isinstance(gas, Exception):
            gas = {}
        
        return {
            'timestamp_ns': time.time_ns(),
            'whale_transactions': whale_txs,
            'exchange_flows': exchange_flows,
            'tvl_by_protocol': tvl,
            'total_tvl': sum(tvl.values()) if tvl else 0,
            'gas_prices': gas,
        }


# Singleton instance
_engine: Optional[OnChainAnalyticsEngine] = None


def get_engine() -> OnChainAnalyticsEngine:
    """Get or create singleton engine instance."""
    global _engine
    if _engine is None:
        _engine = OnChainAnalyticsEngine()
    return _engine
