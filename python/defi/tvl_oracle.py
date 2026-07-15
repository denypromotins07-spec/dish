"""
Lightweight TVL tracker for DeFi protocols with Redis caching.
Prevents redundant network requests and minimizes RAM usage.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib


@dataclass
class ProtocolTVL:
    """TVL data for a single protocol."""
    protocol_name: str
    tvl_usd: float
    tvl_change_24h: float
    timestamp: datetime
    chain: str
    category: str  # "dex", "lending", "derivatives", etc.


@dataclass
class TokenTVL:
    """TVL breakdown by token for a protocol."""
    protocol_name: str
    token_balances: Dict[str, float]  # token -> amount
    token_values_usd: Dict[str, float]  # token -> USD value
    total_tvl_usd: float
    timestamp: datetime


class TvlOracle:
    """
    Lightweight TVL tracker with strict Redis caching.
    Prevents redundant RPC calls and saves RAM.
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 300,  # 5 minutes default
        max_protocols: int = 100,
        redis_client=None,
    ):
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_protocols = max_protocols
        self.redis = redis_client
        
        # In-memory cache (LRU-style)
        self._tvl_cache: Dict[str, tuple] = {}  # key -> (value, expiry)
        self._access_order: List[str] = []
        
        # Known protocols to track
        self.tracked_protocols = {
            "uniswap": {"chains": ["ethereum", "arbitrum", "optimism"], "category": "dex"},
            "aave": {"chains": ["ethereum", "polygon", "avalanche"], "category": "lending"},
            "curve": {"chains": ["ethereum", "arbitrum", "optimism"], "category": "dex"},
            "compound": {"chains": ["ethereum"], "category": "lending"},
            "makerdao": {"chains": ["ethereum"], "category": "lending"},
            "lido": {"chains": ["ethereum"], "category": "staking"},
            "convex": {"chains": ["ethereum"], "category": "yield"},
            "balancer": {"chains": ["ethereum", "arbitrum", "polygon"], "category": "dex"},
            "gmx": {"chains": ["arbitrum", "avalanche"], "category": "derivatives"},
            "pancakeswap": {"chains": ["bsc"], "category": "dex"},
        }
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._rpc_calls = 0

    async def get_protocol_tvl(
        self,
        protocol_name: str,
        chain: str = "ethereum",
        force_refresh: bool = False,
    ) -> Optional[ProtocolTVL]:
        """Get TVL for a protocol, using cache if available."""
        cache_key = f"tvl:{protocol_name}:{chain}"
        
        # Check cache first
        if not force_refresh:
            cached = await self._get_from_cache(cache_key)
            if cached:
                return cached
        
        # Fetch fresh data
        tvl_data = await self._fetch_protocol_tvl(protocol_name, chain)
        
        if tvl_data:
            await self._set_cache(cache_key, tvl_data)
        
        return tvl_data

    async def get_all_protocols_tvl(
        self,
        chains: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[ProtocolTVL]:
        """Get TVL for all tracked protocols."""
        chains = chains or ["ethereum"]
        results = []
        
        for protocol_name, config in self.tracked_protocols.items():
            if categories and config["category"] not in categories:
                continue
            
            for chain in config["chains"]:
                if chain not in chains:
                    continue
                
                tvl = await self.get_protocol_tvl(protocol_name, chain)
                if tvl:
                    results.append(tvl)
        
        return sorted(results, key=lambda x: x.tvl_usd, reverse=True)

    async def _fetch_protocol_tvl(
        self,
        protocol_name: str,
        chain: str,
    ) -> Optional[ProtocolTVL]:
        """Fetch TVL from subgraph or RPC (mock implementation)."""
        self._rpc_calls += 1
        
        # In production, this would:
        # 1. Query The Graph subgraph for the protocol
        # 2. Or make direct RPC calls to get contract balances
        # 3. Calculate USD values using price oracles
        
        # Mock data for demonstration
        mock_tvl_data = {
            "uniswap": {"ethereum": 4500000000, "arbitrum": 150000000, "optimism": 80000000},
            "aave": {"ethereum": 5200000000, "polygon": 350000000, "avalanche": 280000000},
            "curve": {"ethereum": 3800000000, "arbitrum": 120000000, "optimism": 60000000},
            "lido": {"ethereum": 14000000000},
            "gmx": {"arbitrum": 450000000, "avalanche": 180000000},
        }
        
        protocol_data = mock_tvl_data.get(protocol_name, {})
        tvl_usd = protocol_data.get(chain, 0)
        
        if tvl_usd == 0:
            return None
        
        # Mock 24h change
        import random
        change_24h = random.uniform(-0.15, 0.15)
        
        config = self.tracked_protocols.get(protocol_name, {})
        
        return ProtocolTVL(
            protocol_name=protocol_name,
            tvl_usd=tvl_usd,
            tvl_change_24h=change_24h,
            timestamp=datetime.utcnow(),
            chain=chain,
            category=config.get("category", "unknown"),
        )

    async def get_token_breakdown(
        self,
        protocol_name: str,
        chain: str = "ethereum",
    ) -> Optional[TokenTVL]:
        """Get TVL breakdown by token for a protocol."""
        cache_key = f"tvl_tokens:{protocol_name}:{chain}"
        
        cached = await self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Fetch token breakdown (mock)
        token_data = await self._fetch_token_breakdown(protocol_name, chain)
        
        if token_data:
            await self._set_cache(cache_key, token_data)
        
        return token_data

    async def _fetch_token_breakdown(
        self,
        protocol_name: str,
        chain: str,
    ) -> Optional[TokenTVL]:
        """Fetch token-level TVL breakdown (mock)."""
        # Mock token balances
        mock_balances = {
            "uniswap": {
                "ETH": 500000,
                "USDC": 800000000,
                "USDT": 600000000,
                "WBTC": 15000,
            },
            "aave": {
                "ETH": 300000,
                "USDC": 1500000000,
                "DAI": 800000000,
                "WBTC": 25000,
            },
        }
        
        prices = {"ETH": 2000, "USDC": 1, "USDT": 1, "DAI": 1, "WBTC": 30000}
        
        balances = mock_balances.get(protocol_name, {})
        if not balances:
            return None
        
        token_values = {}
        for token, amount in balances.items():
            price = prices.get(token, 0)
            token_values[token] = amount * price
        
        total_tvl = sum(token_values.values())
        
        return TokenTVL(
            protocol_name=protocol_name,
            token_balances=balances,
            token_values_usd=token_values,
            total_tvl_usd=total_tvl,
            timestamp=datetime.utcnow(),
        )

    async def _get_from_cache(self, key: str) -> Optional[object]:
        """Get value from cache if not expired."""
        # Try Redis first if configured
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached:
                    self._cache_hits += 1
                    return cached
            except Exception:
                pass
        
        # Check in-memory cache
        if key in self._tvl_cache:
            value, expiry = self._tvl_cache[key]
            if datetime.utcnow() < expiry:
                # Update access order for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                self._cache_hits += 1
                return value
            else:
                # Expired, remove it
                del self._tvl_cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
        
        self._cache_misses += 1
        return None

    async def _set_cache(self, key: str, value: object):
        """Set value in cache with TTL."""
        expiry = datetime.utcnow() + self.cache_ttl
        
        # Enforce max size with LRU eviction
        while len(self._tvl_cache) >= self.max_protocols and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._tvl_cache:
                del self._tvl_cache[oldest_key]
        
        # Store in memory
        self._tvl_cache[key] = (value, expiry)
        self._access_order.append(key)
        
        # Also store in Redis if configured
        if self.redis:
            try:
                await self.redis.setex(key, int(self.cache_ttl.total_seconds()), value)
            except Exception:
                pass

    def get_total_defi_tvl(
        self,
        chains: Optional[List[str]] = None,
    ) -> float:
        """Get aggregate TVL across all tracked protocols."""
        total = 0.0
        for key, (value, _) in self._tvl_cache.items():
            if isinstance(value, ProtocolTVL):
                if chains is None or value.chain in chains:
                    total += value.tvl_usd
        return total

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._tvl_cache),
            "max_size": self.max_protocols,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "rpc_calls_made": self._rpc_calls,
            "protocols_tracked": len(self.tracked_protocols),
        }

    def clear_cache(self):
        """Clear all cached data."""
        self._tvl_cache.clear()
        self._access_order.clear()


async def main():
    """Example usage of TvlOracle."""
    oracle = TvlOracle(cache_ttl_seconds=60)
    
    # Get TVL for specific protocols
    uniswap_tvl = await oracle.get_protocol_tvl("uniswap", "ethereum")
    if uniswap_tvl:
        print(f"Uniswap TVL: ${uniswap_tvl.tvl_usd:,.0f}")
        print(f"24h Change: {uniswap_tvl.tvl_change_24h:.2%}")
    
    # Get all DEX TVLs
    dex_tvls = await oracle.get_all_protocols_tvl(
        chains=["ethereum"],
        categories=["dex"],
    )
    
    print("\nDEX TVL Rankings:")
    for i, tvl in enumerate(dex_tvls[:5], 1):
        print(f"  {i}. {tvl.protocol_name}: ${tvl.tvl_usd:,.0f}")
    
    # Get token breakdown
    breakdown = await oracle.get_token_breakdown("aave", "ethereum")
    if breakdown:
        print(f"\nAave Token Breakdown:")
        for token, value in breakdown.token_values_usd.items():
            print(f"  {token}: ${value:,.0f}")
    
    # Cache stats
    stats = oracle.get_cache_stats()
    print(f"\nCache Stats:")
    print(f"  Size: {stats['cache_size']}/{stats['max_size']}")
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    print(f"  RPC Calls Saved: {stats['hits']}")


if __name__ == "__main__":
    asyncio.run(main())
