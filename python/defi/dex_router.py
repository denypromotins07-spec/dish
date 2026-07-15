"""
DEX aggregator logic for optimal multi-hop swap routing.
Finds best routes across Uniswap, Curve, Balancer with gas/slippage consideration.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import heapq


class DEXProtocol(Enum):
    """Supported DEX protocols."""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    SUSHISWAP = "sushiswap"
    CURVE = "curve"
    BALANCER = "balancer"
    PANCAKESWAP = "pancakeswap"


@dataclass
class Pool:
    """Represents a liquidity pool on a DEX."""
    pool_id: str
    protocol: DEXProtocol
    token0: str
    token1: str
    reserve0: float
    reserve1: float
    fee_bps: float
    tvl_usd: float
    is_stablecoin_pair: bool = False
    curve_k: float = 0.0  # For Curve pools


@dataclass
class RouteHop:
    """Single hop in a swap route."""
    pool_id: str
    protocol: DEXProtocol
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    price_impact: float
    fee_usd: float


@dataclass
class SwapRoute:
    """Complete swap route from source to destination token."""
    hops: List[RouteHop]
    token_path: List[str]
    total_amount_in: float
    total_amount_out: float
    total_price_impact: float
    total_fees_usd: float
    estimated_gas_cost_usd: float
    net_output_usd: float
    execution_time_ms: float


@dataclass
class GasEstimate:
    """Gas cost estimates for different operations."""
    base_gas: int
    per_hop_gas: int
    current_gas_price_gwei: float
    eth_price_usd: float

    def estimate_cost_usd(self, num_hops: int) -> float:
        """Estimate total gas cost in USD."""
        total_gas = self.base_gas + (num_hops * self.per_hop_gas)
        gas_cost_eth = (total_gas * self.current_gas_price_gwei) / 1e9
        return gas_cost_eth * self.eth_price_usd


class DexRouter:
    """
    DEX aggregator that finds optimal multi-hop swap routes.
    Considers real-time gas costs, slippage, and liquidity depth.
    """

    def __init__(
        self,
        pools: Optional[List[Pool]] = None,
        gas_estimate: Optional[GasEstimate] = None,
        max_hops: int = 3,
        min_liquidity_usd: float = 10_000,
        slippage_tolerance: float = 0.01,
    ):
        self.pools = pools or []
        self.gas_estimate = gas_estimate or GasEstimate(
            base_gas=150000,
            per_hop_gas=50000,
            current_gas_price_gwei=30,
            eth_price_usd=2000,
        )
        self.max_hops = max_hops
        self.min_liquidity_usd = min_liquidity_usd
        self.slippage_tolerance = slippage_tolerance
        
        # Build lookup indices
        self._pools_by_token: Dict[str, List[Pool]] = {}
        self._pools_by_id: Dict[str, Pool] = {}
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Rebuild lookup indices after pool changes."""
        self._pools_by_token.clear()
        self._pools_by_id.clear()
        
        for pool in self.pools:
            self._pools_by_id[pool.pool_id] = pool
            
            if pool.token0 not in self._pools_by_token:
                self._pools_by_token[pool.token0] = []
            self._pools_by_token[pool.token0].append(pool)
            
            if pool.token1 not in self._pools_by_token:
                self._pools_by_token[pool.token1] = []
            self._pools_by_token[pool.token1].append(pool)

    def add_pool(self, pool: Pool):
        """Add a new pool to the router."""
        self.pools.append(pool)
        self._rebuild_indices()

    def remove_pool(self, pool_id: str):
        """Remove a pool from the router."""
        self.pools = [p for p in self.pools if p.pool_id != pool_id]
        self._rebuild_indices()

    async def find_best_route(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        token_prices_usd: Dict[str, float],
    ) -> Optional[SwapRoute]:
        """
        Find the optimal swap route considering output, fees, and gas.
        """
        start_time = datetime.utcnow()
        
        if token_in == token_out:
            return None
        
        # Find all possible routes using DFS
        routes = await self._find_all_routes(
            token_in, token_out, amount_in, [], [token_in], set()
        )
        
        if not routes:
            return None
        
        # Calculate net output for each route
        evaluated_routes = []
        for route in routes:
            net_output = self._evaluate_route(route, token_prices_usd)
            evaluated_routes.append(net_output)
        
        # Sort by net output (highest first)
        evaluated_routes.sort(key=lambda r: r.net_output_usd, reverse=True)
        
        best_route = evaluated_routes[0]
        best_route.execution_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000
        
        return best_route

    async def _find_all_routes(
        self,
        current_token: str,
        target: str,
        amount_in: float,
        hop_ids: List[str],
        token_path: List[str],
        visited_pools: Set[str],
    ) -> List[SwapRoute]:
        """DFS to find all possible routes."""
        if len(hop_ids) >= self.max_hops:
            return []
        
        routes = []
        
        # Get pools containing current token
        candidate_pools = self._pools_by_token.get(current_token, [])
        
        for pool in candidate_pools:
            if pool.pool_id in visited_pools:
                continue
            
            if pool.tvl_usd < self.min_liquidity_usd:
                continue
            
            # Determine output token
            next_token = pool.token1 if pool.token0 == current_token else pool.token0
            
            # Skip if already in path (prevent cycles)
            if next_token in token_path:
                continue
            
            # Calculate output for this hop
            amount_out, price_impact, fee = self._calculate_swap_output(
                pool, current_token, amount_in
            )
            
            if amount_out <= 0:
                continue
            
            new_hop = RouteHop(
                pool_id=pool.pool_id,
                protocol=pool.protocol,
                token_in=current_token,
                token_out=next_token,
                amount_in=amount_in,
                amount_out=amount_out,
                price_impact=price_impact,
                fee_usd=fee,
            )
            
            new_hops = hop_ids + [new_hop]
            new_path = token_path + [next_token]
            new_visited = visited_pools | {pool.pool_id}
            
            if next_token == target:
                # Found complete route
                route = SwapRoute(
                    hops=new_hops,
                    token_path=new_path,
                    total_amount_in=new_hops[0].amount_in,
                    total_amount_out=new_hops[-1].amount_out,
                    total_price_impact=sum(h.price_impact for h in new_hops),
                    total_fees_usd=sum(h.fee_usd for h in new_hops),
                    estimated_gas_cost_usd=0,
                    net_output_usd=0,
                    execution_time_ms=0,
                )
                routes.append(route)
            else:
                # Continue searching
                sub_routes = await self._find_all_routes(
                    next_token, target, amount_out,
                    new_hops, new_path, new_visited
                )
                routes.extend(sub_routes)
        
        return routes

    def _calculate_swap_output(
        self,
        pool: Pool,
        token_in: str,
        amount_in: float,
    ) -> Tuple[float, float, float]:
        """Calculate output amount, price impact, and fee for a swap."""
        is_token0_in = pool.token0 == token_in
        
        if is_token0_in:
            reserve_in = pool.reserve0
            reserve_out = pool.reserve1
        else:
            reserve_in = pool.reserve1
            reserve_out = pool.reserve0
        
        if reserve_in == 0 or reserve_out == 0:
            return 0, 1.0, 0
        
        # Apply fee
        fee_multiplier = 1 - (pool.fee_bps / 10000)
        amount_in_after_fee = amount_in * fee_multiplier
        
        # Constant product formula: x * y = k
        # (reserve_in + amount_in_after_fee) * (reserve_out - amount_out) = k
        # amount_out = reserve_out - k / (reserve_in + amount_in_after_fee)
        # amount_out = reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)
        
        amount_out = (reserve_out * amount_in_after_fee) / (reserve_in + amount_in_after_fee)
        
        # Calculate price impact
        spot_price = reserve_out / reserve_in if reserve_in > 0 else 0
        exec_price = amount_out / amount_in if amount_in > 0 else 0
        price_impact = 1 - (exec_price / spot_price) if spot_price > 0 else 0
        
        # Calculate fee in USD (approximate)
        fee = amount_in * (pool.fee_bps / 10000)
        
        return max(0, amount_out), max(0, price_impact), fee

    def _evaluate_route(
        self,
        route: SwapRoute,
        token_prices_usd: Dict[str, float],
    ) -> SwapRoute:
        """Evaluate route net output after gas costs."""
        # Calculate gas cost
        num_hops = len(route.hops)
        gas_cost = self.gas_estimate.estimate_cost_usd(num_hops)
        route.estimated_gas_cost_usd = gas_cost
        
        # Convert output to USD
        output_token = route.token_path[-1]
        output_price = token_prices_usd.get(output_token, 0)
        output_value_usd = route.total_amount_out * output_price
        
        # Net output
        route.net_output_usd = output_value_usd - route.total_fees_usd - gas_cost
        
        return route

    def get_pools_for_token_pair(
        self,
        token0: str,
        token1: str,
    ) -> List[Pool]:
        """Get all pools trading a specific token pair."""
        result = []
        for pool in self.pools:
            if (pool.token0 == token0 and pool.token1 == token1) or \
               (pool.token0 == token1 and pool.token1 == token0):
                result.append(pool)
        return result

    def get_supported_tokens(self) -> Set[str]:
        """Get all tokens supported by the router."""
        tokens = set()
        for pool in self.pools:
            tokens.add(pool.token0)
            tokens.add(pool.token1)
        return tokens

    def update_gas_estimate(self, gas_price_gwei: float, eth_price_usd: float):
        """Update gas price and ETH price for accurate cost estimation."""
        self.gas_estimate.current_gas_price_gwei = gas_price_gwei
        self.gas_estimate.eth_price_usd = eth_price_usd


async def main():
    """Example usage of DexRouter."""
    router = DexRouter()
    
    # Add some example pools
    router.add_pool(Pool(
        pool_id="uni_eth_usdc",
        protocol=DEXProtocol.UNISWAP_V3,
        token0="ETH",
        token1="USDC",
        reserve0=10000,
        reserve1=20000000,
        fee_bps=5,
        tvl_usd=40000000,
    ))
    
    router.add_pool(Pool(
        pool_id="uni_wbtc_eth",
        protocol=DEXProtocol.UNISWAP_V3,
        token0="WBTC",
        token1="ETH",
        reserve0=500,
        reserve1=8000,
        fee_bps=30,
        tvl_usd=25000000,
    ))
    
    router.add_pool(Pool(
        pool_id="curve_3pool",
        protocol=DEXProtocol.CURVE,
        token0="USDC",
        token1="DAI",
        reserve0=100000000,
        reserve1=100000000,
        fee_bps=4,
        tvl_usd=500000000,
        is_stablecoin_pair=True,
    ))
    
    # Find best route
    prices = {"ETH": 2000, "USDC": 1.0, "WBTC": 30000, "DAI": 1.0}
    
    route = await router.find_best_route(
        token_in="WBTC",
        token_out="USDC",
        amount_in=1.0,
        token_prices_usd=prices,
    )
    
    if route:
        print(f"Best route found:")
        print(f"  Path: {' -> '.join(route.token_path)}")
        print(f"  Input: {route.total_amount_in}")
        print(f"  Output: {route.total_amount_out}")
        print(f"  Price Impact: {route.total_price_impact:.4f}")
        print(f"  Fees: ${route.total_fees_usd:.2f}")
        print(f"  Gas: ${route.estimated_gas_cost_usd:.2f}")
        print(f"  Net Output: ${route.net_output_usd:.2f}")
        print(f"  Execution Time: {route.execution_time_ms:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
