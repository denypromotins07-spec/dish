# python/benchmark/index_tracker.py
"""
Tracks major crypto indices and traditional safe havens.
Calculates Active Return, Tracking Error, and Information Ratio.
Memory-efficient implementation using Polars.
"""

from __future__ import annotations
import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from collections import deque


@dataclass
class IndexDefinition:
    """Definition of a benchmark index."""
    name: str
    symbol: str
    constituents: Dict[str, float]  # asset -> weight
    rebalance_frequency_days: int = 30
    base_value: float = 1000.0


@dataclass
class TrackingMetrics:
    """Portfolio vs benchmark tracking metrics."""
    active_return_annual: float      # Annualized excess return
    tracking_error_annual: float     # Annualized tracking error
    information_ratio: float         # IR = active_return / tracking_error
    correlation: float               # Correlation with benchmark
    beta: float                      # Portfolio beta vs benchmark
    alpha_annual: float              # Jensen's alpha
    up_capture: float                # Capture ratio in up markets
    down_capture: float              # Capture ratio in down markets
    active_share: float              # Percentage not matching benchmark
    
    def to_dict(self) -> dict:
        return {
            "active_return_annual": self.active_return_annual,
            "tracking_error_annual": self.tracking_error_annual,
            "information_ratio": self.information_ratio,
            "correlation": self.correlation,
            "beta": self.beta,
            "alpha_annual": self.alpha_annual,
            "up_capture": self.up_capture,
            "down_capture": self.down_capture,
            "active_share": self.active_share,
        }


class CryptoIndexTracker:
    """
    Tracks custom crypto indices and computes relative performance.
    
    Supports:
    - CRIX-style market-cap weighted indices
    - Equal-weighted baskets
    - Custom factor-tilted indices
    - Traditional safe haven comparison (Gold, DXY proxy)
    """
    
    # Predefined index definitions
    INDICES = {
        "CRIX_LARGE_CAP": IndexDefinition(
            name="CRIX Large Cap",
            symbol="CRIX-LC",
            constituents={
                "BTC": 0.60,
                "ETH": 0.25,
                "BNB": 0.05,
                "SOL": 0.05,
                "XRP": 0.05,
            },
            rebalance_frequency_days=30,
        ),
        "CRIX_DEFI": IndexDefinition(
            name="CRIX DeFi",
            symbol="CRIX-DEFI",
            constituents={
                "UNI": 0.20,
                "AAVE": 0.20,
                "MKR": 0.15,
                "SNX": 0.15,
                "CRV": 0.15,
                "COMP": 0.15,
            },
            rebalance_frequency_days=7,
        ),
        "GOLD_PROXY": IndexDefinition(
            name="Gold Proxy",
            symbol="GLD",
            constituents={"GLD": 1.0},
            rebalance_frequency_days=365,
        ),
        "DXY_PROXY": IndexDefinition(
            name="Dollar Index Proxy",
            symbol="DXY",
            constituents={"UUP": 1.0},
            rebalance_frequency_days=365,
        ),
    }
    
    def __init__(self, max_history_days: int = 730):
        """
        Initialize the tracker.
        
        Args:
            max_history_days: Maximum history to retain (memory constraint)
        """
        self.max_history_days = max_history_days
        
        # Historical returns storage (memory-bounded)
        self._portfolio_returns: deque = deque(maxlen=max_history_days)
        self._benchmark_returns: Dict[str, deque] = {}
        self._dates: deque = deque(maxlen=max_history_days)
        
        # Index values
        self._index_values: Dict[str, deque] = {}
        self._portfolio_value: deque = deque(maxlen=max_history_days)
        
        # Initialize benchmark storages
        for name in self.INDICES:
            self._benchmark_returns[name] = deque(maxlen=max_history_days)
            self._index_values[name] = deque(maxlen=max_history_days)
    
    def update(
        self,
        date: str,
        portfolio_return: float,
        asset_prices: Dict[str, float],
        benchmark_name: str = "CRIX_LARGE_CAP",
    ) -> None:
        """
        Update tracker with new daily data.
        
        Args:
            date: Date string (YYYY-MM-DD)
            portfolio_return: Portfolio's daily return
            asset_prices: Current prices for all assets
            benchmark_name: Which benchmark to compute
        """
        self._dates.append(date)
        self._portfolio_returns.append(portfolio_return)
        
        # Compute benchmark return from constituent prices
        if benchmark_name in self.INDICES:
            index_def = self.INDICES[benchmark_name]
            
            # Calculate weighted return
            bench_return = 0.0
            for asset, weight in index_def.constituents.items():
                if asset in asset_prices:
                    # Simplified: assume we have previous price stored
                    # In production, would fetch historical price
                    asset_return = asset_prices.get(f"{asset}_return", 0.0)
                    bench_return += weight * asset_return
            
            self._benchmark_returns[benchmark_name].append(bench_return)
            
            # Update index value
            if self._index_values[benchmark_name]:
                prev_value = self._index_values[benchmark_name][-1]
                new_value = prev_value * (1 + bench_return)
            else:
                new_value = index_def.base_value
            self._index_values[benchmark_name].append(new_value)
    
    def calculate_tracking_metrics(
        self,
        benchmark_name: str = "CRIX_LARGE_CAP",
        annualization_factor: int = 365,
    ) -> Optional[TrackingMetrics]:
        """
        Calculate comprehensive tracking metrics.
        
        Args:
            benchmark_name: Which benchmark to compare against
            annualization_factor: Days per year for annualization
            
        Returns:
            TrackingMetrics or None if insufficient data
        """
        if benchmark_name not in self._benchmark_returns:
            return None
        
        port_ret = list(self._portfolio_returns)
        bench_ret = list(self._benchmark_returns[benchmark_name])
        
        if len(port_ret) < 30 or len(bench_ret) < 30:
            return None
        
        # Ensure same length
        min_len = min(len(port_ret), len(bench_ret))
        port_ret = port_ret[-min_len:]
        bench_ret = bench_ret[-min_len:]
        
        port_arr = np.array(port_ret)
        bench_arr = np.array(bench_ret)
        
        # Active return (excess return)
        active_returns = port_arr - bench_arr
        avg_active = np.mean(active_returns)
        active_return_annual = avg_active * annualization_factor
        
        # Tracking error (std of active returns)
        tracking_error_daily = np.std(active_returns, ddof=1)
        tracking_error_annual = tracking_error_daily * np.sqrt(annualization_factor)
        
        # Information Ratio
        info_ratio = active_return_annual / tracking_error_annual if tracking_error_annual > 0 else 0.0
        
        # Correlation
        if np.std(port_arr) > 0 and np.std(bench_arr) > 0:
            correlation = np.corrcoef(port_arr, bench_arr)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Beta
        bench_var = np.var(bench_arr, ddof=1)
        if bench_var > 0:
            covariance = np.cov(port_arr, bench_arr)[0, 1]
            beta = covariance / bench_var
        else:
            beta = 1.0
        
        # Alpha (Jensen's alpha)
        # α = Rp - [Rf + β(Rm - Rf)]
        # Simplified: assume Rf = 0 for crypto
        avg_port = np.mean(port_arr) * annualization_factor
        avg_bench = np.mean(bench_arr) * annualization_factor
        alpha_annual = avg_port - beta * avg_bench
        
        # Capture ratios
        up_days = bench_arr > 0
        down_days = bench_arr < 0
        
        if np.sum(up_days) > 0:
            up_capture = np.mean(port_arr[up_days]) / np.mean(bench_arr[up_days]) if np.mean(bench_arr[up_days]) != 0 else 1.0
        else:
            up_capture = 1.0
        
        if np.sum(down_days) > 0:
            down_capture = np.mean(port_arr[down_days]) / np.mean(bench_arr[down_days]) if np.mean(bench_arr[down_days]) != 0 else 1.0
        else:
            down_capture = 1.0
        
        # Active Share (simplified - based on return divergence)
        # True active share requires holdings data
        active_share = np.mean(np.abs(active_returns)) * 100  # Approximate
        
        return TrackingMetrics(
            active_return_annual=active_return_annual,
            tracking_error_annual=tracking_error_annual,
            information_ratio=info_ratio,
            correlation=correlation,
            beta=beta,
            alpha_annual=alpha_annual,
            up_capture=up_capture,
            down_capture=down_capture,
            active_share=active_share,
        )
    
    def get_cumulative_returns(
        self,
        benchmark_name: str = "CRIX_LARGE_CAP",
    ) -> Dict[str, List[float]]:
        """Get cumulative return series for plotting."""
        port_ret = list(self._portfolio_returns)
        bench_ret = list(self._benchmark_returns.get(benchmark_name, []))
        
        port_cum = []
        bench_cum = []
        
        cum_port = 1.0
        cum_bench = 1.0
        
        for p, b in zip(port_ret, bench_ret):
            cum_port *= (1 + p)
            cum_bench *= (1 + b)
            port_cum.append(cum_port - 1)
            bench_cum.append(cum_bench - 1)
        
        return {
            "portfolio": port_cum,
            "benchmark": bench_cum,
        }
    
    def get_active_return_series(self) -> List[float]:
        """Get series of daily active returns."""
        port_ret = list(self._portfolio_returns)
        
        # Use primary benchmark
        if "CRIX_LARGE_CAP" in self._benchmark_returns:
            bench_ret = list(self._benchmark_returns["CRIX_LARGE_CAP"])
        else:
            return []
        
        min_len = min(len(port_ret), len(bench_ret))
        return [port_ret[i] - bench_ret[i] for i in range(min_len)]
    
    def add_custom_benchmark(
        self,
        name: str,
        returns: List[float],
        dates: Optional[List[str]] = None,
    ) -> None:
        """Add a custom benchmark series."""
        self._benchmark_returns[name] = deque(returns, maxlen=self.max_history_days)
    
    def export_to_polars(self) -> pl.DataFrame:
        """Export all data to a Polars DataFrame."""
        data = {
            "date": list(self._dates),
            "portfolio_return": list(self._portfolio_returns),
        }
        
        for name, ret_deque in self._benchmark_returns.items():
            data[f"{name}_return"] = list(ret_deque)
        
        for name, val_deque in self._index_values.items():
            data[f"{name}_value"] = list(val_deque)
        
        return pl.DataFrame(data)


def create_market_cap_index(
    assets: Dict[str, Dict[str, Any]],
    top_n: int = 10,
) -> IndexDefinition:
    """
    Create a market-cap weighted index from asset data.
    
    Args:
        assets: Dict mapping symbol -> {"market_cap": float, ...}
        top_n: Number of top assets to include
        
    Returns:
        IndexDefinition with market-cap weights
    """
    # Sort by market cap
    sorted_assets = sorted(
        assets.items(),
        key=lambda x: x[1].get("market_cap", 0),
        reverse=True,
    )[:top_n]
    
    total_cap = sum(a[1].get("market_cap", 0) for a in sorted_assets)
    
    constituents = {}
    for symbol, data in sorted_assets:
        if total_cap > 0:
            weight = data.get("market_cap", 0) / total_cap
        else:
            weight = 1.0 / len(sorted_assets)
        constituents[symbol] = weight
    
    return IndexDefinition(
        name=f"Top {top_n} Market Cap",
        symbol=f"TOP{top_n}",
        constituents=constituents,
        rebalance_frequency_days=30,
    )


if __name__ == "__main__":
    # Example usage
    import random
    
    tracker = CryptoIndexTracker(max_history_days=365)
    
    # Simulate daily updates
    for day in range(100):
        date = f"2024-{(day // 30) + 1:02d}-{(day % 30) + 1:02d}"
        
        # Random portfolio return with slight alpha
        portfolio_return = random.gauss(0.001, 0.03) + 0.0005
        
        # Asset prices
        asset_prices = {
            "BTC": 50000 + random.gauss(0, 1000),
            "ETH": 3000 + random.gauss(0, 100),
            "BTC_return": random.gauss(0.001, 0.03),
            "ETH_return": random.gauss(0.001, 0.04),
        }
        
        tracker.update(
            date=date,
            portfolio_return=portfolio_return,
            asset_prices=asset_prices,
            benchmark_name="CRIX_LARGE_CAP",
        )
    
    # Calculate metrics
    metrics = tracker.calculate_tracking_metrics("CRIX_LARGE_CAP")
    if metrics:
        print("Tracking Metrics vs CRIX Large Cap:")
        print(f"  Active Return: {metrics.active_return_annual:.2%}")
        print(f"  Tracking Error: {metrics.tracking_error_annual:.2%}")
        print(f"  Information Ratio: {metrics.information_ratio:.3f}")
        print(f"  Beta: {metrics.beta:.3f}")
        print(f"  Alpha: {metrics.alpha_annual:.2%}")
