"""
Real-time Rolling Correlation and Cointegration Calculator.
Computes cross-asset betas for DXY, Gold, Oil, S&P500 vs BTC/ETH.
Uses Polars for fast, memory-efficient computations on AMD Ryzen AI 5.
Designed to stay under 14GB RAM with zero-copy operations.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import polars as pl


@dataclass
class RollingWindowStats:
    """Memory-efficient rolling statistics using fixed-size ring buffer."""
    
    window_size: int
    _buffer_x: deque = field(default_factory=lambda: deque(maxlen=1000))
    _buffer_y: deque = field(default_factory=lambda: deque(maxlen=1000))
    _sum_x: float = 0.0
    _sum_y: float = 0.0
    _sum_xy: float = 0.0
    _sum_x2: float = 0.0
    _sum_y2: float = 0.0
    _count: int = 0
    
    def __post_init__(self):
        self._buffer_x = deque(maxlen=self.window_size)
        self._buffer_y = deque(maxlen=self.window_size)
    
    def update(self, x: float, y: float) -> None:
        """Update rolling statistics with new observation - O(1) complexity."""
        if len(self._buffer_x) == self.window_size:
            # Remove oldest values from sums
            old_x = self._buffer_x[0]
            old_y = self._buffer_y[0]
            self._sum_x -= old_x
            self._sum_y -= old_y
            self._sum_xy -= old_x * old_y
            self._sum_x2 -= old_x * old_x
            self._sum_y2 -= old_y * old_y
        
        # Add new values
        self._buffer_x.append(x)
        self._buffer_y.append(y)
        self._sum_x += x
        self._sum_y += y
        self._sum_xy += x * y
        self._sum_x2 += x * x
        self._sum_y2 += y * y
        self._count += 1
    
    @property
    def correlation(self) -> Optional[float]:
        """Calculate Pearson correlation coefficient."""
        n = len(self._buffer_x)
        if n < 2:
            return None
        
        mean_x = self._sum_x / n
        mean_y = self._sum_y / n
        
        var_x = (self._sum_x2 / n) - (mean_x * mean_x)
        var_y = (self._sum_y2 / n) - (mean_y * mean_y)
        
        if var_x <= 0 or var_y <= 0:
            return None
        
        cov_xy = (self._sum_xy / n) - (mean_x * mean_y)
        
        return cov_xy / np.sqrt(var_x * var_y)
    
    @property
    def beta(self) -> Optional[float]:
        """Calculate beta (slope) of Y vs X."""
        n = len(self._buffer_x)
        if n < 2:
            return None
        
        mean_x = self._sum_x / n
        mean_y = self._sum_y / n
        
        var_x = (self._sum_x2 / n) - (mean_x * mean_x)
        if var_x <= 0:
            return None
        
        cov_xy = (self._sum_xy / n) - (mean_x * mean_y)
        return cov_xy / var_x


class CorrelationEngine:
    """
    Real-time correlation and cointegration engine for cross-asset analysis.
    Tracks DXY, Gold, Oil, S&P500 correlations with BTC/ETH.
    Optimized for microsecond-level updates with minimal RAM usage.
    """
    
    # Asset pairs to track
    ASSET_PAIRS = [
        ("DXY", "BTC"),
        ("DXY", "ETH"),
        ("GOLD", "BTC"),
        ("GOLD", "ETH"),
        ("OIL", "BTC"),
        ("OIL", "ETH"),
        ("SPX", "BTC"),
        ("SPX", "ETH"),
    ]
    
    def __init__(
        self,
        window_size: int = 1000,  # ~16 minutes at 1Hz
        min_samples: int = 50,
    ):
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Initialize rolling stats for each pair
        self._stats: Dict[str, RollingWindowStats] = {
            f"{asset1}_{asset2}": RollingWindowStats(window_size=window_size)
            for asset1, asset2 in self.ASSET_PAIRS
        }
        
        # Latest prices (for alignment)
        self._latest_prices: Dict[str, Optional[float]] = {
            "DXY": None, "GOLD": None, "OIL": None, "SPX": None,
            "BTC": None, "ETH": None,
        }
        
        # Polars DataFrame for batch calculations (memory-mapped)
        self._price_history: Dict[str, deque] = {
            asset: deque(maxlen=window_size)
            for asset in self._latest_prices.keys()
        }
        
        # Cointegration test buffers (Engle-Granger simplified)
        self._cointegration_cache: Dict[str, List[float]] = {}
    
    def update_price(self, asset: str, price: float, timestamp: int) -> None:
        """
        Update price for a single asset and recalculate correlations.
        Uses incremental updates to avoid full recomputation.
        """
        if asset not in self._latest_prices:
            return
        
        self._latest_prices[asset] = price
        self._price_history[asset].append((timestamp, price))
        
        # Update all relevant pairs
        for pair_key, stats in self._stats.items():
            asset1, asset2 = pair_key.split("_")
            
            if asset1 == asset and self._latest_prices[asset2] is not None:
                stats.update(price, self._latest_prices[asset2])
            elif asset2 == asset and self._latest_prices[asset1] is not None:
                stats.update(self._latest_prices[asset1], price)
    
    def get_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """Get current rolling correlation between two assets."""
        key = f"{asset1}_{asset2}"
        if key not in self._stats:
            key = f"{asset2}_{asset1}"
        
        stats = self._stats.get(key)
        if stats is None:
            return None
        
        return stats.correlation
    
    def get_beta(self, asset1: str, asset2: str) -> Optional[float]:
        """Get beta of asset2 relative to asset1 (asset2 = alpha + beta * asset1)."""
        key = f"{asset1}_{asset2}"
        stats = self._stats.get(key)
        if stats is None:
            return None
        
        return stats.beta
    
    def calculate_cointegration(
        self,
        asset1: str,
        asset2: str,
        lookback: int = 252,
    ) -> Optional[Dict[str, float]]:
        """
        Simplified Engle-Granger cointegration test using Polars.
        Returns hedge ratio and residual statistics.
        """
        if asset1 not in self._price_history or asset2 not in self._price_history:
            return None
        
        # Get aligned price series
        history1 = list(self._price_history[asset1])[-lookback:]
        history2 = list(self._price_history[asset2])[-lookback:]
        
        if len(history1) < self.min_samples or len(history2) < self.min_samples:
            return None
        
        # Create Polars DataFrame (zero-copy where possible)
        df = pl.DataFrame({
            "price1": [p for _, p in history1],
            "price2": [p for _, p in history2],
        })
        
        # Calculate hedge ratio via OLS
        try:
            # Normalize prices
            df = df.with_columns([
                (pl.col("price1") - pl.col("price1").mean()) / pl.col("price1").std(),
                (pl.col("price2") - pl.col("price2").mean()) / pl.col("price2").std(),
            ])
            
            # Simple linear regression for hedge ratio
            cov = df.select(pl.corr("price1", "price2")).item()
            std1 = df.select(pl.col("price1").std()).item()
            std2 = df.select(pl.col("price2").std()).item()
            
            hedge_ratio = cov * std2 / std1
            
            # Calculate residuals (spread)
            df = df.with_columns(
                (pl.col("price2") - hedge_ratio * pl.col("price1")).alias("residual")
            )
            
            # ADF-like statistic (simplified - variance ratio test)
            residuals = df["residual"].to_numpy()
            if len(residuals) > 10:
                diff_residuals = np.diff(residuals)
                var_diff = np.var(diff_residuals)
                var_resid = np.var(residuals)
                
                # Variance ratio test statistic
                vr_statistic = var_diff / var_resid if var_resid > 0 else float('inf')
                
                return {
                    "hedge_ratio": hedge_ratio,
                    "variance_ratio": vr_statistic,
                    "mean_reversion_speed": 1.0 - vr_statistic / 2.0,
                    "samples": len(residuals),
                }
        except Exception:
            pass
        
        return None
    
    def get_all_correlations(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Get all current correlations organized by crypto asset."""
        result = {"BTC": {}, "ETH": {}}
        
        for pair_key, stats in self._stats.items():
            asset1, asset2 = pair_key.split("_")
            corr = stats.correlation
            
            if asset2 in result:
                result[asset2][asset1] = corr
            elif asset1 in result:
                result[asset1][asset2] = corr
        
        return result
    
    def get_regime_signal(self) -> Dict[str, str]:
        """
        Detect correlation regime changes for risk management.
        Returns signals for each macro asset vs crypto.
        """
        signals = {}
        
        for pair_key, stats in self._stats.items():
            corr = stats.correlation
            if corr is None:
                signals[pair_key] = "INSUFFICIENT_DATA"
                continue
            
            # Regime detection thresholds
            if abs(corr) > 0.7:
                signals[pair_key] = "STRONG_CORRELATION"
            elif abs(corr) > 0.4:
                signals[pair_key] = "MODERATE_CORRELATION"
            elif abs(corr) > 0.2:
                signals[pair_key] = "WEAK_CORRELATION"
            else:
                signals[pair_key] = "NO_CORRELATION"
        
        return signals
    
    def export_to_polars(self) -> pl.DataFrame:
        """Export current state to Polars DataFrame for further analysis."""
        data = []
        
        for pair_key, stats in self._stats.items():
            asset1, asset2 = pair_key.split("_")
            corr = stats.correlation
            beta = stats.beta
            
            data.append({
                "asset1": asset1,
                "asset2": asset2,
                "correlation": corr if corr is not None else float('nan'),
                "beta": beta if beta is not None else float('nan'),
                "samples": len(stats._buffer_x),
            })
        
        return pl.DataFrame(data)


def main():
    """Example usage of the correlation engine."""
    engine = CorrelationEngine(window_size=500)
    
    # Simulate price updates
    import random
    base_prices = {"DXY": 103.5, "GOLD": 1950.0, "OIL": 80.0, "SPX": 4500.0, "BTC": 43000.0, "ETH": 2300.0}
    
    for i in range(600):
        timestamp = 1700000000 + i
        for asset in base_prices:
            # Random walk with drift
            base_prices[asset] *= (1 + random.gauss(0, 0.001))
            engine.update_price(asset, base_prices[asset], timestamp)
    
    # Get results
    print("Correlations with BTC:")
    for macro in ["DXY", "GOLD", "OIL", "SPX"]:
        corr = engine.get_correlation(macro, "BTC")
        beta = engine.get_beta(macro, "BTC")
        print(f"  {macro}: corr={corr:.4f}, beta={beta:.4f}" if corr else f"  {macro}: insufficient data")
    
    # Cointegration test
    coint = engine.calculate_cointegration("GOLD", "BTC")
    if coint:
        print(f"\nGOLD-BTC Cointegration: hedge_ratio={coint['hedge_ratio']:.4f}")


if __name__ == "__main__":
    main()
