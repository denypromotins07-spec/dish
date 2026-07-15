"""
Backend data aggregator that pushes portfolio-level metrics to the telemetry database.
Tracks: Portfolio Beta, Component VaR, Tracking Error, Active Share, Net Exposure.

Memory-efficient implementation bounded for <14GB total system RAM.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import deque
import json


@dataclass(slots=True)
class PortfolioMetrics:
    """Snapshot of portfolio-level metrics."""
    timestamp: datetime
    portfolio_value: float
    
    # Risk metrics
    portfolio_beta: float
    portfolio_volatility: float
    var_95: float
    var_99: float
    component_var: Dict[str, float]
    
    # Relative metrics
    tracking_error: float
    active_share: float
    information_ratio: float
    
    # Exposure metrics
    net_exposure: float
    gross_exposure: float
    long_exposure: float
    short_exposure: float
    
    # Performance
    daily_return: float
    ytd_return: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Concentration
    top_5_concentration: float
    herfindahl_index: float


@dataclass(slots=True)
class TelemetryBatch:
    """Batch of metrics ready for database insertion."""
    metrics: List[PortfolioMetrics]
    batch_size: int
    created_at: datetime
    
    def to_json(self) -> str:
        """Convert batch to JSON for database insertion."""
        return json.dumps([{
            'timestamp': m.timestamp.isoformat(),
            'portfolio_value': m.portfolio_value,
            'portfolio_beta': m.portfolio_beta,
            'portfolio_volatility': m.portfolio_volatility,
            'var_95': m.var_95,
            'var_99': m.var_99,
            'component_var': m.component_var,
            'tracking_error': m.tracking_error,
            'active_share': m.active_share,
            'information_ratio': m.information_ratio,
            'net_exposure': m.net_exposure,
            'gross_exposure': m.gross_exposure,
            'long_exposure': m.long_exposure,
            'short_exposure': m.short_exposure,
            'daily_return': m.daily_return,
            'ytd_return': m.ytd_return,
            'sharpe_ratio': m.sharpe_ratio,
            'sortino_ratio': m.sortino_ratio,
            'top_5_concentration': m.top_5_concentration,
            'herfindahl_index': m.herfindahl_index
        } for m in self.metrics])


class TelemetryAggregator:
    """
    Aggregates and processes portfolio metrics for telemetry storage.
    Optimized for high-frequency updates with minimal memory overhead.
    """
    
    __slots__ = (
        'asset_names', 'benchmark_returns', 'risk_free_rate',
        'metrics_buffer', 'max_buffer_size', 'batch_size'
    )
    
    def __init__(
        self,
        asset_names: List[str],
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.05,
        max_buffer_size: int = 1000,
        batch_size: int = 100
    ):
        self.asset_names = asset_names
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        
        # Circular buffer for recent metrics
        self.metrics_buffer: deque = deque(maxlen=max_buffer_size)
    
    def calculate_portfolio_beta(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> float:
        """Calculate portfolio beta relative to benchmark."""
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_returns
        
        if benchmark_returns is None or len(returns) < 20:
            return 1.0  # Default market beta
        
        # Ensure same length
        min_len = min(len(returns), len(benchmark_returns))
        port_ret = returns[-min_len:]
        bench_ret = benchmark_returns[-min_len:]
        
        # Calculate beta using covariance/variance
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        variance = np.var(bench_ret)
        
        if variance < 1e-10:
            return 1.0
        
        return covariance / variance
    
    def calculate_component_var(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Component VaR - each asset's contribution to total VaR.
        Uses marginal VaR decomposition.
        """
        n_assets = len(weights)
        
        # Portfolio variance and volatility
        port_var = weights @ covariance @ weights
        port_vol = np.sqrt(port_var)
        
        if port_vol < 1e-10:
            return {name: 0.0 for name in self.asset_names}
        
        # Z-score for confidence level
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence)
        
        # Portfolio VaR
        portfolio_var = -z_score * port_vol
        
        # Marginal VaR for each asset
        marginal_var = (covariance @ weights) / port_vol
        
        # Component VaR = weight * marginal VaR
        component_var = weights * marginal_var * (-z_score)
        
        # Normalize to sum to total VaR
        total_component = np.sum(component_var)
        if abs(total_component) > 1e-10:
            component_var = component_var * portfolio_var / total_component
        
        return {
            self.asset_names[i]: float(component_var[i])
            for i in range(n_assets)
        }
    
    def calculate_tracking_error(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        window: int = 252
    ) -> float:
        """Calculate annualized tracking error."""
        if len(portfolio_returns) < window:
            window = len(portfolio_returns)
        
        if window < 20:
            return 0.0
        
        # Use last 'window' days
        port_ret = portfolio_returns[-window:]
        bench_ret = benchmark_returns[-window:]
        
        # Active returns
        active_returns = port_ret - bench_ret
        
        # Tracking error = std(active returns) * sqrt(252)
        return np.std(active_returns) * np.sqrt(252)
    
    def calculate_active_share(
        self,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray
    ) -> float:
        """
        Calculate Active Share - percentage of portfolio different from benchmark.
        Active Share = 0.5 * sum(|w_p - w_b|)
        """
        if len(portfolio_weights) != len(benchmark_weights):
            return 0.0
        
        diff = np.abs(portfolio_weights - benchmark_weights)
        return 0.5 * np.sum(diff)
    
    def calculate_information_ratio(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        window: int = 252
    ) -> float:
        """Calculate Information Ratio = active return / tracking error."""
        if len(portfolio_returns) < window:
            window = len(portfolio_returns)
        
        if window < 20:
            return 0.0
        
        port_ret = portfolio_returns[-window:]
        bench_ret = benchmark_returns[-window:]
        
        active_returns = port_ret - bench_ret
        active_mean = np.mean(active_returns)
        active_std = np.std(active_returns)
        
        if active_std < 1e-10:
            return 0.0
        
        # Annualize
        return (active_mean * 252) / (active_std * np.sqrt(252))
    
    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        window: int = 252
    ) -> float:
        """Calculate Sortino Ratio (downside deviation adjusted)."""
        if len(returns) < window:
            window = len(returns)
        
        if window < 20:
            return 0.0
        
        ret = returns[-window:]
        mean_ret = np.mean(ret)
        
        # Downside deviation
        downside_returns = ret[ret < 0]
        if len(downside_returns) == 0:
            return 3.0  # Perfect score if no negative returns
        
        downside_std = np.std(downside_returns)
        
        if downside_std < 1e-10:
            return 0.0
        
        # Annualize
        excess_return = (mean_ret - self.risk_free_rate / 252) * 252
        return excess_return / (downside_std * np.sqrt(252))
    
    def calculate_herfindahl_index(self, weights: np.ndarray) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration.
        HHI = sum(w_i^2), ranges from 1/n to 1
        """
        return float(np.sum(weights ** 2))
    
    def aggregate_metrics(
        self,
        weights: np.ndarray,
        prices: np.ndarray,
        returns_history: np.ndarray,
        covariance: np.ndarray,
        benchmark_weights: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> PortfolioMetrics:
        """
        Aggregate all portfolio metrics into a single snapshot.
        """
        n_assets = len(weights)
        timestamp = datetime.utcnow()
        
        # Portfolio value
        portfolio_value = float(np.sum(weights * prices))
        
        # Portfolio returns (weighted sum of asset returns)
        if len(returns_history) > 0:
            latest_returns = returns_history[-1] if returns_history.ndim > 1 else returns_history
            portfolio_return = float(np.dot(weights, latest_returns))
        else:
            portfolio_return = 0.0
        
        # Historical portfolio returns for time-series metrics
        if returns_history.ndim > 1:
            port_returns = returns_history @ weights
        else:
            port_returns = np.array([portfolio_return])
        
        # Risk metrics
        portfolio_beta = self.calculate_portfolio_beta(port_returns, benchmark_returns)
        port_vol = np.sqrt(weights @ covariance @ weights) * np.sqrt(252)
        
        component_var = self.calculate_component_var(weights, covariance)
        
        # VaR calculations
        var_95 = float(np.percentile(port_returns, 5)) if len(port_returns) > 10 else 0.0
        var_99 = float(np.percentile(port_returns, 1)) if len(port_returns) > 10 else 0.0
        
        # Relative metrics
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            tracking_error = self.calculate_tracking_error(port_returns, benchmark_returns)
            info_ratio = self.calculate_information_ratio(port_returns, benchmark_returns)
        else:
            tracking_error = 0.0
            info_ratio = 0.0
        
        if benchmark_weights is not None:
            active_share = self.calculate_active_share(weights, benchmark_weights)
        else:
            active_share = 0.0
        
        # Exposure metrics
        long_mask = weights > 0
        short_mask = weights < 0
        
        long_exposure = float(np.sum(weights[long_mask] * prices[long_mask]))
        short_exposure = float(abs(np.sum(weights[short_mask] * prices[short_mask])))
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        # Performance metrics
        daily_return = portfolio_return
        ytd_return = float(np.sum(port_returns)) if len(port_returns) > 0 else 0.0
        
        sharpe_ratio = 0.0
        if port_vol > 1e-10:
            excess_return = (np.mean(port_returns) * 252 - self.risk_free_rate)
            sharpe_ratio = excess_return / port_vol
        
        sortino_ratio = self.calculate_sortino_ratio(port_returns)
        
        # Concentration metrics
        sorted_weights = np.sort(weights)[::-1]
        top_5_concentration = float(np.sum(sorted_weights[:5]))
        hhi = self.calculate_herfindahl_index(weights)
        
        metrics = PortfolioMetrics(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            portfolio_beta=portfolio_beta,
            portfolio_volatility=port_vol,
            var_95=var_95,
            var_99=var_99,
            component_var=component_var,
            tracking_error=tracking_error,
            active_share=active_share,
            information_ratio=info_ratio,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            daily_return=daily_return,
            ytd_return=ytd_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            top_5_concentration=top_5_concentration,
            herfindahl_index=hhi
        )
        
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        return metrics
    
    def get_batch_for_insertion(self) -> Optional[TelemetryBatch]:
        """Get a batch of metrics ready for database insertion."""
        if len(self.metrics_buffer) < self.batch_size:
            return None
        
        # Extract batch
        batch_metrics = []
        for _ in range(self.batch_size):
            if self.metrics_buffer:
                batch_metrics.append(self.metrics_buffer.popleft())
        
        return TelemetryBatch(
            metrics=batch_metrics,
            batch_size=len(batch_metrics),
            created_at=datetime.utcnow()
        )
    
    def get_latest_metrics(self) -> Optional[PortfolioMetrics]:
        """Get the most recent metrics snapshot."""
        if self.metrics_buffer:
            return self.metrics_buffer[-1]
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of buffered metrics."""
        if not self.metrics_buffer:
            return {}
        
        metrics_list = list(self.metrics_buffer)
        
        betas = [m.portfolio_beta for m in metrics_list]
        vols = [m.portfolio_volatility for m in metrics_list]
        sharpes = [m.sharpe_ratio for m in metrics_list]
        
        return {
            'count': len(metrics_list),
            'avg_beta': float(np.mean(betas)),
            'avg_volatility': float(np.mean(vols)),
            'avg_sharpe': float(np.mean(sharpes)),
            'max_sharpe': float(np.max(sharpes)),
            'min_sharpe': float(np.min(sharpes)),
            'latest_timestamp': metrics_list[-1].timestamp.isoformat() if metrics_list else None
        }


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    asset_names = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
    n_assets = len(asset_names)
    
    aggregator = TelemetryAggregator(asset_names, risk_free_rate=0.05)
    
    # Simulate some data
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    prices = np.array([45000, 2800, 100, 35, 0.8])
    
    # Generate returns history (252 days)
    returns_history = np.random.randn(252, n_assets) * 0.02
    
    # Covariance matrix
    cov = np.cov(returns_history.T)
    
    # Benchmark data
    benchmark_weights = np.ones(n_assets) / n_assets
    benchmark_returns = np.mean(returns_history, axis=1)
    
    # Aggregate metrics
    metrics = aggregator.aggregate_metrics(
        weights=weights,
        prices=prices,
        returns_history=returns_history,
        covariance=cov,
        benchmark_weights=benchmark_weights,
        benchmark_returns=benchmark_returns
    )
    
    print("Portfolio Metrics:")
    print(f"  Portfolio Value: ${metrics.portfolio_value:,.2f}")
    print(f"  Beta: {metrics.portfolio_beta:.3f}")
    print(f"  Volatility: {metrics.portfolio_volatility:.2%}")
    print(f"  VaR (95%): {metrics.var_95:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Tracking Error: {metrics.tracking_error:.2%}")
    print(f"  Active Share: {metrics.active_share:.2%}")
    print(f"  Net Exposure: ${metrics.net_exposure:,.2f}")
    print(f"  Top 5 Concentration: {metrics.top_5_concentration:.1%}")
    print(f"  HHI: {metrics.herfindahl_index:.3f}")
    print(f"\nComponent VaR:")
    for asset, cvar in metrics.component_var.items():
        print(f"  {asset}: {cvar:.4f}")
