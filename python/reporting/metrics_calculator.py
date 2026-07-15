"""
Comprehensive performance analytics module calculating CAGR, Sharpe, Sortino, Calmar, 
Information Ratio, and Ulcer Index directly from Nautilus trade logs using memory-efficient Polars DataFrames.
"""

import polars as pl
from polars import col
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all calculated performance metrics."""
    # Returns
    total_return: float
    cagr: float
    annualized_return: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    ulcer_index: float
    var_95: float
    cvar_95: float
    
    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Additional metrics
    tail_ratio: float
    skewness: float
    kurtosis: float


class MetricsCalculator:
    """
    Memory-efficient performance metrics calculator using Polars.
    
    Optimized for large datasets with minimal RAM usage (<14GB constraint).
    """
    
    def __init__(self, risk_free_rate: float = 0.05, trading_days_per_year: int = 365):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        
    def calculate_all_metrics(self, trades_df: pl.DataFrame) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics from trades DataFrame.
        
        Parameters
        ----------
        trades_df : pl.DataFrame
            DataFrame with columns: timestamp, pnl, cumulative_pnl, etc.
            
        Returns
        -------
        PerformanceMetrics
            All calculated metrics.
        """
        if trades_df.is_empty():
            return self._empty_metrics()
        
        # Ensure we have required columns
        required_cols = ['pnl']
        for col in required_cols:
            if col not in trades_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate equity curve if not present
        if 'equity' not in trades_df.columns:
            trades_df = trades_df.with_columns(
                (col('pnl').cum_sum() + 100000).alias('equity')  # Assume 100k initial
            )
        
        # Calculate returns
        returns = self._calculate_returns(trades_df)
        
        # Calculate all metric categories
        return_metrics = self._calculate_return_metrics(returns)
        risk_metrics = self._calculate_risk_metrics(returns, trades_df)
        trade_metrics = self._calculate_trade_metrics(trades_df)
        
        return PerformanceMetrics(
            **return_metrics,
            **risk_metrics,
            **trade_metrics,
        )
    
    def _calculate_returns(self, df: pl.DataFrame) -> pl.Series:
        """Calculate returns series from equity curve."""
        if 'equity' in df.columns:
            returns = df['equity'].pct_change().fill_null(0)
        else:
            # Use PnL directly normalized by some capital base
            returns = df['pnl'] / 100000
        
        return returns
    
    def _calculate_return_metrics(self, returns: pl.Series) -> Dict[str, float]:
        """Calculate return-based metrics."""
        returns_np = returns.to_numpy()
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return (CAGR)
        n_periods = len(returns)
        years = n_periods / self.trading_days
        cagr = (1 + total_return) ** (1 / max(years, 0.001)) - 1
        
        # Annualized return
        annualized_return = returns.mean() * self.trading_days
        
        return {
            'total_return': float(total_return),
            'cagr': float(cagr),
            'annualized_return': float(annualized_return),
        }
    
    def _calculate_risk_metrics(
        self,
        returns: pl.Series,
        trades_df: pl.DataFrame,
    ) -> Dict[str, float]:
        """Calculate risk-adjusted metrics."""
        returns_np = returns.to_numpy()
        
        # Volatility (annualized)
        volatility = float(returns_np.std()) * np.sqrt(self.trading_days)
        
        # Sharpe Ratio
        excess_return = returns.mean() - (self.risk_free_rate / self.trading_days)
        sharpe = (excess_return * self.trading_days) / max(volatility, 0.0001)
        
        # Downside deviation and Sortino Ratio
        negative_returns = returns.filter(returns < 0)
        downside_dev = float(negative_returns.std()) * np.sqrt(self.trading_days) if len(negative_returns) > 0 else 0.0001
        sortino = (excess_return * self.trading_days) / max(downside_dev, 0.0001)
        
        # Maximum Drawdown
        if 'equity' in trades_df.columns:
            max_dd = self._calculate_max_drawdown(trades_df['equity'])
        else:
            max_dd = self._estimate_max_drawdown(returns)
        
        # Calmar Ratio
        calmar = abs(cagr / max_dd) if max_dd != 0 else 0
        
        # Ulcer Index
        ulcer = self._calculate_ulcer_index(trades_df) if 'equity' in trades_df.columns else 0
        
        # Information Ratio (vs risk-free rate as benchmark)
        tracking_error = volatility  # Simplified
        info_ratio = (returns.mean() * self.trading_days - self.risk_free_rate) / max(tracking_error, 0.0001)
        
        # VaR and CVaR
        var_95 = float(np.percentile(returns_np, 5))
        cvar_95 = float(returns_np[returns_np <= var_95].mean()) if len(returns_np[returns_np <= var_95]) > 0 else var_95
        
        # Tail Ratio
        sorted_returns = np.sort(returns_np)
        p95 = np.percentile(sorted_returns, 95)
        p5 = np.percentile(sorted_returns, 5)
        tail_ratio = abs(p95 / p5) if p5 != 0 else 1.0
        
        # Skewness and Kurtosis
        skewness = float(pl.Series(returns_np).skew()) if len(returns_np) > 2 else 0
        kurtosis = float(pl.Series(returns_np).kurtosis()) if len(returns_np) > 3 else 0
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'information_ratio': info_ratio,
            'downside_deviation': downside_dev,
            'max_drawdown': max_dd,
            'ulcer_index': ulcer,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }
    
    def _calculate_trade_metrics(self, trades_df: pl.DataFrame) -> Dict[str, float]:
        """Calculate trade-level statistics."""
        pnls = trades_df['pnl'].to_numpy()
        
        total_trades = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        
        win_rate = win_count / max(total_trades, 1)
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        
        profit_factor = gross_profit / max(gross_loss, 0.001)
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        largest_win = wins.max() if len(wins) > 0 else 0
        largest_loss = abs(losses.min()) if len(losses) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'largest_win': float(largest_win),
            'largest_loss': float(largest_loss),
        }
    
    def _calculate_max_drawdown(self, equity: pl.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        equity_np = equity.to_numpy()
        
        peak = equity_np[0]
        max_dd = 0.0
        
        for value in equity_np:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _estimate_max_drawdown(self, returns: pl.Series) -> float:
        """Estimate max drawdown from returns when equity not available."""
        cum_returns = (1 + returns).cum_prod()
        return self._calculate_max_drawdown(cum_returns)
    
    def _calculate_ulcer_index(self, trades_df: pl.DataFrame) -> float:
        """Calculate Ulcer Index (measure of drawdown pain)."""
        equity = trades_df['equity'].to_numpy()
        
        peak = equity[0]
        drawdowns_sq = []
        
        for value in equity:
            if value > peak:
                peak = value
            
            dd_pct = (peak - value) / peak * 100 if peak > 0 else 0
            drawdowns_sq.append(dd_pct ** 2)
        
        return np.sqrt(np.mean(drawdowns_sq))
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for edge cases."""
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            volatility=0.0,
            downside_deviation=0.0,
            max_drawdown=0.0,
            ulcer_index=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            tail_ratio=1.0,
            skewness=0.0,
            kurtosis=0.0,
        )


def load_nautilus_trades_to_polars(
    csv_path: str,
    batch_size: int = 10000,
) -> pl.DataFrame:
    """
    Load Nautilus trade logs into Polars DataFrame efficiently.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with trade logs.
    batch_size : int, default 10000
        Batch size for streaming read.
        
    Returns
    -------
    pl.DataFrame
        Trades DataFrame.
    """
    # Use Polars scan for lazy evaluation (memory efficient)
    lf = pl.scan_csv(csv_path)
    
    # Select and rename relevant columns
    lf = lf.select([
        col('timestamp').alias('timestamp'),
        col('pnl').alias('pnl'),
        col('commission').alias('commission'),
        col('symbol').alias('symbol'),
        col('side').alias('side'),
    ])
    
    # Collect with streaming optimization
    return lf.collect(streaming=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample trades data
    np.random.seed(42)
    n_trades = 1000
    
    sample_data = {
        'timestamp': pl.Series(range(n_trades)),
        'pnl': pl.Series(np.random.normal(10, 100, n_trades)),
    }
    
    trades_df = pl.DataFrame(sample_data)
    
    # Calculate metrics
    calculator = MetricsCalculator(risk_free_rate=0.05)
    metrics = calculator.calculate_all_metrics(trades_df)
    
    print("Performance Metrics:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  CAGR: {metrics.cagr:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Total Trades: {metrics.total_trades}")
