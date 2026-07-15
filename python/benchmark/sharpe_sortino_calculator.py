# python/benchmark/sharpe_sortino_calculator.py
"""
Advanced rolling performance metric calculator.
Computes Sharpe, Sortino, Calmar, and Omega ratios.
Memory-efficient Polars implementation with proper weekend/holiday handling.
"""

from __future__ import annotations
import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from collections import deque


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    annualized_return: float
    annualized_vol: float
    skewness: float
    kurtosis: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    
    def to_dict(self) -> dict:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "annualized_return": self.annualized_return,
            "annualized_vol": self.annualized_vol,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


@dataclass 
class RollingMetrics:
    """Rolling window metrics over time."""
    dates: List[str]
    rolling_sharpe: List[float]
    rolling_sortino: List[float]
    rolling_calmar: List[float]
    rolling_max_dd: List[float]


class CryptoPerformanceCalculator:
    """
    Calculates advanced performance metrics for crypto strategies.
    
    Handles:
    - Weekend/holiday gaps in crypto time-series (24/7 markets)
    - Memory-efficient rolling calculations
    - Multiple annualization conventions
    - Downside risk measures
    """
    
    def __init__(
        self,
        annualization_factor: int = 365,  # Crypto trades 24/7
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,  # For Omega ratio
        max_history_days: int = 730,
    ):
        """
        Initialize the calculator.
        
        Args:
            annualization_factor: Days per year (365 for crypto, 252 for stocks)
            risk_free_rate: Annual risk-free rate (decimal)
            target_return: Target return for Omega ratio (decimal)
            max_history_days: Maximum history to retain
        """
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.max_history_days = max_history_days
        
        # Storage (memory-bounded)
        self._returns: deque = deque(maxlen=max_history_days)
        self._dates: deque = deque(maxlen=max_history_days)
        self._cumulative: deque = deque(maxlen=max_history_days)
    
    def add_return(self, date: str, return_pct: float) -> None:
        """
        Add a daily return observation.
        
        Args:
            date: Date string (YYYY-MM-DD)
            return_pct: Daily return as decimal (e.g., 0.01 for 1%)
        """
        self._dates.append(date)
        self._returns.append(return_pct)
        
        # Update cumulative return
        if self._cumulative:
            new_cum = self._cumulative[-1] * (1 + return_pct)
        else:
            new_cum = 1.0 * (1 + return_pct)
        self._cumulative.append(new_cum)
    
    def add_returns_from_polars(self, df: pl.DataFrame, date_col: str, return_col: str) -> None:
        """Bulk load returns from a Polars DataFrame."""
        rows = df.select([date_col, return_col]).to_dicts()
        for row in rows:
            self.add_return(row[date_col], row[return_col])
    
    def calculate_all_metrics(
        self,
        min_observations: int = 30,
    ) -> Optional[PerformanceMetrics]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            min_observations: Minimum data points required
            
        Returns:
            PerformanceMetrics or None if insufficient data
        """
        if len(self._returns) < min_observations:
            return None
        
        returns = np.array(list(self._returns))
        cum_values = np.array(list(self._cumulative))
        
        # Daily risk-free rate
        daily_rf = self.risk_free_rate / self.annualization_factor
        
        # Excess returns
        excess_returns = returns - daily_rf
        
        # Mean and std
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(returns, ddof=1)
        
        # Annualized return and vol
        ann_return = np.mean(returns) * self.annualization_factor
        ann_vol = std_returns * np.sqrt(self.annualization_factor)
        
        # Sharpe Ratio
        if std_returns > 0:
            sharpe = (mean_excess / std_returns) * np.sqrt(self.annualization_factor)
        else:
            sharpe = 0.0
        
        # Sortino Ratio (using downside deviation)
        negative_returns = returns[returns < daily_rf] - daily_rf
        if len(negative_returns) > 0 and np.std(negative_returns, ddof=1) > 0:
            downside_std = np.std(negative_returns, ddof=1)
            sortino = (mean_excess / downside_std) * np.sqrt(self.annualization_factor)
        else:
            sortino = sharpe  # Fall back to Sharpe if no negative returns
        
        # Maximum Drawdown
        max_dd = self._calculate_max_drawdown(cum_values)
        
        # Calmar Ratio
        if abs(max_dd) > 0:
            calmar = ann_return / abs(max_dd)
        else:
            calmar = float('inf') if ann_return > 0 else 0.0
        
        # Omega Ratio
        omega = self._calculate_omega(returns)
        
        # Skewness
        if std_returns > 0 and len(returns) > 2:
            skewness = np.mean(((returns - np.mean(returns)) / std_returns) ** 3)
        else:
            skewness = 0.0
        
        # Kurtosis (excess)
        if std_returns > 0 and len(returns) > 3:
            kurtosis = np.mean(((returns - np.mean(returns)) / std_returns) ** 4) - 3
        else:
            kurtosis = 0.0
        
        # VaR 95% (historical)
        var_95 = np.percentile(returns, 5)
        
        # CVaR 95% (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
        
        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            annualized_return=ann_return,
            annualized_vol=ann_vol,
            skewness=skewness,
            kurtosis=kurtosis,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
        )
    
    def _calculate_max_drawdown(self, cumulative_values: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative values."""
        if len(cumulative_values) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(cumulative_values)
        drawdowns = (cumulative_values - running_max) / running_max
        return np.min(drawdowns)
    
    def _calculate_omega(self, returns: np.ndarray) -> float:
        """
        Calculate Omega ratio.
        
        Omega = sum(gains above threshold) / sum(losses below threshold)
        """
        gains = returns[returns > self.target_return] - self.target_return
        losses = self.target_return - returns[returns <= self.target_return]
        
        sum_gains = np.sum(gains)
        sum_losses = np.sum(losses)
        
        if sum_losses == 0:
            return float('inf') if sum_gains > 0 else 1.0
        
        return sum_gains / sum_losses
    
    def calculate_rolling_metrics(
        self,
        window_days: int = 90,
        step_days: int = 1,
    ) -> Optional[RollingMetrics]:
        """
        Calculate rolling performance metrics.
        
        Args:
            window_days: Rolling window size
            step_days: Step between calculations
            
        Returns:
            RollingMetrics or None if insufficient data
        """
        if len(self._returns) < window_days:
            return None
        
        returns = np.array(list(self._returns))
        cum_values = np.array(list(self._cumulative))
        dates = list(self._dates)
        
        rolling_sharpe = []
        rolling_sortino = []
        rolling_calmar = []
        rolling_max_dd = []
        result_dates = []
        
        n_windows = (len(returns) - window_days) // step_days + 1
        
        for i in range(n_windows):
            start_idx = i * step_days
            end_idx = start_idx + window_days
            
            window_returns = returns[start_idx:end_idx]
            window_cum = cum_values[start_idx:end_idx]
            
            # Sharpe
            if np.std(window_returns, ddof=1) > 0:
                rs = (np.mean(window_returns) / np.std(window_returns, ddof=1)) * np.sqrt(self.annualization_factor)
            else:
                rs = 0.0
            
            # Sortino
            neg_ret = window_returns[window_returns < 0]
            if len(neg_ret) > 0 and np.std(neg_ret, ddof=1) > 0:
                downside_std = np.std(neg_ret, ddof=1)
                rst = (np.mean(window_returns) / downside_std) * np.sqrt(self.annualization_factor)
            else:
                rst = rs
            
            # Max DD
            mdd = self._calculate_max_drawdown(window_cum)
            
            # Calmar
            ann_ret = np.mean(window_returns) * self.annualization_factor
            if abs(mdd) > 0:
                rc = ann_ret / abs(mdd)
            else:
                rc = float('inf') if ann_ret > 0 else 0.0
            
            rolling_sharpe.append(rs)
            rolling_sortino.append(rst)
            rolling_calmar.append(rc)
            rolling_max_dd.append(mdd)
            result_dates.append(dates[end_idx - 1])
        
        return RollingMetrics(
            dates=result_dates,
            rolling_sharpe=rolling_sharpe,
            rolling_sortino=rolling_sortino,
            rolling_calmar=rolling_calmar,
            rolling_max_dd=rolling_max_dd,
        )
    
    def get_best_worst_periods(
        self,
        period_days: int = 30,
        top_n: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify best and worst rolling periods.
        
        Args:
            period_days: Period length to analyze
            top_n: Number of periods to return
            
        Returns:
            Dict with 'best' and 'worst' period lists
        """
        if len(self._returns) < period_days:
            return {"best": [], "worst": []}
        
        returns = np.array(list(self._returns))
        dates = list(self._dates)
        
        period_returns = []
        for i in range(len(returns) - period_days + 1):
            period_ret = np.prod(1 + returns[i:i+period_days]) - 1
            period_returns.append({
                "start_date": dates[i],
                "end_date": dates[i + period_days - 1],
                "return": period_ret,
            })
        
        # Sort by return
        sorted_periods = sorted(period_returns, key=lambda x: x["return"], reverse=True)
        
        return {
            "best": sorted_periods[:top_n],
            "worst": sorted_periods[-top_n:],
        }
    
    def to_polars(self) -> pl.DataFrame:
        """Export data to Polars DataFrame."""
        return pl.DataFrame({
            "date": list(self._dates),
            "return": list(self._returns),
            "cumulative": list(self._cumulative),
        })
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if len(self._returns) < 10:
            return {"error": "Insufficient data"}
        
        returns = np.array(list(self._returns))
        
        return {
            "total_days": len(self._returns),
            "mean_daily_return": float(np.mean(returns)),
            "std_daily_return": float(np.std(returns, ddof=1)),
            "min_daily_return": float(np.min(returns)),
            "max_daily_return": float(np.max(returns)),
            "positive_days": int(np.sum(returns > 0)),
            "negative_days": int(np.sum(returns < 0)),
            "win_rate": float(np.mean(returns > 0)),
            "total_return": float(self._cumulative[-1] - 1) if self._cumulative else 0.0,
        }


def handle_crypto_timegaps(
    df: pl.DataFrame,
    date_col: str = "date",
    freq: str = "1d",
) -> pl.DataFrame:
    """
    Handle potential time gaps in crypto data.
    
    Unlike traditional markets, crypto trades 24/7, but data feeds
    may still have gaps. This function ensures continuous time series.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        freq: Expected frequency (default 1 day)
        
    Returns:
        DataFrame with gaps filled
    """
    # Parse dates
    df = df.with_columns(pl.col(date_col).str.strptime(pl.Date, "%Y-%m-%d"))
    
    # Get date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    if min_date is None or max_date is None:
        return df
    
    # Create complete date range
    date_range = pl.date_range(min_date, max_date, interval=freq, eager=True)
    
    # Create full dataframe
    full_df = pl.DataFrame({date_col: date_range})
    
    # Join with original data
    result = full_df.join(df, on=date_col, how="left")
    
    # Forward fill missing values (crypto doesn't close, so last price carries)
    numeric_cols = [col for col in result.columns if col != date_col]
    if numeric_cols:
        result = result.with_columns([
            pl.col(col).fill_null(strategy="forward") for col in numeric_cols
        ])
    
    return result


if __name__ == "__main__":
    import random
    
    calc = CryptoPerformanceCalculator(
        annualization_factor=365,
        risk_free_rate=0.05,
        max_history_days=500,
    )
    
    # Simulate returns
    for day in range(200):
        date = f"2024-{(day // 30) + 1:02d}-{(day % 30) + 1:02d}"
        # Random return with slight positive drift
        ret = random.gauss(0.001, 0.03)
        calc.add_return(date, ret)
    
    # Calculate metrics
    metrics = calc.calculate_all_metrics()
    if metrics:
        print("Performance Metrics:")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
        print(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")
        print(f"  Omega Ratio: {metrics.omega_ratio:.3f}")
        print(f"  Annualized Return: {metrics.annualized_return:.2%}")
        print(f"  Annualized Vol: {metrics.annualized_vol:.2%}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  VaR 95%: {metrics.var_95:.2%}")
        print(f"  CVaR 95%: {metrics.cvar_95:.2%}")
    
    # Summary
    summary = calc.get_summary_stats()
    print(f"\nSummary: {summary}")
