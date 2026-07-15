"""
Real-time tracker of model predictions vs. actual market outcomes.
Tracks hit rate, profit factor, alpha decay using streaming Polars DataFrames.
Strictly bounded in RAM using rolling windows and periodic flushing.
"""

import polars as pl
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

# Strict memory bounds
MAX_ROWS_PER_SYMBOL = 10_000
FLUSH_THRESHOLD = 50_000


@dataclass
class PerformanceMetrics:
    """Container for real-time performance metrics."""
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    alpha_decay: float = 0.0
    total_predictions: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    cumulative_pnl: float = 0.0
    max_drawdown: float = 0.0


class LivePerformanceTracker:
    """
    Streaming performance tracker for model predictions.
    Uses Polars for efficient in-memory operations with strict RAM bounds.
    """

    def __init__(self, symbols: List[str], window_size: int = 1000):
        self.symbols = symbols
        self.window_size = window_size
        self._dataframes: Dict[str, pl.DataFrame] = {}
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._global_df: Optional[pl.DataFrame] = None
        self._row_counts: Dict[str, int] = {s: 0 for s in symbols}
        
        # Initialize empty DataFrames for each symbol
        for symbol in symbols:
            self._dataframes[symbol] = self._create_empty_dataframe()
            self._metrics[symbol] = PerformanceMetrics()

    @staticmethod
    def _create_empty_dataframe() -> pl.DataFrame:
        """Create schema for prediction tracking DataFrame."""
        return pl.DataFrame(schema={
            'timestamp': pl.Int64,
            'symbol': pl.Utf8,
            'prediction': pl.Float64,
            'actual_return': pl.Float64,
            'pnl': pl.Float64,
            'position_size': pl.Float64,
            'is_winner': pl.Boolean,
            'confidence': pl.Float64,
        })

    def record_prediction(
        self,
        symbol: str,
        prediction: float,
        confidence: float,
        timestamp_ns: int,
    ) -> None:
        """Record a new prediction (pre-outcome)."""
        if symbol not in self._dataframes:
            self._dataframes[symbol] = self._create_empty_dataframe()
            self._metrics[symbol] = PerformanceMetrics()
            self._row_counts[symbol] = 0

        # Pre-register prediction, actual outcome will be filled later
        new_row = pl.DataFrame({
            'timestamp': [timestamp_ns],
            'symbol': [symbol],
            'prediction': [prediction],
            'actual_return': [0.0],  # To be filled on outcome
            'pnl': [0.0],
            'position_size': [0.0],
            'is_winner': [False],
            'confidence': [confidence],
        })
        
        self._dataframes[symbol] = pl.concat([
            self._dataframes[symbol], new_row
        ], how='vertical_relaxed')
        
        self._row_counts[symbol] += 1
        
        # Enforce memory bounds
        if self._row_counts[symbol] > MAX_ROWS_PER_SYMBOL:
            self._trim_dataframe(symbol)

    def record_outcome(
        self,
        symbol: str,
        actual_return: float,
        position_size: float,
        idx: Optional[int] = None,
    ) -> float:
        """Record actual outcome and calculate PnL for a prediction."""
        if symbol not in self._dataframes:
            return 0.0

        df = self._dataframes[symbol]
        
        # Use last unfulfilled prediction if idx not specified
        if idx is None:
            # Find last row where actual_return == 0
            mask = df['actual_return'] == 0.0
            if mask.sum() == 0:
                return 0.0
            idx = mask.arg_max()  # Get last True index

        pnl = actual_return * position_size
        is_winner = pnl > 0

        # Update the row (Polars immutable - recreate with update)
        df = df.with_columns([
            pl.when(pl.col('timestamp') == df[idx, 'timestamp'])
            .then(actual_return)
            .otherwise(pl.col('actual_return')).alias('actual_return'),
            pl.when(pl.col('timestamp') == df[idx, 'timestamp'])
            .then(pnl)
            .otherwise(pl.col('pnl')).alias('pnl'),
            pl.when(pl.col('timestamp') == df[idx, 'timestamp'])
            .then(position_size)
            .otherwise(pl.col('position_size')).alias('position_size'),
            pl.when(pl.col('timestamp') == df[idx, 'timestamp'])
            .then(is_winner)
            .otherwise(pl.col('is_winner')).alias('is_winner'),
        ])
        
        self._dataframes[symbol] = df
        self._update_metrics(symbol)
        
        return pnl

    def _trim_dataframe(self, symbol: str) -> None:
        """Trim DataFrame to maintain memory bounds."""
        df = self._dataframes[symbol]
        if len(df) > MAX_ROWS_PER_SYMBOL:
            # Keep only recent rows
            self._dataframes[symbol] = df.tail(self.window_size)
            self._row_counts[symbol] = self.window_size

    def _update_metrics(self, symbol: str) -> None:
        """Recalculate performance metrics for a symbol."""
        df = self._dataframes[symbol]
        
        if len(df) == 0 or df['pnl'].sum() == 0:
            return

        winners = df.filter(pl.col('is_winner') == True)
        losers = df.filter(pl.col('is_winner') == False)
        
        winning_count = len(winners)
        losing_count = len(losers)
        total_trades = winning_count + losing_count
        
        if total_trades == 0:
            return

        # Hit rate
        hit_rate = winning_count / total_trades

        # Profit factor (gross profit / gross loss)
        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0.0
        gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (annualized)
        pnls = df['pnl'].to_numpy()
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252 * 24)  # Crypto 24/7
        else:
            sharpe_ratio = 0.0

        # Alpha decay (correlation between prediction confidence and actual return)
        if len(df) > 10:
            corr = df.select(pl.corr('confidence', 'actual_return')).item()
            alpha_decay = 1.0 - (corr if corr is not None else 0.0)
        else:
            alpha_decay = 0.0

        # Cumulative PnL
        cumulative_pnl = df['pnl'].sum()

        # Max drawdown
        cumulative = df['pnl'].cum_sum()
        running_max = cumulative.cum_max()
        drawdown = (running_max - cumulative) / running_max.replace(0, 1)
        max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        self._metrics[symbol] = PerformanceMetrics(
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            alpha_decay=alpha_decay,
            total_predictions=len(df),
            winning_trades=winning_count,
            losing_trades=losing_count,
            cumulative_pnl=cumulative_pnl,
            max_drawdown=max_drawdown,
        )

    def get_metrics(self, symbol: str) -> PerformanceMetrics:
        """Get current performance metrics for a symbol."""
        return self._metrics.get(symbol, PerformanceMetrics())

    def get_global_metrics(self) -> PerformanceMetrics:
        """Aggregate metrics across all symbols."""
        all_dfs = [df for df in self._dataframes.values() if len(df) > 0]
        if not all_dfs:
            return PerformanceMetrics()

        combined = pl.concat(all_dfs, how='vertical_relaxed')
        
        winners = combined.filter(pl.col('is_winner') == True)
        losers = combined.filter(pl.col('is_winner') == False)
        
        total = len(combined)
        if total == 0:
            return PerformanceMetrics()

        hit_rate = len(winners) / total
        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0.0
        gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0.01
        
        return PerformanceMetrics(
            hit_rate=hit_rate,
            profit_factor=gross_profit / gross_loss,
            cumulative_pnl=combined['pnl'].sum(),
            total_predictions=total,
            winning_trades=len(winners),
            losing_trades=len(losers),
        )

    def flush_old_data(self, max_age_hours: int = 24) -> int:
        """Remove data older than specified age to free memory."""
        now_ns = int(datetime.now().timestamp() * 1e9)
        max_age_ns = max_age_hours * 3600 * 1e9
        cutoff = now_ns - max_age_ns
        
        flushed = 0
        for symbol in self.symbols:
            df = self._dataframes[symbol]
            old_count = len(df)
            self._dataframes[symbol] = df.filter(pl.col('timestamp') >= cutoff)
            flushed += old_count - len(self._dataframes[symbol])
            self._row_counts[symbol] = len(self._dataframes[symbol])
        
        return flushed


if __name__ == "__main__":
    # Example usage
    tracker = LivePerformanceTracker(['BTC-USDT', 'ETH-USDT'])
    
    # Record predictions
    tracker.record_prediction('BTC-USDT', 0.02, 0.85, int(datetime.now().timestamp() * 1e9))
    tracker.record_outcome('BTC-USDT', 0.015, 1.0)
    
    metrics = tracker.get_metrics('BTC-USDT')
    print(f"Hit Rate: {metrics.hit_rate:.2%}")
    print(f"Cumulative PnL: {metrics.cumulative_pnl:.4f}")
