"""
Trade Distribution Analyzer: Generates statistical distributions of trade PnL,
holding times, and MAE/MFE for frontend histograms.
Strictly caps memory usage during large dataset aggregations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from collections import deque
import polars as pl

# Memory bounds
MAX_TRADES = 100000
MAX_BINS = 100


@dataclass
class TradeMetrics:
    """Container for single trade metrics."""
    trade_id: str
    symbol: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    mae: float  # Maximum Adverse Excursion
    mfe: float  # Maximum Favorable Excursion
    holding_time_us: int
    side: int  # 0=long, 1=short


class TradeDistributionAnalyzer:
    """
    Memory-efficient trade distribution analyzer.
    Uses bounded deques and incremental histogram updates.
    """
    
    def __init__(self, max_trades: int = MAX_TRADES):
        self.max_trades = max_trades
        self._trades: deque[TradeMetrics] = deque(maxlen=max_trades)
        
        # Pre-computed histograms (bounded)
        self._pnl_histogram: Dict[str, int] = {}
        self._holding_time_histogram: Dict[str, int] = {}
        self._mae_histogram: Dict[str, int] = {}
        self._mfe_histogram: Dict[str, int] = {}
        
        # Running statistics
        self._total_pnl: float = 0.0
        self._total_pnl_sq: float = 0.0
        self._win_count: int = 0
        self._loss_count: int = 0
        
    def add_trade(self, trade: TradeMetrics):
        """Add a single trade and update statistics."""
        self._trades.append(trade)
        
        # Update running statistics
        self._total_pnl += trade.pnl
        self._total_pnl_sq += trade.pnl * trade.pnl
        
        if trade.pnl > 0:
            self._win_count += 1
        elif trade.pnl < 0:
            self._loss_count += 1
        
        # Periodically rebuild histograms (every 100 trades)
        if len(self._trades) % 100 == 0:
            self._rebuild_histograms()
    
    def add_batch(self, trades: List[TradeMetrics]):
        """Add a batch of trades."""
        for trade in trades[-self.max_trades:]:
            self.add_trade(trade)
    
    def _rebuild_histograms(self):
        """Rebuild all histograms from current trade data."""
        if len(self._trades) == 0:
            return
        
        trades_list = list(self._trades)
        
        # PnL histogram
        pnl_values = [t.pnl for t in trades_list]
        self._pnl_histogram = self._create_histogram(pnl_values, MAX_BINS)
        
        # Holding time histogram (in seconds)
        holding_times = [t.holding_time_us / 1_000_000 for t in trades_list]
        self._holding_time_histogram = self._create_histogram(holding_times, MAX_BINS)
        
        # MAE histogram
        mae_values = [abs(t.mae) for t in trades_list]
        self._mae_histogram = self._create_histogram(mae_values, MAX_BINS)
        
        # MFE histogram
        mfe_values = [t.mfe for t in trades_list]
        self._mfe_histogram = self._create_histogram(mfe_values, MAX_BINS)
    
    def _create_histogram(self, values: List[float], n_bins: int) -> Dict[str, int]:
        """Create a histogram with fixed number of bins."""
        if len(values) == 0:
            return {}
        
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return {str(min_val): len(values)}
        
        bin_width = (max_val - min_val) / n_bins
        histogram = {}
        
        for val in values:
            bin_idx = min(int((val - min_val) / bin_width), n_bins - 1)
            bin_key = f"{min_val + bin_idx * bin_width:.4f}"
            histogram[bin_key] = histogram.get(bin_key, 0) + 1
        
        return histogram
    
    def get_pnl_statistics(self) -> Dict:
        """Get comprehensive PnL statistics."""
        if len(self._trades) == 0:
            return {'count': 0}
        
        trades_list = list(self._trades)
        pnl_values = [t.pnl for t in trades_list]
        pnl_pcts = [t.pnl_pct for t in trades_list]
        
        mean_pnl = np.mean(pnl_values)
        std_pnl = np.std(pnl_values)
        skew_pnl = self._calculate_skewness(pnl_values)
        kurt_pnl = self._calculate_kurtosis(pnl_values)
        
        return {
            'count': len(trades_list),
            'mean_pnl': float(mean_pnl),
            'std_pnl': float(std_pnl),
            'skewness': float(skew_pnl),
            'kurtosis': float(kurt_pnl),
            'min_pnl': float(min(pnl_values)),
            'max_pnl': float(max(pnl_values)),
            'median_pnl': float(np.median(pnl_values)),
            'mean_pnl_pct': float(np.mean(pnl_pcts)),
            'total_pnl': float(self._total_pnl),
            'win_rate': self.get_win_rate(),
            'profit_factor': self.get_profit_factor(trades_list)
        }
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate sample skewness."""
        if len(values) < 3:
            return 0.0
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std < 1e-10:
            return 0.0
        skew = sum(((x - mean) / std) ** 3 for x in values) * n / ((n - 1) * (n - 2))
        return float(skew)
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate excess kurtosis."""
        if len(values) < 4:
            return 0.0
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std < 1e-10:
            return 0.0
        kurt = sum(((x - mean) / std) ** 4 for x in values) / n
        excess_kurt = kurt - 3.0
        return float(excess_kurt)
    
    def get_holding_time_statistics(self) -> Dict:
        """Get holding time statistics."""
        if len(self._trades) == 0:
            return {'count': 0}
        
        trades_list = list(self._trades)
        holding_times = [t.holding_time_us / 1_000_000 for t in trades_list]  # seconds
        
        return {
            'mean_seconds': float(np.mean(holding_times)),
            'median_seconds': float(np.median(holding_times)),
            'std_seconds': float(np.std(holding_times)),
            'min_seconds': float(min(holding_times)),
            'max_seconds': float(max(holding_times)),
            'percentile_95': float(np.percentile(holding_times, 95)),
        }
    
    def get_mae_mfe_statistics(self) -> Dict:
        """Get MAE/MFE statistics."""
        if len(self._trades) == 0:
            return {'count': 0}
        
        trades_list = list(self._trades)
        mae_values = [abs(t.mae) for t in trades_list]
        mfe_values = [t.mfe for t in trades_list]
        
        # MAE/MFE ratio (lower is better - means we capture more of favorable move)
        mae_mfe_ratio = []
        for t in trades_list:
            if abs(t.mfe) > 1e-10:
                mae_mfe_ratio.append(abs(t.mae) / abs(t.mfe))
        
        return {
            'mean_mae': float(np.mean(mae_values)),
            'mean_mfe': float(np.mean(mfe_values)),
            'mae_mfe_ratio': float(np.mean(mae_mfe_ratio)) if mae_mfe_ratio else None,
            'avg_mae_pct': float(np.mean([abs(t.mae) for t in trades_list])),
            'avg_mfe_pct': float(np.mean([t.mfe for t in trades_list])),
        }
    
    def get_win_rate(self) -> float:
        """Calculate win rate."""
        total = self._win_count + self._loss_count
        if total == 0:
            return 0.0
        return self._win_count / total
    
    def get_profit_factor(self, trades_list: Optional[List[TradeMetrics]] = None) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if trades_list is None:
            trades_list = list(self._trades)
        
        if len(trades_list) == 0:
            return 0.0
        
        gross_profit = sum(t.pnl for t in trades_list if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades_list if t.pnl < 0))
        
        if gross_loss < 1e-10:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_comprehensive_stats(self) -> Dict:
        """Get all statistics for UI display."""
        self._rebuild_histograms()
        
        return {
            'pnl': self.get_pnl_statistics(),
            'holding_time': self.get_holding_time_statistics(),
            'mae_mfe': self.get_mae_mfe_statistics(),
            'win_rate': self.get_win_rate(),
            'profit_factor': self.get_profit_factor(),
            'sharpe_approximation': self._approximate_sharpe()
        }
    
    def _approximate_sharpe(self) -> float:
        """Approximate Sharpe ratio from trade PnLs."""
        if len(self._trades) < 2:
            return 0.0
        
        pnl_values = [t.pnl for t in self._trades]
        mean_pnl = np.mean(pnl_values)
        std_pnl = np.std(pnl_values, ddof=1)
        
        if std_pnl < 1e-10:
            return 0.0
        
        # Annualize (assuming ~252 trading days, average trades per day)
        trades_per_day = len(self._trades) / max(1, self._get_days_span())
        annualization = np.sqrt(252 * max(1, trades_per_day))
        
        return float(mean_pnl / std_pnl * annualization)
    
    def _get_days_span(self) -> int:
        """Get the number of days spanned by trades."""
        if len(self._trades) < 2:
            return 1
        
        trades_list = list(self._trades)
        min_time = min(t.entry_time for t in trades_list)
        max_time = max(t.exit_time for t in trades_list)
        
        return max(1, (max_time - min_time) // (24 * 3600 * 1_000_000))
    
    def export_to_json(self) -> str:
        """Export distribution data to compact JSON for UI histograms."""
        self._rebuild_histograms()
        
        output = {
            'histograms': {
                'pnl': self._pnl_histogram,
                'holding_time': self._holding_time_histogram,
                'mae': self._mae_histogram,
                'mfe': self._mfe_histogram
            },
            'statistics': self.get_comprehensive_stats(),
            'sample_size': len(self._trades)
        }
        
        return json.dumps(output, separators=(',', ':'))
    
    def clear(self):
        """Clear all stored data."""
        self._trades.clear()
        self._pnl_histogram.clear()
        self._holding_time_histogram.clear()
        self._mae_histogram.clear()
        self._mfe_histogram.clear()
        self._total_pnl = 0.0
        self._total_pnl_sq = 0.0
        self._win_count = 0
        self._loss_count = 0


if __name__ == '__main__':
    # Example usage with synthetic trades
    import random
    import string
    
    analyzer = TradeDistributionAnalyzer()
    
    # Generate 1000 synthetic trades
    for i in range(1000):
        pnl = random.gauss(10, 50)  # Mean $10, std $50
        mfe = abs(pnl) * random.uniform(1.0, 3.0) if pnl > 0 else abs(pnl) * random.uniform(0.3, 1.0)
        mae = abs(pnl) * random.uniform(0.5, 2.0)
        
        trade = TradeMetrics(
            trade_id=f"T{i:06d}",
            symbol="BTC-PERP",
            entry_time=1700000000000000 + i * 3600000000,
            exit_time=1700000000000000 + (i + 1) * 3600000000,
            entry_price=50000.0,
            exit_price=50000.0 + pnl,
            quantity=1.0,
            pnl=pnl,
            pnl_pct=pnl / 50000.0,
            mae=-mae,
            mfe=mfe,
            holding_time_us=3600000000,  # 1 hour
            side=random.randint(0, 1)
        )
        analyzer.add_trade(trade)
    
    stats = analyzer.get_comprehensive_stats()
    print(f"Win Rate: {stats['win_rate']:.2%}")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Sharpe Approx: {stats['sharpe_approximation']:.2f}")
    
    json_output = analyzer.export_to_json()
    print(f"JSON size: {len(json_output)} bytes")
