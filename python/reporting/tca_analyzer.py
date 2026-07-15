"""
Transaction Cost Analysis (TCA) engine comparing simulated execution prices against 
theoretical VWAP/TWAP benchmarks, highlighting hidden slippage, market impact costs, and adverse selection.
"""

import polars as pl
from polars import col
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TCAResult:
    """Container for TCA analysis results."""
    # Slippage metrics
    total_slippage_bps: float
    avg_slippage_bps: float
    slippage_std: float
    
    # Market impact
    estimated_market_impact_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    
    # Benchmark comparisons
    vwap_slippage_bps: float
    twap_slippage_bps: float
    arrival_cost_bps: float
    
    # Adverse selection
    adverse_selection_cost_bps: float
    informed_trading_probability: float
    
    # Execution quality
    implementation_shortfall_bps: float
    effective_spread_bps: float
    realized_spread_bps: float
    
    # Summary
    total_transaction_cost_bps: float
    cost_as_pct_of_pnl: float


@dataclass
class TradeAnalysis:
    """Per-trade TCA breakdown."""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    execution_price: float
    benchmark_vwap: float
    benchmark_twap: float
    arrival_price: float
    slippage_vs_vwap_bps: float
    slippage_vs_twap_bps: float
    market_impact_bps: float
    timing_cost_bps: float


class TCAAnalyzer:
    """
    Transaction Cost Analysis engine for evaluating execution quality.
    
    Analyzes:
    - Implementation shortfall
    - VWAP/TWAP benchmark comparison
    - Market impact decomposition
    - Adverse selection costs
    - Effective vs realized spread
    """
    
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
    
    def analyze_trades(
        self,
        trades_df: pl.DataFrame,
        market_data_df: Optional[pl.DataFrame] = None,
    ) -> TCAResult:
        """
        Perform comprehensive TCA on trade data.
        
        Parameters
        ----------
        trades_df : pl.DataFrame
            DataFrame with columns: timestamp, symbol, side, quantity, price, pnl
        market_data_df : Optional[pl.DataFrame]
            Market data for benchmark calculations (OHLCV)
            
        Returns
        -------
        TCAResult
            Complete TCA analysis.
        """
        if trades_df.is_empty():
            return self._empty_result()
        
        # Calculate per-trade metrics
        trade_analyses = self._analyze_individual_trades(trades_df, market_data_df)
        
        # Aggregate to summary statistics
        return self._aggregate_results(trade_analyses, trades_df)
    
    def _analyze_individual_trades(
        self,
        trades_df: pl.DataFrame,
        market_data_df: Optional[pl.DataFrame],
    ) -> List[TradeAnalysis]:
        """Analyze each trade individually."""
        analyses = []
        
        # Convert to pandas for easier iteration (or use Polars throughout)
        trades_pd = trades_df.to_pandas()
        
        for idx, row in trades_pd.iterrows():
            # Get benchmark prices
            vwap_bench = self._get_vwap_benchmark(market_data_df, row['timestamp'], row['symbol']) if market_data_df is not None else row['price']
            twap_bench = self._get_twap_benchmark(market_data_df, row['timestamp'], row['symbol']) if market_data_df is not None else row['price']
            arrival_price = self._get_arrival_price(market_data_df, row['timestamp'], row['symbol']) if market_data_df is not None else row['price']
            
            exec_price = row['price']
            
            # Calculate slippages (direction-aware)
            if row['side'].upper() == 'BUY':
                slippage_vwap = (exec_price - vwap_bench) / vwap_bench * 10000 if vwap_bench > 0 else 0
                slippage_twap = (exec_price - twap_bench) / twap_bench * 10000 if twap_bench > 0 else 0
            else:  # SELL
                slippage_vwap = (vwap_bench - exec_price) / vwap_bench * 10000 if vwap_bench > 0 else 0
                slippage_twap = (twap_bench - exec_price) / twap_bench * 10000 if twap_bench > 0 else 0
            
            # Market impact estimation
            market_impact = abs(exec_price - arrival_price) / arrival_price * 10000 if arrival_price > 0 else 0
            
            # Timing cost
            timing_cost = abs(vwap_bench - twap_bench) / max(vwap_bench, twap_bench) * 10000 if max(vwap_bench, twap_bench) > 0 else 0
            
            analysis = TradeAnalysis(
                trade_id=f"trade_{idx}",
                symbol=row.get('symbol', 'UNKNOWN'),
                side=row['side'],
                quantity=row['quantity'],
                execution_price=exec_price,
                benchmark_vwap=vwap_bench,
                benchmark_twap=twap_bench,
                arrival_price=arrival_price,
                slippage_vs_vwap_bps=slippage_vwap,
                slippage_vs_twap_bps=slippage_twap,
                market_impact_bps=market_impact,
                timing_cost_bps=timing_cost,
            )
            analyses.append(analysis)
        
        return analyses
    
    def _get_vwap_benchmark(
        self,
        market_data: pl.DataFrame,
        timestamp: int,
        symbol: str,
    ) -> float:
        """Get VWAP benchmark for the trade period."""
        if market_data is None:
            return 0.0
        
        # Filter to relevant time window (e.g., 5-minute bar containing timestamp)
        # Simplified: return close price
        try:
            bar = market_data.filter(
                (col('timestamp') <= timestamp) & 
                (col('symbol') == symbol)
            ).sort('timestamp').tail(1)
            
            if not bar.is_empty():
                volume = bar['volume'].item()
                dollar_volume = bar['close'].item() * volume
                return dollar_volume / volume if volume > 0 else bar['close'].item()
        except Exception:
            pass
        
        return 0.0
    
    def _get_twap_benchmark(
        self,
        market_data: pl.DataFrame,
        timestamp: int,
        symbol: str,
    ) -> float:
        """Get TWAP benchmark for the trade period."""
        if market_data is None:
            return 0.0
        
        try:
            # Average of OHLC over the period
            bar = market_data.filter(
                (col('timestamp') <= timestamp) & 
                (col('symbol') == symbol)
            ).sort('timestamp').tail(1)
            
            if not bar.is_empty():
                ohlc_avg = (bar['open'].item() + bar['high'].item() + bar['low'].item() + bar['close'].item()) / 4
                return ohlc_avg
        except Exception:
            pass
        
        return 0.0
    
    def _get_arrival_price(
        self,
        market_data: pl.DataFrame,
        timestamp: int,
        symbol: str,
    ) -> float:
        """Get arrival price (price at order submission time)."""
        if market_data is None:
            return 0.0
        
        try:
            bar = market_data.filter(
                (col('timestamp') <= timestamp) & 
                (col('symbol') == symbol)
            ).sort('timestamp').tail(1)
            
            if not bar.is_empty():
                return bar['open'].item()
        except Exception:
            pass
        
        return 0.0
    
    def _aggregate_results(
        self,
        analyses: List[TradeAnalysis],
        trades_df: pl.DataFrame,
    ) -> TCAResult:
        """Aggregate individual trade analyses into summary metrics."""
        if not analyses:
            return self._empty_result()
        
        # Extract slippage arrays
        vwap_slippages = [a.slippage_vs_vwap_bps for a in analyses]
        twap_slippages = [a.slippage_vs_twap_bps for a in analyses]
        market_impacts = [a.market_impact_bps for a in analyses]
        
        # Total transaction cost
        total_slippage = sum(abs(s) for s in vwap_slippages)
        avg_slippage = np.mean(vwap_slippages)
        slippage_std = np.std(vwap_slippages)
        
        # Market impact decomposition
        est_market_impact = np.mean(market_impacts)
        temp_impact = est_market_impact * 0.7  # Temporary (reverts)
        perm_impact = est_market_impact * 0.3  # Permanent
        
        # VWAP/TWAP comparison
        vwap_slippage = np.mean(vwap_slippages)
        twap_slippage = np.mean(twap_slippages)
        
        # Arrival cost (vs price at order arrival)
        arrival_costs = [abs(a.execution_price - a.arrival_price) / a.arrival_price * 10000 
                        for a in analyses if a.arrival_price > 0]
        arrival_cost = np.mean(arrival_costs) if arrival_costs else 0
        
        # Adverse selection (cost from trading against informed counterparties)
        # Estimated from post-trade price movement
        adverse_selection = self._estimate_adverse_selection(analyses)
        
        # Implementation shortfall
        impl_shortfall = total_slippage / len(analyses)
        
        # Spread measures
        effective_spread = est_market_impact * 2  # Round-trip estimate
        realized_spread = effective_spread - perm_impact
        
        # Total cost as % of PnL
        total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0
        total_notional = sum(a.quantity * a.execution_price for a in analyses)
        total_cost_bps = total_slippage
        cost_pct_pnl = (total_cost_bps / 10000 * total_notional) / abs(total_pnl) * 100 if total_pnl != 0 else 0
        
        return TCAResult(
            total_slippage_bps=total_slippage,
            avg_slippage_bps=avg_slippage,
            slippage_std=slippage_std,
            estimated_market_impact_bps=est_market_impact,
            temporary_impact_bps=temp_impact,
            permanent_impact_bps=perm_impact,
            vwap_slippage_bps=vwap_slippage,
            twap_slippage_bps=twap_slippage,
            arrival_cost_bps=arrival_cost,
            adverse_selection_cost_bps=adverse_selection,
            informed_trading_probability=self._estimate_informed_trading(adverse_selection),
            implementation_shortfall_bps=impl_shortfall,
            effective_spread_bps=effective_spread,
            realized_spread_bps=realized_spread,
            total_transaction_cost_bps=total_cost_bps,
            cost_as_pct_of_pnl=cost_pct_pnl,
        )
    
    def _estimate_adverse_selection(self, analyses: List[TradeAnalysis]) -> float:
        """Estimate adverse selection cost from trade patterns."""
        if not analyses:
            return 0.0
        
        # Simplified: look at consistent negative slippage
        negative_slippage_count = sum(1 for a in analyses if a.slippage_vs_vwap_bps < 0)
        negative_ratio = negative_slippage_count / len(analyses)
        
        # Higher ratio suggests adverse selection
        base_cost = np.mean([abs(a.slippage_vs_vwap_bps) for a in analyses if a.slippage_vs_vwap_bps < 0])
        return base_cost * negative_ratio if negative_ratio > 0.5 else 0
    
    def _estimate_informed_trading(self, adverse_selection: float) -> float:
        """Estimate probability of trading against informed counterparties."""
        # PIN (Probability of Informed Trading) approximation
        # Higher adverse selection = higher PIN
        if adverse_selection <= 0:
            return 0.1  # Base rate
        
        # Map adverse selection to probability (simplified logistic)
        pin = 1 / (1 + np.exp(-adverse_selection / 5))
        return min(0.9, max(0.1, pin))
    
    def _empty_result(self) -> TCAResult:
        """Return empty result for edge cases."""
        return TCAResult(
            total_slippage_bps=0.0,
            avg_slippage_bps=0.0,
            slippage_std=0.0,
            estimated_market_impact_bps=0.0,
            temporary_impact_bps=0.0,
            permanent_impact_bps=0.0,
            vwap_slippage_bps=0.0,
            twap_slippage_bps=0.0,
            arrival_cost_bps=0.0,
            adverse_selection_cost_bps=0.0,
            informed_trading_probability=0.0,
            implementation_shortfall_bps=0.0,
            effective_spread_bps=0.0,
            realized_spread_bps=0.0,
            total_transaction_cost_bps=0.0,
            cost_as_pct_of_pnl=0.0,
        )
    
    def generate_tca_report(
        self,
        trades_df: pl.DataFrame,
        market_data_df: Optional[pl.DataFrame] = None,
    ) -> Dict:
        """Generate a comprehensive TCA report."""
        result = self.analyze_trades(trades_df, market_data_df)
        
        report = {
            'summary': {
                'total_slippage_bps': result.total_slippage_bps,
                'avg_slippage_bps': result.avg_slippage_bps,
                'total_transaction_cost_bps': result.total_transaction_cost_bps,
                'cost_as_pct_of_pnl': result.cost_as_pct_of_pnl,
            },
            'benchmark_comparison': {
                'vwap_slippage_bps': result.vwap_slippage_bps,
                'twap_slippage_bps': result.twap_slippage_bps,
                'arrival_cost_bps': result.arrival_cost_bps,
            },
            'market_impact': {
                'estimated_impact_bps': result.estimated_market_impact_bps,
                'temporary_impact_bps': result.temporary_impact_bps,
                'permanent_impact_bps': result.permanent_impact_bps,
            },
            'adverse_selection': {
                'cost_bps': result.adverse_selection_cost_bps,
                'informed_trading_prob': result.informed_trading_probability,
            },
            'execution_quality': {
                'implementation_shortfall_bps': result.implementation_shortfall_bps,
                'effective_spread_bps': result.effective_spread_bps,
                'realized_spread_bps': result.realized_spread_bps,
            },
        }
        
        logger.info(f"TCA Report generated: Total cost = {result.total_transaction_cost_bps:.2f} bps")
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample trade data
    import pandas as pd
    
    sample_trades = pl.DataFrame({
        'timestamp': list(range(100)),
        'symbol': ['BTCUSDT'] * 100,
        'side': ['BUY'] * 50 + ['SELL'] * 50,
        'quantity': [0.1] * 100,
        'price': [50000 + np.random.normal(0, 50) for _ in range(100)],
        'pnl': np.random.normal(10, 100, 100),
    })
    
    # Create sample market data
    sample_market = pl.DataFrame({
        'timestamp': list(range(100)),
        'symbol': ['BTCUSDT'] * 100,
        'open': [50000 + np.random.normal(0, 50) for _ in range(100)],
        'high': [50100 + np.random.normal(0, 50) for _ in range(100)],
        'low': [49900 + np.random.normal(0, 50) for _ in range(100)],
        'close': [50000 + np.random.normal(0, 50) for _ in range(100)],
        'volume': [1000 + np.random.normal(0, 100) for _ in range(100)],
    })
    
    # Run TCA
    analyzer = TCAAnalyzer(tick_size=0.01)
    report = analyzer.generate_tca_report(sample_trades, sample_market)
    
    print("TCA Report:")
    for section, metrics in report.items():
        print(f"\n{section.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
