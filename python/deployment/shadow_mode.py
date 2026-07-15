"""
Shadow testing framework that runs newly trained models in parallel
with live models, logging hypothetical trades and PnL without
executing them on the exchange.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from pathlib import Path

# Memory bounds
MAX_SHADOW_TRADES = 10_000


@dataclass
class ShadowTrade:
    """Represents a hypothetical trade from shadow mode."""
    timestamp_ns: int
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    model_version: str = ''
    confidence: float = 0.0
    closed: bool = False


@dataclass
class ShadowMetrics:
    """Aggregated metrics from shadow testing."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    avg_trade_duration_ns: int = 0


class ShadowModeTester:
    """
    Runs models in shadow mode to validate performance before
    deploying to live trading.
    """

    def __init__(self, log_dir: str = './shadow_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._open_positions: Dict[str, ShadowTrade] = {}
        self._closed_trades: List[ShadowTrade] = []
        self._pnl_series: List[float] = []
        self._lock = threading.RLock()
        
        self.model_version = ''
        self.start_time_ns = 0

    def start_shadow_session(self, model_version: str) -> None:
        """Begin a new shadow testing session."""
        with self._lock:
            self.model_version = model_version
            self.start_time_ns = int(datetime.now().timestamp() * 1e9)
            self._open_positions.clear()
            self._closed_trades.clear()
            self._pnl_series.clear()

    def process_signal(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        current_price: float,
        timestamp_ns: Optional[int] = None,
    ) -> Optional[ShadowTrade]:
        """
        Process a model signal and create hypothetical trade.
        Positive signal = buy, negative = sell.
        """
        if timestamp_ns is None:
            timestamp_ns = int(datetime.now().timestamp() * 1e9)
        
        with self._lock:
            side = 'buy' if signal > 0 else 'sell'
            
            # Check if we have an opposite position to close
            if symbol in self._open_positions:
                existing = self._open_positions[symbol]
                
                # Close if signal reverses
                if (side == 'buy' and existing.side == 'sell') or \
                   (side == 'sell' and existing.side == 'buy'):
                    self._close_position(symbol, current_price, timestamp_ns)
            
            # Open new position if no existing one
            if symbol not in self._open_positions:
                quantity = abs(signal) * confidence  # Size by signal strength
                trade = ShadowTrade(
                    timestamp_ns=timestamp_ns,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=current_price,
                    model_version=self.model_version,
                    confidence=confidence,
                )
                self._open_positions[symbol] = trade
                return trade
            
            return None

    def update_price(self, symbol: str, price: float, timestamp_ns: Optional[int] = None) -> float:
        """
        Update price for open positions and calculate unrealized PnL.
        Returns unrealized PnL for the symbol.
        """
        if timestamp_ns is None:
            timestamp_ns = int(datetime.now().timestamp() * 1e9)
        
        with self._lock:
            if symbol not in self._open_positions:
                return 0.0
            
            position = self._open_positions[symbol]
            
            if position.side == 'buy':
                unrealized_pnl = (price - position.entry_price) * position.quantity
            else:
                unrealized_pnl = (position.entry_price - price) * position.quantity
            
            return unrealized_pnl

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp_ns: int,
    ) -> ShadowTrade:
        """Close an open position and record the trade."""
        if symbol not in self._open_positions:
            return None
        
        position = self._open_positions.pop(symbol)
        position.exit_price = exit_price
        position.closed = True
        
        # Calculate PnL
        if position.side == 'buy':
            position.pnl = (exit_price - position.entry_price) * position.quantity
        else:
            position.pnl = (position.entry_price - exit_price) * position.quantity
        
        # Record trade
        self._closed_trades.append(position)
        self._pnl_series.append(position.pnl)
        
        # Enforce memory bound
        if len(self._closed_trades) > MAX_SHADOW_TRADES:
            self._closed_trades = self._closed_trades[-MAX_SHADOW_TRADES:]
        if len(self._pnl_series) > MAX_SHADOW_TRADES:
            self._pnl_series = self._pnl_series[-MAX_SHADOW_TRADES:]
        
        return position

    def close_all_positions(self, prices: Dict[str, float], timestamp_ns: Optional[int] = None) -> List[ShadowTrade]:
        """Close all open positions at given prices."""
        if timestamp_ns is None:
            timestamp_ns = int(datetime.now().timestamp() * 1e9)
        
        closed = []
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._open_positions:
                    trade = self._close_position(symbol, price, timestamp_ns)
                    closed.append(trade)
        return closed

    def get_metrics(self) -> ShadowMetrics:
        """Calculate aggregated shadow metrics."""
        with self._lock:
            if not self._closed_trades:
                return ShadowMetrics()
            
            pnls = [t.pnl for t in self._closed_trades]
            winners = [t for t in self._closed_trades if t.pnl > 0]
            losers = [t for t in self._closed_trades if t.pnl <= 0]
            
            # Sharpe ratio
            if len(pnls) > 1 and np.std(pnls) > 0:
                sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252 * 24)
            else:
                sharpe = 0.0
            
            # Max drawdown
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / np.maximum(running_max, 1e-8)
            max_dd = float(np.max(drawdowns))
            
            # Avg duration
            durations = [
                t.timestamp_ns - (int(t.exit_price is not None) * t.timestamp_ns)
                for t in self._closed_trades if t.closed
            ]
            avg_duration = int(np.mean(durations)) if durations else 0
            
            return ShadowMetrics(
                total_trades=len(self._closed_trades),
                winning_trades=len(winners),
                losing_trades=len(losers),
                total_pnl=sum(pnls),
                avg_pnl=np.mean(pnls),
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                hit_rate=len(winners) / len(self._closed_trades),
                avg_trade_duration_ns=avg_duration,
            )

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save shadow test results to file."""
        if filename is None:
            filename = f"shadow_{self.model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        metrics = self.get_metrics()
        
        results = {
            'model_version': self.model_version,
            'start_time_ns': self.start_time_ns,
            'end_time_ns': int(datetime.now().timestamp() * 1e9),
            'metrics': {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'total_pnl': metrics.total_pnl,
                'avg_pnl': metrics.avg_pnl,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'hit_rate': metrics.hit_rate,
            },
            'trades': [
                {
                    'timestamp_ns': t.timestamp_ns,
                    'symbol': t.symbol,
                    'side': t.side,
                    'quantity': t.quantity,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'confidence': t.confidence,
                }
                for t in self._closed_trades[-1000:]  # Last 1000 trades only
            ],
            'open_positions': [
                {
                    'symbol': s,
                    'side': p.side,
                    'entry_price': p.entry_price,
                    'unrealized_pnl': self.update_price(s, p.entry_price),
                }
                for s, p in self._open_positions.items()
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return str(filepath)

    def compare_to_live(self, live_metrics: Dict) -> Dict:
        """Compare shadow metrics to live trading metrics."""
        shadow = self.get_metrics()
        
        comparison = {
            'sharpe_diff': shadow.sharpe_ratio - live_metrics.get('sharpe_ratio', 0),
            'hit_rate_diff': shadow.hit_rate - live_metrics.get('hit_rate', 0),
            'max_dd_diff': shadow.max_drawdown - live_metrics.get('max_drawdown', 0),
            'shadow_sharpe': shadow.sharpe_ratio,
            'live_sharpe': live_metrics.get('sharpe_ratio', 0),
            'shadow_hit_rate': shadow.hit_rate,
            'live_hit_rate': live_metrics.get('hit_rate', 0),
        }
        
        # Deployment recommendation
        if comparison['sharpe_diff'] > 0.1 and comparison['max_dd_diff'] < 0.05:
            comparison['recommendation'] = 'DEPLOY'
        elif comparison['sharpe_diff'] < -0.1:
            comparison['recommendation'] = 'REJECT'
        else:
            comparison['recommendation'] = 'MORE_DATA_NEEDED'
        
        return comparison


if __name__ == "__main__":
    # Example usage
    tester = ShadowModeTester(log_dir='./test_shadow')
    tester.start_shadow_session('v2.5.0')
    
    # Simulate signals and prices
    np.random.seed(42)
    base_price = 50000
    
    for i in range(100):
        timestamp = int(datetime.now().timestamp() * 1e9) + i * 60_000_000_000
        signal = np.random.randn() * 0.1
        confidence = np.random.rand() * 0.5 + 0.5
        price = base_price + np.cumsum(np.random.randn(i+1) * 50)[-1]
        
        tester.process_signal('BTC-USDT', signal, confidence, price, timestamp)
        tester.update_price('BTC-USDT', price, timestamp)
    
    # Close and get metrics
    tester.close_all_positions({'BTC-USDT': base_price})
    metrics = tester.get_metrics()
    
    print(f"Shadow Test Results:")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Hit Rate: {metrics.hit_rate:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"  Total PnL: {metrics.total_pnl:.2f}")
    
    # Save results
    filepath = tester.save_results()
    print(f"\nResults saved to: {filepath}")
