"""
Strategy Decay Monitor
Rolling Sharpe, Sortino, and Maximum Drawdown tracker
Automatically scales down capital or disables sub-strategies experiencing alpha decay
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class DecayStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DISABLED = "disabled"


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy"""
    strategy_id: str
    rolling_sharpe: float
    rolling_sortino: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    recent_returns: List[float]
    timestamp_ns: int


@dataclass
class DecayAssessment:
    """Assessment of strategy decay status"""
    strategy_id: str
    status: DecayStatus
    sharpe_degradation: float  # % decline from peak
    sortino_degradation: float
    drawdown_severity: float
    recommended_action: str
    capital_scale: float  # 0.0 to 1.0
    should_disable: bool
    timestamp_ns: int


class StrategyDecayMonitor:
    """
    Rolling Sharpe, Sortino, and Maximum Drawdown tracker
    Automatically scales down capital allocation or disables strategies
    experiencing alpha decay, overfitting, or regime mismatch
    """
    
    def __init__(
        self,
        lookback_periods: int = 100,
        short_window: int = 20,
        long_window: int = 100,
        sharpe_threshold_warning: float = 0.5,
        sharpe_threshold_critical: float = 0.0,
        sortino_threshold_warning: float = 0.7,
        sortino_threshold_critical: float = 0.3,
        max_drawdown_warning: float = 0.05,
        max_drawdown_critical: float = 0.10,
        degradation_threshold: float = 0.5,  # 50% decline from peak
    ):
        self.lookback_periods = lookback_periods
        self.short_window = short_window
        self.long_window = long_window
        
        # Thresholds
        self.sharpe_threshold_warning = sharpe_threshold_warning
        self.sharpe_threshold_critical = sharpe_threshold_critical
        self.sortino_threshold_warning = sortino_threshold_warning
        self.sortino_threshold_critical = sortino_threshold_critical
        self.max_drawdown_warning = max_drawdown_warning
        self.max_drawdown_critical = max_drawdown_critical
        self.degradation_threshold = degradation_threshold
        
        # Strategy data storage
        self.strategy_returns: Dict[str, List[float]] = {}
        self.strategy_equity: Dict[str, List[float]] = {}
        self.peak_sharpes: Dict[str, float] = {}
        self.peak_sortinos: Dict[str, float] = {}
        self.assessments: Dict[str, DecayAssessment] = {}
        
        # Status tracking
        self.strategy_status: Dict[str, DecayStatus] = {}
    
    def register_strategy(self, strategy_id: str) -> None:
        """Register a new strategy for monitoring"""
        if strategy_id not in self.strategy_returns:
            self.strategy_returns[strategy_id] = []
            self.strategy_equity[strategy_id] = [1.0]  # Start at normalized 1.0
            self.peak_sharpes[strategy_id] = 0.0
            self.peak_sortinos[strategy_id] = 0.0
            self.strategy_status[strategy_id] = DecayStatus.HEALTHY
    
    def unregister_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from monitoring"""
        if strategy_id in self.strategy_returns:
            del self.strategy_returns[strategy_id]
        if strategy_id in self.strategy_equity:
            del self.strategy_equity[strategy_id]
        if strategy_id in self.peak_sharpes:
            del self.peak_sharpes[strategy_id]
        if strategy_id in self.peak_sortinos:
            del self.peak_sortinos[strategy_id]
        if strategy_id in self.assessments:
            del self.assessments[strategy_id]
        if strategy_id in self.strategy_status:
            del self.strategy_status[strategy_id]
    
    def update_returns(self, strategy_id: str, returns: List[float]) -> None:
        """Update returns for a strategy"""
        if strategy_id not in self.strategy_returns:
            self.register_strategy(strategy_id)
        
        self.strategy_returns[strategy_id].extend(returns)
        
        # Update equity curve
        equity = self.strategy_equity[strategy_id][-1]
        for ret in returns:
            equity *= (1.0 + ret)
            self.strategy_equity[strategy_id].append(equity)
        
        # Trim to lookback period
        if len(self.strategy_returns[strategy_id]) > self.lookback_periods:
            self.strategy_returns[strategy_id] = self.strategy_returns[strategy_id][-self.lookback_periods:]
        if len(self.strategy_equity[strategy_id]) > self.lookback_periods + 1:
            self.strategy_equity[strategy_id] = self.strategy_equity[strategy_id][-self.lookback_periods - 1:]
    
    def calculate_rolling_sharpe(self, strategy_id: str, window: Optional[int] = None) -> float:
        """Calculate rolling Sharpe ratio"""
        returns = self.strategy_returns.get(strategy_id, [])
        if len(returns) < 10:
            return 0.0
        
        window = window or self.short_window
        recent_returns = returns[-window:]
        
        mean_ret = np.mean(recent_returns)
        std_ret = np.std(recent_returns)
        
        if std_ret <= 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        return sharpe
    
    def calculate_rolling_sortino(self, strategy_id: str, window: Optional[int] = None) -> float:
        """Calculate rolling Sortino ratio (downside deviation)"""
        returns = self.strategy_returns.get(strategy_id, [])
        if len(returns) < 10:
            return 0.0
        
        window = window or self.short_window
        recent_returns = returns[-window:]
        
        mean_ret = np.mean(recent_returns)
        downside_returns = [r for r in recent_returns if r < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_ret > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std <= 0:
            return 0.0
        
        sortino = (mean_ret / downside_std) * np.sqrt(252)
        return sortino
    
    def calculate_max_drawdown(self, strategy_id: str) -> float:
        """Calculate maximum drawdown from equity curve"""
        equity = self.strategy_equity.get(strategy_id, [])
        if len(equity) < 2:
            return 0.0
        
        peak = equity[0]
        max_dd = 0.0
        
        for eq in equity:
            if eq > peak:
                peak = eq
            drawdown = (peak - eq) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def calculate_current_drawdown(self, strategy_id: str) -> float:
        """Calculate current drawdown from peak"""
        equity = self.strategy_equity.get(strategy_id, [])
        if len(equity) < 2:
            return 0.0
        
        peak = max(equity)
        current = equity[-1]
        
        return (peak - current) / peak
    
    def calculate_win_rate(self, strategy_id: str) -> float:
        """Calculate win rate (percentage of profitable periods)"""
        returns = self.strategy_returns.get(strategy_id, [])
        if len(returns) == 0:
            return 0.0
        
        wins = sum(1 for r in returns if r > 0)
        return wins / len(returns)
    
    def calculate_profit_factor(self, strategy_id: str) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        returns = self.strategy_returns.get(strategy_id, [])
        if len(returns) == 0:
            return 0.0
        
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_strategy_metrics(self, strategy_id: str) -> StrategyMetrics:
        """Get comprehensive metrics for a strategy"""
        returns = self.strategy_returns.get(strategy_id, [])
        
        # Calculate average win/loss
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        
        avg_win = np.mean(winning_returns) if winning_returns else 0.0
        avg_loss = np.mean(losing_returns) if losing_returns else 0.0
        
        return StrategyMetrics(
            strategy_id=strategy_id,
            rolling_sharpe=self.calculate_rolling_sharpe(strategy_id),
            rolling_sortino=self.calculate_rolling_sortino(strategy_id),
            max_drawdown=self.calculate_max_drawdown(strategy_id),
            current_drawdown=self.calculate_current_drawdown(strategy_id),
            win_rate=self.calculate_win_rate(strategy_id),
            profit_factor=self.calculate_profit_factor(strategy_id),
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(returns),
            recent_returns=returns[-10:].copy(),
            timestamp_ns=time.time_ns()
        )
    
    def assess_decay(self, strategy_id: str) -> DecayAssessment:
        """
        Assess strategy for alpha decay
        Returns assessment with recommended actions
        """
        metrics = self.get_strategy_metrics(strategy_id)
        
        # Update peak metrics
        if metrics.rolling_sharpe > self.peak_sharpes.get(strategy_id, 0):
            self.peak_sharpes[strategy_id] = metrics.rolling_sharpe
        if metrics.rolling_sortino > self.peak_sortinos.get(strategy_id, 0):
            self.peak_sortinos[strategy_id] = metrics.rolling_sortino
        
        peak_sharpe = max(self.peak_sharpes.get(strategy_id, 0.5), 0.5)  # Floor at 0.5
        peak_sortino = max(self.peak_sortinos.get(strategy_id, 0.7), 0.7)
        
        # Calculate degradation
        sharpe_degradation = 1.0 - (metrics.rolling_sharpe / peak_sharpe)
        sortino_degradation = 1.0 - (metrics.rolling_sortino / peak_sortino)
        
        # Determine status
        status = DecayStatus.HEALTHY
        recommended_action = "Maintain current allocation"
        capital_scale = 1.0
        should_disable = False
        
        # Check critical conditions
        if (metrics.rolling_sharpe < self.sharpe_threshold_critical or
            metrics.rolling_sortino < self.sortino_threshold_critical or
            metrics.max_drawdown > self.max_drawdown_critical):
            status = DecayStatus.CRITICAL
            recommended_action = "Immediately reduce capital by 75%"
            capital_scale = 0.25
            
            if metrics.max_drawdown > self.max_drawdown_critical * 1.5:
                should_disable = True
                recommended_action = "DISABLE STRATEGY - Critical drawdown exceeded"
        
        # Check warning conditions
        elif (metrics.rolling_sharpe < self.sharpe_threshold_warning or
              metrics.rolling_sortino < self.sortino_threshold_warning or
              metrics.max_drawdown > self.max_drawdown_warning):
            status = DecayStatus.WARNING
            recommended_action = "Reduce capital allocation by 50%"
            capital_scale = 0.5
        
        # Check degradation from peak
        elif sharpe_degradation > self.degradation_threshold:
            status = DecayStatus.WARNING
            recommended_action = f"Alpha decay detected - Sharpe degraded {sharpe_degradation:.1%} from peak"
            capital_scale = 0.6
        
        # Additional check: consistent underperformance
        if metrics.total_trades >= 20 and metrics.win_rate < 0.35:
            status = DecayStatus.WARNING
            recommended_action = "Low win rate indicates potential regime mismatch"
            capital_scale = min(capital_scale, 0.5)
        
        assessment = DecayAssessment(
            strategy_id=strategy_id,
            status=status,
            sharpe_degradation=sharpe_degradation,
            sortino_degradation=sortino_degradation,
            drawdown_severity=metrics.current_drawdown,
            recommended_action=recommended_action,
            capital_scale=capital_scale,
            should_disable=should_disable,
            timestamp_ns=time.time_ns()
        )
        
        self.assessments[strategy_id] = assessment
        self.strategy_status[strategy_id] = status
        
        return assessment
    
    def get_capital_scale(self, strategy_id: str) -> float:
        """Get recommended capital scaling factor for a strategy"""
        if strategy_id in self.assessments:
            return self.assessments[strategy_id].capital_scale
        return 1.0
    
    def is_strategy_enabled(self, strategy_id: str) -> bool:
        """Check if strategy should be enabled"""
        status = self.strategy_status.get(strategy_id, DecayStatus.HEALTHY)
        return status != DecayStatus.DISABLED
    
    def get_all_assessments(self) -> Dict[str, DecayAssessment]:
        """Get assessments for all monitored strategies"""
        return self.assessments.copy()
    
    def reset_peaks(self, strategy_id: str) -> None:
        """Reset peak metrics for a strategy (e.g., after proven regime change)"""
        if strategy_id in self.peak_sharpes:
            self.peak_sharpes[strategy_id] = 0.0
        if strategy_id in self.peak_sortinos:
            self.peak_sortinos[strategy_id] = 0.0


# Example usage
def example_usage():
    """Example of how to use the strategy decay monitor"""
    monitor = StrategyDecayMonitor(
        lookback_periods=200,
        sharpe_threshold_warning=0.8,
        max_drawdown_warning=0.08,
    )
    
    # Register strategies
    monitor.register_strategy("market_making")
    monitor.register_strategy("stat_arb")
    
    # Simulate good performance for market making
    np.random.seed(42)
    good_returns = np.random.normal(0.001, 0.005, 100).tolist()
    monitor.update_returns("market_making", good_returns)
    
    # Simulate degrading performance for stat arb
    bad_returns = list(np.random.normal(-0.0005, 0.01, 100))
    # Add some large losses
    bad_returns[50:55] = [-0.05] * 5
    monitor.update_returns("stat_arb", bad_returns)
    
    # Assess both strategies
    mm_assessment = monitor.assess_decay("market_making")
    sa_assessment = monitor.assess_decay("stat_arb")
    
    print("Market Making Assessment:")
    print(f"  Status: {mm_assessment.status.value}")
    print(f"  Sharpe: {mm_assessment.sharpe_degradation:.1%} degradation")
    print(f"  Capital Scale: {mm_assessment.capital_scale:.1%}")
    print(f"  Action: {mm_assessment.recommended_action}")
    
    print("\nStat Arb Assessment:")
    print(f"  Status: {sa_assessment.status.value}")
    print(f"  Sharpe: {sa_assessment.sharpe_degradation:.1%} degradation")
    print(f"  Max DD: {sa_assessment.drawdown_severity:.1%}")
    print(f"  Capital Scale: {sa_assessment.capital_scale:.1%}")
    print(f"  Action: {sa_assessment.recommended_action}")
    print(f"  Should Disable: {sa_assessment.should_disable}")


if __name__ == "__main__":
    example_usage()
