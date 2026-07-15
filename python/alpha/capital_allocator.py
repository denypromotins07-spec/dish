"""
Dynamic Kelly and Risk-Parity capital reallocation engine.
Automatically shifts funds away from decaying factors and underperforming agents.
Memory-efficient implementation for strict RAM constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy or factor."""
    strategy_id: str
    cumulative_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0


@dataclass
class AllocationResult:
    """Result of capital allocation calculation."""
    strategy_id: str
    target_weight: float
    current_weight: float
    rebalance_delta: float
    kelly_fraction: float
    risk_parity_weight: float
    decay_adjustment: float = 1.0


class KellyCriterion:
    """
    Calculates optimal Kelly fraction for position sizing.
    Includes fractional Kelly for risk management.
    """
    
    def __init__(
        self,
        max_kelly: float = 0.25,
        fractional_kelly: float = 0.5,
        min_win_rate: float = 0.4,
    ):
        self.max_kelly = max_kelly
        self.fractional_kelly = fractional_kelly
        self.min_win_rate = min_win_rate
    
    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            win_rate: Probability of winning trade
            avg_win: Average win amount (positive)
            avg_loss: Average loss amount (positive)
        
        Returns:
            Kelly fraction (0 to max_kelly)
        """
        if win_rate < self.min_win_rate:
            return 0.0
        
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        
        # b = odds received on the wager (win/loss ratio)
        b = avg_win / avg_loss
        
        # p = probability of winning
        p = win_rate
        
        # q = probability of losing
        q = 1 - p
        
        # Kelly formula: f = (bp - q) / b
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        kelly *= self.fractional_kelly
        
        # Cap at maximum
        return max(0.0, min(kelly, self.max_kelly))
    
    def calculate_kelly_from_returns(
        self,
        returns: np.ndarray,
    ) -> float:
        """Calculate Kelly from historical returns series."""
        if len(returns) < 10:
            return 0.0
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        return self.calculate_kelly(win_rate, avg_win, avg_loss)


class RiskParityAllocator:
    """
    Risk parity allocation based on inverse volatility.
    Ensures each strategy contributes equally to portfolio risk.
    """
    
    def __init__(
        self,
        lookback: int = 252,
        min_weight: float = 0.01,
        max_weight: float = 0.4,
    ):
        self.lookback = lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.return_history: Dict[str, deque] = {}
    
    def add_strategy(self, strategy_id: str):
        """Register a new strategy."""
        if strategy_id not in self.return_history:
            self.return_history[strategy_id] = deque(maxlen=self.lookback)
    
    def update_returns(self, strategy_id: str, daily_return: float):
        """Add return observation for a strategy."""
        if strategy_id not in self.return_history:
            self.add_strategy(strategy_id)
        self.return_history[strategy_id].append(daily_return)
    
    def calculate_weights(self) -> Dict[str, float]:
        """Calculate risk parity weights."""
        if len(self.return_history) < 2:
            return {}
        
        # Calculate volatilities
        volatilities = {}
        for strategy_id, returns in self.return_history.items():
            if len(returns) >= 20:
                volatilities[strategy_id] = np.std(returns) + 1e-10
            else:
                volatilities[strategy_id] = 1.0  # Default high risk
        
        # Inverse volatility weighting
        inv_vol = {k: 1.0 / v for k, v in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        
        if total_inv_vol < 1e-10:
            n = len(inv_vol)
            return {k: 1.0 / n for k in inv_vol}
        
        # Normalize weights
        weights = {k: v / total_inv_vol for k, v in inv_vol.items()}
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        return weights
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints and renormalize."""
        # Clip weights
        clipped = {k: np.clip(v, self.min_weight, self.max_weight) 
                   for k, v in weights.items()}
        
        # Renormalize
        total = sum(clipped.values())
        if total > 0:
            return {k: v / total for k, v in clipped.items()}
        else:
            n = len(clipped)
            return {k: 1.0 / n for k in clipped}
    
    def calculate_marginal_risk_contribution(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate marginal risk contribution of each strategy."""
        if len(self.return_history) < 2:
            return {}
        
        # Build return matrix
        strategy_ids = list(self.return_history.keys())
        n_obs = min(len(r) for r in self.return_history.values())
        
        if n_obs < 20:
            return {}
        
        returns_matrix = np.column_stack([
            list(self.return_history[sid])[-n_obs:] for sid in strategy_ids
        ])
        
        # Portfolio volatility
        w = np.array([weights.get(sid, 0) for sid in strategy_ids])
        cov = np.cov(returns_matrix.T)
        
        # Marginal risk contribution
        port_vol = np.sqrt(w @ cov @ w)
        marginal_contrib = (cov @ w) / port_vol
        
        return {sid: marginal_contrib[i] for i, sid in enumerate(strategy_ids)}


class CapitalAllocator:
    """
    Main capital allocation engine combining Kelly and Risk Parity.
    Implements dynamic reallocation based on performance and decay.
    """
    
    def __init__(
        self,
        strategies: List[str],
        total_capital: float = 10_000_000.0,
        rebalance_threshold: float = 0.05,
        decay_lookback: int = 63,
    ):
        self.strategies = strategies
        self.total_capital = total_capital
        self.rebalance_threshold = rebalance_threshold
        self.decay_lookback = decay_lookback
        
        self.kelly = KellyCriterion()
        self.risk_parity = RiskParityAllocator()
        
        # Initialize strategies
        for strategy_id in strategies:
            self.risk_parity.add_strategy(strategy_id)
        
        # Current allocations
        self.current_weights: Dict[str, float] = {s: 1.0 / len(strategies) for s in strategies}
        self.target_weights: Dict[str, float] = self.current_weights.copy()
        
        # Performance tracking
        self.metrics: Dict[str, StrategyMetrics] = {
            s: StrategyMetrics(strategy_id=s) for s in strategies
        }
        
        # Return history for metrics
        self.return_history: Dict[str, deque] = {
            s: deque(maxlen=252) for s in strategies
        }
        
        # Decay scores (0 = fully decayed, 1 = fresh)
        self.decay_scores: Dict[str, float] = {s: 1.0 for s in strategies}
        
        # Rebalance history
        self.rebalance_log: List[Dict] = []
    
    def update_performance(
        self,
        strategy_id: str,
        daily_return: float,
        decay_score: Optional[float] = None,
    ):
        """Update performance metrics for a strategy."""
        if strategy_id not in self.strategies:
            return
        
        # Update return history
        self.return_history[strategy_id].append(daily_return)
        self.risk_parity.update_returns(strategy_id, daily_return)
        
        # Update decay score
        if decay_score is not None:
            self.decay_scores[strategy_id] = decay_score
        
        # Recalculate metrics
        self._update_metrics(strategy_id)
    
    def _update_metrics(self, strategy_id: str):
        """Recalculate all performance metrics."""
        returns = np.array(list(self.return_history[strategy_id]))
        
        if len(returns) < 10:
            return
        
        metrics = self.metrics[strategy_id]
        
        # Basic metrics
        metrics.cumulative_return = np.prod(1 + returns) - 1
        metrics.volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        mean_return = np.mean(returns)
        if metrics.volatility > 0:
            metrics.sharpe_ratio = (mean_return * 252) / metrics.volatility
        else:
            metrics.sharpe_ratio = 0.0
        
        # Drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        metrics.max_drawdown = np.min(drawdowns)
        metrics.current_drawdown = drawdowns[-1]
        
        # Win rate
        metrics.win_rate = np.mean(returns > 0)
        
        # Higher moments
        if len(returns) > 30:
            metrics.skewness = self._skewness(returns)
            metrics.kurtosis = self._kurtosis(returns)
        
        # Calmar ratio
        if abs(metrics.max_drawdown) > 0:
            metrics.calmar_ratio = (mean_return * 252) / abs(metrics.max_drawdown)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            if downside_std > 0:
                metrics.sortino_ratio = (mean_return * 252) / downside_std
    
    def _skewness(self, returns: np.ndarray) -> float:
        n = len(returns)
        if n < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std < 1e-10:
            return 0.0
        return np.mean(((returns - mean) / std) ** 3)
    
    def _kurtosis(self, returns: np.ndarray) -> float:
        n = len(returns)
        if n < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std < 1e-10:
            return 0.0
        return np.mean(((returns - mean) / std) ** 4) - 3
    
    def calculate_optimal_allocation(self) -> Dict[str, AllocationResult]:
        """
        Calculate optimal capital allocation using Kelly and Risk Parity.
        Adjusts for alpha decay.
        """
        results = {}
        
        # Get risk parity weights
        rp_weights = self.risk_parity.calculate_weights()
        
        for strategy_id in self.strategies:
            metrics = self.metrics[strategy_id]
            returns = np.array(list(self.return_history[strategy_id]))
            
            # Kelly fraction
            kelly_fraction = self.kelly.calculate_kelly_from_returns(returns)
            
            # Risk parity weight
            rp_weight = rp_weights.get(strategy_id, 0)
            
            # Decay adjustment
            decay_adj = self.decay_scores.get(strategy_id, 1.0)
            
            # Combined weight (blend Kelly and Risk Parity)
            # Weight by Sharpe ratio confidence
            sharpe_confidence = min(abs(metrics.sharpe_ratio) / 2, 1.0)
            
            blended_weight = (
                sharpe_confidence * kelly_fraction +
                (1 - sharpe_confidence) * rp_weight
            ) * decay_adj
            
            # Store result
            results[strategy_id] = AllocationResult(
                strategy_id=strategy_id,
                target_weight=blended_weight,
                current_weight=self.current_weights.get(strategy_id, 0),
                rebalance_delta=blended_weight - self.current_weights.get(strategy_id, 0),
                kelly_fraction=kelly_fraction,
                risk_parity_weight=rp_weight,
                decay_adjustment=decay_adj,
            )
        
        # Normalize to sum to 1
        total_weight = sum(r.target_weight for r in results.values())
        if total_weight > 0:
            for r in results.values():
                r.target_weight /= total_weight
        
        return results
    
    def rebalance(self) -> List[Tuple[str, float]]:
        """
        Execute rebalancing if needed.
        Returns list of (strategy_id, delta_weight) for trades.
        """
        allocation = self.calculate_optimal_allocation()
        
        trades = []
        for strategy_id, result in allocation.items():
            if abs(result.rebalance_delta) > self.rebalance_threshold:
                trades.append((strategy_id, result.rebalance_delta))
                self.current_weights[strategy_id] = result.target_weight
        
        if trades:
            self.rebalance_log.append({
                'timestamp': len(self.rebalance_log),
                'trades': trades,
                'new_weights': self.current_weights.copy(),
            })
        
        return trades
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get comprehensive allocation summary."""
        allocation = self.calculate_optimal_allocation()
        
        summary = {
            'total_capital': self.total_capital,
            'n_strategies': len(self.strategies),
            'current_weights': self.current_weights.copy(),
            'target_weights': {k: v.target_weight for k, v in allocation.items()},
            'strategies': {},
        }
        
        for strategy_id in self.strategies:
            metrics = self.metrics[strategy_id]
            alloc = allocation[strategy_id]
            
            summary['strategies'][strategy_id] = {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'decay_score': self.decay_scores[strategy_id],
                'kelly_fraction': alloc.kelly_fraction,
                'risk_parity_weight': alloc.risk_parity_weight,
                'current_weight': self.current_weights[strategy_id],
                'target_weight': alloc.target_weight,
                'needs_rebalance': abs(alloc.rebalance_delta) > self.rebalance_threshold,
            }
        
        return summary


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    strategies = ['momentum', 'value', 'stat_arb', 'market_making']
    allocator = CapitalAllocator(strategies, total_capital=1_000_000)
    
    print("Simulating capital allocation...\n")
    
    for day in range(100):
        for strategy in strategies:
            # Generate random returns with different characteristics
            if strategy == 'momentum':
                ret = np.random.randn() * 0.02 + 0.001
            elif strategy == 'value':
                ret = np.random.randn() * 0.015 + 0.0015
            elif strategy == 'stat_arb':
                ret = np.random.randn() * 0.008 + 0.0008
            else:
                ret = np.random.randn() * 0.005 + 0.0005
            
            # Simulate decay for momentum
            decay = 1.0 - (day / 200) if strategy == 'momentum' else 1.0
            
            allocator.update_performance(strategy, ret, decay_score=decay)
        
        if day % 20 == 0 and day > 0:
            trades = allocator.rebalance()
            if trades:
                print(f"Day {day}: Rebalancing")
                for strat, delta in trades:
                    print(f"  {strat}: {'+' if delta > 0 else ''}{delta:.2%}")
    
    print("\nFinal Allocation Summary:")
    summary = allocator.get_allocation_summary()
    for strat, info in summary['strategies'].items():
        print(f"\n{strat}:")
        print(f"  Sharpe: {info['sharpe_ratio']:.2f}")
        print(f"  Decay Score: {info['decay_score']:.2f}")
        print(f"  Current Weight: {info['current_weight']:.2%}")
        print(f"  Target Weight: {info['target_weight']:.2%}")
