"""
Advanced reward shaping functions for RL trading agents.
Implements Kelly Criterion, Drawdown penalties, and Sortino ratio optimization.
Designed to prioritize risk-adjusted returns over raw PnL.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RewardShaping:
    """
    Collection of advanced reward functions for reinforcement learning trading agents.
    All functions are designed to be memory-efficient and computationally lightweight.
    """
    
    @staticmethod
    def kelly_criterion_reward(
        pnl_series: np.ndarray,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_position_pct: float = 0.25,
    ) -> float:
        """
        Calculate reward based on Kelly Criterion optimization.
        
        The Kelly Criterion determines optimal position sizing:
        f* = (p * b - q) / b
        where:
            p = win probability
            q = loss probability (1 - p)
            b = win/loss ratio
        
        Args:
            pnl_series: Array of PnL values
            win_rate: Probability of winning trades
            avg_win: Average winning trade size
            avg_loss: Average losing trade size (absolute value)
            max_position_pct: Maximum position size constraint
        
        Returns:
            Kelly-optimized reward score
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Win/loss ratio
        b = avg_win / avg_loss
        
        # Kelly fraction
        kelly_fraction = (win_rate * b - (1 - win_rate)) / b
        
        # Clamp to reasonable bounds [0, max_position]
        kelly_fraction = np.clip(kelly_fraction, 0, max_position_pct)
        
        # Reward is proportional to how close we are to optimal Kelly
        # Penalize over-betting (f > f*) more than under-betting
        current_exposure = np.abs(pnl_series[-1]) if len(pnl_series) > 0 else 0
        
        if kelly_fraction > 0:
            # Normalize by Kelly-optimal exposure
            optimal_exposure = kelly_fraction * np.mean(np.abs(pnl_series)) if len(pnl_series) > 0 else 1
            if optimal_exposure > 0:
                ratio = current_exposure / optimal_exposure
                if ratio <= 1:
                    reward = ratio  # Under-betting: linear reward
                else:
                    reward = 2 - ratio  # Over-betting: penalty increases quadratically
            else:
                reward = 0.0
        else:
            # Negative Kelly suggests not trading
            reward = -np.abs(current_exposure)
        
        return float(reward)
    
    @staticmethod
    def drawdown_penalty_reward(
        cumulative_pnl: float,
        peak_pnl: float,
        drawdown_threshold: float = 0.05,
        penalty_factor: float = 2.0,
    ) -> float:
        """
        Calculate reward with drawdown penalty.
        
        Implements asymmetric penalty: larger drawdowns receive exponentially larger penalties.
        
        Args:
            cumulative_pnl: Current cumulative PnL
            peak_pnl: Highest cumulative PnL achieved
            drawdown_threshold: Drawdown level where penalty accelerates (default 5%)
            penalty_factor: Multiplier for penalty severity
        
        Returns:
            Reward with drawdown penalty applied
        """
        # Base reward
        base_reward = cumulative_pnl
        
        # Calculate drawdown
        if peak_pnl > 0:
            drawdown = (peak_pnl - cumulative_pnl) / peak_pnl
        else:
            drawdown = -cumulative_pnl if cumulative_pnl < 0 else 0
        
        # Apply penalty
        if drawdown > drawdown_threshold:
            # Exponential penalty beyond threshold
            excess_drawdown = drawdown - drawdown_threshold
            penalty = penalty_factor * (excess_drawdown ** 2) * peak_pnl
        elif drawdown > 0:
            # Linear penalty below threshold
            penalty = penalty_factor * drawdown * peak_pnl
        else:
            penalty = 0
        
        return base_reward - penalty
    
    @staticmethod
    def sortino_ratio_reward(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
        annualization_factor: int = 252,
    ) -> float:
        """
        Calculate reward based on Sortino Ratio.
        
        Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Deviation
        
        Unlike Sharpe Ratio, Sortino only penalizes downside volatility.
        
        Args:
            returns: Array of period returns
            risk_free_rate: Risk-free rate (annualized)
            target_return: Target return for downside calculation
            annualization_factor: Number of periods per year
        
        Returns:
            Sortino ratio as reward signal
        """
        if len(returns) < 2:
            return 0.0
        
        # Mean return
        mean_return = np.mean(returns)
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            # No downside: perfect score
            return mean_return * annualization_factor
        
        downside_std = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        if downside_std == 0:
            return mean_return * annualization_factor
        
        # Annualized Sortino
        excess_return = mean_return * annualization_factor - risk_free_rate
        sortino = excess_return / (downside_std * np.sqrt(annualization_factor))
        
        return sortino
    
    @staticmethod
    def calmar_ratio_reward(
        cumulative_returns: np.ndarray,
        lookback_period: int = 252,
    ) -> float:
        """
        Calculate reward based on Calmar Ratio.
        
        Calmar Ratio = Annualized Return / Max Drawdown
        
        Focuses on worst-case drawdown scenario.
        
        Args:
            cumulative_returns: Array of cumulative returns
            lookback_period: Period for calculating max drawdown
        
        Returns:
            Calmar ratio as reward signal
        """
        if len(cumulative_returns) < 2:
            return 0.0
        
        # Use last lookback_period points
        if len(cumulative_returns) > lookback_period:
            cum_ret = cumulative_returns[-lookback_period:]
        else:
            cum_ret = cumulative_returns
        
        # Annualized return
        total_return = cum_ret[-1] - cum_ret[0]
        years = len(cum_ret) / 252
        if years <= 0:
            return 0.0
        annualized_return = ((1 + total_return) ** (1 / years)) - 1
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - running_max) / running_max
        max_drawdown = np.abs(np.min(drawdown))
        
        if max_drawdown == 0:
            return annualized_return * 10  # Perfect: no drawdown
        
        calmar = annualized_return / max_drawdown
        return calmar
    
    @staticmethod
    def omega_ratio_reward(
        returns: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """
        Calculate reward based on Omega Ratio.
        
        Omega Ratio = Gains above threshold / Losses below threshold
        
        Considers all moments of the return distribution.
        
        Args:
            returns: Array of period returns
            threshold: Minimum acceptable return
        
        Returns:
            Omega ratio as reward signal
        """
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        total_gains = np.sum(gains)
        total_losses = np.sum(losses)
        
        if total_losses == 0:
            return total_gains * 10 if total_gains > 0 else 1.0
        
        omega = total_gains / total_losses
        return omega
    
    @staticmethod
    def composite_reward(
        pnl: float,
        returns: np.ndarray,
        cumulative_pnl: float,
        peak_pnl: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate composite reward combining multiple metrics.
        
        Weights can be adjusted based on trading strategy priorities:
        - Aggressive: higher weight on PnL and Kelly
        - Conservative: higher weight on Sortino and drawdown penalty
        
        Args:
            pnl: Current period PnL
            returns: Array of recent returns
            cumulative_pnl: Total cumulative PnL
            peak_pnl: Peak cumulative PnL
            win_rate: Win rate of trades
            avg_win: Average winning trade
            avg_loss: Average losing trade
            weights: Dictionary of weights for each component
        
        Returns:
            Weighted composite reward
        """
        # Default weights (balanced approach)
        default_weights = {
            "pnl": 0.2,
            "kelly": 0.15,
            "drawdown": 0.25,
            "sortino": 0.25,
            "omega": 0.15,
        }
        
        if weights:
            default_weights.update(weights)
        
        w = default_weights
        
        # Calculate individual components
        kelly_rwd = RewardShaping.kelly_criterion_reward(
            np.array([cumulative_pnl]), win_rate, avg_win, avg_loss
        )
        
        drawdown_rwd = RewardShaping.drawdown_penalty_reward(
            cumulative_pnl, peak_pnl
        )
        
        sortino_rwd = RewardShaping.sortino_ratio_reward(returns)
        
        omega_rwd = RewardShaping.omega_ratio_reward(returns)
        
        # Normalize PnL
        pnl_normalized = pnl / 1000.0  # Scale to reasonable range
        
        # Composite
        composite = (
            w["pnl"] * pnl_normalized +
            w["kelly"] * kelly_rwd +
            w["drawdown"] * drawdown_rwd +
            w["sortino"] * sortino_rwd +
            w["omega"] * omega_rwd
        )
        
        return composite


class AdaptiveRewardShaping:
    """
    Adaptive reward shaping that adjusts weights based on market regime.
    """
    
    def __init__(self):
        self.regime_weights = {
            "trending": {
                "pnl": 0.3,
                "kelly": 0.2,
                "drawdown": 0.15,
                "sortino": 0.2,
                "omega": 0.15,
            },
            "ranging": {
                "pnl": 0.15,
                "kelly": 0.1,
                "drawdown": 0.3,
                "sortino": 0.3,
                "omega": 0.15,
            },
            "volatile": {
                "pnl": 0.1,
                "kelly": 0.1,
                "drawdown": 0.4,
                "sortino": 0.3,
                "omega": 0.1,
            },
        }
        
        self.current_regime = "ranging"
        self.regime_history: List[str] = []
    
    def update_regime(self, regime: str):
        """Update current market regime."""
        self.current_regime = regime
        self.regime_history.append(regime)
        
        # Keep history bounded
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
    
    def get_reward(
        self,
        pnl: float,
        returns: np.ndarray,
        cumulative_pnl: float,
        peak_pnl: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Get reward with regime-adaptive weights."""
        weights = self.regime_weights.get(self.current_regime, self.regime_weights["ranging"])
        
        return RewardShaping.composite_reward(
            pnl=pnl,
            returns=returns,
            cumulative_pnl=cumulative_pnl,
            peak_pnl=peak_pnl,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            weights=weights,
        )


def main():
    """Test reward shaping functions."""
    import psutil
    
    print(f"Initial RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Generate synthetic trading data
    np.random.seed(42)
    n_trades = 100
    
    # Simulate returns
    returns = np.random.normal(0.001, 0.02, n_trades)
    cumulative_pnl = np.cumsum(returns) * 10000  # Scale to dollar PnL
    peak_pnl = np.maximum.accumulate(cumulative_pnl)
    
    # Trade statistics
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns <= 0]
    win_rate = len(winning_trades) / len(returns)
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = np.abs(np.mean(losing_trades)) if len(losing_trades) > 0 else 0
    
    print(f"\nTrade Statistics:")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Avg Win: {avg_win:.4f}")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Final PnL: ${cumulative_pnl[-1]:,.2f}")
    
    # Test individual rewards
    print("\n--- Individual Reward Components ---")
    
    kelly_rwd = RewardShaping.kelly_criterion_reward(
        cumulative_pnl, win_rate, avg_win, avg_loss
    )
    print(f"Kelly Criterion Reward: {kelly_rwd:.4f}")
    
    dd_rwd = RewardShaping.drawdown_penalty_reward(
        cumulative_pnl[-1], peak_pnl[-1]
    )
    print(f"Drawdown Penalty Reward: {dd_rwd:.4f}")
    
    sortino_rwd = RewardShaping.sortino_ratio_reward(returns)
    print(f"Sortino Ratio Reward: {sortino_rwd:.4f}")
    
    calmar_rwd = RewardShaping.calmar_ratio_reward(cumulative_pnl)
    print(f"Calmar Ratio Reward: {calmar_rwd:.4f}")
    
    omega_rwd = RewardShaping.omega_ratio_reward(returns)
    print(f"Omega Ratio Reward: {omega_rwd:.4f}")
    
    # Test composite reward
    print("\n--- Composite Reward ---")
    composite_rwd = RewardShaping.composite_reward(
        pnl=returns[-1] * 10000,
        returns=returns[-20:],
        cumulative_pnl=cumulative_pnl[-1],
        peak_pnl=peak_pnl[-1],
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
    print(f"Composite Reward: {composite_rwd:.4f}")
    
    # Test adaptive reward
    print("\n--- Adaptive Reward by Regime ---")
    adaptive = AdaptiveRewardShaping()
    
    for regime in ["trending", "ranging", "volatile"]:
        adaptive.update_regime(regime)
        rwd = adaptive.get_reward(
            pnl=returns[-1] * 10000,
            returns=returns[-20:],
            cumulative_pnl=cumulative_pnl[-1],
            peak_pnl=peak_pnl[-1],
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )
        print(f"{regime.capitalize()} Regime Reward: {rwd:.4f}")
    
    print(f"\nFinal RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
