"""
Custom objective functions for optimization that heavily penalize high turnover, 
excessive drawdown, and parameter instability. Prioritizes robust out-of-sample 
equity curves over raw in-sample PnL.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Container for all optimization metrics."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    turnover: float
    avg_trade_duration: float
    tail_ratio: float
    ulcer_index: float


def risk_adjusted_objective(
    metrics: OptimizationMetrics,
    target_sharpe: float = 1.5,
    max_acceptable_dd: float = -0.15,
    turnover_penalty_weight: float = 0.3,
) -> float:
    """
    Primary objective function with heavy penalties for undesirable characteristics.
    
    Parameters
    ----------
    metrics : OptimizationMetrics
        Performance metrics from backtest.
    target_sharpe : float, default 1.5
        Target Sharpe ratio.
    max_acceptable_dd : float, default -0.15
        Maximum acceptable drawdown.
    turnover_penalty_weight : float, default 0.3
        Weight of turnover penalty.
        
    Returns
    -------
    float
        Risk-adjusted objective score (higher is better).
    """
    # Base score: Sharpe ratio
    base_score = metrics.sharpe_ratio
    
    # Penalty 1: Excessive drawdown (severe penalty)
    if metrics.max_drawdown < max_acceptable_dd:
        dd_penalty = abs(metrics.max_drawdown - max_acceptable_dd) * 10
        base_score -= dd_penalty
    
    # Penalty 2: High turnover (transaction costs)
    turnover_penalty = metrics.turnover * turnover_penalty_weight
    base_score -= turnover_penalty
    
    # Penalty 3: Low Calmar ratio (return/drawdown efficiency)
    if metrics.calmar_ratio < 1.0:
        calmar_penalty = (1.0 - metrics.calmar_ratio) * 2
        base_score -= calmar_penalty
    
    # Bonus: Consistent returns (high tail ratio)
    if metrics.tail_ratio > 1.5:
        base_score += (metrics.tail_ratio - 1.5) * 0.5
    
    # Penalty 4: High Ulcer Index (sustained drawdown pain)
    if metrics.ulcer_index > 5.0:
        ulcer_penalty = (metrics.ulcer_index - 5.0) * 0.1
        base_score -= ulcer_penalty
    
    return base_score


def stability_weighted_objective(
    metrics: OptimizationMetrics,
    oos_metrics: Optional[OptimizationMetrics] = None,
    stability_weight: float = 0.5,
) -> float:
    """
    Objective function that weights in-sample and out-of-sample performance.
    
    Parameters
    ----------
    metrics : OptimizationMetrics
        In-sample performance metrics.
    oos_metrics : Optional[OptimizationMetrics]
        Out-of-sample metrics (if available).
    stability_weight : float, default 0.5
        Weight given to OOS performance.
        
    Returns
    -------
    float
        Stability-weighted objective score.
    """
    is_score = risk_adjusted_objective(metrics)
    
    if oos_metrics is None:
        # No OOS data, apply uncertainty penalty
        uncertainty_penalty = 0.2
        return is_score * (1 - uncertainty_penalty)
    
    oos_score = risk_adjusted_objective(oos_metrics)
    
    # Calculate degradation from IS to OOS
    degradation = (is_score - oos_score) / max(abs(is_score), 0.001)
    
    # Penalize significant degradation (sign of overfitting)
    if degradation > 0.3:
        overfit_penalty = (degradation - 0.3) * 2
        oos_score -= overfit_penalty
    
    # Weighted combination
    combined_score = (1 - stability_weight) * is_score + stability_weight * oos_score
    
    return combined_score


def turnover_penalized_objective(
    metrics: OptimizationMetrics,
    max_turnover: float = 5.0,
    soft_turnover_limit: float = 2.0,
) -> float:
    """
    Objective function with aggressive turnover penalties.
    
    Parameters
    ----------
    metrics : OptimizationMetrics
        Performance metrics.
    max_turnover : float, default 5.0
        Maximum allowed turnover (hard limit).
    soft_turnover_limit : float, default 2.0
        Soft turnover limit before penalties kick in.
        
    Returns
    -------
    float
        Turnover-penalized objective score.
    """
    base_score = metrics.sharpe_ratio
    
    # Hard cutoff for excessive turnover
    if metrics.turnover > max_turnover:
        return -1000.0  # Reject this parameter set
    
    # Soft penalty for moderate turnover
    if metrics.turnover > soft_turnover_limit:
        excess_turnover = metrics.turnover - soft_turnover_limit
        penalty = excess_turnover * 0.5
        base_score -= penalty
    
    # Additional penalty based on trade frequency
    if metrics.avg_trade_duration < 0.1:  # Very short trades
        hft_penalty = 0.3
        base_score -= hft_penalty
    
    return base_score


def drawdown_focused_objective(
    metrics: OptimizationMetrics,
    max_dd_threshold: float = -0.10,
    dd_penalty_exponent: float = 2.0,
) -> float:
    """
    Objective function that prioritizes drawdown control.
    
    Parameters
    ----------
    metrics : OptimizationMetrics
        Performance metrics.
    max_dd_threshold : float, default -0.10
        Maximum acceptable drawdown.
    dd_penalty_exponent : float, default 2.0
        Exponent for drawdown penalty (higher = more severe).
        
    Returns
    -------
    float
        Drawdown-focused objective score.
    """
    # Start with return-based metric
    base_score = metrics.total_return
    
    # Severe drawdown penalty with exponential scaling
    if metrics.max_drawdown < max_dd_threshold:
        dd_excess = abs(metrics.max_drawdown) - abs(max_dd_threshold)
        penalty = (dd_excess ** dd_penalty_exponent) * 10
        base_score -= penalty
    
    # Reward consistent drawdown control
    if metrics.calmar_ratio > 2.0:
        calmar_bonus = (metrics.calmar_ratio - 2.0) * 0.5
        base_score += calmar_bonus
    
    # Sortino bonus (penalizes downside volatility)
    if metrics.sortino_ratio > 2.0:
        sortino_bonus = (metrics.sortino_ratio - 2.0) * 0.3
        base_score += sortino_bonus
    
    return base_score


def parameter_stability_score(
    params_history: List[Dict[str, float]],
    metrics_history: List[float],
) -> float:
    """
    Calculate parameter stability score across multiple windows.
    
    Parameters
    ----------
    params_history : List[Dict[str, float]]
        Parameter values across windows.
    metrics_history : List[float]
        Objective values across windows.
        
    Returns
    -------
    float
        Stability score (0-1, higher is more stable).
    """
    if len(params_history) < 2 or len(metrics_history) < 2:
        return 0.5  # Neutral score for insufficient data
    
    # Calculate coefficient of variation for metrics
    metrics_std = np.std(metrics_history)
    metrics_mean = abs(np.mean(metrics_history))
    
    if metrics_mean == 0:
        return 0.0
    
    metrics_cv = metrics_std / metrics_mean
    
    # Calculate average parameter variation
    param_variations = []
    if params_history and isinstance(params_history[0], dict):
        for key in params_history[0].keys():
            values = [p.get(key, 0) for p in params_history]
            if np.mean(values) != 0:
                cv = np.std(values) / abs(np.mean(values))
                param_variations.append(cv)
    
    avg_param_cv = np.mean(param_variations) if param_variations else 0
    
    # Combined stability score
    stability = 1.0 / (1.0 + metrics_cv + avg_param_cv)
    
    return min(1.0, max(0.0, stability))


def create_composite_objective(
    weights: Optional[Dict[str, float]] = None,
) -> callable:
    """
    Factory function to create a composite objective function with custom weights.
    
    Parameters
    ----------
    weights : Optional[Dict[str, float]]
        Custom weights for each component.
        
    Returns
    -------
    callable
        Composite objective function.
    """
    default_weights = {
        'sharpe': 0.3,
        'drawdown': 0.25,
        'turnover': 0.15,
        'stability': 0.15,
        'tail_risk': 0.1,
        'consistency': 0.05,
    }
    
    if weights:
        default_weights.update(weights)
    
    def composite_objective(
        metrics: OptimizationMetrics,
        oos_metrics: Optional[OptimizationMetrics] = None,
        params_history: Optional[List[Dict]] = None,
        metrics_history: Optional[List[float]] = None,
    ) -> float:
        """Composite objective combining multiple criteria."""
        
        # Component 1: Risk-adjusted returns
        sharpe_component = metrics.sharpe_ratio * default_weights['sharpe']
        
        # Component 2: Drawdown penalty
        dd_component = 0.0
        if metrics.max_drawdown < -0.10:
            dd_component = metrics.max_drawdown * default_weights['drawdown'] * 5
        else:
            dd_component = -abs(metrics.max_drawdown) * default_weights['drawdown']
        
        # Component 3: Turnover penalty
        turnover_component = -metrics.turnover * default_weights['turnover'] * 0.5
        
        # Component 4: Tail risk (Ulcer index, tail ratio)
        tail_component = 0.0
        if metrics.ulcer_index > 5.0:
            tail_component -= (metrics.ulcer_index - 5.0) * default_weights['tail_risk'] * 0.2
        if metrics.tail_ratio > 1.5:
            tail_component += (metrics.tail_ratio - 1.5) * default_weights['tail_risk'] * 0.3
        
        # Component 5: Stability (if history available)
        stability_component = 0.0
        if params_history and metrics_history:
            stability = parameter_stability_score(params_history, metrics_history)
            stability_component = stability * default_weights['stability']
        
        # Component 6: Consistency (win rate, profit factor)
        consistency_component = (
            (metrics.win_rate - 0.5) * default_weights['consistency'] +
            (metrics.profit_factor - 1.0) * 0.1 * default_weights['consistency']
        )
        
        total_score = (
            sharpe_component +
            dd_component +
            turnover_component +
            tail_component +
            stability_component +
            consistency_component
        )
        
        return total_score
    
    return composite_objective


def early_stopping_criterion(
    current_metrics: OptimizationMetrics,
    best_metrics_so_far: Optional[OptimizationMetrics],
    patience_trials: int = 5,
    min_improvement: float = 0.01,
) -> bool:
    """
    Determine if optimization should stop early.
    
    Parameters
    ----------
    current_metrics : OptimizationMetrics
        Current trial metrics.
    best_metrics_so_far : Optional[OptimizationMetrics]
        Best metrics seen so far.
    patience_trials : int, default 5
        Number of trials without improvement before stopping.
    min_improvement : float, default 0.01
        Minimum improvement required.
        
    Returns
    -------
    bool
        True if should stop early.
    """
    if best_metrics_so_far is None:
        return False
    
    current_score = risk_adjusted_objective(current_metrics)
    best_score = risk_adjusted_objective(best_metrics_so_far)
    
    if current_score < best_score * (1 - min_improvement):
        # Significant underperformance
        return True
    
    return False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample metrics
    sample_metrics = OptimizationMetrics(
        total_return=0.25,
        sharpe_ratio=1.8,
        sortino_ratio=2.5,
        calmar_ratio=2.0,
        max_drawdown=-0.08,
        win_rate=0.55,
        profit_factor=1.8,
        turnover=3.5,
        avg_trade_duration=2.5,
        tail_ratio=1.6,
        ulcer_index=3.2,
    )
    
    print("Sample Metrics:")
    print(f"  Sharpe: {sample_metrics.sharpe_ratio}")
    print(f"  Max DD: {sample_metrics.max_drawdown}")
    print(f"  Turnover: {sample_metrics.turnover}")
    
    # Test different objectives
    score1 = risk_adjusted_objective(sample_metrics)
    print(f"\nRisk-adjusted score: {score1:.4f}")
    
    score2 = turnover_penalized_objective(sample_metrics)
    print(f"Turnover-penalized score: {score2:.4f}")
    
    score3 = drawdown_focused_objective(sample_metrics)
    print(f"Drawdown-focused score: {score3:.4f}")
    
    # Test composite objective
    composite_fn = create_composite_objective()
    score4 = composite_fn(sample_metrics)
    print(f"Composite score: {score4:.4f}")
