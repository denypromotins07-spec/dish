"""
Out-of-sample validation and walk-forward testing module.
Rigorously rejects newly trained models if they don't strictly beat
the incumbent model's out-of-sample Sharpe and Sortino ratios.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings

# Strict validation thresholds
MIN_SHARPE_IMPROVEMENT = 0.05
MIN_SORTINO_IMPROVEMENT = 0.05
MAX_DRAWDOWN_INCREASE = 0.1
MIN_SAMPLES_OOS = 500


@dataclass
class ValidationResult:
    """Result of model validation."""
    passed: bool
    reason: str
    metrics: Dict[str, float]
    incumbent_metrics: Dict[str, float]
    improvement_pct: Dict[str, float]


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""
    sharpe_ratios: List[float]
    sortino_ratios: List[float]
    max_drawdowns: List[float]
    hit_rates: List[float]
    avg_sharpe: float
    avg_sortino: float
    consistency_score: float


class ModelValidator:
    """
    Validates new models against incumbents using rigorous
    out-of-sample and walk-forward testing.
    """

    def __init__(
        self,
        min_oos_samples: int = MIN_SAMPLES_OOS,
        min_sharpe_improvement: float = MIN_SHARPE_IMPROVEMENT,
        min_sortino_improvement: float = MIN_SORTINO_IMPROVEMENT,
    ):
        self.min_oos_samples = min_oos_samples
        self.min_sharpe_improvement = min_sharpe_improvement
        self.min_sortino_improvement = min_sortino_improvement
        self._walk_forward_results: Optional[WalkForwardResult] = None

    def compute_sharpe_ratio(
        self,
        returns: np.ndarray,
        annualization_factor: float = 252 * 24,
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) / np.std(returns)) * np.sqrt(annualization_factor)

    def compute_sortino_ratio(
        self,
        returns: np.ndarray,
        annualization_factor: float = 252 * 24,
    ) -> float:
        """Calculate annualized Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return (np.mean(returns) / downside_std) * np.sqrt(annualization_factor)

    def compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return abs(np.min(drawdowns))

    def compute_hit_rate(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate prediction hit rate (direction accuracy)."""
        if len(predictions) != len(actuals):
            return 0.0
        
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        
        return np.mean(pred_direction == actual_direction)

    def compute_pnl_from_signals(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        position_size: float = 1.0,
    ) -> np.ndarray:
        """Convert prediction signals to PnL series."""
        # Position size based on prediction sign
        positions = np.sign(predictions) * position_size
        return positions * actual_returns

    def validate_out_of_sample(
        self,
        incumbent_predictions: np.ndarray,
        new_predictions: np.ndarray,
        actual_returns: np.ndarray,
        incumbent_metrics: Optional[Dict] = None,
    ) -> ValidationResult:
        """
        Validate new model against incumbent on out-of-sample data.
        Returns ValidationResult with pass/fail decision.
        """
        if len(actual_returns) < self.min_oos_samples:
            return ValidationResult(
                passed=False,
                reason=f'Insufficient OOS samples: {len(actual_returns)} < {self.min_oos_samples}',
                metrics={},
                incumbent_metrics=incumbent_metrics or {},
                improvement_pct={},
            )

        # Ensure arrays are aligned
        min_len = min(len(incumbent_predictions), len(new_predictions), len(actual_returns))
        incumbent_predictions = incumbent_predictions[:min_len]
        new_predictions = new_predictions[:min_len]
        actual_returns = actual_returns[:min_len]

        # Calculate PnL series
        incumbent_pnl = self.compute_pnl_from_signals(incumbent_predictions, actual_returns)
        new_pnl = self.compute_pnl_from_signals(new_predictions, actual_returns)

        # Compute metrics for new model
        new_sharpe = self.compute_sharpe_ratio(new_pnl)
        new_sortino = self.compute_sortino_ratio(new_pnl)
        new_max_dd = self.compute_max_drawdown(new_pnl)
        new_hit_rate = self.compute_hit_rate(new_predictions, actual_returns)

        new_metrics = {
            'sharpe': new_sharpe,
            'sortino': new_sortino,
            'max_drawdown': new_max_dd,
            'hit_rate': new_hit_rate,
            'total_return': np.sum(new_pnl),
        }

        # Use provided incumbent metrics or calculate
        if incumbent_metrics is None:
            inc_sharpe = self.compute_sharpe_ratio(incumbent_pnl)
            inc_sortino = self.compute_sortino_ratio(incumbent_pnl)
            inc_max_dd = self.compute_max_drawdown(incumbent_pnl)
            inc_hit_rate = self.compute_hit_rate(incumbent_predictions, actual_returns)
            
            incumbent_metrics = {
                'sharpe': inc_sharpe,
                'sortino': inc_sortino,
                'max_drawdown': inc_max_dd,
                'hit_rate': inc_hit_rate,
                'total_return': np.sum(incumbent_pnl),
            }

        # Calculate improvements
        improvements = {}
        for key in new_metrics:
            if incumbent_metrics.get(key, 0) != 0:
                improvements[key] = (new_metrics[key] - incumbent_metrics[key]) / abs(incumbent_metrics[key])
            else:
                improvements[key] = new_metrics[key]

        # Validation logic
        reasons = []
        passed = True

        # Sharpe must improve by threshold
        if improvements.get('sharpe', 0) < self.min_sharpe_improvement:
            passed = False
            reasons.append(f"Sharpe improvement {improvements.get('sharpe', 0):.2%} < {self.min_sharpe_improvement:.2%}")

        # Sortino must improve by threshold
        if improvements.get('sortino', 0) < self.min_sortino_improvement:
            passed = False
            reasons.append(f"Sortino improvement {improvements.get('sortino', 0):.2%} < {self.min_sortino_improvement:.2%}")

        # Max drawdown cannot increase beyond threshold
        if improvements.get('max_drawdown', 0) > MAX_DRAWDOWN_INCREASE:
            passed = False
            reasons.append(f"Drawdown increase {improvements.get('max_drawdown', 0):.2%} > {MAX_DRAWDOWN_INCREASE:.2%}")

        return ValidationResult(
            passed=passed,
            reason='; '.join(reasons) if reasons else 'All criteria met',
            metrics=new_metrics,
            incumbent_metrics=incumbent_metrics,
            improvement_pct=improvements,
        )

    def walk_forward_validation(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        n_splits: int = 5,
        train_ratio: float = 0.7,
    ) -> WalkForwardResult:
        """
        Perform walk-forward validation with expanding window.
        Returns statistics across all folds.
        """
        if len(predictions) < n_splits * self.min_oos_samples:
            raise ValueError("Insufficient data for walk-forward validation")

        sharpe_ratios = []
        sortino_ratios = []
        max_drawdowns = []
        hit_rates = []

        total_len = len(predictions)
        fold_size = total_len // n_splits

        for i in range(n_splits):
            # Expanding window: train on all data up to this point
            train_end = fold_size + i * fold_size
            test_start = train_end
            test_end = min(train_end + fold_size, total_len)

            if test_end - test_start < self.min_oos_samples:
                continue

            # Use predictions directly (assuming they're from a model trained on prior data)
            oos_predictions = predictions[test_start:test_end]
            oos_actuals = actual_returns[test_start:test_end]

            pnl = self.compute_pnl_from_signals(oos_predictions, oos_actuals)

            sharpe_ratios.append(self.compute_sharpe_ratio(pnl))
            sortino_ratios.append(self.compute_sortino_ratio(pnl))
            max_drawdowns.append(self.compute_max_drawdown(pnl))
            hit_rates.append(self.compute_hit_rate(oos_predictions, oos_actuals))

        self._walk_forward_results = WalkForwardResult(
            sharpe_ratios=sharpe_ratios,
            sortino_ratios=sortino_ratios,
            max_drawdowns=max_drawdowns,
            hit_rates=hit_rates,
            avg_sharpe=np.mean(sharpe_ratios),
            avg_sortino=np.mean(sortino_ratios),
            consistency_score=1.0 - np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-8),
        )

        return self._walk_forward_results

    def validate_walk_forward(
        self,
        incumbent_wf: WalkForwardResult,
        new_wf: WalkForwardResult,
    ) -> ValidationResult:
        """Compare two walk-forward results."""
        improvements = {
            'avg_sharpe': (new_wf.avg_sharpe - incumbent_wf.avg_sharpe) / (abs(incumbent_wf.avg_sharpe) + 1e-8),
            'avg_sortino': (new_wf.avg_sortino - incumbent_wf.avg_sortino) / (abs(incumbent_wf.avg_sortino) + 1e-8),
            'consistency': (new_wf.consistency_score - incumbent_wf.consistency_score),
        }

        passed = (
            improvements['avg_sharpe'] >= self.min_sharpe_improvement and
            improvements['avg_sortino'] >= self.min_sortino_improvement
        )

        return ValidationResult(
            passed=passed,
            reason='Walk-forward validation ' + ('passed' if passed else 'failed'),
            metrics={
                'avg_sharpe': new_wf.avg_sharpe,
                'avg_sortino': new_wf.avg_sortino,
                'consistency': new_wf.consistency_score,
            },
            incumbent_metrics={
                'avg_sharpe': incumbent_wf.avg_sharpe,
                'avg_sortino': incumbent_wf.avg_sortino,
                'consistency': incumbent_wf.consistency_score,
            },
            improvement_pct=improvements,
        )

    def statistical_significance_test(
        self,
        incumbent_pnl: np.ndarray,
        new_pnl: np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[bool, float]:
        """
        Perform paired t-test to check if new model is statistically better.
        Returns (is_significantly_better, p_value).
        """
        from scipy import stats

        if len(incumbent_pnl) != len(new_pnl):
            return False, 1.0

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(new_pnl, incumbent_pnl)

        # Check if new is significantly better (one-tailed)
        is_better = np.mean(new_pnl) > np.mean(incumbent_pnl)
        is_significant = p_value < (1 - confidence)

        return is_better and is_significant, p_value


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    validator = ModelValidator(min_oos_samples=100)
    
    # Generate sample data
    n_samples = 1000
    actual_returns = np.random.randn(n_samples) * 0.01
    
    # Incumbent predictions (moderate skill)
    incumbent_preds = actual_returns + np.random.randn(n_samples) * 0.02
    
    # New predictions (slightly better)
    new_preds = actual_returns + np.random.randn(n_samples) * 0.015
    
    # Validate
    result = validator.validate_out_of_sample(incumbent_preds, new_preds, actual_returns)
    
    print(f"Validation passed: {result.passed}")
    print(f"Reason: {result.reason}")
    print(f"New Sharpe: {result.metrics.get('sharpe', 0):.4f}")
    print(f"Incumbent Sharpe: {result.incumbent_metrics.get('sharpe', 0):.4f}")
    print(f"Improvement: {result.improvement_pct.get('sharpe', 0):.2%}")
    
    # Walk-forward validation
    wf_result = validator.walk_forward_validation(new_preds, actual_returns, n_splits=5)
    print(f"\nWalk-forward avg Sharpe: {wf_result.avg_sharpe:.4f}")
    print(f"Consistency score: {wf_result.consistency_score:.4f}")
