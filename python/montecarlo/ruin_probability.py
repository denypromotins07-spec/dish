"""
Statistical calculator for Probability of Ruin, Risk of Drawdown, and Confidence Intervals.
Based on Rust Monte Carlo outputs, ensuring the strategy survives black swan events and fat-tail distributions.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RuinAnalysis:
    """Container for ruin probability analysis results."""
    probability_of_ruin: float
    probability_of_gain: float
    expected_growth_rate: float
    kelly_fraction: float
    optimal_f: float
    risk_of_ruin_1yr: float
    risk_of_ruin_5yr: float
    survival_probability: float


@dataclass
class DrawdownAnalysis:
    """Container for drawdown risk analysis."""
    expected_max_drawdown: float
    drawdown_volatility: float
    drawdown_skewness: float
    drawdown_kurtosis: float
    var_95_drawdown: float
    cvar_95_drawdown: float
    time_to_recovery_days: float
    prolonged_drawdown_probability: float


@dataclass
class ConfidenceIntervalResults:
    """Confidence interval calculations for various metrics."""
    returns_ci_90: Tuple[float, float]
    returns_ci_95: Tuple[float, float]
    returns_ci_99: Tuple[float, float]
    sharpe_ci_95: Tuple[float, float]
    final_equity_ci_95: Tuple[float, float]
    max_drawdown_ci_95: Tuple[float, float]


class RuinProbabilityCalculator:
    """
    Calculate probability of ruin and related risk metrics.
    
    Uses multiple methods:
    - Classical gambler's ruin formula
    - Monte Carlo-based empirical estimation
    - Fat-tail adjusted calculations (Student's t)
    """
    
    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
    
    def calculate_classical_ruin(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        target_capital: Optional[float] = None,
    ) -> float:
        """
        Calculate classical gambler's ruin probability.
        
        Parameters
        ----------
        win_rate : float
            Historical win rate.
        avg_win : float
            Average winning trade amount.
        avg_loss : float
            Average losing trade amount (positive value).
        target_capital : Optional[float]
            Target capital level (default: 2x initial).
            
        Returns
        -------
        float
            Probability of ruin (0-1).
        """
        if win_rate <= 0 or win_rate >= 1:
            return 1.0 if win_rate == 0 else 0.0
        
        # Expected value per trade
        ev = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        if ev <= 0:
            return 1.0  # Negative expectation guarantees ruin
        
        # Ratio of loss to win
        r = avg_loss / max(avg_win, 0.001)
        
        # Gambler's ruin formula
        if target_capital is None:
            target_capital = self.initial_capital * 2
        
        # Probability of reaching target before ruin
        p = win_rate
        q = 1 - win_rate
        
        if p == q:
            # Fair game
            prob_success = self.initial_capital / target_capital
        else:
            # Biased game
            ratio = q / p
            numerator = 1 - ratio ** (self.initial_capital / avg_loss)
            denominator = 1 - ratio ** (target_capital / avg_loss)
            prob_success = numerator / denominator if denominator != 0 else 0
        
        return 1 - prob_success
    
    def calculate_empirical_ruin(
        self,
        monte_carlo_results: List[Dict],
        ruin_threshold: float = 0.5,
    ) -> RuinAnalysis:
        """
        Calculate ruin probability from Monte Carlo simulation results.
        
        Parameters
        ----------
        monte_carlo_results : List[Dict]
            List of simulation results with 'final_equity' key.
        ruin_threshold : float, default 0.5
            Fraction of initial capital considered as "ruin".
            
        Returns
        -------
        RuinAnalysis
            Comprehensive ruin analysis.
        """
        if not monte_carlo_results:
            return RuinAnalysis(
                probability_of_ruin=0.0,
                probability_of_gain=0.0,
                expected_growth_rate=0.0,
                kelly_fraction=0.0,
                optimal_f=0.0,
                risk_of_ruin_1yr=0.0,
                risk_of_ruin_5yr=0.0,
                survival_probability=1.0,
            )
        
        final_equities = [r.get('final_equity', 0) for r in monte_carlo_results]
        n_sims = len(final_equities)
        
        # Empirical ruin probability
        ruin_count = sum(1 for e in final_equities if e < self.initial_capital * ruin_threshold)
        prob_ruin = ruin_count / n_sims
        
        # Probability of gain
        gain_count = sum(1 for e in final_equities if e > self.initial_capital)
        prob_gain = gain_count / n_sims
        
        # Expected growth rate (CAGR approximation)
        median_equity = np.median(final_equities)
        expected_cagr = (median_equity / self.initial_capital) ** (1 / 1.0) - 1
        
        # Kelly fraction estimation
        returns = [(e - self.initial_capital) / self.initial_capital for e in final_equities]
        mean_return = np.mean(returns)
        var_return = np.var(returns)
        
        kelly_fraction = mean_return / max(var_return, 0.0001) if var_return > 0 else 0
        kelly_fraction = max(0, min(1, kelly_fraction))  # Bound to [0, 1]
        
        # Optimal f (Vince's method approximation)
        max_loss = max(abs(r) for r in returns if r < 0) if any(r < 0 for r in returns) else 0.01
        optimal_f = abs(mean_return) / max_loss if max_loss > 0 else 0
        optimal_f = max(0, min(1, optimal_f))
        
        # Time-based ruin probabilities (assuming ~252 trading days/year)
        # Using exponential decay model
        daily_ruin_prob = prob_ruin / 252
        risk_1yr = 1 - (1 - daily_ruin_prob) ** 252
        risk_5yr = 1 - (1 - daily_ruin_prob) ** (252 * 5)
        
        return RuinAnalysis(
            probability_of_ruin=prob_ruin,
            probability_of_gain=prob_gain,
            expected_growth_rate=expected_cagr,
            kelly_fraction=kelly_fraction,
            optimal_f=optimal_f,
            risk_of_ruin_1yr=risk_1yr,
            risk_of_ruin_5yr=risk_5yr,
            survival_probability=1 - prob_ruin,
        )
    
    def calculate_fat_tail_ruin(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate ruin probability accounting for fat tails (Student's t distribution).
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns.
        confidence_level : float, default 0.95
            Confidence level for calculations.
            
        Returns
        -------
        Dict[str, float]
            Fat-tail adjusted risk metrics.
        """
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Fit Student's t distribution
        t_params = stats.t.fit(returns)
        df, loc, scale = t_params
        
        # Fat-tail VaR
        fat_tail_var = stats.t.ppf(1 - confidence_level, df, loc=loc, scale=scale)
        
        # Fat-tail CVaR (Expected Shortfall)
        # Numerical integration for CVaR
        tail_returns = returns[returns <= fat_tail_var]
        fat_tail_cvar = np.mean(tail_returns) if len(tail_returns) > 0 else fat_tail_var
        
        # Adjusted ruin probability with fat tails
        # More conservative than normal distribution
        normal_var = stats.norm.ppf(1 - confidence_level, mean, std)
        fat_tail_multiplier = abs(fat_tail_var) / abs(normal_var) if normal_var != 0 else 1.5
        
        return {
            'degrees_of_freedom': df,
            'fat_tail_var': fat_tail_var,
            'fat_tail_cvar': fat_tail_cvar,
            'fat_tail_multiplier': fat_tail_multiplier,
            'is_fat_tailed': df < 30,  # DF < 30 indicates significant fat tails
        }


class DrawdownRiskAnalyzer:
    """
    Analyze drawdown risks from Monte Carlo simulations.
    """
    
    def analyze_drawdowns(self, equity_curves: List[List[float]]) -> DrawdownAnalysis:
        """
        Analyze drawdown characteristics across multiple equity curves.
        
        Parameters
        ----------
        equity_curves : List[List[float]]
            List of equity curve arrays.
            
        Returns
        -------
        DrawdownAnalysis
            Comprehensive drawdown analysis.
        """
        all_max_drawdowns = []
        all_drawdown_series = []
        
        for curve in equity_curves:
            max_dd, dd_series = self._calculate_drawdown_series(curve)
            all_max_drawdowns.append(max_dd)
            all_drawdown_series.extend(dd_series)
        
        if not all_max_drawdowns:
            return DrawdownAnalysis(
                expected_max_drawdown=0.0,
                drawdown_volatility=0.0,
                drawdown_skewness=0.0,
                drawdown_kurtosis=0.0,
                var_95_drawdown=0.0,
                cvar_95_drawdown=0.0,
                time_to_recovery_days=0.0,
                prolonged_drawdown_probability=0.0,
            )
        
        max_dd_array = np.array(all_max_drawdowns)
        
        # Statistics
        expected_max_dd = np.mean(max_dd_array)
        dd_volatility = np.std(max_dd_array)
        dd_skewness = stats.skew(max_dd_array)
        dd_kurtosis = stats.kurtosis(max_dd_array)
        
        # VaR and CVaR of drawdowns
        var_95 = np.percentile(max_dd_array, 95)
        cvar_95 = np.mean(max_dd_array[max_dd_array >= var_95])
        
        # Time to recovery estimation (simplified)
        avg_recovery = self._estimate_recovery_time(all_drawdown_series)
        
        # Prolonged drawdown probability (>30 days underwater)
        prolonged_prob = self._prolonged_drawdown_probability(all_drawdown_series)
        
        return DrawdownAnalysis(
            expected_max_drawdown=expected_max_dd,
            drawdown_volatility=dd_volatility,
            drawdown_skewness=dd_skewness,
            drawdown_kurtosis=dd_kurtosis,
            var_95_drawdown=var_95,
            cvar_95_drawdown=cvar_95,
            time_to_recovery_days=avg_recovery,
            prolonged_drawdown_probability=prolonged_prob,
        )
    
    def _calculate_drawdown_series(self, equity_curve: List[float]) -> Tuple[float, List[float]]:
        """Calculate maximum drawdown and full drawdown series."""
        if not equity_curve:
            return 0.0, []
        
        peak = equity_curve[0]
        max_dd = 0.0
        dd_series = []
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            dd_series.append(dd)
        
        return max_dd, dd_series
    
    def _estimate_recovery_time(self, drawdown_series: List[float]) -> float:
        """Estimate average time to recover from drawdowns."""
        # Simplified: count consecutive periods underwater
        recovery_times = []
        underwater_start = None
        
        for i, dd in enumerate(drawdown_series):
            if dd > 0.01 and underwater_start is None:  # >1% drawdown
                underwater_start = i
            elif dd <= 0.01 and underwater_start is not None:
                recovery_times.append(i - underwater_start)
                underwater_start = None
        
        return np.mean(recovery_times) if recovery_times else 0.0
    
    def _prolonged_drawdown_probability(self, drawdown_series: List[float], threshold_days: int = 30) -> float:
        """Calculate probability of prolonged drawdown periods."""
        consecutive_underwater = 0
        max_consecutive = 0
        
        for dd in drawdown_series:
            if dd > 0.05:  # >5% drawdown
                consecutive_underwater += 1
                max_consecutive = max(max_consecutive, consecutive_underwater)
            else:
                consecutive_underwater = 0
        
        return 1.0 if max_consecutive >= threshold_days else max_consecutive / threshold_days


def calculate_confidence_intervals(monte_carlo_results: List[Dict]) -> ConfidenceIntervalResults:
    """
    Calculate comprehensive confidence intervals from Monte Carlo results.
    
    Parameters
    ----------
    monte_carlo_results : List[Dict]
        Simulation results with various metrics.
        
    Returns
    -------
    ConfidenceIntervalResults
        Confidence intervals for key metrics.
    """
    if not monte_carlo_results:
        return ConfidenceIntervalResults(
            returns_ci_90=(0, 0),
            returns_ci_95=(0, 0),
            returns_ci_99=(0, 0),
            sharpe_ci_95=(0, 0),
            final_equity_ci_95=(0, 0),
            max_drawdown_ci_95=(0, 0),
        )
    
    # Extract metrics
    final_equities = [r.get('final_equity', 0) for r in monte_carlo_results]
    sharpe_ratios = [r.get('sharpe_ratio', 0) for r in monte_carlo_results]
    max_drawdowns = [r.get('max_drawdown', 0) for r in monte_carlo_results]
    
    # Calculate returns
    initial = 100000  # Assumed
    returns = [(e - initial) / initial for e in final_equities]
    
    return ConfidenceIntervalResults(
        returns_ci_90=(np.percentile(returns, 5), np.percentile(returns, 95)),
        returns_ci_95=(np.percentile(returns, 2.5), np.percentile(returns, 97.5)),
        returns_ci_99=(np.percentile(returns, 0.5), np.percentile(returns, 99.5)),
        sharpe_ci_95=(np.percentile(sharpe_ratios, 2.5), np.percentile(sharpe_ratios, 97.5)),
        final_equity_ci_95=(np.percentile(final_equities, 2.5), np.percentile(final_equities, 97.5)),
        max_drawdown_ci_95=(np.percentile(max_drawdowns, 2.5), np.percentile(max_drawdowns, 97.5)),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    calculator = RuinProbabilityCalculator(initial_capital=100000)
    
    # Mock Monte Carlo results
    mock_results = [
        {'final_equity': 120000, 'sharpe_ratio': 1.5, 'max_drawdown': 0.08},
        {'final_equity': 95000, 'sharpe_ratio': 0.8, 'max_drawdown': 0.15},
        {'final_equity': 150000, 'sharpe_ratio': 2.1, 'max_drawdown': 0.05},
        {'final_equity': 80000, 'sharpe_ratio': -0.2, 'max_drawdown': 0.25},
        {'final_equity': 110000, 'sharpe_ratio': 1.2, 'max_drawdown': 0.10},
    ] * 20  # Repeat for better statistics
    
    # Calculate ruin analysis
    ruin_analysis = calculator.calculate_empirical_ruin(mock_results, ruin_threshold=0.5)
    
    print("Ruin Analysis:")
    print(f"  Probability of Ruin: {ruin_analysis.probability_of_ruin:.2%}")
    print(f"  Probability of Gain: {ruin_analysis.probability_of_gain:.2%}")
    print(f"  Kelly Fraction: {ruin_analysis.kelly_fraction:.2%}")
    print(f"  Risk of Ruin (1yr): {ruin_analysis.risk_of_ruin_1yr:.2%}")
    print(f"  Risk of Ruin (5yr): {ruin_analysis.risk_of_ruin_5yr:.2%}")
    
    # Calculate confidence intervals
    ci_results = calculate_confidence_intervals(mock_results)
    
    print("\nConfidence Intervals (95%):")
    print(f"  Returns: {ci_results.returns_ci_95}")
    print(f"  Sharpe: {ci_results.sharpe_ci_95}")
    print(f"  Final Equity: {ci_results.final_equity_ci_95}")
    print(f"  Max Drawdown: {ci_results.max_drawdown_ci_95}")
