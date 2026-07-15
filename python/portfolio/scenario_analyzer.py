"""
Scenario analysis tool applying macroeconomic shocks to the portfolio's correlation matrix
to predict hidden factor exposures and tail risks.

Memory-efficient implementation bounded for <14GB total system RAM.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class MacroFactorType(Enum):
    INTEREST_RATES = "interest_rates"
    DXY = "dxy"  # Dollar Index
    VIX = "vix"  # Volatility
    CREDIT_SPREAD = "credit_spread"
    LIQUIDITY = "liquidity"
    RISK_APPETITE = "risk_appetite"


@dataclass(slots=True)
class MacroShock:
    """Defines a macroeconomic shock scenario."""
    name: str
    factor_type: MacroFactorType
    magnitude: float  # In standard deviations or basis points
    description: str
    
    # Factor loadings per asset (how each asset responds to this factor)
    factor_loadings: Optional[np.ndarray] = None
    
    # Correlation impact (how correlations change under stress)
    correlation_multiplier: float = 1.0  # >1 means correlations increase


@dataclass(slots=True)
class FactorExposure:
    """Portfolio exposure to a macro factor."""
    factor_name: str
    exposure: float
    sensitivity: float  # Change in portfolio value per unit factor move
    contribution_to_risk: float


@dataclass(slots=True)
class ScenarioAnalysisResult:
    """Results from scenario analysis."""
    scenario_name: str
    base_portfolio_value: float
    stressed_portfolio_value: float
    value_change: float
    value_change_pct: float
    
    # Factor exposures
    factor_exposures: List[FactorExposure]
    
    # Risk metrics
    base_correlation: float
    stressed_correlation: float
    correlation_change: float
    
    # Tail risk indicators
    tail_risk_score: float  # 0-100 scale
    hidden_risks: List[str]


class ScenarioAnalyzer:
    """
    Analyzes portfolio under various macroeconomic scenarios.
    Identifies hidden factor exposures and tail risks.
    """
    
    __slots__ = (
        'portfolio_weights', 'asset_names', 'base_correlation',
        'base_covariance', 'factor_models', 'current_scenarios'
    )
    
    def __init__(
        self,
        portfolio_weights: np.ndarray,
        asset_names: List[str],
        base_correlation: np.ndarray,
        base_covariance: Optional[np.ndarray] = None
    ):
        self.portfolio_weights = portfolio_weights
        self.asset_names = asset_names
        self.base_correlation = base_correlation
        self.base_covariance = base_covariance
        self.factor_models: Dict[MacroFactorType, np.ndarray] = {}
        self.current_scenarios: List[MacroShock] = []
        
        n_assets = len(asset_names)
        if base_covariance is None:
            # Derive from correlation assuming 20% vol
            vols = np.full(n_assets, 0.20)
            self.base_covariance = np.outer(vols, vols) * base_correlation
    
    def set_factor_loadings(
        self,
        factor_type: MacroFactorType,
        loadings: np.ndarray
    ) -> None:
        """Set factor loadings for a macro factor."""
        assert len(loadings) == len(self.asset_names)
        self.factor_models[factor_type] = loadings.copy()
    
    def add_scenario(self, scenario: MacroShock) -> None:
        """Add a scenario to analyze."""
        self.current_scenarios.append(scenario)
    
    def clear_scenarios(self) -> None:
        """Clear all scenarios."""
        self.current_scenarios.clear()
    
    def _apply_correlation_stress(
        self,
        base_corr: np.ndarray,
        multiplier: float
    ) -> np.ndarray:
        """
        Apply correlation stress.
        Under stress, correlations tend toward 1 (or -1).
        """
        n = base_corr.shape[0]
        
        # Correlations move toward 1 under stress
        stressed_corr = base_corr + (1 - base_corr) * (multiplier - 1) / multiplier
        
        # Ensure valid correlation matrix
        np.fill_diagonal(stressed_corr, 1.0)
        
        # Clip to valid range
        stressed_corr = np.clip(stressed_corr, -1.0, 1.0)
        
        # Make symmetric
        stressed_corr = (stressed_corr + stressed_corr.T) / 2
        
        return stressed_corr
    
    def _calculate_factor_exposure(
        self,
        factor_type: MacroFactorType,
        scenario: MacroShock
    ) -> FactorExposure:
        """Calculate portfolio exposure to a factor."""
        if factor_type not in self.factor_models:
            # Default: assume moderate positive exposure for all assets
            loadings = np.ones(len(self.asset_names)) * 0.5
        else:
            loadings = self.factor_models[factor_type]
        
        # Portfolio-level exposure
        portfolio_loading = np.dot(self.portfolio_weights, loadings)
        
        # Sensitivity: change in portfolio value per unit factor move
        sensitivity = portfolio_loading * scenario.magnitude
        
        # Contribution to risk (simplified)
        contribution = abs(sensitivity) * np.std(loadings)
        
        return FactorExposure(
            factor_name=factor_type.value,
            exposure=portfolio_loading,
            sensitivity=sensitivity,
            contribution_to_risk=contribution
        )
    
    def _calculate_tail_risk_score(
        self,
        base_corr: np.ndarray,
        stressed_corr: np.ndarray,
        scenarios: List[MacroShock]
    ) -> Tuple[float, List[str]]:
        """
        Calculate tail risk score and identify hidden risks.
        Score is 0-100, higher = more risk.
        """
        risks = []
        score = 0.0
        
        # Factor 1: Correlation increase
        avg_base = np.mean(base_corr[np.triu_indices(len(base_corr), 1)])
        avg_stressed = np.mean(stressed_corr[np.triu_indices(len(stressed_corr), 1)])
        corr_increase = avg_stressed - avg_base
        
        if corr_increase > 0.2:
            score += 30
            risks.append("Severe correlation breakdown (diversification failure)")
        elif corr_increase > 0.1:
            score += 15
            risks.append("Moderate correlation increase")
        
        # Factor 2: Number of simultaneous shocks
        n_shocks = len(scenarios)
        if n_shocks >= 3:
            score += 25
            risks.append("Multiple concurrent macro shocks")
        elif n_shocks >= 2:
            score += 10
        
        # Factor 3: Magnitude of shocks
        total_magnitude = sum(abs(s.magnitude) for s in scenarios)
        if total_magnitude > 3:  # More than 3 std moves
            score += 25
            risks.append("Extreme shock magnitudes")
        elif total_magnitude > 2:
            score += 15
            risks.append("Large shock magnitudes")
        
        # Factor 4: Concentration risk
        max_weight = np.max(self.portfolio_weights)
        if max_weight > 0.4:
            score += 20
            risks.append(f"High concentration ({max_weight:.1%} in single asset)")
        
        return min(score, 100.0), risks
    
    def analyze_scenario(self, scenario: MacroShock) -> ScenarioAnalysisResult:
        """Analyze a single scenario."""
        n_assets = len(self.asset_names)
        
        # Apply correlation stress
        stressed_corr = self._apply_correlation_stress(
            self.base_correlation,
            scenario.correlation_multiplier
        )
        
        # Calculate factor exposure
        factor_exposure = self._calculate_factor_exposure(scenario.factor_type, scenario)
        
        # Estimate portfolio value change
        # Simplified: linear response to factor
        base_value = 1_000_000  # Normalized
        value_change = -base_value * factor_exposure.sensitivity
        stressed_value = base_value + value_change
        
        # Calculate average correlations
        triu_idx = np.triu_indices(n_assets, 1)
        base_avg_corr = np.mean(self.base_correlation[triu_idx])
        stressed_avg_corr = np.mean(stressed_corr[triu_idx])
        
        # Tail risk
        tail_score, hidden_risks = self._calculate_tail_risk_score(
            self.base_correlation,
            stressed_corr,
            [scenario]
        )
        
        return ScenarioAnalysisResult(
            scenario_name=scenario.name,
            base_portfolio_value=base_value,
            stressed_portfolio_value=stressed_value,
            value_change=value_change,
            value_change_pct=value_change / base_value,
            factor_exposures=[factor_exposure],
            base_correlation=base_avg_corr,
            stressed_correlation=stressed_avg_corr,
            correlation_change=stressed_avg_corr - base_avg_corr,
            tail_risk_score=tail_score,
            hidden_risks=hidden_risks
        )
    
    def analyze_combined_scenarios(self) -> ScenarioAnalysisResult:
        """Analyze all current scenarios combined."""
        if not self.current_scenarios:
            raise ValueError("No scenarios to analyze")
        
        n_assets = len(self.asset_names)
        
        # Combined correlation stress
        total_multiplier = 1.0
        for scenario in self.current_scenarios:
            total_multiplier *= scenario.correlation_multiplier
        
        stressed_corr = self._apply_correlation_stress(
            self.base_correlation,
            total_multiplier
        )
        
        # Combined factor exposures
        factor_exposures = []
        total_sensitivity = 0.0
        
        for scenario in self.current_scenarios:
            exposure = self._calculate_factor_exposure(scenario.factor_type, scenario)
            factor_exposures.append(exposure)
            total_sensitivity += exposure.sensitivity
        
        # Portfolio value impact
        base_value = 1_000_000
        value_change = -base_value * total_sensitivity
        stressed_value = base_value + value_change
        
        # Correlations
        triu_idx = np.triu_indices(n_assets, 1)
        base_avg_corr = np.mean(self.base_correlation[triu_idx])
        stressed_avg_corr = np.mean(stressed_corr[triu_idx])
        
        # Tail risk
        tail_score, hidden_risks = self._calculate_tail_risk_score(
            self.base_correlation,
            stressed_corr,
            self.current_scenarios
        )
        
        return ScenarioAnalysisResult(
            scenario_name="Combined Scenarios",
            base_portfolio_value=base_value,
            stressed_portfolio_value=stressed_value,
            value_change=value_change,
            value_change_pct=value_change / base_value,
            factor_exposures=factor_exposures,
            base_correlation=base_avg_corr,
            stressed_correlation=stressed_avg_corr,
            correlation_change=stressed_avg_corr - base_avg_corr,
            tail_risk_score=tail_score,
            hidden_risks=hidden_risks
        )
    
    def generate_report(self, results: List[ScenarioAnalysisResult]) -> str:
        """Generate human-readable scenario analysis report."""
        lines = [
            "=" * 70,
            "MACROECONOMIC SCENARIO ANALYSIS REPORT",
            "=" * 70,
            f"Portfolio: {len(self.asset_names)} assets",
            "",
            "SCENARIO RESULTS:",
            "-" * 70
        ]
        
        for result in results:
            lines.extend([
                f"\n{result.scenario_name}:",
                f"  Portfolio Value Change: ${result.value_change:,.2f} ({result.value_change_pct:.2%})",
                f"  ",
                f"  Factor Exposures:",
            ])
            
            for exp in result.factor_exposures:
                lines.append(
                    f"    {exp.factor_name}: exposure={exp.exposure:.3f}, "
                    f"sensitivity=${exp.sensitivity:,.2f}"
                )
            
            lines.extend([
                f"  ",
                f"  Correlation Analysis:",
                f"    Base Avg Correlation: {result.base_correlation:.3f}",
                f"    Stressed Avg Correlation: {result.stressed_correlation:.3f}",
                f"    Change: {result.correlation_change:+.3f}",
                f"  ",
                f"  Tail Risk Score: {result.tail_risk_score:.0f}/100",
            ])
            
            if result.hidden_risks:
                lines.append("  Hidden Risks Identified:")
                for risk in result.hidden_risks:
                    lines.append(f"    ⚠ {risk}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


# Pre-built scenario templates
def create_rate_hike_scenario(magnitude_bps: float = 200) -> MacroShock:
    """Create an interest rate hike scenario."""
    return MacroShock(
        name=f"+{magnitude_bps}bps Rate Hike",
        factor_type=MacroFactorType.INTEREST_RATES,
        magnitude=magnitude_bps / 10000,  # Convert to decimal
        description=f"Federal Reserve raises rates by {magnitude_bps} basis points",
        correlation_multiplier=1.3  # Correlations increase under rate stress
    )


def create_dxy_spike_scenario(magnitude_pct: float = 10) -> MacroShock:
    """Create a dollar spike scenario."""
    return MacroShock(
        name=f"DXY +{magnitude_pct}% Spike",
        factor_type=MacroFactorType.DXY,
        magnitude=magnitude_pct / 100,
        description=f"US Dollar Index spikes {magnitude_pct}%",
        correlation_multiplier=1.4  # Strong dollar increases correlations
    )


def create_vix_surge_scenario(magnitude_pct: float = 50) -> MacroShock:
    """Create a volatility surge scenario."""
    return MacroShock(
        name=f"VIX +{magnitude_pct}% Surge",
        factor_type=MacroFactorType.VIX,
        magnitude=magnitude_pct / 100,
        description=f"VIX volatility index surges {magnitude_pct}%",
        correlation_multiplier=1.5  # High vol = high correlations
    )


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Create sample portfolio
    n_assets = 5
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    asset_names = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
    
    # Base correlation matrix (moderate correlations)
    base_corr = np.array([
        [1.00, 0.70, 0.60, 0.55, 0.50],
        [0.70, 1.00, 0.65, 0.60, 0.55],
        [0.60, 0.65, 1.00, 0.70, 0.60],
        [0.55, 0.60, 0.70, 1.00, 0.65],
        [0.50, 0.55, 0.60, 0.65, 1.00]
    ])
    
    # Initialize analyzer
    analyzer = ScenarioAnalyzer(weights, asset_names, base_corr)
    
    # Set some factor loadings (how assets respond to factors)
    # Negative loadings for rate hikes (crypto typically falls when rates rise)
    analyzer.set_factor_loadings(
        MacroFactorType.INTEREST_RATES,
        np.array([-0.8, -0.7, -0.9, -0.85, -0.95])
    )
    
    # Positive loadings for DXY (inverse relationship)
    analyzer.set_factor_loadings(
        MacroFactorType.DXY,
        np.array([-0.6, -0.5, -0.7, -0.6, -0.65])
    )
    
    # Add scenarios
    analyzer.add_scenario(create_rate_hike_scenario(200))
    analyzer.add_scenario(create_dxy_spike_scenario(10))
    analyzer.add_scenario(create_vix_surge_scenario(50))
    
    # Run analysis
    results = []
    for scenario in analyzer.current_scenarios:
        result = analyzer.analyze_scenario(scenario)
        results.append(result)
    
    # Combined analysis
    combined = analyzer.analyze_combined_scenarios()
    results.append(combined)
    
    # Print report
    print(analyzer.generate_report(results))
