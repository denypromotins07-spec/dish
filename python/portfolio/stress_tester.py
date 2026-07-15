"""
Historical and hypothetical stress testing engine.
Evaluates portfolio drawdown limits and margin call risks without bloating RAM.

Memory-efficient implementation bounded for <14GB total system RAM.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime


class StressScenarioType(Enum):
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"


@dataclass(slots=True)
class StressScenario:
    """Defines a stress test scenario."""
    name: str
    scenario_type: StressScenarioType
    description: str
    
    # For historical scenarios
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    historical_returns: Optional[np.ndarray] = None
    
    # For hypothetical scenarios
    shock_magnitude: float = 0.0  # e.g., -0.50 for -50% shock
    shock_correlation: float = 1.0  # How correlated is the shock across assets
    
    # For Monte Carlo
    n_simulations: int = 10000
    confidence_levels: Tuple[float, ...] = (0.95, 0.99)


@dataclass(slots=True)
class StressTestResult:
    """Results from a stress test."""
    scenario_name: str
    portfolio_return: float
    portfolio_drawdown: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_loss: float
    margin_call_probability: float
    breach_count: int  # Number of simulations breaching threshold
    total_simulations: int


@dataclass(slots=True)
class PortfolioSnapshot:
    """Snapshot of portfolio for stress testing."""
    weights: np.ndarray
    asset_names: List[str]
    current_value: float
    leverage: float = 1.0
    margin_threshold: float = 0.1  # Margin call at 10% equity


class StressTester:
    """
    High-performance stress testing engine with minimal memory footprint.
    """
    
    __slots__ = ('portfolio', 'covariance_matrix', 'returns_history', 'custom_scenarios')
    
    def __init__(
        self,
        portfolio: PortfolioSnapshot,
        covariance_matrix: Optional[np.ndarray] = None,
        returns_history: Optional[np.ndarray] = None
    ):
        self.portfolio = portfolio
        self.covariance_matrix = covariance_matrix
        self.returns_history = returns_history
        self.custom_scenarios: List[StressScenario] = []
    
    def add_historical_scenario(
        self,
        name: str,
        description: str,
        returns: np.ndarray,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> None:
        """Add a historical scenario (e.g., March 2020 crash)."""
        scenario = StressScenario(
            name=name,
            scenario_type=StressScenarioType.HISTORICAL,
            description=description,
            start_date=start_date,
            end_date=end_date,
            historical_returns=returns.copy()
        )
        self.custom_scenarios.append(scenario)
    
    def add_hypothetical_scenario(
        self,
        name: str,
        description: str,
        shock_magnitude: float,
        shock_correlation: float = 1.0
    ) -> None:
        """Add a hypothetical scenario (e.g., BTC -50%)."""
        scenario = StressScenario(
            name=name,
            scenario_type=StressScenarioType.HYPOTHETICAL,
            description=description,
            shock_magnitude=shock_magnitude,
            shock_correlation=shock_correlation
        )
        self.custom_scenarios.append(scenario)
    
    def _get_stress_returns(self, scenario: StressScenario) -> np.ndarray:
        """Generate stressed returns based on scenario type."""
        n_assets = len(self.portfolio.asset_names)
        
        if scenario.scenario_type == StressScenarioType.HISTORICAL:
            if scenario.historical_returns is not None:
                return scenario.historical_returns
            # Fallback: use stored history
            if self.returns_history is not None:
                return self.returns_history
        
        elif scenario.scenario_type == StressScenarioType.HYPOTHETICAL:
            # Generate correlated shock
            base_shock = scenario.shock_magnitude
            
            if scenario.shock_correlation >= 1.0:
                # Perfect correlation - all assets move together
                shocks = np.full(n_assets, base_shock)
            else:
                # Partial correlation
                idio_vol = np.sqrt(1 - scenario.shock_correlation ** 2)
                idio_shocks = np.random.randn(n_assets) * idio_vol * abs(base_shock)
                systematic = base_shock * scenario.shock_correlation
                shocks = systematic + idio_shocks
            
            return shocks.reshape(1, -1)
        
        # Default: no shock
        return np.zeros((1, n_assets))
    
    def run_stress_test(self, scenario: StressScenario) -> StressTestResult:
        """Run a single stress test scenario."""
        n_assets = len(self.portfolio.asset_names)
        weights = self.portfolio.weights
        
        # Get stressed returns
        stress_returns = self._get_stress_returns(scenario)
        
        if scenario.scenario_type == StressScenarioType.MONTE_CARLO:
            # Monte Carlo simulation
            n_sims = scenario.n_simulations
            
            if self.covariance_matrix is not None:
                # Generate correlated returns
                mean_returns = np.zeros(n_assets)
                simulated_returns = np.random.multivariate_normal(
                    mean_returns,
                    self.covariance_matrix,
                    size=n_sims
                )
                
                # Apply stress scaling
                if scenario.shock_magnitude != 0:
                    simulated_returns *= (1 + scenario.shock_magnitude)
            else:
                # Simple independent shocks
                vol = 0.05  # Default 5% daily vol
                simulated_returns = np.random.randn(n_sims, n_assets) * vol
            
            # Calculate portfolio returns
            port_returns = simulated_returns @ weights
            
            # Account for leverage
            if self.portfolio.leverage > 1:
                port_returns *= self.portfolio.leverage
            
        else:
            # Single scenario or historical
            port_returns = stress_returns @ weights
            if self.portfolio.leverage > 1:
                port_returns *= self.portfolio.leverage
            
            # Expand for uniform interface
            port_returns = port_returns.flatten()
            n_sims = len(port_returns)
        
        # Calculate metrics
        portfolio_return = np.mean(port_returns)
        portfolio_drawdown = np.min(port_returns)  # Worst case
        
        # VaR calculations
        var_95 = np.percentile(port_returns, 5)
        var_99 = np.percentile(port_returns, 1)
        
        # Expected Shortfall (CVaR)
        es_95 = np.mean(port_returns[port_returns <= var_95]) if len(port_returns[port_returns <= var_95]) > 0 else var_95
        
        # Max loss
        max_loss = np.min(port_returns)
        
        # Margin call probability
        # Margin call if loss exceeds (1 / leverage - margin_threshold)
        if self.portfolio.leverage > 1:
            margin_call_threshold = -(1.0 / self.portfolio.leverage - self.portfolio.margin_threshold)
            margin_calls = port_returns < margin_call_threshold
            margin_call_prob = np.mean(margin_calls)
            breach_count = np.sum(margin_calls)
        else:
            margin_call_threshold = -self.portfolio.margin_threshold
            margin_calls = port_returns < margin_call_threshold
            margin_call_prob = np.mean(margin_calls)
            breach_count = np.sum(margin_calls)
        
        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_return=portfolio_return,
            portfolio_drawdown=portfolio_drawdown,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=es_95,
            max_loss=max_loss,
            margin_call_probability=margin_call_prob,
            breach_count=int(breach_count),
            total_simulations=n_sims
        )
    
    def run_all_scenarios(self) -> List[StressTestResult]:
        """Run all registered scenarios."""
        results = []
        
        # Built-in scenarios
        built_in = [
            StressScenario(
                name="BTC Flash Crash -50%",
                scenario_type=StressScenarioType.HYPOTHETICAL,
                description="Bitcoin drops 50% in one day",
                shock_magnitude=-0.50,
                shock_correlation=0.7
            ),
            StressScenario(
                name="Market Wide Crash -30%",
                scenario_type=StressScenarioType.HYPOTHETICAL,
                description="All crypto assets drop 30%",
                shock_magnitude=-0.30,
                shock_correlation=0.9
            ),
            StressScenario(
                name="Mild Correction -10%",
                scenario_type=StressScenarioType.HYPOTHETICAL,
                description="Market correction of 10%",
                shock_magnitude=-0.10,
                shock_correlation=0.8
            ),
        ]
        
        all_scenarios = built_in + self.custom_scenarios
        
        for scenario in all_scenarios:
            try:
                result = self.run_stress_test(scenario)
                results.append(result)
            except Exception as e:
                print(f"Error running scenario {scenario.name}: {e}")
                continue
        
        return results
    
    def get_worst_case(self, results: List[StressTestResult]) -> Optional[StressTestResult]:
        """Find the worst-case scenario from results."""
        if not results:
            return None
        
        return min(results, key=lambda r: r.max_loss)
    
    def generate_report(self, results: List[StressTestResult]) -> str:
        """Generate human-readable stress test report."""
        lines = [
            "=" * 60,
            "STRESS TEST REPORT",
            "=" * 60,
            f"Portfolio Value: ${self.portfolio.current_value:,.2f}",
            f"Leverage: {self.portfolio.leverage}x",
            f"Assets: {len(self.portfolio.asset_names)}",
            "",
            "SCENARIO RESULTS:",
            "-" * 60
        ]
        
        for result in results:
            lines.extend([
                f"\n{result.scenario_name}:",
                f"  Expected Return: {result.portfolio_return:.2%}",
                f"  Worst Case: {result.max_loss:.2%}",
                f"  VaR (95%): {result.var_95:.2%}",
                f"  VaR (99%): {result.var_99:.2%}",
                f"  Expected Shortfall: {result.expected_shortfall:.2%}",
                f"  Margin Call Probability: {result.margin_call_probability:.2%}",
                f"  Breaches: {result.breach_count}/{result.total_simulations}"
            ])
        
        worst = self.get_worst_case(results)
        if worst:
            lines.extend([
                "",
                "-" * 60,
                f"WORST CASE SCENARIO: {worst.scenario_name}",
                f"  Maximum Loss: {worst.max_loss:.2%}",
                f"  Dollar Loss: ${self.portfolio.current_value * abs(worst.max_loss):,.2f}",
                "=" * 60
            ])
        
        return "\n".join(lines)


# Pre-defined historical scenarios
def create_march_2020_scenario() -> StressScenario:
    """March 2020 COVID crash scenario."""
    return StressScenario(
        name="March 2020 COVID Crash",
        scenario_type=StressScenarioType.HISTORICAL,
        description="COVID-19 market crash, March 2020",
        shock_magnitude=-0.40,
        shock_correlation=0.85
    )


def create_luna_collapse_scenario() -> StressScenario:
    """LUNA/UST collapse scenario."""
    return StressScenario(
        name="LUNA/UST Collapse",
        scenario_type=StressScenarioType.HISTORICAL,
        description="Terra/LUNA ecosystem collapse, May 2022",
        shock_magnitude=-0.35,
        shock_correlation=0.6  # More idiosyncratic
    )


def create_ftx_collapse_scenario() -> StressScenario:
    """FTX collapse scenario."""
    return StressScenario(
        name="FTX Collapse",
        scenario_type=StressScenarioType.HISTORICAL,
        description="FTX exchange collapse, November 2022",
        shock_magnitude=-0.25,
        shock_correlation=0.7
    )


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Create a sample portfolio
    portfolio = PortfolioSnapshot(
        weights=np.array([0.4, 0.3, 0.2, 0.1]),
        asset_names=['BTC', 'ETH', 'SOL', 'AVAX'],
        current_value=1_000_000,
        leverage=3.0,
        margin_threshold=0.1
    )
    
    # Create covariance matrix
    cov = np.array([
        [0.04, 0.02, 0.015, 0.01],
        [0.02, 0.06, 0.025, 0.015],
        [0.015, 0.025, 0.09, 0.03],
        [0.01, 0.015, 0.03, 0.10]
    ])
    
    # Initialize stress tester
    tester = StressTester(portfolio, covariance_matrix=cov)
    
    # Add custom scenarios
    tester.add_hypothetical_scenario(
        "Depeg Event",
        "Stablecoin depeg causes market panic",
        shock_magnitude=-0.20,
        shock_correlation=0.5
    )
    
    # Run stress tests
    results = tester.run_all_scenarios()
    
    # Print report
    print(tester.generate_report(results))
