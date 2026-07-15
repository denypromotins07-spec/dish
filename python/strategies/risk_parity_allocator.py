"""
Risk Parity and Inverse-Volatility Capital Allocation Engine
Distributes funds across active sub-strategies safely
Ensures no single strategy can blow up the portfolio during regime shift
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class RiskMetric(Enum):
    VOLATILITY = "volatility"
    VAR = "var"  # Value at Risk
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class StrategyAllocation:
    """Capital allocation for a single strategy"""
    strategy_id: str
    allocated_capital: float
    weight: float
    risk_contribution: float
    volatility: float
    sharpe_ratio: float
    max_position_size: float
    timestamp_ns: int


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_capital: float
    allocated_capital: float
    available_capital: float
    portfolio_volatility: float
    portfolio_var: float
    strategy_weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    timestamp_ns: int


class RiskParityAllocator:
    """
    Risk parity and inverse-volatility capital allocation engine
    Distributes available funds across active sub-strategies safely
    """
    
    def __init__(
        self,
        total_capital: float,
        max_single_strategy_weight: float = 0.4,
        min_strategy_weight: float = 0.05,
        target_portfolio_volatility: float = 0.15,
        lookback_days: int = 60,
    ):
        self.total_capital = total_capital
        self.max_single_strategy_weight = max_single_strategy_weight
        self.min_strategy_weight = min_strategy_weight
        self.target_portfolio_volatility = target_portfolio_volatility
        self.lookback_days = lookback_days
        
        # Strategy metrics storage
        self.strategy_returns: Dict[str, List[float]] = {}
        self.strategy_volatilities: Dict[str, float] = {}
        self.strategy_correlations: Dict[Tuple[str, str], float] = {}
        self.strategy_sharpes: Dict[str, float] = {}
        
        # Current allocations
        self.allocations: Dict[str, StrategyAllocation] = {}
        
        # Constraints
        self.max_leverage = 1.0  # No leverage by default
        self.min_diversification = 0.5  # Minimum effective number of strategies
    
    def register_strategy(self, strategy_id: str) -> None:
        """Register a new strategy for allocation"""
        if strategy_id not in self.strategy_returns:
            self.strategy_returns[strategy_id] = []
            self.strategy_volatilities[strategy_id] = 0.0
            self.strategy_sharpes[strategy_id] = 0.0
    
    def unregister_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from allocation"""
        if strategy_id in self.strategy_returns:
            del self.strategy_returns[strategy_id]
        if strategy_id in self.strategy_volatilities:
            del self.strategy_volatilities[strategy_id]
        if strategy_id in self.strategy_sharpes:
            del self.strategy_sharpes[strategy_id]
        if strategy_id in self.allocations:
            del self.allocations[strategy_id]
    
    def update_strategy_returns(self, strategy_id: str, returns: List[float]) -> None:
        """Update historical returns for a strategy"""
        if strategy_id in self.strategy_returns:
            self.strategy_returns[strategy_id].extend(returns)
            # Keep only lookback period
            max_returns = self.lookback_days * 24  # Assuming hourly returns
            if len(self.strategy_returns[strategy_id]) > max_returns:
                self.strategy_returns[strategy_id] = self.strategy_returns[strategy_id][-max_returns:]
            
            # Update volatility estimate
            self._update_volatility(strategy_id)
            self._update_sharpe(strategy_id)
    
    def _update_volatility(self, strategy_id: str) -> None:
        """Update volatility estimate for a strategy"""
        returns = self.strategy_returns.get(strategy_id, [])
        if len(returns) >= 10:
            vol = np.std(returns) * np.sqrt(252)  # Annualized
            self.strategy_volatilities[strategy_id] = vol
    
    def _update_sharpe(self, strategy_id: str) -> None:
        """Update Sharpe ratio estimate"""
        returns = self.strategy_returns.get(strategy_id, [])
        if len(returns) >= 10:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret > 0:
                self.strategy_sharpes[strategy_id] = (mean_ret / std_ret) * np.sqrt(252)
    
    def calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix between strategies"""
        strategy_ids = list(self.strategy_returns.keys())
        n = len(strategy_ids)
        
        if n < 2:
            return np.array([[1.0]])
        
        # Build return matrix
        min_length = min(len(self.strategy_returns[s]) for s in strategy_ids)
        if min_length < 10:
            return np.eye(n)
        
        returns_matrix = np.zeros((min_length, n))
        for i, sid in enumerate(strategy_ids):
            returns_matrix[:, i] = self.strategy_returns[sid][-min_length:]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_matrix.T)
        return np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    def calculate_risk_parity_weights(self) -> Dict[str, float]:
        """
        Calculate risk parity weights using inverse volatility with correlation adjustment
        """
        strategy_ids = list(self.strategy_returns.keys())
        n = len(strategy_ids)
        
        if n == 0:
            return {}
        
        if n == 1:
            return {strategy_ids[0]: 1.0}
        
        # Get volatilities
        vols = np.array([self.strategy_volatilities.get(s, 0.1) for s in strategy_ids])
        vols = np.clip(vols, 0.01, None)  # Floor volatility
        
        # Get correlation matrix
        corr_matrix = self.calculate_correlation_matrix()
        
        # Calculate covariance matrix
        cov_matrix = np.outer(vols, vols) * corr_matrix
        
        # Risk parity: equal risk contribution
        # Solve for weights where each strategy contributes equally to portfolio risk
        try:
            # Inverse volatility weighting as starting point
            inv_vols = 1.0 / vols
            initial_weights = inv_vols / np.sum(inv_vols)
            
            # Apply constraints
            initial_weights = np.clip(
                initial_weights,
                self.min_strategy_weight,
                self.max_single_strategy_weight
            )
            initial_weights = initial_weights / np.sum(initial_weights)
            
            # Calculate risk contributions
            portfolio_var = initial_weights @ cov_matrix @ initial_weights
            marginal_risk = cov_matrix @ initial_weights
            risk_contrib = initial_weights * marginal_risk
            
            # Normalize to get percentage risk contribution
            total_risk_contrib = np.sum(risk_contrib)
            if total_risk_contrib > 0:
                risk_pct = risk_contrib / total_risk_contrib
            else:
                risk_pct = np.ones(n) / n
            
            # Store risk contributions
            for i, sid in enumerate(strategy_ids):
                if sid in self.allocations:
                    self.allocations[sid].risk_contribution = risk_pct[i]
            
            weights = {sid: w for sid, w in zip(strategy_ids, initial_weights)}
            
        except Exception as e:
            # Fallback to equal weighting
            weights = {sid: 1.0 / n for sid in strategy_ids}
        
        return weights
    
    def allocate_capital(self) -> PortfolioState:
        """
        Allocate capital across strategies using risk parity
        Returns current portfolio state
        """
        weights = self.calculate_risk_parity_weights()
        
        if not weights:
            return PortfolioState(
                total_capital=self.total_capital,
                allocated_capital=0.0,
                available_capital=self.total_capital,
                portfolio_volatility=0.0,
                portfolio_var=0.0,
                strategy_weights={},
                risk_contributions={},
                timestamp_ns=time.time_ns()
            )
        
        # Calculate allocated capital per strategy
        allocated_total = 0.0
        strategy_weights = {}
        risk_contributions = {}
        
        for strategy_id, weight in weights.items():
            allocated = self.total_capital * weight
            
            # Apply position size limits
            max_position = self._calculate_max_position(strategy_id)
            allocated = min(allocated, max_position)
            
            strategy_weights[strategy_id] = allocated / self.total_capital
            allocated_total += allocated
            
            # Update or create allocation
            vol = self.strategy_volatilities.get(strategy_id, 0.0)
            sharpe = self.strategy_sharpes.get(strategy_id, 0.0)
            
            self.allocations[strategy_id] = StrategyAllocation(
                strategy_id=strategy_id,
                allocated_capital=allocated,
                weight=weight,
                risk_contribution=1.0 / len(weights),  # Approximate for risk parity
                volatility=vol,
                sharpe_ratio=sharpe,
                max_position_size=max_position,
                timestamp_ns=time.time_ns()
            )
        
        # Calculate portfolio-level metrics
        portfolio_vol = self._calculate_portfolio_volatility(strategy_weights)
        portfolio_var = self._calculate_portfolio_var(strategy_weights)
        
        return PortfolioState(
            total_capital=self.total_capital,
            allocated_capital=allocated_total,
            available_capital=self.total_capital - allocated_total,
            portfolio_volatility=portfolio_vol,
            portfolio_var=portfolio_var,
            strategy_weights=strategy_weights,
            risk_contributions=risk_contributions,
            timestamp_ns=time.time_ns()
        )
    
    def _calculate_max_position(self, strategy_id: str) -> float:
        """Calculate maximum position size for a strategy"""
        base_max = self.total_capital * self.max_single_strategy_weight
        
        # Reduce position for high volatility strategies
        vol = self.strategy_volatilities.get(strategy_id, 0.1)
        vol_adjustment = self.target_portfolio_volatility / max(vol, 0.01)
        vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
        
        return base_max * vol_adjustment
    
    def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate expected portfolio volatility"""
        if not weights:
            return 0.0
        
        strategy_ids = list(weights.keys())
        w = np.array([weights[s] for s in strategy_ids])
        vols = np.array([self.strategy_volatilities.get(s, 0.1) for s in strategy_ids])
        corr = self.calculate_correlation_matrix()
        
        cov = np.outer(vols, vols) * corr
        portfolio_var = w @ cov @ w
        
        return np.sqrt(portfolio_var)
    
    def _calculate_portfolio_var(self, weights: Dict[str, float], confidence: float = 0.99) -> float:
        """Calculate portfolio Value at Risk"""
        portfolio_vol = self._calculate_portfolio_volatility(weights)
        # Parametric VaR assuming normal distribution
        z_score = 2.33 for 99% confidence
        z_score = 2.33 if confidence == 0.99 else 1.645
        return portfolio_vol * z_score * self.total_capital
    
    def rebalance(self) -> Dict[str, float]:
        """
        Rebalance portfolio to target risk parity weights
        Returns dictionary of target allocations
        """
        state = self.allocate_capital()
        return {
            sid: alloc.allocated_capital 
            for sid, alloc in self.allocations.items()
        }
    
    def get_allocation(self, strategy_id: str) -> Optional[StrategyAllocation]:
        """Get current allocation for a specific strategy"""
        return self.allocations.get(strategy_id)
    
    def get_all_allocations(self) -> Dict[str, StrategyAllocation]:
        """Get all current allocations"""
        return self.allocations.copy()


# Example usage
def example_usage():
    """Example of how to use the risk parity allocator"""
    allocator = RiskParityAllocator(
        total_capital=1_000_000,
        max_single_strategy_weight=0.3,
        target_portfolio_volatility=0.12
    )
    
    # Register strategies
    allocator.register_strategy("market_making")
    allocator.register_strategy("stat_arb")
    allocator.register_strategy("microstructure")
    
    # Add some historical returns (simulated)
    np.random.seed(42)
    allocator.update_strategy_returns("market_making", np.random.normal(0.0001, 0.005, 100).tolist())
    allocator.update_strategy_returns("stat_arb", np.random.normal(0.0002, 0.008, 100).tolist())
    allocator.update_strategy_returns("microstructure", np.random.normal(0.00015, 0.006, 100).tolist())
    
    # Allocate capital
    state = allocator.allocate_capital()
    
    print(f"Total Capital: ${state.total_capital:,.2f}")
    print(f"Allocated Capital: ${state.allocated_capital:,.2f}")
    print(f"Portfolio Volatility: {state.portfolio_volatility:.2%}")
    print(f"Portfolio VaR (99%): ${state.portfolio_var:,.2f}")
    print("\nStrategy Weights:")
    for strategy_id, weight in state.strategy_weights.items():
        print(f"  {strategy_id}: {weight:.2%}")
    
    # Rebalance
    targets = allocator.rebalance()
    print("\nTarget Allocations:")
    for strategy_id, amount in targets.items():
        print(f"  {strategy_id}: ${amount:,.2f}")


if __name__ == "__main__":
    example_usage()
