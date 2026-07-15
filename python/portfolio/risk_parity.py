"""
Newton-Raphson based Risk Parity allocator ensuring each asset and sub-strategy
contributes equally to the overall portfolio variance.

Memory-efficient implementation bounded for <14GB total system RAM.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass(slots=True)
class RiskParityResult:
    """Result of risk parity optimization."""
    weights: np.ndarray
    risk_contributions: np.ndarray
    total_risk: float
    converged: bool
    iterations: int


def risk_budget_objective(
    weights: np.ndarray,
    covariance: np.ndarray,
    target_risk_budget: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate risk parity objective function value, gradient, and Hessian.
    
    Args:
        weights: Portfolio weights
        covariance: Covariance matrix
        target_risk_budget: Target risk contribution for each asset (default: equal)
    
    Returns:
        objective_value, gradient, hessian_diagonal
    """
    n = len(weights)
    
    if target_risk_budget is None:
        target_risk_budget = np.ones(n) / n
    
    # Portfolio variance and volatility
    port_var = weights @ covariance @ weights
    port_vol = np.sqrt(port_var)
    
    if port_vol < 1e-10:
        return 0.0, np.zeros(n), np.ones(n)
    
    # Marginal risk contributions
    marginal_risk = covariance @ weights / port_vol
    
    # Actual risk contributions
    risk_contrib = weights * marginal_risk
    
    # Relative risk contributions (should equal target budget)
    rel_risk_contrib = risk_contrib / port_vol
    
    # Objective: sum of squared deviations from target
    diff = rel_risk_contrib - target_risk_budget
    objective = np.sum(diff ** 2)
    
    # Gradient calculation
    # d(objective)/d(w_i) = 2 * sum_j (diff_j * d(rel_risk_j)/d(w_i))
    gradient = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            # Derivative of relative risk contribution
            d_rc_dw = (covariance[i, j] * port_vol - marginal_risk[j] * covariance[i, :] @ weights / port_vol) / port_var
            d_rel_rc_dw = (d_rc_dw * port_vol - risk_contrib[j] * marginal_risk[i]) / (port_vol ** 2)
            gradient[i] += 2 * diff[j] * d_rel_rc_dw
    
    # Hessian diagonal approximation (for Newton step)
    hessian_diag = np.zeros(n)
    for i in range(n):
        # Second derivative approximation
        hessian_diag[i] = 2 * covariance[i, i] / port_var
    
    return objective, gradient, hessian_diag


def newton_raphson_risk_parity(
    covariance: np.ndarray,
    target_risk_budget: Optional[np.ndarray] = None,
    initial_weights: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    min_weight: float = 1e-6,
    max_weight: float = 1.0
) -> RiskParityResult:
    """
    Solve risk parity using Newton-Raphson method with line search.
    
    Args:
        covariance: Asset covariance matrix
        target_risk_budget: Target risk contribution (default: equal)
        initial_weights: Starting weights (default: equal)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        min_weight: Minimum weight bound
        max_weight: Maximum weight bound
    
    Returns:
        RiskParityResult with optimized weights
    """
    n = covariance.shape[0]
    
    if target_risk_budget is None:
        target_risk_budget = np.ones(n) / n
    
    if initial_weights is None:
        weights = np.ones(n) / n
    else:
        weights = initial_weights.copy()
    
    # Ensure weights are valid
    weights = np.clip(weights, min_weight, max_weight)
    weights /= np.sum(weights)
    
    converged = False
    iterations = 0
    
    for iteration in range(max_iterations):
        iterations = iteration + 1
        
        # Evaluate objective, gradient, Hessian
        obj, grad, hess_diag = risk_budget_objective(weights, covariance, target_risk_budget)
        
        # Check convergence
        if obj < tolerance:
            converged = True
            break
        
        # Newton step: w_new = w - H^{-1} * g
        # Use diagonal Hessian approximation for efficiency
        hess_inv_diag = np.where(hess_diag > 1e-10, 1.0 / hess_diag, 1.0)
        delta = -grad * hess_inv_diag
        
        # Line search with backtracking
        step_size = 1.0
        for _ in range(20):
            new_weights = weights + step_size * delta
            
            # Project onto simplex
            new_weights = project_to_simplex(new_weights, min_weight)
            
            # Check if improvement
            new_obj, _, _ = risk_budget_objective(new_weights, covariance, target_risk_budget)
            
            if new_obj < obj:
                break
            
            step_size *= 0.5
        else:
            # Line search failed, take small gradient step
            new_weights = weights - 0.01 * grad
            new_weights = project_to_simplex(new_weights, min_weight)
        
        weights = new_weights
        
        # Check weight change convergence
        if np.max(np.abs(delta * step_size)) < tolerance:
            converged = True
            break
    
    # Calculate final risk contributions
    port_var = weights @ covariance @ weights
    port_vol = np.sqrt(port_var)
    marginal_risk = covariance @ weights / port_vol
    risk_contrib = weights * marginal_risk
    
    return RiskParityResult(
        weights=weights,
        risk_contributions=risk_contrib,
        total_risk=port_vol,
        converged=converged,
        iterations=iterations
    )


def project_to_simplex(x: np.ndarray, min_val: float = 0.0) -> np.ndarray:
    """
    Project vector onto the probability simplex.
    
    Args:
        x: Input vector
        min_val: Minimum value for each element
    
    Returns:
        Projected vector that sums to 1
    """
    n = len(x)
    
    # Clip to minimum
    x = np.maximum(x, min_val)
    
    # If already on simplex, return
    sum_x = np.sum(x)
    if np.abs(sum_x - 1.0) < 1e-10:
        return x
    
    # Sort for efficient projection
    u = np.sort(x)[::-1]
    
    # Find threshold
    cssv = np.cumsum(u)
    ind = np.arange(n) + 1
    cond = u - (cssv - 1.0) / ind > 0
    rho = ind[cond][-1]
    theta = (cssv[rho - 1] - 1.0) / rho
    
    # Project
    w = np.maximum(x - theta, min_val)
    
    # Renormalize
    w_sum = np.sum(w)
    if w_sum > 1e-10:
        w /= w_sum
    else:
        w = np.ones(n) / n
    
    return w


def cyclic_coordinate_descent_risk_parity(
    covariance: np.ndarray,
    target_risk_budget: Optional[np.ndarray] = None,
    max_iterations: int = 500,
    tolerance: float = 1e-8
) -> RiskParityResult:
    """
    Alternative risk parity solver using cyclic coordinate descent.
    Often more stable than Newton-Raphson for ill-conditioned matrices.
    """
    n = covariance.shape[0]
    
    if target_risk_budget is None:
        target_risk_budget = np.ones(n) / n
    
    weights = np.ones(n) / n
    
    converged = False
    
    for iteration in range(max_iterations):
        weights_old = weights.copy()
        
        for i in range(n):
            # Update weight i while keeping others fixed
            # Closed-form update for single coordinate
            
            # Current portfolio variance without asset i
            w_excl = weights.copy()
            w_excl[i] = 0
            var_excl = w_excl @ covariance @ w_excl
            
            # Optimal weight for asset i
            # Derived from setting marginal risk contribution proportional to target
            cov_i = covariance[i, :]
            
            # Solve quadratic for optimal weight
            a = covariance[i, i]
            b = 2 * (cov_i @ w_excl)
            c = var_excl
            
            if a < 1e-10:
                continue
            
            # Target: w_i * (a*w_i + b/2) / sqrt(total_var) = target_budget * sqrt(total_var)
            # Simplified iterative update
            total_cov = cov_i @ weights
            if total_cov > 1e-10:
                new_w_i = target_risk_budget[i] * (weights @ covariance @ weights) / total_cov
                new_w_i = np.clip(new_w_i, 1e-6, 1.0)
                weights[i] = new_w_i
        
        # Normalize
        weights /= np.sum(weights)
        
        # Check convergence
        if np.max(np.abs(weights - weights_old)) < tolerance:
            converged = True
            break
    
    # Calculate final metrics
    port_var = weights @ covariance @ weights
    port_vol = np.sqrt(port_var)
    marginal_risk = covariance @ weights / port_vol
    risk_contrib = weights * marginal_risk
    
    return RiskParityResult(
        weights=weights,
        risk_contributions=risk_contrib,
        total_risk=port_vol,
        converged=converged,
        iterations=iteration + 1
    )


class RiskParityAllocator:
    """
    High-level risk parity allocator with caching and incremental updates.
    """
    
    __slots__ = ('n_assets', 'covariance', 'target_budget', 'last_weights', 'cache_valid')
    
    def __init__(self, n_assets: int, target_budget: Optional[np.ndarray] = None):
        self.n_assets = n_assets
        self.covariance: Optional[np.ndarray] = None
        self.target_budget = target_budget if target_budget is not None else np.ones(n_assets) / n_assets
        self.last_weights: Optional[np.ndarray] = None
        self.cache_valid = False
    
    def update_covariance(self, covariance: np.ndarray) -> None:
        """Update covariance matrix and invalidate cache."""
        assert covariance.shape == (self.n_assets, self.n_assets)
        self.covariance = covariance
        self.cache_valid = False
    
    def compute(self, use_newton: bool = True) -> RiskParityResult:
        """Compute risk parity allocation."""
        if self.covariance is None:
            raise ValueError("Covariance matrix not set")
        
        if self.cache_valid and self.last_weights is not None:
            # Return cached result
            port_var = self.last_weights @ self.covariance @ self.last_weights
            port_vol = np.sqrt(port_var)
            marginal_risk = self.covariance @ self.last_weights / port_vol
            risk_contrib = self.last_weights * marginal_risk
            
            return RiskParityResult(
                weights=self.last_weights,
                risk_contributions=risk_contrib,
                total_risk=port_vol,
                converged=True,
                iterations=0
            )
        
        # Compute fresh allocation
        if use_newton:
            result = newton_raphson_risk_parity(
                self.covariance,
                self.target_budget,
                initial_weights=self.last_weights
            )
        else:
            result = cyclic_coordinate_descent_risk_parity(
                self.covariance,
                self.target_budget
            )
        
        # Cache result
        if result.converged:
            self.last_weights = result.weights.copy()
            self.cache_valid = True
        
        return result
    
    def get_risk_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get risk decomposition (weights and risk contributions)."""
        result = self.compute()
        return result.weights, result.risk_contributions


if __name__ == '__main__':
    # Test risk parity allocation
    np.random.seed(42)
    n = 5
    
    # Generate random positive definite covariance
    A = np.random.randn(n, n)
    cov = A @ A.T / n + np.eye(n) * 0.1
    
    print("Testing Newton-Raphson Risk Parity:")
    result = newton_raphson_risk_parity(cov)
    print(f"  Converged: {result.converged} in {result.iterations} iterations")
    print(f"  Weights: {result.weights}")
    print(f"  Risk contributions: {result.risk_contributions}")
    print(f"  Total risk: {result.total_risk:.4f}")
    
    print("\nTesting Cyclic Coordinate Descent:")
    result2 = cyclic_coordinate_descent_risk_parity(cov)
    print(f"  Converged: {result2.converged} in {result2.iterations} iterations")
    print(f"  Weights: {result2.weights}")
    print(f"  Risk contributions: {result2.risk_contributions}")
    
    # Verify risk parity property
    print("\nRisk parity check (should all be ~0.2 for equal budget):")
    rel_contrib = result.risk_contributions / result.total_risk
    print(f"  Relative contributions: {rel_contrib}")
