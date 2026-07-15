"""
Friction-Aware Loss Functions for Deep Hedging.
Explicitly penalizes transaction costs, spread crossing, and market impact.
Designed for capital efficiency under strict memory constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class FrictionAwareLoss(nn.Module):
    """
    Composite loss function for deep hedging that accounts for:
    1. Transaction costs (explicit fees)
    2. Bid-ask spread crossing penalties
    3. Market impact (slippage based on order size)
    4. Position regularization (smooth hedging)
    """
    
    def __init__(
        self,
        tx_cost_rate: float = 0.0005,      # 5 bps transaction cost
        spread_penalty: float = 0.001,     # Penalty for crossing spread
        impact_coefficient: float = 0.0001, # Market impact coefficient
        position_reg_lambda: float = 0.01,  # Position regularization
        turnover_penalty: float = 0.005,    # Penalize excessive trading
    ):
        super().__init__()
        self.tx_cost_rate = tx_cost_rate
        self.spread_penalty = spread_penalty
        self.impact_coefficient = impact_coefficient
        self.position_reg_lambda = position_reg_lambda
        self.turnover_penalty = turnover_penalty
        
    def forward(
        self,
        actions: torch.Tensor,           # Hedging actions (delta changes)
        prev_positions: torch.Tensor,    # Previous positions
        spreads: torch.Tensor,           # Current bid-ask spreads
        volumes: torch.Tensor,           # Trading volumes (for impact)
        pnl: torch.Tensor,               # Realized PnL
        target_pnl: torch.Tensor,        # Target PnL (usually 0 for hedging)
    ) -> torch.Tensor:
        """
        Calculate friction-aware hedging loss.
        
        Args:
            actions: Delta changes proposed by the model
            prev_positions: Portfolio positions before action
            spreads: Current bid-ask spreads per asset
            volumes: Market volumes for impact calculation
            pnl: Realized PnL from hedging
            target_pnl: Target PnL (typically zero for pure hedging)
            
        Returns:
            Total loss tensor
        """
        # 1. Transaction Cost Loss
        tx_costs = torch.abs(actions) * self.tx_cost_rate
        tx_loss = tx_costs.mean()
        
        # 2. Spread Crossing Penalty
        # Penalize when action direction crosses the spread
        action_sign = torch.sign(actions)
        position_change = actions - prev_positions
        cross_spread = (torch.sign(position_change) != torch.sign(prev_positions + actions)).float()
        spread_loss = (cross_spread * spreads * self.spread_penalty).mean()
        
        # 3. Market Impact Loss
        # Larger orders relative to volume cause more slippage
        relative_size = torch.abs(actions) / (volumes + 1e-8)
        impact_loss = (self.impact_coefficient * relative_size.pow(2)).mean()
        
        # 4. PnL Tracking Error
        tracking_error = (pnl - target_pnl).pow(2).mean()
        
        # 5. Position Regularization (smooth hedging)
        position_reg = (prev_positions + actions).pow(2).mean() * self.position_reg_lambda
        
        # 6. Turnover Penalty (discourage excessive trading)
        turnover_loss = torch.abs(actions).mean() * self.turnover_penalty
        
        # Total loss
        total_loss = (
            tracking_error +
            tx_loss +
            spread_loss +
            impact_loss +
            position_reg +
            turnover_loss
        )
        
        return total_loss


class SharpnessAwareHedgingLoss(nn.Module):
    """
    Loss function that incorporates sharpness awareness for robust hedging.
    Penalizes strategies that are sensitive to small perturbations in inputs.
    """
    
    def __init__(
        self,
        base_loss: FrictionAwareLoss,
        perturbation_scale: float = 0.01,
        sharpness_weight: float = 0.1,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.perturbation_scale = perturbation_scale
        self.sharpness_weight = sharpness_weight
        
    def forward(
        self,
        model: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate base loss plus sharpness penalty.
        """
        # Base loss
        base_loss_val = self.base_loss(actions, **kwargs)
        
        # Sharpness estimation via input perturbation
        perturbation = torch.randn_like(states) * self.perturbation_scale
        perturbed_states = states + perturbation
        
        with torch.enable_grad():
            perturbed_actions, _ = model(perturbed_states)
            perturbed_loss = self.base_loss(perturbed_actions, **kwargs)
        
        # Sharpness is the difference in loss under perturbation
        sharpness = (perturbed_loss - base_loss_val).abs().mean()
        
        return base_loss_val + self.sharpness_weight * sharpness


class AdaptiveFrictionScheduler:
    """
    Dynamically adjusts friction parameters based on market regime.
    Increases penalties during high volatility or low liquidity periods.
    """
    
    def __init__(
        self,
        base_loss: FrictionAwareLoss,
        volatility_threshold: float = 0.02,
        liquidity_threshold: float = 0.5,
    ):
        self.base_loss = base_loss
        self.volatility_threshold = volatility_threshold
        self.liquidity_threshold = liquidity_threshold
        
    def adjust_for_regime(
        self,
        volatility: float,
        liquidity_ratio: float,
    ) -> None:
        """
        Adjust friction parameters based on current market regime.
        """
        # Increase penalties in high volatility
        vol_multiplier = 1.0 + max(0, (volatility - self.volatility_threshold) * 10)
        
        # Increase penalties in low liquidity
        liq_multiplier = 1.0 + max(0, (self.liquidity_threshold - liquidity_ratio) * 2)
        
        combined_multiplier = vol_multiplier * liq_multiplier
        
        # Update loss parameters
        self.base_loss.tx_cost_rate *= combined_multiplier
        self.base_loss.spread_penalty *= combined_multiplier
        self.base_loss.impact_coefficient *= combined_multiplier
        
    def reset(self) -> None:
        """Reset to base parameters."""
        self.base_loss.tx_cost_rate = 0.0005
        self.base_loss.spread_penalty = 0.001
        self.base_loss.impact_coefficient = 0.0001


class EntropicRiskLoss(nn.Module):
    """
    Entropic risk measure for hedging loss.
    More conservative than MSE, penalizes tail risks heavily.
    """
    
    def __init__(self, risk_aversion: float = 5.0):
        super().__init__()
        self.risk_aversion = risk_aversion
        
    def forward(
        self,
        hedging_errors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate entropic risk of hedging errors.
        
        Entropic Risk = (1/risk_aversion) * log(E[exp(-risk_aversion * errors)])
        """
        exp_term = torch.exp(-self.risk_aversion * hedging_errors)
        expectation = exp_term.mean()
        
        # Clamp to prevent log(0)
        expectation = torch.clamp(expectation, min=1e-8)
        
        return (1.0 / self.risk_aversion) * torch.log(expectation)


if __name__ == "__main__":
    # Example usage
    batch_size = 64
    
    # Dummy data
    actions = torch.randn(batch_size, 5) * 0.1
    prev_positions = torch.randn(batch_size, 5)
    spreads = torch.ones(batch_size, 5) * 0.001
    volumes = torch.rand(batch_size, 5) * 1000 + 100
    pnl = torch.randn(batch_size, 1)
    target_pnl = torch.zeros(batch_size, 1)
    
    # Initialize loss
    loss_fn = FrictionAwareLoss()
    
    loss = loss_fn(
        actions=actions,
        prev_positions=prev_positions,
        spreads=spreads,
        volumes=volumes,
        pnl=pnl,
        target_pnl=target_pnl,
    )
    
    print(f"Friction-aware loss: {loss.item():.6f}")
    
    # Test adaptive scheduler
    scheduler = AdaptiveFrictionScheduler(loss_fn)
    scheduler.adjust_for_regime(volatility=0.05, liquidity_ratio=0.3)
    print("Adjusted for high volatility, low liquidity regime")
    
    loss_adjusted = loss_fn(
        actions=actions,
        prev_positions=prev_positions,
        spreads=spreads,
        volumes=volumes,
        pnl=pnl,
        target_pnl=target_pnl,
    )
    print(f"Adjusted loss: {loss_adjusted.item():.6f}")
