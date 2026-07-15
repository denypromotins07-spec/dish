"""
Delta-neutral gamma scalping strategy.
Dynamically hedges options portfolios by buying low and selling high
in the underlying spot/perp market to harvest gamma while remaining delta-neutral.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HedgeAction(Enum):
    BUY_SPOT = "buy_spot"
    SELL_SPOT = "sell_spot"
    BUY_PERP = "buy_perp"
    SELL_PERP = "sell_perp"
    HOLD = "hold"


@dataclass
class GammaPosition:
    """Options position with gamma exposure"""
    symbol: str
    option_type: str  # call or put
    strike: float
    expiry: int
    quantity: int
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class HedgeDecision:
    """Hedging decision output"""
    action: HedgeAction
    quantity: float
    target_delta: float
    current_delta: float
    hedge_ratio: float
    expected_gamma_pnl: float
    transaction_cost_estimate: float


class GammaScalpingStrategy:
    """
    Delta-neutral gamma scalping strategy.
    
    Maintains delta neutrality while harvesting gamma through:
    - Dynamic delta hedging at price levels
    - Rebalancing based on gamma thresholds
    - Optimizing hedge frequency vs transaction costs
    """

    def __init__(
        self,
        gamma_threshold: float = 0.05,
        delta_tolerance: float = 0.10,
        hedge_band_pct: float = 0.02,
        max_hedge_frequency_sec: int = 60,
        taker_fee_bps: float = 4.0,
    ):
        self.gamma_threshold = gamma_threshold
        self.delta_tolerance = delta_tolerance
        self.hedge_band_pct = hedge_band_pct
        self.max_hedge_frequency_sec = max_hedge_frequency_sec
        self.taker_fee_bps = taker_fee_bps
        
        self._positions: List[GammaPosition] = []
        self._underlying_position: float = 0.0  # Positive = long
        self._last_hedge_time: float = 0.0
        self._cumulative_gamma_pnl: float = 0.0
        self._hedge_count: int = 0
        
    def add_options_position(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiry: int,
        quantity: int,
        greeks: Dict[str, float],
    ) -> None:
        """Add an options position to the portfolio"""
        pos = GammaPosition(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            quantity=quantity,
            delta=greeks.get('delta', 0.0),
            gamma=greeks.get('gamma', 0.0),
            theta=greeks.get('theta', 0.0),
            vega=greeks.get('vega', 0.0),
        )
        self._positions.append(pos)
        logger.info(f"Added options position: {pos}")
        
    def remove_options_position(self, index: int) -> None:
        """Remove an options position"""
        if 0 <= index < len(self._positions):
            self._positions.pop(index)
    
    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate aggregate portfolio Greeks"""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        
        for pos in self._positions:
            qty = pos.quantity
            total_delta += pos.delta * qty
            total_gamma += pos.gamma * qty
            total_theta += pos.theta * qty
            total_vega += pos.vega * qty
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
        }
    
    def calculate_hedge_decision(
        self,
        spot_price: float,
        portfolio_greeks: Optional[Dict[str, float]] = None,
    ) -> HedgeDecision:
        """
        Calculate optimal hedge decision to maintain delta neutrality.
        
        Uses gamma-weighted rebalancing to optimize trade-off between
        gamma capture and transaction costs.
        """
        if portfolio_greeks is None:
            portfolio_greeks = self.get_portfolio_greeks()
        
        options_delta = portfolio_greeks['delta']
        portfolio_gamma = portfolio_greeks['gamma']
        
        # Current total delta (options + underlying)
        current_total_delta = options_delta + self._underlying_position
        
        # Target: delta neutral
        target_delta = 0.0
        
        # Calculate required hedge
        hedge_needed = target_delta - current_total_delta
        
        # Apply hedge band to avoid over-trading
        hedge_band = spot_price * self.hedge_band_pct * abs(portfolio_gamma)
        
        if abs(hedge_needed) < hedge_band:
            return HedgeDecision(
                action=HedgeAction.HOLD,
                quantity=0.0,
                target_delta=target_delta,
                current_delta=current_total_delta,
                hedge_ratio=0.0,
                expected_gamma_pnl=0.0,
                transaction_cost_estimate=0.0,
            )
        
        # Determine action
        if hedge_needed > 0:
            action = HedgeAction.BUY_SPOT
        else:
            action = HedgeAction.SELL_SPOT
        
        hedge_quantity = abs(hedge_needed)
        
        # Estimate transaction cost
        notional = hedge_quantity * spot_price
        tx_cost = notional * (self.taker_fee_bps / 10000)
        
        # Estimate gamma PnL from rebalancing
        # Gamma PnL ≈ 0.5 * gamma * (price_move)^2
        expected_move = spot_price * 0.02  # Assume 2% move
        expected_gamma_pnl = 0.5 * portfolio_gamma * (expected_move ** 2)
        
        # Hedge ratio (underlying / options delta)
        hedge_ratio = abs(self._underlying_position) / max(abs(options_delta), 0.01)
        
        return HedgeDecision(
            action=action,
            quantity=hedge_quantity,
            target_delta=target_delta,
            current_delta=current_total_delta,
            hedge_ratio=hedge_ratio,
            expected_gamma_pnl=expected_gamma_pnl,
            transaction_cost_estimate=tx_cost,
        )
    
    def execute_hedge(
        self,
        decision: HedgeDecision,
        spot_price: float,
    ) -> bool:
        """Execute a hedge decision"""
        if decision.action == HedgeAction.HOLD:
            return True
        
        # Update underlying position
        if decision.action in [HedgeAction.BUY_SPOT, HedgeAction.BUY_PERP]:
            self._underlying_position += decision.quantity
        else:
            self._underlying_position -= decision.quantity
        
        self._hedge_count += 1
        
        logger.info(
            f"Executed hedge: {decision.action.value} {decision.quantity:.4f} @ {spot_price:.2f}"
        )
        
        return True
    
    def update_underlying_price(self, new_price: float, old_price: float) -> float:
        """
        Update underlying price and calculate gamma PnL from price movement.
        
        Returns the gamma PnL from the price move.
        """
        price_change = new_price - old_price
        
        # Gamma PnL from the move
        portfolio_greeks = self.get_portfolio_greeks()
        gamma_pnl = 0.5 * portfolio_greeks['gamma'] * (price_change ** 2)
        
        # Delta PnL
        delta_pnl = (portfolio_greeks['delta'] + self._underlying_position) * price_change
        
        total_pnl = gamma_pnl + delta_pnl
        self._cumulative_gamma_pnl += gamma_pnl
        
        logger.debug(
            f"Price update: {old_price:.2f} -> {new_price:.2f}, "
            f"Gamma PnL: {gamma_pnl:.2f}, Total PnL: {total_pnl:.2f}"
        )
        
        return gamma_pnl
    
    def check_rebalance_trigger(
        self,
        spot_price: float,
        time_since_last_hedge: float,
    ) -> bool:
        """Check if rebalancing is triggered"""
        # Time-based trigger
        if time_since_last_hedge >= self.max_hedge_frequency_sec:
            return True
        
        # Delta drift trigger
        portfolio_greeks = self.get_portfolio_greeks()
        total_delta = portfolio_greeks['delta'] + self._underlying_position
        
        if abs(total_delta) > self.delta_tolerance:
            return True
        
        # Gamma threshold trigger
        if abs(portfolio_greeks['gamma']) > self.gamma_threshold:
            return True
        
        return False
    
    def get_optimal_hedge_levels(
        self,
        spot_price: float,
        n_levels: int = 5,
    ) -> List[Tuple[float, float]]:
        """
        Calculate optimal hedge entry levels based on gamma profile.
        
        Returns list of (price_level, hedge_quantity) tuples.
        """
        portfolio_greeks = self.get_portfolio_greeks()
        gamma = portfolio_greeks['gamma']
        delta = portfolio_greeks['delta']
        
        if gamma == 0:
            return []
        
        # Calculate price levels where delta changes significantly
        levels = []
        
        for i in range(-n_levels // 2, n_levels // 2 + 1):
            if i == 0:
                continue
            
            # Price level based on gamma
            price_offset = (i * 0.01) / gamma * spot_price
            price_level = spot_price + price_offset
            
            # Hedge quantity needed at this level
            delta_change = gamma * price_offset
            hedge_qty = abs(delta + delta_change)
            
            levels.append((price_level, hedge_qty))
        
        return sorted(levels, key=lambda x: x[0])
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        portfolio_greeks = self.get_portfolio_greeks()
        
        return {
            'portfolio_delta': portfolio_greeks['delta'],
            'portfolio_gamma': portfolio_greeks['gamma'],
            'portfolio_theta': portfolio_greeks['theta'],
            'portfolio_vega': portfolio_greeks['vega'],
            'underlying_position': self._underlying_position,
            'total_delta': portfolio_greeks['delta'] + self._underlying_position,
            'cumulative_gamma_pnl': self._cumulative_gamma_pnl,
            'hedge_count': self._hedge_count,
            'is_delta_neutral': abs(portfolio_greeks['delta'] + self._underlying_position) < self.delta_tolerance,
        }
    
    def reset(self) -> None:
        """Reset strategy state"""
        self._positions.clear()
        self._underlying_position = 0.0
        self._cumulative_gamma_pnl = 0.0
        self._hedge_count = 0
