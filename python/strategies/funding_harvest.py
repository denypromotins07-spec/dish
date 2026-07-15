"""
Automated funding rate harvesting strategy.
Opens delta-neutral spot-perp positions to collect funding payouts,
dynamically adjusting position sizes based on Kelly Criterion and funding predictions.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionState(Enum):
    OPENING = "opening"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class FundingArbPosition:
    """A single funding arbitrage position"""
    symbol: str
    # Spot leg
    spot_side: str  # long or short
    spot_quantity: float
    spot_entry_price: float
    # Perp leg
    perp_side: str  # opposite of spot
    perp_quantity: float
    perp_entry_price: float
    # Metrics
    expected_funding_rate: float
    annualized_return: float
    state: PositionState
    cumulative_funding_collected: float
    realized_pnl: float


@dataclass
class FundingSignal:
    """Funding rate arbitrage signal"""
    symbol: str
    predicted_funding_rate: float
    annualized_rate: float
    confidence: float
    recommended_size: float
    direction: int  # +1 = long spot/short perp, -1 = short spot/long perp


class FundingHarvestStrategy:
    """
    Automated funding rate harvesting strategy.
    
    Exploits positive/negative funding rates by:
    - Opening delta-neutral spot-perp pairs
    - Collecting funding payments
    - Managing position exits based on rate changes
    
    Uses Kelly Criterion for optimal position sizing.
    """

    def __init__(
        self,
        min_annualized_rate: float = 10.0,  # Minimum 10% annualized
        max_position_notional: float = 500_000.0,
        kelly_fraction: float = 0.25,
        funding_threshold_bps: float = 50.0,
        exit_threshold_bps: float = 20.0,
        taker_fee_bps: float = 4.0,
    ):
        self.min_annualized_rate = min_annualized_rate
        self.max_position_notional = max_position_notional
        self.kelly_fraction = kelly_fraction
        self.funding_threshold_bps = funding_threshold_bps
        self.exit_threshold_bps = exit_threshold_bps
        self.taker_fee_bps = taker_fee_bps
        
        self._positions: List[FundingArbPosition] = []
        self._cumulative_funding: float = 0.0
        self._total_realized_pnl: float = 0.0
        self._trade_count: int = 0
        
    def analyze_funding_opportunity(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        predicted_funding_rate: float,
        prediction_confidence: float,
    ) -> Optional[FundingSignal]:
        """
        Analyze funding rate arbitrage opportunity.
        
        Args:
            symbol: Trading pair symbol
            spot_price: Current spot price
            perp_price: Current perpetual price
            predicted_funding_rate: Predicted next funding rate (as decimal)
            prediction_confidence: Confidence in prediction (0-1)
        
        Returns:
            FundingSignal if opportunity exists, None otherwise
        """
        # Calculate basis
        basis = (perp_price - spot_price) / spot_price
        
        # Annualized funding rate (3 payments per day, 365 days)
        annualized_rate = predicted_funding_rate * 3 * 365 * 100  # As percentage
        
        # Check minimum threshold
        if abs(annualized_rate) < self.min_annualized_rate:
            return None
        
        # Determine direction
        # Positive funding = longs pay shorts → short perp, long spot
        # Negative funding = shorts pay longs → long perp, short spot
        if predicted_funding_rate > 0:
            direction = 1  # Long spot, short perp
        else:
            direction = -1  # Short spot, long perp
        
        # Calculate optimal size using Kelly
        edge = abs(annualized_rate) / 100  # Normalize
        win_prob = 0.5 + prediction_confidence * 0.3  # Map confidence to win prob
        
        if edge <= 0:
            return None
            
        kelly = (win_prob * edge - (1 - win_prob)) / edge
        kelly = max(0, min(kelly, self.kelly_fraction))
        
        recommended_size = kelly * self.max_position_notional
        
        # Account for transaction costs (entry + exit)
        tx_cost_bps = self.taker_fee_bps * 2
        net_annualized = annualized_rate - tx_cost_bps * 3  # Rough estimate
        
        if net_annualized < self.min_annualized_rate:
            return None
        
        return FundingSignal(
            symbol=symbol,
            predicted_funding_rate=predicted_funding_rate,
            annualized_rate=net_annualized,
            confidence=prediction_confidence,
            recommended_size=recommended_size,
            direction=direction,
        )
    
    def open_position(
        self,
        signal: FundingSignal,
        spot_price: float,
        perp_price: float,
    ) -> Optional[FundingArbPosition]:
        """Open a funding arbitrage position"""
        if signal.recommended_size <= 0:
            return None
        
        # Calculate quantities
        spot_qty = signal.recommended_size / spot_price
        
        if signal.direction > 0:
            # Long spot, short perp
            spot_side = "long"
            perp_side = "short"
        else:
            # Short spot, long perp
            spot_side = "short"
            perp_side = "long"
        
        position = FundingArbPosition(
            symbol=signal.symbol,
            spot_side=spot_side,
            spot_quantity=spot_qty,
            spot_entry_price=spot_price,
            perp_side=perp_side,
            perp_quantity=spot_qty,  # Delta neutral
            perp_entry_price=perp_price,
            expected_funding_rate=signal.predicted_funding_rate,
            annualized_return=signal.annualized_rate,
            state=PositionState.ACTIVE,
            cumulative_funding_collected=0.0,
            realized_pnl=0.0,
        )
        
        self._positions.append(position)
        self._trade_count += 1
        
        logger.info(
            f"Opened funding arb: {signal.symbol}, "
            f"size={signal.recommended_size:.2f}, "
            f"expected_annual={signal.annualized_rate:.2f}%"
        )
        
        return position
    
    def collect_funding(
        self,
        position: FundingArbPosition,
        actual_funding_rate: float,
    ) -> float:
        """
        Process funding payment collection.
        
        Returns the funding amount collected (positive) or paid (negative).
        """
        # Funding is paid/received on perp notional
        perp_notional = position.perp_quantity * position.perp_entry_price
        
        # Direction determines if we receive or pay
        if position.perp_side == "short":
            # Short perp receives positive funding
            funding_amount = perp_notional * actual_funding_rate
        else:
            # Long perp pays positive funding (receives negative)
            funding_amount = -perp_notional * actual_funding_rate
        
        position.cumulative_funding_collected += funding_amount
        self._cumulative_funding += funding_amount
        
        logger.debug(
            f"Funding collected on {position.symbol}: {funding_amount:.2f} "
            f"(rate={actual_funding_rate})"
        )
        
        return funding_amount
    
    def check_exit_condition(
        self,
        position: FundingArbPosition,
        current_funding_rate: float,
    ) -> bool:
        """Check if position should be closed"""
        # Exit if funding rate has reversed significantly
        rate_change = abs(current_funding_rate - position.expected_funding_rate)
        
        if rate_change * 10000 > self.exit_threshold_bps:
            logger.info(f"Exit triggered for {position.symbol}: rate changed significantly")
            return True
        
        # Exit if funding rate flipped sign
        if position.expected_funding_rate * current_funding_rate < 0:
            logger.info(f"Exit triggered for {position.symbol}: rate flipped sign")
            return True
        
        return False
    
    def close_position(
        self,
        position: FundingArbPosition,
        spot_exit_price: float,
        perp_exit_price: float,
    ) -> float:
        """Close a funding arbitrage position"""
        position.state = PositionState.CLOSING
        
        # Calculate PnL on spot leg
        if position.spot_side == "long":
            spot_pnl = (spot_exit_price - position.spot_entry_price) * position.spot_quantity
        else:
            spot_pnl = (position.spot_entry_price - spot_exit_price) * position.spot_quantity
        
        # Calculate PnL on perp leg
        if position.perp_side == "long":
            perp_pnl = (perp_exit_price - position.perp_entry_price) * position.perp_quantity
        else:
            perp_pnl = (position.perp_entry_price - perp_exit_price) * position.perp_quantity
        
        # Total realized PnL
        total_pnl = spot_pnl + perp_pnl + position.cumulative_funding_collected
        
        position.realized_pnl = total_pnl
        position.state = PositionState.CLOSED
        
        self._total_realized_pnl += total_pnl
        
        logger.info(
            f"Closed position {position.symbol}: "
            f"spot_pnl={spot_pnl:.2f}, perp_pnl={perp_pnl:.2f}, "
            f"funding={position.cumulative_funding_collected:.2f}, "
            f"total={total_pnl:.2f}"
        )
        
        return total_pnl
    
    def get_active_positions(self) -> List[FundingArbPosition]:
        """Get all active positions"""
        return [p for p in self._positions if p.state == PositionState.ACTIVE]
    
    def get_strategy_metrics(self) -> Dict:
        """Get comprehensive strategy metrics"""
        active_positions = self.get_active_positions()
        
        total_notional = sum(
            p.spot_quantity * p.spot_entry_price for p in active_positions
        )
        
        avg_annualized = np.mean([p.annualized_return for p in active_positions]) if active_positions else 0.0
        
        return {
            'active_positions': len(active_positions),
            'total_notional': total_notional,
            'cumulative_funding': self._cumulative_funding,
            'total_realized_pnl': self._total_realized_pnl,
            'average_annualized_return': avg_annualized,
            'trade_count': self._trade_count,
            'positions': [
                {
                    'symbol': p.symbol,
                    'annualized_return': p.annualized_return,
                    'funding_collected': p.cumulative_funding_collected,
                }
                for p in active_positions
            ],
        }
    
    def reset(self) -> None:
        """Reset strategy state"""
        self._positions.clear()
        self._cumulative_funding = 0.0
        self._total_realized_pnl = 0.0
        self._trade_count = 0
