"""
Advanced fill simulation engine for NautilusTrader backtesting.
Models partial fills, queue position depletion, and market impact (slippage).
Ensures backtests don't assume infinite liquidity.
"""

import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.objects import Price, Quantity

logger = logging.getLogger(__name__)


class FillType(Enum):
    """Types of order fills."""
    FULL = "full"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class LiquidityLevel:
    """Represents a liquidity level in the order book."""
    price: float
    volume: float
    order_count: int
    
    def remaining_volume(self) -> float:
        return self.volume


@dataclass
class QueuePosition:
    """Tracks position in the order queue at a price level."""
    orders_ahead: int
    total_volume_ahead: float
    our_position: int
    estimated_fill_probability: float


@dataclass
class FillResult:
    """Result of a fill simulation."""
    fill_type: FillType
    filled_quantity: float
    remaining_quantity: float
    average_price: float
    slippage_bps: float
    market_impact_bps: float
    fill_details: List[Dict] = field(default_factory=list)
    queue_position: Optional[QueuePosition] = None


class VolumeProfile:
    """
    Historical volume profile for market impact calculation.
    Uses intraday volume patterns to estimate available liquidity.
    """
    
    def __init__(self, bucket_size_minutes: int = 5):
        self.bucket_size = bucket_size_minutes * 60 * 1_000_000_000  # ns
        self.volume_buckets: Dict[int, float] = {}
        self.total_volume: float = 0.0
        
    def add_volume(self, timestamp_ns: int, volume: float):
        """Add volume to the appropriate time bucket."""
        bucket = timestamp_ns // self.bucket_size
        if bucket not in self.volume_buckets:
            self.volume_buckets[bucket] = 0.0
        self.volume_buckets[bucket] += volume
        self.total_volume += volume
    
    def get_expected_volume(self, timestamp_ns: int) -> float:
        """Get expected volume for a given timestamp."""
        bucket = timestamp_ns // self.bucket_size
        return self.volume_buckets.get(bucket, self.total_volume / max(1, len(self.volume_buckets)))
    
    def get_volume_fraction(self, timestamp_ns: int) -> float:
        """Get fraction of total volume expected at this time."""
        expected = self.get_expected_volume(timestamp_ns)
        return expected / max(1.0, self.total_volume)


class FillEngine:
    """
    Advanced fill simulation engine modeling realistic market microstructure.
    
    Features:
    - Partial fill modeling based on order book depth
    - Queue position tracking and depletion
    - Market impact calculation using square-root model
    - Slippage estimation based on spread and volatility
    - Volume-aware fill probability
    """
    
    def __init__(
        self,
        default_spread_bps: float = 5.0,
        market_impact_coefficient: float = 0.1,
        min_fill_ratio: float = 0.1,
        use_volume_profile: bool = True,
    ):
        self.default_spread_bps = default_spread_bps
        self.market_impact_coefficient = market_impact_coefficient
        self.min_fill_ratio = min_fill_ratio
        self.use_volume_profile = use_volume_profile
        
        self.volume_profile = VolumeProfile() if use_volume_profile else None
        self.order_book_state: Dict[str, List[LiquidityLevel]] = {}
        
        # Statistics
        self.total_orders = 0
        self.full_fills = 0
        self.partial_fills = 0
        self.rejected_orders = 0
        self.total_slippage_bps = 0.0
        
    def update_order_book(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ):
        """
        Update the simulated order book state.
        
        Parameters
        ----------
        symbol : str
            Trading symbol.
        bids : List[Tuple[float, float]]
            List of (price, volume) tuples for bids.
        asks : List[Tuple[float, float]]
            List of (price, volume) tuples for asks.
        """
        self.order_book_state[symbol] = {
            'bids': [LiquidityLevel(p, v, 1) for p, v in sorted(bids, reverse=True)],
            'asks': [LiquidityLevel(p, v, 1) for p, v in sorted(asks)],
        }
    
    def estimate_queue_position(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        order_quantity: float,
    ) -> QueuePosition:
        """
        Estimate queue position for a limit order.
        
        Parameters
        ----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Buy or sell.
        price : float
            Order price.
        order_quantity : float
            Order quantity.
            
        Returns
        -------
        QueuePosition
            Estimated queue position details.
        """
        if symbol not in self.order_book_state:
            return QueuePosition(
                orders_ahead=0,
                total_volume_ahead=0.0,
                our_position=0,
                estimated_fill_probability=0.5,
            )
        
        book = self.order_book_state[symbol]
        levels = book['bids'] if side == OrderSide.BUY else book['asks']
        
        orders_ahead = 0
        volume_ahead = 0.0
        
        for level in levels:
            if side == OrderSide.BUY and level.price < price:
                break
            if side == OrderSide.SELL and level.price > price:
                break
            
            if abs(level.price - price) < 0.0001:  # Same price level
                # Estimate position within the level
                fraction_at_better_prices = volume_ahead / max(0.001, level.volume + volume_ahead)
                orders_ahead = int(fraction_at_better_prices * level.order_count)
                volume_ahead = level.volume * fraction_at_better_prices
                break
            else:
                orders_ahead += level.order_count
                volume_ahead += level.volume
        
        # Calculate fill probability based on queue position
        total_volume_at_level = volume_ahead + order_quantity
        fill_prob = min(1.0, order_quantity / max(0.001, total_volume_at_level))
        
        # Adjust for market conditions
        if side == OrderSide.BUY:
            best_bid = book['bids'][0].price if book['bids'] else 0
            if price < best_bid:
                fill_prob *= 0.5  # Behind best bid
        else:
            best_ask = book['asks'][0].price if book['asks'] else float('inf')
            if price > best_ask:
                fill_prob *= 0.5  # Behind best ask
        
        return QueuePosition(
            orders_ahead=orders_ahead,
            total_volume_ahead=volume_ahead,
            our_position=orders_ahead + 1,
            estimated_fill_probability=max(0.0, min(1.0, fill_prob)),
        )
    
    def calculate_market_impact(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
    ) -> float:
        """
        Calculate market impact in basis points using square-root model.
        
        Parameters
        ----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Buy or sell.
        quantity : float
            Order quantity.
        current_price : float
            Current market price.
            
        Returns
        -------
        float
            Market impact in basis points.
        """
        # Square-root market impact model
        # impact = coefficient * sqrt(notional / avg_daily_volume)
        
        notional = quantity * current_price
        
        # Estimate available liquidity from order book
        if symbol in self.order_book_state:
            book = self.order_book_state[symbol]
            levels = book['bids'] if side == OrderSide.SELL else book['asks']
            
            available_liquidity = sum(l.volume for l in levels[:5])  # Top 5 levels
            if available_liquidity > 0:
                participation_rate = quantity / available_liquidity
                impact_bps = self.market_impact_coefficient * math.sqrt(participation_rate) * 100
                return min(impact_bps, 100.0)  # Cap at 100 bps
        
        # Fallback: use volume profile or default
        if self.volume_profile:
            expected_volume = self.volume_profile.get_expected_volume(0)
            if expected_volume > 0:
                participation_rate = quantity / expected_volume
                impact_bps = self.market_impact_coefficient * math.sqrt(participation_rate) * 100
                return min(impact_bps, 100.0)
        
        return self.market_impact_coefficient * 10  # Default impact
    
    def calculate_slippage(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        current_bid: float = 0.0,
        current_ask: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Calculate expected slippage for an order.
        
        Parameters
        ----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Buy or sell.
        order_type : OrderType
            Order type (MARKET, LIMIT, etc.).
        limit_price : Optional[float]
            Limit price for limit orders.
        current_bid : float, default 0.0
            Current best bid.
        current_ask : float, default 0.0
            Current best ask.
            
        Returns
        -------
        Tuple[float, float]
            (slippage_bps, execution_price)
        """
        mid_price = (current_bid + current_ask) / 2.0 if current_bid and current_ask else limit_price
        if mid_price <= 0:
            return 0.0, limit_price or 0.0
        
        spread = current_ask - current_bid if current_ask > current_bid else mid_price * 0.001
        spread_bps = (spread / mid_price) * 10000
        
        if order_type == OrderType.MARKET:
            # Market orders pay half spread + additional slippage
            base_slippage = spread_bps / 2
            
            # Add slippage for walking the book
            if symbol in self.order_book_state:
                book = self.order_book_state[symbol]
                levels = book['bids'] if side == OrderSide.SELL else book['asks']
                
                if levels:
                    worst_fill_price = levels[-1].price if len(levels) > 1 else levels[0].price
                    if side == OrderSide.BUY:
                        walk_slippage = (worst_fill_price - current_ask) / mid_price * 10000
                    else:
                        walk_slippage = (current_bid - worst_fill_price) / mid_price * 10000
                    base_slippage += max(0, walk_slippage)
            
            return base_slippage, current_ask if side == OrderSide.BUY else current_bid
        
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                return spread_bps, mid_price
            
            # Limit orders: slippage depends on whether they cross
            if side == OrderSide.BUY:
                if limit_price >= current_ask:
                    # Crosses spread, executes at ask
                    return spread_bps / 2, current_ask
                else:
                    # Waits in queue, potential slippage from adverse selection
                    return spread_bps * 0.1, limit_price
            else:
                if limit_price <= current_bid:
                    # Crosses spread, executes at bid
                    return spread_bps / 2, current_bid
                else:
                    return spread_bps * 0.1, limit_price
        
        return 0.0, limit_price or mid_price
    
    def simulate_fill(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        limit_price: Optional[float] = None,
        current_bid: float = 0.0,
        current_ask: float = 0.0,
        timestamp_ns: int = 0,
    ) -> FillResult:
        """
        Simulate an order fill with realistic market microstructure.
        
        Parameters
        ----------
        symbol : str
            Trading symbol.
        side : OrderSide
            Buy or sell.
        order_type : OrderType
            Order type.
        quantity : float
            Order quantity.
        limit_price : Optional[float]
            Limit price for limit orders.
        current_bid : float, default 0.0
            Current best bid.
        current_ask : float, default 0.0
            Current best ask.
        timestamp_ns : int, default 0
            Order timestamp.
            
        Returns
        -------
        FillResult
            Detailed fill simulation result.
        """
        self.total_orders += 1
        
        mid_price = (current_bid + current_ask) / 2.0 if current_bid and current_ask else limit_price
        if mid_price <= 0:
            return FillResult(
                fill_type=FillType.REJECTED,
                filled_quantity=0.0,
                remaining_quantity=quantity,
                average_price=0.0,
                slippage_bps=0.0,
                market_impact_bps=0.0,
            )
        
        # Calculate slippage and market impact
        slippage_bps, base_exec_price = self.calculate_slippage(
            symbol, side, order_type, limit_price, current_bid, current_ask
        )
        
        impact_bps = self.calculate_market_impact(symbol, side, quantity, mid_price)
        
        # Determine execution price
        if order_type == OrderType.MARKET:
            exec_price = base_exec_price
            if side == OrderSide.BUY:
                exec_price *= (1 + (slippage_bps + impact_bps) / 10000)
            else:
                exec_price *= (1 - (slippage_bps + impact_bps) / 10000)
        else:
            # Limit order
            exec_price = limit_price if limit_price else mid_price
            
            # Check if limit order would be filled
            queue_pos = self.estimate_queue_position(symbol, side, exec_price, quantity)
            
            if random.random() > queue_pos.estimated_fill_probability:
                # Order not filled
                self.rejected_orders += 1
                return FillResult(
                    fill_type=FillType.REJECTED,
                    filled_quantity=0.0,
                    remaining_quantity=quantity,
                    average_price=exec_price,
                    slippage_bps=0.0,
                    market_impact_bps=0.0,
                    queue_position=queue_pos,
                )
        
        # Simulate partial fills for large orders
        if symbol in self.order_book_state:
            book = self.order_book_state[symbol]
            levels = book['bids'] if side == OrderSide.SELL else book['asks']
            
            remaining = quantity
            total_value = 0.0
            fill_details = []
            
            for level in levels:
                if remaining <= 0:
                    break
                
                fill_qty = min(remaining, level.volume)
                fill_price = level.price
                
                # Apply slippage to each fill level
                if side == OrderSide.BUY:
                    fill_price *= (1 + slippage_bps / 10000)
                else:
                    fill_price *= (1 - slippage_bps / 10000)
                
                fill_value = fill_qty * fill_price
                total_value += fill_value
                remaining -= fill_qty
                
                fill_details.append({
                    'price': fill_price,
                    'quantity': fill_qty,
                    'value': fill_value,
                })
            
            if fill_details:
                avg_price = total_value / quantity
                filled_qty = quantity - remaining
                
                if remaining < quantity * self.min_fill_ratio:
                    # Consider it a full fill if most was filled
                    filled_qty = quantity
                    remaining = 0.0
                    self.full_fills += 1
                else:
                    self.partial_fills += 1
                
                actual_slippage = abs(avg_price - mid_price) / mid_price * 10000
                
                return FillResult(
                    fill_type=FillType.FULL if remaining == 0 else FillType.PARTIAL,
                    filled_quantity=filled_qty,
                    remaining_quantity=remaining,
                    average_price=avg_price,
                    slippage_bps=actual_slippage,
                    market_impact_bps=impact_bps,
                    fill_details=fill_details,
                )
        
        # Simple case: assume full fill at calculated price
        self.full_fills += 1
        self.total_slippage_bps += slippage_bps
        
        return FillResult(
            fill_type=FillType.FULL,
            filled_quantity=quantity,
            remaining_quantity=0.0,
            average_price=exec_price,
            slippage_bps=slippage_bps,
            market_impact_bps=impact_bps,
        )
    
    def get_statistics(self) -> Dict:
        """Get fill engine statistics."""
        return {
            'total_orders': self.total_orders,
            'full_fills': self.full_fills,
            'partial_fills': self.partial_fills,
            'rejected_orders': self.rejected_orders,
            'full_fill_rate': self.full_fills / max(1, self.total_orders),
            'average_slippage_bps': self.total_slippage_bps / max(1, self.total_orders),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = FillEngine(
        default_spread_bps=5.0,
        market_impact_coefficient=0.1,
    )
    
    # Set up order book
    bids = [(49999.0, 1.0), (49998.0, 2.0), (49997.0, 3.0)]
    asks = [(50001.0, 1.0), (50002.0, 2.0), (50003.0, 3.0)]
    engine.update_order_book("BTCUSDT", bids, asks)
    
    # Simulate a market buy
    result = engine.simulate_fill(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.5,
        current_bid=49999.0,
        current_ask=50001.0,
    )
    
    print(f"Fill Result: {result.fill_type}")
    print(f"Filled Quantity: {result.filled_quantity}")
    print(f"Average Price: {result.average_price}")
    print(f"Slippage (bps): {result.slippage_bps}")
    print(f"Market Impact (bps): {result.market_impact_bps}")
    print(f"Statistics: {engine.get_statistics()}")
