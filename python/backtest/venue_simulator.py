"""
Custom Binance venue simulator for NautilusTrader backtesting.
Defines precise maker/taker fee schedules, tick sizes, lot sizes, and realistic latency distributions.
"""

import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.enums import AccountType, OMSType, OrderSide, OrderType
from nautilus_trader.backtest.config import BacktestVenueConfig

logger = logging.getLogger(__name__)


class LatencyDistributionType(Enum):
    """Types of latency distributions for simulation."""
    CONSTANT = "constant"
    UNIFORM = "uniform"
    LOG_NORMAL = "log_normal"
    EXPONENTIAL = "exponential"


@dataclass
class LatencyConfig:
    """Configuration for network/exchange latency simulation."""
    distribution: LatencyDistributionType = LatencyDistributionType.LOG_NORMAL
    base_latency_ms: float = 10.0  # Base latency in milliseconds
    jitter_ms: float = 5.0  # Additional jitter in milliseconds
    mean_ln: float = 2.3  # For log-normal: mean of log(latency)
    std_ln: float = 0.5  # For log-normal: std dev of log(latency)
    rate: float = 0.1  # For exponential distribution
    
    def sample_latency_ms(self) -> float:
        """Sample a latency value based on the configured distribution."""
        if self.distribution == LatencyDistributionType.CONSTANT:
            return self.base_latency_ms
        
        elif self.distribution == LatencyDistributionType.UNIFORM:
            return self.base_latency_ms + random.uniform(0, self.jitter_ms)
        
        elif self.distribution == LatencyDistributionType.LOG_NORMAL:
            # Log-normal distribution for realistic network delays
            latency = random.lognormvariate(self.mean_ln, self.std_ln)
            return max(self.base_latency_ms, latency)
        
        elif self.distribution == LatencyDistributionType.EXPONENTIAL:
            # Exponential for rare high-latency events
            latency = random.expovariate(self.rate)
            return self.base_latency_ms + latency
        
        return self.base_latency_ms


@dataclass
class FeeSchedule:
    """Maker/taker fee schedule configuration."""
    maker_fee_bps: float = 1.0  # Basis points (0.01% = 1 bp)
    taker_fee_bps: float = 1.0
    maker_rebate_bps: float = 0.0  # Negative fee = rebate
    volume_discount_threshold: float = 1_000_000.0  # USD volume for discount
    volume_discount_rate: float = 0.2  # 20% discount after threshold
    
    def calculate_maker_fee(self, notional: float, volume_30d: float = 0.0) -> float:
        """Calculate maker fee for a given notional amount."""
        fee_rate = self.maker_fee_bps / 10000.0
        
        # Apply volume discount if applicable
        if volume_30d >= self.volume_discount_threshold:
            fee_rate *= (1 - self.volume_discount_rate)
        
        # Apply rebate (negative fee)
        fee = notional * (fee_rate - self.maker_rebate_bps / 10000.0)
        return max(0.0, fee) if self.maker_rebate_bps == 0 else fee
    
    def calculate_taker_fee(self, notional: float, volume_30d: float = 0.0) -> float:
        """Calculate taker fee for a given notional amount."""
        fee_rate = self.taker_fee_bps / 10000.0
        
        if volume_30d >= self.volume_discount_threshold:
            fee_rate *= (1 - self.volume_discount_rate)
        
        return notional * fee_rate


@dataclass
class InstrumentSpec:
    """Specification for a trading instrument."""
    symbol: str
    price_precision: int
    size_precision: int
    tick_size: float
    lot_size: float
    min_qty: float
    max_qty: float
    min_notional: float = 10.0  # Minimum order value in quote currency
    
    def round_price(self, price: float) -> float:
        """Round price to valid tick size."""
        return round(price / self.tick_size) * self.tick_size
    
    def round_quantity(self, quantity: float) -> float:
        """Round quantity to valid lot size."""
        return round(quantity / self.lot_size) * self.lot_size
    
    def validate_order(self, price: float, quantity: float) -> Tuple[bool, Optional[str]]:
        """Validate an order against instrument specifications."""
        rounded_price = self.round_price(price)
        rounded_qty = self.round_quantity(quantity)
        
        if rounded_qty < self.min_qty:
            return False, f"Quantity {rounded_qty} below minimum {self.min_qty}"
        
        if rounded_qty > self.max_qty:
            return False, f"Quantity {rounded_qty} above maximum {self.max_qty}"
        
        notional = rounded_price * rounded_qty
        if notional < self.min_notional:
            return False, f"Notional {notional} below minimum {self.min_notional}"
        
        return True, None


@dataclass
class VenueState:
    """Runtime state for the venue simulator."""
    total_orders: int = 0
    total_trades: int = 0
    total_volume: float = 0.0
    total_fees_collected: float = 0.0
    total_rebates_paid: float = 0.0
    active_orders: Dict[int, dict] = field(default_factory=dict)
    

class BinanceVenueSimulator:
    """
    Custom Binance venue simulator with realistic market microstructure.
    
    Features:
    - Precise maker/taker fee schedules with volume discounts
    - Tick size and lot size validation
    - Realistic latency distributions (log-normal network delay)
    - Exchange processing lag simulation
    - Partial fill modeling
    - Queue position tracking
    """
    
    def __init__(
        self,
        venue_name: str = "BINANCE",
        account_type: AccountType = AccountType.MARGIN,
        oms_type: OMSType = OMSType.NETTING,
        starting_balance: float = 100_000.0,
        base_currency: str = "USDT",
        fee_schedule: Optional[FeeSchedule] = None,
        latency_config: Optional[LatencyConfig] = None,
    ):
        self.venue = Venue(venue_name.upper())
        self.account_type = account_type
        self.oms_type = oms_type
        self.starting_balance = Money(starting_balance, base_currency)
        self.base_currency = base_currency
        
        self.fee_schedule = fee_schedule or FeeSchedule()
        self.latency_config = latency_config or LatencyConfig()
        
        self.state = VenueState()
        self.instruments: Dict[str, InstrumentSpec] = {}
        
        logger.info(f"BinanceVenueSimulator initialized for {venue_name}")
    
    def add_instrument(self, spec: InstrumentSpec):
        """Add a trading instrument to the venue."""
        self.instruments[spec.symbol] = spec
        logger.debug(f"Added instrument: {spec.symbol}")
    
    def create_nautilus_config(self) -> BacktestVenueConfig:
        """
        Create a NautilusTrader BacktestVenueConfig from this simulator's settings.
        
        Returns
        -------
        BacktestVenueConfig
            Configuration object for Nautilus backtesting.
        """
        return BacktestVenueConfig(
            name=self.venue.value,
            oms_type=self.oms_type,
            account_type=self.account_type,
            starting_balances=[self.starting_balance],
            modules=[],  # External data/execution modules
            base_currency=self.base_currency,
            # Fee configuration
            maker_fee=self.fee_schedule.maker_fee_bps / 10000.0,
            taker_fee=self.fee_schedule.taker_fee_bps / 10000.0,
            # Realistic settings
            fill_limit=True,  # Enable partial fills
            fill_stop=True,   # Enable stop orders
        )
    
    def simulate_order_latency(self) -> float:
        """
        Simulate network + exchange processing latency.
        
        Returns
        -------
        float
            Latency in milliseconds.
        """
        network_latency = self.latency_config.sample_latency_ms()
        exchange_processing = random.uniform(1.0, 5.0)  # 1-5ms exchange processing
        return network_latency + exchange_processing
    
    def calculate_execution_price(
        self,
        order_price: float,
        order_side: OrderSide,
        current_bid: float,
        current_ask: float,
        is_market_order: bool = False,
    ) -> float:
        """
        Calculate the actual execution price considering spread and slippage.
        
        Parameters
        ----------
        order_price : float
            The order's limit price.
        order_side : OrderSide
            Buy or sell.
        current_bid : float
            Current best bid price.
        current_ask : float
            Current best ask price.
        is_market_order : bool, default False
            Whether this is a market order.
            
        Returns
        -------
        float
            Executed price.
        """
        if is_market_order:
            # Market orders execute at opposite side of book
            if order_side == OrderSide.BUY:
                return current_ask
            else:
                return current_bid
        
        # Limit orders: check if they cross the spread
        if order_side == OrderSide.BUY:
            if order_price >= current_ask:
                # Crosses spread, executes at ask
                return current_ask
            return order_price
        else:
            if order_price <= current_bid:
                # Crosses spread, executes at bid
                return current_bid
            return order_price
    
    def calculate_fill_probability(
        self,
        order_price: float,
        order_side: OrderSide,
        current_bid: float,
        current_ask: float,
        queue_position: int = 0,
        time_in_force: str = "GTC",
    ) -> float:
        """
        Estimate the probability of a limit order being filled.
        
        Parameters
        ----------
        order_price : float
            Order limit price.
        order_side : OrderSide
            Buy or sell.
        current_bid : float
            Current best bid.
        current_ask : float
            Current best ask.
        queue_position : int, default 0
            Position in the order queue.
        time_in_force : str, default "GTC"
            Time in force instruction.
            
        Returns
        -------
        float
            Fill probability (0.0 to 1.0).
        """
        mid_price = (current_bid + current_ask) / 2.0
        spread = current_ask - current_bid
        spread_pct = spread / mid_price if mid_price > 0 else 0
        
        if order_side == OrderSide.BUY:
            distance_from_mid = (mid_price - order_price) / mid_price if mid_price > 0 else 0
        else:
            distance_from_mid = (order_price - mid_price) / mid_price if mid_price > 0 else 0
        
        # Base probability decreases with distance from mid
        base_prob = max(0.0, 1.0 - abs(distance_from_mid) / (spread_pct * 10))
        
        # Queue position penalty
        queue_penalty = min(0.5, queue_position * 0.01)
        
        # Time in force adjustment
        tif_factor = 1.0 if time_in_force == "GTC" else 0.7
        
        return max(0.0, min(1.0, (base_prob - queue_penalty) * tif_factor))
    
    def process_fill(
        self,
        order_id: int,
        symbol: str,
        side: OrderSide,
        price: float,
        quantity: float,
        is_maker: bool = False,
    ) -> Dict:
        """
        Process a trade fill and calculate fees.
        
        Parameters
        ----------
        order_id : int
            Order identifier.
        symbol : str
            Trading symbol.
        side : OrderSide
            Buy or sell.
        price : float
            Fill price.
        quantity : float
            Fill quantity.
        is_maker : bool, default False
            Whether the order provided liquidity.
            
        Returns
        -------
        Dict
            Fill details including fees.
        """
        notional = price * quantity
        
        # Calculate fees
        if is_maker:
            fee = self.fee_schedule.calculate_maker_fee(notional)
            fee_type = "maker_fee"
            if self.fee_schedule.maker_rebate_bps > 0:
                self.state.total_rebates_paid += abs(fee) if fee < 0 else 0
        else:
            fee = self.fee_schedule.calculate_taker_fee(notional)
            fee_type = "taker_fee"
        
        # Update state
        self.state.total_trades += 1
        self.state.total_volume += notional
        self.state.total_fees_collected += max(0, fee)
        
        fill_result = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side.name,
            "price": price,
            "quantity": quantity,
            "notional": notional,
            "fee": fee,
            "fee_type": fee_type,
            "is_maker": is_maker,
            "latency_ms": self.simulate_order_latency(),
        }
        
        logger.debug(f"Fill processed: {fill_result}")
        return fill_result
    
    def get_statistics(self) -> Dict:
        """Get venue statistics."""
        return {
            "venue": self.venue.value,
            "total_orders": self.state.total_orders,
            "total_trades": self.state.total_trades,
            "total_volume": self.state.total_volume,
            "total_fees_collected": self.state.total_fees_collected,
            "total_rebates_paid": self.state.total_rebates_paid,
            "net_fees": self.state.total_fees_collected - self.state.total_rebates_paid,
            "active_instruments": len(self.instruments),
        }


def create_binance_spot_instruments() -> List[InstrumentSpec]:
    """
    Create standard Binance spot instrument specifications.
    
    Returns
    -------
    List[InstrumentSpec]
        List of major crypto instruments.
    """
    instruments = [
        InstrumentSpec(
            symbol="BTCUSDT",
            price_precision=2,
            size_precision=6,
            tick_size=0.01,
            lot_size=0.00001,
            min_qty=0.00001,
            max_qty=9000.0,
            min_notional=10.0,
        ),
        InstrumentSpec(
            symbol="ETHUSDT",
            price_precision=2,
            size_precision=5,
            tick_size=0.01,
            lot_size=0.0001,
            min_qty=0.0001,
            max_qty=10000.0,
            min_notional=10.0,
        ),
        InstrumentSpec(
            symbol="BNBUSDT",
            price_precision=2,
            size_precision=4,
            tick_size=0.01,
            lot_size=0.001,
            min_qty=0.001,
            max_qty=100000.0,
            min_notional=10.0,
        ),
    ]
    
    return instruments


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create simulator with realistic settings
    simulator = BinanceVenueSimulator(
        venue_name="BINANCE",
        fee_schedule=FeeSchedule(
            maker_fee_bps=1.0,
            taker_fee_bps=1.0,
            maker_rebate_bps=0.0,
        ),
        latency_config=LatencyConfig(
            distribution=LatencyDistributionType.LOG_NORMAL,
            base_latency_ms=10.0,
            mean_ln=2.3,
            std_ln=0.5,
        ),
    )
    
    # Add instruments
    for inst in create_binance_spot_instruments():
        simulator.add_instrument(inst)
    
    print(f"Venue: {simulator.venue}")
    print(f"Instruments: {list(simulator.instruments.keys())}")
    print(f"Statistics: {simulator.get_statistics()}")
