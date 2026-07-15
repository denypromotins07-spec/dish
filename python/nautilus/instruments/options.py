"""
Custom Nautilus Instruments for Crypto Derivatives.
Defines CryptoOption and CryptoPerpetual instrument classes
with complex derivatives parameters mapped to Nautilus execution engine.
"""

from __future__ import annotations
from decimal import Decimal
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.enums import AssetClass, OptionKind
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Size, Money


class OptionExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class CryptoOption:
    """
    Custom crypto options instrument definition for Nautilus.
    
    Maps complex derivatives parameters (strike, expiry, put/call, multiplier)
    directly into the Nautilus execution engine.
    """
    
    id: InstrumentId
    raw_symbol: Symbol
    base_currency: str
    quote_currency: str
    settlement_currency: str
    asset_class: AssetClass
    underlying: str
    
    # Options-specific parameters
    strike_price: Price
    expiry_ns: int  # Nanoseconds timestamp
    exercise_style: OptionExerciseStyle
    option_kind: OptionKind  # CALL or PUT
    
    # Contract specifications
    contract_size: Decimal
    contract_multiplier: Decimal
    
    # Price/size precision
    price_precision: int
    size_precision: int
    
    # Trading limits
    max_quantity: Decimal
    min_quantity: Decimal
    max_notional: Optional[Money]
    min_notional: Optional[Money]
    
    # Margin requirements
    initial_margin_pct: Decimal
    maintenance_margin_pct: Decimal
    
    # Tick/lot sizes
    tick_size: Price
    lot_size: Size
    
    def __post_init__(self):
        """Validate instrument parameters"""
        if self.strike_price <= 0:
            raise ValueError("Strike price must be positive")
        
        if self.expiry_ns <= 0:
            raise ValueError("Expiry must be a valid timestamp")
        
        if self.contract_size <= 0:
            raise ValueError("Contract size must be positive")
        
        if self.contract_multiplier <= 0:
            raise ValueError("Contract multiplier must be positive")
    
    @property
    def expiry_dt(self) -> Optional:
        """Get expiry as datetime"""
        try:
            from datetime import datetime
            return datetime.fromtimestamp(self.expiry_ns / 1_000_000_000)
        except Exception:
            return None
    
    @property
    def is_expired(self) -> bool:
        """Check if option is expired"""
        import time
        return self.expiry_ns < time.time_ns()
    
    @property
    def is_call(self) -> bool:
        """Check if option is a call"""
        return self.option_kind == OptionKind.CALL
    
    @property
    def is_put(self) -> bool:
        """Check if option is a put"""
        return self.option_kind == OptionKind.PUT
    
    def get_underlying_notional(self, underlying_price: Decimal) -> Decimal:
        """Calculate the notional value of underlying controlled by one contract"""
        return underlying_price * self.contract_size * self.contract_multiplier
    
    def get_intrinsic_value(self, underlying_price: Decimal) -> Decimal:
        """Calculate intrinsic value at given underlying price"""
        strike = self.strike_price.as_decimal()
        
        if self.is_call:
            return max(underlying_price - strike, Decimal("0"))
        else:
            return max(strike - underlying_price, Decimal("0"))
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "type": "CryptoOption",
            "id": str(self.id),
            "symbol": str(self.raw_symbol),
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "settlement_currency": self.settlement_currency,
            "asset_class": self.asset_class.name,
            "underlying": self.underlying,
            "strike_price": str(self.strike_price),
            "expiry_ns": self.expiry_ns,
            "exercise_style": self.exercise_style.value,
            "option_kind": self.option_kind.name,
            "contract_size": str(self.contract_size),
            "contract_multiplier": str(self.contract_multiplier),
            "price_precision": self.price_precision,
            "size_precision": self.size_precision,
            "max_quantity": str(self.max_quantity),
            "min_quantity": str(self.min_quantity),
            "tick_size": str(self.tick_size),
            "lot_size": str(self.lot_size),
        }


@dataclass(frozen=True)
class CryptoPerpetual:
    """
    Custom crypto perpetual swap instrument definition for Nautilus.
    
    Includes funding rate mechanics and mark price tracking.
    """
    
    id: InstrumentId
    raw_symbol: Symbol
    base_currency: str
    quote_currency: str
    settlement_currency: str
    asset_class: AssetClass
    underlying: str
    
    # Perpetual-specific parameters
    funding_rate_cap: Decimal  # Maximum funding rate
    funding_rate_floor: Decimal  # Minimum funding rate
    funding_interval_secs: int  # Seconds between funding payments
    
    # Contract specifications
    contract_size: Decimal
    contract_multiplier: Decimal
    
    # Price/size precision
    price_precision: int
    size_precision: int
    
    # Trading limits
    max_quantity: Decimal
    min_quantity: Decimal
    max_leverage: Decimal
    max_notional: Optional[Money]
    min_notional: Optional[Money]
    
    # Margin requirements
    initial_margin_pct: Decimal
    maintenance_margin_pct: Decimal
    
    # Tick/lot sizes
    tick_size: Price
    lot_size: Size
    
    def __post_init__(self):
        """Validate instrument parameters"""
        if self.contract_size <= 0:
            raise ValueError("Contract size must be positive")
        
        if self.max_leverage <= 0:
            raise ValueError("Max leverage must be positive")
        
        if self.funding_interval_secs <= 0:
            raise ValueError("Funding interval must be positive")
    
    @property
    def funding_intervals_per_day(self) -> int:
        """Number of funding intervals per day"""
        return 86400 // self.funding_interval_secs
    
    def get_funding_timestamps(self, start_ns: int, count: int) -> list:
        """Generate upcoming funding payment timestamps"""
        interval_ns = self.funding_interval_secs * 1_000_000_000
        
        # Align to interval boundary
        aligned_start = (start_ns // interval_ns) * interval_ns
        
        return [aligned_start + i * interval_ns for i in range(count)]
    
    def calculate_funding_payment(
        self,
        position_size: Decimal,
        entry_price: Decimal,
        funding_rate: Decimal,
        is_long: bool,
    ) -> Decimal:
        """
        Calculate funding payment for a position.
        
        Positive result = receive funding
        Negative result = pay funding
        """
        notional = position_size * entry_price * self.contract_multiplier
        
        if is_long:
            # Long pays when funding is positive
            return -notional * funding_rate
        else:
            # Short receives when funding is positive
            return notional * funding_rate
    
    def get_required_margin(
        self,
        quantity: Decimal,
        price: Decimal,
        leverage: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate required margin for a position"""
        notional = quantity * price * self.contract_multiplier
        
        if leverage is not None:
            leverage_margin = notional / leverage
            return max(leverage_margin, notional * self.maintenance_margin_pct)
        
        return notional * self.initial_margin_pct
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "type": "CryptoPerpetual",
            "id": str(self.id),
            "symbol": str(self.raw_symbol),
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "settlement_currency": self.settlement_currency,
            "asset_class": self.asset_class.name,
            "underlying": self.underlying,
            "funding_rate_cap": str(self.funding_rate_cap),
            "funding_rate_floor": str(self.funding_rate_floor),
            "funding_interval_secs": self.funding_interval_secs,
            "contract_size": str(self.contract_size),
            "contract_multiplier": str(self.contract_multiplier),
            "price_precision": self.price_precision,
            "size_precision": self.size_precision,
            "max_leverage": str(self.max_leverage),
            "initial_margin_pct": str(self.initial_margin_pct),
            "maintenance_margin_pct": str(self.maintenance_margin_pct),
        }


def create_crypto_option(
    venue: str,
    symbol: str,
    underlying: str,
    base_currency: str,
    quote_currency: str,
    strike: float,
    expiry_ts: float,
    option_kind: str,
    contract_size: float = 1.0,
    contract_multiplier: float = 1.0,
    price_precision: int = 2,
    size_precision: int = 4,
) -> CryptoOption:
    """
    Factory function to create a CryptoOption instrument.
    
    Args:
        venue: Exchange venue name
        symbol: Instrument symbol
        underlying: Underlying asset symbol
        base_currency: Base currency (e.g., BTC)
        quote_currency: Quote currency (e.g., USD)
        strike: Strike price
        expiry_ts: Expiry timestamp (seconds since epoch)
        option_kind: 'call' or 'put'
        contract_size: Size of one contract
        contract_multiplier: Contract multiplier
        price_precision: Decimal places for price
        size_precision: Decimal places for size
    
    Returns:
        Configured CryptoOption instance
    """
    instrument_id = InstrumentId.from_str(f"{symbol}.{venue}")
    raw_symbol = Symbol(symbol)
    
    ok = OptionKind.CALL if option_kind.lower() == "call" else OptionKind.PUT
    exercise = OptionExerciseStyle.EUROPEAN  # Default to European for crypto
    
    return CryptoOption(
        id=instrument_id,
        raw_symbol=raw_symbol,
        base_currency=base_currency,
        quote_currency=quote_currency,
        settlement_currency=quote_currency,
        asset_class=AssetClass.DIGITAL_CURRENCY,
        underlying=underlying,
        strike_price=Price(strike, price_precision),
        expiry_ns=int(expiry_ts * 1_000_000_000),
        exercise_style=exercise,
        option_kind=ok,
        contract_size=Decimal(str(contract_size)),
        contract_multiplier=Decimal(str(contract_multiplier)),
        price_precision=price_precision,
        size_precision=size_precision,
        max_quantity=Decimal("1000000"),
        min_quantity=Decimal("0.0001"),
        max_notional=None,
        min_notional=None,
        initial_margin_pct=Decimal("0.15"),
        maintenance_margin_pct=Decimal("0.05"),
        tick_size=Price(0.01, price_precision),
        lot_size=Size(0.0001, size_precision),
    )


def create_crypto_perpetual(
    venue: str,
    symbol: str,
    underlying: str,
    base_currency: str,
    quote_currency: str,
    contract_size: float = 1.0,
    contract_multiplier: float = 1.0,
    max_leverage: float = 100.0,
    price_precision: int = 2,
    size_precision: int = 4,
) -> CryptoPerpetual:
    """
    Factory function to create a CryptoPerpetual instrument.
    
    Args:
        venue: Exchange venue name
        symbol: Instrument symbol
        underlying: Underlying asset symbol
        base_currency: Base currency (e.g., BTC)
        quote_currency: Quote currency (e.g., USD)
        contract_size: Size of one contract
        contract_multiplier: Contract multiplier
        max_leverage: Maximum allowed leverage
        price_precision: Decimal places for price
        size_precision: Decimal places for size
    
    Returns:
        Configured CryptoPerpetual instance
    """
    instrument_id = InstrumentId.from_str(f"{symbol}.{venue}")
    raw_symbol = Symbol(symbol)
    
    return CryptoPerpetual(
        id=instrument_id,
        raw_symbol=raw_symbol,
        base_currency=base_currency,
        quote_currency=quote_currency,
        settlement_currency=quote_currency,
        asset_class=AssetClass.DIGITAL_CURRENCY,
        underlying=underlying,
        funding_rate_cap=Decimal("0.01"),
        funding_rate_floor=Decimal("-0.01"),
        funding_interval_secs=28800,  # 8 hours
        contract_size=Decimal(str(contract_size)),
        contract_multiplier=Decimal(str(contract_multiplier)),
        price_precision=price_precision,
        size_precision=size_precision,
        max_quantity=Decimal("1000000"),
        min_quantity=Decimal("0.0001"),
        max_leverage=Decimal(str(max_leverage)),
        max_notional=None,
        min_notional=None,
        initial_margin_pct=Decimal("0.01"),
        maintenance_margin_pct=Decimal("0.005"),
        tick_size=Price(0.01, price_precision),
        lot_size=Size(0.0001, size_precision),
    )
