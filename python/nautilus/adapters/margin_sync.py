"""
Zero-copy PyO3 bridge for margin engine synchronization.
Synchronizes Rust margin engine state with Nautilus Portfolio component,
ensuring Python strategy layer never over-leverages the account.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# PyO3 imports (would be compiled as Rust extension)
# In production, this would use: from rust_margin_engine import MarginEngineBridge
# For now, we define the interface that the Rust bridge would implement

logger = logging.getLogger(__name__)


class MarginMode(Enum):
    CROSS = "cross"
    ISOLATED = "isolated"


@dataclass
class MarginState:
    """Snapshot of margin account state from Rust engine"""
    wallet_balance: float
    available_balance: float
    total_margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    equity: float
    margin_ratio: float
    is_at_liquidation_risk: bool
    timestamp_ns: int


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    liquidation_price: float
    margin_used: float
    leverage: float


@dataclass
class MarginCheckResult:
    """Result of margin check before order submission"""
    allowed: bool
    reason: str
    available_margin: float
    required_margin: float
    post_order_equity: float
    post_order_margin_ratio: float


class MarginSyncBridge:
    """
    Zero-copy PyO3 bridge to Rust margin engine.
    
    This class provides the Python interface to the high-performance
    Rust margin calculation engine, ensuring:
    - No memory copies between Rust and Python
    - Microsecond-level latency for margin checks
    - Atomic state updates
    - Prevention of over-leverage in Python strategies
    """

    def __init__(self, engine_ptr: Optional[int] = None):
        """
        Initialize the margin sync bridge.
        
        Args:
            engine_ptr: Pointer to Rust margin engine (when using PyO3 FFI)
                       If None, uses mock/simulation mode
        """
        self._engine_ptr = engine_ptr
        self._last_sync_ns: int = 0
        self._cached_state: Optional[MarginState] = None
        self._position_risks: Dict[str, PositionRisk] = {}
        self._is_connected: bool = False
        
        # Safety thresholds
        self._max_margin_ratio: float = 0.8  # Never exceed 80% margin usage
        self._safety_buffer_pct: float = 0.1  # 10% safety buffer
        
    def connect(self, engine_config: Dict[str, Any]) -> bool:
        """
        Connect to Rust margin engine.
        
        In production with PyO3, this would:
        1. Load the Rust shared library
        2. Get function pointers to margin engine APIs
        3. Initialize zero-copy shared memory region
        """
        try:
            # Simulated connection for demonstration
            # Real implementation would use PyO3 FFI
            self._is_connected = True
            self._cached_state = MarginState(
                wallet_balance=engine_config.get('initial_balance', 100000.0),
                available_balance=engine_config.get('initial_balance', 100000.0),
                total_margin_used=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                equity=engine_config.get('initial_balance', 100000.0),
                margin_ratio=0.0,
                is_at_liquidation_risk=False,
                timestamp_ns=0,
            )
            logger.info("Connected to Rust margin engine")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to margin engine: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Rust engine"""
        self._is_connected = False
        logger.info("Disconnected from margin engine")
    
    def sync_state(self) -> Optional[MarginState]:
        """
        Synchronize state from Rust engine (zero-copy read).
        
        Returns current margin state or None if not connected.
        """
        if not self._is_connected:
            return None
        
        # In real PyO3 implementation, this would directly read
        # from shared memory without copying
        # state_ptr = unsafe { rust_get_margin_state(self._engine_ptr) }
        # self._cached_state = MarginState.from_ptr(state_ptr)
        
        # Update cached state timestamp
        import time
        self._last_sync_ns = time.time_ns()
        
        return self._cached_state
    
    def check_margin_before_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        leverage: float,
    ) -> MarginCheckResult:
        """
        Check if new order would violate margin constraints.
        
        This is called BEFORE submitting any order to ensure
        the Python strategy doesn't over-leverage.
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Order price
            leverage: Requested leverage
        
        Returns:
            MarginCheckResult indicating if order is allowed
        """
        if not self._is_connected or self._cached_state is None:
            return MarginCheckResult(
                allowed=False,
                reason="Not connected to margin engine",
                available_margin=0.0,
                required_margin=0.0,
                post_order_equity=0.0,
                post_order_margin_ratio=0.0,
            )
        
        # Calculate required margin for new position
        notional = quantity * price
        required_margin = notional / leverage
        
        # Add safety buffer
        required_with_buffer = required_margin * (1 + self._safety_buffer_pct)
        
        available = self._cached_state.available_balance
        
        if required_with_buffer > available:
            return MarginCheckResult(
                allowed=False,
                reason="Insufficient available margin",
                available_margin=available,
                required_margin=required_with_buffer,
                post_order_equity=self._cached_state.equity,
                post_order_margin_ratio=self._cached_state.margin_ratio,
            )
        
        # Calculate post-order margin ratio
        new_margin_used = self._cached_state.total_margin_used + required_margin
        post_equity = self._cached_state.equity
        post_margin_ratio = new_margin_used / post_equity if post_equity > 0 else 1.0
        
        # Check against maximum allowed margin ratio
        if post_margin_ratio > self._max_margin_ratio:
            return MarginCheckResult(
                allowed=False,
                reason=f"Would exceed max margin ratio ({self._max_margin_ratio:.1%})",
                available_margin=available,
                required_margin=required_with_buffer,
                post_order_equity=post_equity,
                post_order_margin_ratio=post_margin_ratio,
            )
        
        return MarginCheckResult(
            allowed=True,
            reason="OK",
            available_margin=available,
            required_margin=required_with_buffer,
            post_order_equity=post_equity,
            post_order_margin_ratio=post_margin_ratio,
        )
    
    def update_position_risk(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        mark_price: float,
        liquidation_price: float,
        leverage: float,
    ) -> None:
        """Update risk metrics for a position"""
        unrealized_pnl = (mark_price - entry_price) * size if side == 'long' else (entry_price - mark_price) * size
        margin_used = (size * mark_price) / leverage
        
        self._position_risks[symbol] = PositionRisk(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            mark_price=mark_price,
            unrealized_pnl=unrealized_pnl,
            liquidation_price=liquidation_price,
            margin_used=margin_used,
            leverage=leverage,
        )
    
    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary"""
        if self._cached_state is None:
            return {}
        
        total_unrealized = sum(p.unrealized_pnl for p in self._position_risks.values())
        total_margin = sum(p.margin_used for p in self._position_risks.values())
        
        # Find riskiest positions
        sorted_positions = sorted(
            self._position_risks.values(),
            key=lambda p: abs(p.unrealized_pnl),
            reverse=True,
        )
        
        return {
            'equity': self._cached_state.equity,
            'margin_ratio': self._cached_state.margin_ratio,
            'available_balance': self._cached_state.available_balance,
            'total_unrealized_pnl': total_unrealized,
            'total_margin_used': total_margin,
            'is_at_risk': self._cached_state.is_at_liquidation_risk,
            'position_count': len(self._position_risks),
            'largest_loss_position': sorted_positions[0].symbol if sorted_positions else None,
            'last_sync_ns': self._last_sync_ns,
        }
    
    def get_max_safe_position_size(
        self,
        symbol: str,
        price: float,
        leverage: float,
    ) -> float:
        """Calculate maximum safe position size for a new order"""
        if self._cached_state is None:
            return 0.0
        
        # Account for safety buffer
        available = self._cached_state.available_balance * (1 - self._safety_buffer_pct)
        
        # Max notional based on available margin
        max_notional = available * leverage
        
        # Convert to quantity
        max_qty = max_notional / price
        
        return max(0.0, max_qty)
    
    def set_safety_thresholds(
        self,
        max_margin_ratio: float,
        safety_buffer_pct: float,
    ) -> None:
        """Configure safety thresholds"""
        self._max_margin_ratio = min(1.0, max(0.0, max_margin_ratio))
        self._safety_buffer_pct = min(0.5, max(0.0, safety_buffer_pct))
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Rust engine"""
        return self._is_connected
    
    @property
    def last_sync_age_ns(self) -> int:
        """Get age of last sync in nanoseconds"""
        import time
        if self._last_sync_ns == 0:
            return float('inf')
        return time.time_ns() - self._last_sync_ns


class NautilusPortfolioAdapter:
    """
    Adapter to integrate margin sync with Nautilus Portfolio.
    
    Intercepts order submissions and validates against Rust margin engine.
    """

    def __init__(self, margin_bridge: MarginSyncBridge):
        self.margin_bridge = margin_bridge
        self._blocked_orders: List[Dict] = []
    
    def validate_order(self, order_dict: Dict) -> tuple[bool, str]:
        """
        Validate an order before submission to Nautilus.
        
        Returns (is_valid, rejection_reason)
        """
        # Extract order parameters
        symbol = order_dict.get('symbol', '')
        side = order_dict.get('side', '')
        quantity = order_dict.get('quantity', 0)
        price = order_dict.get('price', 0)
        leverage = order_dict.get('leverage', 1)
        
        # Check margin
        result = self.margin_bridge.check_margin_before_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            leverage=leverage,
        )
        
        if not result.allowed:
            self._blocked_orders.append({
                'order': order_dict,
                'reason': result.reason,
                'timestamp': self.margin_bridge._last_sync_ns,
            })
            return False, result.reason
        
        return True, "OK"
    
    def get_blocked_orders_report(self) -> List[Dict]:
        """Get report of blocked orders"""
        return self._blocked_orders.copy()
