"""
Real-time calculator for Exchange Inflow/Outflow deltas and Stablecoin metrics.
Normalizes low-frequency on-chain events into high-frequency tick timeline.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import statistics


class ExchangeType(Enum):
    """Types of exchanges tracked."""
    CEX_SPOT = "cex_spot"
    CEX_FUTURES = "cex_futures"
    DEX = "dex"
    BRIDGE = "bridge"


@dataclass
class ExchangeFlowEvent:
    """Single exchange flow event."""
    timestamp: datetime
    exchange_id: str
    exchange_type: ExchangeType
    token_symbol: str
    amount: float  # Positive for inflow, negative for outflow
    usd_value: float
    tx_hash: str
    from_address: str
    to_address: str
    is_deposit: bool  # True if depositing to exchange


@dataclass
class StablecoinEvent:
    """Stablecoin mint/burn event."""
    timestamp: datetime
    token_symbol: str  # USDT, USDC, DAI, etc.
    event_type: str  # "mint" or "burn"
    amount: float
    usd_value: float
    tx_hash: str
    protocol: str  # e.g., "tether", "circle", "makerdao"


@dataclass
class FlowMetrics:
    """Aggregated flow metrics for a time window."""
    window_start: datetime
    window_end: datetime
    total_inflow_usd: float = 0.0
    total_outflow_usd: float = 0.0
    net_flow_usd: float = 0.0
    inflow_count: int = 0
    outflow_count: int = 0
    largest_inflow_usd: float = 0.0
    largest_outflow_usd: float = 0.0
    avg_inflow_size_usd: float = 0.0
    avg_outflow_size_usd: float = 0.0


@dataclass
class StablecoinMetrics:
    """Aggregated stablecoin metrics."""
    token_symbol: str
    total_minted_usd: float = 0.0
    total_burned_usd: float = 0.0
    net_supply_change_usd: float = 0.0
    mint_count: int = 0
    burn_count: int = 0
    current_supply_estimate_usd: float = 0.0


class ExchangeFlowCalculator:
    """
    Real-time calculator for exchange flows and stablecoin metrics.
    Normalizes events into configurable time windows for analysis.
    """

    def __init__(
        self,
        window_size_seconds: int = 3600,  # 1 hour default
        known_exchanges: Optional[Dict[str, ExchangeType]] = None,
        stablecoin_tokens: Optional[List[str]] = None,
    ):
        self.window_size = timedelta(seconds=window_size_seconds)
        self.known_exchanges = known_exchanges or {}
        self.stablecoin_tokens = stablecoin_tokens or ["USDT", "USDC", "DAI", "BUSD"]
        
        # Event buffers (production would use Redis/time-series DB)
        self.flow_events: List[ExchangeFlowEvent] = []
        self.stablecoin_events: List[StablecoinEvent] = []
        
        # Aggregated metrics by window
        self.flow_metrics: Dict[Tuple[str, datetime], FlowMetrics] = {}
        self.stablecoin_metrics: Dict[str, StablecoinMetrics] = {}
        
        # Running totals per exchange
        self.exchange_totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Callbacks for significant events
        self.significant_flow_callbacks = []
        self.significant_threshold_usd = 10_000_000  # $10M

    async def record_flow(
        self,
        timestamp: datetime,
        exchange_id: str,
        token_symbol: str,
        amount: float,
        price_usd: float,
        tx_hash: str,
        from_address: str,
        to_address: str,
        is_deposit: bool,
    ) -> Optional[ExchangeFlowEvent]:
        """Record an exchange flow event."""
        usd_value = amount * price_usd
        
        event = ExchangeFlowEvent(
            timestamp=timestamp,
            exchange_id=exchange_id,
            exchange_type=self.known_exchanges.get(
                exchange_id, ExchangeType.CEX_SPOT
            ),
            token_symbol=token_symbol,
            amount=amount if is_deposit else -amount,
            usd_value=usd_value,
            tx_hash=tx_hash,
            from_address=from_address,
            to_address=to_address,
            is_deposit=is_deposit,
        )
        
        async with self._lock:
            self.flow_events.append(event)
            
            # Update exchange totals
            direction = "inflow" if is_deposit else "outflow"
            self.exchange_totals[exchange_id][f"{token_symbol}_{direction}"] += amount
            self.exchange_totals[exchange_id]["total_usd"] += usd_value
            
            # Update aggregated metrics
            await self._update_flow_metrics(event)
            
            # Check for significant flows
            if usd_value >= self.significant_threshold_usd:
                await self._notify_significant_flow(event)
        
        return event

    async def record_stablecoin_event(
        self,
        timestamp: datetime,
        token_symbol: str,
        event_type: str,
        amount: float,
        price_usd: float,
        tx_hash: str,
        protocol: str,
    ) -> Optional[StablecoinEvent]:
        """Record a stablecoin mint/burn event."""
        if token_symbol not in self.stablecoin_tokens:
            return None
        
        usd_value = amount * price_usd
        
        event = StablecoinEvent(
            timestamp=timestamp,
            token_symbol=token_symbol,
            event_type=event_type,
            amount=amount,
            usd_value=usd_value,
            tx_hash=tx_hash,
            protocol=protocol,
        )
        
        async with self._lock:
            self.stablecoin_events.append(event)
            await self._update_stablecoin_metrics(event)
        
        return event

    async def _update_flow_metrics(self, event: ExchangeFlowEvent):
        """Update aggregated flow metrics for the event's time window."""
        window_start = self._get_window_start(event.timestamp)
        key = (event.exchange_id, window_start)
        
        if key not in self.flow_metrics:
            self.flow_metrics[key] = FlowMetrics(
                window_start=window_start,
                window_end=window_start + self.window_size,
            )
        
        metrics = self.flow_metrics[key]
        
        if event.is_deposit:
            metrics.total_inflow_usd += event.usd_value
            metrics.inflow_count += 1
            metrics.largest_inflow_usd = max(
                metrics.largest_inflow_usd, event.usd_value
            )
        else:
            metrics.total_outflow_usd += event.usd_value
            metrics.outflow_count += 1
            metrics.largest_outflow_usd = max(
                metrics.largest_outflow_usd, event.usd_value
            )
        
        metrics.net_flow_usd = metrics.total_inflow_usd - metrics.total_outflow_usd
        
        if metrics.inflow_count > 0:
            metrics.avg_inflow_size_usd = (
                metrics.total_inflow_usd / metrics.inflow_count
            )
        if metrics.outflow_count > 0:
            metrics.avg_outflow_size_usd = (
                metrics.total_outflow_usd / metrics.outflow_count
            )

    async def _update_stablecoin_metrics(self, event: StablecoinEvent):
        """Update aggregated stablecoin metrics."""
        if event.token_symbol not in self.stablecoin_metrics:
            self.stablecoin_metrics[event.token_symbol] = StablecoinMetrics(
                token_symbol=event.token_symbol
            )
        
        metrics = self.stablecoin_metrics[event.token_symbol]
        
        if event.event_type == "mint":
            metrics.total_minted_usd += event.usd_value
            metrics.mint_count += 1
            metrics.net_supply_change_usd += event.usd_value
        elif event.event_type == "burn":
            metrics.total_burned_usd += event.usd_value
            metrics.burn_count += 1
            metrics.net_supply_change_usd -= event.usd_value
        
        # Update running supply estimate
        metrics.current_supply_estimate_usd += (
            event.usd_value if event.event_type == "mint" else -event.usd_value
        )

    async def _notify_significant_flow(self, event: ExchangeFlowEvent):
        """Notify callbacks of significant flows."""
        for callback in self.significant_flow_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                # Log but don't fail the main operation
                pass

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """Get the start of the time window containing the timestamp."""
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (timestamp - epoch).total_seconds()
        window_seconds = self.window_size.total_seconds()
        window_number = int(seconds_since_epoch // window_seconds)
        return epoch + timedelta(seconds=window_number * window_seconds)

    def get_exchange_net_flow(
        self,
        exchange_id: str,
        lookback_hours: int = 24,
    ) -> float:
        """Get net flow for an exchange over the specified period."""
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        total_net = 0.0
        for (ex_id, window_start), metrics in self.flow_metrics.items():
            if ex_id == exchange_id and window_start >= cutoff:
                total_net += metrics.net_flow_usd
        
        return total_net

    def get_all_exchanges_net_flow(
        self,
        lookback_hours: int = 24,
    ) -> Dict[str, float]:
        """Get net flows for all tracked exchanges."""
        result = {}
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        for (ex_id, window_start), metrics in self.flow_metrics.items():
            if window_start >= cutoff:
                if ex_id not in result:
                    result[ex_id] = 0.0
                result[ex_id] += metrics.net_flow_usd
        
        return result

    def get_stablecoin_net_supply_change(
        self,
        token_symbol: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get net supply change for stablecoins."""
        if token_symbol:
            if token_symbol in self.stablecoin_metrics:
                return {
                    token_symbol: self.stablecoin_metrics[
                        token_symbol
                    ].net_supply_change_usd
                }
            return {token_symbol: 0.0}
        
        return {
            symbol: metrics.net_supply_change_usd
            for symbol, metrics in self.stablecoin_metrics.items()
        }

    def get_flow_trend(
        self,
        exchange_id: Optional[str] = None,
        window_count: int = 24,
    ) -> Dict:
        """
        Analyze flow trends over recent windows.
        Returns trend direction, strength, and volatility.
        """
        cutoff_idx = len(self.flow_metrics) - window_count
        
        net_flows = []
        for idx, ((ex_id, _), metrics) in enumerate(
            sorted(self.flow_metrics.items(), key=lambda x: x[0][1])
        ):
            if idx < cutoff_idx:
                continue
            if exchange_id and ex_id != exchange_id:
                continue
            net_flows.append(metrics.net_flow_usd)
        
        if len(net_flows) < 2:
            return {
                "trend": "insufficient_data",
                "strength": 0.0,
                "volatility": 0.0,
                "avg_net_flow_usd": 0.0,
            }
        
        # Calculate trend using linear regression slope
        x_mean = (len(net_flows) - 1) / 2
        y_mean = statistics.mean(net_flows)
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(net_flows))
        denominator = sum((i - x_mean) ** 2 for i in range(len(net_flows)))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine trend direction
        if slope > y_mean * 0.1:
            trend = "increasing_inflows"
        elif slope < -y_mean * 0.1:
            trend = "increasing_outflows"
        else:
            trend = "neutral"
        
        # Volatility as standard deviation
        volatility = statistics.stdev(net_flows) if len(net_flows) > 1 else 0
        
        return {
            "trend": trend,
            "strength": abs(slope),
            "volatility": volatility,
            "avg_net_flow_usd": y_mean,
            "sample_count": len(net_flows),
        }

    def register_significant_flow_callback(self, callback):
        """Register callback for significant flow notifications."""
        self.significant_flow_callbacks.append(callback)

    def set_significant_threshold(self, threshold_usd: float):
        """Set the threshold for significant flow alerts."""
        self.significant_threshold_usd = threshold_usd

    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        return {
            "total_flow_events": len(self.flow_events),
            "total_stablecoin_events": len(self.stablecoin_events),
            "tracked_exchanges": len(self.known_exchanges),
            "stablecoin_tokens": self.stablecoin_tokens,
            "window_size_hours": self.window_size.total_seconds() / 3600,
        }


async def main():
    """Example usage of ExchangeFlowCalculator."""
    calculator = ExchangeFlowCalculator(window_size_seconds=3600)
    
    # Add known exchanges
    calculator.known_exchanges = {
        "binance": ExchangeType.CEX_SPOT,
        "coinbase": ExchangeType.CEX_SPOT,
        "uniswap_v3": ExchangeType.DEX,
    }
    
    # Record some flows
    now = datetime.utcnow()
    
    await calculator.record_flow(
        timestamp=now,
        exchange_id="binance",
        token_symbol="BTC",
        amount=100.0,
        price_usd=30000.0,
        tx_hash="0xabc123",
        from_address="0xuser1",
        to_address="0xbinance",
        is_deposit=True,
    )
    
    await calculator.record_flow(
        timestamp=now,
        exchange_id="binance",
        token_symbol="ETH",
        amount=500.0,
        price_usd=2000.0,
        tx_hash="0xdef456",
        from_address="0xbinance",
        to_address="0xuser2",
        is_deposit=False,
    )
    
    # Record stablecoin mint
    await calculator.record_stablecoin_event(
        timestamp=now,
        token_symbol="USDT",
        event_type="mint",
        amount=100_000_000,
        price_usd=1.0,
        tx_hash="0xghi789",
        protocol="tether",
    )
    
    # Get results
    print("Exchange Net Flows (24h):")
    for ex, flow in calculator.get_all_exchanges_net_flow().items():
        print(f"  {ex}: ${flow:,.2f}")
    
    print("\nStablecoin Supply Changes:")
    for token, change in calculator.get_stablecoin_net_supply_change().items():
        print(f"  {token}: ${change:,.2f}")
    
    print("\nFlow Trend:")
    trend = calculator.get_flow_trend()
    print(f"  Trend: {trend['trend']}")
    print(f"  Avg Net Flow: ${trend['avg_net_flow_usd']:,.2f}")
    
    print("\nStatistics:", calculator.get_statistics())


if __name__ == "__main__":
    asyncio.run(main())
