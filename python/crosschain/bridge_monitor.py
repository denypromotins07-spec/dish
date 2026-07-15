"""
Bridge monitor for cross-chain liquidity and message tracking.
Monitors LayerZero, Wormhole, Arbitrum delayed inbox for capital rotation detection.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class BridgeProtocol(Enum):
    """Supported bridge protocols."""
    LAYERZERO = "layerzero"
    WORMHOLE = "wormhole"
    ARBITRUM_DELAYED = "arbitrum_delayed"
    OPTIMISM = "optimism"
    POLYGON_POS = "polygon_pos"
    STARGATE = "stargate"
    SYNAPSE = "synapse"
    HOP = "hop"


class ChainId(Enum):
    """Common chain IDs."""
    ETHEREUM = 1
    ARBITRUM = 42161
    OPTIMISM = 10
    POLYGON = 137
    BSC = 56
    AVALANCHE = 43114
    BASE = 8453


@dataclass
class BridgeTransfer:
    """Represents a cross-chain bridge transfer."""
    tx_hash: str
    bridge_protocol: BridgeProtocol
    source_chain: ChainId
    destination_chain: ChainId
    token_symbol: str
    amount: float
    usd_value: float
    sender: str
    recipient: str
    timestamp: datetime
    status: str  # "pending", "completed", "failed"
    estimated_completion: Optional[datetime]


@dataclass
class LiquidityMovement:
    """Aggregated liquidity movement between chains."""
    source_chain: ChainId
    destination_chain: ChainId
    total_value_usd: float
    transfer_count: int
    time_window_start: datetime
    time_window_end: datetime
    net_flow_usd: float
    largest_transfer_usd: float


@dataclass
class BridgeAlert:
    """Alert for significant bridge activity."""
    alert_type: str
    severity: str
    message: str
    bridge_protocol: BridgeProtocol
    usd_value: float
    timestamp: datetime
    details: Dict


class BridgeMonitor:
    """
    Monitors major bridge contracts for large liquidity movements
    and cross-chain message passing to detect capital rotation.
    """

    def __init__(
        self,
        large_transfer_threshold_usd: float = 5_000_000,
        alert_channel=None,
    ):
        self.large_threshold = large_transfer_threshold_usd
        self.alert_channel = alert_channel
        
        # Transfer history (bounded)
        self._transfers: List[BridgeTransfer] = []
        self.max_history_size = 10000
        
        # Known bridge contract addresses
        self.bridge_contracts = {
            BridgeProtocol.LAYERZERO: {
                ChainId.ETHEREUM: "0x...",
                ChainId.ARBITRUM: "0x...",
            },
            BridgeProtocol.WORMHOLE: {
                ChainId.ETHEREUM: "0x...",
                ChainId.BSC: "0x...",
            },
            BridgeProtocol.ARBITRUM_DELAYED: {
                ChainId.ETHEREUM: "0x1234...",  # Delayed inbox
            },
            BridgeProtocol.STARGATE: {
                ChainId.ETHEREUM: "0x...",
                ChainId.ARBITRUM: "0x...",
            },
        }
        
        # Alert callbacks
        self._alert_callbacks = []
        
        # Statistics
        self._stats = {
            "total_transfers": 0,
            "total_volume_usd": 0,
            "by_protocol": {},
            "by_chain_pair": {},
        }

    async def process_bridge_transfer(
        self,
        tx_hash: str,
        bridge_protocol: BridgeProtocol,
        source_chain: ChainId,
        destination_chain: ChainId,
        token_symbol: str,
        amount: float,
        price_usd: float,
        sender: str,
        recipient: str,
        status: str = "pending",
        estimated_completion: Optional[datetime] = None,
    ) -> Optional[BridgeTransfer]:
        """Process a detected bridge transfer."""
        usd_value = amount * price_usd
        
        transfer = BridgeTransfer(
            tx_hash=tx_hash,
            bridge_protocol=bridge_protocol,
            source_chain=source_chain,
            destination_chain=destination_chain,
            token_symbol=token_symbol,
            amount=amount,
            usd_value=usd_value,
            sender=sender,
            recipient=recipient,
            timestamp=datetime.utcnow(),
            status=status,
            estimated_completion=estimated_completion,
        )
        
        # Add to history
        self._transfers.append(transfer)
        if len(self._transfers) > self.max_history_size:
            self._transfers = self._transfers[-self.max_history_size // 2:]
        
        # Update statistics
        self._update_stats(transfer)
        
        # Check for alerts
        await self._check_alerts(transfer)
        
        return transfer

    def _update_stats(self, transfer: BridgeTransfer):
        """Update internal statistics."""
        self._stats["total_transfers"] += 1
        self._stats["total_volume_usd"] += transfer.usd_value
        
        # By protocol
        proto_key = transfer.bridge_protocol.value
        if proto_key not in self._stats["by_protocol"]:
            self._stats["by_protocol"][proto_key] = {"count": 0, "volume": 0}
        self._stats["by_protocol"][proto_key]["count"] += 1
        self._stats["by_protocol"][proto_key]["volume"] += transfer.usd_value
        
        # By chain pair
        pair_key = f"{transfer.source_chain.name}->{transfer.destination_chain.name}"
        if pair_key not in self._stats["by_chain_pair"]:
            self._stats["by_chain_pair"][pair_key] = {"count": 0, "volume": 0}
        self._stats["by_chain_pair"][pair_key]["count"] += 1
        self._stats["by_chain_pair"][pair_key]["volume"] += transfer.usd_value

    async def _check_alerts(self, transfer: BridgeTransfer):
        """Check if transfer triggers any alerts."""
        alerts = []
        
        # Large transfer alert
        if transfer.usd_value >= self.large_threshold:
            alerts.append(BridgeAlert(
                alert_type="large_transfer",
                severity="high" if transfer.usd_value >= self.large_threshold * 2 else "medium",
                message=f"Large bridge transfer: ${transfer.usd_value:,.0f} from {transfer.source_chain.name} to {transfer.destination_chain.name}",
                bridge_protocol=transfer.bridge_protocol,
                usd_value=transfer.usd_value,
                timestamp=datetime.utcnow(),
                details={
                    "tx_hash": transfer.tx_hash,
                    "token": transfer.token_symbol,
                    "amount": transfer.amount,
                    "sender": transfer.sender,
                    "recipient": transfer.recipient,
                },
            ))
        
        # Unusual chain pair alert (would use ML in production)
        pair_key = f"{transfer.source_chain.name}->{transfer.destination_chain.name}"
        pair_stats = self._stats["by_chain_pair"].get(pair_key, {})
        if pair_stats.get("count", 0) == 1 and transfer.usd_value > 1_000_000:
            alerts.append(BridgeAlert(
                alert_type="unusual_route",
                severity="medium",
                message=f"First large transfer on route {pair_key}",
                bridge_protocol=transfer.bridge_protocol,
                usd_value=transfer.usd_value,
                timestamp=datetime.utcnow(),
                details={"route": pair_key},
            ))
        
        # Send alerts
        for alert in alerts:
            if self.alert_channel:
                await self.alert_channel.send(alert)
            
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception:
                    pass

    def get_liquidity_movements(
        self,
        window_minutes: int = 60,
        min_value_usd: float = 100_000,
    ) -> List[LiquidityMovement]:
        """Get aggregated liquidity movements by chain pair."""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        # Group by chain pair
        pairs: Dict[str, List[BridgeTransfer]] = {}
        for transfer in self._transfers:
            if transfer.timestamp < cutoff:
                continue
            if transfer.usd_value < min_value_usd:
                continue
            
            key = f"{transfer.source_chain.value}-{transfer.destination_chain.value}"
            if key not in pairs:
                pairs[key] = []
            pairs[key].append(transfer)
        
        # Aggregate
        movements = []
        for key, transfers in pairs.items():
            if not transfers:
                continue
            
            inflows = sum(t.usd_value for t in transfers)
            outflows = 0  # Would calculate reverse direction
            
            movements.append(LiquidityMovement(
                source_chain=transfers[0].source_chain,
                destination_chain=transfers[0].destination_chain,
                total_value_usd=inflows,
                transfer_count=len(transfers),
                time_window_start=min(t.timestamp for t in transfers),
                time_window_end=max(t.timestamp for t in transfers),
                net_flow_usd=inflows - outflows,
                largest_transfer_usd=max(t.usd_value for t in transfers),
            ))
        
        return sorted(movements, key=lambda m: m.total_value_usd, reverse=True)

    def get_capital_rotation_signal(self) -> Dict:
        """Detect capital rotation between ecosystems."""
        movements = self.get_liquidity_movements(window_minutes=120)
        
        if not movements:
            return {"signal": "none", "details": {}}
        
        # Analyze flow patterns
        ethereum_outflow = sum(
            m.total_value_usd for m in movements 
            if m.source_chain == ChainId.ETHEREUM
        )
        ethereum_inflow = sum(
            m.total_value_usd for m in movements 
            if m.destination_chain == ChainId.ETHEREUM
        )
        
        l2_total = sum(
            m.total_value_usd for m in movements 
            if m.destination_chain in [ChainId.ARBITRUM, ChainId.OPTIMISM, ChainId.BASE]
        )
        
        # Determine signal
        if ethereum_outflow > ethereum_inflow * 1.5 and l2_total > ethereum_outflow * 0.5:
            signal = "rotation_to_l2"
        elif ethereum_inflow > ethereum_outflow * 1.5:
            signal = "rotation_from_l2"
        else:
            signal = "neutral"
        
        return {
            "signal": signal,
            "confidence": min(1.0, len(movements) / 10),
            "details": {
                "ethereum_outflow_usd": ethereum_outflow,
                "ethereum_inflow_usd": ethereum_inflow,
                "l2_inflow_usd": l2_total,
                "movement_count": len(movements),
            },
        }

    def register_alert_callback(self, callback):
        """Register callback for bridge alerts."""
        self._alert_callbacks.append(callback)

    def get_statistics(self) -> Dict:
        """Get monitoring statistics."""
        return {
            **self._stats,
            "history_size": len(self._transfers),
            "max_history_size": self.max_history_size,
        }

    def get_pending_transfers(self) -> List[BridgeTransfer]:
        """Get all pending transfers."""
        return [t for t in self._transfers if t.status == "pending"]

    def update_transfer_status(self, tx_hash: str, new_status: str):
        """Update the status of a transfer."""
        for transfer in self._transfers:
            if transfer.tx_hash == tx_hash:
                transfer.status = new_status
                break


async def main():
    """Example usage of BridgeMonitor."""
    monitor = BridgeMonitor(large_transfer_threshold_usd=1_000_000)
    
    # Simulate some bridge transfers
    await monitor.process_bridge_transfer(
        tx_hash="0xabc123",
        bridge_protocol=BridgeProtocol.LAYERZERO,
        source_chain=ChainId.ETHEREUM,
        destination_chain=ChainId.ARBITRUM,
        token_symbol="USDC",
        amount=5_000_000,
        price_usd=1.0,
        sender="0xuser1",
        recipient="0xuser2",
        status="completed",
    )
    
    await monitor.process_bridge_transfer(
        tx_hash="0xdef456",
        bridge_protocol=BridgeProtocol.STARGATE,
        source_chain=ChainId.ETHEREUM,
        destination_chain=ChainId.OPTIMISM,
        token_symbol="ETH",
        amount=2000,
        price_usd=2000,
        sender="0xuser3",
        recipient="0xuser4",
        status="pending",
    )
    
    # Get liquidity movements
    movements = monitor.get_liquidity_movements()
    print("Liquidity Movements:")
    for m in movements:
        print(f"  {m.source_chain.name} -> {m.destination_chain.name}: ${m.total_value_usd:,.0f}")
    
    # Capital rotation signal
    signal = monitor.get_capital_rotation_signal()
    print(f"\nCapital Rotation Signal: {signal['signal']}")
    print(f"  Confidence: {signal['confidence']:.2f}")
    print(f"  Details: {signal['details']}")
    
    # Statistics
    stats = monitor.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
