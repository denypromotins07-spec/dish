"""
Asynchronous whale tracker for large on-chain transfers.
Classifies wallets as Exchanges, OTC desks, or Institutional Whales.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
import hashlib


class WalletType(Enum):
    """Classification of wallet types based on heuristics."""
    UNKNOWN = "unknown"
    EXCHANGE = "exchange"
    OTC_DESK = "otc_desk"
    INSTITUTIONAL_WHALE = "institutional_whale"
    RETAIL_WHALE = "retail_whale"
    CONTRACT = "contract"
    MEV_BOT = "mev_bot"


@dataclass
class WhaleAlert:
    """Alert generated when significant on-chain activity detected."""
    tx_hash: str
    from_address: str
    to_address: str
    amount: float
    token_symbol: str
    usd_value: float
    timestamp: datetime
    wallet_type_from: WalletType
    wallet_type_to: WalletType
    confidence_score: float
    alert_type: str  # "large_transfer", "exchange_inflow", "exchange_outflow", "whale_accumulation"


@dataclass
class WalletProfile:
    """Profile information for a tracked wallet."""
    address: str
    wallet_type: WalletType = WalletType.UNKNOWN
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_transactions: int = 0
    total_volume_usd: float = 0.0
    avg_transaction_size_usd: float = 0.0
    counterparties: Set[str] = field(default_factory=set)
    tags: List[str] = field(default_factory=list)
    risk_score: float = 0.0


class WhaleTracker:
    """
    Asynchronous tracker for large on-chain transfers.
    Applies heuristic tagging to classify wallets and push alerts.
    """

    def __init__(
        self,
        min_transfer_threshold_usd: float = 100_000.0,
        exchange_addresses: Optional[Set[str]] = None,
        otc_addresses: Optional[Set[str]] = None,
        alert_channel=None,
    ):
        self.min_transfer_threshold_usd = min_transfer_threshold_usd
        self.exchange_addresses = exchange_addresses or set()
        self.otc_addresses = otc_addresses or set()
        self.alert_channel = alert_channel
        
        # In-memory wallet profiles (production would use Redis/DB)
        self.wallet_profiles: Dict[str, WalletProfile] = {}
        self.recent_alerts: List[WhaleAlert] = []
        self._lock = asyncio.Lock()
        
        # Known exchange labels (would be loaded from configuration)
        self.known_exchanges = {
            "binance": {"hot_wallet", "cold_wallet", "deposit"},
            "coinbase": {"custody", "prime"},
            "kraken": {"trading", "staking"},
        }

    async def process_transfer(
        self,
        tx_hash: str,
        from_address: str,
        to_address: str,
        amount: float,
        token_symbol: str,
        price_usd: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[WhaleAlert]:
        """Process a single transfer and generate alert if threshold exceeded."""
        timestamp = timestamp or datetime.utcnow()
        usd_value = amount * price_usd
        
        # Skip if below threshold
        if usd_value < self.min_transfer_threshold_usd:
            return None
        
        async with self._lock:
            # Update wallet profiles
            await self._update_wallet_profile(from_address, timestamp, usd_value, to_address)
            await self._update_wallet_profile(to_address, timestamp, usd_value, from_address)
            
            # Classify wallets
            wallet_type_from = self._classify_wallet(from_address)
            wallet_type_to = self._classify_wallet(to_address)
            
            # Determine alert type
            alert_type = self._determine_alert_type(
                wallet_type_from, wallet_type_to, usd_value
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                wallet_type_from, wallet_type_to, usd_value
            )
            
            # Create alert
            alert = WhaleAlert(
                tx_hash=tx_hash,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                token_symbol=token_symbol,
                usd_value=usd_value,
                timestamp=timestamp,
                wallet_type_from=wallet_type_from,
                wallet_type_to=wallet_type_to,
                confidence_score=confidence,
                alert_type=alert_type,
            )
            
            self.recent_alerts.append(alert)
            
            # Keep only recent alerts
            if len(self.recent_alerts) > 1000:
                self.recent_alerts = self.recent_alerts[-500:]
        
        # Push to alert channel if configured
        if self.alert_channel:
            await self.alert_channel.send(alert)
        
        return alert

    async def _update_wallet_profile(
        self,
        address: str,
        timestamp: datetime,
        volume_usd: float,
        counterparty: str,
    ):
        """Update or create wallet profile."""
        if address not in self.wallet_profiles:
            self.wallet_profiles[address] = WalletProfile(
                address=address,
                first_seen=timestamp,
            )
        
        profile = self.wallet_profiles[address]
        profile.last_seen = timestamp
        profile.total_transactions += 1
        profile.total_volume_usd += volume_usd
        profile.avg_transaction_size_usd = (
            profile.total_volume_usd / profile.total_transactions
        )
        profile.counterparties.add(counterparty)

    def _classify_wallet(self, address: str) -> WalletType:
        """Classify wallet based on heuristics and known addresses."""
        # Check known exchange addresses
        if address.lower() in {a.lower() for a in self.exchange_addresses}:
            return WalletType.EXCHANGE
        
        # Check known OTC desks
        if address.lower() in {a.lower() for a in self.otc_addresses}:
            return WalletType.OTC_DESK
        
        # Check if we have existing profile
        if address in self.wallet_profiles:
            profile = self.wallet_profiles[address]
            
            # High transaction count with many counterparties suggests exchange
            if profile.total_transactions > 10000 and len(profile.counterparties) > 1000:
                return WalletType.EXCHANGE
            
            # Large average transaction size suggests institutional
            if profile.avg_transaction_size_usd > 1_000_000:
                return WalletType.INSTITUTIONAL_WHALE
            
            # Contract detection (simplified - would check bytecode in production)
            if address.startswith("0x") and len(address) == 42:
                # Heuristic: contracts often have specific patterns
                hash_digest = hashlib.sha256(address.encode()).hexdigest()
                if hash_digest[:2] in {"00", "01", "02"}:
                    return WalletType.CONTRACT
        
        return WalletType.UNKNOWN

    def _determine_alert_type(
        self,
        from_type: WalletType,
        to_type: WalletType,
        usd_value: float,
    ) -> str:
        """Determine the type of alert based on wallet classifications."""
        if from_type == WalletType.EXCHANGE and to_type != WalletType.EXCHANGE:
            return "exchange_outflow"
        elif to_type == WalletType.EXCHANGE and from_type != WalletType.EXCHANGE:
            return "exchange_inflow"
        elif from_type == WalletType.EXCHANGE and to_type == WalletType.EXCHANGE:
            return "exchange_internal"
        elif (
            from_type in {WalletType.INSTITUTIONAL_WHALE, WalletType.RETAIL_WHALE}
            and to_type == WalletType.EXCHANGE
        ):
            return "whale_distribution"
        elif (
            to_type in {WalletType.INSTITUTIONAL_WHALE, WalletType.RETAIL_WHALE}
            and from_type == WalletType.EXCHANGE
        ):
            return "whale_accumulation"
        elif usd_value > 10_000_000:
            return "mega_transfer"
        else:
            return "large_transfer"

    def _calculate_confidence(
        self,
        from_type: WalletType,
        to_type: WalletType,
        usd_value: float,
    ) -> float:
        """Calculate confidence score for the classification."""
        base_confidence = 0.5
        
        # Higher confidence for known addresses
        if from_type == WalletType.EXCHANGE or to_type == WalletType.EXCHANGE:
            base_confidence += 0.3
        
        # Higher confidence for larger transfers
        if usd_value > 5_000_000:
            base_confidence += 0.1
        if usd_value > 50_000_000:
            base_confidence += 0.1
        
        # Cap at 1.0
        return min(base_confidence, 1.0)

    async def add_known_exchange(self, address: str, label: str = ""):
        """Add a known exchange address."""
        async with self._lock:
            self.exchange_addresses.add(address.lower())
            if label:
                if address in self.wallet_profiles:
                    self.wallet_profiles[address].tags.append(f"exchange:{label}")
                else:
                    self.wallet_profiles[address] = WalletProfile(
                        address=address,
                        wallet_type=WalletType.EXCHANGE,
                        tags=[f"exchange:{label}"],
                    )

    async def add_known_otc(self, address: str, label: str = ""):
        """Add a known OTC desk address."""
        async with self._lock:
            self.otc_addresses.add(address.lower())
            if label:
                if address in self.wallet_profiles:
                    self.wallet_profiles[address].tags.append(f"otc:{label}")
                else:
                    self.wallet_profiles[address] = WalletProfile(
                        address=address,
                        wallet_type=WalletType.OTC_DESK,
                        tags=[f"otc:{label}"],
                    )

    def get_wallet_profile(self, address: str) -> Optional[WalletProfile]:
        """Get profile for a specific wallet."""
        return self.wallet_profiles.get(address.lower())

    def get_recent_alerts(
        self,
        limit: int = 50,
        alert_type: Optional[str] = None,
    ) -> List[WhaleAlert]:
        """Get recent alerts, optionally filtered by type."""
        alerts = self.recent_alerts[-limit:]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        return alerts

    def get_statistics(self) -> Dict:
        """Get tracker statistics."""
        wallet_types = {}
        for profile in self.wallet_profiles.values():
            wt = profile.wallet_type.value
            wallet_types[wt] = wallet_types.get(wt, 0) + 1
        
        return {
            "total_wallets_tracked": len(self.wallet_profiles),
            "wallet_type_distribution": wallet_types,
            "total_alerts_generated": len(self.recent_alerts),
            "known_exchanges": len(self.exchange_addresses),
            "known_otc_desks": len(self.otc_addresses),
        }


async def main():
    """Example usage of WhaleTracker."""
    tracker = WhaleTracker(min_transfer_threshold_usd=500_000)
    
    # Add some known exchange addresses
    await tracker.add_known_exchange("0x28C6c06298d514Db089934071355E5743bf21d60", "binance_hot")
    await tracker.add_known_exchange("0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549", "binance_cold")
    
    # Simulate a large transfer
    alert = await tracker.process_transfer(
        tx_hash="0xabc123...",
        from_address="0x28C6c06298d514Db089934071355E5743bf21d60",
        to_address="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        amount=5000.0,
        token_symbol="ETH",
        price_usd=2000.0,
    )
    
    if alert:
        print(f"Alert: {alert.alert_type}")
        print(f"  From: {alert.from_address} ({alert.wallet_type_from.value})")
        print(f"  To: {alert.to_address} ({alert.wallet_type_to.value})")
        print(f"  Value: ${alert.usd_value:,.2f}")
        print(f"  Confidence: {alert.confidence_score:.2f}")
    
    stats = tracker.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
