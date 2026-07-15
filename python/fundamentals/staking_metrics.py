"""
ETH staking metrics tracker for network security and LSD peg analysis.
Monitors validator queue, staking rates, and liquid staking derivative ratios.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class StakingMetricType(Enum):
    """Types of staking metrics."""
    VALIDATOR_COUNT = "validator_count"
    TOTAL_STAKED = "total_staked"
    STAKING_RATE = "staking_rate"
    QUEUE_LENGTH = "queue_length"
    APR = "apr"
    LSD_PEG = "lsd_peg"


@dataclass
class ValidatorMetrics:
    """Ethereum validator statistics."""
    total_validators: int
    active_validators: int
    pending_validators: int
    exiting_validators: int
    slashed_validators: int
    timestamp: datetime


@dataclass
class QueueMetrics:
    """Validator queue statistics."""
    activation_queue_length: int
    exit_queue_length: int
    estimated_activation_days: float
    estimated_exit_days: float
    queue_churn_limit: int


@dataclass
class LSDPegData:
    """Liquid Staking Derivative peg information."""
    token_symbol: str  # e.g., "stETH", "rETH", "cbETH"
    underlying_token: str  # e.g., "ETH"
    price_ratio: float  # lsd_price / eth_price
    peg_deviation_percent: float
    market_cap_usd: float
    total_supply: float
    timestamp: datetime


@dataclass
class StakingOverview:
    """Complete staking overview snapshot."""
    total_eth_staked: float
    staking_rate_percent: float
    current_apr: float
    validator_count: int
    queue_status: str  # "normal", "congested", "emptying"
    risk_level: str  # "low", "medium", "high"
    timestamp: datetime


class StakingMetricsTracker:
    """
    Tracks ETH staking rates, validator queues, and LSD pegs.
    Gauges network security and liquidation risks.
    """

    def __init__(self):
        # Historical data storage (limited size)
        self._validator_history: List[ValidatorMetrics] = []
        self._queue_history: List[QueueMetrics] = []
        self._lsd_pegs: Dict[str, List[LSDPegData]] = {}
        
        # Known LSD tokens to track
        self.tracked_lsds = [
            {"symbol": "stETH", "name": "Lido Staked ETH", "underlying": "ETH"},
            {"symbol": "rETH", "name": "Rocket Pool ETH", "underlying": "ETH"},
            {"symbol": "cbETH", "name": "Coinbase Wrapped Staked ETH", "underlying": "ETH"},
            {"symbol": "wstETH", "name": "Wrapped Lido Staked ETH", "underlying": "ETH"},
            {"symbol": "frxETH", "name": "Frax Ether", "underlying": "ETH"},
            {"symbol": "sfrxETH", "name": "Staked Frax Ether", "underlying": "ETH"},
        ]
        
        # Thresholds for risk assessment
        self.risk_thresholds = {
            "staking_rate_high": 30.0,  # % of total supply
            "staking_rate_critical": 40.0,
            "queue_days_normal": 5,
            "queue_days_congested": 30,
            "peg_deviation_warning": 1.0,  # %
            "peg_deviation_critical": 3.0,
        }
        
        # Mock current state (would be fetched from beacon chain API)
        self._current_state = {
            "total_validators": 800_000,
            "active_validators": 780_000,
            "pending_validators": 15_000,
            "exiting_validators": 5_000,
            "slashed_validators": 150,
            "total_eth_staked": 25_000_000,
            "eth_total_supply": 120_000_000,
            "current_apr": 4.2,
        }

    async def fetch_validator_metrics(self) -> ValidatorMetrics:
        """Fetch current validator statistics from beacon chain."""
        # In production, this would query:
        # - Beaconcha.in API
        # - Ethereum Consensus Layer RPC
        # - Prysm/Lighthouse client APIs
        
        metrics = ValidatorMetrics(
            total_validators=self._current_state["total_validators"],
            active_validators=self._current_state["active_validators"],
            pending_validators=self._current_state["pending_validators"],
            exiting_validators=self._current_state["exiting_validators"],
            slashed_validators=self._current_state["slashed_validators"],
            timestamp=datetime.utcnow(),
        )
        
        self._validator_history.append(metrics)
        if len(self._validator_history) > 1000:
            self._validator_history = self._validator_history[-500:]
        
        return metrics

    async def fetch_queue_metrics(self) -> QueueMetrics:
        """Fetch validator queue statistics."""
        # Mock calculation based on current state
        churn_limit = max(4, self._current_state["active_validators"] // 65536)
        
        activation_days = (
            self._current_state["pending_validators"] / churn_limit 
            if churn_limit > 0 else 0
        ) / 225  # ~225 epochs per day
        
        exit_days = (
            self._current_state["exiting_validators"] / churn_limit 
            if churn_limit > 0 else 0
        ) / 225
        
        metrics = QueueMetrics(
            activation_queue_length=self._current_state["pending_validators"],
            exit_queue_length=self._current_state["exiting_validators"],
            estimated_activation_days=activation_days,
            estimated_exit_days=exit_days,
            queue_churn_limit=churn_limit,
        )
        
        self._queue_history.append(metrics)
        if len(self._queue_history) > 1000:
            self._queue_history = self._queue_history[-500:]
        
        return metrics

    async def fetch_lsd_peg(
        self,
        token_symbol: str,
        lsd_price_usd: float,
        eth_price_usd: float,
        market_cap_usd: float,
        total_supply: float,
    ) -> LSDPegData:
        """Fetch LSD peg data for a specific token."""
        if eth_price_usd == 0:
            eth_price_usd = 2000  # Default fallback
        
        price_ratio = lsd_price_usd / eth_price_usd
        peg_deviation = (price_ratio - 1.0) * 100
        
        peg_data = LSDPegData(
            token_symbol=token_symbol,
            underlying_token="ETH",
            price_ratio=price_ratio,
            peg_deviation_percent=peg_deviation,
            market_cap_usd=market_cap_usd,
            total_supply=total_supply,
            timestamp=datetime.utcnow(),
        )
        
        if token_symbol not in self._lsd_pegs:
            self._lsd_pegs[token_symbol] = []
        
        self._lsd_pegs[token_symbol].append(peg_data)
        if len(self._lsd_pegs[token_symbol]) > 1000:
            self._lsd_pegs[token_symbol] = self._lsd_pegs[token_symbol][-500:]
        
        return peg_data

    async def get_staking_overview(self) -> StakingOverview:
        """Get comprehensive staking overview."""
        validators = await self.fetch_validator_metrics()
        queue = await self.fetch_queue_metrics()
        
        total_supply = self._current_state["eth_total_supply"]
        total_staked = self._current_state["total_eth_staked"]
        staking_rate = (total_staked / total_supply) * 100
        
        # Determine queue status
        if queue.estimated_activation_days < self.risk_thresholds["queue_days_normal"]:
            queue_status = "normal"
        elif queue.estimated_activation_days < self.risk_thresholds["queue_days_congested"]:
            queue_status = "congested"
        else:
            queue_status = "severely_congested"
        
        # Assess overall risk level
        risk_level = self._assess_risk(staking_rate, queue, validators)
        
        return StakingOverview(
            total_eth_staked=total_staked,
            staking_rate_percent=staking_rate,
            current_apr=self._current_state["current_apr"],
            validator_count=validators.active_validators,
            queue_status=queue_status,
            risk_level=risk_level,
            timestamp=datetime.utcnow(),
        )

    def _assess_risk(
        self,
        staking_rate: float,
        queue: QueueMetrics,
        validators: ValidatorMetrics,
    ) -> str:
        """Assess overall staking risk level."""
        risk_score = 0
        
        # High staking rate concentration risk
        if staking_rate >= self.risk_thresholds["staking_rate_critical"]:
            risk_score += 3
        elif staking_rate >= self.risk_thresholds["staking_rate_high"]:
            risk_score += 2
        
        # Queue congestion risk
        if queue.estimated_activation_days > self.risk_thresholds["queue_days_congested"]:
            risk_score += 2
        elif queue.estimated_activation_days > self.risk_thresholds["queue_days_normal"]:
            risk_score += 1
        
        # Slashing risk
        slashing_rate = (validators.slashed_validators / validators.total_validators) * 100
        if slashing_rate > 0.1:
            risk_score += 2
        elif slashing_rate > 0.01:
            risk_score += 1
        
        # Map score to risk level
        if risk_score >= 5:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"

    async def get_lsd_peg_alerts(self) -> List[Dict]:
        """Get alerts for significant LSD peg deviations."""
        alerts = []
        
        for token_symbol, history in self._lsd_pegs.items():
            if not history:
                continue
            
            latest = history[-1]
            
            if abs(latest.peg_deviation_percent) >= self.risk_thresholds["peg_deviation_critical"]:
                alerts.append({
                    "severity": "critical",
                    "token": token_symbol,
                    "deviation": latest.peg_deviation_percent,
                    "message": f"{token_symbol} peg deviation critical: {latest.peg_deviation_percent:.2f}%",
                })
            elif abs(latest.peg_deviation_percent) >= self.risk_thresholds["peg_deviation_warning"]:
                alerts.append({
                    "severity": "warning",
                    "token": token_symbol,
                    "deviation": latest.peg_deviation_percent,
                    "message": f"{token_symbol} peg deviation warning: {latest.peg_deviation_percent:.2f}%",
                })
        
        return alerts

    def get_historical_trend(
        self,
        metric_type: StakingMetricType,
        days: int = 30,
    ) -> List[Dict]:
        """Get historical trend data for a specific metric."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        if metric_type == StakingMetricType.VALIDATOR_COUNT:
            return [
                {"timestamp": m.timestamp, "value": m.total_validators}
                for m in self._validator_history
                if m.timestamp >= cutoff
            ]
        elif metric_type == StakingMetricType.QUEUE_LENGTH:
            return [
                {"timestamp": m.timestamp, "value": m.activation_queue_length}
                for m in self._queue_history
                if m.timestamp >= cutoff
            ]
        elif metric_type == StakingMetricType.LSD_PEG:
            result = {}
            for token, history in self._lsd_pegs.items():
                result[token] = [
                    {"timestamp": d.timestamp, "value": d.price_ratio}
                    for d in history
                    if d.timestamp >= cutoff
                ]
            return result
        
        return []

    def get_statistics(self) -> Dict:
        """Get tracker statistics."""
        return {
            "validator_samples": len(self._validator_history),
            "queue_samples": len(self._queue_history),
            "lsd_tokens_tracked": len(self._lsd_pegs),
            "tracked_lsd_symbols": list(self._lsd_pegs.keys()),
        }


async def main():
    """Example usage of StakingMetricsTracker."""
    tracker = StakingMetricsTracker()
    
    # Get staking overview
    overview = await tracker.get_staking_overview()
    print("Staking Overview:")
    print(f"  Total ETH Staked: {overview.total_eth_staked:,.0f}")
    print(f"  Staking Rate: {overview.staking_rate_percent:.2f}%")
    print(f"  Current APR: {overview.current_apr:.2f}%")
    print(f"  Active Validators: {overview.validator_count:,}")
    print(f"  Queue Status: {overview.queue_status}")
    print(f"  Risk Level: {overview.risk_level}")
    
    # Fetch LSD pegs (mock data)
    print("\nLSD Peg Data:")
    lsd_prices = {"stETH": 1998, "rETH": 2150, "cbETH": 2050}
    eth_price = 2000
    
    for lsd in tracker.tracked_lsds[:3]:
        symbol = lsd["symbol"]
        price = lsd_prices.get(symbol, eth_price)
        peg = await tracker.fetch_lsd_peg(
            token_symbol=symbol,
            lsd_price_usd=price,
            eth_price_usd=eth_price,
            market_cap_usd=1_000_000_000,
            total_supply=500_000,
        )
        print(f"  {symbol}: Ratio={peg.price_ratio:.4f}, Deviation={peg.peg_deviation_percent:.2f}%")
    
    # Check for peg alerts
    alerts = await tracker.get_lsd_peg_alerts()
    if alerts:
        print("\nPeg Alerts:")
        for alert in alerts:
            print(f"  [{alert['severity']}] {alert['message']}")
    
    # Statistics
    stats = tracker.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
