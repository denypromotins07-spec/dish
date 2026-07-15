"""
Scheduler and scraper for token vesting schedules and cliff unlocks.
Maps future supply shocks into regime detection and volatility forecasting models.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import heapq


class VestingType(Enum):
    """Types of vesting schedules."""
    LINEAR = "linear"
    CLIFF = "cliff"
    GRADUATED = "graduated"
    CUSTOM = "custom"


@dataclass
class TokenUnlock:
    """Represents a scheduled token unlock event."""
    token_symbol: str
    token_name: str
    unlock_timestamp: datetime
    unlock_amount: float
    unlock_value_usd: float
    percent_of_circulating: float
    percent_of_total_supply: float
    recipient_type: str  # "team", "investor", "foundation", "ecosystem"
    vesting_type: VestingType
    is_cliff: bool
    source: str  # Data source for verification


@dataclass
class VestingSchedule:
    """Complete vesting schedule for a token allocation."""
    token_symbol: str
    recipient: str
    recipient_type: str
    total_allocation: float
    start_date: datetime
    end_date: datetime
    cliff_date: Optional[datetime]
    vesting_type: VestingType
    vesting_period_days: int
    tokens_released: float
    tokens_remaining: float


@dataclass
class SupplyShockEvent:
    """Calculated supply shock from upcoming unlocks."""
    date: datetime
    total_unlock_value_usd: float
    percent_of_circulating: float
    affected_tokens: List[str]
    severity: str  # "low", "medium", "high", "critical"
    confidence_score: float


class TokenUnlockScheduler:
    """
    Scheduler for token vesting schedules and cliff unlocks.
    Maps future supply shocks for trading strategy adjustments.
    """

    def __init__(self):
        # Scheduled unlocks sorted by timestamp
        self._upcoming_unlocks: List[TokenUnlock] = []
        
        # Vesting schedules being tracked
        self._vesting_schedules: Dict[str, VestingSchedule] = {}
        
        # Historical unlock data
        self._historical_unlocks: List[TokenUnlock] = []
        
        # Severity thresholds (percent of circulating supply)
        self.severity_thresholds = {
            "low": 0.5,      # < 0.5%
            "medium": 2.0,   # 0.5% - 2%
            "high": 5.0,     # 2% - 5%
            "critical": 10.0, # > 5%
        }
        
        # Known token allocations (would be loaded from configuration/database)
        self.known_allocations = {
            "ARB": {"total_supply": 10_000_000_000, "circulating": 1_275_000_000},
            "OP": {"total_supply": 4_294_967_296, "circulating": 780_000_000},
            "SUI": {"total_supply": 10_000_000_000, "circulating": 1_000_000_000},
            "APT": {"total_supply": 1_000_000_000, "circulating": 150_000_000},
        }

    async def add_vesting_schedule(
        self,
        schedule: VestingSchedule,
        token_price_usd: float,
    ):
        """Add a new vesting schedule and calculate unlock events."""
        self._vesting_schedules[schedule.token_symbol + "_" + schedule.recipient] = schedule
        
        # Generate unlock events
        unlocks = self._generate_unlock_events(schedule, token_price_usd)
        
        for unlock in unlocks:
            self._add_unlock(unlock)

    def _generate_unlock_events(
        self,
        schedule: VestingSchedule,
        price_usd: float,
    ) -> List[TokenUnlock]:
        """Generate individual unlock events from a vesting schedule."""
        unlocks = []
        
        total_supply = self.known_allocations.get(
            schedule.token_symbol, {"total_supply": 1_000_000_000}
        )["total_supply"]
        
        circulating = self.known_allocations.get(
            schedule.token_symbol, {"circulating": 100_000_000}
        )["circulating"]
        
        if schedule.vesting_type == VestingType.CLIFF:
            # Single cliff unlock
            if schedule.cliff_date:
                unlock = TokenUnlock(
                    token_symbol=schedule.token_symbol,
                    token_name=schedule.token_symbol,
                    unlock_timestamp=schedule.cliff_date,
                    unlock_amount=schedule.total_allocation,
                    unlock_value_usd=schedule.total_allocation * price_usd,
                    percent_of_circulating=(schedule.total_allocation / circulating) * 100,
                    percent_of_total_supply=(schedule.total_allocation / total_supply) * 100,
                    recipient_type=schedule.recipient_type,
                    vesting_type=VestingType.CLIFF,
                    is_cliff=True,
                    source="manual",
                )
                unlocks.append(unlock)
        
        elif schedule.vesting_type == VestingType.LINEAR:
            # Linear vesting over period
            days_total = (schedule.end_date - schedule.start_date).days
            tokens_per_day = schedule.total_allocation / max(days_total, 1)
            
            current_date = schedule.start_date
            while current_date <= schedule.end_date:
                # Check if past cliff
                if schedule.cliff_date and current_date < schedule.cliff_date:
                    current_date += timedelta(days=1)
                    continue
                
                unlock_amount = tokens_per_day * schedule.vesting_period_days
                
                if unlock_amount > 0:
                    unlock = TokenUnlock(
                        token_symbol=schedule.token_symbol,
                        token_name=schedule.token_symbol,
                        unlock_timestamp=current_date,
                        unlock_amount=unlock_amount,
                        unlock_value_usd=unlock_amount * price_usd,
                        percent_of_circulating=(unlock_amount / circulating) * 100,
                        percent_of_total_supply=(unlock_amount / total_supply) * 100,
                        recipient_type=schedule.recipient_type,
                        vesting_type=VestingType.LINEAR,
                        is_cliff=False,
                        source="manual",
                    )
                    unlocks.append(unlock)
                
                current_date += timedelta(days=schedule.vesting_period_days)
        
        return unlocks

    def _add_unlock(self, unlock: TokenUnlock):
        """Add unlock to the priority queue."""
        heapq.heappush(self._upcoming_unlocks, (unlock.unlock_timestamp, unlock))

    async def get_upcoming_unlocks(
        self,
        days_ahead: int = 30,
        min_value_usd: float = 1_000_000,
    ) -> List[TokenUnlock]:
        """Get upcoming unlocks within specified time window."""
        cutoff = datetime.utcnow() + timedelta(days=days_ahead)
        results = []
        
        temp_list = []
        while self._upcoming_unlocks:
            timestamp, unlock = heapq.heappop(self._upcoming_unlocks)
            temp_list.append((timestamp, unlock))
            
            if timestamp <= cutoff and unlock.unlock_value_usd >= min_value_usd:
                results.append(unlock)
        
        # Restore the heap
        for item in temp_list:
            heapq.heappush(self._upcoming_unlocks, item)
        
        return sorted(results, key=lambda x: x.unlock_timestamp)

    async def get_supply_shock_forecast(
        self,
        window_days: int = 7,
    ) -> List[SupplyShockEvent]:
        """Generate supply shock forecast for trading strategy."""
        upcoming = await self.get_upcoming_unlocks(days_ahead=window_days, min_value_usd=0)
        
        if not upcoming:
            return []
        
        # Group by date
        daily_unlocks: Dict[datetime.date, List[TokenUnlock]] = {}
        for unlock in upcoming:
            date_key = unlock.unlock_timestamp.date()
            if date_key not in daily_unlocks:
                daily_unlocks[date_key] = []
            daily_unlocks[date_key].append(unlock)
        
        # Calculate shock events
        shocks = []
        for date, day_unlocks in sorted(daily_unlocks.items()):
            total_value = sum(u.unlock_value_usd for u in day_unlocks)
            avg_percent_circulating = sum(u.percent_of_circulating for u in day_unlocks) / len(day_unlocks)
            
            # Determine severity
            severity = self._calculate_severity(avg_percent_circulating)
            
            # Calculate confidence based on data quality
            confidence = min(1.0, len(day_unlocks) * 0.2)  # More sources = higher confidence
            
            shock = SupplyShockEvent(
                date=datetime.combine(date, datetime.min.time()),
                total_unlock_value_usd=total_value,
                percent_of_circulating=avg_percent_circulating,
                affected_tokens=list(set(u.token_symbol for u in day_unlocks)),
                severity=severity,
                confidence_score=confidence,
            )
            shocks.append(shock)
        
        return shocks

    def _calculate_severity(self, percent_circulating: float) -> str:
        """Calculate severity level based on percent of circulating supply."""
        if percent_circulating >= self.severity_thresholds["critical"]:
            return "critical"
        elif percent_circulating >= self.severity_thresholds["high"]:
            return "high"
        elif percent_circulating >= self.severity_thresholds["medium"]:
            return "medium"
        else:
            return "low"

    def get_token_cumulative_unlocks(
        self,
        token_symbol: str,
        months_ahead: int = 12,
    ) -> List[Tuple[datetime, float, float]]:
        """Get cumulative unlock schedule for a specific token."""
        cutoff = datetime.utcnow() + timedelta(days=months_ahead * 30)
        
        token_unlocks = [
            u for u in self._upcoming_unlocks 
            if u[1].token_symbol == token_symbol and u[0] <= cutoff
        ]
        
        if not token_unlocks:
            return []
        
        cumulative = []
        running_total = 0
        
        for timestamp, unlock in sorted(token_unlocks):
            running_total += unlock.unlock_amount
            cumulative.append((timestamp, running_total, unlock.percent_of_circulating))
        
        return cumulative

    def record_historical_unlock(self, unlock: TokenUnlock):
        """Record a completed unlock for analysis."""
        self._historical_unlocks.append(unlock)
        
        # Keep only recent history
        if len(self._historical_unlocks) > 1000:
            self._historical_unlocks = self._historical_unlocks[-500:]

    def get_unlock_statistics(self) -> Dict:
        """Get statistics about tracked unlocks."""
        upcoming_count = len(self._upcoming_unlocks)
        historical_count = len(self._historical_unlocks)
        
        # Calculate total upcoming value
        total_upcoming_value = sum(u.unlock_value_usd for _, u in self._upcoming_unlocks)
        
        # Get distribution by recipient type
        by_recipient = {}
        for _, unlock in self._upcoming_unlocks:
            rtype = unlock.recipient_type
            by_recipient[rtype] = by_recipient.get(rtype, 0) + unlock.unlock_value_usd
        
        return {
            "upcoming_unlocks": upcoming_count,
            "historical_unlocks": historical_count,
            "total_upcoming_value_usd": total_upcoming_value,
            "by_recipient_type": by_recipient,
            "tracked_schedules": len(self._vesting_schedules),
        }


async def main():
    """Example usage of TokenUnlockScheduler."""
    scheduler = TokenUnlockScheduler()
    
    # Add a sample vesting schedule
    schedule = VestingSchedule(
        token_symbol="TEST",
        recipient="Early Investors",
        recipient_type="investor",
        total_allocation=10_000_000,
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=365),
        cliff_date=datetime.utcnow() + timedelta(days=90),
        vesting_type=VestingType.LINEAR,
        vesting_period_days=30,
        tokens_released=0,
        tokens_remaining=10_000_000,
    )
    
    await scheduler.add_vesting_schedule(schedule, token_price_usd=1.5)
    
    # Get upcoming unlocks
    upcoming = await scheduler.get_upcoming_unlocks(days_ahead=365)
    print(f"Upcoming Unlocks ({len(upcoming)} events):")
    for unlock in upcoming[:5]:
        print(f"  {unlock.unlock_timestamp.date()}: {unlock.unlock_amount:,.0f} {unlock.token_symbol} (${unlock.unlock_value_usd:,.0f})")
    
    # Get supply shock forecast
    shocks = await scheduler.get_supply_shock_forecast(window_days=30)
    print(f"\nSupply Shock Forecast:")
    for shock in shocks:
        print(f"  {shock.date.date()}: ${shock.total_unlock_value_usd:,.0f} ({shock.severity})")
    
    # Statistics
    stats = scheduler.get_unlock_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
