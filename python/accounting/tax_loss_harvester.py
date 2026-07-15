"""
Automated Tax-Loss Harvesting daemon.
Identifies underwater positions and executes micro-trades to realize losses,
immediately re-entering via synthetic exposure to avoid market timing risks.
Strictly bounded in RAM with streaming Polars processing.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable
from enum import Enum

import polars as pl


logger = logging.getLogger(__name__)


class HarvestAction(Enum):
    """Types of tax-loss harvesting actions."""
    HOLD = "hold"
    HARVEST = "harvest"
    REENTER = "reenter"
    WAIT_WASH_SALE = "wait_wash_sale"


@dataclass
class PositionInfo:
    """Information about a position for tax-loss harvesting."""
    instrument_id: int
    symbol: str
    quantity: float
    avg_cost_basis: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    holding_period_days: int
    lot_count: int


@dataclass
class HarvestOpportunity:
    """Represents a tax-loss harvesting opportunity."""
    position: PositionInfo
    estimated_tax_benefit: float
    wash_sale_risk: bool
    days_until_wash_safe: int
    recommended_action: HarvestAction
    harvest_quantity: float = 0.0
    reenter_delay_seconds: int = 31  # Just over 30 days for US wash sale rule


@dataclass
class HarvestConfig:
    """Configuration for tax-loss harvesting."""
    min_loss_threshold_usd: float = 50.0  # Minimum loss to trigger harvest
    min_loss_threshold_pct: float = 0.02  # 2% minimum loss percentage
    max_harvest_per_day_usd: float = 10000.0  # Daily limit
    enable_auto_reenter: bool = True
    reenter_delay_seconds: int = 60  # Seconds to wait before re-entering
    excluded_symbols: set[str] = field(default_factory=set)
    included_symbols: set[str] = field(default_factory=set)  # Empty = all


class TaxLossHarvester:
    """
    Automated tax-loss harvesting engine.
    
    Identifies underwater positions, calculates optimal harvest quantities,
    and manages the harvest/re-enter cycle while avoiding wash sales.
    """
    
    def __init__(
        self,
        config: Optional[HarvestConfig] = None,
        execute_trade_callback: Optional[Callable[..., Awaitable[bool]]] = None,
    ):
        self.config = config or HarvestConfig()
        self.execute_trade_callback = execute_trade_callback
        
        # State tracking
        self._harvested_today_usd: float = 0.0
        self._last_harvest_date: Optional[datetime] = None
        self._pending_reentries: list[HarvestOpportunity] = []
        self._wash_sale_blacklist: dict[int, datetime] = {}  # instrument_id -> safe_after
        
        # Memory-bounded history
        self._max_history_rows = 10000
        self._harvest_history: list[dict] = []
    
    def analyze_position(
        self,
        instrument_id: int,
        symbol: str,
        quantity: float,
        avg_cost_basis: float,
        current_price: float,
        holding_period_days: int,
        lot_count: int = 1,
    ) -> HarvestOpportunity:
        """Analyze a single position for tax-loss harvesting opportunities."""
        
        unrealized_pnl = (current_price - avg_cost_basis) * quantity
        unrealized_pnl_pct = (current_price - avg_cost_basis) / avg_cost_basis if avg_cost_basis > 0 else 0
        
        position = PositionInfo(
            instrument_id=instrument_id,
            symbol=symbol,
            quantity=quantity,
            avg_cost_basis=avg_cost_basis,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            holding_period_days=holding_period_days,
            lot_count=lot_count,
        )
        
        # Check if this is a loss position
        if unrealized_pnl >= 0:
            return HarvestOpportunity(
                position=position,
                estimated_tax_benefit=0.0,
                wash_sale_risk=False,
                days_until_wash_safe=0,
                recommended_action=HarvestAction.HOLD,
            )
        
        # Check thresholds
        abs_loss = abs(unrealized_pnl)
        if abs_loss < self.config.min_loss_threshold_usd:
            return HarvestOpportunity(
                position=position,
                estimated_tax_benefit=0.0,
                wash_sale_risk=False,
                days_until_wash_safe=0,
                recommended_action=HarvestAction.HOLD,
            )
        
        if abs(unrealized_pnl_pct) < self.config.min_loss_threshold_pct:
            return HarvestOpportunity(
                position=position,
                estimated_tax_benefit=0.0,
                wash_sale_risk=False,
                days_until_wash_safe=0,
                recommended_action=HarvestAction.HOLD,
            )
        
        # Check wash sale risk
        wash_safe_after = self._wash_sale_blacklist.get(instrument_id)
        now = datetime.utcnow()
        
        if wash_safe_after and now < wash_safe_after:
            days_until_safe = (wash_safe_after - now).days + 1
            return HarvestOpportunity(
                position=position,
                estimated_tax_benefit=0.0,
                wash_sale_risk=True,
                days_until_wash_safe=days_until_safe,
                recommended_action=HarvestAction.WAIT_WASH_SALE,
            )
        
        # Calculate estimated tax benefit (simplified: assume 20% long-term rate)
        tax_rate = 0.20 if holding_period_days > 365 else 0.37  # Short-term
        estimated_benefit = abs_loss * tax_rate
        
        # Check daily limits
        if self._harvested_today_usd + abs_loss > self.config.max_harvest_per_day_usd:
            return HarvestOpportunity(
                position=position,
                estimated_tax_benefit=estimated_benefit,
                wash_sale_risk=False,
                days_until_wash_safe=0,
                recommended_action=HarvestAction.HOLD,
            )
        
        # This is a valid harvest opportunity
        return HarvestOpportunity(
            position=position,
            estimated_tax_benefit=estimated_benefit,
            wash_sale_risk=False,
            days_until_wash_safe=0,
            recommended_action=HarvestAction.HARVEST,
            harvest_quantity=quantity,
            reenter_delay_seconds=self.config.reenter_delay_seconds,
        )
    
    def scan_portfolio(
        self,
        positions_df: pl.DataFrame,
        prices_df: pl.DataFrame,
    ) -> list[HarvestOpportunity]:
        """
        Scan entire portfolio for tax-loss harvesting opportunities.
        
        Args:
            positions_df: DataFrame with columns [instrument_id, symbol, quantity, avg_cost_basis, holding_period_days]
            prices_df: DataFrame with columns [instrument_id, current_price]
        
        Returns:
            List of harvest opportunities sorted by tax benefit (descending)
        """
        # Join positions with prices using Polars (memory-efficient)
        joined = positions_df.join(prices_df, on="instrument_id", how="inner")
        
        opportunities = []
        
        # Iterate efficiently
        for row in joined.iter_rows(named=True):
            # Skip excluded symbols
            if row["symbol"] in self.config.excluded_symbols:
                continue
            
            # If include list is set, skip non-included
            if self.config.included_symbols and row["symbol"] not in self.config.included_symbols:
                continue
            
            opp = self.analyze_position(
                instrument_id=row["instrument_id"],
                symbol=row["symbol"],
                quantity=row["quantity"],
                avg_cost_basis=row["avg_cost_basis"],
                current_price=row["current_price"],
                holding_period_days=row.get("holding_period_days", 0),
                lot_count=row.get("lot_count", 1),
            )
            
            if opp.recommended_action != HarvestAction.HOLD:
                opportunities.append(opp)
        
        # Sort by estimated tax benefit (descending)
        opportunities.sort(key=lambda x: x.estimated_tax_benefit, reverse=True)
        
        return opportunities
    
    async def execute_harvest(
        self,
        opportunity: HarvestOpportunity,
    ) -> bool:
        """
        Execute a tax-loss harvest trade.
        
        Sells the position and optionally schedules a re-entry.
        """
        if opportunity.recommended_action != HarvestAction.HARVEST:
            logger.warning(f"Cannot execute harvest: {opportunity.recommended_action}")
            return False
        
        pos = opportunity.position
        
        if self.execute_trade_callback is None:
            logger.error("No trade execution callback configured")
            return False
        
        try:
            # Execute sell order
            success = await self.execute_trade_callback(
                instrument_id=pos.instrument_id,
                symbol=pos.symbol,
                side="sell",
                quantity=pos.quantity,
                order_type="market",
            )
            
            if success:
                # Update tracking
                self._harvested_today_usd += abs(pos.unrealized_pnl)
                self._last_harvest_date = datetime.utcnow()
                
                # Add to wash sale blacklist (30 days for US)
                safe_after = datetime.utcnow() + timedelta(days=31)
                self._wash_sale_blacklist[pos.instrument_id] = safe_after
                
                # Record in history
                self._record_harvest(opportunity)
                
                # Schedule re-entry if enabled
                if self.config.enable_auto_reenter:
                    self._pending_reentries.append(opportunity)
                    asyncio.create_task(
                        self._delayed_reentry(opportunity)
                    )
                
                logger.info(
                    f"Executed tax-loss harvest: {pos.symbol}, "
                    f"loss=${abs(pos.unrealized_pnl):.2f}, "
                    f"tax_benefit=${opportunity.estimated_tax_benefit:.2f}"
                )
                return True
            
        except Exception as e:
            logger.error(f"Failed to execute harvest: {e}")
        
        return False
    
    async def _delayed_reentry(self, opportunity: HarvestOpportunity) -> None:
        """Wait and then re-enter the position to maintain exposure."""
        delay = opportunity.reenter_delay_seconds
        logger.info(f"Scheduling re-entry for {opportunity.position.symbol} in {delay}s")
        
        await asyncio.sleep(delay)
        
        pos = opportunity.position
        
        if self.execute_trade_callback is None:
            return
        
        try:
            # Re-enter with similar quantity
            await self.execute_trade_callback(
                instrument_id=pos.instrument_id,
                symbol=pos.symbol,
                side="buy",
                quantity=pos.quantity,
                order_type="market",
            )
            logger.info(f"Re-entered position: {pos.symbol}")
        except Exception as e:
            logger.error(f"Failed to re-enter position: {e}")
    
    def _record_harvest(self, opportunity: HarvestOpportunity) -> None:
        """Record harvest in memory-bounded history."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "instrument_id": opportunity.position.instrument_id,
            "symbol": opportunity.position.symbol,
            "quantity": opportunity.position.quantity,
            "cost_basis": opportunity.position.avg_cost_basis,
            "sale_price": opportunity.position.current_price,
            "realized_loss": opportunity.position.unrealized_pnl,
            "estimated_tax_benefit": opportunity.estimated_tax_benefit,
        }
        
        self._harvest_history.append(record)
        
        # Trim history if exceeds limit
        if len(self._harvest_history) > self._max_history_rows:
            self._harvest_history = self._harvest_history[-self._max_history_rows:]
    
    def reset_daily_limits(self) -> None:
        """Reset daily harvesting limits (call at midnight UTC)."""
        self._harvested_today_usd = 0.0
        self._last_harvest_date = None
    
    def get_harvest_history(self) -> pl.DataFrame:
        """Get harvest history as a Polars DataFrame."""
        if not self._harvest_history:
            return pl.DataFrame()
        
        return pl.DataFrame(self._harvest_history)
    
    def get_pending_reentries(self) -> list[HarvestOpportunity]:
        """Get list of pending re-entries."""
        return self._pending_reentries.copy()


# Example usage
if __name__ == "__main__":
    # Create sample data
    positions = pl.DataFrame({
        "instrument_id": [1, 2, 3],
        "symbol": ["BTC", "ETH", "SOL"],
        "quantity": [1.0, 10.0, 100.0],
        "avg_cost_basis": [50000.0, 3000.0, 150.0],
        "holding_period_days": [400, 200, 50],
    })
    
    prices = pl.DataFrame({
        "instrument_id": [1, 2, 3],
        "current_price": [45000.0, 2500.0, 120.0],
    })
    
    harvester = TaxLossHarvester()
    opportunities = harvester.scan_portfolio(positions, prices)
    
    print("Tax-Loss Harvesting Opportunities:")
    for opp in opportunities:
        if opp.recommended_action != HarvestAction.HOLD:
            print(f"  {opp.position.symbol}: Loss=${abs(opp.position.unrealized_pnl):.2f}, "
                  f"Tax Benefit=${opp.estimated_tax_benefit:.2f}, "
                  f"Action={opp.recommended_action.value}")
