#!/usr/bin/env python3
"""
On-Demand State Snapshot Generator.
Serializes portfolio, active strategies, and system health into JSON for UI hydration.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Snapshot of a single position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str  # "long" or "short"
    timestamp_us: int


@dataclass
class StrategySnapshot:
    """Snapshot of an active strategy."""
    name: str
    status: str  # "active", "paused", "stopped"
    symbols: List[str]
    total_pnl: float
    trade_count: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    last_signal_ts: int


@dataclass
class SystemHealthSnapshot:
    """Snapshot of system health metrics."""
    cpu_usage_pct: float
    memory_used_mb: float
    memory_total_mb: float
    disk_free_gb: float
    network_latency_ms: float
    uptime_seconds: float
    active_connections: int
    message_queue_depth: int


@dataclass
class FullStateSnapshot:
    """Complete state snapshot for UI hydration."""
    timestamp_us: int
    positions: List[PositionSnapshot]
    strategies: List[StrategySnapshot]
    system_health: SystemHealthSnapshot
    account_balance: float
    total_equity: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    margin_used: float
    margin_available: float
    risk_metrics: Dict[str, float]


class StateSnapshotter:
    """Generates on-demand state snapshots for frontend hydration."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Optional[FullStateSnapshot] = None
        self._cache_timestamp: int = 0
        self._cache_ttl_us: int = 100_000  # 100ms cache TTL
        
    def get_portfolio_positions(self) -> List[PositionSnapshot]:
        """Fetch current portfolio positions."""
        # In production, fetch from LMDB or shared memory
        # This is a simulation with sample data
        return [
            PositionSnapshot(
                symbol="BTC-USD",
                quantity=0.5,
                entry_price=42000.0,
                current_price=43500.0,
                unrealized_pnl=750.0,
                unrealized_pnl_pct=1.79,
                side="long",
                timestamp_us=int(time.time() * 1_000_000)
            ),
            PositionSnapshot(
                symbol="ETH-USD",
                quantity=5.0,
                entry_price=2200.0,
                current_price=2150.0,
                unrealized_pnl=-250.0,
                unrealized_pnl_pct=-2.27,
                side="long",
                timestamp_us=int(time.time() * 1_000_000)
            )
        ]
    
    def get_active_strategies(self) -> List[StrategySnapshot]:
        """Fetch active strategy states."""
        # In production, fetch from strategy manager
        return [
            StrategySnapshot(
                name="StatArb_BTCETH",
                status="active",
                symbols=["BTC-USD", "ETH-USD"],
                total_pnl=12500.0,
                trade_count=156,
                win_rate=0.58,
                sharpe_ratio=2.1,
                max_drawdown=0.08,
                last_signal_ts=int(time.time() * 1_000_000) - 5_000_000
            ),
            StrategySnapshot(
                name="Momentum_SOL",
                status="active",
                symbols=["SOL-USD"],
                total_pnl=3200.0,
                trade_count=89,
                win_rate=0.52,
                sharpe_ratio=1.5,
                max_drawdown=0.12,
                last_signal_ts=int(time.time() * 1_000_000) - 120_000_000
            )
        ]
    
    def get_system_health(self) -> SystemHealthSnapshot:
        """Fetch system health metrics."""
        import psutil
        
        try:
            cpu_pct = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            return SystemHealthSnapshot(
                cpu_usage_pct=cpu_pct,
                memory_used_mb=mem.used / (1024 * 1024),
                memory_total_mb=mem.total / (1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_latency_ms=15.0,  # Simulated
                uptime_seconds=time.time() - psutil.boot_time(),
                active_connections=len(psutil.net_connections(kind='tcp')),
                message_queue_depth=0  # Would fetch from queue
            )
        except Exception as e:
            logger.warning(f"Failed to get system health: {e}")
            return SystemHealthSnapshot(
                cpu_usage_pct=0.0,
                memory_used_mb=0.0,
                memory_total_mb=16384.0,
                disk_free_gb=0.0,
                network_latency_ms=0.0,
                uptime_seconds=0.0,
                active_connections=0,
                message_queue_depth=0
            )
    
    def get_account_summary(self) -> Dict[str, float]:
        """Fetch account balance and margin info."""
        # In production, fetch from exchange API or local state
        return {
            "account_balance": 100000.0,
            "total_equity": 115250.0,
            "total_pnl": 15250.0,
            "realized_pnl": 14750.0,
            "unrealized_pnl": 500.0,
            "margin_used": 25000.0,
            "margin_available": 75000.0
        }
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Fetch current risk metrics."""
        return {
            "var_95": 2500.0,
            "cvar_95": 3200.0,
            "max_position_size": 50000.0,
            "leverage": 1.5,
            "beta_btc": 0.85,
            "volatility_30d": 0.45
        }
    
    def generate_snapshot(self, force_refresh: bool = False) -> FullStateSnapshot:
        """Generate a complete state snapshot."""
        now_us = int(time.time() * 1_000_000)
        
        # Check cache
        if not force_refresh and self._cache is not None:
            if now_us - self._cache_timestamp < self._cache_ttl_us:
                return self._cache
        
        with self._lock:
            # Double-check after acquiring lock
            if not force_refresh and self._cache is not None:
                if now_us - self._cache_timestamp < self._cache_ttl_us:
                    return self._cache
            
            # Generate new snapshot
            snapshot = FullStateSnapshot(
                timestamp_us=now_us,
                positions=self.get_portfolio_positions(),
                strategies=self.get_active_strategies(),
                system_health=self.get_system_health(),
                **self.get_account_summary(),
                risk_metrics=self.get_risk_metrics()
            )
            
            self._cache = snapshot
            self._cache_timestamp = now_us
            
            logger.info(f"Generated state snapshot at {now_us}")
            return snapshot
    
    def to_json(self, snapshot: Optional[FullStateSnapshot] = None) -> str:
        """Convert snapshot to JSON string."""
        if snapshot is None:
            snapshot = self.generate_snapshot()
        
        # Convert dataclass to dict recursively
        def convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj
        
        data = convert(snapshot)
        return json.dumps(data, indent=2)
    
    def save_to_file(self, filepath: str) -> str:
        """Generate snapshot and save to file."""
        snapshot = self.generate_snapshot(force_refresh=True)
        json_str = self.to_json(snapshot)
        
        with open(filepath, 'w') as f:
            f.write(json_str)
        
        logger.info(f"Saved state snapshot to {filepath}")
        return filepath


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="State Snapshot Generator")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--format", choices=["json", "pretty"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    snapshotter = StateSnapshotter()
    
    if args.output:
        filepath = snapshotter.save_to_file(args.output)
        print(f"Snapshot saved to: {filepath}")
    else:
        snapshot = snapshotter.generate_snapshot(force_refresh=True)
        
        if args.format == "pretty":
            print(snapshotter.to_json(snapshot))
        else:
            # Compact JSON
            data = {k: v for k, v in asdict(snapshot).items()}
            print(json.dumps(data))


if __name__ == "__main__":
    main()
