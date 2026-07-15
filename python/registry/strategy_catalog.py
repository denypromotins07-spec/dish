"""
Strategy Catalog: Centralized catalog of all available strategies.
Tracks active parameters, live/shadow/backtest status.
Single source of truth for frontend's "Strategy Manager" tab.
Memory-bounded with efficient lookups.
"""

import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from enum import Enum
from collections import OrderedDict


class StrategyStatus(Enum):
    """Strategy execution status."""
    INACTIVE = "inactive"
    LIVE = "live"
    SHADOW = "shadow"  # Running but not executing trades
    BACKTEST = "backtest"
    PAUSED = "paused"


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    name: str
    strategy_type: str
    version: str
    status: StrategyStatus
    parameters: Dict[str, float]
    risk_limits: Dict[str, float]
    symbols: List[str]
    capital_allocation: float  # 0.0 to 1.0
    max_position_size: float
    created_at: int  # timestamp_us
    updated_at: int  # timestamp_us
    description: str = ""
    tags: List[str] = field(default_factory=list)


class StrategyCatalog:
    """
    Thread-safe, memory-bounded strategy catalog.
    Uses OrderedDict for LRU-style management if needed.
    """
    
    def __init__(self, max_strategies: int = 100):
        self.max_strategies = max_strategies
        self._lock = threading.RLock()
        self._strategies: OrderedDict[str, StrategyConfig] = OrderedDict()
        self._status_index: Dict[StrategyStatus, Set[str]] = {
            status: set() for status in StrategyStatus
        }
        self._type_index: Dict[str, Set[str]] = {}
        
    def register_strategy(self, config: StrategyConfig) -> bool:
        """Register a new strategy or update existing one."""
        with self._lock:
            # Check if we need to evict oldest
            if config.name not in self._strategies and len(self._strategies) >= self.max_strategies:
                # Remove oldest inactive strategy
                self._evict_oldest_inactive()
            
            # Update indexes
            if config.name in self._strategies:
                old_config = self._strategies[config.name]
                self._status_index[old_config.status].discard(config.name)
                if old_config.strategy_type in self._type_index:
                    self._type_index[old_config.strategy_type].discard(config.name)
            
            # Add/update strategy
            self._strategies[config.name] = config
            
            # Update indexes
            self._status_index[config.status].add(config.name)
            
            if config.strategy_type not in self._type_index:
                self._type_index[config.strategy_type] = set()
            self._type_index[config.strategy_type].add(config.name)
            
            # Move to end (most recently used)
            self._strategies.move_to_end(config.name)
            
            return True
    
    def _evict_oldest_inactive(self):
        """Evict the oldest inactive strategy."""
        for name, config in list(self._strategies.items()):
            if config.status == StrategyStatus.INACTIVE:
                self.remove_strategy(name)
                return
        
        # If no inactive, warn but don't evict active strategies
        # In production, this might trigger an alert
    
    def remove_strategy(self, name: str) -> bool:
        """Remove a strategy from the catalog."""
        with self._lock:
            if name not in self._strategies:
                return False
            
            config = self._strategies.pop(name)
            self._status_index[config.status].discard(name)
            if config.strategy_type in self._type_index:
                self._type_index[config.strategy_type].discard(name)
            
            return True
    
    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get a specific strategy by name."""
        with self._lock:
            return self._strategies.get(name)
    
    def get_all_strategies(self) -> List[StrategyConfig]:
        """Get all strategies."""
        with self._lock:
            return list(self._strategies.values())
    
    def get_by_status(self, status: StrategyStatus) -> List[StrategyConfig]:
        """Get all strategies with a specific status."""
        with self._lock:
            names = self._status_index.get(status, set())
            return [self._strategies[name] for name in names if name in self._strategies]
    
    def get_by_type(self, strategy_type: str) -> List[StrategyConfig]:
        """Get all strategies of a specific type."""
        with self._lock:
            names = self._type_index.get(strategy_type, set())
            return [self._strategies[name] for name in names if name in self._strategies]
    
    def get_live_strategies(self) -> List[StrategyConfig]:
        """Get all currently live strategies."""
        return self.get_by_status(StrategyStatus.LIVE)
    
    def update_status(self, name: str, new_status: StrategyStatus) -> bool:
        """Update a strategy's status."""
        with self._lock:
            if name not in self._strategies:
                return False
            
            config = self._strategies[name]
            old_status = config.status
            
            # Update index
            self._status_index[old_status].discard(name)
            self._status_index[new_status].add(name)
            
            # Update config
            config.status = new_status
            config.updated_at = self._current_time_us()
            
            return True
    
    def update_parameters(self, name: str, new_params: Dict[str, float]) -> bool:
        """Update strategy parameters atomically."""
        with self._lock:
            if name not in self._strategies:
                return False
            
            config = self._strategies[name]
            config.parameters.update(new_params)
            config.updated_at = self._current_time_us()
            
            return True
    
    def get_total_capital_allocated(self) -> float:
        """Get total capital allocated across all active strategies."""
        with self._lock:
            total = 0.0
            for config in self._strategies.values():
                if config.status in [StrategyStatus.LIVE, StrategyStatus.SHADOW]:
                    total += config.capital_allocation
            return total
    
    def validate_allocation(self, proposed_allocation: float) -> bool:
        """Check if proposed allocation would exceed 100%."""
        current = self.get_total_capital_allocated()
        return (current + proposed_allocation) <= 1.0
    
    def export_to_json(self) -> str:
        """Export catalog to JSON for UI."""
        with self._lock:
            data = []
            for config in self._strategies.values():
                item = asdict(config)
                item['status'] = config.status.value
                data.append(item)
            
            return json.dumps({
                'strategies': data,
                'total_count': len(data),
                'live_count': len(self._status_index[StrategyStatus.LIVE]),
                'shadow_count': len(self._status_index[StrategyStatus.SHADOW]),
                'total_allocated': self.get_total_capital_allocated()
            }, separators=(',', ':'))
    
    def _current_time_us(self) -> int:
        """Get current time in microseconds."""
        import time
        return int(time.time() * 1_000_000)
    
    def clear(self):
        """Clear all strategies."""
        with self._lock:
            self._strategies.clear()
            for s in self._status_index.values():
                s.clear()
            self._type_index.clear()


# Singleton instance
_catalog_instance: Optional[StrategyCatalog] = None
_instance_lock = threading.Lock()


def get_catalog() -> StrategyCatalog:
    """Get or create the singleton StrategyCatalog instance."""
    global _catalog_instance
    if _catalog_instance is None:
        with _instance_lock:
            if _catalog_instance is None:
                _catalog_instance = StrategyCatalog()
    return _catalog_instance


if __name__ == '__main__':
    # Example usage
    catalog = get_catalog()
    
    # Register some strategies
    strategies = [
        StrategyConfig(
            name="momentum_v1",
            strategy_type="momentum",
            version="1.0.0",
            status=StrategyStatus.LIVE,
            parameters={'lookback': 20.0, 'threshold': 0.02},
            risk_limits={'max_dd': 0.05, 'var_limit': 0.03},
            symbols=['BTC-PERP', 'ETH-PERP'],
            capital_allocation=0.3,
            max_position_size=100000.0,
            created_at=1700000000000000,
            updated_at=1700000000000000,
            description="Momentum-based trend following",
            tags=['trend', 'crypto']
        ),
        StrategyConfig(
            name="mean_reversion_v2",
            strategy_type="mean_reversion",
            version="2.1.0",
            status=StrategyStatus.SHADOW,
            parameters={'zscore_threshold': 2.0, 'half_life': 60},
            risk_limits={'max_dd': 0.03, 'var_limit': 0.02},
            symbols=['BTC-PERP', 'ETH-PERP', 'SOL-PERP'],
            capital_allocation=0.2,
            max_position_size=50000.0,
            created_at=1700000000000000,
            updated_at=1700000000000000,
            description="Statistical mean reversion",
            tags=['mean-reversion', 'stat-arb']
        ),
    ]
    
    for s in strategies:
        catalog.register_strategy(s)
    
    print(f"Total strategies: {len(catalog.get_all_strategies())}")
    print(f"Live strategies: {len(catalog.get_live_strategies())}")
    print(f"Total allocated: {catalog.get_total_capital_allocated():.1%}")
    
    # Export to JSON
    json_output = catalog.export_to_json()
    print(f"JSON size: {len(json_output)} bytes")
