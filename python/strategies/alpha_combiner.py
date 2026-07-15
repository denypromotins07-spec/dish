"""
Ray-based Alpha Combiner Ensemble Model
Combines signals from Market Making, Stat Arb, and Microstructure strategies
Uses dynamic weighting based on recent out-of-sample performance
"""

import ray
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class StrategyType(Enum):
    MARKET_MAKING = "market_making"
    STAT_ARB = "stat_arb"
    MICROSTRUCTURE = "microstructure"
    ORDERFLOW = "orderflow"


@dataclass
class Signal:
    """Trading signal from a sub-strategy"""
    strategy_id: str
    strategy_type: StrategyType
    direction: float  # -1.0 to 1.0 (short to long)
    strength: float   # 0.0 to 1.0 (confidence)
    timestamp_ns: int
    expected_return: float
    risk_estimate: float


@dataclass
class CombinedSignal:
    """Master signal after combination"""
    direction: float
    strength: float
    weighted_expected_return: float
    weighted_risk: float
    active_strategies: int
    timestamp_ns: int


@ray.remote
class SignalEvaluator:
    """Remote actor for evaluating individual strategy signals"""
    
    def __init__(self, strategy_id: str, window_size: int = 100):
        self.strategy_id = strategy_id
        self.window_size = window_size
        self.recent_returns: List[float] = []
        self.recent_sharpes: List[float] = []
        self.total_signals = 0
        self profitable_signals = 0
    
    def evaluate_signal(self, signal: Signal, realized_return: Optional[float] = None) -> Dict:
        """Evaluate signal quality and update metrics"""
        self.total_signals += 1
        
        if realized_return is not None:
            self.recent_returns.append(realized_return)
            if len(self.recent_returns) > self.window_size:
                self.recent_returns.pop(0)
            
            if realized_return > 0:
                self.profitable_signals += 1
        
        # Calculate recent Sharpe ratio
        sharpe = 0.0
        if len(self.recent_returns) >= 10:
            returns = np.array(self.recent_returns)
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        self.recent_sharpes.append(sharpe)
        if len(self.recent_sharpes) > self.window_size:
            self.recent_sharpes.pop(0)
        
        win_rate = self.profitable_signals / self.total_signals if self.total_signals > 0 else 0.0
        
        return {
            'strategy_id': self.strategy_id,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'avg_return': np.mean(self.recent_returns) if self.recent_returns else 0.0,
            'return_std': np.std(self.recent_returns) if self.recent_returns else 0.0,
        }
    
    def get_weight(self) -> float:
        """Calculate current weight for this strategy"""
        if len(self.recent_sharpes) < 5:
            return 0.1  # Default low weight during warmup
        
        recent_sharpe = np.mean(self.recent_sharpes[-10:])
        # Normalize Sharpe to 0-1 range (assuming max Sharpe of 3)
        normalized = np.clip(recent_sharpe / 3.0, 0.0, 1.0)
        return normalized


@ray.remote
class AlphaCombiner:
    """
    Ray-based ensemble model that combines signals from multiple strategies
    Uses dynamic weighting based on recent out-of-sample performance
    """
    
    def __init__(self, min_strategies: int = 2, consensus_threshold: float = 0.3):
        self.min_strategies = min_strategies
        self.consensus_threshold = consensus_threshold
        self.evaluators: Dict[str, ray.actor.ActorHandle] = {}
        self.signal_history: List[CombinedSignal] = []
        self.max_history = 1000
        self.is_initialized = False
    
    def register_strategy(self, strategy_id: str, strategy_type: StrategyType) -> None:
        """Register a new strategy for signal combination"""
        if strategy_id not in self.evaluators:
            self.evaluators[strategy_id] = SignalEvaluator.remote(strategy_id)
    
    def unregister_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from combination"""
        if strategy_id in self.evaluators:
            del self.evaluators[strategy_id]
    
    async def combine_signals(self, signals: List[Signal]) -> CombinedSignal:
        """
        Combine multiple strategy signals into a master signal
        Uses performance-weighted averaging
        """
        if not signals:
            return CombinedSignal(
                direction=0.0,
                strength=0.0,
                weighted_expected_return=0.0,
                weighted_risk=0.0,
                active_strategies=0,
                timestamp_ns=time.time_ns()
            )
        
        # Get weights for all strategies
        weight_futures = []
        for signal in signals:
            if signal.strategy_id in self.evaluators:
                weight_futures.append(
                    self.evaluators[signal.strategy_id].get_weight.remote()
                )
            else:
                weight_futures.append(ray.put(0.1))  # Default weight for unknown
        
        weights = await ray.get(weight_futures)
        
        # Filter signals by consensus
        valid_signals = []
        valid_weights = []
        
        for signal, weight in zip(signals, weights):
            if signal.strength >= self.consensus_threshold and weight > 0:
                valid_signals.append(signal)
                valid_weights.append(weight)
        
        if len(valid_signals) < self.min_strategies:
            # Not enough consensus, reduce position
            return CombinedSignal(
                direction=0.0,
                strength=0.1,
                weighted_expected_return=0.0,
                weighted_risk=0.0,
                active_strategies=len(valid_signals),
                timestamp_ns=time.time_ns()
            )
        
        # Normalize weights
        total_weight = sum(valid_weights)
        normalized_weights = [w / total_weight for w in valid_weights]
        
        # Calculate weighted combination
        weighted_direction = sum(
            s.direction * s.strength * w 
            for s, w in zip(valid_signals, normalized_weights)
        )
        
        weighted_strength = sum(
            s.strength * w 
            for s, w in zip(valid_signals, normalized_weights)
        )
        
        weighted_expected_return = sum(
            s.expected_return * w 
            for s, w in zip(valid_signals, normalized_weights)
        )
        
        weighted_risk = sum(
            s.risk_estimate * w 
            for s, w in zip(valid_signals, normalized_weights)
        )
        
        combined = CombinedSignal(
            direction=np.clip(weighted_direction, -1.0, 1.0),
            strength=np.clip(weighted_strength, 0.0, 1.0),
            weighted_expected_return=weighted_expected_return,
            weighted_risk=weighted_risk,
            active_strategies=len(valid_signals),
            timestamp_ns=time.time_ns()
        )
        
        self.signal_history.append(combined)
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)
        
        return combined
    
    async def update_strategy_performance(
        self, 
        strategy_id: str, 
        signal: Signal, 
        realized_return: float
    ) -> Dict:
        """Update strategy performance metrics with realized return"""
        if strategy_id in self.evaluators:
            result = await self.evaluators[strategy_id].evaluate_signal.remote(
                signal, realized_return
            )
            return result
        return {}
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current weights for all registered strategies"""
        # This would be called synchronously for monitoring
        pass
    
    def get_combined_signal_history(self) -> List[CombinedSignal]:
        """Return recent combined signal history"""
        return self.signal_history.copy()


class AlphaCombinerOrchestrator:
    """
    High-level orchestrator for the alpha combination system
    Manages Ray actors and coordinates signal flow
    """
    
    def __init__(self, num_workers: int = 4):
        if not ray.is_initialized:
            ray.init(num_cpus=num_workers, ignore_reinit_error=True)
        self.combiner = AlphaCombiner.remote()
        self.strategy_types: Dict[str, StrategyType] = {}
    
    def add_strategy(self, strategy_id: str, strategy_type: StrategyType) -> None:
        """Add a new strategy to the ensemble"""
        self.strategy_types[strategy_id] = strategy_type
        ray.get(self.combiner.register_strategy.remote(strategy_id, strategy_type))
    
    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the ensemble"""
        if strategy_id in self.strategy_types:
            del self.strategy_types[strategy_id]
        ray.get(self.combiner.unregister_strategy.remote(strategy_id))
    
    async def process_signals(self, signals: List[Signal]) -> CombinedSignal:
        """Process incoming signals and return combined master signal"""
        return await self.combiner.combine_signals.remote(signals)
    
    async def record_outcome(
        self, 
        strategy_id: str, 
        original_signal: Signal,
        realized_return: float
    ) -> None:
        """Record the outcome of a signal for performance tracking"""
        await self.combiner.update_strategy_performance.remote(
            strategy_id, original_signal, realized_return
        )
    
    def shutdown(self) -> None:
        """Shutdown Ray cluster"""
        ray.shutdown()


# Example usage
async def example_usage():
    """Example of how to use the alpha combiner"""
    orchestrator = AlphaCombinerOrchestrator(num_workers=4)
    
    # Register strategies
    orchestrator.add_strategy("mm_btc", StrategyType.MARKET_MAKING)
    orchestrator.add_strategy("statarb_ethbtc", StrategyType.STAT_ARB)
    orchestrator.add_strategy("micro_vpin", StrategyType.MICROSTRUCTURE)
    
    # Create signals
    signals = [
        Signal(
            strategy_id="mm_btc",
            strategy_type=StrategyType.MARKET_MAKING,
            direction=0.5,
            strength=0.8,
            timestamp_ns=time.time_ns(),
            expected_return=0.001,
            risk_estimate=0.02
        ),
        Signal(
            strategy_id="statarb_ethbtc",
            strategy_type=StrategyType.STAT_ARB,
            direction=-0.3,
            strength=0.6,
            timestamp_ns=time.time_ns(),
            expected_return=0.0005,
            risk_estimate=0.015
        ),
    ]
    
    # Get combined signal
    combined = await orchestrator.process_signals(signals)
    print(f"Combined direction: {combined.direction}")
    print(f"Combined strength: {combined.strength}")
    
    # Later, record outcome
    await orchestrator.record_outcome("mm_btc", signals[0], 0.002)
    
    orchestrator.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
