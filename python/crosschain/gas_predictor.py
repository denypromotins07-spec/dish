"""
EIP-1559 base fee and priority fee predictor across L1 and L2 networks.
Uses time-series forecasting to optimize transaction submission costs.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


class NetworkType(Enum):
    """Network types for gas prediction."""
    L1_ETHEREUM = "l1_ethereum"
    L2_ARBITRUM = "l2_arbitrum"
    L2_OPTIMISM = "l2_optimism"
    L2_BASE = "l2_base"
    L1_POLYGON = "l1_polygon"


@dataclass
class GasSample:
    """Single gas price sample."""
    timestamp: datetime
    block_number: int
    base_fee_gwei: float
    priority_fee_gwei: float
    total_gas_price_gwei: float
    gas_used: int
    gas_limit: int
    utilization_percent: float


@dataclass
class GasPrediction:
    """Predicted gas prices for different time horizons."""
    network: NetworkType
    prediction_time: datetime
    horizon_minutes: int
    predicted_base_fee_gwei: float
    predicted_priority_fee_gwei: float
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: float
    recommended_action: str  # "submit_now", "wait", "urgent"


@dataclass
class FeeOptimization:
    """Optimal fee recommendation for transaction submission."""
    optimal_base_fee_gwei: float
    optimal_priority_fee_gwei: float
    expected_confirmation_time_seconds: float
    success_probability: float
    cost_savings_percent: float


class GasPredictor:
    """
    EIP-1559 base fee and priority fee predictor using time-series forecasting.
    Optimizes transaction submission costs across L1 and L2 networks.
    """

    def __init__(
        self,
        history_size: int = 1000,
        target_utilization: float = 0.5,
    ):
        self.history_size = history_size
        self.target_utilization = target_utilization
        
        # Per-network gas history
        self._gas_history: Dict[NetworkType, deque] = {
            net: deque(maxlen=history_size) for net in NetworkType
        }
        
        # Model parameters (would be trained in production)
        self._model_params = {
            NetworkType.L1_ETHEREUM: {"volatility": 0.15, "mean_reversion": 0.1},
            NetworkType.L2_ARBITRUM: {"volatility": 0.08, "mean_reversion": 0.2},
            NetworkType.L2_OPTIMISM: {"volatility": 0.07, "mean_reversion": 0.2},
            NetworkType.L2_BASE: {"volatility": 0.06, "mean_reversion": 0.25},
            NetworkType.L1_POLYGON: {"volatility": 0.12, "mean_reversion": 0.15},
        }
        
        # Default base fees per network
        self._base_defaults = {
            NetworkType.L1_ETHEREUM: 20.0,
            NetworkType.L2_ARBITRUM: 0.1,
            NetworkType.L2_OPTIMISM: 0.05,
            NetworkType.L2_BASE: 0.05,
            NetworkType.L1_POLYGON: 30.0,
        }

    def add_gas_sample(
        self,
        network: NetworkType,
        block_number: int,
        base_fee_gwei: float,
        priority_fee_gwei: float,
        gas_used: int,
        gas_limit: int,
    ) -> GasSample:
        """Add a new gas sample for analysis."""
        utilization = gas_used / gas_limit if gas_limit > 0 else 0
        total_price = base_fee_gwei + priority_fee_gwei
        
        sample = GasSample(
            timestamp=datetime.utcnow(),
            block_number=block_number,
            base_fee_gwei=base_fee_gwei,
            priority_fee_gwei=priority_fee_gwei,
            total_gas_price_gwei=total_price,
            gas_used=gas_used,
            gas_limit=gas_limit,
            utilization_percent=utilization * 100,
        )
        
        self._gas_history[network].append(sample)
        return sample

    def predict_gas(
        self,
        network: NetworkType,
        horizon_minutes: int = 5,
    ) -> GasPrediction:
        """Predict gas prices for a future time horizon."""
        history = list(self._gas_history[network])
        
        if len(history) < 10:
            # Not enough data, use defaults
            base_default = self._base_defaults.get(network, 10.0)
            return GasPrediction(
                network=network,
                prediction_time=datetime.utcnow(),
                horizon_minutes=horizon_minutes,
                predicted_base_fee_gwei=base_default,
                predicted_priority_fee_gwei=base_default * 0.1,
                confidence_interval_low=base_default * 0.5,
                confidence_interval_high=base_default * 2.0,
                confidence_level=0.3,
                recommended_action="wait",
            )
        
        # Extract time series
        base_fees = [s.base_fee_gwei for s in history]
        priority_fees = [s.priority_fee_gwei for s in history]
        utilizations = [s.utilization_percent / 100 for s in history]
        
        # Simple AR(1) prediction with mean reversion
        params = self._model_params.get(network, {"volatility": 0.1, "mean_reversion": 0.1})
        
        # Calculate mean and recent trend
        mean_base = sum(base_fees[-50:]) / min(len(base_fees[-50:]), 50)
        recent_base = base_fees[-1]
        
        # Mean reversion prediction
        reversion_speed = params["mean_reversion"]
        predicted_base = mean_base + (recent_base - mean_base) * (1 - reversion_speed)
        
        # Adjust for utilization trend
        recent_util = sum(utilizations[-5:]) / 5
        if recent_util > 0.8:
            predicted_base *= 1.2  # High utilization -> higher fees
        elif recent_util < 0.3:
            predicted_base *= 0.8  # Low utilization -> lower fees
        
        # Priority fee prediction (based on congestion)
        mean_priority = sum(priority_fees[-50:]) / min(len(priority_fees[-50:]), 50)
        predicted_priority = mean_priority * (recent_util / self.target_utilization)
        
        # Confidence interval based on volatility
        volatility = params["volatility"]
        ci_width = predicted_base * volatility * math.sqrt(horizon_minutes / 5)
        
        # Determine recommended action
        current_base = base_fees[-1]
        if predicted_base < current_base * 0.8:
            action = "wait"
        elif predicted_base > current_base * 1.3:
            action = "submit_now"
        elif recent_util > 0.9:
            action = "urgent"
        else:
            action = "submit_now"
        
        return GasPrediction(
            network=network,
            prediction_time=datetime.utcnow(),
            horizon_minutes=horizon_minutes,
            predicted_base_fee_gwei=max(0.001, predicted_base),
            predicted_priority_fee_gwei=max(0.001, predicted_priority),
            confidence_interval_low=max(0.001, predicted_base - ci_width),
            confidence_interval_high=predicted_base + ci_width,
            confidence_level=min(0.95, 0.5 + len(history) / 2000),
            recommended_action=action,
        )

    def optimize_fee(
        self,
        network: NetworkType,
        urgency: str = "normal",  # "low", "normal", "high", "urgent"
        max_fee_gwei: Optional[float] = None,
    ) -> FeeOptimization:
        """Calculate optimal fee for transaction submission."""
        prediction = self.predict_gas(network, horizon_minutes=5)
        
        # Urgency multipliers
        urgency_multipliers = {
            "low": 0.7,
            "normal": 1.0,
            "high": 1.5,
            "urgent": 2.5,
        }
        multiplier = urgency_multipliers.get(urgency, 1.0)
        
        # Calculate optimal fees
        optimal_base = prediction.predicted_base_fee_gwei * multiplier
        optimal_priority = prediction.predicted_priority_fee_gwei * multiplier
        
        # Apply max fee constraint
        if max_fee_gwei:
            total = optimal_base + optimal_priority
            if total > max_fee_gwei:
                scale = max_fee_gwei / total
                optimal_base *= scale
                optimal_priority *= scale
        
        # Estimate confirmation time
        history = list(self._gas_history[network])
        if history:
            avg_util = sum(s.utilization_percent for s in history[-20:]) / min(len(history[-20:]), 20)
            base_time = 12 if network == NetworkType.L1_ETHEREUM else 2
            
            if urgency == "urgent":
                confirm_time = base_time * 1.5
            elif urgency == "high":
                confirm_time = base_time * 2
            elif urgency == "normal":
                confirm_time = base_time * 3
            else:
                confirm_time = base_time * 6
        else:
            confirm_time = 60  # Default 1 minute
        
        # Success probability based on fee level
        current_total = prediction.predicted_base_fee_gwei + prediction.predicted_priority_fee_gwei
        proposed_total = optimal_base + optimal_priority
        success_prob = min(0.99, 0.5 + (proposed_total / current_total) * 0.4) if current_total > 0 else 0.5
        
        # Cost savings vs using current high fees
        if history:
            avg_total = sum(s.total_gas_price_gwei for s in history[-20:]) / min(len(history[-20:]), 20)
            savings = max(0, (avg_total - proposed_total) / avg_total * 100) if avg_total > 0 else 0
        else:
            savings = 0
        
        return FeeOptimization(
            optimal_base_fee_gwei=optimal_base,
            optimal_priority_fee_gwei=optimal_priority,
            expected_confirmation_time_seconds=confirm_time,
            success_probability=success_prob,
            cost_savings_percent=savings,
        )

    def get_gas_trend(
        self,
        network: NetworkType,
        lookback_blocks: int = 100,
    ) -> Dict:
        """Analyze gas price trend over recent blocks."""
        history = list(self._gas_history[network])
        if len(history) < 2:
            return {"trend": "insufficient_data"}
        
        samples = history[-lookback_blocks:]
        base_fees = [s.base_fee_gwei for s in samples]
        
        # Calculate trend direction
        first_half_avg = sum(base_fees[:len(base_fees)//2]) / (len(base_fees)//2)
        second_half_avg = sum(base_fees[len(base_fees)//2:]) / (len(base_fees) - len(base_fees)//2)
        
        if second_half_avg > first_half_avg * 1.1:
            trend = "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Calculate volatility
        mean = sum(base_fees) / len(base_fees)
        variance = sum((x - mean) ** 2 for x in base_fees) / len(base_fees)
        volatility = math.sqrt(variance) / mean if mean > 0 else 0
        
        return {
            "trend": trend,
            "current_base_fee": base_fees[-1],
            "average_base_fee": mean,
            "min_base_fee": min(base_fees),
            "max_base_fee": max(base_fees),
            "volatility": volatility,
            "sample_count": len(samples),
        }

    def get_best_submission_window(
        self,
        network: NetworkType,
        next_hours: int = 6,
    ) -> Dict:
        """Predict the best time window for transaction submission."""
        # This would use more sophisticated forecasting in production
        prediction = self.predict_gas(network, horizon_minutes=30)
        
        # Simple heuristic based on current state
        current_hour = datetime.utcnow().hour
        
        # Historical patterns (UTC):
        # - Lowest activity: 02:00-06:00 UTC (weekend especially)
        # - Highest activity: 14:00-18:00 UTC (US trading hours)
        
        if 2 <= current_hour <= 6:
            window_quality = "excellent"
            expected_savings = 30
        elif 14 <= current_hour <= 18:
            window_quality = "poor"
            expected_savings = -20
        else:
            window_quality = "moderate"
            expected_savings = 0
        
        return {
            "current_hour_utc": current_hour,
            "window_quality": window_quality,
            "expected_savings_percent": expected_savings,
            "recommendation": f"Consider waiting for off-peak hours" if window_quality == "poor" else "Good time to submit",
            "prediction_confidence": prediction.confidence_level,
        }

    def get_statistics(self) -> Dict:
        """Get predictor statistics."""
        return {
            "networks": {
                net.name: {
                    "samples": len(self._gas_history[net]),
                    "max_samples": self.history_size,
                }
                for net in NetworkType
            },
            "model_params": {
                net.name: params for net, params in self._model_params.items()
            },
        }


async def main():
    """Example usage of GasPredictor."""
    predictor = GasPredictor()
    
    # Add some historical samples
    for i in range(100):
        predictor.add_gas_sample(
            network=NetworkType.L1_ETHEREUM,
            block_number=18000000 + i,
            base_fee_gwei=20 + (i % 20) - 10 + (i * 0.1),
            priority_fee_gwei=2 + (i % 5),
            gas_used=12_000_000 + (i % 3_000_000),
            gas_limit=15_000_000,
        )
    
    # Get prediction
    prediction = predictor.predict_gas(NetworkType.L1_ETHEREUM, horizon_minutes=10)
    print(f"Gas Prediction (10 min):")
    print(f"  Base Fee: {prediction.predicted_base_fee_gwei:.2f} gwei")
    print(f"  Priority Fee: {prediction.predicted_priority_fee_gwei:.2f} gwei")
    print(f"  Action: {prediction.recommended_action}")
    
    # Optimize fee
    optimization = predictor.optimize_fee(
        NetworkType.L1_ETHEREUM,
        urgency="normal",
        max_fee_gwei=50,
    )
    print(f"\nOptimized Fee:")
    print(f"  Base: {optimization.optimal_base_fee_gwei:.2f} gwei")
    print(f"  Priority: {optimization.optimal_priority_fee_gwei:.2f} gwei")
    print(f"  Success Probability: {optimization.success_probability:.2%}")
    print(f"  Cost Savings: {optimization.cost_savings_percent:.1f}%")
    
    # Trend analysis
    trend = predictor.get_gas_trend(NetworkType.L1_ETHEREUM)
    print(f"\nTrend: {trend['trend']}")
    print(f"  Current: {trend.get('current_base_fee', 'N/A'):.2f} gwei")
    print(f"  Average: {trend.get('average_base_fee', 0):.2f} gwei")
    
    # Best submission window
    window = predictor.get_best_submission_window(NetworkType.L1_ETHEREUM)
    print(f"\nBest Window: {window['window_quality']}")
    print(f"  Recommendation: {window['recommendation']}")


if __name__ == "__main__":
    asyncio.run(main())
