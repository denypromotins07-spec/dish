# python/journal/tagging_engine.py
"""
Automated metadata tagging engine for trade journal.
Appends contextual tags to every trade for multi-dimensional filtering.
Memory-efficient with streaming tag computation.
"""

from __future__ import annotations
import polars as pl
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from enum import Enum, auto
from collections import deque
import time


class TagCategory(Enum):
    """Categories of tags for organization."""
    MARKET_REGIME = auto()
    STRATEGY = auto()
    EXECUTION = auto()
    RISK = auto()
    COST = auto()
    TIMING = auto()
    LIQUIDITY = auto()
    CUSTOM = auto()


@dataclass
class TradeTag:
    """A single tag applied to a trade."""
    name: str
    value: str
    category: TagCategory
    confidence: float  # 0-1 confidence in tag applicability
    source: str        # Which rule/model generated this tag
    timestamp: int     # When tag was computed
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "category": self.category.name,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass
class TaggedTrade:
    """A trade with all its associated tags."""
    order_id: int
    timestamp_ns: int
    tags: List[TradeTag] = field(default_factory=list)
    
    def add_tag(self, tag: TradeTag) -> None:
        self.tags.append(tag)
    
    def get_tags_by_category(self, category: TagCategory) -> List[TradeTag]:
        return [t for t in self.tags if t.category == category]
    
    def has_tag(self, name: str) -> bool:
        return any(t.name == name for t in self.tags)
    
    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "timestamp_ns": self.timestamp_ns,
            "tags": [t.to_dict() for t in self.tags],
        }


class TagRule:
    """Base class for tag generation rules."""
    
    def __init__(self, name: str, category: TagCategory):
        self.name = name
        self.category = category
        self.hit_count: int = 0
        self.last_hit_time: int = 0
    
    def evaluate(self, trade_data: Dict[str, Any]) -> Optional[TradeTag]:
        """
        Evaluate if this rule applies to the trade.
        
        Args:
            trade_data: Dictionary of trade attributes
            
        Returns:
            TradeTag if rule applies, None otherwise
        """
        raise NotImplementedError


class MarketRegimeRule(TagRule):
    """Tags trades based on market regime at execution time."""
    
    def __init__(
        self,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.01,
    ):
        super().__init__("market_regime", TagCategory.MARKET_REGIME)
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
    
    def evaluate(self, trade_data: Dict[str, Any]) -> Optional[TradeTag]:
        volatility = trade_data.get("realized_volatility", 0.0)
        trend = trade_data.get("trend_signal", 0.0)
        
        # Determine regime
        regimes = []
        
        if volatility > self.volatility_threshold * 2:
            regimes.append("EXTREME_VOL")
        elif volatility > self.volatility_threshold:
            regimes.append("HIGH_VOL")
        else:
            regimes.append("LOW_VOL")
        
        if abs(trend) > self.trend_threshold * 2:
            direction = "STRONG_BULL" if trend > 0 else "STRONG_BEAR"
            regimes.append(direction)
        elif abs(trend) > self.trend_threshold:
            direction = "BULL" if trend > 0 else "BEAR"
            regimes.append(direction)
        else:
            regimes.append("SIDEWAYS")
        
        regime_str = "_".join(regimes)
        
        return TradeTag(
            name="market_regime",
            value=regime_str,
            category=self.category,
            confidence=0.9,
            source="market_regime_rule",
            timestamp=int(time.time_ns()),
        )


class StrategyTagRule(TagRule):
    """Tags trades with strategy metadata."""
    
    def __init__(self, strategy_map: Dict[int, str]):
        super().__init__("strategy", TagCategory.STRATEGY)
        self.strategy_map = strategy_map  # strategy_id -> name
    
    def evaluate(self, trade_data: Dict[str, Any]) -> Optional[TradeTag]:
        strategy_id = trade_data.get("strategy_id")
        if strategy_id is None:
            return None
        
        strategy_name = self.strategy_map.get(strategy_id, f"UNKNOWN_{strategy_id}")
        
        return TradeTag(
            name="strategy",
            value=strategy_name,
            category=self.category,
            confidence=1.0,
            source="strategy_map",
            timestamp=int(time.time_ns()),
        )


class FundingRateRule(TagRule):
    """Tags trades based on funding rate conditions."""
    
    def __init__(
        self,
        extreme_threshold: float = 0.001,  # 0.1% per 8hr
    ):
        super().__init__("funding_rate", TagCategory.LIQUIDITY)
        self.extreme_threshold = extreme_threshold
    
    def evaluate(self, trade_data: Dict[str, Any]) -> Optional[TradeTag]:
        funding_rate = trade_data.get("funding_rate", 0.0)
        
        if funding_rate > self.extreme_threshold:
            value = "EXTREME_POSITIVE"
        elif funding_rate < -self.extreme_threshold:
            value = "EXTREME_NEGATIVE"
        elif funding_rate > 0:
            value = "POSITIVE"
        elif funding_rate < 0:
            value = "NEGATIVE"
        else:
            value = "NEUTRAL"
        
        return TradeTag(
            name="funding_rate",
            value=value,
            category=self.category,
            confidence=1.0,
            source="funding_rate_rule",
            timestamp=int(time.time_ns()),
        )


class ExecutionQualityRule(TagRule):
    """Tags trades based on execution quality metrics."""
    
    def __init__(
        self,
        slippage_threshold_bps: float = 5.0,
    ):
        super().__init__("execution_quality", TagCategory.EXECUTION)
        self.slippage_threshold_bps = slippage_threshold_bps
    
    def evaluate(self, trade_data: Dict[str, Any]) -> Optional[TradeTag]:
        slippage_bps = trade_data.get("slippage_bps", 0.0)
        
        if slippage_bps < 0:
            # Negative slippage = we beat the benchmark
            if slippage_bps < -self.slippage_threshold_bps:
                value = "EXCELLENT"
            else:
                value = "GOOD"
        elif slippage_bps < self.slippage_threshold_bps:
            value = "ACCEPTABLE"
        elif slippage_bps < self.slippage_threshold_bps * 2:
            value = "POOR"
        else:
            value = "VERY_POOR"
        
        return TradeTag(
            name="execution_quality",
            value=value,
            category=self.category,
            confidence=0.95,
            source="execution_quality_rule",
            timestamp=int(time.time_ns()),
        )


class TimeOfDayRule(TagRule):
    """Tags trades based on time of day / session."""
    
    SESSIONS = {
        "ASIA": (0, 8),      # 00:00 - 08:00 UTC
        "EUROPE": (7, 16),   # 07:00 - 16:00 UTC
        "US": (13, 22),      # 13:00 - 22:00 UTC
        "OVERLAP_EU_US": (13, 16),  # 13:00 - 16:00 UTC
        "OVERLAP_ASIA_EU": (7, 8),  # 07:00 - 08:00 UTC
    }
    
    def evaluate(self, trade_data: Dict[str, Any]) -> Optional[TradeTag]:
        timestamp_ns = trade_data.get("timestamp_ns", 0)
        hour = (timestamp_ns // 3_600_000_000_000) % 24
        
        sessions = []
        for session_name, (start, end) in self.SESSIONS.items():
            if start <= hour < end:
                sessions.append(session_name)
        
        if not sessions:
            sessions.append("QUIET")
        
        return TradeTag(
            name="session",
            value="_".join(sessions),
            category=TagCategory.TIMING,
            confidence=1.0,
            source="time_of_day_rule",
            timestamp=int(time.time_ns()),
        )


class RiskLevelRule(TagRule):
    """Tags trades based on risk level."""
    
    def __init__(
        self,
        var_threshold_pct: float = 2.0,
        position_size_threshold: float = 0.1,  # 10% of portfolio
    ):
        super().__init__("risk_level", TagCategory.RISK)
        self.var_threshold_pct = var_threshold_pct
        self.position_size_threshold = position_size_threshold
    
    def evaluate(self, trade_data: Dict[str, Any]) -> Optional[TradeTag]:
        var_pct = trade_data.get("var_contribution_pct", 0.0)
        position_size = trade_data.get("position_size_pct", 0.0)
        
        risk_score = 0
        if var_pct > self.var_threshold_pct * 2:
            risk_score += 2
        elif var_pct > self.var_threshold_pct:
            risk_score += 1
        
        if position_size > self.position_size_threshold * 2:
            risk_score += 2
        elif position_size > self.position_size_threshold:
            risk_score += 1
        
        if risk_score >= 4:
            value = "CRITICAL"
        elif risk_score >= 3:
            value = "HIGH"
        elif risk_score >= 2:
            value = "MEDIUM"
        else:
            value = "LOW"
        
        return TradeTag(
            name="risk_level",
            value=value,
            category=self.category,
            confidence=0.85,
            source="risk_level_rule",
            timestamp=int(time.time_ns()),
        )


class TaggingEngine:
    """
    Main tagging engine that applies multiple rules to trades.
    
    Features:
    - Pluggable rule system
    - Memory-bounded history
    - Batch processing with Polars
    - Streaming tag computation
    """
    
    def __init__(self, max_history_size: int = 100_000):
        """
        Initialize the tagging engine.
        
        Args:
            max_history_size: Maximum tagged trades to keep in memory
        """
        self._rules: List[TagRule] = []
        self._history: deque[TaggedTrade] = deque(maxlen=max_history_size)
        self._tag_counts: Dict[str, int] = {}
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register the default set of tagging rules."""
        self._rules = [
            MarketRegimeRule(),
            StrategyTagRule({
                1: "StatArb",
                2: "Momentum",
                3: "MeanReversion",
                4: "MarketMaking",
                5: "TrendFollowing",
            }),
            FundingRateRule(),
            ExecutionQualityRule(),
            TimeOfDayRule(),
            RiskLevelRule(),
        ]
    
    def add_rule(self, rule: TagRule) -> None:
        """Add a custom tagging rule."""
        self._rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                return True
        return False
    
    def tag_trade(self, trade_data: Dict[str, Any]) -> TaggedTrade:
        """
        Apply all rules to a single trade.
        
        Args:
            trade_data: Dictionary containing trade attributes
            
        Returns:
            TaggedTrade with all applicable tags
        """
        order_id = trade_data.get("order_id", 0)
        timestamp_ns = trade_data.get("timestamp_ns", 0)
        
        tagged = TaggedTrade(order_id=order_id, timestamp_ns=timestamp_ns)
        
        for rule in self._rules:
            try:
                tag = rule.evaluate(trade_data)
                if tag:
                    tagged.add_tag(tag)
                    rule.hit_count += 1
                    rule.last_hit_time = timestamp_ns
                    
                    # Update counts
                    tag_key = f"{tag.category.name}:{tag.name}"
                    self._tag_counts[tag_key] = self._tag_counts.get(tag_key, 0) + 1
            except Exception as e:
                # Log error but continue with other rules
                pass
        
        self._history.append(tagged)
        return tagged
    
    def tag_batch_polars(
        self,
        df: pl.DataFrame,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> pl.DataFrame:
        """
        Tag a batch of trades using Polars for efficiency.
        
        Args:
            df: DataFrame with trade data
            context_data: Additional context (market regime, etc.)
            
        Returns:
            DataFrame with added 'tags' column (list of dicts)
        """
        context = context_data or {}
        
        # Convert to rows for rule evaluation
        rows = df.to_dicts()
        all_tags = []
        
        for row in rows:
            # Merge row data with context
            trade_data = {**row, **context}
            tagged = self.tag_trade(trade_data)
            all_tags.append([t.to_dict() for t in tagged.tags])
        
        # Add tags column
        return df.with_columns(
            pl.Series("tags", all_tags, dtype=pl.List(pl.Struct))
        )
    
    def get_trades_by_tag(
        self,
        tag_name: str,
        tag_value: Optional[str] = None,
    ) -> List[TaggedTrade]:
        """
        Retrieve trades matching a specific tag.
        
        Args:
            tag_name: Name of tag to filter
            tag_value: Optional value to match
            
        Returns:
            List of matching TaggedTrade objects
        """
        results = []
        for trade in self._history:
            for tag in trade.tags:
                if tag.name == tag_name:
                    if tag_value is None or tag.value == tag_value:
                        results.append(trade)
                        break
        return results
    
    def get_tag_statistics(self) -> Dict[str, Any]:
        """Get statistics about tag usage."""
        total_trades = len(self._history)
        
        rule_stats = []
        for rule in self._rules:
            hit_rate = rule.hit_count / total_trades if total_trades > 0 else 0
            rule_stats.append({
                "rule_name": rule.name,
                "category": rule.category.name,
                "hit_count": rule.hit_count,
                "hit_rate": hit_rate,
                "last_hit_time": rule.last_hit_time,
            })
        
        return {
            "total_trades_tagged": total_trades,
            "unique_tags": len(self._tag_counts),
            "tag_distribution": dict(self._tag_counts),
            "rule_statistics": rule_stats,
        }
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """
        Get available filter options for UI.
        
        Returns:
            Dict mapping tag names to list of observed values
        """
        filters: Dict[str, set] = {}
        
        for trade in self._history:
            for tag in trade.tags:
                if tag.name not in filters:
                    filters[tag.name] = set()
                filters[tag.name].add(tag.value)
        
        return {k: sorted(list(v)) for k, v in filters.items()}


def create_custom_rule(
    name: str,
    category: TagCategory,
    condition: Callable[[Dict[str, Any]], Optional[str]],
) -> TagRule:
    """
    Factory function to create a custom tag rule from a condition function.
    
    Args:
        name: Rule name
        category: Tag category
        condition: Function that returns tag value or None
        
    Returns:
        Configured TagRule
    """
    class CustomRule(TagRule):
        def __init__(self):
            super().__init__(name, category)
            self._condition = condition
        
        def evaluate(self, trade_data: Dict[str, Any]) -> Optional[TradeTag]:
            value = self._condition(trade_data)
            if value is None:
                return None
            
            return TradeTag(
                name=name,
                value=value,
                category=self.category,
                confidence=0.8,
                source=f"custom:{name}",
                timestamp=int(time.time_ns()),
            )
    
    return CustomRule()


if __name__ == "__main__":
    # Example usage
    engine = TaggingEngine()
    
    # Tag a sample trade
    trade_data = {
        "order_id": 12345,
        "timestamp_ns": int(time.time_ns()),
        "strategy_id": 1,
        "price": 50000.0,
        "quantity": 0.5,
        "side": 0,
        "realized_volatility": 0.025,
        "trend_signal": 0.015,
        "funding_rate": 0.0008,
        "slippage_bps": 3.2,
        "var_contribution_pct": 1.5,
        "position_size_pct": 0.05,
    }
    
    tagged = engine.tag_trade(trade_data)
    
    print(f"Trade {tagged.order_id} tags:")
    for tag in tagged.tags:
        print(f"  [{tag.category.name}] {tag.name} = {tag.value}")
    
    # Statistics
    stats = engine.get_tag_statistics()
    print(f"\nTotal trades: {stats['total_trades_tagged']}")
    print(f"Unique tags: {stats['unique_tags']}")
