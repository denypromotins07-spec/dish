"""
Cross-margin portfolio hedger.
Identifies offsetting positions across spot, perps, and options to minimize global margin requirements
and maximize capital efficiency on Binance/Bybit.

Memory-efficient implementation bounded for <14GB total system RAM.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class PositionType(Enum):
    SPOT = "spot"
    PERP = "perp"
    OPTION_CALL = "call"
    OPTION_PUT = "put"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(slots=True)
class Position:
    """Represents a trading position."""
    symbol: str
    position_type: PositionType
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    notional_value: float = field(init=False)
    delta: float = 1.0  # Options will have different delta
    
    def __post_init__(self):
        self.notional_value = self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity


@dataclass(slots=True)
class HedgeGroup:
    """Group of positions that hedge each other."""
    group_id: str
    positions: List[Position]
    net_delta: float
    net_notional: float
    margin_offset_pct: float  # Percentage of margin saved due to hedging


@dataclass(slots=True)
class MarginAnalysis:
    """Result of margin analysis."""
    total_gross_margin: float
    total_net_margin: float
    margin_savings: float
    savings_percentage: float
    hedge_groups: List[HedgeGroup]
    largest_exposure: str
    largest_exposure_size: float


class CrossMarginHedger:
    """
    Analyzes portfolio positions to identify hedging opportunities
    and calculate optimized margin requirements.
    """
    
    __slots__ = ('positions', 'margin_rates', 'correlation_threshold')
    
    def __init__(self, margin_rates: Optional[Dict[str, float]] = None, correlation_threshold: float = 0.7):
        self.positions: List[Position] = []
        self.margin_rates = margin_rates or {}
        self.correlation_threshold = correlation_threshold
    
    def add_position(
        self,
        symbol: str,
        position_type: PositionType,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        current_price: float,
        delta: float = 1.0
    ) -> None:
        """Add a position to the portfolio."""
        pos = Position(
            symbol=symbol,
            position_type=position_type,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            delta=delta
        )
        self.positions.append(pos)
    
    def clear_positions(self) -> None:
        """Clear all positions."""
        self.positions.clear()
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for comparison (e.g., BTCUSDT, BTC-PERP -> BTC)."""
        # Remove suffixes
        base = symbol.upper()
        for suffix in ['USDT', 'USDC', 'PERP', '-PERP', '-CALL', '-PUT']:
            base = base.replace(suffix, '')
        return base
    
    def _calculate_effective_delta(self, position: Position) -> float:
        """Calculate effective delta-adjusted exposure."""
        return position.notional_value * position.delta
    
    def identify_hedge_groups(self) -> List[HedgeGroup]:
        """
        Identify groups of positions that hedge each other.
        Groups positions by underlying asset and opposite sides.
        """
        # Group by normalized symbol
        symbol_groups: Dict[str, List[Position]] = {}
        
        for pos in self.positions:
            base_symbol = self._normalize_symbol(pos.symbol)
            if base_symbol not in symbol_groups:
                symbol_groups[base_symbol] = []
            symbol_groups[base_symbol].append(pos)
        
        hedge_groups = []
        group_id = 0
        
        for base_symbol, positions in symbol_groups.items():
            if len(positions) < 2:
                continue
            
            # Calculate net delta
            long_delta = sum(
                self._calculate_effective_delta(p) 
                for p in positions if p.side == PositionSide.LONG
            )
            short_delta = sum(
                self._calculate_effective_delta(p) 
                for p in positions if p.side == PositionSide.SHORT
            )
            
            net_delta = long_delta - short_delta
            net_notional = sum(
                self._calculate_effective_delta(p) for p in positions
            )
            
            # Calculate margin offset
            # If positions hedge, margin requirement is reduced
            gross_notional = long_delta + short_delta
            if gross_notional > 0:
                hedge_ratio = 1.0 - abs(net_delta) / gross_notional
                margin_offset_pct = hedge_ratio * 0.5  # Typically 50% of hedge gets offset
            else:
                margin_offset_pct = 0.0
            
            if margin_offset_pct > 0.1:  # At least 10% hedge
                hedge_groups.append(HedgeGroup(
                    group_id=f"hedge_{base_symbol}_{group_id}",
                    positions=positions,
                    net_delta=net_delta,
                    net_notional=net_notional,
                    margin_offset_pct=margin_offset_pct
                ))
                group_id += 1
        
        return hedge_groups
    
    def calculate_margin_requirements(self) -> MarginAnalysis:
        """
        Calculate gross and net margin requirements.
        """
        # Get hedge groups
        hedge_groups = self.identify_hedge_groups()
        hedged_symbols = set()
        for group in hedge_groups:
            for pos in group.positions:
                hedged_symbols.add(self._normalize_symbol(pos.symbol))
        
        # Calculate gross margin (no hedging benefit)
        total_gross_margin = 0.0
        for pos in self.positions:
            margin_rate = self.margin_rates.get(
                self._normalize_symbol(pos.symbol), 
                0.1  # Default 10% margin
            )
            total_gross_margin += pos.notional_value * margin_rate
        
        # Calculate net margin (with hedging benefit)
        total_net_margin = 0.0
        processed_positions = set()
        
        # First, process hedged groups
        for group in hedge_groups:
            group_gross = 0.0
            for pos in group.positions:
                margin_rate = self.margin_rates.get(
                    self._normalize_symbol(pos.symbol),
                    0.1
                )
                group_gross += pos.notional_value * margin_rate
                processed_positions.add(id(pos))
            
            # Apply hedge offset
            group_net = group_gross * (1.0 - group.margin_offset_pct)
            total_net_margin += group_net
        
        # Then, process unhedged positions
        for pos in self.positions:
            if id(pos) in processed_positions:
                continue
            
            margin_rate = self.margin_rates.get(
                self._normalize_symbol(pos.symbol),
                0.1
            )
            total_net_margin += pos.notional_value * margin_rate
        
        # Find largest exposure
        exposures: Dict[str, float] = {}
        for pos in self.positions:
            base = self._normalize_symbol(pos.symbol)
            delta_adj = self._calculate_effective_delta(pos)
            if base not in exposures:
                exposures[base] = 0.0
            if pos.side == PositionSide.LONG:
                exposures[base] += delta_adj
            else:
                exposures[base] -= delta_adj
        
        largest_exposure = max(exposures.keys(), key=lambda k: abs(exposures[k])) if exposures else "N/A"
        largest_exposure_size = exposures.get(largest_exposure, 0.0)
        
        # Calculate savings
        margin_savings = total_gross_margin - total_net_margin
        savings_pct = margin_savings / total_gross_margin if total_gross_margin > 0 else 0.0
        
        return MarginAnalysis(
            total_gross_margin=total_gross_margin,
            total_net_margin=total_net_margin,
            margin_savings=margin_savings,
            savings_percentage=savings_pct,
            hedge_groups=hedge_groups,
            largest_exposure=largest_exposure,
            largest_exposure_size=largest_exposure_size
        )
    
    def get_recommended_hedges(self, target_delta_range: Tuple[float, float] = (-0.1, 0.1)) -> List[Dict]:
        """
        Get recommended hedge trades to bring portfolio delta within target range.
        """
        recommendations = []
        
        # Calculate current delta by symbol
        deltas: Dict[str, float] = {}
        prices: Dict[str, float] = {}
        
        for pos in self.positions:
            base = self._normalize_symbol(pos.symbol)
            delta_adj = self._calculate_effective_delta(pos)
            
            if base not in deltas:
                deltas[base] = 0.0
                prices[base] = pos.current_price
            
            if pos.side == PositionSide.LONG:
                deltas[base] += delta_adj
            else:
                deltas[base] -= delta_adj
        
        # Check each symbol against target range
        for symbol, delta in deltas.items():
            if delta < target_delta_range[0] or delta > target_delta_range[1]:
                # Need to hedge
                price = prices.get(symbol, 1.0)
                
                if delta > 0:
                    # Long bias - recommend short hedge
                    hedge_quantity = delta / price
                    recommendations.append({
                        'symbol': f"{symbol}-PERP",
                        'action': 'SELL',
                        'quantity': abs(hedge_quantity),
                        'reason': f'Reduce {symbol} long exposure of ${delta:.2f}'
                    })
                else:
                    # Short bias - recommend long hedge
                    hedge_quantity = abs(delta) / price
                    recommendations.append({
                        'symbol': f"{symbol}-PERP",
                        'action': 'BUY',
                        'quantity': abs(hedge_quantity),
                        'reason': f'Reduce {symbol} short exposure of ${abs(delta):.2f}'
                    })
        
        return recommendations
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary of portfolio positions and hedges."""
        analysis = self.calculate_margin_requirements()
        
        long_notional = sum(
            p.notional_value for p in self.positions if p.side == PositionSide.LONG
        )
        short_notional = sum(
            p.notional_value for p in self.positions if p.side == PositionSide.SHORT
        )
        
        return {
            'total_positions': len(self.positions),
            'long_notional': long_notional,
            'short_notional': short_notional,
            'net_notional': long_notional - short_notional,
            'gross_margin': analysis.total_gross_margin,
            'net_margin': analysis.total_net_margin,
            'margin_savings': analysis.margin_savings,
            'savings_pct': analysis.savings_percentage,
            'hedge_groups': len(analysis.hedge_groups),
            'largest_exposure': {
                'symbol': analysis.largest_exposure,
                'size': analysis.largest_exposure_size
            }
        }


# Example usage
if __name__ == '__main__':
    hedger = CrossMarginHedger(margin_rates={'BTC': 0.1, 'ETH': 0.1, 'SOL': 0.15})
    
    # Add some positions
    hedger.add_position('BTCUSDT', PositionType.SPOT, PositionSide.LONG, 1.0, 45000, 46000)
    hedger.add_position('BTC-PERP', PositionType.PERP, PositionSide.SHORT, 0.5, 45500, 46000)
    hedger.add_position('ETHUSDT', PositionType.SPOT, PositionSide.LONG, 10.0, 2800, 2900)
    hedger.add_position('ETH-PERP', PositionType.PERP, PositionSide.SHORT, 5.0, 2850, 2900)
    hedger.add_position('SOLUSDT', PositionType.SPOT, PositionSide.LONG, 100.0, 100, 105)
    
    # Get analysis
    summary = hedger.get_portfolio_summary()
    print("Portfolio Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get hedge recommendations
    recommendations = hedger.get_recommended_hedges()
    print("\nHedge Recommendations:")
    for rec in recommendations:
        print(f"  {rec['action']} {rec['quantity']:.4f} {rec['symbol']} - {rec['reason']}")
