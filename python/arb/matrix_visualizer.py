"""
Backend data formatter that compresses the high-dimensional cross-venue arbitrage matrix
into a lightweight JSON/Protobuf stream for the frontend's "Arbitrage Heatmap" dashboard.
Strictly memory-bounded to prevent RAM spikes during high-frequency updates.
"""

import json
import struct
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import IntEnum
import time


class VenueId(IntEnum):
    """Venue identifiers matching Rust enum"""
    BINANCE = 0
    BYBIT = 1
    OKX = 2
    COINBASE = 3
    KRAKEN = 4
    HUOBI = 5
    KUCOIN = 6
    GATEIO = 7


@dataclass(slots=True)
class ArbOpportunity:
    """Arbitrage opportunity data structure"""
    buy_venue: int
    sell_venue: int
    symbol_idx: int
    profit_bps: int
    recommended_size: int
    timestamp_ns: int


@dataclass(slots=True)
class VenueData:
    """Compressed venue data for frontend"""
    venue_id: int
    symbol_idx: int
    mid_price: int
    best_bid: int
    best_ask: int
    bid_depth: int
    ask_depth: int
    spread_bps: int


class MatrixVisualizer:
    """
    Compresses high-dimensional arbitrage matrix into lightweight streams.
    Uses Protobuf-like binary encoding for minimal bandwidth.
    """
    
    # Maximum opportunities to send per update (prevents frontend overload)
    MAX_OPPORTUNITIES_PER_UPDATE = 20
    # Maximum venues to track
    MAX_VENUES = 8
    # Maximum symbols to track
    MAX_SYMBOLS = 100
    
    def __init__(self):
        self._last_update_ns = 0
        self._opportunities_cache: List[ArbOpportunity] = []
        self._venue_data_cache: Dict[Tuple[int, int], VenueData] = {}
    
    def update_opportunities(self, opportunities: List[ArbOpportunity]) -> None:
        """
        Update cached opportunities, keeping only top N by profit.
        Memory-bounded operation.
        """
        # Sort by profit descending and keep top N
        sorted_opps = sorted(opportunities, key=lambda x: x.profit_bps, reverse=True)
        self._opportunities_cache = sorted_opps[:self.MAX_OPPORTUNITIES_PER_UPDATE]
        self._last_update_ns = time.time_ns()
    
    def update_venue_data(self, venue_id: int, symbol_idx: int, 
                          mid_price: int, best_bid: int, best_ask: int,
                          bid_depth: int, ask_depth: int, spread_bps: int) -> None:
        """Update cached venue data for a specific venue/symbol pair."""
        key = (venue_id, symbol_idx)
        self._venue_data_cache[key] = VenueData(
            venue_id=venue_id,
            symbol_idx=symbol_idx,
            mid_price=mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            spread_bps=spread_bps
        )
    
    def to_json(self) -> str:
        """
        Convert current state to compact JSON for WebSocket streaming.
        Uses minimal field names and integer encoding.
        """
        payload = {
            'ts': self._last_update_ns,
            'opps': [
                {
                    'bv': o.buy_venue,
                    'sv': o.sell_venue,
                    'sym': o.symbol_idx,
                    'prof': o.profit_bps,
                    'sz': o.recommended_size,
                }
                for o in self._opportunities_cache
            ],
            'venues': [
                {
                    'v': v.venue_id,
                    's': v.symbol_idx,
                    'mp': v.mid_price,
                    'bb': v.best_bid,
                    'ba': v.best_ask,
                    'bd': v.bid_depth,
                    'ad': v.ask_depth,
                    'sp': v.spread_bps,
                }
                for v in list(self._venue_data_cache.values())[:50]  # Limit venue data
            ]
        }
        return json.dumps(payload, separators=(',', ':'))
    
    def to_binary(self) -> bytes:
        """
        Convert to compact binary format (Protobuf-like).
        Format: [header][opportunities][venue_data]
        
        Header: timestamp (8 bytes) + opp_count (2 bytes) + venue_count (2 bytes)
        Opportunity: buy_venue (1) + sell_venue (1) + symbol (2) + profit (4) + size (8)
        Venue Data: venue (1) + symbol (2) + mid (8) + bid (8) + ask (8) + bid_d (8) + ask_d (8) + spread (4)
        """
        opps = self._opportunities_cache
        venues = list(self._venue_data_cache.values())[:50]
        
        # Calculate total size
        header_size = 12
        opp_size = 1 + 1 + 2 + 4 + 8  # 16 bytes per opportunity
        venue_size = 1 + 2 + 8 + 8 + 8 + 8 + 8 + 4  # 47 bytes per venue
        
        total_size = header_size + (len(opps) * opp_size) + (len(venues) * venue_size)
        buffer = bytearray(total_size)
        offset = 0
        
        # Write header
        struct.pack_into('<Q', buffer, offset, self._last_update_ns)
        offset += 8
        struct.pack_into('<H', buffer, offset, len(opps))
        offset += 2
        struct.pack_into('<H', buffer, offset, len(venues))
        offset += 2
        
        # Write opportunities
        for opp in opps:
            struct.pack_into('<B', buffer, offset, opp.buy_venue)
            offset += 1
            struct.pack_into('<B', buffer, offset, opp.sell_venue)
            offset += 1
            struct.pack_into('<H', buffer, offset, opp.symbol_idx)
            offset += 2
            struct.pack_into('<I', buffer, offset, opp.profit_bps)
            offset += 4
            struct.pack_into('<Q', buffer, offset, opp.recommended_size)
            offset += 8
        
        # Write venue data
        for venue in venues:
            struct.pack_into('<B', buffer, offset, venue.venue_id)
            offset += 1
            struct.pack_into('<H', buffer, offset, venue.symbol_idx)
            offset += 2
            struct.pack_into('<Q', buffer, offset, venue.mid_price)
            offset += 8
            struct.pack_into('<Q', buffer, offset, venue.best_bid)
            offset += 8
            struct.pack_into('<Q', buffer, offset, venue.best_ask)
            offset += 8
            struct.pack_into('<Q', buffer, offset, venue.bid_depth)
            offset += 8
            struct.pack_into('<Q', buffer, offset, venue.ask_depth)
            offset += 8
            struct.pack_into('<I', buffer, offset, venue.spread_bps)
            offset += 4
        
        return bytes(buffer)
    
    def get_heatmap_data(self) -> Dict[str, List[List[float]]]:
        """
        Generate 2D heatmap matrix for frontend visualization.
        Returns dict with 'matrix' key containing [venue x venue] profit values.
        """
        # Initialize NxN matrix with zeros
        matrix = [[0.0 for _ in range(self.MAX_VENUES)] for _ in range(self.MAX_VENUES)]
        
        # Fill in profit values from opportunities
        for opp in self._opportunities_cache:
            if opp.buy_venue < self.MAX_VENUES and opp.sell_venue < self.MAX_VENUES:
                # Store profit as decimal (bps / 10000)
                matrix[opp.buy_venue][opp.sell_venue] = opp.profit_bps / 10000.0
        
        return {'matrix': matrix}
    
    def clear_cache(self) -> None:
        """Clear all cached data - call periodically to prevent memory growth."""
        self._opportunities_cache.clear()
        self._venue_data_cache.clear()
    
    @property
    def opportunity_count(self) -> int:
        """Get current number of cached opportunities."""
        return len(self._opportunities_cache)
    
    @property
    def venue_data_count(self) -> int:
        """Get current number of cached venue data points."""
        return len(self._venue_data_cache)


# Singleton instance for global access
_visualizer_instance: Optional[MatrixVisualizer] = None


def get_visualizer() -> MatrixVisualizer:
    """Get or create singleton visualizer instance."""
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = MatrixVisualizer()
    return _visualizer_instance


if __name__ == '__main__':
    # Test/demo code
    viz = get_visualizer()
    
    # Add test opportunities
    test_opps = [
        ArbOpportunity(0, 1, 0, 25, 1000, time.time_ns()),
        ArbOpportunity(1, 2, 0, 15, 500, time.time_ns()),
        ArbOpportunity(0, 2, 1, 30, 2000, time.time_ns()),
    ]
    viz.update_opportunities(test_opps)
    
    # Add test venue data
    viz.update_venue_data(0, 0, 10000, 9999, 10001, 1000, 1000, 2)
    
    print(f"JSON output: {viz.to_json()}")
    print(f"Binary size: {len(viz.to_binary())} bytes")
    print(f"Heatmap: {viz.get_heatmap_data()}")
