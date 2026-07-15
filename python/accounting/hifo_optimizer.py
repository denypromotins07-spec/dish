"""
Highest-In, First-Out (HIFO) optimization logic.
Minimizes taxable gains by selecting the most expensive tax lots to close during profitable exits.
Strictly bounded in RAM usage with streaming processing.
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Iterator, Optional
from collections import defaultdict


@dataclass(order=True)
class TaxLot:
    """Represents a single tax lot for HIFO optimization."""
    cost_basis_per_unit: float  # For sorting (negative for max-heap behavior)
    lot_id: int = field(compare=False)
    instrument_id: int = field(compare=False)
    quantity: float = field(compare=False)
    total_cost_basis: float = field(compare=False)
    entry_timestamp_ns: int = field(compare=False)
    is_closed: bool = field(default=False, compare=False)


class HifoOptimizer:
    """
    Highest-In, First-Out optimizer for tax-efficient lot selection.
    
    Uses a heap-based approach for O(log n) lot selection while maintaining
    strict memory bounds through streaming processing.
    """
    
    def __init__(self, max_lots_per_instrument: int = 4096):
        self.max_lots_per_instrument = max_lots_per_instrument
        # Per-instrument heaps (negative cost for max-heap behavior)
        self._lot_heaps: dict[int, list[TaxLot]] = defaultdict(list)
        # All lots by ID for quick lookup
        self._lots_by_id: dict[int, TaxLot] = {}
        # Closed lot IDs for cleanup
        self._closed_lots: set[int] = set()
        # Memory tracking
        self._memory_footprint_bytes: int = 0
    
    def add_lot(
        self,
        lot_id: int,
        instrument_id: int,
        quantity: float,
        total_cost_basis: float,
        entry_timestamp_ns: int,
    ) -> None:
        """Add a new tax lot to the optimizer."""
        if len(self._lot_heaps[instrument_id]) >= self.max_lots_per_instrument:
            # Memory bound exceeded - force close oldest lot
            self._evict_oldest_lot(instrument_id)
        
        cost_per_unit = total_cost_basis / quantity if quantity > 0 else 0
        
        lot = TaxLot(
            cost_basis_per_unit=-cost_per_unit,  # Negative for max-heap
            lot_id=lot_id,
            instrument_id=instrument_id,
            quantity=quantity,
            total_cost_basis=total_cost_basis,
            entry_timestamp_ns=entry_timestamp_ns,
        )
        
        heapq.heappush(self._lot_heaps[instrument_id], lot)
        self._lots_by_id[lot_id] = lot
        
        # Update memory footprint estimate
        self._memory_footprint_bytes += 128  # Approximate size per lot
    
    def _evict_oldest_lot(self, instrument_id: int) -> None:
        """Evict the oldest lot when memory limit is reached."""
        heap = self._lot_heaps[instrument_id]
        if not heap:
            return
        
        # Find oldest lot (minimum timestamp)
        oldest = min(heap, key=lambda x: x.entry_timestamp_ns)
        oldest.is_closed = True
        self._closed_lots.add(oldest.lot_id)
        
        # Remove from heap (inefficient but rare operation)
        heap.remove(oldest)
        heapq.heapify(heap)
        
        if oldest.lot_id in self._lots_by_id:
            del self._lots_by_id[oldest.lot_id]
        
        self._memory_footprint_bytes -= 128
    
    def select_lots_hifo(
        self,
        instrument_id: int,
        quantity_to_close: float,
    ) -> Iterator[tuple[int, float]]:
        """
        Select lots using HIFO strategy to close specified quantity.
        
        Yields tuples of (lot_id, quantity_to_close_from_lot).
        Generator pattern ensures streaming/low-memory operation.
        """
        remaining = quantity_to_close
        heap = self._lot_heaps[instrument_id]
        
        # Create a temporary list to hold popped items
        temp_popped: list[TaxLot] = []
        
        try:
            while remaining > 0 and heap:
                # Get highest cost basis lot
                lot = heapq.heappop(heap)
                temp_popped.append(lot)
                
                if lot.is_closed:
                    continue
                
                close_qty = min(remaining, lot.quantity)
                
                if close_qty > 0:
                    yield (lot.lot_id, close_qty)
                    remaining -= close_qty
                    
                    # Update lot state
                    lot.quantity -= close_qty
                    lot.total_cost_basis -= close_qty * (-lot.cost_basis_per_unit)
                    
                    if lot.quantity <= 0:
                        lot.is_closed = True
                        self._closed_lots.add(lot.lot_id)
                        if lot.lot_id in self._lots_by_id:
                            del self._lots_by_id[lot.lot_id]
                    else:
                        # Put back modified lot
                        heapq.heappush(heap, lot)
                        temp_popped.pop()  # Remove from temp since we put it back
        finally:
            # Restore any unpopped lots back to heap
            for lot in temp_popped:
                if not lot.is_closed and lot.quantity > 0:
                    if lot not in heap:
                        heapq.heappush(heap, lot)
    
    def calculate_tax_impact(
        self,
        instrument_id: int,
        quantity_to_close: float,
        exit_price_per_unit: float,
    ) -> dict:
        """
        Calculate the tax impact of closing a position using HIFO.
        
        Returns dict with:
        - realized_gain: Total realized gain/loss
        - lots_used: Number of lots touched
        - avg_cost_basis: Average cost basis of closed lots
        """
        total_proceeds = 0.0
        total_cost = 0.0
        lots_used = 0
        
        for lot_id, close_qty in self.select_lots_hifo(instrument_id, quantity_to_close):
            lot = self._lots_by_id.get(lot_id)
            if lot is None:
                continue
            
            cost_per_unit = -lot.cost_basis_per_unit
            lot_cost = close_qty * cost_per_unit
            lot_proceeds = close_qty * exit_price_per_unit
            
            total_cost += lot_cost
            total_proceeds += lot_proceeds
            lots_used += 1
        
        realized_gain = total_proceeds - total_cost
        avg_cost = total_cost / quantity_to_close if quantity_to_close > 0 else 0
        
        return {
            "realized_gain": realized_gain,
            "lots_used": lots_used,
            "avg_cost_basis": avg_cost,
            "total_proceeds": total_proceeds,
            "total_cost": total_cost,
        }
    
    def get_optimal_lot_for_single_unit(
        self,
        instrument_id: int,
    ) -> Optional[int]:
        """Get the optimal lot ID for closing a single unit (highest cost basis)."""
        heap = self._lot_heaps[instrument_id]
        
        # Find first non-closed lot
        for lot in sorted(heap, key=lambda x: x.cost_basis_per_unit):
            if not lot.is_closed and lot.quantity > 0:
                return lot.lot_id
        
        return None
    
    def get_memory_footprint(self) -> int:
        """Return current memory footprint in bytes."""
        return self._memory_footprint_bytes
    
    def get_open_lot_count(self, instrument_id: Optional[int] = None) -> int:
        """Get count of open lots, optionally filtered by instrument."""
        if instrument_id is not None:
            return sum(
                1 for lot in self._lot_heaps[instrument_id]
                if not lot.is_closed and lot.quantity > 0
            )
        
        return sum(
            1 for lots in self._lot_heaps.values()
            for lot in lots
            if not lot.is_closed and lot.quantity > 0
        )
    
    def clear(self) -> None:
        """Clear all lots and reset state."""
        self._lot_heaps.clear()
        self._lots_by_id.clear()
        self._closed_lots.clear()
        self._memory_footprint_bytes = 0


# Example usage and testing
if __name__ == "__main__":
    optimizer = HifoOptimizer()
    
    # Add test lots with different cost bases
    optimizer.add_lot(1, 1, 100, 5000.0, 1000)  # $50/unit
    optimizer.add_lot(2, 1, 100, 6000.0, 2000)  # $60/unit
    optimizer.add_lot(3, 1, 100, 5500.0, 3000)  # $55/unit
    
    # Test HIFO selection - should pick $60 lot first
    result = optimizer.calculate_tax_impact(1, 50, 70.0)
    
    print(f"HIFO Tax Impact Analysis:")
    print(f"  Realized Gain: ${result['realized_gain']:.2f}")
    print(f"  Lots Used: {result['lots_used']}")
    print(f"  Avg Cost Basis: ${result['avg_cost_basis']:.2f}")
    print(f"  Memory Footprint: {optimizer.get_memory_footprint()} bytes")
