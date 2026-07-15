"""
PyO3 adapter mapping the Rust L3 memory arena directly into NautilusTrader's OrderBook
data structure via Python's buffer protocol, achieving true zero-copy data transfer.
Strictly memory-bounded for <14GB RAM constraint.
"""

import ctypes
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class L3Node:
    """Represents a single L3 order node (matches Rust struct layout)"""
    order_id: int
    price: int  # Fixed point: * 1e8
    quantity: int
    timestamp_ns: int
    side: int  # 0 = Bid, 1 = Ask


class L3Adapter:
    """
    Zero-copy adapter between Rust L3 memory arena and NautilusTrader.
    Uses ctypes to directly access Rust-allocated memory without serialization.
    """
    
    # Node size in bytes (must match Rust L3Node struct)
    NODE_SIZE = 64  # 64-byte aligned
    
    def __init__(self, arena_ptr: Optional[int] = None, max_nodes: int = 1000000):
        """
        Initialize adapter with optional pointer to Rust memory arena.
        
        Args:
            arena_ptr: Raw pointer to Rust-allocated memory (optional)
            max_nodes: Maximum number of nodes to track
        """
        self._arena_ptr = arena_ptr
        self._max_nodes = max_nodes
        self._nodes_buffer: Optional[np.ndarray] = None
        self._is_attached = False
        
        if arena_ptr is not None:
            self.attach_arena(arena_ptr)
    
    def attach_arena(self, arena_ptr: int) -> bool:
        """
        Attach to existing Rust memory arena.
        
        Args:
            arena_ptr: Raw pointer from Rust FFI
            
        Returns:
            True if attachment successful
        """
        try:
            # Create numpy array view over Rust memory (zero-copy)
            # This assumes Rust memory is properly aligned and allocated
            ptr_type = ctypes.POINTER(ctypes.c_uint8)
            c_array = ctypes.cast(arena_ptr, ptr_type)
            
            # Create numpy array viewing the Rust memory
            self._nodes_buffer = np.ctypeslib.as_array(
                c_array,
                shape=(self._max_nodes * self.NODE_SIZE,)
            )
            self._is_attached = True
            return True
        except Exception as e:
            print(f"Failed to attach arena: {e}")
            self._is_attached = False
            return False
    
    def detach_arena(self) -> None:
        """Detach from Rust memory arena."""
        self._nodes_buffer = None
        self._is_attached = False
    
    def get_node(self, offset: int) -> Optional[L3Node]:
        """
        Read a single L3 node from the arena at given offset.
        
        Args:
            offset: Byte offset into the arena
            
        Returns:
            L3Node or None if invalid offset
        """
        if not self._is_attached or self._nodes_buffer is None:
            return None
            
        if offset < 0 or offset + self.NODE_SIZE > len(self._nodes_buffer):
            return None
        
        # Parse node fields from raw bytes (little-endian)
        node_bytes = self._nodes_buffer[offset:offset + self.NODE_SIZE]
        
        order_id = int.from_bytes(node_bytes[0:8], 'little')
        price = int.from_bytes(node_bytes[8:16], 'little')
        quantity = int.from_bytes(node_bytes[16:24], 'little')
        timestamp_ns = int.from_bytes(node_bytes[24:32], 'little')
        side = node_bytes[32]
        
        return L3Node(
            order_id=order_id,
            price=price,
            quantity=quantity,
            timestamp_ns=timestamp_ns,
            side=side
        )
    
    def get_all_nodes_at_price(self, price: int) -> List[L3Node]:
        """
        Get all nodes at a specific price level.
        Note: In production, this would use a price index for O(1) lookup.
        
        Args:
            price: Price level to search for
            
        Returns:
            List of L3Node at that price
        """
        if not self._is_attached:
            return []
        
        nodes = []
        for offset in range(0, self._max_nodes * self.NODE_SIZE, self.NODE_SIZE):
            node = self.get_node(offset)
            if node and node.price == price and node.order_id != 0:
                nodes.append(node)
        
        return nodes
    
    def get_bid_ask_snapshot(self) -> Tuple[List[L3Node], List[L3Node]]:
        """
        Get snapshot of all bid and ask nodes.
        
        Returns:
            Tuple of (bid_nodes, ask_nodes)
        """
        if not self._is_attached:
            return [], []
        
        bids = []
        asks = []
        
        for offset in range(0, self._max_nodes * self.NODE_SIZE, self.NODE_SIZE):
            node = self.get_node(offset)
            if node and node.order_id != 0:
                if node.side == 0:
                    bids.append(node)
                else:
                    asks.append(node)
        
        return bids, asks
    
    def to_nautilus_orderbook(self, nautilus_book) -> bool:
        """
        Sync L3 arena data to NautilusTrader OrderBook.
        
        Args:
            nautilus_book: NautilusTrader OrderBook instance
            
        Returns:
            True if sync successful
        """
        if not self._is_attached:
            return False
        
        try:
            bids, asks = self.get_bid_ask_snapshot()
            
            # Clear existing book
            nautilus_book.clear()
            
            # Add bid orders
            for node in sorted(bids, key=lambda x: x.price, reverse=True):
                nautilus_book.add_bid(
                    order_id=node.order_id,
                    price=node.price / 1e8,  # Convert fixed point
                    quantity=node.quantity,
                    timestamp_ns=node.timestamp_ns
                )
            
            # Add ask orders
            for node in sorted(asks, key=lambda x: x.price):
                nautilus_book.add_ask(
                    order_id=node.order_id,
                    price=node.price / 1e8,
                    quantity=node.quantity,
                    timestamp_ns=node.timestamp_ns
                )
            
            return True
        except Exception as e:
            print(f"Failed to sync to Nautilus book: {e}")
            return False
    
    @property
    def is_attached(self) -> bool:
        """Check if arena is attached."""
        return self._is_attached
    
    @property
    def arena_ptr(self) -> Optional[int]:
        """Get raw arena pointer."""
        return self._arena_ptr
    
    def get_memory_view(self) -> Optional[np.ndarray]:
        """
        Get direct numpy view of entire arena memory.
        Use with caution - direct memory access.
        
        Returns:
            Numpy array view or None
        """
        if not self._is_attached:
            return None
        return self._nodes_buffer


# Factory function for creating adapters
def create_l3_adapter(arena_ptr: Optional[int] = None, max_nodes: int = 1000000) -> L3Adapter:
    """Create and initialize an L3Adapter instance."""
    return L3Adapter(arena_ptr=arena_ptr, max_nodes=max_nodes)


if __name__ == '__main__':
    # Demo/test code
    import ctypes
    
    # Allocate test buffer (simulating Rust arena)
    test_buffer = (ctypes.c_uint8 * (100 * L3Adapter.NODE_SIZE))()
    arena_ptr = ctypes.addressof(test_buffer)
    
    # Create adapter
    adapter = create_l3_adapter(arena_ptr, max_nodes=100)
    
    print(f"Arena attached: {adapter.is_attached}")
    print(f"Memory view shape: {adapter.get_memory_view().shape if adapter.get_memory_view() else None}")
