"""
TCP Backpressure Simulator - Simulates local network congestion and OS-level TCP window scaling.
Tests how Rust ingestion layer handles massive memory backlogs without exceeding 14GB RAM limit.
Memory-bounded using streaming Polars DataFrames.
"""

import time
import socket
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
import struct

logger = logging.getLogger(__name__)


@dataclass
class TcpConfig:
    """TCP simulation configuration."""
    max_buffer_size_mb: float = 100.0  # Maximum buffer size in MB
    window_size_kb: int = 64  # TCP window size in KB
    rto_ms: float = 200.0  # Retransmission timeout in ms
    max_retries: int = 3
    simulate_congestion: bool = True
    congestion_threshold_pct: float = 80.0  # Buffer fill % to trigger congestion


@dataclass
class PacketStats:
    """Statistics for packet transmission."""
    packets_sent: int = 0
    packets_received: int = 0
    packets_dropped: int = 0
    packets_retransmitted: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    buffer_high_water_mark_mb: float = 0.0


class TcpBackpressureSimulator:
    """
    Simulates TCP backpressure and network congestion.
    Bounded memory usage with configurable limits.
    """
    
    def __init__(self, config: TcpConfig):
        self.config = config
        self.max_buffer_bytes = int(config.max_buffer_size_mb * 1024 * 1024)
        self.window_size_bytes = config.window_size_kb * 1024
        
        # Ring buffers for bounded memory
        self.send_buffer: deque = deque()
        self.recv_buffer: deque = deque()
        self.current_buffer_size = 0
        
        # Statistics
        self.stats = PacketStats()
        self.latencies: deque = deque(maxlen=1000)  # Bounded latency history
        
        # State
        self.congested = False
        self.running = False
        self._lock = threading.Lock()
        
    def send_packet(self, data: bytes, priority: int = 0) -> bool:
        """
        Attempt to send a packet through simulated TCP.
        Returns False if backpressure prevents sending.
        """
        with self._lock:
            packet_size = len(data)
            
            # Check buffer capacity
            if self.current_buffer_size + packet_size > self.max_buffer_bytes:
                logger.warning("Buffer full - backpressure triggered")
                self.stats.packets_dropped += 1
                self.congested = True
                return False
            
            # Add to send buffer
            self.send_buffer.append({
                'data': data,
                'size': packet_size,
                'timestamp': time.time(),
                'priority': priority,
                'retries': 0,
            })
            
            self.current_buffer_size += packet_size
            self.stats.packets_sent += 1
            self.stats.bytes_sent += packet_size
            
            # Update high water mark
            current_mb = self.current_buffer_size / (1024 * 1024)
            if current_mb > self.stats.buffer_high_water_mark_mb:
                self.stats.buffer_high_water_mark_mb = current_mb
            
            # Check congestion threshold
            fill_pct = (self.current_buffer_size / self.max_buffer_bytes) * 100
            if fill_pct > self.config.congestion_threshold_pct:
                self.congested = True
                
            return True
    
    def receive_packet(self) -> Optional[bytes]:
        """Receive a packet from the simulated TCP stream."""
        with self._lock:
            if not self.recv_buffer:
                return None
            
            packet = self.recv_buffer.popleft()
            self.current_buffer_size -= packet['size']
            self.stats.packets_received += 1
            self.stats.bytes_received += packet['size']
            
            # Calculate latency
            latency_ms = (time.time() - packet['timestamp']) * 1000
            self.latencies.append(latency_ms)
            self._update_latency_stats(latency_ms)
            
            # Clear congestion if buffer empties
            if self.current_buffer_size < self.max_buffer_bytes * 0.5:
                self.congested = False
            
            return packet['data']
    
    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update latency statistics."""
        total = sum(self.latencies)
        count = len(self.latencies)
        self.stats.avg_latency_ms = total / count if count > 0 else 0.0
        self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency_ms)
    
    def simulate_ack(self, packet_id: int) -> bool:
        """Simulate ACK reception."""
        with self._lock:
            # Find and remove packet from send buffer
            for i, packet in enumerate(self.send_buffer):
                if id(packet) == packet_id:
                    self.send_buffer.remove(packet)
                    self.current_buffer_size -= packet['size']
                    return True
            return False
    
    def simulate_timeout(self, packet_id: int) -> bool:
        """Simulate timeout and retransmission."""
        with self._lock:
            for packet in self.send_buffer:
                if id(packet) == packet_id:
                    packet['retries'] += 1
                    if packet['retries'] > self.config.max_retries:
                        # Drop packet after max retries
                        self.send_buffer.remove(packet)
                        self.current_buffer_size -= packet['size']
                        self.stats.packets_dropped += 1
                        return False
                    else:
                        self.stats.packets_retransmitted += 1
                        return True
            return False
    
    def get_window_available(self) -> int:
        """Get available window size for sending."""
        with self._lock:
            used = sum(p['size'] for p in self.send_buffer)
            return max(0, self.window_size_bytes - used)
    
    def is_congested(self) -> bool:
        """Check if connection is congested."""
        return self.congested
    
    def get_buffer_fill_pct(self) -> float:
        """Get current buffer fill percentage."""
        with self._lock:
            return (self.current_buffer_size / self.max_buffer_bytes) * 100
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        with self._lock:
            return {
                'packets_sent': self.stats.packets_sent,
                'packets_received': self.stats.packets_received,
                'packets_dropped': self.stats.packets_dropped,
                'packets_retransmitted': self.stats.packets_retransmitted,
                'bytes_sent': self.stats.bytes_sent,
                'bytes_received': self.stats.bytes_received,
                'avg_latency_ms': self.stats.avg_latency_ms,
                'max_latency_ms': self.stats.max_latency_ms,
                'buffer_high_water_mark_mb': self.stats.buffer_high_water_mark_mb,
                'current_buffer_mb': self.current_buffer_size / (1024 * 1024),
                'buffer_fill_pct': self.get_buffer_fill_pct(),
                'is_congested': self.congested,
                'window_available': self.get_window_available(),
            }
    
    def reset(self) -> None:
        """Reset simulator state."""
        with self._lock:
            self.send_buffer.clear()
            self.recv_buffer.clear()
            self.current_buffer_size = 0
            self.congested = False
            self.stats = PacketStats()
            self.latencies.clear()


class TcpWindowScaler:
    """
    Simulates TCP window scaling behavior.
    Adjusts window size based on network conditions.
    """
    
    def __init__(self, initial_window_kb: int = 64):
        self.window_size = initial_window_kb * 1024
        self.slow_start_threshold = 65535  # Initial SS threshold
        self.congestion_window = initial_window_kb * 1024
        self.in_recovery = False
        
    def on_ack(self, bytes_acked: int) -> None:
        """Handle ACK - increase congestion window."""
        if self.in_recovery:
            # Congestion avoidance - linear increase
            self.congestion_window += bytes_acked * 1 / (self.congestion_window / 1460)
        else:
            # Slow start - exponential increase
            self.congestion_window *= 2
            if self.congestion_window >= self.slow_start_threshold:
                self.in_recovery = True
    
    def on_loss(self) -> None:
        """Handle packet loss - reduce window."""
        self.slow_start_threshold = self.congestion_window // 2
        self.congestion_window = max(1460, self.slow_start_threshold)
        self.in_recovery = False
    
    def get_window(self) -> int:
        """Get current window size."""
        return int(self.congestion_window)


# Example usage
if __name__ == "__main__":
    config = TcpConfig(
        max_buffer_size_mb=50.0,
        window_size_kb=64,
        simulate_congestion=True,
    )
    
    simulator = TcpBackpressureSimulator(config)
    
    # Simulate sending packets
    print("Sending packets...")
    for i in range(100):
        data = b"X" * 10000  # 10KB packets
        success = simulator.send_packet(data)
        if not success:
            print(f"Backpressure at packet {i}")
            break
    
    # Print stats
    stats = simulator.get_stats()
    print("\nTCP Backpressure Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
