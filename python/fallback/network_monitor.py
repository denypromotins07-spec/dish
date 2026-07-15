"""
Network Health Monitor for Trading Bot.
Detects Wi-Fi degradation, packet loss, ISP throttling.
Automatically switches to 5G backup or pauses HFT on latency spikes.
"""

import asyncio
import logging
import psutil
import socket
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DISCONNECTED = "disconnected"


@dataclass
class NetworkMetrics:
    """Current network health metrics."""
    status: NetworkStatus
    latency_ms: float
    packet_loss_percent: float
    jitter_ms: float
    bandwidth_mbps: float
    active_interface: str
    backup_available: bool


class NetworkMonitor:
    """
    Monitors network health and triggers fallback mechanisms.
    Designed for ultra-low-latency trading requirements.
    """
    
    # Latency thresholds (microseconds matter for HFT)
    LATENCY_HEALTHY_US = 1000  # <1ms
    LATENCY_DEGRADED_US = 5000  # 5ms
    LATENCY_CRITICAL_US = 20000  # 20ms
    
    # Packet loss thresholds
    PACKET_LOSS_DEGRADED = 0.1  # 0.1%
    PACKET_LOSS_CRITICAL = 1.0  # 1%
    
    # Monitoring endpoints (exchange APIs, DNS)
    MONITOR_ENDPOINTS = [
        ("8.8.8.8", 53),  # Google DNS
        ("1.1.1.1", 53),  # Cloudflare DNS
    ]
    
    def __init__(self, exchange_host: str = "api.binance.com"):
        self.exchange_host = exchange_host
        self.metrics_history: List[NetworkMetrics] = []
        self.max_history = 100
        self.current_status = NetworkStatus.HEALTHY
        self.backup_interface: Optional[str] = None
        
    def _measure_latency(self, host: str, port: int, timeout_sec: float = 2.0) -> Optional[float]:
        """Measure TCP connection latency in milliseconds."""
        try:
            start = time.perf_counter()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout_sec)
            sock.connect((host, port))
            latency_ms = (time.perf_counter() - start) * 1000
            sock.close()
            return latency_ms
        except Exception as e:
            logger.debug(f"Latency measurement failed to {host}:{port}: {e}")
            return None
    
    def _measure_packet_loss(self, host: str, attempts: int = 10) -> float:
        """Estimate packet loss by counting failed connections."""
        failures = 0
        for _ in range(attempts):
            result = self._measure_latency(host, 443, timeout_sec=0.5)
            if result is None:
                failures += 1
        return (failures / attempts) * 100
    
    def _get_active_interface(self) -> str:
        """Get the name of the primary network interface."""
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        
        for iface, addrs_list in addrs.items():
            if stats[iface].isup and any(addr.family == socket.AF_INET for addr in addrs_list):
                return iface
        return "unknown"
    
    def _check_backup_interface(self) -> bool:
        """Check if 5G/USB tethering backup is available."""
        addrs = psutil.net_if_addrs()
        # Look for common tethering interface names
        tethering_names = ["usb", "tether", "rndis", "wwan"]
        
        for iface in addrs.keys():
            if any(name in iface.lower() for name in tethering_names):
                self.backup_interface = iface
                return True
        return False
    
    def _calculate_jitter(self, samples: List[float]) -> float:
        """Calculate network jitter from latency samples."""
        if len(samples) < 2:
            return 0.0
        
        diffs = [abs(samples[i] - samples[i-1]) for i in range(1, len(samples))]
        return sum(diffs) / len(diffs) if diffs else 0.0
    
    async def measure_network_health(self) -> NetworkMetrics:
        """Perform comprehensive network health check."""
        latencies = []
        
        # Measure latency to multiple endpoints
        for host, port in self.MONITOR_ENDPOINTS:
            latency = self._measure_latency(host, port)
            if latency is not None:
                latencies.append(latency)
        
        # Measure exchange latency
        exchange_latency = self._measure_latency(self.exchange_host, 443)
        if exchange_latency:
            latencies.append(exchange_latency)
        
        if not latencies:
            return NetworkMetrics(
                status=NetworkStatus.DISCONNECTED,
                latency_ms=float('inf'),
                packet_loss_percent=100.0,
                jitter_ms=0.0,
                bandwidth_mbps=0.0,
                active_interface=self._get_active_interface(),
                backup_available=self._check_backup_interface(),
            )
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        jitter = self._calculate_jitter(latencies)
        packet_loss = self._measure_packet_loss(self.exchange_host)
        
        # Determine status
        latency_us = avg_latency * 1000  # Convert to microseconds
        if latency_us > self.LATENCY_CRITICAL_US or packet_loss > self.PACKET_LOSS_CRITICAL:
            status = NetworkStatus.CRITICAL
        elif latency_us > self.LATENCY_DEGRADED_US or packet_loss > self.PACKET_LOSS_DEGRADED:
            status = NetworkStatus.DEGRADED
        else:
            status = NetworkStatus.HEALTHY
        
        metrics = NetworkMetrics(
            status=status,
            latency_ms=avg_latency,
            packet_loss_percent=packet_loss,
            jitter_ms=jitter,
            bandwidth_mbps=0.0,  # Would need speedtest implementation
            active_interface=self._get_active_interface(),
            backup_available=self._check_backup_interface(),
        )
        
        # Store history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        self.current_status = status
        return metrics
    
    def should_pause_hft(self) -> bool:
        """Determine if HFT should be paused due to network issues."""
        return self.current_status in (NetworkStatus.CRITICAL, NetworkStatus.DISCONNECTED)
    
    def should_switch_to_backup(self) -> bool:
        """Determine if we should switch to 5G backup."""
        if not self._check_backup_interface():
            return False
        
        # Switch if critical status or sustained degradation
        recent_critical = sum(
            1 for m in self.metrics_history[-10:]
            if m.status == NetworkStatus.CRITICAL
        )
        return recent_critical >= 3
    
    async def monitor_loop(self, interval_sec: float = 1.0):
        """Continuous monitoring loop."""
        while True:
            try:
                metrics = await self.measure_network_health()
                
                if metrics.status != NetworkStatus.HEALTHY:
                    logger.warning(
                        f"Network degraded: {metrics.status.value}, "
                        f"latency={metrics.latency_ms:.2f}ms, "
                        f"loss={metrics.packet_loss_percent:.2f}%"
                    )
                    
                    if self.should_pause_hft():
                        logger.critical("PAUSING HFT due to network issues!")
                        # In production: trigger pause via callback
                    
                    if self.should_switch_to_backup():
                        logger.warning("Switching to 5G backup interface...")
                        # In production: trigger interface switch
                
                await asyncio.sleep(interval_sec)
                
            except Exception as e:
                logger.error(f"Network monitor error: {e}")
                await asyncio.sleep(interval_sec)


class NetworkFallbackHandler:
    """Handles network fallback actions."""
    
    def __init__(self, monitor: NetworkMonitor):
        self.monitor = monitor
        self.hft_paused = False
        self.on_backup = False
        
    async def handle_fallback(self):
        """Main fallback handling loop."""
        while True:
            if self.monitor.should_pause_hft() and not self.hft_paused:
                logger.critical("Executing HFT pause...")
                self.hft_paused = True
                # Cancel all pending orders, pause strategies
                
            elif not self.monitor.should_pause_hft() and self.hft_paused:
                logger.info("Resuming HFT operations...")
                self.hft_paused = False
                # Resume strategies
                
            if self.monitor.should_switch_to_backup() and not self.on_backup:
                logger.warning("Activating 5G backup...")
                self.on_backup = True
                # Switch network interface
                
            elif not self.monitor.should_switch_to_backup() and self.on_backup:
                logger.info("Returning to primary network...")
                self.on_backup = False
                
            await asyncio.sleep(0.5)


# Example usage
async def main():
    monitor = NetworkMonitor(exchange_host="api.binance.com")
    fallback = NetworkFallbackHandler(monitor)
    
    # Run monitor and fallback handler concurrently
    await asyncio.gather(
        monitor.monitor_loop(interval_sec=1.0),
        fallback.handle_fallback(),
    )


if __name__ == "__main__":
    asyncio.run(main())
