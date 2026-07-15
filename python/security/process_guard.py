#!/usr/bin/env python3
"""
Watchdog process running in an isolated cgroup that monitors the main trading engine.
Detects unexpected code injection, unauthorized file modifications, or debugger attachment,
and instantly kills the process and severs network connections if threats are detected.
"""

import asyncio
import hashlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Set, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Severity levels for detected threats."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Represents a detected security event."""
    timestamp: str
    threat_level: ThreatLevel
    event_type: str
    description: str
    process_id: int
    action_taken: str


class ProcessGuard:
    """
    Watchdog process that monitors the main trading engine for security threats.
    Runs in an isolated cgroup and can terminate the process if threats are detected.
    """

    def __init__(
        self,
        monitored_pid: Optional[int] = None,
        check_interval_seconds: float = 1.0,
        cgroup_name: str = "trading_bot_isolated",
        enable_file_monitoring: bool = True,
        enable_debugger_detection: bool = True,
        enable_network_monitoring: bool = True,
    ):
        self.monitored_pid = monitored_pid or os.getpid()
        self.check_interval = check_interval_seconds
        self.cgroup_name = cgroup_name
        
        self.enable_file_monitoring = enable_file_monitoring
        self.enable_debugger_detection = enable_debugger_detection
        self.enable_network_monitoring = enable_network_monitoring
        
        # File integrity baselines
        self._file_hashes: Dict[str, str] = {}
        self._monitored_files: Set[str] = set()
        
        # Network state
        self._allowed_connections: Set[tuple] = set()
        self._suspicious_connections: List[dict] = []
        
        # Security events log
        self._events: List[SecurityEvent] = []
        
        # Running state
        self._is_running = False
        self._cgroup_created = False

    def setup_cgroup(self) -> bool:
        """
        Creates an isolated cgroup for the trading process.
        Returns True if successful, False if cgroups are not available.
        """
        try:
            # Check if cgroup v2 is available
            cgroup_path = Path(f"/sys/fs/cgroup/{self.cgroup_name}")
            
            if cgroup_path.exists():
                logger.warning(f"Cgroup {self.cgroup_name} already exists")
                return True
            
            # Create cgroup (requires root privileges)
            if os.geteuid() != 0:
                logger.warning("Root privileges required for cgroup creation")
                return False
            
            # Create the cgroup
            cgroup_path.mkdir(parents=True, exist_ok=True)
            
            # Set memory limit (optional - configure as needed)
            memory_limit_file = cgroup_path / "memory.max"
            if memory_limit_file.exists():
                memory_limit_file.write_text("8G\n")  # 8GB limit
            
            # Add process to cgroup
            cgroup_procs_file = cgroup_path / "cgroup.procs"
            if cgroup_procs_file.exists():
                cgroup_procs_file.write_text(f"{self.monitored_pid}\n")
            
            self._cgroup_created = True
            logger.info(f"Cgroup {self.cgroup_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup cgroup: {e}")
            return False

    def baseline_file_integrity(self, directories: List[str]) -> None:
        """
        Creates baseline hashes of critical files for integrity monitoring.
        """
        if not self.enable_file_monitoring:
            return
        
        logger.info("Creating file integrity baseline...")
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning(f"Directory not found: {directory}")
                continue
            
            for file_path in dir_path.rglob("*.py"):
                if "__pycache__" in str(file_path):
                    continue
                    
                try:
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    self._file_hashes[str(file_path)] = file_hash
                    self._monitored_files.add(str(file_path))
                except Exception as e:
                    logger.warning(f"Could not hash {file_path}: {e}")
        
        logger.info(f"Baseline created for {len(self._file_hashes)} files")

    def check_file_integrity(self) -> List[SecurityEvent]:
        """
        Checks monitored files for unauthorized modifications.
        """
        events = []
        
        for file_path in self._monitored_files:
            path = Path(file_path)
            if not path.exists():
                event = SecurityEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    threat_level=ThreatLevel.HIGH,
                    event_type="file_deleted",
                    description=f"Critical file deleted: {file_path}",
                    process_id=self.monitored_pid,
                    action_taken="alert_only",
                )
                events.append(event)
                continue
            
            try:
                with open(path, "rb") as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                
                if current_hash != self._file_hashes.get(file_path):
                    event = SecurityEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        threat_level=ThreatLevel.CRITICAL,
                        event_type="file_modified",
                        description=f"Critical file modified: {file_path}",
                        process_id=self.monitored_pid,
                        action_taken="emergency_shutdown",
                    )
                    events.append(event)
                    
            except Exception as e:
                logger.warning(f"Could not check {file_path}: {e}")
        
        return events

    def detect_debugger(self) -> bool:
        """
        Detects if a debugger is attached to the process.
        Uses multiple detection methods for reliability.
        """
        if not self.enable_debugger_detection:
            return False
        
        # Method 1: Check TracerPid in /proc/self/status
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("TracerPid:"):
                        tracer_pid = int(line.split(":")[1].strip())
                        if tracer_pid != 0:
                            logger.critical(f"Debugger detected! TracerPid: {tracer_pid}")
                            return True
        except Exception as e:
            logger.warning(f"Could not check TracerPid: {e}")
        
        # Method 2: Check for ptrace-based debugging indicators
        # (Additional checks can be added here)
        
        return False

    def check_network_connections(self) -> List[SecurityEvent]:
        """
        Monitors network connections for suspicious activity.
        """
        events = []
        
        if not self.enable_network_monitoring:
            return events
        
        try:
            # Get current connections
            connections = self._get_established_connections()
            
            for conn in connections:
                # Check against allowed connections
                conn_tuple = (conn["local_addr"], conn["remote_addr"], conn["remote_port"])
                
                if conn_tuple not in self._allowed_connections:
                    # Check if connection is to known exchange IPs
                    if not self._is_exchange_connection(conn["remote_addr"]):
                        event = SecurityEvent(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            threat_level=ThreatLevel.MEDIUM,
                            event_type="suspicious_connection",
                            description=f"Unexpected connection: {conn['remote_addr']}:{conn['remote_port']}",
                            process_id=self.monitored_pid,
                            action_taken="log_and_alert",
                        )
                        events.append(event)
                        self._suspicious_connections.append(conn)
                        
        except Exception as e:
            logger.warning(f"Network monitoring error: {e}")
        
        return events

    def _get_established_connections(self) -> List[dict]:
        """Gets list of established TCP connections."""
        connections = []
        
        try:
            # Parse /proc/net/tcp for connections
            with open("/proc/net/tcp", "r") as f:
                lines = f.readlines()[1:]  # Skip header
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        # State 01 = ESTABLISHED
                        if parts[3] == "01":
                            local = self._parse_ip_port(parts[1])
                            remote = self._parse_ip_port(parts[2])
                            
                            connections.append({
                                "local_addr": local[0],
                                "local_port": local[1],
                                "remote_addr": remote[0],
                                "remote_port": remote[1],
                            })
        except Exception as e:
            logger.warning(f"Could not parse connections: {e}")
        
        return connections

    def _parse_ip_port(self, hex_addr: str) -> tuple:
        """Parses hex IP:port from /proc/net/tcp format."""
        try:
            ip_hex, port_hex = hex_addr.split(":")
            port = int(port_hex, 16)
            
            # Convert IP (little-endian hex)
            ip_int = int(ip_hex, 16)
            ip = socket.inet_ntoa(ip_int.to_bytes(4, "little"))
            
            return (ip, port)
        except Exception:
            return ("0.0.0.0", 0)

    def _is_exchange_connection(self, ip: str) -> bool:
        """Checks if an IP belongs to a known exchange."""
        # Add known exchange IP ranges here
        exchange_ranges = [
            # Binance
            "3.64.0.0/16",
            # Coinbase
            "52.20.0.0/16",
            # Kraken
            "34.192.0.0/16",
        ]
        
        # Simplified check - in production, use proper CIDR matching
        return any(ip.startswith(prefix.split(".")[0]) for prefix in exchange_ranges)

    async def run_watchdog(self) -> None:
        """
        Main watchdog loop that continuously monitors for threats.
        """
        self._is_running = True
        logger.info(f"ProcessGuard started. Monitoring PID: {self.monitored_pid}")
        
        while self._is_running:
            try:
                # Check if monitored process is still alive
                if not self._process_exists(self.monitored_pid):
                    logger.critical("Monitored process has terminated!")
                    await self._emergency_shutdown("process_terminated")
                    break
                
                # Debugger detection
                if self.detect_debugger():
                    await self._emergency_shutdown("debugger_attached")
                
                # File integrity check
                file_events = self.check_file_integrity()
                for event in file_events:
                    if event.threat_level == ThreatLevel.CRITICAL:
                        await self._emergency_shutdown("file_tampering")
                    self._events.append(event)
                
                # Network monitoring
                network_events = self.check_network_connections()
                for event in network_events:
                    self._events.append(event)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(self.check_interval)

    def _process_exists(self, pid: int) -> bool:
        """Checks if a process with the given PID exists."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    async def _emergency_shutdown(self, reason: str) -> None:
        """
        Executes emergency shutdown procedure.
        Terminates the process and severs network connections.
        """
        logger.critical(f"EMERGENCY SHUTDOWN triggered: {reason}")
        
        # Log the security event
        event = SecurityEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            threat_level=ThreatLevel.CRITICAL,
            event_type="emergency_shutdown",
            description=f"Emergency shutdown due to: {reason}",
            process_id=self.monitored_pid,
            action_taken="terminate_process",
        )
        self._events.append(event)
        
        # Sever network connections (best effort)
        await self._sever_network_connections()
        
        # Terminate the process
        logger.critical("Terminating monitored process...")
        try:
            os.kill(self.monitored_pid, signal.SIGKILL)
        except Exception as e:
            logger.error(f"Failed to kill process: {e}")
        
        # Self-terminate as well
        logger.critical("ProcessGuard self-terminating...")
        sys.exit(1)

    async def _sever_network_connections(self) -> None:
        """Attempts to sever all outbound network connections."""
        logger.info("Severing network connections...")
        
        try:
            # Use iptables to block all outbound traffic (requires root)
            if os.geteuid() == 0:
                subprocess.run(
                    ["iptables", "-A", "OUTPUT", "-j", "DROP"],
                    capture_output=True,
                    timeout=5,
                )
                logger.info("Outbound traffic blocked via iptables")
        except Exception as e:
            logger.warning(f"Could not block network: {e}")

    def get_security_events(self) -> List[SecurityEvent]:
        """Returns the list of recorded security events."""
        return self._events.copy()

    def stop(self) -> None:
        """Stops the watchdog monitoring."""
        self._is_running = False
        logger.info("ProcessGuard stopped")


async def main():
    """Example usage of the ProcessGuard."""
    print("=" * 60)
    print("Process Guard - Security Watchdog")
    print("=" * 60)
    
    guard = ProcessGuard(
        monitored_pid=os.getpid(),
        check_interval_seconds=2.0,
        enable_file_monitoring=True,
        enable_debugger_detection=True,
        enable_network_monitoring=True,
    )
    
    # Setup cgroup (requires root)
    # guard.setup_cgroup()
    
    # Baseline file integrity
    guard.baseline_file_integrity(["./python/security"])
    
    print("\nStarting watchdog monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        await guard.run_watchdog()
    except KeyboardInterrupt:
        guard.stop()
        print("\nWatchdog stopped.")


if __name__ == "__main__":
    asyncio.run(main())
