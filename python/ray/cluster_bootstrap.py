"""
Ray Cluster Bootstrap for Low-Memory Distributed Computing.

Initializes a local, lightweight Ray cluster strictly bounded to 2GB RAM
for parallel data processing, ensuring it doesn't starve the live trading
engine which has priority access to the 14GB system RAM limit.
"""

import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import psutil

log = logging.getLogger(__name__)


@dataclass
class RayClusterConfig:
    """Configuration for low-memory Ray cluster."""
    
    # Memory limits (critical for 14GB system constraint)
    max_memory_gb: float = 2.0  # Strict 2GB limit for Ray
    object_store_memory_gb: float = 1.0  # 50% of Ray memory for Plasma
    
    # CPU configuration
    num_cpus: int = 4  # Use only 4 cores (leave others for trading engine)
    num_gpus: int = 0  # No GPU for background tasks (save for ML if needed)
    
    # Head node settings
    head_node_host: str = "127.0.0.1"
    head_node_port: int = 6379
    
    # Worker settings
    num_workers: int = 2  # Minimal workers for low memory
    worker_memory_gb: float = 0.5  # 500MB per worker
    
    # Performance tuning
    enable_profiling: bool = False
    logging_level: str = "WARNING"  # Minimize logging overhead
    
    # Spill-to-disk settings (when memory pressure hits)
    spill_to_disk_enabled: bool = True
    spill_directory: str = "/tmp/ray_spill"
    
    # Health monitoring
    health_check_interval_s: int = 10
    runtime_env_timeout_s: int = 300


class RayClusterBootstrap:
    """
    Bootstrap and manage a low-memory Ray cluster.
    
    Designed to coexist with the live trading engine by strictly
    limiting resource usage and automatically scaling down under
    memory pressure.
    """
    
    def __init__(self, config: Optional[RayClusterConfig] = None):
        self.config = config or RayClusterConfig()
        self.cluster_started = False
        self.head_process: Optional[subprocess.Popen] = None
        self.worker_processes: list = []
        
    def _check_available_memory(self) -> float:
        """Check available system memory in GB."""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        log.info(f"Available system memory: {available_gb:.2f} GB")
        return available_gb
        
    def _check_trading_engine_memory(self) -> float:
        """Estimate memory reserved for trading engine."""
        # Assume trading engine needs ~8-10GB for optimal operation
        trading_reserved_gb = 10.0
        return trading_reserved_gb
        
    def validate_resources(self) -> bool:
        """Validate that we can start Ray without starving the trading engine."""
        available = self._check_available_memory()
        trading_reserved = self._check_trading_engine_memory()
        
        required_for_ray = self.config.max_memory_gb
        required_total = trading_reserved + required_for_ray
        
        if available < required_for_ray:
            log.warning(
                f"Insufficient memory for Ray: need {required_for_ray:.2f}GB, "
                f"have {available:.2f}GB available"
            )
            return False
            
        if (trading_reserved + required_for_ray) > available:
            log.warning(
                f"Starting Ray may impact trading engine: "
                f"trading({trading_reserved:.2f}GB) + ray({required_for_ray:.2f}GB) "
                f"> available({available:.2f}GB)"
            )
            # Still allow but with reduced allocation
            self.config.max_memory_gb = min(
                self.config.max_memory_gb,
                available - trading_reserved - 0.5  # Leave 500MB buffer
            )
            log.info(f"Reduced Ray memory limit to {self.config.max_memory_gb:.2f}GB")
            
        return True
        
    def _build_ray_start_command(self) -> list:
        """Build the command to start Ray head node."""
        cmd = [
            "ray", "start", "--head",
            f"--node-ip-address={self.config.head_node_host}",
            f"--port={self.config.head_node_port}",
            f"--num-cpus={self.config.num_cpus}",
            f"--memory={int(self.config.max_memory_gb * 1024 * 1024 * 1024)}",
            f"--object-store-memory={int(self.config.object_store_memory_gb * 1024 * 1024 * 1024)}",
            "--include-dashboard=false",  # Disable dashboard to save memory
            f"--log-style=pretty",
            f"--logging-level={self.config.logging_level}",
        ]
        
        if self.config.spill_to_disk_enabled:
            os.makedirs(self.config.spill_directory, exist_ok=True)
            cmd.append(f"--temp-dir={self.config.spill_directory}")
            
        if not self.config.enable_profiling:
            cmd.append("--disable-usage-stats")
            
        return cmd
        
    def start_head_node(self) -> bool:
        """Start the Ray head node."""
        if not self.validate_resources():
            log.error("Resource validation failed, cannot start Ray head node")
            return False
            
        log.info("Starting Ray head node...")
        log.info(f"Memory limit: {self.config.max_memory_gb:.2f}GB")
        log.info(f"Object store: {self.config.object_store_memory_gb:.2f}GB")
        log.info(f"CPUs: {self.config.num_cpus}")
        
        cmd = self._build_ray_start_command()
        log.debug(f"Ray command: {' '.join(cmd)}")
        
        try:
            self.head_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Wait for startup
            import time
            time.sleep(5)
            
            # Check if process is still running
            if self.head_process.poll() is not None:
                stdout, stderr = self.head_process.communicate()
                log.error(f"Ray head node failed to start: {stderr}")
                return False
                
            self.cluster_started = True
            log.info("Ray head node started successfully")
            return True
            
        except Exception as e:
            log.error(f"Failed to start Ray head node: {e}")
            return False
            
    def start_workers(self) -> bool:
        """Start Ray worker nodes."""
        if not self.cluster_started:
            log.error("Cannot start workers: head node not running")
            return False
            
        log.info(f"Starting {self.config.num_workers} Ray workers...")
        
        redis_address = f"{self.config.head_node_host}:{self.config.head_node_port}"
        
        for i in range(self.config.num_workers):
            cmd = [
                "ray", "start",
                f"--address={redis_address}",
                f"--num-cpus={self.config.num_cpus // self.config.num_workers}",
                f"--memory={int(self.config.worker_memory_gb * 1024 * 1024 * 1024)}",
                "--block=false",
            ]
            
            try:
                worker_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.worker_processes.append(worker_proc)
                log.info(f"Worker {i+1} started")
                
            except Exception as e:
                log.error(f"Failed to start worker {i+1}: {e}")
                return False
                
        return True
        
    def get_connection_string(self) -> str:
        """Get the Ray connection string."""
        return f"ray://{self.config.head_node_host}:{self.config.head_node_port}"
        
    def stop(self) -> None:
        """Stop all Ray processes gracefully."""
        log.info("Stopping Ray cluster...")
        
        # Stop workers first
        for proc in self.worker_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception as e:
                log.warning(f"Error stopping worker: {e}")
                try:
                    proc.kill()
                except:
                    pass
                    
        self.worker_processes.clear()
        
        # Stop head node
        if self.head_process:
            try:
                self.head_process.terminate()
                self.head_process.wait(timeout=10)
            except Exception as e:
                log.warning(f"Error stopping head node: {e}")
                try:
                    self.head_process.kill()
                except:
                    pass
                    
        self.head_process = None
        self.cluster_started = False
        
        # Final cleanup
        try:
            subprocess.run(["ray", "stop"], timeout=10, capture_output=True)
        except:
            pass
            
        log.info("Ray cluster stopped")


def initialize_ray_cluster(
    max_memory_gb: float = 2.0,
    num_cpus: int = 4,
    num_workers: int = 2,
) -> str:
    """
    Convenience function to initialize a low-memory Ray cluster.
    
    Args:
        max_memory_gb: Maximum memory for Ray (default 2GB)
        num_cpus: Number of CPUs to allocate
        num_workers: Number of worker processes
        
    Returns:
        Ray connection string
    """
    config = RayClusterConfig(
        max_memory_gb=max_memory_gb,
        num_cpus=num_cpus,
        num_workers=num_workers,
        worker_memory_gb=max_memory_gb / num_workers if num_workers > 0 else 0.5,
    )
    
    bootstrap = RayClusterBootstrap(config)
    
    if not bootstrap.start_head_node():
        raise RuntimeError("Failed to start Ray head node")
        
    if not bootstrap.start_workers():
        bootstrap.stop()
        raise RuntimeError("Failed to start Ray workers")
        
    connection_str = bootstrap.get_connection_string()
    log.info(f"Ray cluster ready at: {connection_str}")
    
    return connection_str


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        conn_str = initialize_ray_cluster()
        print(f"Ray cluster initialized: {conn_str}")
        print("Press Ctrl+C to stop...")
        
        import signal
        import sys
        
        def signal_handler(sig, frame):
            print("\nShutting down Ray cluster...")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # Keep running
        import time
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
