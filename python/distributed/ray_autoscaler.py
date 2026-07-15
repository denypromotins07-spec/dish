"""
Ray Autoscaler with Strict Memory Quotas for 14GB RAM Ceiling.
Dynamically scales workers up/down based on real-time memory pressure.
Aggressively kills idle actors to protect the live trading engine.
"""

import ray
from ray.autoscaler.sdk import request_resources
from typing import Dict, List, Optional
import psutil
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Strict constraints for 16GB total RAM system
MAX_SYSTEM_RAM_GB = 14.0
RESERVED_RAM_FOR_ENGINE_GB = 2.0  # Reserve for live Rust engine
AVAILABLE_RAM_FOR_RAY_GB = MAX_SYSTEM_RAM_GB - RESERVED_RAM_FOR_ENGINE_GB

# Worker configuration
WORKER_RAM_GB = 0.5  # 500MB per worker
MIN_WORKERS = 1
MAX_WORKERS = int(AVAILABLE_RAM_FOR_RAY_GB / WORKER_RAM_GB)  # ~24 workers max


class RayAutoscaler:
    """
    Custom autoscaler that respects strict RAM limits.
    Monitors system memory and adjusts Ray cluster size dynamically.
    """
    
    def __init__(self, head_node_ip: str = "127.0.0.1"):
        self.head_node_ip = head_node_ip
        self.active_workers: Dict[str, dict] = {}
        self.last_scale_time = 0
        self.scale_cooldown_sec = 5.0  # Prevent thrashing
        
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory usage in GB."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent_used": mem.percent
        }
    
    def calculate_safe_worker_count(self) -> int:
        """
        Calculate the maximum number of workers that can run safely
        without exceeding the 14GB RAM ceiling.
        """
        mem_info = self.get_system_memory_info()
        available_for_workers = mem_info["available_gb"] - RESERVED_RAM_FOR_ENGINE_GB
        
        if available_for_workers <= 0:
            return 0
            
        safe_count = int(available_for_workers / WORKER_RAM_GB)
        return min(safe_count, MAX_WORKERS)
    
    def scale_cluster(self, force: bool = False) -> bool:
        """
        Scale the Ray cluster up or down based on current memory pressure.
        Returns True if scaling occurred, False otherwise.
        """
        current_time = time.time()
        if not force and (current_time - self.last_scale_time) < self.scale_cooldown_sec:
            return False
            
        safe_worker_count = self.calculate_safe_worker_count()
        current_worker_count = len(self.active_workers)
        
        if safe_worker_count == current_worker_count:
            return False
            
        logger.info(f"Scaling cluster: {current_worker_count} -> {safe_worker_count} workers")
        
        # Request new resource allocation
        request_resources(
            num_cpus=safe_worker_count * 0.5,  # 0.5 CPU per worker
            memory=int(safe_worker_count * WORKER_RAM_GB * 1024 * 1024 * 1024)  # Convert to bytes
        )
        
        self.last_scale_time = current_time
        return True
    
    def kill_idle_actors(self, idle_threshold_sec: float = 60.0) -> int:
        """
        Aggressively terminate actors that have been idle beyond threshold.
        Returns count of killed actors.
        """
        killed_count = 0
        current_time = time.time()
        
        try:
            # Get all running actors
            actor_list = ray.state.list_actors()
            
            for actor in actor_list:
                if actor["State"] == "ALIVE":
                    # Check last task timestamp (simplified - in production use actor metrics)
                    # This is a placeholder for actual idle detection logic
                    actor_info = ray.state.get_actor(actor["ActorID"])
                    
                    # If actor has no recent activity, mark for termination
                    # In production, integrate with Ray's internal metrics
                    if self._is_actor_idle(actor_info, idle_threshold_sec):
                        ray.kill(ray.get_actor(actor["Name"]))
                        killed_count += 1
                        logger.warning(f"Killed idle actor: {actor['Name']}")
                        
        except Exception as e:
            logger.error(f"Error killing idle actors: {e}")
            
        return killed_count
    
    def _is_actor_idle(self, actor_info: dict, threshold_sec: float) -> bool:
        """Check if an actor has been idle beyond threshold."""
        # Placeholder - implement actual idle detection using Ray metrics
        # In production, check actor's last task completion time
        return False
    
    def monitor_and_scale(self, interval_sec: float = 10.0):
        """
        Continuous monitoring loop that scales cluster based on memory pressure.
        Run this in a background thread.
        """
        while True:
            try:
                mem_info = self.get_system_memory_info()
                
                # Critical: if we're approaching the limit, scale down aggressively
                if mem_info["used_gb"] > MAX_SYSTEM_RAM_GB - 1.0:
                    logger.critical(f"Memory pressure critical: {mem_info['used_gb']:.2f}GB used")
                    self.kill_idle_actors(idle_threshold_sec=10.0)  # Aggressive cleanup
                    
                self.scale_cluster()
                
                time.sleep(interval_sec)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(interval_sec)


def initialize_ray_cluster():
    """Initialize Ray with strict memory bounds."""
    if not ray.is_initialized():
        ray.init(
            address="auto",
            _memory=int(AVAILABLE_RAM_FOR_RAY_GB * 1024 * 1024 * 1024),  # Total memory limit
            object_store_memory=int(AVAILABLE_RAM_FOR_RAY_GB * 1024 * 1024 * 1024 * 0.3),  # 30% for object store
            runtime_env={
                "env_vars": {
                    "RAY_memory_monitor_refresh_ms": "1000",  # Fast memory monitoring
                    "RAY_object_store_memory_mib": str(int(AVAILABLE_RAM_FOR_RAY_GB * 1024 * 0.3))
                }
            }
        )
        logger.info(f"Ray initialized with {AVAILABLE_RAM_FOR_RAY_GB}GB memory limit")


if __name__ == "__main__":
    initialize_ray_cluster()
    scaler = RayAutoscaler()
    
    # Start monitoring in background (in production, use threading)
    logger.info("Starting Ray autoscaler monitor...")
    scaler.monitor_and_scale(interval_sec=5.0)
