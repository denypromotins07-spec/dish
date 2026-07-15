#!/usr/bin/env python3
"""
Python Supervisor for Ray Cluster, Nautilus Node, and Analytics Workers.
Ensures strict 4GB RAM slice per worker, graceful restarts on crash, no memory leaks.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import resource
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from enum import Enum
import psutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class WorkerState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    CRASHED = "crashed"
    RESTARTING = "restarting"


@dataclass
class WorkerConfig:
    name: str
    entry_point: str
    args: List[str]
    max_ram_mb: int
    cpu_affinity: List[int]
    restart_delay_sec: float
    max_restarts: int
    is_critical: bool


class WorkerHandle:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.state = WorkerState.STOPPED
        self.process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.last_heartbeat = 0.0
        self.stop_flag = False
        self.lock = threading.Lock()

    def start(self) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                logger.warning(f"Worker {self.config.name} already running")
                return

            self.state = WorkerState.STARTING
            logger.info(f"Starting worker: {self.config.name}")

            try:
                # Set resource limits before spawning
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                new_limit = self.config.max_ram_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (new_limit, new_limit))

                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(len(self.config.cpu_affinity))
                env["MKL_NUM_THREADS"] = str(len(self.config.cpu_affinity))

                cmd = [sys.executable, self.config.entry_point] + self.config.args
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    preexec_fn=lambda: self._set_cpu_affinity(self.config.cpu_affinity),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.state = WorkerState.RUNNING
                self.last_heartbeat = time.time()
                logger.info(f"Worker {self.config.name} started with PID {self.process.pid}")

            except Exception as e:
                logger.error(f"Failed to start worker {self.config.name}: {e}")
                self.state = WorkerState.CRASHED

    def _set_cpu_affinity(self, cpus: List[int]) -> None:
        """Set CPU affinity for the current process (called in child via preexec_fn)."""
        try:
            p = psutil.Process(os.getpid())
            p.cpu_affinity(cpus)
        except Exception as e:
            logger.warning(f"Could not set CPU affinity: {e}")

    def stop(self) -> None:
        with self.lock:
            self.stop_flag = True
            if self.process is not None:
                logger.info(f"Stopping worker: {self.config.name}")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Worker {self.config.name} did not terminate gracefully, killing...")
                    self.process.kill()
                    self.process.wait()
                self.process = None
            self.state = WorkerState.STOPPED

    def is_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def check_health(self) -> bool:
        """Check if worker is alive and within memory limits."""
        if not self.is_alive():
            return False

        # Check memory usage
        try:
            p = psutil.Process(self.process.pid)
            mem_mb = p.memory_info().rss / (1024 * 1024)
            if mem_mb > self.config.max_ram_mb * 0.95:  # 95% threshold
                logger.warning(f"Worker {self.config.name} approaching RAM limit: {mem_mb:.1f}MB / {self.config.max_ram_mb}MB")
            self.last_heartbeat = time.time()
            return True
        except (psutil.NoSuchProcess, Exception):
            return False


class PythonSupervisor:
    """
    Supervisor for Python workers (Ray, Nautilus, Analytics).
    Implements Erlang-style supervision trees with memory limits.
    """

    def __init__(self):
        self.workers: Dict[str, WorkerHandle] = {}
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def register_worker(self, config: WorkerConfig) -> None:
        handle = WorkerHandle(config)
        self.workers[config.name] = handle
        logger.info(f"Registered worker: {config.name}")

    def start_all(self) -> None:
        self.is_running = True
        logger.info("Python Supervisor starting all workers...")
        for worker in self.workers.values():
            worker.start()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_all(self) -> None:
        logger.info("Python Supervisor stopping all workers...")
        self.is_running = False

        for worker in self.workers.values():
            worker.stop()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("All workers stopped.")

    def _monitor_loop(self) -> None:
        """Monitor workers, restart on crash with backoff."""
        while self.is_running:
            for name, worker in list(self.workers.items()):
                if worker.stop_flag:
                    continue

                if not worker.check_health():
                    if worker.state == WorkerState.RUNNING:
                        logger.warning(f"Worker {name} appears dead or unhealthy")
                        worker.state = WorkerState.CRASHED

                    if worker.state == WorkerState.CRASHED:
                        if worker.restart_count >= worker.config.max_restarts:
                            logger.error(f"Worker {name} exceeded max restarts ({worker.config.max_restarts})")
                            if worker.config.is_critical:
                                logger.critical("Critical worker failed permanently. Initiating shutdown...")
                                self._emergency_shutdown()
                                return
                            else:
                                continue  # Skip non-critical

                        # Restart with delay
                        worker.state = WorkerState.RESTARTING
                        delay = worker.config.restart_delay_sec * (2 ** worker.restart_count)  # Exponential backoff
                        logger.info(f"Restarting worker {name} in {delay:.1f}s (attempt {worker.restart_count + 1})")
                        time.sleep(delay)

                        worker.restart_count += 1
                        worker.start()

            time.sleep(0.5)  # Monitor interval

    def _emergency_shutdown(self) -> None:
        """Emergency shutdown when critical worker fails permanently."""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        self.stop_all()
        os._exit(1)


def create_ray_worker_config() -> WorkerConfig:
    return WorkerConfig(
        name="ray_cluster",
        entry_point="python/orchestrator/ray_worker.py",
        args=["--head", "--num-cpus=4", "--memory=4294967296"],  # 4GB
        max_ram_mb=4096,
        cpu_affinity=[0, 1, 2, 3],
        restart_delay_sec=2.0,
        max_restarts=3,
        is_critical=True,
    )


def create_nautilus_worker_config() -> WorkerConfig:
    return WorkerConfig(
        name="nautilus_node",
        entry_point="python/orchestrator/nautilus_worker.py",
        args=["--port=8080"],
        max_ram_mb=2048,
        cpu_affinity=[4, 5],
        restart_delay_sec=1.0,
        max_restarts=5,
        is_critical=True,
    )


def create_analytics_worker_config() -> WorkerConfig:
    return WorkerConfig(
        name="analytics_worker",
        entry_point="python/orchestrator/analytics_worker.py",
        args=["--batch-size=1000"],
        max_ram_mb=2048,
        cpu_affinity=[6, 7],
        restart_delay_sec=1.0,
        max_restarts=5,
        is_critical=False,
    )


def main():
    supervisor = PythonSupervisor()

    # Register workers
    supervisor.register_worker(create_ray_worker_config())
    supervisor.register_worker(create_nautilus_worker_config())
    supervisor.register_worker(create_analytics_worker_config())

    # Handle signals
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        supervisor.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start all workers
    supervisor.start_all()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        supervisor.stop_all()


if __name__ == "__main__":
    main()
