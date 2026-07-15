"""
Cloud Offloader for Heavy Parameter Optimizations.
Pushes non-latency-critical work to cloud VPS when local resources are saturated.
Uses secure SSH/API for communication.
"""

import asyncio
import logging
import paramiko
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import psutil
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Configuration for cloud offloading."""
    vps_host: str
    vps_port: int = 22
    username: str
    private_key_path: str
    max_offload_size_mb: float = 100.0
    timeout_sec: float = 300.0


class CloudOffloader:
    """
    Securely offloads heavy computations to cloud VPS.
    Automatically triggers when local CPU/RAM exceeds thresholds.
    """
    
    # Local resource thresholds for triggering offload
    CPU_THRESHOLD_PERCENT = 80.0
    RAM_THRESHOLD_GB = 12.0  # Trigger offload before hitting 14GB ceiling
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.ssh_client: Optional[paramiko.SSHClient] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def connect(self):
        """Establish SSH connection to cloud VPS."""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.ssh_client.connect(
                hostname=self.config.vps_host,
                port=self.config.vps_port,
                username=self.config.username,
                key_filename=self.config.private_key_path,
                timeout=10.0,
            )
            
            logger.info(f"Connected to cloud VPS: {self.config.vps_host}")
            
        except Exception as e:
            logger.error(f"Failed to connect to cloud VPS: {e}")
            raise
    
    def disconnect(self):
        """Close SSH connection."""
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
            logger.info("Disconnected from cloud VPS")
    
    def should_offload(self) -> bool:
        """Check if local resources are saturated and offload is needed."""
        cpu_percent = psutil.cpu_percent(interval=1.0)
        ram_used_gb = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 ** 3)
        
        if cpu_percent > self.CPU_THRESHOLD_PERCENT:
            logger.info(f"CPU saturation detected: {cpu_percent}%")
            return True
            
        if ram_used_gb > self.RAM_THRESHOLD_GB:
            logger.info(f"RAM saturation detected: {ram_used_gb:.2f}GB")
            return True
            
        return False
    
    async def offload_optimization(
        self,
        optimization_params: Dict[str, Any],
        data_paths: List[str],
        script_path: str = "/opt/trading/optimize.py",
    ) -> str:
        """
        Offload parameter optimization to cloud VPS.
        Returns job ID for tracking.
        """
        if not self.ssh_client:
            await self.connect()
        
        # Serialize parameters
        params_json = json.dumps({
            "params": optimization_params,
            "data_paths": data_paths,
            "max_memory_gb": 8.0,  # Cloud can use more RAM
        })
        
        # Create remote command
        cmd = f"python3 {script_path} --config stdin"
        
        # Execute via SSH
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        stdin.write(params_json)
        stdin.channel.shutdown_write()
        
        # Get job ID from output
        job_id = stdout.read().decode().strip()
        error = stderr.read().decode()
        
        if error:
            logger.warning(f"Cloud job warning: {error}")
        
        logger.info(f"Offloaded optimization job: {job_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of offloaded job."""
        if not self.ssh_client:
            raise RuntimeError("Not connected to cloud VPS")
        
        cmd = f"python3 /opt/trading/job_status.py --job-id {job_id}"
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        
        result = stdout.read().decode()
        error = stderr.read().decode()
        
        if error:
            return {"status": "error", "message": error}
        
        return json.loads(result)
    
    async def retrieve_results(self, job_id: str, local_path: str) -> bool:
        """Retrieve optimization results from cloud."""
        if not self.ssh_client:
            raise RuntimeError("Not connected to cloud VPS")
        
        try:
            sftp = self.ssh_client.open_sftp()
            remote_path = f"/opt/trading/results/{job_id}.json"
            sftp.get(remote_path, local_path)
            sftp.close()
            
            logger.info(f"Retrieved results for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            return False


class AutoOffloadManager:
    """
    Automatically manages cloud offloading based on system load.
    """
    
    def __init__(self, cloud_config: CloudConfig):
        self.offloader = CloudOffloader(cloud_config)
        self.pending_jobs: Dict[str, Dict] = {}
        self.monitoring = False
        
    async def start_monitoring(self, check_interval_sec: float = 30.0):
        """Start monitoring system resources and auto-offload."""
        self.monitoring = True
        
        while self.monitoring:
            try:
                if self.offloader.should_offload():
                    logger.info("Resource saturation detected, ready for offload")
                    # In production, queue pending optimizations for offload
                    
                await asyncio.sleep(check_interval_sec)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(check_interval_sec)
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.monitoring = False
        self.offloader.disconnect()


# Example usage
async def example_offload():
    config = CloudConfig(
        vps_host="your-vps-ip.example.com",
        username="trader",
        private_key_path="/home/user/.ssh/cloud_key",
    )
    
    manager = AutoOffloadManager(config)
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(manager.start_monitoring())
    
    # Simulate heavy optimization that needs offloading
    if manager.offloader.should_offload():
        job_id = await manager.offloader.offload_optimization(
            optimization_params={"lookback": 100, "threshold": 0.05},
            data_paths=["/data/btc_usdt.parquet"],
        )
        
        # Poll for completion
        while True:
            status = await manager.offloader.get_job_status(job_id)
            if status.get("status") == "completed":
                await manager.offloader.retrieve_results(job_id, "/tmp/results.json")
                break
            await asyncio.sleep(10.0)
    
    monitor_task.cancel()


if __name__ == "__main__":
    asyncio.run(example_offload())
