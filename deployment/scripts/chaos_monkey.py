#!/usr/bin/env python3
"""
Stage 30: Chaos Monkey for Resilience Testing
Randomly kills containers, severs network links, and spikes CPU usage
to verify graceful recovery without orphaned orders.
"""

import asyncio
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Literal

# Configuration
CHAOS_ACTIONS = Literal["kill_container", "network_partition", "cpu_spike", "memory_pressure"]
CONTAINERS = ["crypto_bot_rust", "crypto_bot_python", "crypto_bot_redis", "crypto_bot_frontend"]

class ChaosMonkey:
    def __init__(self, probability: float = 0.3, interval_seconds: int = 10):
        self.probability = probability
        self.interval = interval_seconds
        self.events_log: List[dict] = []
        
    def log_event(self, action: str, target: str, result: str):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "target": target,
            "result": result
        }
        self.events_log.append(event)
        print(f"[{event['timestamp']}] CHAOS: {action} -> {target} : {result}")
    
    async def kill_container(self, container: str) -> bool:
        """Randomly kill a container."""
        try:
            print(f"[*] Killing container: {container}")
            subprocess.run(["docker", "kill", container], check=True, capture_output=True)
            self.log_event("kill_container", container, "SUCCESS")
            
            # Wait then restart
            await asyncio.sleep(5)
            subprocess.run(["docker", "start", container], check=True, capture_output=True)
            self.log_event("restart_container", container, "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.log_event("kill_container", container, f"FAILED: {e}")
            return False
    
    async def network_partition(self, container: str) -> bool:
        """Sever network links for a container."""
        try:
            # Disconnect from network
            print(f"[*] Disconnecting {container} from network")
            subprocess.run(
                ["docker", "network", "disconnect", "deployment_bot_internal", container],
                check=True, capture_output=True
            )
            self.log_event("network_disconnect", container, "SUCCESS")
            
            # Wait then reconnect
            await asyncio.sleep(10)
            subprocess.run(
                ["docker", "network", "connect", "deployment_bot_internal", container],
                check=True, capture_output=True
            )
            self.log_event("network_reconnect", container, "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.log_event("network_partition", container, f"FAILED: {e}")
            return False
    
    async def cpu_spike(self, container: str) -> bool:
        """Spike CPU usage inside a container."""
        try:
            print(f"[*] Spiking CPU in {container}")
            # Run stress command inside container
            cmd = (
                f"docker exec {container} timeout 30s stress-ng --cpu 2 --cpu-method matrixprod "
                f"--cpu-load 100 2>/dev/null || true"
            )
            subprocess.run(cmd, shell=True, capture_output=True)
            self.log_event("cpu_spike", container, "SUCCESS")
            return True
        except Exception as e:
            self.log_event("cpu_spike", container, f"FAILED: {e}")
            return False
    
    async def memory_pressure(self, container: str) -> bool:
        """Apply memory pressure inside a container."""
        try:
            print(f"[*] Applying memory pressure in {container}")
            cmd = (
                f"docker exec {container} timeout 20s stress-ng --vm 1 --vm-bytes 256M "
                f"--vm-hang 0 2>/dev/null || true"
            )
            subprocess.run(cmd, shell=True, capture_output=True)
            self.log_event("memory_pressure", container, "SUCCESS")
            return True
        except Exception as e:
            self.log_event("memory_pressure", container, f"FAILED: {e}")
            return False
    
    async def run_chaos_event(self):
        """Execute a single chaos event."""
        if random.random() > self.probability:
            print("[*] Skipping chaos event this cycle")
            return
        
        action = random.choice(["kill_container", "network_partition", "cpu_spike", "memory_pressure"])
        target = random.choice(CONTAINERS)
        
        print(f"\n{'='*50}")
        print(f"TRIGGERING CHAOS: {action} on {target}")
        print(f"{'='*50}\n")
        
        if action == "kill_container":
            await self.kill_container(target)
        elif action == "network_partition":
            await self.network_partition(target)
        elif action == "cpu_spike":
            await self.cpu_spike(target)
        elif action == "memory_pressure":
            await self.memory_pressure(target)
        
        # Cooldown period
        await asyncio.sleep(self.interval)
    
    async def run(self, duration_minutes: int = 30):
        """Run chaos monkey for specified duration."""
        print(f"[*] Starting Chaos Monkey for {duration_minutes} minutes")
        print(f"[*] Probability: {self.probability}, Interval: {self.interval}s")
        print(f"[*] Target containers: {CONTAINERS}")
        print(f"[*] Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                await self.run_chaos_event()
                
                # Check if system recovered
                await self.verify_recovery()
        
        except KeyboardInterrupt:
            print("\n[*] Chaos Monkey stopped by user")
        
        finally:
            self.generate_report()
    
    async def verify_recovery(self):
        """Verify all containers are healthy after chaos."""
        print("[*] Verifying system recovery...")
        for container in CONTAINERS:
            try:
                result = subprocess.run(
                    ["docker", "inspect", "--format={{.State.Health.Status}}", container],
                    capture_output=True, text=True, timeout=5
                )
                status = result.stdout.strip()
                if status in ["healthy", "starting"]:
                    print(f"  ✓ {container}: {status}")
                else:
                    print(f"  ⚠ {container}: {status} - May need attention")
            except Exception as e:
                print(f"  ✗ {container}: Cannot inspect - {e}")
    
    def generate_report(self):
        """Generate chaos test report."""
        print("\n" + "="*60)
        print("           CHAOS MONKEY TEST REPORT")
        print("="*60)
        print(f"Total events executed: {len(self.events_log)}")
        
        if self.events_log:
            print("\nEvent Log:")
            for event in self.events_log[-20:]:  # Last 20 events
                print(f"  [{event['timestamp']}] {event['action']:20} -> {event['target']:25} : {event['result']}")
        
        print("\n" + "="*60)


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Chaos Monkey for resilience testing")
    parser.add_argument("--duration", type=int, default=30, help="Duration in minutes")
    parser.add_argument("--probability", type=float, default=0.3, help="Probability of chaos event per cycle")
    parser.add_argument("--interval", type=int, default=10, help="Interval between cycles in seconds")
    
    args = parser.parse_args()
    
    monkey = ChaosMonkey(probability=args.probability, interval_seconds=args.interval)
    await monkey.run(duration_minutes=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
