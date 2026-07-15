"""
CPU Frequency Governor for AMD Ryzen AI 5.
Dynamically switches between performance and powersave modes.
Pins CPU to performance during active trading, powersave during flat periods.
"""

import os
import logging
import subprocess
from typing import Optional, List
from enum import Enum
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GovernorMode(Enum):
    PERFORMANCE = "performance"
    POWERSAVE = "powersave"
    ONDEMAND = "ondemand"
    CONSERVATIVE = "conservative"
    SCHEDUTIL = "schedutil"


class AMDRyzenCPUGovernor:
    """
    Manages CPU frequency governor for AMD Ryzen AI 5.
    Optimizes for low latency during trading, power savings during idle.
    """
    
    # Sysfs paths for CPU frequency control
    CPUFREQ_PATH = "/sys/devices/system/cpu"
    GOVERNOR_PATH = "cpufreq/scaling_governor"
    
    def __init__(self):
        self.cpus: List[int] = self._detect_cpus()
        self.original_governors: dict[int, str] = {}
        self.current_mode: Optional[GovernorMode] = None
        
    def _detect_cpus(self) -> List[int]:
        """Detect all logical CPU cores."""
        cpus = []
        for cpu_dir in os.listdir(self.CPUFREQ_PATH):
            if cpu_dir.startswith("cpu"):
                try:
                    cpu_id = int(cpu_dir[3:])
                    governor_path = os.path.join(
                        self.CPUFREQ_PATH, cpu_dir, self.GOVERNOR_PATH
                    )
                    if os.path.exists(governor_path):
                        cpus.append(cpu_id)
                except (ValueError, FileNotFoundError):
                    continue
        return sorted(cpus)
    
    def _read_governor(self, cpu_id: int) -> str:
        """Read current governor for a CPU."""
        path = os.path.join(self.CPUFREQ_PATH, f"cpu{cpu_id}", self.GOVERNOR_PATH)
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Cannot read governor for cpu{cpu_id}: {e}")
            return "unknown"
    
    def _write_governor(self, cpu_id: int, governor: str) -> bool:
        """Set governor for a CPU."""
        path = os.path.join(self.CPUFREQ_PATH, f"cpu{cpu_id}", self.GOVERNOR_PATH)
        try:
            with open(path, 'w') as f:
                f.write(governor)
            return True
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Cannot set governor for cpu{cpu_id}: {e}")
            return False
    
    def save_current_state(self):
        """Save current governor settings for restoration."""
        self.original_governors = {
            cpu_id: self._read_governor(cpu_id) 
            for cpu_id in self.cpus
        }
        logger.info(f"Saved original governors: {self.original_governors}")
    
    def set_mode(self, mode: GovernorMode) -> bool:
        """Set all CPUs to specified governor mode."""
        success_count = 0
        
        for cpu_id in self.cpus:
            if self._write_governor(cpu_id, mode.value):
                success_count += 1
        
        self.current_mode = mode
        logger.info(f"Set {success_count}/{len(self.cpus)} CPUs to {mode.value}")
        
        return success_count == len(self.cpus)
    
    def restore_original_state(self):
        """Restore original governor settings."""
        if not self.original_governors:
            logger.warning("No original state saved")
            return
        
        for cpu_id, governor in self.original_governors.items():
            self._write_governor(cpu_id, governor)
        
        self.current_mode = None
        logger.info("Restored original governor settings")
    
    def set_performance_for_cpu(self, cpu_id: int) -> bool:
        """Set specific CPU to performance mode (for pinning critical threads)."""
        return self._write_governor(cpu_id, GovernorMode.PERFORMANCE.value)
    
    def set_powersave_for_cpu(self, cpu_id: int) -> bool:
        """Set specific CPU to powersave mode."""
        return self._write_governor(cpu_id, GovernorMode.POWERSAVE.value)


class BoostControl:
    """
    Controls AMD Precision Boost and core performance boost.
    """
    
    BOOST_PATH = "/sys/devices/system/cpu/cpufreq/boost"
    
    def __init__(self):
        self.original_boost: Optional[int] = None
        
    def is_boost_available(self) -> bool:
        """Check if boost control is available."""
        return os.path.exists(self.BOOST_PATH)
    
    def get_boost_status(self) -> bool:
        """Get current boost status."""
        try:
            with open(self.BOOST_PATH, 'r') as f:
                return int(f.read().strip()) == 1
        except Exception as e:
            logger.warning(f"Cannot read boost status: {e}")
            return False
    
    def set_boost(self, enabled: bool) -> bool:
        """Enable or disable CPU boost."""
        if not self.is_boost_available():
            logger.warning("Boost control not available")
            return False
        
        try:
            with open(self.BOOST_PATH, 'w') as f:
                f.write('1' if enabled else '0')
            logger.info(f"CPU boost {'enabled' if enabled else 'disabled'}")
            return True
        except PermissionError as e:
            logger.error(f"Permission denied setting boost: {e}")
            return False
    
    def save_boost_state(self):
        """Save current boost state."""
        self.original_boost = 1 if self.get_boost_status() else 0
    
    def restore_boost_state(self):
        """Restore original boost state."""
        if self.original_boost is not None:
            self.set_boost(bool(self.original_boost))


class TradingCPUPolicy:
    """
    High-level CPU policy manager for trading bot.
    Automatically adjusts based on trading activity.
    """
    
    def __init__(self):
        self.governor = AMDRyzenCPUGovernor()
        self.boost = BoostControl()
        self.is_trading_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start CPU management."""
        # Save current state
        self.governor.save_current_state()
        self.boost.save_boost_state()
        
        # Set initial trading mode
        await self.set_trading_mode(True)
        
        logger.info("CPU governor management started")
    
    async def stop(self):
        """Stop CPU management and restore original settings."""
        await self.set_trading_mode(False)
        self.governor.restore_original_state()
        self.boost.restore_boost_state()
        logger.info("CPU governor management stopped")
    
    async def set_trading_mode(self, active: bool):
        """Switch CPU mode based on trading activity."""
        self.is_trading_active = active
        
        if active:
            logger.info("Activating PERFORMANCE mode for trading...")
            self.governor.set_mode(GovernorMode.PERFORMANCE)
            self.boost.set_boost(True)  # Enable boost for max frequency
            
            # Pin critical CPUs to performance
            for cpu_id in self.governor.cpus[:4]:  # First 4 cores for critical tasks
                self.governor.set_performance_for_cpu(cpu_id)
        else:
            logger.info("Switching to POWERSAVE mode (idle)...")
            self.governor.set_mode(Governor.POWERSAVE)
            self.boost.set_boost(False)  # Disable boost to save power
    
    def get_status(self) -> dict:
        """Get current CPU governor status."""
        return {
            "is_trading_active": self.is_trading_active,
            "current_mode": self.governor.current_mode.value if self.governor.current_mode else None,
            "boost_enabled": self.boost.get_boost_status(),
            "cpu_count": len(self.governor.cpus),
            "governors": {
                f"cpu{i}": self.governor._read_governor(i)
                for i in self.governor.cpus[:4]  # Report first 4 CPUs
            }
        }


# Example usage
async def main():
    policy = TradingCPUPolicy()
    
    await policy.start()
    
    print("Initial status:", policy.get_status())
    
    # Simulate trading session
    print("\nStarting trading session...")
    await policy.set_trading_mode(True)
    print("Status:", policy.get_status())
    
    await asyncio.sleep(10)  # Trade for 10 seconds
    
    # End trading session
    print("\nEnding trading session...")
    await policy.set_trading_mode(False)
    print("Status:", policy.get_status())
    
    await policy.stop()


if __name__ == "__main__":
    # Check for root permissions
    if os.geteuid() != 0:
        print("WARNING: This script requires root privileges to change CPU governors")
        print("Run with: sudo python cpu_governor.py")
    
    asyncio.run(main())
