"""
Linux perf event integration for AMD hardware performance counters
Reads PMC (Performance Monitoring Counters) in real-time
Tracks cache misses, branch mispredictions, and instruction cycles per trade
Memory-efficient design for <14GB RAM constraint
"""

import os
import struct
import ctypes
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from collections import deque


# AMD-specific perf event types and configurations
PERF_TYPE_HARDWARE = 0
PERF_TYPE_SOFTWARE = 1
PERF_TYPE_HW_CACHE = 2

# AMD Zen architecture event codes
AMD_EVENTS = {
    'L1_DCACHE_MISS': 0x0041,      # L1 Data Cache Miss
    'L2_CACHE_MISS': 0x0060,       # L2 Cache Miss  
    'L3_CACHE_MISS': 0x004D,       # L3 Cache Miss (Last Level)
    'BRANCH_MISPRED': 0x00C3,      # Branch Misprediction
    'INSTRUCTIONS': 0x00C0,        # Instructions Retired
    'CPU_CYCLES': 0x0076,          # CPU Cycles
    'TLB_MISS': 0x0054,            # TLB Miss
}


@dataclass
class PerfEventAttr:
    """perf_event_attr structure for perf_event_open"""
    type: int
    size: int
    config: int
    sample_period: int
    sample_type: int
    read_format: int
    flags: int
    wakeup_events: int
    bp_type: int
    bp_addr: int
    bp_len: int
    branch_sample_type: int


@dataclass
class PerfCounter:
    """Single performance counter"""
    name: str
    event_code: int
    umask: int
    fd: Optional[int]
    value: int
    enabled: bool


class PerfEventReader:
    """
    Low-level perf event reader for AMD PMCs
    Uses perf_event_open syscall for direct hardware counter access
    """
    
    def __init__(self, pid: int = -1, cpu: int = -1):
        self.pid = pid  # -1 for current process
        self.cpu = cpu  # -1 for any CPU
        
        # Load libc for syscall
        self.libc = ctypes.CDLL('libc.so.6', use_errno=True)
        
        # Counters
        self.counters: List[PerfCounter] = []
        self.mmap_buffer: Optional[ctypes.c_void_p] = None
        self.mmap_size: int = 0
        
        # Statistics
        self.read_count: int = 0
        self.last_values: Dict[str, int] = {}
    
    def _perf_event_open(self, attr: PerfEventAttr, pid: int, cpu: int, 
                         group_fd: int, flags: int) -> int:
        """Call perf_event_open syscall"""
        # Serialize attr structure
        attr_data = struct.pack(
            'IIQQQIIIHHI',
            attr.type,
            attr.size,
            attr.config,
            attr.sample_period,
            attr.sample_type,
            attr.read_format,
            attr.flags,
            attr.wakeup_events,
            attr.bp_type,
            attr.bp_addr,
            attr.bp_len,
        )
        
        result = self.libc.syscall(
            298,  # perf_event_open syscall number on x86_64
            attr_data,
            pid,
            cpu,
            group_fd,
            flags
        )
        
        if result < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"perf_event_open failed: {os.strerror(errno)}")
        
        return result
    
    def create_counter(self, name: str, event_code: int, umask: int = 0x00) -> PerfCounter:
        """Create a new hardware performance counter"""
        config = (umask << 8) | (event_code & 0xFF)
        
        attr = PerfEventAttr(
            type=PERF_TYPE_HARDWARE,
            size=ctypes.sizeof(ctypes.c_uint64),
            config=config,
            sample_period=0,  # Counting mode
            sample_type=0,
            read_format=0x00000002,  # PERF_FORMAT_TOTAL_TIME_ENABLED
            flags=0x00000001,  # PERF_FLAG_FD_CLOEXEC
            wakeup_events=0,
            bp_type=0,
            bp_addr=0,
            bp_len=0,
            branch_sample_type=0,
        )
        
        try:
            fd = self._perf_event_open(attr, self.pid, self.cpu, -1, 0)
            counter = PerfCounter(name, event_code, umask, fd, 0, True)
            self.counters.append(counter)
            return counter
        except OSError as e:
            print(f"Warning: Could not create counter {name}: {e}")
            counter = PerfCounter(name, event_code, umask, None, 0, False)
            self.counters.append(counter)
            return counter
    
    def setup_amd_counters(self) -> List[PerfCounter]:
        """Setup standard AMD Ryzen performance counters"""
        counters = []
        
        # Instructions retired
        counters.append(self.create_counter('instructions', AMD_EVENTS['INSTRUCTIONS']))
        
        # CPU cycles
        counters.append(self.create_counter('cycles', AMD_EVENTS['CPU_CYCLES']))
        
        # L1 cache misses
        counters.append(self.create_counter('l1_miss', AMD_EVENTS['L1_DCACHE_MISS']))
        
        # L2 cache misses
        counters.append(self.create_counter('l2_miss', AMD_EVENTS['L2_CACHE_MISS']))
        
        # L3 cache misses
        counters.append(self.create_counter('l3_miss', AMD_EVENTS['L3_CACHE_MISS']))
        
        # Branch mispredictions
        counters.append(self.create_counter('branch_miss', AMD_EVENTS['BRANCH_MISPRED']))
        
        return counters
    
    def read_counter(self, counter: PerfCounter) -> int:
        """Read current counter value"""
        if counter.fd is None or not counter.enabled:
            return 0
        
        try:
            data = os.read(counter.fd, 24)  # 3 x 8-byte values
            if len(data) >= 8:
                value = struct.unpack('Q', data[:8])[0]
                counter.value = value
                self.last_values[counter.name] = value
                return value
        except OSError:
            pass
        
        return 0
    
    def read_all_counters(self) -> Dict[str, int]:
        """Read all active counters"""
        results = {}
        for counter in self.counters:
            if counter.enabled:
                results[counter.name] = self.read_counter(counter)
        self.read_count += 1
        return results
    
    def enable_counter(self, counter: PerfCounter):
        """Enable a counter"""
        if counter.fd is not None:
            try:
                self.libc.ioctl(counter.fd, 0x20009401)  # PERF_EVENT_IOC_ENABLE
                counter.enabled = True
            except:
                pass
    
    def disable_counter(self, counter: PerfCounter):
        """Disable a counter"""
        if counter.fd is not None:
            try:
                self.libc.ioctl(counter.fd, 0x20009400)  # PERF_EVENT_IOC_DISABLE
                counter.enabled = False
            except:
                pass
    
    def enable_all(self):
        """Enable all counters"""
        for counter in self.counters:
            self.enable_counter(counter)
    
    def disable_all(self):
        """Disable all counters"""
        for counter in self.counters:
            self.disable_counter(counter)
    
    def close(self):
        """Close all counter file descriptors"""
        for counter in self.counters:
            if counter.fd is not None:
                try:
                    os.close(counter.fd)
                except:
                    pass
        self.counters.clear()


class TradeLatencyProfiler:
    """
    Profile latency and hardware events per trade
    Tracks IPC (Instructions Per Cycle), cache miss rates, etc.
    """
    
    def __init__(self, window_size: int = 1000):
        self.perf_reader = PerfEventReader()
        self.window_size = window_size
        
        # Fixed-size history buffers
        self.latency_history: deque = deque(maxlen=window_size)
        self.ipc_history: deque = deque(maxlen=window_size)
        self.cache_miss_history: deque = deque(maxlen=window_size)
        
        # Baseline measurements
        self.baseline_instructions: int = 0
        self.baseline_cycles: int = 0
        self.baseline_time_ns: int = 0
        
        # Setup counters
        self.perf_reader.setup_amd_counters()
    
    def start_trade_measurement(self):
        """Start measuring a trade execution"""
        self.perf_reader.enable_all()
        readings = self.perf_reader.read_all_counters()
        
        self.baseline_instructions = readings.get('instructions', 0)
        self.baseline_cycles = readings.get('cycles', 0)
        self.baseline_time_ns = time.time_ns()
    
    def end_trade_measurement(self, trade_id: str) -> Dict:
        """End trade measurement and calculate metrics"""
        end_time_ns = time.time_ns()
        self.perf_reader.disable_all()
        
        readings = self.perf_reader.read_all_counters()
        
        # Calculate deltas
        instructions = readings.get('instructions', 0) - self.baseline_instructions
        cycles = readings.get('cycles', 0) - self.baseline_cycles
        l1_miss = readings.get('l1_miss', 0) - self.baseline_instructions  # Approximate
        l2_miss = readings.get('l2_miss', 0) - self.baseline_instructions
        l3_miss = readings.get('l3_miss', 0) - self.baseline_instructions
        
        latency_us = (end_time_ns - self.baseline_time_ns) / 1000
        
        # Calculate IPC
        ipc = instructions / max(cycles, 1)
        
        # Calculate cache miss rates
        total_accesses = max(instructions, 1)  # Approximate
        l1_miss_rate = l1_miss / total_accesses
        l2_miss_rate = l2_miss / total_accesses
        l3_miss_rate = l3_miss / total_accesses
        
        # Store in history
        self.latency_history.append(latency_us)
        self.ipc_history.append(ipc)
        self.cache_miss_history.append(l3_miss_rate)
        
        return {
            'trade_id': trade_id,
            'latency_us': latency_us,
            'instructions': instructions,
            'cycles': cycles,
            'ipc': ipc,
            'l1_miss_rate': l1_miss_rate,
            'l2_miss_rate': l2_miss_rate,
            'l3_miss_rate': l3_miss_rate,
            'timestamp_ns': end_time_ns,
        }
    
    def get_statistics(self) -> Dict:
        """Get aggregated statistics"""
        if not self.latency_history:
            return {}
        
        import numpy as np
        
        latencies = np.array(self.latency_history)
        ipcs = np.array(self.ipc_history)
        cache_misses = np.array(self.cache_miss_history)
        
        return {
            'avg_latency_us': float(np.mean(latencies)),
            'p50_latency_us': float(np.percentile(latencies, 50)),
            'p99_latency_us': float(np.percentile(latencies, 99)),
            'max_latency_us': float(np.max(latencies)),
            'avg_ipc': float(np.mean(ipcs)),
            'avg_cache_miss_rate': float(np.mean(cache_misses)),
            'sample_count': len(self.latency_history),
        }
    
    def reset(self):
        """Reset all measurements"""
        self.latency_history.clear()
        self.ipc_history.clear()
        self.cache_miss_history.clear()
        self.perf_reader.disable_all()


class AMDCPUMonitor:
    """
    Monitor AMD Ryzen CPU-specific metrics
    Includes CCX topology awareness and NUMA considerations
    """
    
    def __init__(self):
        self.perf_readers: Dict[int, PerfEventReader] = {}
        self.ccx_topology: Dict[int, List[int]] = {}
        
        # Detect CPU topology
        self._detect_topology()
    
    def _detect_topology(self):
        """Detect AMD Ryzen CCX topology"""
        try:
            # Read CPU topology from sysfs
            for cpu_dir in os.listdir('/sys/devices/system/cpu/'):
                if cpu_dir.startswith('cpu'):
                    cpu_id = int(cpu_dir[3:])
                    
                    # Read core ID and die ID
                    try:
                        with open(f'/sys/devices/system/cpu/{cpu_dir}/topology/core_id') as f:
                            core_id = int(f.read().strip())
                        
                        # Group by CCX (simplified detection)
                        ccx_id = core_id // 6  # 6 cores per CCX typically
                        if ccx_id not in self.ccx_topology:
                            self.ccx_topology[ccx_id] = []
                        self.ccx_topology[ccx_id].append(cpu_id)
                    except:
                        pass
        except Exception as e:
            print(f"Topology detection warning: {e}")
    
    def create_per_cpu_monitors(self, cpu_ids: Optional[List[int]] = None):
        """Create perf readers for specific CPUs"""
        if cpu_ids is None:
            cpu_ids = list(range(os.cpu_count()))
        
        for cpu_id in cpu_ids:
            reader = PerfEventReader(pid=-1, cpu=cpu_id)
            reader.setup_amd_counters()
            self.perf_readers[cpu_id] = reader
    
    def get_ccx_metrics(self, ccx_id: int) -> Dict:
        """Get aggregated metrics for a CCX"""
        if ccx_id not in self.ccx_topology:
            return {}
        
        cpu_ids = self.ccx_topology[ccx_id]
        metrics = {'instructions': 0, 'cycles': 0, 'l3_miss': 0}
        
        for cpu_id in cpu_ids:
            if cpu_id in self.perf_readers:
                readings = self.perf_readers[cpu_id].read_all_counters()
                metrics['instructions'] += readings.get('instructions', 0)
                metrics['cycles'] += readings.get('cycles', 0)
                metrics['l3_miss'] += readings.get('l3_miss', 0)
        
        return metrics
    
    def shutdown(self):
        """Cleanup all resources"""
        for reader in self.perf_readers.values():
            reader.close()
        self.perf_readers.clear()


# Import time module
import time

if __name__ == "__main__":
    # Example usage
    profiler = TradeLatencyProfiler()
    
    # Simulate trade measurement
    profiler.start_trade_measurement()
    time.sleep(0.001)  # Simulate work
    result = profiler.end_trade_measurement("TRADE_001")
    
    print(f"Trade latency: {result['latency_us']:.2f} us")
    print(f"IPC: {result['ipc']:.2f}")
    print(f"L3 miss rate: {result['l3_miss_rate']:.4f}")
    
    stats = profiler.get_statistics()
    print(f"\nStatistics: {stats}")
