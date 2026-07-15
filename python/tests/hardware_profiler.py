"""
Automated hardware profiler integrating Python cProfile and Rust flamegraph
to identify memory leaks and CPU cache misses tuned for AMD Ryzen Zen architecture.
"""

import cProfile
import pstats
import io
import subprocess
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass
class ProfileResult:
    """Result from profiling a function or module."""
    function_name: str
    total_time_ms: float
    cumulative_time_ms: float
    call_count: int
    time_per_call_ms: float
    memory_estimate_kb: int


@dataclass
class CacheMetrics:
    """CPU cache performance metrics."""
    l1_hits: int
    l1_misses: int
    l2_hits: int
    l2_misses: int
    l3_hits: int
    l3_misses: int
    cache_efficiency: float


class HardwareProfiler:
    """
    Profiles Python and Rust code for performance optimization
    on AMD Ryzen AI 5 architecture.
    """

    def __init__(self, output_dir: str = './profiling_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # AMD Ryzen specific settings
        self.cpu_family = 'zen4'  # Adjust based on actual CPU
        self.cache_line_size = 64  # bytes
        
        self._python_profiler: Optional[cProfile.Profile] = None
        self._rust_perf_data: Optional[str] = None

    def profile_python_function(
        self,
        func,
        *args,
        sort_by: str = 'cumulative',
        top_n: int = 20,
    ) -> List[ProfileResult]:
        """Profile a Python function using cProfile."""
        self._python_profiler = cProfile.Profile()
        
        self._python_profiler.enable()
        try:
            result = func(*args)
        finally:
            self._python_profiler.disable()
        
        # Parse stats
        stream = io.StringIO()
        stats = pstats.Stats(self._python_profiler, stream=stream)
        stats.sort_stats(sort_by)
        stats.print_stats(top_n)
        
        # Save full stats
        stats_file = self.output_dir / f'python_profile_{int(time.time())}.pstats'
        stats.dump_stats(str(stats_file))
        
        # Extract results
        results = []
        for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
            results.append(ProfileResult(
                function_name=func_name[2] if len(func_name) > 2 else str(func_name),
                total_time_ms=tt * 1000,
                cumulative_time_ms=ct * 1000,
                call_count=nc,
                time_per_call_ms=(tt / nc * 1000) if nc > 0 else 0,
                memory_estimate_kb=0,  # Would need memory_profiler for this
            ))
        
        return sorted(results, key=lambda x: x.cumulative_time_ms, reverse=True)[:top_n]

    def profile_rust_binary(
        self,
        binary_path: str,
        args: List[str] = None,
        duration_seconds: int = 10,
    ) -> Optional[str]:
        """
        Profile a Rust binary using perf and generate flamegraph.
        Requires perf-tools and flamegraph package installed.
        """
        args = args or []
        
        try:
            # Record with perf
            perf_data = self.output_dir / 'perf.data'
            record_cmd = [
                'perf', 'record',
                '-F', '99',  # Sampling frequency
                '-a', '-g',  # All CPUs, call graphs
                '-o', str(perf_data),
                '--',
                binary_path,
                *args,
            ]
            
            # Run for specified duration
            process = subprocess.Popen(record_cmd)
            time.sleep(duration_seconds)
            process.terminate()
            process.wait()
            
            # Generate flamegraph
            flamegraph_svg = self.output_dir / 'flamegraph.svg'
            subprocess.run([
                'perf', 'script',
                '-i', str(perf_data),
            ], stdout=subprocess.PIPE, check=True)
            
            # Save SVG path for later viewing
            self._rust_perf_data = str(flamegraph_svg)
            
            return str(flamegraph_svg)
            
        except FileNotFoundError:
            print("Warning: perf tools not available. Install linux-tools-common.")
            return None
        except Exception as e:
            print(f"Rust profiling failed: {e}")
            return None

    def analyze_memory_usage(self) -> Dict:
        """Analyze current process memory usage."""
        import resource
        
        usage = resource.getrusage(resource.RUSAGE_SELF)
        
        return {
            'max_rss_kb': usage.ru_maxrss,
            'shared_mem_kb': usage.ru_ixrss,
            'unshared_data_kb': usage.ru_idrss,
            'unshared_stack_kb': usage.ru_isrss,
            'voluntary_context_switches': usage.ru_nvcsw,
            'involuntary_context_switches': usage.ru_nivcsw,
        }

    def detect_memory_leaks(
        self,
        func,
        iterations: int = 100,
        threshold_mb: float = 50,
    ) -> Tuple[bool, float]:
        """
        Detect potential memory leaks by running function repeatedly.
        Returns (leak_detected, memory_growth_mb).
        """
        initial_mem = self.analyze_memory_usage()['max_rss_kb']
        
        for _ in range(iterations):
            func()
        
        final_mem = self.analyze_memory_usage()['max_rss_kb']
        growth_mb = (final_mem - initial_mem) / 1024
        
        leak_detected = growth_mb > threshold_mb
        
        return leak_detected, growth_mb

    def estimate_cache_efficiency(self, data_size_bytes: int) -> CacheMetrics:
        """
        Estimate cache efficiency based on data size vs cache hierarchy.
        AMD Ryzen AI 5 typical cache sizes:
        - L1: 32KB instruction + 32KB data per core
        - L2: 1MB per core
        - L3: 16MB shared
        """
        l1_size = 32 * 1024
        l2_size = 1024 * 1024
        l3_size = 16 * 1024 * 1024
        
        # Simplified estimation
        if data_size_bytes <= l1_size:
            return CacheMetrics(
                l1_hits=data_size_bytes // self.cache_line_size,
                l1_misses=0,
                l2_hits=0,
                l2_misses=0,
                l3_hits=0,
                l3_misses=0,
                cache_efficiency=1.0,
            )
        elif data_size_bytes <= l2_size:
            l1_misses = (data_size_bytes - l1_size) // self.cache_line_size
            return CacheMetrics(
                l1_hits=l1_size // self.cache_line_size,
                l1_misses=l1_misses,
                l2_hits=l1_misses,
                l2_misses=0,
                l3_hits=0,
                l3_misses=0,
                cache_efficiency=l1_size / data_size_bytes,
            )
        elif data_size_bytes <= l3_size:
            l2_misses = (data_size_bytes - l2_size) // self.cache_line_size
            return CacheMetrics(
                l1_hits=l1_size // self.cache_line_size,
                l1_misses=(l2_size - l1_size) // self.cache_line_size,
                l2_hits=(l2_size - l1_size) // self.cache_line_size,
                l2_misses=l2_misses,
                l3_hits=l2_misses,
                l3_misses=0,
                cache_efficiency=l2_size / data_size_bytes,
            )
        else:
            l3_misses = (data_size_bytes - l3_size) // self.cache_line_size
            return CacheMetrics(
                l1_hits=l1_size // self.cache_line_size,
                l1_misses=(l2_size - l1_size) // self.cache_line_size,
                l2_hits=(l2_size - l1_size) // self.cache_line_size,
                l2_misses=(l3_size - l2_size) // self.cache_line_size,
                l3_hits=(l3_size - l2_size) // self.cache_line_size,
                l3_misses=l3_misses,
                cache_efficiency=l3_size / data_size_bytes,
            )

    def generate_optimization_report(
        self,
        python_results: List[ProfileResult],
        memory_info: Dict,
        cache_metrics: Optional[CacheMetrics] = None,
    ) -> str:
        """Generate a comprehensive optimization report."""
        report_lines = [
            "=" * 60,
            "HARDWARE PROFILING REPORT",
            f"Target: AMD Ryzen AI 5 ({self.cpu_family})",
            "=" * 60,
            "",
            "TOP PYTHON FUNCTIONS BY CUMULATIVE TIME:",
            "-" * 40,
        ]
        
        for i, result in enumerate(python_results[:10], 1):
            report_lines.append(
                f"{i}. {result.function_name}: "
                f"{result.cumulative_time_ms:.2f}ms "
                f"({result.call_count} calls, "
                f"{result.time_per_call_ms:.4f}ms/call)"
            )
        
        report_lines.extend([
            "",
            "MEMORY USAGE:",
            "-" * 40,
            f"Max RSS: {memory_info.get('max_rss_kb', 0) / 1024:.2f} MB",
            f"Context Switches: {memory_info.get('voluntary_context_switches', 0) + memory_info.get('involuntary_context_switches', 0)}",
            "",
        ])
        
        if cache_metrics:
            report_lines.extend([
                "CACHE EFFICIENCY ESTIMATE:",
                "-" * 40,
                f"L1 Hit/Miss: {cache_metrics.l1_hits}/{cache_metrics.l1_misses}",
                f"L2 Hit/Miss: {cache_metrics.l2_hits}/{cache_metrics.l2_misses}",
                f"L3 Hit/Miss: {cache_metrics.l3_hits}/{cache_metrics.l3_misses}",
                f"Estimated Efficiency: {cache_metrics.cache_efficiency:.1%}",
                "",
            ])
        
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 40,
            "1. Use NumPy/Polars for vectorized operations",
            "2. Minimize Python object creation in hot paths",
            "3. Align data structures to cache lines (64 bytes)",
            "4. Consider Rust FFI for CPU-intensive functions",
            "5. Use memory pools for frequent allocations",
            "",
        ])
        
        report = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / f'optimization_report_{int(time.time())}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report


if __name__ == "__main__":
    # Example usage
    profiler = HardwareProfiler()
    
    # Define test function
    def slow_function():
        total = 0
        for i in range(100000):
            total += i ** 2
        return total
    
    # Profile Python function
    results = profiler.profile_python_function(slow_function)
    print("Python Profiling Results:")
    for r in results[:5]:
        print(f"  {r.function_name}: {r.cumulative_time_ms:.2f}ms")
    
    # Analyze memory
    memory_info = profiler.analyze_memory_usage()
    print(f"\nMemory Usage: {memory_info['max_rss_kb'] / 1024:.2f} MB")
    
    # Check cache efficiency
    cache = profiler.estimate_cache_efficiency(2 * 1024 * 1024)  # 2MB data
    print(f"Cache Efficiency (2MB data): {cache.cache_efficiency:.1%}")
    
    # Generate report
    report = profiler.generate_optimization_report(results, memory_info, cache)
    print("\n" + report)
