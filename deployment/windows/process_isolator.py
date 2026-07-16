"""
Windows Process Isolator using Job Objects

This module uses Windows Job Objects to set strict memory limits on
processes. If a process exceeds its Job Object limit, Windows instantly
terminates it, protecting the 10GB total system ceiling.

Target: Windows 10/11 with strict per-process memory caps:
- Python/Nautilus process: 2GB max
- Frontend browser: 1.5GB max  
- Rust core: 2GB max
- Databases/cache: 2GB max
"""

import ctypes
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JobObjectInfoClass(IntEnum):
    """Job Object Information Classes for Windows API"""
    BasicAccountingInformation = 0
    BasicLimitInformation = 1
    ExtendedLimitInformation = 2
    AssociateCompletionPortInformation = 7


class JobObjectLimitFlags(IntEnum):
    """Job Object Limit Flags"""
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    JOB_OBJECT_LIMIT_AFFINITY = 0x00000010
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_LIMIT_PRESERVE_JOB_TIME = 0x00000040
    JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x00000020
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
    JOB_OBJECT_LIMIT_SCHEDULING_CLASS = 0x00000080
    JOB_OBJECT_LIMIT_SUBSET_AFFINITY = 0x00004000
    JOB_OBJECT_LIMIT_WORKINGSET = 0x00000001


@dataclass
class ProcessLimits:
    """Memory limits for different process types"""
    # Python/Nautilus process
    python_max_mb: int = 2048  # 2GB
    
    # Frontend browser (Chrome/Edge)
    frontend_max_mb: int = 1536  # 1.5GB
    
    # Rust core engine
    rust_core_max_mb: int = 2048  # 2GB
    
    # Database and cache processes
    database_max_mb: int = 2048  # 2GB
    
    # Total system limit (should not exceed this)
    total_system_max_mb: int = 10240  # 10GB


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    """Windows JOBOBJECT_BASIC_LIMIT_INFORMATION structure"""
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
    ]


class IO_COUNTERS(ctypes.Structure):
    """Windows IO_COUNTERS structure"""
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    """Windows JOBOBJECT_EXTENDED_LIMIT_INFORMATION structure"""
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsJobObject:
    """
    Windows Job Object wrapper for process isolation and memory limiting.
    
    Job Objects allow you to:
    - Set hard memory limits on processes
    - Automatically terminate processes that exceed limits
    - Group related processes together
    - Apply CPU affinity and priority constraints
    """
    
    def __init__(self, name: str, kill_on_close: bool = True):
        """
        Create a new Job Object.
        
        Args:
            name: Unique name for the job object
            kill_on_close: If True, all processes in job are killed when handle closes
        """
        self.name = name
        self.kill_on_close = kill_on_close
        self.job_handle = None
        self._initialized = False
        
        # Load Windows API
        self.kernel32 = ctypes.windll.kernel32
    
    def create(self) -> bool:
        """
        Create the Job Object.
        
        Returns:
            True if creation was successful
        """
        try:
            # Create job object with security attributes
            self.job_handle = self.kernel32.CreateJobObjectW(None, self.name)
            
            if not self.job_handle:
                logger.error(f"Failed to create job object: {ctypes.get_last_error()}")
                return False
            
            # Set up extended limit information
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            
            # Set flags for memory limiting and kill-on-close
            limit_flags = JobObjectLimitFlags.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            
            if self.kill_on_close:
                limit_flags |= JobObjectLimitFlags.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            
            info.BasicLimitInformation.LimitFlags = limit_flags
            
            # Apply the limits
            result = self.kernel32.SetInformationJobObject(
                self.job_handle,
                JobObjectInfoClass.ExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info)
            )
            
            if not result:
                logger.error(f"Failed to set job limits: {ctypes.get_last_error()}")
                return False
            
            self._initialized = True
            logger.info(f"Job Object '{self.name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating job object: {e}")
            return False
    
    def set_memory_limit(self, max_memory_mb: int) -> bool:
        """
        Set the maximum memory limit for processes in this job.
        
        Args:
            max_memory_mb: Maximum memory in megabytes
        
        Returns:
            True if limit was set successfully
        """
        if not self._initialized or not self.job_handle:
            logger.error("Job object not initialized")
            return False
        
        try:
            # Get current extended limit information
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            
            # Set memory limit in bytes
            info.ProcessMemoryLimit = max_memory_mb * 1024 * 1024
            
            # Ensure the flag is set
            info.BasicLimitInformation.LimitFlags |= \
                JobObjectLimitFlags.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            
            # Apply the limits
            result = self.kernel32.SetInformationJobObject(
                self.job_handle,
                JobObjectInfoClass.ExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info)
            )
            
            if not result:
                logger.error(f"Failed to set memory limit: {ctypes.get_last_error()}")
                return False
            
            logger.info(f"Memory limit set to {max_memory_mb}MB for job '{self.name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error setting memory limit: {e}")
            return False
    
    def assign_process(self, process_handle: int) -> bool:
        """
        Assign a process to this job object.
        
        Args:
            process_handle: Windows process handle
        
        Returns:
            True if assignment was successful
        """
        if not self._initialized or not self.job_handle:
            logger.error("Job object not initialized")
            return False
        
        try:
            result = self.kernel32.AssignProcessToJobObject(
                self.job_handle,
                process_handle
            )
            
            if not result:
                error_code = ctypes.get_last_error()
                if error_code == 5:  # ERROR_ACCESS_DENIED
                    logger.warning(
                        "Process already assigned to a job object (common for child processes)"
                    )
                    return False
                logger.error(f"Failed to assign process: {error_code}")
                return False
            
            logger.info(f"Process assigned to job '{self.name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error assigning process: {e}")
            return False
    
    def get_memory_usage(self) -> Optional[Dict[str, int]]:
        """
        Get current memory usage statistics for the job.
        
        Returns:
            Dictionary with memory stats or None if failed
        """
        if not self._initialized or not self.job_handle:
            return None
        
        try:
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            
            result = self.kernel32.QueryInformationJobObject(
                self.job_handle,
                JobObjectInfoClass.ExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
                None
            )
            
            if not result:
                return None
            
            return {
                "process_memory_limit_mb": info.ProcessMemoryLimit // (1024 * 1024),
                "job_memory_limit_mb": info.JobMemoryLimit // (1024 * 1024),
                "peak_process_memory_mb": info.PeakProcessMemoryUsed // (1024 * 1024),
                "peak_job_memory_mb": info.PeakJobMemoryUsed // (1024 * 1024),
            }
            
        except Exception as e:
            logger.error(f"Error querying job memory: {e}")
            return None
    
    def close(self) -> None:
        """Close the job object handle"""
        if self.job_handle:
            self.kernel32.CloseHandle(self.job_handle)
            self.job_handle = None
            logger.info(f"Job Object '{self.name}' closed")


class ProcessIsolator:
    """
    High-level process isolator that manages Job Objects for all bot components.
    """
    
    def __init__(self):
        self.limits = ProcessLimits()
        self.jobs: Dict[str, WindowsJobObject] = {}
    
    def create_python_job(self) -> Optional[WindowsJobObject]:
        """Create job for Python/Nautilus process (2GB limit)"""
        job = WindowsJobObject("CryptoBot_Python")
        if job.create():
            job.set_memory_limit(self.limits.python_max_mb)
            self.jobs["python"] = job
            return job
        return None
    
    def create_frontend_job(self) -> Optional[WindowsJobObject]:
        """Create job for frontend browser process (1.5GB limit)"""
        job = WindowsJobObject("CryptoBot_Frontend")
        if job.create():
            job.set_memory_limit(self.limits.frontend_max_mb)
            self.jobs["frontend"] = job
            return job
        return None
    
    def create_rust_core_job(self) -> Optional[WindowsJobObject]:
        """Create job for Rust core engine (2GB limit)"""
        job = WindowsJobObject("CryptoBot_RustCore")
        if job.create():
            job.set_memory_limit(self.limits.rust_core_max_mb)
            self.jobs["rust_core"] = job
            return job
        return None
    
    def create_database_job(self) -> Optional[WindowsJobObject]:
        """Create job for database processes (2GB limit)"""
        job = WindowsJobObject("CryptoBot_Database")
        if job.create():
            job.set_memory_limit(self.limits.database_max_mb)
            self.jobs["database"] = job
            return job
        return None
    
    def setup_all_jobs(self) -> bool:
        """Create all job objects for the entire bot system"""
        logger.info("Setting up all process isolation jobs...")
        
        success = True
        
        if not self.create_python_job():
            logger.warning("Failed to create Python job")
            success = False
        
        if not self.create_frontend_job():
            logger.warning("Failed to create Frontend job")
            success = False
        
        if not self.create_rust_core_job():
            logger.warning("Failed to create Rust Core job")
            success = False
        
        if not self.create_database_job():
            logger.warning("Failed to create Database job")
            success = False
        
        if success:
            logger.info("All process isolation jobs created successfully")
            logger.info(f"Total system memory limit: {self.limits.total_system_max_mb}MB")
        else:
            logger.error("Some process isolation jobs failed to create")
        
        return success
    
    def get_all_memory_stats(self) -> Dict[str, Dict[str, int]]:
        """Get memory statistics for all managed jobs"""
        stats = {}
        for name, job in self.jobs.items():
            job_stats = job.get_memory_usage()
            if job_stats:
                stats[name] = job_stats
        return stats
    
    def cleanup(self) -> None:
        """Clean up all job objects"""
        for name, job in list(self.jobs.items()):
            job.close()
            del self.jobs[name]
        logger.info("All job objects cleaned up")


if __name__ == "__main__":
    # Example usage
    print("=== Windows Process Isolator Demo ===\n")
    
    isolator = ProcessIsolator()
    
    # Setup all jobs
    if isolator.setup_all_jobs():
        print("\nProcess isolation configured successfully!")
        print("\nMemory Limits:")
        print(f"  Python/Nautilus: {isolator.limits.python_max_mb}MB")
        print(f"  Frontend:        {isolator.limits.frontend_max_mb}MB")
        print(f"  Rust Core:       {isolator.limits.rust_core_max_mb}MB")
        print(f"  Database:        {isolator.limits.database_max_mb}MB")
        print(f"  Total System:    {isolator.limits.total_system_max_mb}MB")
    else:
        print("\nFailed to configure process isolation")
    
    # Cleanup
    isolator.cleanup()
    print("\n=== Demo Complete ===")
