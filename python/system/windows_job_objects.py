# Chapter 3, File 1: Windows Job Objects for Process Isolation
# python/system/windows_job_objects.py
# Replaces Linux cgroups with Windows Job Objects for memory hard-capping

import ctypes
from ctypes import wintypes
from typing import Optional
import logging

# Windows API constants
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
JOB_OBJECT_BASIC_LIMIT_INFORMATION = 2
ERROR_ALREADY_EXISTS = 183

# Memory limits (bytes)
PYTHON_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024  # 4GB hard cap for Python workers

logger = logging.getLogger(__name__)


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", ctypes.c_void_p),  # IO_COUNTERS
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsJobObject:
    """
    Windows Job Object wrapper for process isolation and memory limiting.
    Replaces Linux cgroups for HFT Python worker management.
    """
    
    def __init__(self, job_name: str = "HFTPythonWorkers"):
        self.job_name = job_name
        self.job_handle: Optional[ctypes.c_void_p] = None
        self._kernel32 = ctypes.windll.kernel32
        
        # Function prototypes
        self._kernel32.CreateJobObjectW.argtypes = [
            ctypes.c_void_p,  # LPSECURITY_ATTRIBUTES
            wintypes.LPCWSTR,  # LPCWSTR
        ]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        
        self._kernel32.OpenJobObjectW.argtypes = [
            wintypes.DWORD,  # DWORD dwDesiredAccess
            wintypes.BOOL,   # BOOL bInheritHandle
            wintypes.LPCWSTR,  # LPCWSTR lpName
        ]
        self._kernel32.OpenJobObjectW.restype = wintypes.HANDLE
        
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,  # HANDLE hJob
            wintypes.DWORD,   # JOBOBJECTINFOCLASS JobObjectInfoClass
            ctypes.c_void_p,  # LPVOID lpJobObjectInfo
            wintypes.DWORD,   # DWORD cbJobObjectInfoLength
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        
        self._kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,  # HANDLE hJob
            wintypes.HANDLE,  # HANDLE hProcess
        ]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        
        self._kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,  # HANDLE hJob
            wintypes.DWORD,   # JOBOBJECTINFOCLASS JobObjectInfoClass
            ctypes.c_void_p,  # LPVOID lpJobObjectInfo
            wintypes.DWORD,   # DWORD cbJobObjectInfoLength
            ctypes.POINTER(wintypes.DWORD),  # LPDWORD lpReturnLength
        ]
        self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def create(self, memory_limit_bytes: int = PYTHON_MEMORY_LIMIT_BYTES) -> bool:
        """Create a new job object with memory limits."""
        try:
            # Try to create new job object
            self.job_handle = self._kernel32.CreateJobObjectW(None, self.job_name)
            
            if not self.job_handle:
                # If already exists, open it
                self.job_handle = self._kernel32.OpenJobObjectW(
                    0x000F01FF,  # JOB_OBJECT_ALL_ACCESS
                    False,
                    self.job_name
                )
                
            if not self.job_handle:
                logger.error(f"Failed to create/open job object: {ctypes.get_last_error()}")
                return False
            
            # Set up extended limit information
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = memory_limit_bytes
            
            result = self._kernel32.SetInformationJobObject(
                self.job_handle,
                9,  # JobObjectExtendedLimitInformation
                ctypes.byref(info),
                ctypes.sizeof(info)
            )
            
            if not result:
                logger.error(f"Failed to set job object limits: {ctypes.get_last_error()}")
                return False
            
            logger.info(f"Created job object '{self.job_name}' with {memory_limit_bytes / (1024**3):.1f}GB memory limit")
            return True
            
        except Exception as e:
            logger.error(f"Error creating job object: {e}")
            return False

    def assign_process(self, process_handle: int) -> bool:
        """Assign a process to this job object."""
        if not self.job_handle:
            logger.error("Job object not created")
            return False
        
        result = self._kernel32.AssignProcessToJobObject(
            self.job_handle,
            process_handle
        )
        
        if not result:
            logger.error(f"Failed to assign process to job: {ctypes.get_last_error()}")
            return False
        
        logger.info(f"Assigned process {process_handle} to job object")
        return True

    def get_memory_usage(self) -> dict:
        """Query current memory usage of the job object."""
        if not self.job_handle:
            return {"error": "Job object not created"}
        
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        return_length = wintypes.DWORD()
        
        result = self._kernel32.QueryInformationJobObject(
            self.job_handle,
            9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(return_length)
        )
        
        if not result:
            return {"error": f"Query failed: {ctypes.get_last_error()}"}
        
        return {
            "process_memory_limit": info.ProcessMemoryLimit,
            "job_memory_limit": info.JobMemoryLimit,
            "peak_process_memory": info.PeakProcessMemoryUsed,
            "peak_job_memory": info.PeakJobMemoryUsed,
        }

    def close(self):
        """Close the job object handle."""
        if self.job_handle:
            self._kernel32.CloseHandle(self.job_handle)
            self.job_handle = None
            logger.info("Closed job object handle")


def wrap_python_workers_in_job_object():
    """
    Wrap all Python Ray workers in a Windows Job Object with 4GB memory limit.
    Call this at the start of your Python application.
    """
    import os
    
    job = WindowsJobObject("HFTRayWorkers")
    
    if not job.create(PYTHON_MEMORY_LIMIT_BYTES):
        logger.error("Failed to create job object for Python workers")
        return None
    
    # Assign current process to job object
    current_process = ctypes.windll.kernel32.GetCurrentProcess()
    if not job.assign_process(current_process):
        logger.error("Failed to assign current process to job object")
        job.close()
        return None
    
    logger.info(f"Python workers wrapped in job object with {PYTHON_MEMORY_LIMIT_BYTES / (1024**3):.1f}GB limit")
    return job


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test job object creation
    job = wrap_python_workers_in_job_object()
    
    if job:
        usage = job.get_memory_usage()
        print(f"Job object memory usage: {usage}")
        job.close()
