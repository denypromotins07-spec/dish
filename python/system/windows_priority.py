# Chapter 3, File 3: Windows Process Priority Management
# python/system/windows_priority.py
# Sets process and thread priorities for HFT Python workers

import ctypes
from ctypes import wintypes
import logging
from typing import Optional
import threading

logger = logging.getLogger(__name__)

# Windows priority constants
IDLE_PRIORITY_CLASS = 0x00000040
NORMAL_PRIORITY_CLASS = 0x00000020
HIGH_PRIORITY_CLASS = 0x00000080
REALTIME_PRIORITY_CLASS = 0x00000100

THREAD_PRIORITY_IDLE = -15
THREAD_PRIORITY_LOWEST = -2
THREAD_PRIORITY_BELOW_NORMAL = -1
THREAD_PRIORITY_NORMAL = 0
THREAD_PRIORITY_ABOVE_NORMAL = 1
THREAD_PRIORITY_HIGHEST = 2
THREAD_PRIORITY_TIME_CRITICAL = 15

# Access rights for process manipulation
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_ALL_ACCESS = 0x001F0FFF

THREAD_SET_INFORMATION = 0x0020
THREAD_QUERY_INFORMATION = 0x0080


class WindowsPriorityManager:
    """
    Manages Windows process and thread priorities for HFT workloads.
    Ensures Rust core has REALTIME_PRIORITY while Python workers get HIGH_PRIORITY.
    """
    
    def __init__(self):
        self._kernel32 = ctypes.windll.kernel32
        
        # Setup function prototypes
        self._kernel32.GetCurrentProcess.argtypes = []
        self._kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        
        self._kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._kernel32.SetPriorityClass.restype = wintypes.BOOL
        
        self._kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        self._kernel32.GetPriorityClass.restype = wintypes.DWORD
        
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        
        self._kernel32.GetCurrentThreadId.argtypes = []
        self._kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        
        self._kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenThread.restype = wintypes.HANDLE
        
        self._kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        self._kernel32.SetThreadPriority.restype = wintypes.BOOL
        
        self._kernel32.GetThreadPriority.argtypes = [wintypes.HANDLE]
        self._kernel32.GetThreadPriority.restype = ctypes.c_int
        
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
    
    def set_current_process_priority(self, priority_class: int) -> bool:
        """Set priority class for current process."""
        try:
            current_process = self._kernel32.GetCurrentProcess()
            
            result = self._kernel32.SetPriorityClass(current_process, priority_class)
            
            if result:
                logger.info(f"Set process priority to {priority_class:#x}")
                return True
            else:
                logger.error(f"Failed to set priority: {ctypes.get_last_error()}")
                return False
        except Exception as e:
            logger.error(f"Error setting process priority: {e}")
            return False
    
    def get_current_process_priority(self) -> Optional[int]:
        """Get current process priority class."""
        try:
            current_process = self._kernel32.GetCurrentProcess()
            return self._kernel32.GetPriorityClass(current_process)
        except Exception as e:
            logger.error(f"Error getting process priority: {e}")
            return None
    
    def set_current_thread_priority(self, priority: int) -> bool:
        """Set priority for current thread."""
        try:
            current_thread_id = self._kernel32.GetCurrentThreadId()
            
            thread_handle = self._kernel32.OpenThread(
                THREAD_SET_INFORMATION,
                False,
                current_thread_id
            )
            
            if not thread_handle:
                logger.error(f"Failed to open thread: {ctypes.get_last_error()}")
                return False
            
            result = self._kernel32.SetThreadPriority(thread_handle, priority)
            self._kernel32.CloseHandle(thread_handle)
            
            if result:
                logger.info(f"Set thread priority to {priority}")
                return True
            else:
                logger.error(f"Failed to set thread priority: {ctypes.get_last_error()}")
                return False
        except Exception as e:
            logger.error(f"Error setting thread priority: {e}")
            return False
    
    def get_current_thread_priority(self) -> Optional[int]:
        """Get current thread priority."""
        try:
            current_thread_id = self._kernel32.GetCurrentThreadId()
            
            thread_handle = self._kernel32.OpenThread(
                THREAD_QUERY_INFORMATION,
                False,
                current_thread_id
            )
            
            if not thread_handle:
                return None
            
            priority = self._kernel32.GetThreadPriority(thread_handle)
            self._kernel32.CloseHandle(thread_handle)
            
            return priority
        except Exception as e:
            logger.error(f"Error getting thread priority: {e}")
            return None
    
    def set_process_priority_by_pid(self, pid: int, priority_class: int) -> bool:
        """Set priority class for a process by PID."""
        try:
            process_handle = self._kernel32.OpenProcess(
                PROCESS_SET_INFORMATION,
                False,
                pid
            )
            
            if not process_handle:
                logger.error(f"Failed to open process {pid}: {ctypes.get_last_error()}")
                return False
            
            result = self._kernel32.SetPriorityClass(process_handle, priority_class)
            self._kernel32.CloseHandle(process_handle)
            
            if result:
                logger.info(f"Set process {pid} priority to {priority_class:#x}")
                return True
            else:
                logger.error(f"Failed to set process priority: {ctypes.get_last_error()}")
                return False
        except Exception as e:
            logger.error(f"Error setting process priority: {e}")
            return False
    
    def elevate_python_worker(self) -> bool:
        """
        Elevate current Python worker to HIGH_PRIORITY_CLASS.
        Use this in Python Ray workers to ensure they don't starve but
        still yield to Rust core (REALTIME_PRIORITY).
        """
        success = self.set_current_process_priority(HIGH_PRIORITY_CLASS)
        if success:
            # Also elevate main thread
            self.set_current_thread_priority(THREAD_PRIORITY_ABOVE_NORMAL)
        return success
    
    def set_rust_core_realtime(self, rust_pid: int) -> bool:
        """
        Set Rust HFT core process to REALTIME_PRIORITY_CLASS.
        This gives it maximum scheduler priority.
        
        Args:
            rust_pid: Process ID of the Rust core
        """
        return self.set_process_priority_by_pid(rust_pid, REALTIME_PRIORITY_CLASS)
    
    def configure_hft_priorities(self, rust_core_pid: Optional[int] = None) -> dict:
        """
        Configure optimal priorities for entire HFT system.
        
        Returns:
            dict with configuration results
        """
        results = {
            "python_worker_elevated": False,
            "rust_core_realtime": False,
            "current_python_priority": None,
        }
        
        # Elevate Python workers
        results["python_worker_elevated"] = self.elevate_python_worker()
        results["current_python_priority"] = self.get_current_process_priority()
        
        # Set Rust core to realtime if PID provided
        if rust_core_pid is not None:
            results["rust_core_realtime"] = self.set_rust_core_realtime(rust_core_pid)
        
        logger.info(f"HFT priority configuration: {results}")
        return results


def apply_hft_priority_config(rust_core_pid: Optional[int] = None):
    """
    Apply HFT-optimized priority configuration.
    Call this at the start of Python workers.
    """
    manager = WindowsPriorityManager()
    return manager.configure_hft_priorities(rust_core_pid)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Configuring Windows priorities for HFT...")
    
    manager = WindowsPriorityManager()
    
    # Show current priorities
    current_proc_priority = manager.get_current_process_priority()
    current_thread_priority = manager.get_current_thread_priority()
    
    print(f"Current process priority: {current_proc_priority:#x}")
    print(f"Current thread priority: {current_thread_priority}")
    
    # Apply HFT configuration
    results = manager.configure_hft_priorities()
    print(f"\nConfiguration results: {results}")
    
    # Verify changes
    new_proc_priority = manager.get_current_process_priority()
    print(f"\nNew process priority: {new_proc_priority:#x}")
    
    if new_proc_priority == HIGH_PRIORITY_CLASS:
        print("SUCCESS: Python worker elevated to HIGH_PRIORITY_CLASS")
    if new_proc_priority == REALTIME_PRIORITY_CLASS:
        print("WARNING: Python running at REALTIME - should be HIGH only")
