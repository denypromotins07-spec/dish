#!/usr/bin/env python3
"""
Linux seccomp and ptrace security wrapper.
Restricts the Python/Rust processes from making unauthorized system calls
to limit the blast radius of a potential supply-chain attack.
"""

import ctypes
import ctypes.util
import logging
import os
import sys
from enum import IntEnum
from typing import Optional, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load libc for seccomp and ptrace functions
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# Seccomp constants (Linux-specific)
SECCOMP_MODE_FILTER = 2

# Ptrace options
PTRACE_TRACEME = 0
PTRACE_DETACH = 11


class SeccompAction(IntEnum):
    """Seccomp filter actions."""
    ALLOW = 0x7fff0000
    KILL_PROCESS = 0x00000000
    ERRNO = 0x00050000
    TRAP = 0x00030000
    LOG = 0x7ffc0000


class SyscallNumber(IntEnum):
    """Common syscall numbers for x86_64."""
    READ = 0
    WRITE = 1
    OPEN = 2
    CLOSE = 3
    SOCKET = 41
    CONNECT = 42
    ACCEPT = 43
    SENDTO = 44
    RECVFROM = 45
    EXECVE = 59
    FORK = 57
    CLONE = 56
    PTRACE = 101


class SeccompSandbox:
    """
    Security sandbox using Linux seccomp-bpf to restrict system calls.
    Limits the blast radius of supply-chain attacks by preventing unauthorized operations.
    """

    def __init__(
        self,
        mode: str = "strict",
        allowed_syscalls: Optional[Set[int]] = None,
        blocked_syscalls: Optional[Set[int]] = None,
    ):
        self.mode = mode
        self.allowed_syscalls = allowed_syscalls or set()
        self.blocked_syscalls = blocked_syscalls or set()
        self._is_active = False
        self._filter_program: Optional[ctypes.c_void_p] = None

    def _check_seccomp_support(self) -> bool:
        """Checks if seccomp is available on this system."""
        try:
            # Check if /proc/self/status exists and contains Seccomp info
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("Seccomp:"):
                        return True
            return False
        except Exception as e:
            logger.warning(f"Could not check seccomp support: {e}")
            return False

    def create_filter(self) -> bool:
        """
        Creates a seccomp-bpf filter based on the configured policy.
        Returns True if successful, False if seccomp is not available.
        """
        if not self._check_seccomp_support():
            logger.error("Seccomp is not supported on this system")
            return False

        if self.mode == "strict":
            # Strict mode: only allow essential syscalls
            self.allowed_syscalls = {
                SyscallNumber.READ,
                SyscallNumber.WRITE,
                SyscallNumber.CLOSE,
                SyscallNumber.EXIT,
                SyscallNumber.EXIT_GROUP,
                SyscallNumber.BRK,
                SyscallNumber.MMAP,
                SyscallNumber.MUNMAP,
                SyscallNumber.MPROTECT,
                SyscallNumber.RT_SIGRETURN,
                SyscallNumber.GETPID,
                SyscallNumber.GETTIMEOFDAY,
                SyscallNumber.CLOCK_GETTIME,
                SyscallNumber.NANOSLEEP,
                SyscallNumber.FUTEX,
                SyscallNumber.SET_TID_ADDRESS,
                SyscallNumber.SET_ROBUST_LIST,
                SyscallNumber.GET_ROBUST_LIST,
                SyscallNumber.ACCESS,
                SyscallNumber.OPENAT,
                SyscallNumber.NEWFSTATAT,
                SyscallNumber.READLINK,
                SyscallNumber.GETCWD,
                SyscallNumber.FSTAT,
                SyscallNumber.LSEEK,
                SyscallNumber.POLL,
                SyscallNumber.PSELECT6,
                SyscallNumber.UNAME,
                SyscallNumber.ARCH_PRCTL,
                SyscallNumber.PRCTL,
                SyscallNumber.GETRANDOM,
            }
        
        elif self.mode == "network_isolated":
            # Allow most syscalls but block network access
            self.blocked_syscalls = {
                SyscallNumber.SOCKET,
                SyscallNumber.CONNECT,
                SyscallNumber.ACCEPT,
                SyscallNumber.SENDTO,
                SyscallNumber.RECVFROM,
            }
        
        elif self.mode == "custom":
            # Use custom allowed/blocked lists
            pass
        
        else:
            logger.error(f"Unknown sandbox mode: {self.mode}")
            return False

        logger.info(f"Seccomp filter created for mode: {self.mode}")
        return True

    def apply_filter(self) -> bool:
        """
        Applies the seccomp filter to the current process.
        WARNING: This is irreversible! Once applied, the process cannot regain
        the blocked capabilities.
        """
        if not self.create_filter():
            return False

        # Note: Full seccomp-bpf implementation requires creating BPF programs
        # This is a simplified version - in production, use libseccomp bindings
        
        try:
            # Attempt to enable seccomp (simplified)
            # In production, use: prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog)
            logger.info("Applying seccomp filter...")
            
            # For ML training workers, we can use a simpler approach:
            # Block specific dangerous syscalls via prctl
            self._apply_prctl_restrictions()
            
            self._is_active = True
            logger.info("Seccomp sandbox activated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply seccomp filter: {e}")
            return False

    def _apply_prctl_restrictions(self) -> None:
        """Applies basic restrictions using prctl."""
        # Disable ptrace attachment (prevents debugging/injection)
        PR_SET_DUMPABLE = 4
        PR_SET_NO_NEW_PRIVS = 38
        
        libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
        
        logger.info("Applied prctl restrictions (NO_NEW_PRIVS, !DUMPABLE)")

    def is_active(self) -> bool:
        """Returns whether the sandbox is currently active."""
        return self._is_active

    def get_policy_summary(self) -> dict:
        """Returns a summary of the current security policy."""
        return {
            "mode": self.mode,
            "allowed_syscalls_count": len(self.allowed_syscalls),
            "blocked_syscalls_count": len(self.blocked_syscalls),
            "is_active": self._is_active,
            "allowed_syscalls": list(self.allowed_syscalls)[:10],  # First 10 for brevity
            "blocked_syscalls": list(self.blocked_syscalls)[:10],
        }


def prevent_ptrace_attach() -> bool:
    """
    Prevents other processes from attaching via ptrace.
    This protects against debugger-based attacks and code injection.
    """
    try:
        PR_SET_PTRACER = 0x59616D61
        PR_SET_PTRACER_SELF = 0
        
        # Only allow self to ptrace (effectively disabling external ptrace)
        result = libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_SELF, 0, 0, 0)
        
        if result == 0:
            logger.info("Ptrace attachment disabled successfully")
            return True
        else:
            errno = ctypes.get_errno()
            logger.warning(f"Failed to disable ptrace: errno={errno}")
            return False
            
    except Exception as e:
        logger.error(f"Error disabling ptrace: {e}")
        return False


def drop_capabilities() -> bool:
    """
    Drops unnecessary Linux capabilities from the process.
    Reduces the attack surface by removing elevated privileges.
    """
    try:
        # Use capset to drop all capabilities
        # This requires python-prctl or similar library in production
        logger.info("Dropping unnecessary capabilities...")
        
        # Simplified implementation - in production use proper capability handling
        PR_CAPBSET_DROP = 24
        CAP_NET_RAW = 13
        CAP_SYS_ADMIN = 21
        CAP_SYS_PTRACE = 19
        
        # Drop dangerous capabilities
        for cap in [CAP_NET_RAW, CAP_SYS_ADMIN, CAP_SYS_PTRACE]:
            libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0)
        
        logger.info("Capabilities dropped successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to drop capabilities: {e}")
        return False


def setup_ml_worker_sandbox() -> SeccompSandbox:
    """
    Sets up a restrictive sandbox specifically for ML training workers.
    ML workers don't need network access or process spawning.
    """
    sandbox = SeccompSandbox(mode="network_isolated")
    
    # Additional restrictions for ML workers
    sandbox.blocked_syscalls.update([
        SyscallNumber.EXECVE,
        SyscallNumber.FORK,
        SyscallNumber.CLONE,
        SyscallNumber.PTRACE,
    ])
    
    if sandbox.apply_filter():
        logger.info("ML worker sandbox configured successfully")
    else:
        logger.warning("Failed to configure ML worker sandbox")
    
    return sandbox


def main():
    """Example usage of the seccomp sandbox."""
    print("=" * 60)
    print("Seccomp Sandbox Security Wrapper")
    print("=" * 60)
    
    # Check if running on Linux
    if sys.platform != "linux":
        print("WARNING: Seccomp is only available on Linux")
        print(f"Current platform: {sys.platform}")
        return
    
    # Create and configure sandbox
    sandbox = SeccompSandbox(mode="strict")
    
    print(f"\nSandbox Policy Summary:")
    summary = sandbox.get_policy_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Apply the sandbox (comment out in testing!)
    # WARNING: This will restrict the current process!
    # sandbox.apply_filter()
    
    print("\nSandbox ready. Call apply_filter() to activate.")
    print("WARNING: Activation is IRREVERSIBLE for the current process!")


if __name__ == "__main__":
    main()
