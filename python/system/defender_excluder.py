# Chapter 3, File 2: Windows Defender Excluder
# python/system/defender_excluder.py
# Adds bot executables and data files to Windows Defender exclusion list

import ctypes
from ctypes import wintypes
import subprocess
import os
import sys
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# PowerShell execution policy bypass command
PS_BYPASS = "-ExecutionPolicy Bypass -NoProfile"


class WindowsDefenderExcluder:
    """
    Automates adding HFT bot files to Windows Defender exclusion list.
    Prevents real-time AV scanning from causing I/O latency spikes.
    
    Requires Administrator privileges.
    """
    
    def __init__(self):
        self._exclusions_added: List[str] = []
        
    def is_admin(self) -> bool:
        """Check if running with administrator privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    
    def add_path_exclusion(self, path: str) -> bool:
        """Add a path to Windows Defender exclusions."""
        if not self.is_admin():
            logger.error("Administrator privileges required for Defender exclusions")
            return False
        
        # Normalize path
        normalized_path = os.path.normpath(path)
        
        # PowerShell command to add exclusion
        ps_command = (
            f'Add-MpPreference -ExclusionPath "{normalized_path}" -ErrorAction SilentlyContinue'
        )
        
        try:
            result = subprocess.run(
                ["powershell", PS_BYPASS, "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self._exclusions_added.append(normalized_path)
                logger.info(f"Added Defender exclusion: {normalized_path}")
                return True
            else:
                logger.warning(f"Failed to add exclusion: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout adding exclusion for {path}")
            return False
        except Exception as e:
            logger.error(f"Error adding exclusion: {e}")
            return False
    
    def add_process_exclusion(self, process_name: str) -> bool:
        """Add a process to Windows Defender exclusions."""
        if not self.is_admin():
            logger.error("Administrator privileges required")
            return False
        
        ps_command = (
            f'Add-MpPreference -ExclusionProcess "{process_name}" -ErrorAction SilentlyContinue'
        )
        
        try:
            result = subprocess.run(
                ["powershell", PS_BYPASS, "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Added process exclusion: {process_name}")
                return True
            else:
                logger.warning(f"Failed to add process exclusion: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding process exclusion: {e}")
            return False
    
    def add_extension_exclusion(self, extension: str) -> bool:
        """Add a file extension to Windows Defender exclusions."""
        if not self.is_admin():
            logger.error("Administrator privileges required")
            return False
        
        ps_command = (
            f'Add-MpPreference -ExclusionExtension "{extension}" -ErrorAction SilentlyContinue'
        )
        
        try:
            result = subprocess.run(
                ["powershell", PS_BYPASS, "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Added extension exclusion: {extension}")
                return True
            else:
                logger.warning(f"Failed to add extension exclusion: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding extension exclusion: {e}")
            return False
    
    def remove_path_exclusion(self, path: str) -> bool:
        """Remove a path from Windows Defender exclusions."""
        if not self.is_admin():
            return False
        
        normalized_path = os.path.normpath(path)
        ps_command = f'Remove-MpPreference -ExclusionPath "{normalized_path}" -ErrorAction SilentlyContinue'
        
        try:
            result = subprocess.run(
                ["powershell", PS_BYPASS, "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                if normalized_path in self._exclusions_added:
                    self._exclusions_added.remove(normalized_path)
                logger.info(f"Removed Defender exclusion: {normalized_path}")
                return True
            else:
                logger.warning(f"Failed to remove exclusion: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing exclusion: {e}")
            return False
    
    def get_current_exclusions(self) -> dict:
        """Get current Windows Defender exclusions."""
        ps_command = "Get-MpPreference | Select-Object ExclusionPath,ExclusionProcess,ExclusionExtension"
        
        try:
            result = subprocess.run(
                ["powershell", PS_BYPASS, "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {"output": result.stdout}
            else:
                return {"error": result.stderr}
                
        except Exception as e:
            return {"error": str(e)}
    
    def apply_hft_exclusions(self, base_dir: str) -> bool:
        """
        Apply all recommended exclusions for HFT bot.
        
        Args:
            base_dir: Base directory of the HFT bot installation
            
        Returns:
            bool: Success status
        """
        if not self.is_admin():
            logger.error("Must run as Administrator to apply Defender exclusions")
            return False
        
        base_path = Path(base_dir).resolve()
        
        # Directories to exclude
        directories = [
            base_path / "crates" / "hft" / "target" / "release",
            base_path / "python",
            base_path / "data" / "lmdb",
            base_path / "data" / "parquet",
            base_path / "logs",
        ]
        
        # Executables to exclude
        executables = [
            "hft_bot.exe",
            "python.exe",
            "raylet.exe",
            "node.exe",  # For frontend
        ]
        
        # Extensions to exclude
        extensions = [
            "*.lmdb",
            "*.mdb",
            "*.parquet",
            "*.bin",
        ]
        
        success_count = 0
        total_count = 0
        
        # Add directory exclusions
        for dir_path in directories:
            if dir_path.exists():
                total_count += 1
                if self.add_path_exclusion(str(dir_path)):
                    success_count += 1
        
        # Add process exclusions
        for exe_name in executables:
            total_count += 1
            if self.add_process_exclusion(exe_name):
                success_count += 1
        
        # Add extension exclusions
        for ext in extensions:
            total_count += 1
            if self.add_extension_exclusion(ext):
                success_count += 1
        
        logger.info(f"Applied {success_count}/{total_count} Defender exclusions")
        return success_count == total_count


def main():
    """Main entry point for applying HFT Defender exclusions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    excluder = WindowsDefenderExcluder()
    
    if not excluder.is_admin():
        print("ERROR: This script must be run as Administrator")
        print("Right-click and select 'Run as Administrator'")
        sys.exit(1)
    
    # Get base directory (parent of this script's directory)
    base_dir = Path(__file__).parent.parent.parent.resolve()
    
    print(f"Applying Windows Defender exclusions for HFT bot at: {base_dir}")
    
    if excluder.apply_hft_exclusions(str(base_dir)):
        print("SUCCESS: All Defender exclusions applied")
    else:
        print("WARNING: Some exclusions may have failed")
    
    # Show current exclusions
    print("\nCurrent Defender exclusions:")
    exclusions = excluder.get_current_exclusions()
    print(exclusions.get("output", exclusions.get("error", "Unknown")))


if __name__ == "__main__":
    main()
